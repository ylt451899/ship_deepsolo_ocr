# -*- coding: utf-8 -*-
import os
import cv2
import torch
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Detectron2 imports
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T

# AdelaiDet imports
from adet.config import get_cfg as get_adet_cfg
# from adet.data.augmentation import Pad
from ultralytics import YOLO
# --- 設定參數 ---
CONFIG_FILE = "configs/r_50/rects/pretrain.yaml"
MODEL_WEIGHTS = "res50_pretrain_synch-art-lsvt-rects.pth"
INPUT_IMAGE = "data/20251125153149.jpeg"  # 請確認檔名
OUTPUT_IMAGE = "data/result_chinese_test.jpg"
CHN_CLS_LIST_PATH = "chn_cls_list"
FONT_PATH = "simsun.ttc"
OUTPUT_BASE_DIR = "data/output" # *** 修改輸出基礎目錄 ***
if not os.path.exists(FONT_PATH):
    FONT_PATH = "font/Arial-Unicode-MS.ttf"

class SimplePadTransform:
    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def apply_image(self, img):
        return cv2.copyMakeBorder(
            img,
            self.top, self.bottom,
            self.left, self.right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )


class SimplePad:
    def __init__(self, divisible_size=32):
        self.divisible_size = divisible_size

    def get_transform(self, image):
        h, w = image.shape[:2]
        div = self.divisible_size

        new_h = int(np.ceil(h / div) * div)
        new_w = int(np.ceil(w / div) * div)

        pad_h = new_h - h
        pad_w = new_w - w

        return SimplePadTransform(0, pad_h, 0, pad_w)

class DeepSoloPredictor:
    """
    基於 DeepSolo 官方 predictor.py 修改的預測器。
    確保包含 Resize 和 Pad 操作，以支援 ViTAE 和 Transformer 架構。
    """
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()

        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        # 這是關鍵：DeepSolo 需要將圖片 Resize 並 Pad 到 32 的倍數
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.pad = SimplePad(divisible_size=32)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): (H, W, C) BGR 格式圖片
        """
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]

            height, width = original_image.shape[:2]

            # 1. Resize
            image = self.aug.get_transform(original_image).apply_image(original_image)
            # 2. Pad (這一步解決了維度報錯的問題)
            image = self.pad.get_transform(image).apply_image(image)
            # 3. 轉換為 Tensor (C, H, W)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}

            # 模型推論
            predictions = self.model([inputs])[0]
            return predictions

def setup_cfg():
    """初始化配置"""
    cfg = get_adet_cfg() # 這裡不需要參數

    # 載入設定檔
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"找不到設定檔: {CONFIG_FILE}")
    cfg.merge_from_file(CONFIG_FILE)

    # 設定權重
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"找不到權重檔: {MODEL_WEIGHTS}")
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS

    # 設定裝置
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # === 重要：不要設定 FCOS 相關閾值，因為 DeepSolo 是 Transformer 架構 ===
    # 我們只設定通用的測試閾值 (如果配置檔中有定義)
    if hasattr(cfg.MODEL, "RETINANET"):
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3
    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3

    cfg.freeze()
    return cfg

def load_dictionary(dict_path):
    if not os.path.exists(dict_path):
        print(f"警告: 找不到字表 {dict_path}")
        return None
    with open(dict_path, 'rb') as f:
        chars = pickle.load(f)
    return chars

def decode_text(rec, vocab):
    """解碼模型輸出的索引序列"""
    if vocab is None:
        return str(rec)

    text = ""
    for idx in rec:
        # EOS
        if idx == len(vocab):
            continue

        if 0 <= idx < len(vocab):
            ch = vocab[idx]

            # ✅ 若是 int Unicode -> 轉字元
            if isinstance(ch, int):
                ch = chr(ch)

            text += ch

    return text

# --- 輔助函式：取得資料夾內的圖片清單 ---
def get_image_paths(folder_path, valid_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """
    掃描指定資料夾，並返回所有圖片檔案的完整路徑列表。
    """
    image_list = []
    
    if not os.path.isdir(folder_path):
        print(f"錯誤：資料夾 '{folder_path}' 不存在。")
        return image_list

    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(folder_path):
        # 將檔名轉為小寫，並檢查副檔名是否在有效清單中
        if filename.lower().endswith(valid_extensions):
            # 建立圖片的完整路徑
            full_path = str(os.path.join(folder_path, filename))
            image_list.append(full_path)
            
    return image_list

def main():
    print("正在初始化環境與模型...")
    cfg = setup_cfg()

    # 使用我們自定義的 Predictor (包含 Pad 邏輯)
    predictor = DeepSoloPredictor(cfg)

    vocab = load_dictionary(CHN_CLS_LIST_PATH)
    # 2. 讀取圖片
    # DATA_DIR = "data" 
    DATA_DIR = "step2_ship_crops"
    image_list = get_image_paths(DATA_DIR)
    for image in image_list:
        # 根據圖片路徑生成輸出檔案名
        # 範例：step2_ship_crops/image1.jpg -> image1
        base_name = os.path.splitext(os.path.basename(image))[0]
        output_txt_path = os.path.join(OUTPUT_BASE_DIR, f"{base_name}_result.txt")
        output_img_path = os.path.join(OUTPUT_BASE_DIR, f"{base_name}_vis.jpg")

        print(f"\n--- 處理圖片: {image} ---")
        img = cv2.imread(image)
        if img is None:
            print(f"錯誤: 無法讀取 {image}，跳過。")
            continue
        
        # 開啟 TXT 檔案準備寫入
        with open(output_txt_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"--- 圖片檔案: {os.path.basename(image)} ---\n\n")

            print("正在執行推論 (Inference)...")
            # 執行推論
            outputs = predictor(img)
            
            # 提取 CPU 上的實例
            instances = outputs["instances"].to("cpu")
            
            # 準備繪圖
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                font = ImageFont.truetype(FONT_PATH, 24)
            except:
                font = ImageFont.load_default()

            f_out.write("--- 辨識結果 ---\n")
            
            if instances.has("recs"):
                recs = instances.recs.tolist()
                scores = instances.scores.tolist()

                # 處理邊界框
                if instances.has("pred_boxes"):
                    boxes = instances.pred_boxes.tensor.numpy()
                else:
                    boxes = None

                for i, rec in enumerate(recs):
                    score = scores[i]
                    if score < 0.3: continue # 過濾低分

                    text = decode_text(rec, vocab)
                    result_line = f"文字 {i+1}: {text} (信心度: {score:.2f})\n"
                    
                    # 寫入 TXT 檔案
                    f_out.write(result_line)
                    # 輸出到終端機
                    print(result_line.strip())

                    # 繪圖
                    if boxes is not None:
                        x1, y1, x2, y2 = boxes[i]
                        # 畫框
                        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                        # 畫文字
                        draw.text((x1, y1 - 25), f"{text} ({score:.2f})", font=font, fill=(255, 0, 0))
            
            print(f"✅ 文字結果已儲存至: {output_txt_path}")
            
        # 儲存結果圖片
        vis_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_img_path, vis_img)
        print(f"🖼️ 視覺化圖片已儲存至: {output_img_path}")
        # print(f"讀取圖片: {image}")
        # img = cv2.imread(image)
        # if img is None:
        #     print(f"錯誤: 無法讀取 {image}")
        #     return


        # print("正在執行推論 (Inference)...")
        # # 執行推論
        # outputs = predictor(img)
        # # print(outputs)
        # # 提取 CPU 上的實例
        # instances = outputs["instances"].to("cpu")
        # fields = instances.get_fields()
        # # print("✅ OCR 回傳欄位:", fields.keys())
        # # 準備繪圖
        # img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # draw = ImageDraw.Draw(img_pil)
        # try:
        #     font = ImageFont.truetype(FONT_PATH, 24)
        # except:
        #     font = ImageFont.load_default()

        # print("\n--- 辨識結果 ---")
        # if instances.has("recs"):
        #     recs = instances.recs.tolist()
        #     scores = instances.scores.tolist()

        #     # 處理邊界框 (DeepSolo 輸出的是 Bezier 曲線或 Boxes)
        #     if instances.has("pred_boxes"):
        #         boxes = instances.pred_boxes.tensor.numpy()
        #     else:
        #         boxes = None

        #     for i, rec in enumerate(recs):
        #         score = scores[i]
        #         if score < 0.3: continue # 過濾低分

        #         text = decode_text(rec, vocab)
        #         print(f"文字 {i+1}: {text} (信心度: {score:.2f})")

        #         # 繪圖
        #         if boxes is not None:
        #             x1, y1, x2, y2 = boxes[i]
        #             # 畫框
        #             draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        #             # 畫文字
        #             draw.text((x1, y1 - 25), f"{text} ({score:.2f})", font=font, fill=(255, 0, 0))

        # vis_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(OUTPUT_IMAGE, vis_img)
        # print(f"\n結果已儲存至: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()


