import os
import cv2
import numpy as np
from ultralytics import YOLO

class ShipDetectorAdvanced:
    """
        進階船隻檢測器
        1. 船隻檢測：使用 YOLO 定位船隻。
        2. 尺寸過濾：剔除過小的無效目標。
        3. ROI 裁切：將偵測到的船隻區域裁切下來供後續 OCR 使用。
        4. 重疊框抑制 (NMS) 與 連續截圖去重複。
    """
    def __init__(self, input_dir, output_dir, model_path='./weights/yolo11n.pt'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        print(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path) 
        
        # 參數設定
        self.min_area = 5000       # 面積過濾
        self.target_class_id = 8   # 8=boat
        self.conf_threshold = 0.25 # 信心度
        self.iou_threshold = 0.3   # NMS 重疊閾值：數值越小，越不會有重疊框
        
        # 截圖去重複參數
        self.save_similarity_threshold = 0.85  # 如果兩張截圖相似度 > 85%，視為重複不存
        self.last_saved_img = None             # 用來存上一張船的特徵

        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_similarity(self, img1, img2):
        """
        計算兩張圖片的直方圖相似度 (0~1)
        """
        try:
            # 轉為 HSV 色彩空間比較準確
            hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            
            # 計算直方圖
            hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
            
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # 比較相似度 (Correlation)
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        except:
            return 0.0

    def process(self):
        print(f"Start Detection with De-duplication: {self.input_dir}")
        files = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        saved_count = 0

        for f in files:
            path = os.path.join(self.input_dir, f)
            img = cv2.imread(path)
            if img is None: continue
            
            # [修改點 1] 加入 agnositc_nms 與 iou 參數，解決單張圖重疊框問題
            results = self.model(
                img, 
                conf=self.conf_threshold, 
                iou=self.iou_threshold,     # NMS 閾值
                agnostic_nms=True,          # 強制不同類別的框如果重疊也要過濾 (避免船被誤判為車疊在一起)
                verbose=False
            )
            
            crop_idx = 0
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != self.target_class_id:
                        continue
                        
                    # 取得座標
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    
                    # 尺寸過濾
                    if area < self.min_area:
                        continue
                    
                    # 裁切圖片
                    pad = 10
                    h_img, w_img = img.shape[:2]
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w_img, x2 + pad)
                    y2 = min(h_img, y2 + pad)
                    crop_img = img[y1:y2, x1:x2]
                    
                    # [修改點 2] 連續截圖去重複 (Similarity Check)
                    # 如果這張裁切圖跟上一張存的圖非常像，就跳過
                    if self.last_saved_img is not None:
                        sim = self.calculate_similarity(self.last_saved_img, crop_img)
                        if sim > self.save_similarity_threshold:
                            # print(f" -> [跳過重複] {f} 相似度: {sim:.2f}")
                            continue # 太像了，不存
                    
                    # 存檔
                    save_name = f"{os.path.splitext(f)[0]}_ship{crop_idx}.jpg"
                    cv2.imwrite(os.path.join(self.output_dir, save_name), crop_img)
                    
                    # 更新上一張圖
                    self.last_saved_img = crop_img
                    saved_count += 1
                    crop_idx += 1
                    
        print(f"處理完成。共儲存 {saved_count} 張具備差異性的船隻截圖。")

if __name__ == "__main__":
    detector = ShipDetectorAdvanced(input_dir="step1_cleaned", output_dir="step2_ship_crops")
    detector.process()