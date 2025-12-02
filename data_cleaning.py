import os
import cv2
import numpy as np
import imagehash
from PIL import Image

class ImagePreprocessor:
    """
        影像前處理類別
        功能包含：
        1. [cite_start]亮度檢查 (剔除過曝/過暗) [cite: 18]
        2. [cite_start]去重複 (感知雜湊演算法) [cite: 19]
        3. [cite_start]影像增強 (CLAHE) [cite: 25]
    """
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 參數設定
        self.min_brightness = 40    # 過暗閾值
        self.max_brightness = 220   # 過曝閾值
        self.hash_threshold = 5     # 去重複閾值

        # CLAHE 設定 (clipLimit=2.0: 限制對比度放大的倍數，避免過度放大雜訊, tileGridSize=(8, 8): 將圖像切分為 8x8 的區塊分別進行均化)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  
        
        # 儲存上一張保留圖片的 Hash 值，供去重複比對使用
        self.prev_hash = None

        # 輸出資料夾
        os.makedirs(self.output_dir, exist_ok=True)

    def apply_clahe_color(self, img):
        """
        使用 CLAHE 增強對比度 (針對彩色圖片的 L 通道處理)
        """
        # 將 BGR 轉為 LAB 
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # 分離 L (Lightness/亮度), A (紅綠), B (黃藍)
        l, a, b = cv2.split(lab)

        # 對 L (亮度) 通道做 CLAHE
        cl = self.clahe.apply(l)

        # 合併回 BGR
        limg = cv2.merge((cl, a, b))

        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def process(self):
        print(f"Start Processing: {self.input_dir} -> {self.output_dir}")

        # 排序檔案 (for 去重複)
        files = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        processed_count = 0
        
        for f in files:
            path = os.path.join(self.input_dir, f)
            img = cv2.imread(path)
            if img is None: continue
            
            # 亮度檢查
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_bright = np.mean(gray)
            if mean_bright < self.min_brightness or mean_bright > self.max_brightness:
                print(f"[剔除-亮度異常] {f}: {mean_bright:.1f}")
                continue
                
            # 去重複檢查
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            curr_hash = imagehash.phash(pil_img)
            if self.prev_hash and (curr_hash - self.prev_hash < self.hash_threshold):
                print(f"[剔除-重複影像] {f}")
                continue
            self.prev_hash = curr_hash
            
            # 影像增強 (CLAHE)
            enhanced_img = self.apply_clahe_color(img)
            
            # 儲存結果
            cv2.imwrite(os.path.join(self.output_dir, f), enhanced_img)
            processed_count += 1
            
        print(f"處理完成，共保留 {processed_count} 張圖片。")

if __name__ == "__main__":
    processor = ImagePreprocessor(input_dir="data", output_dir="step1_cleaned")
    processor.process()