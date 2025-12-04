# Ship-OCR: 智慧型船隻影像辨識系統

本專案是一個完整的三階段船隻影像辨識流程，旨在從原始影像中自動偵測船隻、裁切目標，並對船隻上的船名、船號等文字進行光學字元辨識 (OCR)。

## 功能特色

- **影像前處理**: 自動過濾光線過曝或過暗的影像，並使用感知雜湊演算法去除連續的重複影像。
- **對比度增強**: 採用 CLAHE (對比度限制自適應直方圖等化) 技術，在突顯細節的同時抑制雜訊。
- **高效率船隻偵測**: 整合 YOLOv11 模型，快速且準確地定位影像中的船隻。
- **高精度文字辨識**: 採用基於 Transformer 的 DeepSolo 模型，實現精準的船名與船號 OCR。

## 處理流程

本專案的處理流程如下，使用者僅需依序執行三個 Python 腳本即可完成所有操作。

```
[原始圖片] -> [Step 1: data_cleaning.py] -> [純淨圖片] -> [Step 2: ship_detection.py] -> [船隻裁切圖] -> [Step 3: ship_ocr_detection.py] -> [辨識結果 (文字檔)]
```

## 安裝與設定

請依照以下步驟設定您的開發環境。

**1. 複製專案**

```bash
git clone <your-repository-url>
cd deepsolo-main
```

**2. 安裝相依套件**

建議使用 `environment.yml`，建立 Conda 虛擬環境
或是使用 Conda 建立虛擬環境，並透過 `requirements.txt` 安裝所需套件。

```bash
# 建議先安裝 PyTorch，請參考 PyTorch 官網選擇符合您 CUDA 版本的指令
# https://pytorch.org/get-started/locally/

# 安裝其他套件
pip install -r requirements.txt
```
**注意**: `ship_ocr_detection.py` 依賴 `Detectron2` 和 `AdelaiDet`。請確保它們已成功安裝。

**3. 下載預訓練模型**

本專案需要兩個預訓練模型，請手動下載並放置到正確的位置。

*   **YOLOv8n (船隻偵測模型)**
    *   執行 `ship_detection.py` 會自動下載檔案 `yolov11n.pt`
    *   **下載檔案**: `yolov11n.pt`
    *   **放置路徑**: 請在專案根目錄下建立 `weights` 資料夾，並將模型放入。最終路徑為 `./weights/yolov11n.pt`。
    *   *(注意: `ship_detection.py` 中模型路徑為 `'../weights/yolo11n.pt'`，請自行確認模型檔名與路徑是否一致，或修改程式碼中的路徑。此處以 `yolov11n.pt` 為例)*

*   **DeepSolo (OCR 模型)**
    *   **下載檔案**: `res50_pretrain_synch-art-lsvt-rects.pth`
    *   **放置路徑**: 請將模型檔案直接放置在專案的根目錄下，與 `ship_ocr_detection.py` 同層。
    *   **開源模行下載網址**: https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaW1CZ1lWN0pqVGxnY2gxcEgwOGJvbGhnS0VCUVE%5FZT15ZVN3d1E&cid=E534267B85818129&id=E534267B85818129%2125717&parId=E534267B85818129%2125575&o=OneUp

**4. 準備字體檔案**

為了讓辨識結果可以正確地視覺化 (尤其是中文字)，請確保字體檔案存在。
- 專案預設使用 `simsun.ttc` (宋體)，請將其放置在專案根目錄。
- 如果找不到，程式會嘗試使用備用的 `Arial-Unicode-MS.ttf`。

## 使用指南

請將您要處理的原始影像全部放入 `data` 資料夾，然後依序執行以下三個步驟。

**第一步：資料清洗**

此步驟會讀取 `data` 資料夾中的圖片，進行亮度檢查、去重複與影像增強，並將處理後的圖片儲存到 `step1_cleaned` 資料夾。

```bash
python data_cleaning.py
```

**第二步：船隻偵測與裁切**

此步驟會讀取 `step1_cleaned` 中的圖片，使用 YOLO 模型偵測船隻，並將每個偵測到的船隻裁切成獨立的圖片，儲存到 `step2_ship_crops` 資料夾。

```bash
python ship_detection.py
```

**第三步：船名船號 OCR 辨識**

此步驟是最後一步，它會處理 `step2_ship_crops` 中所有的船隻裁切圖，並利用 DeepSolo 模型進行文字辨識。

```bash
python ship_ocr_detection.py
```

辨識完成後，結果會儲存在 `data/output` 資料夾中，每張輸入的船隻圖都會對應產生兩個檔案：
- `*_result.txt`: 包含辨識出的文字與其信心分數。
- `*_vis.jpg`: 一張視覺化圖片，其中辨識出的文字會被綠色框標記出來。

## 專案結構

```
.
├── AdelaiDet.egg-info/   # AdelaiDet 專案的 Python egg-info 元資料 (安裝時生成)
├── adet/                 # DeepSolo 框架核心程式碼
├── build/                # Python 專案構建相關檔案 (如編譯的 C/C++ 擴展模組)
├── configs/              # DeepSolo 模型設定檔
├── char_map/             # DeepSolo 所使用的字元集與索引對應檔
├── tools/                # DeepSolo 官方提供的訓練/評估工具
├── demo/                 # DeepSolo 官方提供的範例腳本
├── data/                 # 存放原始圖片
│   └── output/           # 存放最終辨識結果
├── step1_cleaned/        # 存放第一步清洗後的圖片
├── step2_ship_crops/     # 存放第二步裁切出的船隻圖片
├── weights/              # 存放 YOLO 模型
├── data_cleaning.py      # 第一步：資料清洗腳本
├── ship_detection.py     # 第二步：船隻偵測腳本
├── ship_ocr_detection.py # 第三步：OCR 辨識腳本
├── requirements.txt      # Python 相依套件列表
├── environment.yml       # Conda 環境套件列表
└── README.md             # 本說明文件
```

## 致謝

- 本專案的 OCR 功能基於 [DeepSolo](https://github.com/ViTAE-Transformer/DeepSolo)。
- 船隻偵測功能使用了 [YOLO (You Only Look Once)](https://github.com/ultralytics/ultralytics) 模型。

感謝這些優秀的開源專案。
>>>>>>> Stashed changes
