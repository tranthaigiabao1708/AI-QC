# 🔍 Hệ Thống Nhận Diện Sản Phẩm Lỗi — Computer Vision (Hugging Face + PyTorch)

Dự án này là một hệ thống kiểm tra chất lượng sản phẩm **cos đồng crimping** tự động (Quality Control) sử dụng kết hợp **Computer Vision truyền thống (OpenCV)** để tiền xử lý/cắt vùng đặc trưng (ROI) và **Deep Learning (Hugging Face + PyTorch)** để phân loại lỗi sản phẩm (OK/NG).

**Tính năng nổi bật:**
- ✅ Pipeline 6 bước bền vững — phát hiện đồng lộ bằng HSV + LAB direct range
- ✅ Nhận diện live từ camera ở 20fps (multi-thread, temporal smoothing, OSD overlay)
- ✅ Web dashboard Streamlit tương tác
- ✅ Hỗ trợ ONNX Runtime tăng tốc inference

---

## 🚀 Clone Về Máy Khác & Chạy Ngay

> ⚠️ **YÊU CẦU: Python 3.10 - 3.12** (Python 3.13+ hoặc 3.9- sẽ lỗi numpy)
>
> Tải Python 3.12: https://www.python.org/downloads/release/python-3129/
>
> Khi cài **bắt buộc tick ✅ "Add python.exe to PATH"**

### Bước 1: Clone repository

```bash
git clone https://github.com/tranthaigiabao1708/AI-QC.git
cd AI-QC/defect_detection_huggingface
```

> Nếu đã clone rồi, chỉ cần cập nhật code mới:
> ```bash
> cd AI-QC
> git stash
> git pull origin main
> cd defect_detection_huggingface
> ```

### Bước 2: Cài đặt thư viện

```bash
py -m pip install --upgrade pip setuptools wheel
py -m pip install -r requirements.txt
```

> **Lưu ý:** Lần đầu chạy, hệ thống sẽ tự động tải model `microsoft/resnet-18` (~44MB) từ Hugging Face Hub. Cần có kết nối Internet.

### Bước 3: Chạy thử inference trên ảnh mẫu

```bash
py -m inference.predict
```

Kết quả sẽ được lưu tại `output/predictions/`.

---

## 📹 Chạy Nhận Diện Live Từ Camera (Chế Độ Thực Tế)

Đây là chế độ chính để kiểm tra sản phẩm trên dây chuyền thực tế.

### Cách 1: Lệnh đơn giản nhất

```bash
py live_inspector.py --camera 0
```

### Cách 2: Chọn chế độ hoạt động

```bash
# Nhận diện liên tục mọi frame (mặc định)
py live_inspector.py --camera 0 --fps 20 --mode continuous

# Tự động chụp khi sản phẩm ổn định (cho dây chuyền)
py live_inspector.py --camera 0 --mode auto-capture

# Preview live, nhấn Space để trigger phân tích (kiểm tra thủ công)
py live_inspector.py --camera 0 --mode manual-trigger
```

### Cách 3: Tùy chỉnh camera và resolution

```bash
# Dùng camera USB thứ 2 (ID=1), resolution 800x600
py live_inspector.py --camera 1 --resolution 800x600 --fps 15
```

### Phím tắt khi chạy Live

| Phím | Chức năng |
|------|-----------|
| `q` | Thoát |
| `s` | Chụp screenshot |
| `Space` | Manual trigger (chế độ manual-trigger) |
| `m` | Chuyển đổi chế độ hoạt động |

### 3 Chế Độ Hoạt Động

| Chế độ | Mô tả | Phù hợp cho |
|--------|--------|-------------|
| **continuous** | Nhận diện liên tục mọi frame | Demo, test nhanh |
| **auto-capture** | Tự động chụp + log khi sản phẩm ổn định | **Dây chuyền sản xuất** |
| **manual-trigger** | Preview live, Space để trigger | Kiểm tra thủ công |

Tính năng live:
- Multi-thread (camera + processing + display)
- Temporal smoothing + hysteresis (tránh kết quả nhảy)
- OSD overlay (bounding box, confidence bar, FPS, statistics)
- Product tracking (phát hiện sản phẩm mới)
- Auto-save ảnh NG + CSV log

---

## 🌐 Mở Web Dashboard (Tùy chọn)

```bash
streamlit run app.py
```

Hoặc double-click file **`run_app.bat`**. Trình duyệt sẽ tự mở tại `http://localhost:8501`.

---

## 🎓 Huấn Luyện Model (Tùy chọn)

```bash
# 1. Bỏ ảnh sản phẩm OK vào: data/training_images/labeled/OK/
# 2. Bỏ ảnh sản phẩm NG vào: data/training_images/labeled/NG/
# 3. Chạy train:
py training/train_hf.py
```

---

## 📁 Cấu Trúc Thư Mục

```
defect_detection_huggingface/
├── README.md                  # File này
├── requirements.txt           # Thư viện phụ thuộc
├── config.py                  # Cấu hình tập trung
├── run_demo.bat               # CLI Runner tương tác (Windows)
├── run_app.bat                # Chạy Streamlit 1-click
│
├── data/
│   ├── raw_images/            # Ảnh gốc để test inference
│   ├── reference_templates/   # Ảnh reference cho Feature Matching
│   └── training_images/labeled/
│       ├── OK/                # Ảnh sản phẩm đạt chuẩn
│       └── NG/                # Ảnh sản phẩm lỗi
│
├── preprocessing/
│   ├── image_processor.py     # Pipeline 6 bước (HSV+LAB copper detection)
│   └── object_detector.py     # Feature Matching product detector
│
├── training/
│   ├── dataset.py             # Dataset + Data Augmentation
│   └── train_hf.py            # Fine-tuning ResNet-18
│
├── inference/
│   └── predict.py             # QualityInspector (CV-first + AI-boost)
│
├── live_inspector.py          # 📹 Nhận diện live từ camera 20fps
├── app.py                     # 🌐 Web Dashboard Streamlit
│
└── output/
    ├── model/                 # Model đã train
    ├── predictions/           # Ảnh kết quả inference
    └── ng_captures/           # Ảnh NG tự động capture (live mode)
```

---

## 🤖 Pipeline Xử Lý Ảnh (V3)

```
[Ảnh đầu vào (bất kỳ góc camera / live camera)]
        │
        ▼ (OpenCV 6 bước)
1. Tách nền (HSV blue detection + Otsu)
2. PCA xác định trục chính
3. Xoay thẳng + thickness-based xác định hướng đầu cos
4. Cắt adaptive (terminal end detection)
5. HSV + LAB direct range copper detection
6. ROI crop + pipeline confidence scoring
        │
        ▼ (ROI 224×224)
[Hugging Face ImageProcessor] → [ResNet-18] → [OK / NG + confidence %]
```

---

## ⚙️ Cấu Hình Quan Trọng (`config.py`)

| Biến | Mặc định | Mô tả |
|---|---|---|
| `MIN_COPPER_RATIO` | 0.02 | Ngưỡng dưới: copper < 2% → OK |
| `MAX_COPPER_RATIO` | 0.15 | Ngưỡng trên: copper > 15% → NG |
| `CONFIDENCE_THRESHOLD` | 0.70 | Ngưỡng tin cậy phân loại OK/NG |
| `CAMERA_ID` | 0 | ID camera (0 = webcam mặc định) |
| `TARGET_FPS` | 20 | FPS mục tiêu live mode |
| `SMOOTHING_WINDOW` | 7 | Số frame dùng cho temporal smoothing |
| `USE_ONNX` | False | Bật ONNX Runtime tăng tốc |

---

## 🎯 Mô Hình Có Thể Thay Thế

Đổi `MODEL_NAME` trong `config.py`:

| Mô hình | Model ID | Tham số | Ưu điểm |
| :--- | :--- | :--- | :--- |
| **ResNet-18** *(mặc định)* | `microsoft/resnet-18` | ~11M | Nhẹ, nhanh trên CPU |
| **ConvNeXt** | `facebook/convnext-tiny-224` | ~28M | Chính xác cao hơn |
| **MobileNetV2** | `google/mobilenet_v2_1.0_224` | ~3.5M | Siêu nhẹ, cho edge devices |
| **ViT** | `google/vit-tiny-patch16-224` | ~5.7M | Transformer-based |

---

## ❓ Xử Lý Lỗi Thường Gặp

| Lỗi | Nguyên nhân | Cách fix |
|-----|-------------|----------|
| `metadata-generation-failed` numpy | Python version không tương thích | Cài Python 3.12, dùng `py -3.12 -m pip install -r requirements.txt` |
| `fatal: not a git repository` | Chưa clone repo | Chạy `git clone` trước, không phải `git pull` |
| `local changes would be overwritten` | File local bị conflict | Chạy `git stash` rồi `git pull origin main` |
| Không thể mở camera | Camera không kết nối hoặc sai ID | Thử `--camera 1`, `--camera 2` |
| Model tải chậm lần đầu | Đang tải từ Hugging Face Hub | Chờ ~1 phút, cần Internet |



