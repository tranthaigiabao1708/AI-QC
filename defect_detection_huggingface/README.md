# 🔍 Hệ Thống Nhận Diện Sản Phẩm Lỗi — Computer Vision (Hugging Face + PyTorch)

Dự án này là một hệ thống kiểm tra chất lượng sản phẩm **cos đồng crimping** tự động (Quality Control) sử dụng kết hợp **Computer Vision truyền thống (OpenCV)** để tiền xử lý/cắt vùng đặc trưng (ROI) và **Deep Learning (Hugging Face + PyTorch)** để phân loại lỗi sản phẩm (OK/NG).

**Tính năng nổi bật:**
- ✅ Pipeline 6 bước bền vững — hoạt động với mọi góc camera (K-means + multi-colorspace fusion)
- ✅ Nhận diện live từ camera ở 20fps (multi-thread, temporal smoothing, OSD overlay)
- ✅ Web dashboard Streamlit tương tác
- ✅ Hỗ trợ ONNX Runtime tăng tốc inference

---

## 🚀 Clone Về Máy Khác & Chạy Ngay

### Bước 1: Clone repository

```bash
git clone https://github.com/tranthaigiabao1708/AI-QC.git
cd AI-QC/defect_detection_huggingface
```

### Bước 2: Cài đặt thư viện

```bash
# Khuyến nghị dùng Python 3.10 - 3.12
py -m pip install -r requirements.txt
```

> **Lưu ý:** Lần đầu chạy, hệ thống sẽ tự động tải model `microsoft/resnet-18` (~44MB) từ Hugging Face Hub. Cần có kết nối Internet.

### Bước 3: Chạy thử inference trên ảnh mẫu

```bash
py inference/predict.py
```

Kết quả sẽ được lưu tại `output/predictions/`.

### Bước 4 (Tùy chọn): Huấn luyện model trên dữ liệu của bạn

```bash
# 1. Bỏ ảnh sản phẩm OK vào: data/training_images/labeled/OK/
# 2. Bỏ ảnh sản phẩm NG vào: data/training_images/labeled/NG/
# 3. Chạy train:
py training/train_hf.py
```

### Bước 5 (Tùy chọn): Chạy Live Camera Detection

```bash
# Nhận diện liên tục từ camera USB/webcam
py live_inspector.py --camera 0 --fps 20 --mode continuous

# Tự động chụp khi sản phẩm ổn định
py live_inspector.py --camera 0 --mode auto-capture

# Preview live, nhấn Space để trigger phân tích
py live_inspector.py --camera 0 --mode manual-trigger
```

**Phím tắt khi chạy Live:** `q` thoát | `s` screenshot | `Space` trigger | `m` đổi mode

### Bước 6 (Tùy chọn): Mở Web Dashboard

```bash
streamlit run app.py
```

Trình duyệt sẽ tự mở tại `http://localhost:8501`.

---

## 📁 Cấu Trúc Thư Mục

```
defect_detection_huggingface/
├── README.md                  # File này
├── requirements.txt           # Thư viện phụ thuộc
├── config.py                  # Cấu hình tập trung (model, paths, thresholds, live camera)
├── run_demo.bat               # CLI Runner tương tác (Windows)
├── run_app.bat                # Chạy Streamlit 1-click
│
├── data/                      # DỮ LIỆU (ảnh mẫu + training data)
│   ├── raw_images/            # Ảnh gốc để test inference
│   └── training_images/
│       └── labeled/
│           ├── OK/            # Ảnh sản phẩm đạt chuẩn
│           └── NG/            # Ảnh sản phẩm lỗi
│
├── preprocessing/             # Pipeline OpenCV 6 bước (Robust V2)
│   ├── __init__.py
│   └── image_processor.py     # HSV bg removal, PCA rotation, K-means copper detection
│
├── training/                  # Huấn luyện model
│   ├── __init__.py
│   ├── dataset.py             # Dataset + Data Augmentation (perspective, shear, scale, cutout)
│   └── train_hf.py            # Fine-tuning ResNet-18 từ Hugging Face
│
├── inference/                 # Suy luận
│   ├── __init__.py
│   └── predict.py             # QualityInspector (PyTorch + ONNX support)
│
├── tools/
│   └── export_onnx.py         # Xuất model sang ONNX để tăng tốc inference
│
├── live_inspector.py          # 📹 Nhận diện live từ camera 20fps (MỚI)
├── app.py                     # 🌐 Web Dashboard Streamlit (ảnh tĩnh + live camera)
│
└── output/                    # Kết quả đầu ra (tự tạo khi chạy)
    ├── model/                 # Model đã train (safetensors)
    ├── predictions/           # Ảnh kết quả inference
    └── ng_captures/           # Ảnh NG tự động capture (live mode)
```

---

## 🤖 Pipeline Xử Lý Ảnh (Robust V2)

```
[Ảnh đầu vào (bất kỳ góc camera)]
        │
        ▼ (OpenCV 6 bước — Adaptive Pipeline)
1. Tách nền (HSV blue detection + Otsu)
2. PCA xác định trục chính
3. Xoay thẳng + phân tích sáng xác định hướng đầu cos
4. Cắt adaptive (% chiều dài sản phẩm)
5. K-means copper detection + multi-colorspace voting (LAB+HSV+YCrCb)
6. ROI + pipeline confidence scoring
        │
        ▼ (ROI 224×224)
[Hugging Face ImageProcessor] → [ResNet-18 Fine-tuned] → [OK / NG + confidence %]
```

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

## 📹 Live Camera Detection

3 chế độ hoạt động:

| Chế độ | Mô tả | Lệnh |
|---|---|---|
| **continuous** | Nhận diện liên tục mọi frame | `py live_inspector.py --mode continuous` |
| **auto-capture** | Tự động chụp + log khi sản phẩm ổn định | `py live_inspector.py --mode auto-capture` |
| **manual-trigger** | Preview live, Space để trigger | `py live_inspector.py --mode manual-trigger` |

Tính năng:
- Multi-thread (camera + processing + display)
- Temporal smoothing + hysteresis (tránh kết quả nhảy)
- OSD overlay (bounding box, confidence bar, FPS, statistics)
- Product tracking (phát hiện sản phẩm mới)
- Auto-save ảnh NG + CSV log

---

## ⚙️ Cấu Hình Quan Trọng (`config.py`)

| Biến | Mặc định | Mô tả |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.70 | Ngưỡng tin cậy phân loại OK/NG |
| `CROP_RATIO` | 0.45 | Tỷ lệ crop so với chiều dài sản phẩm |
| `CAMERA_ID` | 0 | ID camera (0 = webcam mặc định) |
| `TARGET_FPS` | 20 | FPS mục tiêu live mode |
| `SMOOTHING_WINDOW` | 7 | Số frame dùng cho temporal smoothing |
| `USE_ONNX` | False | Bật ONNX Runtime tăng tốc |
