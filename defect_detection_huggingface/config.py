import os
from pathlib import Path

# Thư mục gốc của project
PROJECT_DIR = Path(__file__).resolve().parent

# Thư mục dữ liệu (kế thừa từ thư mục đã có trong workspace)
WORKSPACE_DIR = PROJECT_DIR.parent
DATA_DIR = WORKSPACE_DIR / "training_images" / "labeled"
RAW_DATA_DIR = WORKSPACE_DIR / "raw_images"

# Thư mục kết quả đầu ra
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = OUTPUT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# CẤU HÌNH AI ENGINEER & HUGGING FACE
# -------------------------------------------------------------

# Lựa chọn mô hình mặc định từ Hugging Face Hub
# Bạn có thể đổi sang các mô hình khác dễ dàng như:
# - "google/vit-tiny-patch16-224" (Vision Transformer cực nhẹ)
# - "google/vit-base-patch16-224" (Vision Transformer chuẩn)
# - "facebook/convnext-tiny-224" (ConvNeXt hiện đại)
# - "google/mobilenet_v2_1.0_224" (MobileNetV2 siêu nhẹ cho di động)
MODEL_NAME = "microsoft/resnet-18"

# -------------------------------------------------------------
# THAM SỐ HUẤN LUYỆN & XỬ LÝ ẢNH
# -------------------------------------------------------------
IMAGE_SIZE = (224, 224)  # Kích thước ảnh đầu vào mô hình
BATCH_SIZE = 4           # Kích thước batch (phù hợp với tập dữ liệu nhỏ và chạy CPU)
EPOCHS = 15              # Số lượng epoch huấn luyện
LEARNING_RATE = 1e-4     # Tốc độ học (thấp để tránh phá hỏng trọng số pre-trained)

# Số lượng ảnh sinh thêm bằng Data Augmentation cho mỗi ảnh gốc
# Vì dữ liệu gốc rất nhỏ (3 ảnh OK, 3 ảnh NG), augmentation giúp nhân bản dữ liệu
N_AUGMENT = 15 

# Ngưỡng tin cậy dùng trong suy luận thực tế (OK vs NG)
CONFIDENCE_THRESHOLD = 0.70

# Chiều dài cắt cố định của sản phẩm (pixel)
FIXED_CROP_LENGTH = 300

# Ngưỡng vật lý bổ trợ (Tỷ lệ diện tích đồng lộ ra trong ROI)
# Nếu tỷ lệ đồng nằm ngoài khoảng [MIN, MAX], sản phẩm sẽ bị coi là NG
MIN_COPPER_RATIO = 0.02
MAX_COPPER_RATIO = 0.15

# Nhãn tương ứng với chỉ số lớp
CLASS_NAMES = ["OK", "NG"]
