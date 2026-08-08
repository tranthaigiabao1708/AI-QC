import os
from pathlib import Path

# Thư mục gốc của project
PROJECT_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────
# ĐƯỜNG DẪN DỮ LIỆU
# Tất cả đường dẫn đều relative so với PROJECT_DIR để hoạt động
# đúng trên mọi máy khi clone từ GitHub.
# ─────────────────────────────────────────────────────────────

# Thư mục dữ liệu training (nằm trong project)
DATA_DIR = PROJECT_DIR / "data" / "training_images" / "labeled"

# Thư mục ảnh gốc để test inference (nằm trong project)
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw_images"

# Thư mục ảnh reference templates cho Feature Matching detector
REFERENCE_DIR = PROJECT_DIR / "data" / "reference_templates"
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

# Thư mục kết quả đầu ra
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = OUTPUT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# CẤU HÌNH AI ENGINEER & HUGGING FACE
# -------------------------------------------------------------

# Lựa chọn mô hình mặc định từ Hugging Face Hub
MODEL_NAME = "microsoft/resnet-18"

# -------------------------------------------------------------
# THAM SỐ HUẤN LUYỆN & XỬ LÝ ẢNH
# -------------------------------------------------------------
IMAGE_SIZE = (224, 224)  # Kích thước ảnh đầu vào mô hình
BATCH_SIZE = 4           # Kích thước batch (phù hợp với tập dữ liệu nhỏ và chạy CPU)
EPOCHS = 15              # Số lượng epoch huấn luyện
LEARNING_RATE = 1e-4     # Tốc độ học (thấp để tránh phá hỏng trọng số pre-trained)

# Số lượng ảnh sinh thêm bằng Data Augmentation cho mỗi ảnh gốc
N_AUGMENT = 15

# Ngưỡng tin cậy dùng trong suy luận thực tế (OK vs NG)
CONFIDENCE_THRESHOLD = 0.70

# Chiều dài cắt cố định của sản phẩm (pixel) — giữ lại cho backward compatibility
FIXED_CROP_LENGTH = 300

# Ngưỡng vật lý bổ trợ (Tỷ lệ diện tích đồng lộ ra trong ROI)
MIN_COPPER_RATIO = 0.02
MAX_COPPER_RATIO = 0.15

# Nhãn tương ứng với chỉ số lớp
CLASS_NAMES = ["OK", "NG"]

# -------------------------------------------------------------
# CẤU HÌNH PIPELINE ADAPTIVE (MỚI)
# -------------------------------------------------------------
# Tỷ lệ crop so với chiều dài sản phẩm (thay thế FIXED_CROP_LENGTH khi dùng pipeline mới)
CROP_RATIO = 0.45

# Tỷ lệ diện tích contour tối thiểu so với diện tích ảnh (thay thế min_contour_area cố định)
MIN_CONTOUR_RATIO = 0.02

# Số cluster cho K-means khi phát hiện đồng lộ
COPPER_KMEANS_CLUSTERS = 5

# -------------------------------------------------------------
# CẤU HÌNH LIVE CAMERA DETECTION (MỚI)
# -------------------------------------------------------------

# Camera
CAMERA_ID = 0                         # ID camera mặc định (0 = webcam tích hợp/USB đầu tiên)
TARGET_FPS = 20                       # FPS mục tiêu cho live mode
LIVE_RESOLUTION = (640, 480)          # Resolution cho live mode (width, height)

# Xử lý frame
PROCESS_EVERY_N_FRAMES = 3            # Chạy full pipeline mỗi N frame, frame giữa dùng cache
FRAME_RESIZE_FOR_PROCESSING = (640, 480)  # Resize ảnh trước khi xử lý pipeline

# Temporal Smoothing (Ổn định kết quả giữa các frame)
SMOOTHING_WINDOW = 7                  # Số frame gần nhất dùng cho moving average
HYSTERESIS_OK_THRESHOLD = 0.85        # Ngưỡng confidence để chuyển NG → OK
HYSTERESIS_NG_THRESHOLD = 0.80        # Ngưỡng confidence để chuyển OK → NG
HYSTERESIS_MIN_FRAMES = 5             # Số frame liên tục cần vượt ngưỡng để chuyển trạng thái

# Product Tracking
NEW_PRODUCT_DISTANCE = 100            # Pixel di chuyển tâm contour để nhận là sản phẩm mới
PRODUCT_STABLE_DELAY = 0.5            # Giây chờ sản phẩm ổn định trước khi auto-capture

# ONNX Runtime (tùy chọn, tăng tốc inference ~2-3x)
USE_ONNX = False                      # True = dùng ONNX runtime, False = dùng PyTorch
ONNX_MODEL_PATH = OUTPUT_DIR / "model.onnx"

# Thư mục lưu ảnh NG tự động capture và log CSV
NG_CAPTURES_DIR = OUTPUT_DIR / "ng_captures"
NG_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
LIVE_LOG_PATH = OUTPUT_DIR / "live_qc_log.csv"
