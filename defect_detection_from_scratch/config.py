import os
from pathlib import Path

# Thư mục gốc của project thực hành
PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent

# Thư mục dữ liệu (Kế thừa dữ liệu có sẵn từ workspace)
DATA_DIR = WORKSPACE_DIR / "training_images" / "labeled"
RAW_DATA_DIR = WORKSPACE_DIR / "raw_images"

# Thư mục kết quả đầu ra
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = OUTPUT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# TODO: HÃY TỰ ĐIỀN CÁC THAM SỐ CẤU HÌNH DƯỚI ĐÂY
# =====================================================================

# 1. Điền Model ID từ Hugging Face Hub (Gợi ý: "microsoft/resnet-18" hoặc "google/vit-tiny-patch16-224")
MODEL_NAME = None # TODO: Hãy gõ tên mô hình dạng chuỗi ký tự ở đây (Ví dụ: "microsoft/resnet-18")

# 2. Định nghĩa kích thước ảnh đầu vào của mô hình (Dòng ResNet và ViT mặc định dùng 224x224)
IMAGE_SIZE = (224, 224) 

# 3. Điền kích thước batch size (Gợi ý: 4 hoặc 8 cho tập dữ liệu nhỏ chạy trên CPU)
BATCH_SIZE = None # TODO: Điền số nguyên thích hợp (Ví dụ: 4)

# 4. Điền số lượng vòng lặp huấn luyện (Epochs) mong muốn (Gợi ý: từ 10 đến 20)
EPOCHS = None # TODO: Điền số nguyên thích hợp (Ví dụ: 15)

# 5. Tốc độ học (Learning Rate) khi huấn luyện chuyển giao (Gợi ý: từ 1e-4 đến 3e-4)
LEARNING_RATE = None # TODO: Điền số thực thích hợp (Ví dụ: 1e-4)

# 6. Số lượng ảnh sinh thêm bằng Data Augmentation từ mỗi ảnh gốc (Gợi ý: 10 đến 15)
N_AUGMENT = None # TODO: Điền số nguyên thích hợp (Ví dụ: 15)

# 7. Ngưỡng tin cậy phân loại OK vs NG (Gợi ý: 0.70)
CONFIDENCE_THRESHOLD = 0.70

# Các nhãn phân loại tương ứng với vị trí (0: OK, 1: NG)
CLASS_NAMES = ["OK", "NG"]
