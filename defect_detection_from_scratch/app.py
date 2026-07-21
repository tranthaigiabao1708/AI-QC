import os
import sys
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Thêm thư mục hiện tại vào PYTHONPATH để import
sys.path.append(str(Path(__file__).resolve().parent))
import config
from inference import QualityInspector

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="QC Dashboard Practice - From Scratch",
    page_icon="🔍",
    layout="wide"
)

# Khởi tạo QualityInspector
@st.cache_resource
def get_inspector():
    return QualityInspector()

try:
    inspector = get_inspector()
    model_loaded = True
except Exception as e:
    st.error(f"Lỗi tải mô hình: {e}")
    model_loaded = False

# --- SIDEBAR CẤU HÌNH ---
st.sidebar.title("⚙️ Cấu hình QC")

# Thanh trượt thiết lập Ngưỡng tin cậy (Confidence Threshold)
conf_threshold = st.sidebar.slider(
    "Ngưỡng tin cậy phân loại (Threshold)",
    min_value=0.50,
    max_value=0.99,
    value=0.70,
    step=0.01
)
config.CONFIDENCE_THRESHOLD = conf_threshold

# Lựa chọn dữ liệu mẫu để chạy thử
st.sidebar.subheader("🖼️ Dữ liệu chạy thử")
sample_images = ["Tải lên ảnh mới..."]
if config.RAW_DATA_DIR.exists():
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    raw_files = sorted([f.name for f in config.RAW_DATA_DIR.iterdir() if f.suffix.lower() in exts])
    sample_images.extend(raw_files)

selected_sample = st.sidebar.selectbox("Chọn ảnh mẫu:", sample_images)

# --- PANEL CHÍNH ---
st.title("🔍 Thực Hành: AI Quality Control Dashboard")
st.markdown("---")

img_bgr = None

# Đọc ảnh được chọn/tải lên
if selected_sample == "Tải lên ảnh mới...":
    uploaded_file = st.file_uploader("Chọn ảnh sản phẩm cần kiểm tra:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
else:
    img_path = config.RAW_DATA_DIR / selected_sample
    if img_path.exists():
        img_bgr = cv2.imread(str(img_path))
        st.info(f"Đang phân tích ảnh: **{selected_sample}**")

# =====================================================================
# TODO: HÃY TỰ VIẾT LOGIC GIAO DIỆN WEB HIỂN THỊ KẾT QUẢ QC
# =====================================================================

if img_bgr is not None and model_loaded:
    
    # 1. Chạy inspect ảnh đầu vào bằng đối tượng inspector
    # Gợi ý: result = inspector.inspect(img_bgr)
    result = None # TODO: Thực hiện chạy suy luận
    
    if result is not None and result["success"]:
        decision = result["decision"]
        confidence = result["confidence"]
        vis_img = result["vis_img"]
        roi = result["roi"]
        vis_steps = result["vis_steps"]
        
        # 2. Tạo một Banner lớn thông báo kết quả OK hoặc NG
        # Gợi ý: Dùng st.success() hoặc st.error() tùy thuộc vào biến decision chứa nhãn "OK" hay "NG"
        # TODO: Tạo banner kết quả lớn
        
        # 3. Dựng layout 2 cột hiển thị ảnh trực quan và số liệu thống kê
        # Gợi ý: col1, col2 = st.columns([2, 1])
        # Cột 1 hiển thị vis_img vẽ bounding box (Chú ý chuyển BGR -> RGB trước khi hiển thị).
        # Cột 2 hiển thị ảnh ROI (roi) và số liệu % confidence của lớp dự đoán.
        # TODO: Tạo layout 2 cột chính
        
        # 4. Tạo Expander để hiển thị trực quan các bước xử lý ảnh OpenCV
        # Gợi ý:
        # with st.expander("Chi tiết 6 bước xử lý ảnh OpenCV"):
        #   Dựng lưới cột (ví dụ: st.columns(3)) để vẽ các ảnh lưu trong dict vis_steps
        # TODO: Dựng expander và hiển thị 6 ảnh của pipeline OpenCV
        
        pass # Xóa dòng pass này khi viết code
        
    else:
        st.error("Không thể xử lý ảnh này.")
        
else:
    st.markdown("### 📥 Hướng dẫn:")
    st.write("Hãy chọn ảnh mẫu ở sidebar bên trái hoặc tải lên ảnh mới để chạy QC.")
