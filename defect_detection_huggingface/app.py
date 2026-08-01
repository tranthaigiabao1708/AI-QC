import os
import sys
import time
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent))
import config
from inference import QualityInspector

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="AI QC Dashboard - Hugging Face + PyTorch",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để giao diện trông hiện đại và chuyên nghiệp hơn
st.markdown("""
<style>
    .main {
        background-color: #0f111a;
        color: #ffffff;
    }
    .stApp {
        background-color: #0f111a;
    }
    h1, h2, h3 {
        color: #00ffcc !important;
        font-family: 'Outfit', sans-serif;
    }
    .reportview-container {
        background: #0f111a;
    }
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #1f2937;
        border-radius: 10px;
        background-color: #111827;
    }
    .stSlider > label {
        color: #00ffcc;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo Inspector trực tiếp để tránh các lỗi cache của Streamlit trên Windows
def get_inspector():
    return QualityInspector()

try:
    inspector = get_inspector()
    model_loaded = True
except Exception as e:
    import traceback
    st.error(f"Lỗi khởi tạo mô hình: {e}")
    st.code(traceback.format_exc())
    model_loaded = False

# --- GIAO DIỆN SIDEBAR ---
st.sidebar.title("⚙️ Cấu hình Hệ thống")

# Hiển thị trạng thái mô hình
is_fine_tuned = (Path(config.MODEL_DIR) / "config.json").exists()
if is_fine_tuned:
    st.sidebar.success("🟢 Model: Fine-tuned Local model")
else:
    st.sidebar.warning("🟡 Model: Hugging Face Baseline (Chưa train)")

st.sidebar.text(f"Backbone: {config.MODEL_NAME}")

# === CHẾ ĐỘ HOẠT ĐỘNG ===
st.sidebar.subheader("📌 Chế độ hoạt động")
app_mode = st.sidebar.radio(
    "Chọn chế độ:",
    ["📷 Ảnh tĩnh (Upload/Chọn mẫu)", "📹 Live Camera"],
    index=0
)

# Thanh trượt điều chỉnh Ngưỡng tin cậy (Confidence Threshold)
conf_threshold = st.sidebar.slider(
    "Ngưỡng tin cậy phân loại (Threshold)",
    min_value=0.50,
    max_value=0.99,
    value=float(config.CONFIDENCE_THRESHOLD),
    step=0.01,
    help="Nếu mô hình dự đoán OK với độ tin cậy thấp hơn ngưỡng này, hệ thống sẽ tự động gán nhãn NG để kiểm tra lại bằng tay."
)

# Thanh trượt điều chỉnh tỷ lệ đồng lộ tối thiểu và tối đa (Quy tắc vật lý)
min_copper_ratio = st.sidebar.slider(
    "Tỷ lệ đồng lộ tối thiểu (Min Copper %)",
    min_value=0.00,
    max_value=0.20,
    value=float(config.MIN_COPPER_RATIO),
    step=0.01,
    help="Sản phẩm bị coi là lỗi (NG) nếu tỷ lệ đồng lộ ra nhỏ hơn ngưỡng này."
)

max_copper_ratio = st.sidebar.slider(
    "Tỷ lệ đồng lộ tối đa (Max Copper %)",
    min_value=0.05,
    max_value=0.40,
    value=float(config.MAX_COPPER_RATIO),
    step=0.01,
    help="Sản phẩm bị coi là lỗi (NG) nếu tỷ lệ đồng lộ ra lớn hơn ngưỡng này."
)

# Cập nhật cấu hình vào config động cho phiên chạy này
config.CONFIDENCE_THRESHOLD = conf_threshold
config.MIN_COPPER_RATIO = min_copper_ratio
config.MAX_COPPER_RATIO = max_copper_ratio

# Gán trực tiếp giá trị cấu hình cho đối tượng inspector để cập nhật tức thời
if model_loaded:
    inspector.confidence_threshold = conf_threshold
    inspector.min_copper_ratio = min_copper_ratio
    inspector.max_copper_ratio = max_copper_ratio

# --- TIÊU ĐỀ CHÍNH ---
st.title("🔍 Hệ Thống Kiểm Tra Chất Lượng Sản Phẩm Cos Đồng")
st.markdown("---")


# ═════════════════════════════════════════════════════════════════
# CHẾ ĐỘ 1: ẢNH TĨNH (Upload/Chọn mẫu) — GIỮ NGUYÊN LOGIC CŨ
# ═════════════════════════════════════════════════════════════════
if app_mode == "📷 Ảnh tĩnh (Upload/Chọn mẫu)":
    # Lựa chọn ảnh test nhanh
    st.sidebar.subheader("🖼️ Dữ liệu chạy thử nghiệm")
    sample_images = ["Tải lên ảnh mới..."]
    if config.RAW_DATA_DIR.exists():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        raw_files = sorted([f.name for f in config.RAW_DATA_DIR.iterdir() if f.suffix.lower() in exts])
        sample_images.extend(raw_files)

    selected_sample = st.sidebar.selectbox("Chọn ảnh mẫu để test nhanh:", sample_images)

    # Xử lý chọn ảnh
    img_bgr = None
    uploaded_file = None

    if selected_sample == "Tải lên ảnh mới...":
        uploaded_file = st.file_uploader("Kéo và thả hoặc Chọn ảnh từ máy tính của bạn:", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        img_path = config.RAW_DATA_DIR / selected_sample
        if img_path.exists():
            img_bgr = cv2.imread(str(img_path))
            st.info(f"Đang hiển thị ảnh mẫu: **{selected_sample}**")

    # --- XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ ---
    if img_bgr is not None and model_loaded:
        # Chạy suy luận qua hệ thống QualityInspector
        with st.spinner("Đang xử lý ảnh qua pipeline OpenCV và suy luận với mô hình Hugging Face..."):
            result = inspector.inspect(img_bgr)

        if result["success"]:
            # Tách kết quả để hiển thị
            decision = result["decision"]
            confidence = result["confidence"]
            vis_img = result["vis_img"]
            roi = result["roi"]
            vis_steps = result["vis_steps"]
            copper_ratio = result["copper_ratio"]
            pipeline_conf = result.get("pipeline_confidence", 1.0)

            # 1. Hiển thị Banner Kết Quả Lớn
            if "OK" in decision and "NG" not in decision:
                st.markdown(
                    f"<div style='background-color:#1b5e20; padding:20px; border-radius:10px; border-left: 8px solid #4caf50; margin-bottom: 20px;'>"
                    f"<h2 style='margin:0; color:#4caf50 !important;'>✅ KẾT QUẢ QC: ĐẠT CHUẨN (OK)</h2>"
                    f"<p style='margin:5px 0 0 0; font-size:16px; color:#ffffff;'>Mô hình dự đoán: {result['raw_decision']} | "
                    f"Độ tin cậy: {confidence:.2%} | Tỷ lệ đồng lộ: {copper_ratio:.1%} | Pipeline: {pipeline_conf:.0%}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background-color:#b71c1c; padding:20px; border-radius:10px; border-left: 8px solid #f44336; margin-bottom: 20px;'>"
                    f"<h2 style='margin:0; color:#f44336 !important;'>❌ KẾT QUẢ QC: LỖI SẢN PHẨM (NG)</h2>"
                    f"<p style='margin:5px 0 0 0; font-size:16px; color:#ffffff;'>Mô hình dự đoán: {result['raw_decision']} | "
                    f"Độ tin cậy: {confidence:.2%} | Tỷ lệ đồng lộ: {copper_ratio:.1%} | Pipeline: {pipeline_conf:.0%}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # 2. Layout hai cột chính: Ảnh kết quả & Đồ thị phân phối lớp
            col_img, col_metrics = st.columns([2, 1])

            with col_img:
                st.subheader("🖼️ Trực quan hóa suy luận")
                vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                st.image(vis_img_rgb, use_container_width=True, caption="Ảnh gốc được vẽ bounding box và kết quả phân loại")

            with col_metrics:
                st.subheader("📊 Số liệu chi tiết")

                st.markdown("**Vùng đặc trưng ROI cắt bởi OpenCV (224x224):**")
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                st.image(roi_rgb, width=180, caption="Vùng ROI đưa vào model")

                st.markdown(f"**Phân tích xác suất mô hình:**")

                ok_prob = confidence if result["raw_decision"] == "OK" else (1.0 - confidence)
                ng_prob = confidence if result["raw_decision"] == "NG" else (1.0 - confidence)

                st.write("Độ tin cậy lớp OK:")
                st.progress(int(ok_prob * 100))
                st.caption(f"{ok_prob:.2%}")

                st.write("Độ tin cậy lớp NG:")
                st.progress(int(ng_prob * 100))
                st.caption(f"{ng_prob:.2%}")

                st.markdown("---")
                st.markdown("**Thông số đo đạc vật lý (OpenCV):**")
                st.write(f"- Tỷ lệ diện tích đồng lộ: `{copper_ratio:.2%}`")
                st.write(f"- Kích thước vùng đồng: `{result['copper_w']} x {result['copper_h']} px`")
                st.write(f"- Pipeline confidence: `{pipeline_conf:.0%}`")

            # 3. Mục chi tiết Pipeline xử lý ảnh (OpenCV 6 bước)
            st.markdown("---")
            with st.expander("⚙️ Xem chi tiết Pipeline tiền xử lý ảnh 6 bước bằng OpenCV"):
                st.markdown("""
                Trước khi đưa vào mô hình học sâu của Hugging Face, ảnh gốc phải trải qua một pipeline xử lý bằng OpenCV.
                Quá trình này giúp loại bỏ background nhiễu, căn thẳng đầu cos và chỉ cắt đúng vùng ROI đặc trưng cần kiểm tra.
                """)

                steps_keys = [
                    ("01_bg_removed", "Bước 1: Tách nền", "HSV-based + Otsu adaptive (bỏ ngưỡng cứng)."),
                    ("02_contour", "Bước 2: Tìm contour", "PCA xác định trục chính và tâm trọng lực."),
                    ("03_rotated", "Bước 3: Xoay thẳng", "Xoay nắn thẳng + xác định hướng bằng phân tích sáng."),
                    ("04_standardized", "Bước 4: Cắt chuẩn hóa", "Crop adaptive theo % chiều dài sản phẩm."),
                    ("05_copper_detect", "Bước 5: Vùng đồng lộ", "K-means clustering + multi-colorspace fusion."),
                    ("06_roi_final", "Bước 6: Trích ROI", "Cắt vùng ROI và resize về 224x224.")
                ]

                for r in range(2):
                    cols = st.columns(3)
                    for c in range(3):
                        idx = r * 3 + c
                        key, title, desc = steps_keys[idx]
                        with cols[c]:
                            st.markdown(f"**{title}**")
                            st.caption(desc)
                            if key in vis_steps:
                                step_img_rgb = cv2.cvtColor(vis_steps[key], cv2.COLOR_BGR2RGB)
                                st.image(step_img_rgb, use_container_width=True)
                            else:
                                st.write("Không có hình ảnh bước này.")

        else:
            st.error(f"Xử lý ảnh thất bại: {result['error']}")

    elif img_bgr is None:
        st.markdown("### 📥 Hướng dẫn sử dụng:")
        st.markdown("""
        1. **Chọn một ảnh mẫu** ở cột bên trái hoặc **Tải lên ảnh mới** để kiểm tra ngay lập tức.
        2. **Điều chỉnh ngưỡng phân loại** (Confidence Threshold) nếu muốn kiểm soát chặt chẽ hơn tỷ lệ lọt lỗi.
        3. Hệ thống sẽ tự động hiển thị kết quả phân loại OK/NG cùng quá trình tiền xử lý ảnh OpenCV.

        *Lưu ý: Nếu mô hình đang ở trạng thái Baseline (chưa được train), kết quả dự đoán sẽ ngẫu nhiên. Hãy chạy huấn luyện trước bằng cách chạy script huấn luyện `train_hf.py`.*
        """)


# ═════════════════════════════════════════════════════════════════
# CHẾ ĐỘ 2: LIVE CAMERA — XỬ LÝ LIÊN TỤC KHÔNG CẦN ẤN CHỤP
# ═════════════════════════════════════════════════════════════════
elif app_mode == "📹 Live Camera":

    st.subheader("📹 Nhận Diện Trực Tiếp Từ Camera")

    # Sidebar config cho live mode
    st.sidebar.subheader("📹 Cấu hình Live Camera")
    camera_id = st.sidebar.number_input("Camera ID", min_value=0, max_value=10, value=config.CAMERA_ID)
    process_every = st.sidebar.slider("Xử lý mỗi N frame", min_value=1, max_value=10, value=3,
                                       help="Giảm để tăng tốc độ phản hồi, tăng để giảm tải CPU")

    # Khởi tạo session state
    if "live_running" not in st.session_state:
        st.session_state.live_running = False
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "stats" not in st.session_state:
        st.session_state.stats = {"total": 0, "ok": 0, "ng": 0}

    # Nút điều khiển
    col_start, col_stop, col_reset = st.columns(3)
    with col_start:
        start_btn = st.button("▶️ Bắt đầu Live", type="primary", use_container_width=True)
    with col_stop:
        stop_btn = st.button("⏹️ Dừng", use_container_width=True)
    with col_reset:
        reset_btn = st.button("🔄 Reset thống kê", use_container_width=True)

    if start_btn:
        st.session_state.live_running = True
    if stop_btn:
        st.session_state.live_running = False
    if reset_btn:
        st.session_state.stats = {"total": 0, "ok": 0, "ng": 0}

    st.markdown("---")

    # Layout chính: video feed + kết quả
    col_video, col_result = st.columns([2, 1])

    with col_video:
        video_placeholder = st.empty()
    with col_result:
        result_banner = st.empty()
        result_details = st.empty()
        roi_placeholder = st.empty()
        stats_placeholder = st.empty()

    status_bar = st.empty()

    # Pipeline debug section
    pipeline_expander = st.expander("⚙️ Debug Pipeline — Xem chi tiết từng bước xử lý ảnh", expanded=True)
    pipeline_placeholder = pipeline_expander.empty()

    # ─── VÒNG LẶP LIVE ───
    if st.session_state.live_running and model_loaded:
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            st.error(f"❌ Không thể mở camera ID={camera_id}. Kiểm tra camera đã kết nối chưa.")
            st.session_state.live_running = False
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            frame_count = 0
            last_result = st.session_state.last_result
            fps_start = time.time()
            fps_count = 0

            try:
                while st.session_state.live_running:
                    ret, frame = cap.read()
                    if not ret:
                        status_bar.warning("⚠️ Mất kết nối camera...")
                        break

                    frame_count += 1
                    fps_count += 1
                    display_frame = frame.copy()

                    # Xử lý pipeline mỗi N frame
                    if frame_count % process_every == 0:
                        try:
                            result = inspector.inspect(frame)
                            if result["success"]:
                                last_result = result
                                st.session_state.last_result = result

                                # Cập nhật thống kê
                                st.session_state.stats["total"] += 1
                                is_ok = "OK" in result["decision"] and "NG" not in result["decision"]
                                if is_ok:
                                    st.session_state.stats["ok"] += 1
                                else:
                                    st.session_state.stats["ng"] += 1
                        except Exception:
                            pass

                    # Vẽ overlay lên frame hiển thị
                    if last_result and last_result["success"]:
                        decision = last_result["decision"]
                        confidence = last_result["confidence"]
                        copper_ratio = last_result["copper_ratio"]
                        pipeline_conf = last_result.get("pipeline_confidence", 1.0)
                        is_ok = "OK" in decision and "NG" not in decision

                        # Vẽ text overlay trên frame
                        color = (0, 255, 0) if is_ok else (0, 0, 255)
                        label = f"{'OK' if is_ok else 'NG'} ({confidence:.0%})"
                        cv2.putText(display_frame, label, (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                        cv2.putText(display_frame, f"Copper: {copper_ratio:.1%} | Pipeline: {pipeline_conf:.0%}",
                                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        # Vẽ viền màu theo kết quả
                        border_color = (0, 200, 0) if is_ok else (0, 0, 200)
                        cv2.rectangle(display_frame, (0, 0),
                                      (display_frame.shape[1]-1, display_frame.shape[0]-1),
                                      border_color, 4)

                    # Tính FPS
                    elapsed = time.time() - fps_start
                    if elapsed > 0:
                        current_fps = fps_count / elapsed
                    else:
                        current_fps = 0
                    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Reset FPS counter mỗi 2 giây
                    if elapsed > 2.0:
                        fps_start = time.time()
                        fps_count = 0

                    # Hiển thị frame
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    # Cập nhật panel kết quả
                    if last_result and last_result["success"]:
                        decision = last_result["decision"]
                        is_ok = "OK" in decision and "NG" not in decision

                        if is_ok:
                            result_banner.success(f"✅ **OK** — {last_result['confidence']:.1%}")
                        else:
                            result_banner.error(f"❌ **NG** — {decision}")

                        result_details.markdown(
                            f"- Đồng lộ: **{last_result['copper_ratio']:.1%}**\n"
                            f"- Pipeline: **{last_result.get('pipeline_confidence', 0):.0%}**\n"
                            f"- Copper: {last_result['copper_w']}×{last_result['copper_h']}px"
                        )

                        try:
                            roi_rgb = cv2.cvtColor(last_result["roi"], cv2.COLOR_BGR2RGB)
                            roi_placeholder.image(roi_rgb, width=160, caption="ROI")
                        except Exception:
                            pass

                    # Cập nhật pipeline steps visualization
                    if last_result and last_result.get("vis_steps"):
                        vis_steps = last_result["vis_steps"]
                        steps_keys = [
                            ("01_bg_removed", "Bước 1: Tách nền", "Multi-strategy: HSV blue + wire detect + edge + Otsu"),
                            ("02_contour", "Bước 2: Tìm contour", "PCA trục chính + tâm trọng lực"),
                            ("03_rotated", "Bước 3: Xoay thẳng", "Xoay + phân tích sáng xác định hướng đầu cos"),
                            ("04_standardized", "Bước 4: Cắt chuẩn hóa", "Crop adaptive theo % chiều dài sản phẩm"),
                            ("05_copper_detect", "Bước 5: Vùng đồng lộ", "K-means + multi-colorspace voting"),
                            ("06_roi_final", "Bước 6: ROI", "Cắt ROI → resize 224×224")
                        ]
                        with pipeline_placeholder.container():
                            for r in range(2):
                                cols_pipe = st.columns(3)
                                for c in range(3):
                                    idx = r * 3 + c
                                    key, title, desc = steps_keys[idx]
                                    with cols_pipe[c]:
                                        st.markdown(f"**{title}**")
                                        st.caption(desc)
                                        if key in vis_steps:
                                            step_rgb = cv2.cvtColor(vis_steps[key], cv2.COLOR_BGR2RGB)
                                            st.image(step_rgb, use_container_width=True)
                                        else:
                                            st.warning("❌ Bước này thất bại")

                    # Cập nhật thống kê
                    s = st.session_state.stats
                    stats_placeholder.markdown(
                        f"### 📊 Thống kê\n"
                        f"- Tổng: **{s['total']}**\n"
                        f"- ✅ OK: **{s['ok']}**\n"
                        f"- ❌ NG: **{s['ng']}**"
                    )

                    # Delay nhỏ để không block hoàn toàn
                    time.sleep(0.03)

            except Exception as e:
                st.error(f"Lỗi camera: {e}")
            finally:
                cap.release()
                st.session_state.live_running = False
                status_bar.info("📷 Camera đã dừng.")

    elif not st.session_state.live_running:
        video_placeholder.info("📷 Nhấn **▶️ Bắt đầu Live** để bắt đầu nhận diện liên tục từ camera.")

        # Hiển thị kết quả cuối cùng nếu có
        if st.session_state.last_result and st.session_state.last_result["success"]:
            lr = st.session_state.last_result
            decision = lr["decision"]
            is_ok = "OK" in decision and "NG" not in decision
            if is_ok:
                result_banner.success(f"✅ Kết quả cuối: **OK** ({lr['confidence']:.1%})")
            else:
                result_banner.error(f"❌ Kết quả cuối: **NG** — {decision}")

            # Hiển thị pipeline steps cuối cùng khi đã dừng
            if lr.get("vis_steps"):
                vis_steps = lr["vis_steps"]
                steps_keys = [
                    ("01_bg_removed", "Bước 1: Tách nền", "Multi-strategy background removal"),
                    ("02_contour", "Bước 2: Tìm contour", "PCA + tâm trọng lực"),
                    ("03_rotated", "Bước 3: Xoay thẳng", "Xoay + xác định hướng"),
                    ("04_standardized", "Bước 4: Cắt chuẩn hóa", "Crop adaptive"),
                    ("05_copper_detect", "Bước 5: Vùng đồng lộ", "K-means + colorspace voting"),
                    ("06_roi_final", "Bước 6: ROI", "ROI 224×224")
                ]
                with pipeline_placeholder.container():
                    for r in range(2):
                        cols_pipe = st.columns(3)
                        for c in range(3):
                            idx = r * 3 + c
                            key, title, desc = steps_keys[idx]
                            with cols_pipe[c]:
                                st.markdown(f"**{title}**")
                                st.caption(desc)
                                if key in vis_steps:
                                    step_rgb = cv2.cvtColor(vis_steps[key], cv2.COLOR_BGR2RGB)
                                    st.image(step_rgb, use_container_width=True)
                                else:
                                    st.warning("❌ Bước này thất bại")

        s = st.session_state.stats
        if s["total"] > 0:
            stats_placeholder.markdown(
                f"### 📊 Thống kê phiên\n"
                f"- Tổng: **{s['total']}**\n"
                f"- ✅ OK: **{s['ok']}**\n"
                f"- ❌ NG: **{s['ng']}**"
            )

