import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from loguru import logger

# Thêm đường dẫn gốc vào python path để import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from preprocessing import ImageProcessor

class QualityInspector:
    """
    Hệ thống kiểm tra chất lượng tích hợp:
    1. Chạy pipeline tiền xử lý OpenCV -> Cắt ROI sản phẩm
    2. Chạy mô hình Deep Learning của Hugging Face -> Phân loại lỗi OK/NG
    """
    def __init__(self, model_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Đường dẫn tải model
        if model_dir is None:
            model_dir = config.MODEL_DIR
            
        # Tắt progress bar của Hugging Face để tránh lỗi flush() của tqdm trên Windows khi chạy Streamlit
        from transformers.utils.logging import disable_progress_bar
        disable_progress_bar()

        # Kiểm tra xem mô hình đã được huấn luyện chưa
        if (Path(model_dir) / "config.json").exists():
            model_dir_str = str(Path(model_dir).resolve())
            logger.info(f"Đang tải mô hình đã huấn luyện từ: {model_dir_str}...")
            self.image_processor = AutoImageProcessor.from_pretrained(model_dir_str)
            self.model = AutoModelForImageClassification.from_pretrained(model_dir_str)
        else:
            logger.warning(f"Không tìm thấy mô hình đã huấn luyện tại: {model_dir}!")
            logger.warning(f"Tải mô hình gốc (chưa tinh chỉnh) từ Hugging Face Hub: {config.MODEL_NAME}...")
            self.image_processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)
            self.model = AutoModelForImageClassification.from_pretrained(config.MODEL_NAME, num_labels=2)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Khởi tạo OpenCV Image Processor
        self.opencv_processor = ImageProcessor(target_size=config.IMAGE_SIZE)
        logger.info("Khởi tạo hệ thống QualityInspector thành công!")

    def inspect(self, img_bgr):
        """
        Kiểm tra chất lượng một ảnh sản phẩm (BGR).
        
        Trả về:
            dict: Chứa thông tin nhãn dự đoán, độ tin cậy, thông số đo đạc, và ảnh trực quan hóa.
        """
        if img_bgr is None or img_bgr.size == 0:
            return {"success": False, "error": "Ảnh đầu vào rỗng"}
            
        # 1. Chạy qua pipeline tiền xử lý OpenCV để lấy ROI và các bước trực quan hóa
        proc_result = self.opencv_processor.process(img_bgr)
        if proc_result is None or proc_result.roi is None:
            return {"success": False, "error": "Không thể tách và định vị sản phẩm bằng OpenCV"}
            
        roi = proc_result.roi
        
        # 2. Chuẩn bị ảnh ROI đưa vào mô hình Hugging Face
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        
        # 3. Suy luận bằng mô hình PyTorch + Hugging Face
        inputs = self.image_processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze(0)
            
        # Lấy nhãn và độ tin cậy
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        
        # Đưa ra quyết định cuối cùng dựa trên Ngưỡng tin cậy (Confidence Threshold)
        # Nếu mô hình dự đoán NG nhưng độ tin cậy quá thấp, hoặc ngược lại, ta có thể điều chỉnh
        pred_label = config.CLASS_NAMES[pred_idx]
        
        # Lấy ngưỡng động từ self nếu có để tương thích tốt với Streamlit Sidebar
        conf_thresh = getattr(self, "confidence_threshold", config.CONFIDENCE_THRESHOLD)
        min_ratio = getattr(self, "min_copper_ratio", config.MIN_COPPER_RATIO)
        max_ratio = getattr(self, "max_copper_ratio", config.MAX_COPPER_RATIO)
        
        # Phân loại thực tế (áp dụng kết hợp AI và quy tắc vật lý bổ trợ)
        if pred_label == "OK" and confidence < conf_thresh:
            # Nếu dự đoán là OK nhưng độ tin cậy thấp -> chuyển sang NG để kiểm tra thủ công cho an toàn
            final_decision = "NG (Độ tin cậy OK thấp)"
            color = (0, 165, 255) # Màu cam cảnh báo
        elif pred_label == "OK" and (proc_result.copper_ratio < min_ratio or proc_result.copper_ratio > max_ratio):
            # Nếu dự đoán là OK nhưng tỷ lệ đồng lộ ra không đạt tiêu chuẩn vật lý -> báo lỗi NG
            final_decision = f"NG (Lỗi tỷ lệ đồng lộ: {proc_result.copper_ratio:.2%})"
            color = (0, 0, 255) # Màu đỏ báo lỗi
        else:
            final_decision = pred_label
            color = (0, 255, 0) if final_decision == "OK" else (0, 0, 255) # Xanh cho OK, Đỏ cho NG
            
        # 4. Vẽ trực quan hóa trên ảnh gốc
        vis_img = img_bgr.copy()
        
        # Vẽ oriented bounding box bao quanh sản phẩm từ outer_rect
        if proc_result.outer_rect is not None:
            box = cv2.boxPoints(proc_result.outer_rect)
            box = np.intp(box)
            cv2.drawContours(vis_img, [box], 0, color, 3)
            
        # Ghi kết quả lên ảnh
        text = f"QC DECISION: {final_decision} ({confidence:.1%})"
        cv2.putText(vis_img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        
        # Thêm thông tin thông số đo đạc từ pipeline OpenCV làm thông tin bổ trợ (AI + Rule-Based)
        spec_text = f"Copper Area Ratio: {proc_result.copper_ratio:.1%}"
        cv2.putText(vis_img, spec_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return {
            "success": True,
            "decision": final_decision,
            "raw_decision": pred_label,
            "confidence": confidence,
            "copper_ratio": proc_result.copper_ratio,
            "copper_w": proc_result.copper_w,
            "copper_h": proc_result.copper_h,
            "vis_img": vis_img,
            "roi": roi,
            "vis_steps": proc_result.vis_steps
        }

def main():
    parser = argparse.ArgumentParser(description="Chạy suy luận kiểm tra chất lượng sản phẩm.")
    parser.add_argument("--image", type=str, help="Đường dẫn tới ảnh sản phẩm cần kiểm tra")
    parser.add_argument("--model-dir", type=str, default=None, help="Đường dẫn thư mục chứa model")
    args = parser.parse_args()
    
    inspector = QualityInspector(model_dir=args.model_dir)
    
    # Tạo thư mục lưu kết quả suy luận
    pred_output_dir = config.OUTPUT_DIR / "predictions"
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.image:
        # Chạy suy luận trên 1 ảnh chỉ định
        img_path = Path(args.image)
        if not img_path.exists():
            logger.error(f"File không tồn tại: {img_path}")
            sys.exit(1)
            
        img = cv2.imread(str(img_path))
        result = inspector.inspect(img)
        
        if result["success"]:
            logger.info(f"Kết quả phân loại [{img_path.name}]: {result['decision']} (Độ tin cậy: {result['confidence']:.1%})")
            out_path = pred_output_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(out_path), result["vis_img"])
            logger.info(f"Đã lưu ảnh trực quan hóa tại: {out_path}")
        else:
            logger.error(f"Suy luận lỗi: {result['error']}")
    else:
        # Chạy suy luận mặc định trên tất cả ảnh gốc trong raw_images
        logger.info(f"Đang quét tất cả ảnh trong thư mục ảnh gốc: {config.RAW_DATA_DIR}")
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        raw_files = [f for f in config.RAW_DATA_DIR.iterdir() if f.suffix.lower() in exts]
        
        if not raw_files:
            logger.warning(f"Không có ảnh nào trong thư mục {config.RAW_DATA_DIR}")
            sys.exit(0)
            
        for img_path in raw_files:
            img = cv2.imread(str(img_path))
            result = inspector.inspect(img)
            
            if result["success"]:
                logger.info(f"Ảnh: {img_path.name} | Kết quả: {result['decision']} ({result['confidence']:.1%})")
                out_path = pred_output_dir / f"pred_{img_path.name}"
                cv2.imwrite(str(out_path), result["vis_img"])
            else:
                logger.error(f"Ảnh: {img_path.name} | Lỗi: {result['error']}")
                
        logger.info(f"Hoàn tất! Các ảnh kết quả suy luận đã được lưu tại: {pred_output_dir}")

if __name__ == "__main__":
    main()
