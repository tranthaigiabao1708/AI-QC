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
    Hệ thống kiểm tra chất lượng kết hợp OpenCV và Deep Learning Hugging Face.
    """
    def __init__(self, model_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_dir is None:
            model_dir = config.MODEL_DIR
            
        # Kiểm tra và tải mô hình
        if (Path(model_dir) / "config.json").exists():
            logger.info(f"Đang tải mô hình đã huấn luyện từ: {model_dir}...")
            self.image_processor = AutoImageProcessor.from_pretrained(model_dir)
            self.model = AutoModelForImageClassification.from_pretrained(model_dir)
        else:
            logger.warning("Không tìm thấy mô hình huấn luyện cục bộ. Tải mô hình gốc...")
            self.image_processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)
            self.model = AutoModelForImageClassification.from_pretrained(config.MODEL_NAME, num_labels=2)
            
        self.model.to(self.device)
        self.model.eval()
        
        self.opencv_processor = ImageProcessor(target_size=config.IMAGE_SIZE)

    def inspect(self, img_bgr):
        """
        Thực hiện kiểm tra chất lượng một sản phẩm.
        """
        if img_bgr is None or img_bgr.size == 0:
            return {"success": False, "error": "Ảnh đầu vào rỗng"}
            
        # =====================================================================
        # TODO: HÃY TỰ VIẾT LOGIC LIÊN KẾT TIỀN XỬ LÝ OPENCV VÀ SUY LUẬN AI
        # =====================================================================
        
        # 1. Chạy qua pipeline OpenCV để lấy ROI và các bước ảnh trực quan
        # Gợi ý: Gọi proc_result = self.opencv_processor.process(img_bgr)
        # Kiểm tra nếu proc_result rỗng hoặc proc_result.roi rỗng thì báo lỗi.
        proc_result = None # TODO: Thực hiện tiền xử lý
        
        if proc_result is None or proc_result.roi is None:
            return {"success": False, "error": "Không thể tách sản phẩm"}
            
        roi = proc_result.roi
        
        # 2. Chuyển đổi màu BGR sang RGB và chuyển đổi sang ảnh PIL RGB
        # Gợi ý: Dùng cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) và Image.fromarray(...)
        pil_img = None # TODO: Thực hiện đổi định dạng màu
        
        # 3. Chuẩn hóa bằng Hugging Face ImageProcessor và đưa lên thiết bị tính toán
        # Gợi ý: 
        # inputs = self.image_processor(images=pil_img, return_tensors="pt")
        # pixel_values = inputs["pixel_values"].to(self.device)
        pixel_values = None # TODO: Chuẩn hóa ảnh
        
        # 4. Chạy mô hình (Inference) và tính xác suất lớp bằng Softmax
        # Gợi ý:
        # with torch.no_grad():
        #     outputs = self.model(pixel_values=pixel_values)
        #     logits = outputs.logits
        #     probs = torch.softmax(logits, dim=1).squeeze(0)
        # Lấy nhãn dự đoán pred_idx = torch.argmax(probs).item()
        # Lấy độ tin cậy confidence = probs[pred_idx].item()
        pred_idx = 0     # TODO: Thực hiện suy luận và lấy index nhãn lớn nhất
        confidence = 0.0 # TODO: Lấy độ tin cậy dự đoán của nhãn đó
        
        pred_label = config.CLASS_NAMES[pred_idx]
        
        # Quyết định kết quả cuối cùng áp dụng ngưỡng Confidence Threshold
        if pred_label == "OK" and confidence < config.CONFIDENCE_THRESHOLD:
            final_decision = "NG (OK thấp tin cậy)"
            color = (0, 165, 255) # Cảnh báo màu cam
        else:
            final_decision = pred_label
            color = (0, 255, 0) if pred_label == "OK" else (0, 0, 255)
            
        # 5. Vẽ trực quan hóa lên ảnh gốc
        vis_img = img_bgr.copy()
        
        # Vẽ bounding box xoay quanh sản phẩm
        if proc_result.outer_rect is not None:
            box = cv2.boxPoints(proc_result.outer_rect)
            box = np.intp(box)
            cv2.drawContours(vis_img, [box], 0, color, 3)
            
        text = f"QC: {final_decision} ({confidence:.1%})"
        cv2.putText(vis_img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        
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
    args = parser.parse_args()
    
    inspector = QualityInspector()
    
    # Tạo thư mục lưu
    pred_output_dir = config.OUTPUT_DIR / "predictions"
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            logger.error(f"File không tồn tại: {img_path}")
            sys.exit(1)
            
        img = cv2.imread(str(img_path))
        result = inspector.inspect(img)
        
        if result["success"]:
            logger.info(f"Dự đoán [{img_path.name}]: {result['decision']} ({result['confidence']:.1%})")
            out_path = pred_output_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(out_path), result["vis_img"])
        else:
            logger.error(f"Lỗi: {result['error']}")
    else:
        # Chạy mặc định trên tất cả ảnh gốc
        logger.info(f"Đang quét tất cả ảnh trong: {config.RAW_DATA_DIR}")
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        raw_files = [f for f in config.RAW_DATA_DIR.iterdir() if f.suffix.lower() in exts]
        
        for img_path in raw_files:
            img = cv2.imread(str(img_path))
            result = inspector.inspect(img)
            
            if result["success"]:
                logger.info(f"Ảnh: {img_path.name} | Kết quả: {result['decision']} ({result['confidence']:.1%})")
                out_path = pred_output_dir / f"pred_{img_path.name}"
                cv2.imwrite(str(out_path), result["vis_img"])
            else:
                logger.error(f"Ảnh: {img_path.name} | Lỗi: {result['error']}")

if __name__ == "__main__":
    main()
