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
    Hệ thống kiểm tra chất lượng tích hợp (V2 — Robust + Live-ready):
    1. Chạy pipeline tiền xử lý OpenCV (adaptive) -> Cắt ROI sản phẩm
    2. Chạy mô hình Deep Learning của Hugging Face -> Phân loại lỗi OK/NG
    3. Hỗ trợ ONNX runtime tùy chọn để tăng tốc inference
    """
    def __init__(self, model_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Đường dẫn tải model
        if model_dir is None:
            model_dir = config.MODEL_DIR

        # Tắt progress bar của Hugging Face
        from transformers.utils.logging import disable_progress_bar
        disable_progress_bar()

        # Kiểm tra xem mô hình đã được huấn luyện chưa
        self.use_onnx = False
        if (Path(model_dir) / "config.json").exists():
            model_dir_str = str(Path(model_dir).resolve())
            logger.info(f"Đang tải mô hình đã huấn luyện từ: {model_dir_str}...")
            self.image_processor = AutoImageProcessor.from_pretrained(model_dir_str)

            # Thử tải ONNX nếu được bật và file tồn tại
            if getattr(config, 'USE_ONNX', False) and config.ONNX_MODEL_PATH.exists():
                try:
                    import onnxruntime as ort
                    self.onnx_session = ort.InferenceSession(
                        str(config.ONNX_MODEL_PATH),
                        providers=['CPUExecutionProvider']
                    )
                    self.use_onnx = True
                    logger.info(f"Đang sử dụng ONNX Runtime để tăng tốc inference!")
                except ImportError:
                    logger.warning("onnxruntime chưa cài. Dùng PyTorch thay thế.")
                except Exception as e:
                    logger.warning(f"Lỗi tải ONNX model: {e}. Dùng PyTorch thay thế.")

            if not self.use_onnx:
                self.model = AutoModelForImageClassification.from_pretrained(model_dir_str)
                self.model.to(self.device)
                self.model.eval()
        else:
            logger.warning(f"Không tìm thấy mô hình đã huấn luyện tại: {model_dir}!")
            logger.warning(f"Tải mô hình gốc (chưa tinh chỉnh) từ Hugging Face Hub: {config.MODEL_NAME}...")
            self.image_processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)
            self.model = AutoModelForImageClassification.from_pretrained(config.MODEL_NAME, num_labels=2)
            self.model.to(self.device)
            self.model.eval()

        # Khởi tạo OpenCV Image Processor (phiên bản robust V2)
        self.opencv_processor = ImageProcessor(
            target_size=config.IMAGE_SIZE,
            crop_ratio=getattr(config, 'CROP_RATIO', 0.45),
            fixed_crop_length=config.FIXED_CROP_LENGTH,
            min_contour_ratio=getattr(config, 'MIN_CONTOUR_RATIO', 0.02),
            copper_kmeans_clusters=getattr(config, 'COPPER_KMEANS_CLUSTERS', 4),
        )
        logger.info("Khởi tạo hệ thống QualityInspector V2 thành công!")

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

        # 2. Chuẩn bị ảnh ROI đưa vào mô hình
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)

        # 3. Suy luận
        inputs = self.image_processor(images=pil_img, return_tensors="pt" if not self.use_onnx else "np")

        if self.use_onnx:
            # ONNX Runtime inference (nhanh hơn ~2-3x)
            pixel_values = inputs["pixel_values"]
            ort_inputs = {"pixel_values": pixel_values}
            ort_outputs = self.onnx_session.run(None, ort_inputs)
            logits = ort_outputs[0]
            probs = self._softmax(logits[0])
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
        else:
            # PyTorch inference
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.inference_mode():
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).squeeze(0)

            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

        # Đưa ra quyết định cuối cùng
        pred_label = config.CLASS_NAMES[pred_idx]

        # Lấy ngưỡng động từ self nếu có
        conf_thresh = getattr(self, "confidence_threshold", config.CONFIDENCE_THRESHOLD)
        min_ratio = getattr(self, "min_copper_ratio", config.MIN_COPPER_RATIO)
        max_ratio = getattr(self, "max_copper_ratio", config.MAX_COPPER_RATIO)

        # Phân loại thực tế (AI + quy tắc vật lý + pipeline confidence)
        if proc_result.pipeline_confidence < 0.4:
            final_decision = "NG (Pipeline confidence thấp — cần kiểm tra thủ công)"
            color = (0, 165, 255)
        elif pred_label == "OK" and confidence < conf_thresh:
            final_decision = "NG (Độ tin cậy OK thấp)"
            color = (0, 165, 255)
        elif pred_label == "OK" and (proc_result.copper_ratio < min_ratio or proc_result.copper_ratio > max_ratio):
            final_decision = f"NG (Lỗi tỷ lệ đồng lộ: {proc_result.copper_ratio:.2%})"
            color = (0, 0, 255)
        else:
            final_decision = pred_label
            color = (0, 255, 0) if final_decision == "OK" else (0, 0, 255)

        # 4. Vẽ trực quan hóa trên ảnh gốc
        vis_img = img_bgr.copy()

        if proc_result.outer_rect is not None:
            box = cv2.boxPoints(proc_result.outer_rect)
            box = np.intp(box)
            cv2.drawContours(vis_img, [box], 0, color, 3)

        text = f"QC DECISION: {final_decision} ({confidence:.1%})"
        cv2.putText(vis_img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        spec_text = f"Copper Area Ratio: {proc_result.copper_ratio:.1%} | Pipeline Conf: {proc_result.pipeline_confidence:.0%}"
        cv2.putText(vis_img, spec_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return {
            "success": True,
            "decision": final_decision,
            "raw_decision": pred_label,
            "confidence": confidence,
            "copper_ratio": proc_result.copper_ratio,
            "copper_w": proc_result.copper_w,
            "copper_h": proc_result.copper_h,
            "pipeline_confidence": proc_result.pipeline_confidence,
            "centroid": proc_result.centroid,
            "product_length": proc_result.product_length,
            "vis_img": vis_img,
            "roi": roi,
            "vis_steps": proc_result.vis_steps
        }

    def inspect_fast(self, img_bgr):
        """
        Version nhẹ: chỉ chạy Bước 1 (tách nền) để detect có sản phẩm không.
        Dùng cho live mode frame skipping — kiểm tra nhanh trước khi chạy full pipeline.

        Trả về:
            dict: {"has_product": bool, "centroid": tuple hoặc None}
        """
        if img_bgr is None or img_bgr.size == 0:
            return {"has_product": False, "centroid": None}

        h, w = img_bgr.shape[:2]
        fg_mask, contour, _ = self.opencv_processor._step1_remove_background(img_bgr, h, w)

        if fg_mask is None or contour is None:
            return {"has_product": False, "centroid": None}

        M = cv2.moments(contour)
        if M["m00"] > 0:
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            centroid = (w // 2, h // 2)

        return {"has_product": True, "centroid": centroid}

    @staticmethod
    def _softmax(x):
        """Softmax cho ONNX output (numpy)."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


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
            logger.info(f"  Pipeline confidence: {result['pipeline_confidence']:.0%}")
            logger.info(f"  Copper ratio: {result['copper_ratio']:.2%}")
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
                logger.info(f"Ảnh: {img_path.name} | Kết quả: {result['decision']} ({result['confidence']:.1%}) | Pipeline: {result['pipeline_confidence']:.0%}")
                out_path = pred_output_dir / f"pred_{img_path.name}"
                cv2.imwrite(str(out_path), result["vis_img"])
            else:
                logger.error(f"Ảnh: {img_path.name} | Lỗi: {result['error']}")

        logger.info(f"Hoàn tất! Các ảnh kết quả suy luận đã được lưu tại: {pred_output_dir}")

if __name__ == "__main__":
    main()
