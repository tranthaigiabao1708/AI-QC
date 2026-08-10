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

        # Kiểm tra xem mô hình đã được huấn luyện chưa (cần có file weights thực tế)
        self.use_onnx = False
        has_weights = (
            (Path(model_dir) / "model.safetensors").exists() or
            (Path(model_dir) / "pytorch_model.bin").exists()
        )
        if has_weights and (Path(model_dir) / "config.json").exists():
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
            self.has_trained_model = True
        else:
            logger.warning(f"Không tìm thấy mô hình đã huấn luyện tại: {model_dir}!")
            logger.warning(f"Tải mô hình gốc (chưa tinh chỉnh) từ Hugging Face Hub: {config.MODEL_NAME}...")
            self.image_processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)
            self.model = AutoModelForImageClassification.from_pretrained(config.MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
            self.model.to(self.device)
            self.model.eval()
            self.has_trained_model = False

        # Khởi tạo OpenCV Image Processor (phiên bản robust V2 + YOLO-seg)
        self.opencv_processor = ImageProcessor(
            target_size=config.IMAGE_SIZE,
            crop_ratio=getattr(config, 'CROP_RATIO', 0.45),
            fixed_crop_length=config.FIXED_CROP_LENGTH,
            min_contour_ratio=getattr(config, 'MIN_CONTOUR_RATIO', 0.02),
            copper_kmeans_clusters=getattr(config, 'COPPER_KMEANS_CLUSTERS', 4),
            enable_presharpening=getattr(config, 'ENABLE_PRESHARPENING', False),
            use_yolo_seg=getattr(config, 'USE_YOLO_SEG', False),
            yolo_model_path=getattr(config, 'YOLO_SEG_MODEL_PATH', None),
            yolo_confidence=getattr(config, 'YOLO_SEG_CONFIDENCE', 0.5),
        )
        mode_str = "CV+AI" if self.has_trained_model else "CV-only (model chưa train)"
        logger.info(f"Khởi tạo QualityInspector V3 thành công! Mode: {mode_str}")

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

        # ═══ QUYẾT ĐỊNH CUỐI CÙNG: CV-FIRST + AI-BOOST ═══
        # Ưu tiên kết quả Computer Vision (copper_ratio, pipeline_confidence)
        # Model AI chỉ bổ trợ khi đã được train đầy đủ
        pred_label = config.CLASS_NAMES[pred_idx]

        # Lấy ngưỡng động
        conf_thresh = getattr(self, "confidence_threshold", config.CONFIDENCE_THRESHOLD)
        min_ratio = getattr(self, "min_copper_ratio", config.MIN_COPPER_RATIO)
        max_ratio = getattr(self, "max_copper_ratio", config.MAX_COPPER_RATIO)

        # ── Bước A: Kiểm tra pipeline có đáng tin không ──
        pipeline_conf = proc_result.pipeline_confidence
        copper_ratio = proc_result.copper_ratio

        if pipeline_conf < 0.3:
            final_decision = "NG (Pipeline không detect được sản phẩm)"
            color = (0, 165, 255)
            effective_confidence = pipeline_conf

        # ── Bước B: Quyết định bằng Computer Vision (copper_ratio) ──
        elif copper_ratio > max_ratio:
            # Đồng lộ quá nhiều → NG chắc chắn (CV)
            final_decision = f"NG (Đồng lộ quá cao: {copper_ratio:.1%})"
            color = (0, 0, 255)
            effective_confidence = 0.9  # CV rất tự tin

        elif copper_ratio < min_ratio and pipeline_conf > 0.5:
            # Không thấy đồng NHƯNG pipeline detect tốt → OK (CV)
            # Sản phẩm OK = đồng không lộ ra ngoài
            if self.has_trained_model and pred_label == "NG" and confidence > 0.85:
                # Model đã train VÀ rất tự tin NG → tin model
                final_decision = f"NG (AI phát hiện lỗi: {confidence:.0%})"
                color = (0, 0, 255)
                effective_confidence = confidence
            else:
                final_decision = "OK"
                color = (0, 255, 0)
                # Confidence = trung bình pipeline + (model nếu đã train)
                if self.has_trained_model:
                    effective_confidence = 0.5 * pipeline_conf + 0.5 * confidence
                else:
                    effective_confidence = pipeline_conf * 0.9

        elif min_ratio <= copper_ratio <= max_ratio:
            # Đồng lộ trong khoảng cho phép → có thể OK hoặc NG tùy mức
            if self.has_trained_model:
                # Model đã train → dùng model quyết định
                final_decision = pred_label
                effective_confidence = confidence
                color = (0, 255, 0) if pred_label == "OK" else (0, 0, 255)
            else:
                # Model chưa train → dùng quy tắc: copper gần min → OK, gần max → NG
                mid_point = (min_ratio + max_ratio) / 2
                if copper_ratio < mid_point:
                    final_decision = "OK"
                    color = (0, 255, 0)
                    effective_confidence = pipeline_conf * 0.8
                else:
                    final_decision = f"NG (Đồng lộ: {copper_ratio:.1%})"
                    color = (0, 0, 255)
                    effective_confidence = pipeline_conf * 0.7

        else:
            # Fallback
            if self.has_trained_model:
                final_decision = pred_label
                effective_confidence = confidence
            else:
                final_decision = "OK"
                effective_confidence = pipeline_conf * 0.7
            color = (0, 255, 0) if final_decision == "OK" else (0, 0, 255)

        # 4. Vẽ trực quan hóa 3 thành phần trên ảnh gốc theo đúng yêu cầu
        vis_img = img_bgr.copy()

        # 1. Border xanh lá cây ngoài cùng: Bo sát 100% theo dạng thực tế của vật thể sau khi remove background
        if proc_result.smooth_contour is not None:
            cv2.polylines(vis_img, [proc_result.smooth_contour], True, (0, 255, 0), 3)
        elif proc_result.outer_rect is not None:
            box = cv2.boxPoints(proc_result.outer_rect)
            box = np.intp(box)
            cv2.polylines(vis_img, [box], True, (0, 255, 0), 3)

        # 2. Border đen trung tâm: Bo sát 100% đường viền phần kim loại cos (crimp barrel + ring head)
        if proc_result.terminal_contour is not None:
            img_h, img_w = vis_img.shape[:2]
            border_canvas = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.polylines(border_canvas, [proc_result.terminal_contour], True, 255, 3)
            if hasattr(proc_result, 'metal_end_y') and proc_result.metal_end_y > 0:
                border_canvas[proc_result.metal_end_y - 2:, :] = 0
            vis_img[border_canvas > 0] = (0, 0, 0)
        elif proc_result.terminal_pts is not None:
            cv2.drawContours(vis_img, [proc_result.terminal_pts], 0, (0, 0, 0), 3)
        elif proc_result.terminal_bbox is not None:
            tx1, ty1, tx2, ty2 = proc_result.terminal_bbox
            cv2.rectangle(vis_img, (tx1, ty1), (tx2, ty2), (0, 0, 0), 3)

        # 3. Khoanh vùng đồng lộ (Xanh lá / Đỏ): Bo sát 100% đường viền dải đồng (contours bám khoang màu) + tỉ lệ % diện tích
        spec_ratios = []
        if proc_result.copper_details:
            for item in proc_result.copper_details:
                (c_x, c_y, c_w, c_h) = item['box']
                r_pct = item['ratio_pct']
                p_name = item['pos_name']
                
                # Vẽ border bám sát khoang màu đồng
                if 'contours' in item and item['contours']:
                    cv2.drawContours(vis_img, item['contours'], -1, color, 2)
                else:
                    cv2.rectangle(vis_img, (c_x, c_y), (c_x + c_w, c_y + c_h), color, 2)

                # Vẽ nhãn % diện tích đồng ngay bên cạnh khoang màu đồng
                lbl = f"{p_name}: {r_pct:.2f}%"
                cv2.putText(vis_img, lbl, (c_x + c_w + 10, c_y + c_h // 2 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                spec_ratios.append(f"{p_name}={r_pct:.2f}%")
        elif proc_result.copper_boxes:
            for (c_x, c_y, c_w, c_h) in proc_result.copper_boxes:
                cv2.rectangle(vis_img, (c_x, c_y), (c_x + c_w, c_y + c_h), color, 2)
        elif proc_result.copper_pts is not None:
            cv2.drawContours(vis_img, [proc_result.copper_pts], 0, color, 2)
        elif proc_result.copper_bbox_orig is not None:
            cx1, cy1, cx2, cy2 = proc_result.copper_bbox_orig
            cv2.rectangle(vis_img, (cx1, cy1), (cx2, cy2), color, 2)

        text = f"QC DECISION: {final_decision} ({effective_confidence:.1%})"
        cv2.putText(vis_img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        ratio_summary = " | ".join(spec_ratios) if spec_ratios else f"Copper: {copper_ratio:.1%}"
        mode_text = "CV+AI" if self.has_trained_model else "CV-only"
        spec_text = f"Ratios: {ratio_summary} | Metal: {int(proc_result.metal_area_px)}px2 | Mode: {mode_text}"
        cv2.putText(vis_img, spec_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return {
            "success": True,
            "decision": final_decision,
            "raw_decision": pred_label,
            "confidence": effective_confidence,
            "copper_ratio": copper_ratio,
            "copper_w": proc_result.copper_w,
            "copper_h": proc_result.copper_h,
            "pipeline_confidence": pipeline_conf,
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
