"""
preprocessing/yolo_segmentor.py
───────────────────────────────
YOLOv8-seg wrapper để segment sản phẩm.
Thay thế _step1_remove_background trong pipeline.

Trả về output tương thích với pipeline hiện tại:
  - fg_mask: binary mask (uint8, 0/255)
  - product_contour: contour lớn nhất
  - confidence: detection confidence
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


class YOLOSegmentor:
    """
    Segment sản phẩm bằng YOLOv8-seg model đã train.
    
    Usage:
        segmentor = YOLOSegmentor("output/yolo_seg_best.pt")
        mask, contour, confidence = segmentor.segment(img)
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        if not HAS_ULTRALYTICS:
            raise ImportError("ultralytics chưa cài. Chạy: pip install ultralytics")

        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model = None

        if self.model_path.exists():
            self.model = YOLO(str(self.model_path))
            logger.info(f"YOLOSegmentor loaded: {self.model_path.name}")
        else:
            logger.warning(f"YOLO model không tồn tại: {self.model_path}")

    @property
    def is_ready(self) -> bool:
        """Kiểm tra model đã load thành công chưa."""
        return self.model is not None

    def segment(self, img: np.ndarray):
        """
        Segment sản phẩm trong ảnh.

        Args:
            img: BGR image (numpy array)

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - fg_mask: binary mask (uint8, 0/255), None nếu không detect được
                - product_contour: contour lớn nhất, None nếu không detect
                - confidence: float [0, 1]
        """
        if not self.is_ready:
            return None, None, 0.0

        h, w = img.shape[:2]

        # Run inference
        results = self.model(img, verbose=False, conf=self.confidence_threshold)

        if not results or len(results) == 0:
            return None, None, 0.0

        result = results[0]

        # Kiểm tra có masks không
        if result.masks is None or len(result.masks) == 0:
            return None, None, 0.0

        # Lấy detection có confidence cao nhất
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None, None, 0.0

        # Tìm detection tốt nhất (confidence cao nhất)
        confidences = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confidences))
        best_conf = float(confidences[best_idx])

        # Lấy mask tương ứng
        mask_data = result.masks.data[best_idx].cpu().numpy()

        # Resize mask về kích thước ảnh gốc (YOLO mask có thể nhỏ hơn)
        if mask_data.shape != (h, w):
            mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_resized = mask_data

        # Chuyển sang binary mask uint8
        fg_mask = (mask_resized > 0.5).astype(np.uint8) * 255

        # Morphological cleanup nhẹ
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Tìm contour lớn nhất
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return fg_mask, None, best_conf

        product_contour = max(contours, key=cv2.contourArea)

        # Tạo mask sạch từ contour lớn nhất (bỏ noise nhỏ)
        clean_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(clean_mask, [product_contour], -1, 255, -1)

        return clean_mask, product_contour, best_conf

    def segment_smooth_contour(self, img: np.ndarray):
        """
        Segment + tạo smooth contour cho visualization (viền xanh).
        Thay thế hoàn toàn smooth_contour generation từ pipeline CV.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
                - fg_mask, product_contour, smooth_contour, confidence
        """
        fg_mask, product_contour, conf = self.segment(img)

        if fg_mask is None:
            return None, None, None, 0.0

        # Smooth contour: dilate nhẹ + blur + threshold
        h, w = img.shape[:2]
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(fg_mask, k_dilate, iterations=1)
        blurred = cv2.GaussianBlur(dilated, (9, 9), 0)
        _, smooth_mask = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        smooth_contour = None
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            epsilon = 0.001 * cv2.arcLength(largest, True)
            smooth_contour = cv2.approxPolyDP(largest, epsilon, True)

        return fg_mask, product_contour, smooth_contour, conf
