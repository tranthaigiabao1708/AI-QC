"""
preprocessing/image_processor.py
─────────────────────────────────
Pipeline 6 bước xử lý ảnh sản phẩm crimping:
  Bước 1: Tách nền (background removal)
  Bước 2: Phát hiện contour + border ngoài (minAreaRect)
  Bước 3: Xoay nắn thẳng + xác định hướng đầu cos
  Bước 4: Cắt chuẩn hóa từ đỉnh sản phẩm
  Bước 5: Phát hiện vùng đồng lộ (LAB + HSV)
  Bước 6: Cắt ROI cuối cùng

Lưu ý: File này chứa thuật toán xử lý ảnh bằng OpenCV truyền thống đã được tối ưu hóa.
Bạn có thể giữ nguyên file này để làm nền tảng xử lý dữ liệu đầu vào.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np


@dataclass
class ProcessingResult:
    """Kết quả xử lý ảnh qua pipeline 6 bước."""
    roi: np.ndarray                          # ROI cuối cùng (resized)
    copper_w: int = 0                        # Chiều rộng vùng đồng (px)
    copper_h: int = 0                        # Chiều cao vùng đồng (px)
    copper_ratio: float = 0.0               # Tỷ lệ diện tích đồng / sản phẩm
    copper_mask: Optional[np.ndarray] = None # Mask vùng đồng
    outer_rect: Optional[Any] = None         # minAreaRect sản phẩm
    long_angle: float = 0.0                  # Góc xoay
    is_flipped: bool = False                 # Có đảo chiều không

    # Ảnh trực quan hóa từng bước
    vis_steps: Dict[str, np.ndarray] = field(default_factory=dict)


class ImageProcessor:
    """
    Pipeline xử lý ảnh sản phẩm crimping.
    Sử dụng:
        processor = ImageProcessor()
        result = processor.process(img_bgr)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        fixed_crop_length: int = 300,
        bg_diff_threshold: int = 30,
        morph_kernel_size: int = 9,
        min_contour_area: int = 5000,
        min_aspect_ratio: float = 2.5,
        lab_a_threshold: int = 138,
        hsv_s_threshold: int = 150,
        hsv_h_max: int = 25,
        search_region_ratio: float = 0.55,
        save_vis_steps: bool = True,
    ):
        self.target_size = target_size
        self.fixed_crop_length = fixed_crop_length
        self.bg_diff_threshold = bg_diff_threshold
        self.morph_kernel_size = morph_kernel_size
        self.min_contour_area = min_contour_area
        self.min_aspect_ratio = min_aspect_ratio
        self.lab_a_threshold = lab_a_threshold
        self.hsv_s_threshold = hsv_s_threshold
        self.hsv_h_max = hsv_h_max
        self.search_region_ratio = search_region_ratio
        self.save_vis_steps = save_vis_steps

    def process(self, img_in: np.ndarray) -> Optional[ProcessingResult]:
        """
        Chạy pipeline 6 bước trên ảnh đầu vào.
        """
        if img_in is None or img_in.size == 0:
            return None

        h, w = img_in.shape[:2]
        vis = {}

        # ═══ BƯỚC 1: TÁCH NỀN ═══
        fg_mask, product_contour = self._step1_remove_background(img_in, h, w)
        if fg_mask is None:
            return None

        img_no_bg = cv2.bitwise_and(img_in, img_in, mask=fg_mask)
        if self.save_vis_steps:
            vis["01_bg_removed"] = img_no_bg.copy()

        # ═══ BƯỚC 2: BORDER NGOÀI ═══
        outer_rect, cx, cy, length, thickness, long_angle = \
            self._step2_find_border(fg_mask)
        if outer_rect is None:
            return None

        if self.save_vis_steps:
            vis_border = img_in.copy()
            contours_mask, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours_mask:
                cv2.drawContours(vis_border,
                    [max(contours_mask, key=cv2.contourArea)], -1, (0, 255, 0), 2)
            cv2.circle(vis_border, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            vis["02_contour"] = vis_border

        # ═══ BƯỚC 3: XOAY THẲNG ═══
        rotated, rotated_mask, long_angle, is_flipped = \
            self._step3_rotate(img_no_bg, fg_mask, cx, cy, length, long_angle, h, w)
        if rotated is None:
            return None

        # Cắt sát sản phẩm sau xoay
        cropped, cropped_mask = self._crop_tight(rotated, rotated_mask, thickness, h, w)
        if cropped is None:
            return None

        if self.save_vis_steps:
            vis["03_rotated"] = cropped.copy()

        # ═══ BƯỚC 4: CẮT CHUẨN HÓA TỪ ĐỈNH ═══
        standardized, std_mask = self._step4_standardize(
            cropped, cropped_mask
        )
        if standardized is None:
            return None

        if self.save_vis_steps:
            vis["04_standardized"] = standardized.copy()

        h_std, w_std = standardized.shape[:2]

        # ═══ BƯỚC 5: PHÁT HIỆN ĐỒNG LỘ ═══
        copper_w, copper_h, copper_ratio, copper_mask = \
            self._step5_detect_copper(standardized, std_mask, h_std, w_std)

        if self.save_vis_steps:
            vis_copper = standardized.copy()
            overlay = np.zeros_like(standardized)
            overlay[:, :, 2] = copper_mask
            vis_copper = cv2.addWeighted(vis_copper, 0.7, overlay, 0.3, 0)
            search_x = int(w_std * self.search_region_ratio)
            cv2.line(vis_copper, (search_x, 0), (search_x, h_std),
                     (255, 255, 0), 1)
            if copper_w > 0:
                cx_start = self._last_copper_x
                cy_start = self._last_copper_y
                cv2.rectangle(vis_copper, (cx_start, cy_start),
                    (cx_start + copper_w, cy_start + copper_h), (0, 255, 0), 2)
            vis["05_copper_detect"] = vis_copper

        # ═══ BƯỚC 6: CẮT ROI ═══
        roi_final = self._step6_crop_roi(
            standardized, std_mask, copper_w, copper_h, h_std, w_std
        )
        if roi_final is None:
            return None

        if self.save_vis_steps:
            vis["06_roi_final"] = roi_final.copy()

        return ProcessingResult(
            roi=roi_final,
            copper_w=copper_w,
            copper_h=copper_h,
            copper_ratio=copper_ratio,
            copper_mask=copper_mask,
            outer_rect=outer_rect,
            long_angle=long_angle,
            is_flipped=is_flipped,
            vis_steps=vis,
        )

    # ──────────────────────────────────────────────────────────
    # Bước 1: Tách nền
    # ──────────────────────────────────────────────────────────
    def _step1_remove_background(self, img, h, w):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        median_val = np.median(blurred)
        lo = int(max(0, 0.5 * median_val))
        hi = int(min(255, 1.5 * median_val))
        edges = cv2.Canny(blurred, lo, hi)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for c in contours:
            if cv2.contourArea(c) < self.min_contour_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            cy_center = by + bh / 2.0
            if not (0.15 * h < cy_center < 0.85 * h):
                continue
            rect = cv2.minAreaRect(c)
            _, (rw, rh), _ = rect
            ar = max(rw, rh) / max(min(rw, rh), 1)
            if ar > self.min_aspect_ratio:
                candidates.append(c)

        if not candidates:
            return None, None

        product = max(candidates, key=cv2.contourArea)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [product], -1, 255, thickness=cv2.FILLED)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        return mask, product

    # ──────────────────────────────────────────────────────────
    # Bước 2: Border ngoài
    # ──────────────────────────────────────────────────────────
    def _step2_find_border(self, fg_mask):
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, 0, 0, 0, 0, 0

        product = max(contours, key=cv2.contourArea)
        outer_rect = cv2.minAreaRect(product)
        (cx, cy), (rect_w, rect_h), angle = outer_rect

        if rect_w < rect_h:
            length, thickness, long_angle = rect_h, rect_w, angle + 90
        else:
            length, thickness, long_angle = rect_w, rect_h, angle

        return outer_rect, cx, cy, length, thickness, long_angle

    # ──────────────────────────────────────────────────────────
    # Bước 3: Xoay thẳng
    # ──────────────────────────────────────────────────────────
    def _step3_rotate(self, img_no_bg, fg_mask, cx, cy, length, long_angle, h, w):
        rad = np.deg2rad(long_angle)
        dx, dy = np.cos(rad), np.sin(rad)
        nx, ny = -dy, dx

        pt1 = (int(cx + 0.33 * length * dx), int(cy + 0.33 * length * dy))
        pt2 = (int(cx - 0.33 * length * dx), int(cy - 0.33 * length * dy))

        thick1 = self._measure_thickness(fg_mask, *pt1, nx, ny)
        thick2 = self._measure_thickness(fg_mask, *pt2, nx, ny)

        is_flipped = False
        if thick1 < thick2:
            dx, dy = -dx, -dy
            long_angle += 180
            is_flipped = True

        M = cv2.getRotationMatrix2D((cx, cy), -long_angle, 1.0)
        rotated = cv2.warpAffine(img_no_bg, M, (w, h), flags=cv2.INTER_CUBIC)
        rot_mask = cv2.warpAffine(fg_mask, M, (w, h), flags=cv2.INTER_NEAREST)

        return rotated, rot_mask, long_angle, is_flipped

    def _crop_tight(self, rotated, rot_mask, thickness, h, w):
        rows = np.where(np.any(rot_mask > 0, axis=1))[0]
        cols = np.where(np.any(rot_mask > 0, axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return None, None

        margin = 5
        y1 = max(0, rows[0] - margin)
        y2 = min(h, rows[-1] + margin)
        x1 = max(0, cols[0] - margin)
        x2 = min(w, cols[-1] + margin)

        cropped = rotated[y1:y2, x1:x2]
        cropped_mask = rot_mask[y1:y2, x1:x2]

        if cropped.size == 0:
            return None, None
        return cropped, cropped_mask

    # ──────────────────────────────────────────────────────────
    # Bước 4: Cắt chuẩn hóa từ đỉnh
    # ──────────────────────────────────────────────────────────
    def _step4_standardize(self, cropped, cropped_mask):
        h_c, w_c = cropped.shape[:2]
        col_has_fg = np.any(cropped_mask > 0, axis=0)
        fg_cols = np.where(col_has_fg)[0]
        if len(fg_cols) == 0:
            return None, None

        x_tip = fg_cols[-1]
        x_start = max(0, x_tip - self.fixed_crop_length)
        x_end = min(w_c, x_tip + 5)

        std = cropped[:, x_start:x_end]
        std_mask = cropped_mask[:, x_start:x_end]

        if std.size == 0:
            return None, None
        return std, std_mask

    # ──────────────────────────────────────────────────────────
    # Bước 5: Phát hiện đồng lộ
    # ──────────────────────────────────────────────────────────
    def _step5_detect_copper(self, standardized, std_mask, h_std, w_std):
        self._last_copper_x = 0
        self._last_copper_y = 0

        lab = cv2.cvtColor(standardized, cv2.COLOR_BGR2LAB)
        _, a_ch, _ = cv2.split(lab)
        copper_mask = (a_ch > self.lab_a_threshold).astype(np.uint8) * 255

        hsv = cv2.cvtColor(standardized, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, _ = cv2.split(hsv)
        hsv_copper = ((s_ch > self.hsv_s_threshold) &
                      (h_ch < self.hsv_h_max)).astype(np.uint8) * 255
        copper_mask = copper_mask | hsv_copper
        copper_mask = cv2.bitwise_and(copper_mask, std_mask)

        # Giới hạn vùng tìm kiếm (phần đầu sản phẩm)
        search_x = int(w_std * self.search_region_ratio)
        copper_mask[:, search_x:] = 0

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        copper_mask = cv2.morphologyEx(copper_mask, cv2.MORPH_CLOSE, k, iterations=2)
        copper_mask = cv2.morphologyEx(copper_mask, cv2.MORPH_OPEN, k)

        contours, _ = cv2.findContours(
            copper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        c_x, c_y, c_w, c_h = 0, 0, 0, 0
        if contours:
            valid = [c for c in contours if cv2.contourArea(c) > 30]
            if valid:
                all_pts = np.vstack(valid)
                c_x, c_y, c_w, c_h = cv2.boundingRect(all_pts)

        self._last_copper_x = c_x
        self._last_copper_y = c_y

        copper_pixels = np.sum(copper_mask > 0)
        total_pixels = max(np.sum(std_mask > 0), 1)
        copper_ratio = copper_pixels / total_pixels

        return c_w, c_h, copper_ratio, copper_mask

    # ──────────────────────────────────────────────────────────
    # Bước 6: Cắt ROI cuối cùng
    # ──────────────────────────────────────────────────────────
    def _step6_crop_roi(self, standardized, std_mask, copper_w, copper_h,
                        h_std, w_std):
        if copper_w > 0 and copper_h > 0:
            pad = 10
            y1 = max(0, self._last_copper_y - pad)
            y2 = min(h_std, self._last_copper_y + copper_h + pad)
            x1 = max(0, self._last_copper_x - pad)
            x2 = min(w_std, self._last_copper_x + copper_w + pad)
            roi_raw = standardized[y1:y2, x1:x2]
        else:
            # Fallback: inner border cố định
            ix_l = int(w_std * 0.15)
            ix_r = int(w_std * 0.50)
            col_slice = std_mask[:, ix_l:ix_r]
            rows = np.where(np.any(col_slice > 0, axis=1))[0]
            if len(rows) > 0:
                iy_t = max(0, rows[0] - 5)
                iy_b = min(h_std, rows[-1] + 5)
            else:
                iy_t, iy_b = int(h_std * 0.2), int(h_std * 0.8)
            roi_raw = standardized[iy_t:iy_b, ix_l:ix_r]

        if roi_raw.size == 0:
            return None
        return cv2.resize(roi_raw, self.target_size, interpolation=cv2.INTER_CUBIC)

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _measure_thickness(mask, px, py, nx, ny, max_search=200):
        mh, mw = mask.shape[:2]
        t1 = 0
        while t1 < max_search:
            x, y = int(px + t1 * nx), int(py + t1 * ny)
            if x < 0 or x >= mw or y < 0 or y >= mh or mask[y, x] == 0:
                break
            t1 += 1
        t2 = 0
        while t2 < max_search:
            x, y = int(px - t2 * nx), int(py - t2 * ny)
            if x < 0 or x >= mw or y < 0 or y >= mh or mask[y, x] == 0:
                break
            t2 += 1
        return t1 + t2
