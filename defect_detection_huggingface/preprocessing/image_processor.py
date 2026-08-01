"""
preprocessing/image_processor.py
─────────────────────────────────
Pipeline 6 bước xử lý ảnh sản phẩm crimping — PHIÊN BẢN BỀN VỮNG (Robust V2):
  Bước 1: Tách nền bền vững (HSV-based + Otsu, không ngưỡng cứng)
  Bước 2: Phát hiện contour + PCA xác định trục chính
  Bước 3: Xoay thẳng + xác định hướng đầu cos bằng phân tích sáng/màu
  Bước 4: Cắt tự thích ứng (adaptive crop theo tỷ lệ chiều dài sản phẩm)
  Bước 5: Phát hiện vùng đồng lộ bằng K-means clustering + multi-colorspace
  Bước 6: Cắt ROI cuối cùng + pipeline confidence scoring
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass
class ProcessingResult:
    """Kết quả xử lý ảnh qua pipeline 6 bước."""
    roi: np.ndarray                          # ROI cuối cùng (resized)
    copper_w: int = 0                        # Chiều rộng vùng đồng (px)
    copper_h: int = 0                        # Chiều cao vùng đồng (px)
    copper_ratio: float = 0.0               # Tỷ lệ diện tích đồng / sản phẩm
    copper_mask: Optional[np.ndarray] = None # Mask vùng đồng
    outer_rect: Optional[Any] = None         # minAreaRect sản phẩm (backward compat)
    long_angle: float = 0.0                  # Góc xoay
    is_flipped: bool = False                 # Có đảo chiều không

    # === FIELDS MỚI (V2) ===
    pipeline_confidence: float = 1.0         # Mức tin cậy tổng thể pipeline (0.0-1.0)
    product_length: int = 0                  # Chiều dài sản phẩm đo được (px)
    centroid: Tuple[int, int] = (0, 0)       # Tâm sản phẩm trên ảnh gốc (cho tracking)

    # Ảnh trực quan hóa từng bước
    vis_steps: Dict[str, np.ndarray] = field(default_factory=dict)


class ImageProcessor:
    """
    Pipeline xử lý ảnh sản phẩm crimping — Phiên bản bền vững (Robust V2).
    Hoạt động ổn định với mọi góc camera.

    Sử dụng:
        processor = ImageProcessor()
        result = processor.process(img_bgr)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        crop_ratio: float = 0.45,
        fixed_crop_length: int = 300,
        min_contour_ratio: float = 0.02,
        copper_kmeans_clusters: int = 4,
        search_region_ratio: float = 0.60,
        save_vis_steps: bool = True,
    ):
        self.target_size = target_size
        self.crop_ratio = crop_ratio
        self.fixed_crop_length = fixed_crop_length
        self.min_contour_ratio = min_contour_ratio
        self.copper_kmeans_clusters = copper_kmeans_clusters
        self.search_region_ratio = search_region_ratio
        self.save_vis_steps = save_vis_steps

        # Biến lưu trữ kết quả trung gian cho visualization
        self._last_copper_x = 0
        self._last_copper_y = 0

    def process(self, img_in: np.ndarray) -> Optional[ProcessingResult]:
        """
        Chạy pipeline 6 bước trên ảnh đầu vào.
        Trả về ProcessingResult hoặc None nếu thất bại.
        """
        if img_in is None or img_in.size == 0:
            return None

        h, w = img_in.shape[:2]
        vis = {}
        confidence_scores = []  # Thu thập điểm tin cậy từ mỗi bước

        # ═══ BƯỚC 1: TÁCH NỀN BỀN VỮNG ═══
        fg_mask, product_contour, conf1 = self._step1_remove_background(img_in, h, w)
        confidence_scores.append(conf1)
        if fg_mask is None:
            return None

        img_no_bg = cv2.bitwise_and(img_in, img_in, mask=fg_mask)
        if self.save_vis_steps:
            vis["01_bg_removed"] = img_no_bg.copy()

        # Tính centroid trên ảnh gốc (cho tracking)
        M_moments = cv2.moments(product_contour)
        if M_moments["m00"] > 0:
            centroid = (int(M_moments["m10"] / M_moments["m00"]),
                        int(M_moments["m01"] / M_moments["m00"]))
        else:
            centroid = (w // 2, h // 2)

        # ═══ BƯỚC 2: BORDER NGOÀI + PCA ═══
        outer_rect, cx, cy, length, thickness, long_angle, conf2 = \
            self._step2_find_border_pca(fg_mask, product_contour)
        confidence_scores.append(conf2)
        if outer_rect is None:
            return None

        if self.save_vis_steps:
            vis_border = img_in.copy()
            cv2.drawContours(vis_border, [product_contour], -1, (0, 255, 0), 2)
            cv2.circle(vis_border, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            vis["02_contour"] = vis_border

        # ═══ BƯỚC 3: XOAY THẲNG + XÁC ĐỊNH HƯỚNG ═══
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

        # ═══ BƯỚC 4: CẮT CHUẨN HÓA TỰ THÍCH ỨNG ═══
        standardized, std_mask = self._step4_standardize_adaptive(
            cropped, cropped_mask, length
        )
        if standardized is None:
            return None

        if self.save_vis_steps:
            vis["04_standardized"] = standardized.copy()

        h_std, w_std = standardized.shape[:2]

        # ═══ BƯỚC 5: PHÁT HIỆN ĐỒNG LỘ BẰNG K-MEANS ═══
        copper_w, copper_h, _, copper_mask, conf5 = \
            self._step5_detect_copper_adaptive(standardized, std_mask, h_std, w_std)
        confidence_scores.append(conf5)

        if self.save_vis_steps:
            vis_copper = standardized.copy()
            overlay = np.zeros_like(standardized)
            overlay[:, :, 2] = copper_mask
            vis_copper = cv2.addWeighted(vis_copper, 0.7, overlay, 0.3, 0)
            search_x = int(w_std * self.search_region_ratio)
            cv2.line(vis_copper, (search_x, 0), (search_x, h_std),
                     (255, 255, 0), 1)
            if copper_w > 0:
                cv2.rectangle(vis_copper,
                    (self._last_copper_x, self._last_copper_y),
                    (self._last_copper_x + copper_w, self._last_copper_y + copper_h),
                    (0, 255, 0), 2)
            vis["05_copper_detect"] = vis_copper

        # ═══ BƯỚC 6: CẮT ROI ═══
        roi_final, copper_ratio = self._step6_crop_roi(
            standardized, std_mask, copper_w, copper_h, h_std, w_std, copper_mask
        )
        if roi_final is None:
            return None

        if self.save_vis_steps:
            vis["06_roi_final"] = roi_final.copy()

        # Tính pipeline confidence tổng thể
        pipeline_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.5

        return ProcessingResult(
            roi=roi_final,
            copper_w=copper_w,
            copper_h=copper_h,
            copper_ratio=copper_ratio,
            copper_mask=copper_mask,
            outer_rect=outer_rect,
            long_angle=long_angle,
            is_flipped=is_flipped,
            pipeline_confidence=pipeline_confidence,
            product_length=int(length),
            centroid=centroid,
            vis_steps=vis,
        )

    # ──────────────────────────────────────────────────────────
    # Bước 1: Tách nền bền vững (HSV-based + Otsu)
    # ──────────────────────────────────────────────────────────
    def _step1_remove_background(self, img, h, w):
        """
        Tách nền đa chiến lược — hoạt động với BẤT KỲ nền nào:
        1. Phát hiện nền xanh dương (HSV)
        2. Phát hiện dây trắng + kim loại sáng (LAB + saturation)
        3. Edge-based (Canny + morphological) cho nền có texture
        4. CLAHE enhanced edge (cho nền trắng/xám nhạt)
        5. Otsu với blur mạnh (fallback cuối)

        QUAN TRỌNG: Mỗi candidate contour được chấm điểm theo:
        - Aspect ratio (sản phẩm dài)
        - Diện tích hợp lý
        - Edge density BÊN TRONG contour (sản phẩm thật có texture, nền trơn thì không)
        """
        min_area = int(h * w * self.min_contour_ratio)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Tính edge map toàn cục (dùng cho scoring sau)
        edges_global = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)

        candidate_masks = []

        # ── Chiến lược 1: Nền xanh dương ──
        lower_blue = np.array([85, 30, 50])
        upper_blue = np.array([135, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / (h * w)
        if blue_ratio > 0.15:
            fg1 = cv2.bitwise_not(blue_mask)
            candidate_masks.append(("blue_bg", fg1))

        # ── Chiến lược 2: Phát hiện dây trắng + kim loại ──
        l_channel = lab[:, :, 0]
        s_channel = hsv[:, :, 1]
        bright_threshold = np.percentile(l_channel, 65)
        low_sat_threshold = np.percentile(s_channel, 50)
        wire_mask = ((l_channel > bright_threshold) & (s_channel < low_sat_threshold)).astype(np.uint8) * 255
        candidate_masks.append(("wire_detect", wire_mask))

        # ── Chiến lược 3: Edge-based (cho nền có texture) ──
        blurred_edge = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred_edge, 30, 100)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edge, iterations=4)
        edges_filled = edges_closed.copy()
        flood_fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(edges_filled, flood_fill_mask, (0, 0), 255)
        edges_filled = cv2.bitwise_not(edges_filled)
        edges_combined = cv2.bitwise_or(edges_closed, edges_filled)
        candidate_masks.append(("edge_based", edges_combined))

        # ── Chiến lược 4: CLAHE enhanced (cho nền trắng/xám nhạt) ──
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        edges_clahe = cv2.Canny(enhanced, 40, 120)
        kernel_clahe = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        clahe_closed = cv2.morphologyEx(edges_clahe, cv2.MORPH_CLOSE, kernel_clahe, iterations=5)
        clahe_filled = clahe_closed.copy()
        flood_fill_mask2 = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(clahe_filled, flood_fill_mask2, (0, 0), 255)
        clahe_filled = cv2.bitwise_not(clahe_filled)
        clahe_combined = cv2.bitwise_or(clahe_closed, clahe_filled)
        candidate_masks.append(("clahe_edge", clahe_combined))

        # ── Chiến lược 5: Otsu với blur mạnh (fallback) ──
        blurred_heavy = cv2.GaussianBlur(gray, (21, 21), 0)
        _, otsu_mask = cv2.threshold(blurred_heavy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_ratio = np.sum(otsu_mask > 0) / (h * w)
        if fg_ratio > 0.6:
            otsu_mask = cv2.bitwise_not(otsu_mask)
        candidate_masks.append(("otsu_heavy", otsu_mask))

        # ── Đánh giá và chọn chiến lược tốt nhất ──
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        best_product = None
        best_score = -1
        best_mask = None
        best_strategy = ""

        for name, raw_mask in candidate_masks:
            # Morphological cleanup
            cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open, iterations=2)

            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            candidates = [c for c in contours if cv2.contourArea(c) > min_area]
            if not candidates:
                continue

            # Đánh giá TẤT CẢ contour đủ lớn, không chỉ lớn nhất
            for product in sorted(candidates, key=cv2.contourArea, reverse=True)[:3]:
                area = cv2.contourArea(product)

                # Aspect ratio
                rect = cv2.minAreaRect(product)
                rect_w, rect_h = rect[1]
                if min(rect_w, rect_h) > 0:
                    aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
                else:
                    aspect = 1.0

                area_ratio = area / (h * w)

                # ═══ EDGE DENSITY: phân biệt sản phẩm thật vs nền trơn ═══
                # Sản phẩm thật (dây + kim loại + đồng) có nhiều edge bên trong
                # Nền trơn (trắng, xám) gần như không có edge
                contour_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(contour_mask, [product], -1, 255, thickness=cv2.FILLED)
                edge_inside = cv2.bitwise_and(edges_global, contour_mask)
                contour_pixels = max(np.sum(contour_mask > 0), 1)
                edge_density = np.sum(edge_inside > 0) / contour_pixels

                # ═══ Tính điểm tổng hợp ═══
                score = 0.0

                # Aspect ratio: sản phẩm cos đồng luôn dài
                if aspect > 2.5:
                    score += min(aspect, 10.0) * 10
                elif aspect > 1.5:
                    score += aspect * 5
                else:
                    score += aspect * 1

                # Diện tích: ưu tiên 3-40% ảnh
                if 0.03 < area_ratio < 0.40:
                    score += 50
                elif 0.01 < area_ratio <= 0.03:
                    score += 20
                elif area_ratio >= 0.40:
                    score += 5  # Quá lớn = nghi ngờ

                # Edge density: QUAN TRỌNG NHẤT — sản phẩm thật có edge density > 0.02
                if edge_density > 0.05:
                    score += 80  # Rất nhiều edge = chắc chắn là sản phẩm
                elif edge_density > 0.02:
                    score += 40  # Có edge = có thể là sản phẩm
                elif edge_density > 0.01:
                    score += 10
                else:
                    score -= 30  # Gần như không có edge = có thể là nền trơn

                if score > best_score:
                    best_score = score
                    best_product = product
                    best_mask = cleaned
                    best_strategy = name

        if best_product is None:
            return None, None, 0.0

        # Tạo mask sạch chỉ chứa sản phẩm tốt nhất
        clean_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(clean_mask, [best_product], -1, 255, thickness=cv2.FILLED)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_open)

        # Tính confidence
        product_ratio = cv2.contourArea(best_product) / (h * w)
        rect = cv2.minAreaRect(best_product)
        rect_w, rect_h = rect[1]
        aspect = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1)

        # Tính edge density cho confidence
        final_mask_check = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(final_mask_check, [best_product], -1, 255, thickness=cv2.FILLED)
        edge_in = cv2.bitwise_and(edges_global, final_mask_check)
        final_edge_density = np.sum(edge_in > 0) / max(np.sum(final_mask_check > 0), 1)

        conf = 0.3
        if aspect > 3.0:
            conf += 0.25
        elif aspect > 2.0:
            conf += 0.1
        if 0.05 < product_ratio < 0.30:
            conf += 0.15
        if final_edge_density > 0.03:
            conf += 0.3
        elif final_edge_density > 0.01:
            conf += 0.1
        conf = max(0.2, min(1.0, conf))

        return clean_mask, best_product, conf

    # ──────────────────────────────────────────────────────────
    # Bước 2: Border ngoài + PCA
    # ──────────────────────────────────────────────────────────
    def _step2_find_border_pca(self, fg_mask, product_contour):
        """
        Xác định trục chính bằng PCA trên tọa độ contour.
        Bền vững hơn minAreaRect khi có biến dạng phối cảnh.
        """
        # Vẫn tính minAreaRect cho backward compatibility (outer_rect field)
        outer_rect = cv2.minAreaRect(product_contour)
        (cx, cy), (rect_w, rect_h), angle = outer_rect

        # PCA trên tọa độ contour để tìm trục chính
        pts = product_contour.reshape(-1, 2).astype(np.float64)
        mean_pt = np.mean(pts, axis=0)
        centered = pts - mean_pt

        # Tính ma trận hiệp phương sai và eigenvectors
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Eigenvector ứng với eigenvalue lớn nhất = trục chính
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Góc trục chính (tính từ trục x)
        principal_axis = eigenvectors[:, 0]
        long_angle = np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))

        # Chiều dài và chiều rộng ước tính từ eigenvalues
        length = 4.0 * np.sqrt(eigenvalues[0])  # ~95% dải phân bố
        thickness = 4.0 * np.sqrt(max(eigenvalues[1], 1.0))

        # Confidence: dựa trên tỷ lệ eigenvalues (sản phẩm dài → tỷ lệ cao → tin cậy)
        eigen_ratio = eigenvalues[0] / max(eigenvalues[1], 1.0)
        conf = min(1.0, eigen_ratio / 10.0)
        conf = max(0.3, conf)

        cx, cy = mean_pt[0], mean_pt[1]

        return outer_rect, cx, cy, length, thickness, long_angle, conf

    # ──────────────────────────────────────────────────────────
    # Bước 3: Xoay thẳng + xác định hướng đầu cos
    # ──────────────────────────────────────────────────────────
    def _step3_rotate(self, img_no_bg, fg_mask, cx, cy, length, long_angle, h, w):
        """
        Xoay sản phẩm nằm ngang.
        Xác định hướng đầu cos bằng phân tích sáng/màu thay vì chỉ so sánh độ dày.
        """
        rad = np.deg2rad(long_angle)
        dx, dy = np.cos(rad), np.sin(rad)
        nx, ny = -dy, dx

        # Lấy 2 điểm ở 2 đầu sản phẩm dọc theo trục chính
        pt1 = (int(cx + 0.33 * length * dx), int(cy + 0.33 * length * dy))
        pt2 = (int(cx - 0.33 * length * dx), int(cy - 0.33 * length * dy))

        # Phương pháp 1: So sánh độ sáng trung bình (phần cos kim loại sáng hơn)
        brightness1 = self._measure_local_brightness(img_no_bg, fg_mask, *pt1, radius=30)
        brightness2 = self._measure_local_brightness(img_no_bg, fg_mask, *pt2, radius=30)

        # Phương pháp 2: So sánh độ dày (giữ lại như backup)
        thick1 = self._measure_thickness(fg_mask, *pt1, nx, ny)
        thick2 = self._measure_thickness(fg_mask, *pt2, nx, ny)

        # Quyết định hướng: đầu cos = đầu sáng hơn VÀ dày hơn
        # Nếu 2 phương pháp đồng ý → tin cậy cao; nếu không → ưu tiên brightness
        brightness_vote = 1 if brightness1 > brightness2 else -1
        thickness_vote = 1 if thick1 > thick2 else -1

        is_flipped = False
        if brightness_vote == -1:
            # Đầu 2 sáng hơn → đổi hướng
            dx, dy = -dx, -dy
            long_angle += 180
            is_flipped = True
        elif brightness_vote == 0 and thickness_vote == -1:
            # Brightness bằng nhau → dùng thickness
            dx, dy = -dx, -dy
            long_angle += 180
            is_flipped = True

        M = cv2.getRotationMatrix2D((cx, cy), -long_angle, 1.0)
        rotated = cv2.warpAffine(img_no_bg, M, (w, h), flags=cv2.INTER_CUBIC)
        rot_mask = cv2.warpAffine(fg_mask, M, (w, h), flags=cv2.INTER_NEAREST)

        return rotated, rot_mask, long_angle, is_flipped

    def _crop_tight(self, rotated, rot_mask, thickness, h, w):
        """Cắt sát sản phẩm sau xoay."""
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
    # Bước 4: Cắt chuẩn hóa tự thích ứng
    # ──────────────────────────────────────────────────────────
    def _step4_standardize_adaptive(self, cropped, cropped_mask, product_length):
        """
        Cắt chuẩn hóa theo tỷ lệ chiều dài sản phẩm thay vì pixel cố định.
        Đảm bảo vùng crop bao phủ đầu cos + vùng crimp + phần dây.
        """
        h_c, w_c = cropped.shape[:2]
        col_has_fg = np.any(cropped_mask > 0, axis=0)
        fg_cols = np.where(col_has_fg)[0]
        if len(fg_cols) == 0:
            return None, None

        # Chiều dài thực tế đo được trong ảnh đã cắt
        actual_length = fg_cols[-1] - fg_cols[0]

        # Tính chiều dài crop: dùng tỷ lệ crop_ratio so với chiều dài sản phẩm
        crop_length = int(actual_length * self.crop_ratio)
        # Fallback: nếu crop_length quá nhỏ, dùng fixed_crop_length
        crop_length = max(crop_length, min(self.fixed_crop_length, actual_length))

        x_tip = fg_cols[-1]
        x_start = max(0, x_tip - crop_length)
        x_end = min(w_c, x_tip + 5)

        std = cropped[:, x_start:x_end]
        std_mask = cropped_mask[:, x_start:x_end]

        if std.size == 0:
            return None, None
        return std, std_mask

    # ──────────────────────────────────────────────────────────
    # Bước 5: Phát hiện đồng lộ bằng K-means + multi-colorspace
    # ──────────────────────────────────────────────────────────
    def _step5_detect_copper_adaptive(self, standardized, std_mask, h_std, w_std):
        """
        Phát hiện vùng đồng lộ bằng K-means clustering thay vì ngưỡng cứng.
        Kết hợp xác nhận từ nhiều color space (LAB, HSV, YCrCb).
        """
        self._last_copper_x = 0
        self._last_copper_y = 0

        # Giới hạn vùng tìm kiếm (phần đầu sản phẩm)
        search_x = int(w_std * self.search_region_ratio)

        # Lấy pixels thuộc sản phẩm trong vùng tìm kiếm
        search_mask = std_mask.copy()
        search_mask[:, search_x:] = 0
        fg_pixels_idx = np.where(search_mask > 0)

        if len(fg_pixels_idx[0]) < 100:
            # Quá ít pixel → không đủ dữ liệu
            return 0, 0, 0.0, np.zeros((h_std, w_std), dtype=np.uint8), 0.3

        # === K-means trên LAB color space ===
        lab = cv2.cvtColor(standardized, cv2.COLOR_BGR2LAB)
        fg_lab = lab[fg_pixels_idx[0], fg_pixels_idx[1]]

        n_clusters = min(self.copper_kmeans_clusters, len(fg_lab) // 10)
        n_clusters = max(2, n_clusters)

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3, batch_size=256)
        labels = kmeans.fit_predict(fg_lab)
        centers = kmeans.cluster_centers_

        # Tìm cluster có đặc tính đồng: kênh a (LAB) cao nhất = màu đỏ/cam
        # Đồng có a_channel cao (hướng đỏ) và b_channel dương (hướng vàng)
        copper_scores = []
        for i, center in enumerate(centers):
            l_val, a_val, b_val = center
            # Score: ưu tiên a cao (đỏ/cam), b dương (vàng), l trung bình
            score = a_val * 1.5 + max(b_val - 128, 0) * 0.5 - abs(l_val - 140) * 0.3
            copper_scores.append(score)

        copper_cluster_idx = np.argmax(copper_scores)
        copper_mask_lab = np.zeros((h_std, w_std), dtype=np.uint8)
        copper_pixels = labels == copper_cluster_idx
        copper_mask_lab[fg_pixels_idx[0][copper_pixels], fg_pixels_idx[1][copper_pixels]] = 255

        # === Xác nhận bằng HSV ===
        hsv = cv2.cvtColor(standardized, cv2.COLOR_BGR2HSV)
        fg_hsv = hsv[fg_pixels_idx[0], fg_pixels_idx[1]]
        # Đồng: saturation tương đối cao, hue trong vùng cam/đỏ
        h_ch = fg_hsv[:, 0]
        s_ch = fg_hsv[:, 1]
        # Ngưỡng adaptive: dùng percentile thay vì giá trị cố định
        s_threshold = max(np.percentile(s_ch, 70), 80)
        hsv_copper = ((s_ch > s_threshold) & ((h_ch < 30) | (h_ch > 160))).astype(np.uint8) * 255
        copper_mask_hsv = np.zeros((h_std, w_std), dtype=np.uint8)
        copper_mask_hsv[fg_pixels_idx[0], fg_pixels_idx[1]] = hsv_copper

        # === Xác nhận bằng YCrCb ===
        ycrcb = cv2.cvtColor(standardized, cv2.COLOR_BGR2YCrCb)
        fg_ycrcb = ycrcb[fg_pixels_idx[0], fg_pixels_idx[1]]
        cr_ch = fg_ycrcb[:, 1]
        # Đồng: Cr (red chroma) cao
        cr_threshold = max(np.percentile(cr_ch, 75), 140)
        ycrcb_copper = (cr_ch > cr_threshold).astype(np.uint8) * 255
        copper_mask_ycrcb = np.zeros((h_std, w_std), dtype=np.uint8)
        copper_mask_ycrcb[fg_pixels_idx[0], fg_pixels_idx[1]] = ycrcb_copper

        # === Multi-colorspace fusion: ít nhất 2/3 color space đồng ý ===
        vote_sum = (copper_mask_lab.astype(np.float32) / 255.0 +
                    copper_mask_hsv.astype(np.float32) / 255.0 +
                    copper_mask_ycrcb.astype(np.float32) / 255.0)
        copper_mask = (vote_sum >= 2.0).astype(np.uint8) * 255

        # Morphological cleanup
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        copper_mask = cv2.morphologyEx(copper_mask, cv2.MORPH_CLOSE, k, iterations=2)
        copper_mask = cv2.morphologyEx(copper_mask, cv2.MORPH_OPEN, k)

        # Connected component analysis: giữ chỉ các component lớn
        num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
            copper_mask, connectivity=8
        )
        min_component_area = 30
        clean_mask = np.zeros_like(copper_mask)
        for i in range(1, num_labels):  # Bỏ qua background (label 0)
            if stats[i, cv2.CC_STAT_AREA] > min_component_area:
                clean_mask[labels_cc == i] = 255
        copper_mask = clean_mask

        # Tìm bounding box tổng thể của vùng đồng
        contours, _ = cv2.findContours(
            copper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        c_x, c_y, c_w, c_h = 0, 0, 0, 0
        if contours:
            valid = [c for c in contours if cv2.contourArea(c) > min_component_area]
            if valid:
                all_pts = np.vstack(valid)
                c_x, c_y, c_w, c_h = cv2.boundingRect(all_pts)

        self._last_copper_x = c_x
        self._last_copper_y = c_y

        copper_pixels_count = np.sum(copper_mask > 0)
        total_pixels = max(np.sum(search_mask > 0), 1)
        copper_ratio = copper_pixels_count / total_pixels

        # Confidence: dựa trên số pixel đồng phát hiện được
        conf = min(1.0, copper_pixels_count / 200.0) if copper_pixels_count > 0 else 0.2
        conf = max(0.2, conf)

        return c_w, c_h, copper_ratio, copper_mask, conf

    # ──────────────────────────────────────────────────────────
    # Bước 6: Cắt ROI cuối cùng
    # ──────────────────────────────────────────────────────────
    def _step6_crop_roi(self, standardized, std_mask, copper_w, copper_h,
                        h_std, w_std, copper_mask):
        """Cắt ROI cuối cùng với fallback cải thiện."""
        if copper_w > 0 and copper_h > 0:
            pad = 10
            y1 = max(0, self._last_copper_y - pad)
            y2 = min(h_std, self._last_copper_y + copper_h + pad)
            x1 = max(0, self._last_copper_x - pad)
            x2 = min(w_std, self._last_copper_x + copper_w + pad)
            roi_raw = standardized[y1:y2, x1:x2]
            roi_std_mask = std_mask[y1:y2, x1:x2]
            roi_copper_mask = copper_mask[y1:y2, x1:x2]
        else:
            # Fallback cải thiện: crop vùng trung tâm sản phẩm theo tỷ lệ
            fg_cols = np.where(np.any(std_mask > 0, axis=0))[0]
            fg_rows = np.where(np.any(std_mask > 0, axis=1))[0]

            if len(fg_cols) > 0 and len(fg_rows) > 0:
                # Crop 30-60% chiều dài sản phẩm (vùng giữa có khả năng chứa đồng lộ)
                x_range = fg_cols[-1] - fg_cols[0]
                ix_l = fg_cols[0] + int(x_range * 0.15)
                ix_r = fg_cols[0] + int(x_range * 0.55)
                iy_t = max(0, fg_rows[0] - 5)
                iy_b = min(h_std, fg_rows[-1] + 5)
            else:
                ix_l = int(w_std * 0.15)
                ix_r = int(w_std * 0.50)
                iy_t, iy_b = int(h_std * 0.2), int(h_std * 0.8)

            roi_raw = standardized[iy_t:iy_b, ix_l:ix_r]
            roi_std_mask = std_mask[iy_t:iy_b, ix_l:ix_r]
            roi_copper_mask = copper_mask[iy_t:iy_b, ix_l:ix_r]

        if roi_raw.size == 0:
            return None, 0.0

        # Tính tỷ lệ diện tích đồng trên diện tích sản phẩm (foreground) trong ROI
        roi_copper_pixels = np.sum(roi_copper_mask > 0)
        roi_total_pixels = max(np.sum(roi_std_mask > 0), 1)
        copper_ratio = float(roi_copper_pixels / roi_total_pixels)

        return cv2.resize(roi_raw, self.target_size, interpolation=cv2.INTER_CUBIC), copper_ratio

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _measure_thickness(mask, px, py, nx, ny, max_search=200):
        """Đo độ dày sản phẩm tại điểm (px, py) theo hướng pháp tuyến (nx, ny)."""
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

    @staticmethod
    def _measure_local_brightness(img_bgr, mask, px, py, radius=30):
        """
        Đo độ sáng trung bình trong vùng tròn xung quanh điểm (px, py).
        Chỉ tính trên pixel thuộc foreground (mask > 0).
        """
        h, w = img_bgr.shape[:2]
        px, py = int(px), int(py)

        # Giới hạn bounding box
        y1 = max(0, py - radius)
        y2 = min(h, py + radius)
        x1 = max(0, px - radius)
        x2 = min(w, px + radius)

        if y2 <= y1 or x2 <= x1:
            return 0.0

        region = img_bgr[y1:y2, x1:x2]
        region_mask = mask[y1:y2, x1:x2]

        # Chuyển sang grayscale để đo brightness
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Chỉ tính mean trên pixel foreground
        fg_pixels = gray_region[region_mask > 0]
        if len(fg_pixels) == 0:
            return 0.0

        return float(np.mean(fg_pixels))
