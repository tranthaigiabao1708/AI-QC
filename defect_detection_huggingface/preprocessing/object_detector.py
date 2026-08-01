"""
preprocessing/object_detector.py
────────────────────────────────
Module phát hiện sản phẩm cos đồng (crimped terminal) bằng Feature Matching.

Sử dụng SIFT features từ ảnh reference (đã tách nền) để tìm vị trí sản phẩm
trong ảnh mới — bất kể nền, ánh sáng, hay góc camera.

Pipeline:
  1. Trích SIFT features từ ảnh reference (chỉ 1 lần khi khởi tạo)
  2. Với mỗi ảnh mới: trích SIFT → FLANN matching → RANSAC → homography
  3. Trả về bounding box + mask của sản phẩm
  4. Nếu feature matching thất bại → fallback về rule-based (bảo toàn hệ thống cũ)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np
from loguru import logger


class ProductDetector:
    """
    Phát hiện sản phẩm cos đồng bằng SIFT Feature Matching.
    
    Tự động tải ảnh reference từ thư mục chỉ định và trích features.
    Khi detect(), so khớp features với ảnh đầu vào để tìm vị trí sản phẩm.
    """

    MIN_MATCH_COUNT = 8          # Số matches tối thiểu để chấp nhận
    LOWE_RATIO = 0.7             # Lowe ratio test threshold
    RANSAC_REPROJ_THRESH = 5.0   # RANSAC reprojection threshold

    def __init__(self, reference_dir: Optional[str] = None, min_match_count: int = 8):
        """
        Args:
            reference_dir: Thư mục chứa ảnh reference đã tách nền.
                           Nếu None, sẽ dùng thư mục mặc định (data/reference_templates/)
            min_match_count: Số feature matches tối thiểu
        """
        self.min_match_count = min_match_count
        self.sift = cv2.SIFT_create(nfeatures=1000)
        
        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Lưu reference features
        self.ref_features: List[dict] = []
        
        # Tải reference images
        if reference_dir:
            self._load_references(Path(reference_dir))
        
        self._is_ready = len(self.ref_features) > 0
        if self._is_ready:
            logger.info(f"ProductDetector sẵn sàng với {len(self.ref_features)} ảnh reference.")
        else:
            logger.warning("ProductDetector: Không có ảnh reference — sẽ dùng fallback detection.")

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _load_references(self, ref_dir: Path):
        """Tải và trích features từ tất cả ảnh trong thư mục reference."""
        if not ref_dir.exists():
            logger.warning(f"Thư mục reference không tồn tại: {ref_dir}")
            return

        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for img_path in sorted(ref_dir.iterdir()):
            if img_path.suffix.lower() not in image_exts:
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Tăng cường contrast trước khi trích features
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            kp, des = self.sift.detectAndCompute(enhanced, None)
            
            if des is not None and len(kp) >= 10:
                # Lưu mask cho ảnh đã tách nền (tự động detect vùng có nội dung)
                _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                
                self.ref_features.append({
                    'name': img_path.stem,
                    'keypoints': kp,
                    'descriptors': des,
                    'image': img,
                    'gray': enhanced,
                    'shape': img.shape[:2],
                    'mask': mask,
                })
                logger.info(f"  → Đã tải reference: {img_path.name} ({len(kp)} features)")

    def add_reference_image(self, img: np.ndarray, name: str = "runtime_ref"):
        """Thêm 1 ảnh reference lúc runtime (không cần từ file)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        kp, des = self.sift.detectAndCompute(enhanced, None)
        
        if des is not None and len(kp) >= 10:
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            self.ref_features.append({
                'name': name,
                'keypoints': kp,
                'descriptors': des,
                'image': img,
                'gray': enhanced,
                'shape': img.shape[:2],
                'mask': mask,
            })
            self._is_ready = True
            logger.info(f"  → Thêm reference: {name} ({len(kp)} features)")

    def detect(self, img: np.ndarray) -> Optional[dict]:
        """
        Phát hiện sản phẩm trong ảnh đầu vào bằng feature matching.
        
        Returns:
            dict với keys:
                - bbox: (x, y, w, h) bounding box
                - mask: binary mask vùng sản phẩm
                - contour: contour polygon
                - confidence: float 0.0-1.0
                - match_count: số matches tìm được
                - ref_name: tên ảnh reference khớp nhất
            Hoặc None nếu không tìm thấy.
        """
        if not self._is_ready:
            return None
        
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        kp_query, des_query = self.sift.detectAndCompute(enhanced, None)
        
        if des_query is None or len(kp_query) < 5:
            return None

        best_result = None
        best_match_count = 0

        for ref in self.ref_features:
            result = self._match_single_ref(
                kp_query, des_query, ref, h, w
            )
            if result is not None and result['match_count'] > best_match_count:
                best_match_count = result['match_count']
                best_result = result

        return best_result

    def _match_single_ref(
        self, kp_query, des_query, ref: dict, h: int, w: int
    ) -> Optional[dict]:
        """So khớp features với 1 ảnh reference."""
        des_ref = ref['descriptors']
        kp_ref = ref['keypoints']
        ref_h, ref_w = ref['shape']

        # Cần ít nhất 2 features để match
        if len(kp_ref) < 2 or len(kp_query) < 2:
            return None

        try:
            matches = self.flann.knnMatch(des_ref, des_query, k=2)
        except cv2.error:
            return None

        # Lowe ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.LOWE_RATIO * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.min_match_count:
            return None

        # RANSAC homography
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_query[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask_ransac = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.RANSAC_REPROJ_THRESH)
        
        if H is None:
            return None

        inliers = int(mask_ransac.sum()) if mask_ransac is not None else 0
        if inliers < self.min_match_count * 0.5:
            return None

        # Biến đổi bounding box reference → ảnh query
        ref_corners = np.float32([
            [0, 0],
            [ref_w, 0],
            [ref_w, ref_h],
            [0, ref_h]
        ]).reshape(-1, 1, 2)

        dst_corners = cv2.perspectiveTransform(ref_corners, H)
        
        if dst_corners is None:
            return None

        # Kiểm tra tính hợp lệ của bounding box
        dst_pts_flat = dst_corners.reshape(-1, 2)
        
        # Bounding box phải nằm trong ảnh (cho phép margin 10%)
        margin = max(h, w) * 0.1
        if (np.any(dst_pts_flat < -margin) or 
            np.any(dst_pts_flat[:, 0] > w + margin) or
            np.any(dst_pts_flat[:, 1] > h + margin)):
            return None

        # Diện tích bounding box phải hợp lý (1-80% ảnh)
        area = cv2.contourArea(dst_corners.astype(np.int32))
        area_ratio = area / (h * w)
        if area_ratio < 0.01 or area_ratio > 0.80:
            return None

        # Kiểm tra aspect ratio (sản phẩm cos đồng phải dài)
        rect = cv2.minAreaRect(dst_corners.astype(np.int32))
        rect_w_val, rect_h_val = rect[1]
        if min(rect_w_val, rect_h_val) > 0:
            aspect = max(rect_w_val, rect_h_val) / min(rect_w_val, rect_h_val)
        else:
            return None
        
        # Sản phẩm phải có aspect > 1.5 (hình dài)
        if aspect < 1.3:
            return None

        # Tạo mask từ polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        polygon = dst_corners.astype(np.int32)
        cv2.fillConvexPoly(mask, polygon.reshape(-1, 2), 255)

        # Tạo contour từ mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        product_contour = max(contours, key=cv2.contourArea)

        # Bounding box
        x, y, bw, bh = cv2.boundingRect(product_contour)

        # Confidence dựa trên inlier ratio + match count
        inlier_ratio = inliers / len(good_matches) if good_matches else 0
        match_confidence = min(1.0, len(good_matches) / 30)  # Normalize: 30+ matches = 1.0
        confidence = 0.4 * inlier_ratio + 0.6 * match_confidence
        confidence = max(0.3, min(1.0, confidence))

        return {
            'bbox': (x, y, bw, bh),
            'mask': mask,
            'contour': product_contour,
            'confidence': confidence,
            'match_count': len(good_matches),
            'inliers': inliers,
            'ref_name': ref['name'],
            'homography': H,
            'dst_corners': dst_corners,
        }


def create_reference_from_pipeline(
    raw_image: np.ndarray,
    bg_removed_image: np.ndarray,
    output_dir: Path,
    name: str = "ref"
) -> Optional[Path]:
    """
    Tiện ích: Tạo ảnh reference từ ảnh đã chạy pipeline (đã tách nền).
    Lưu ảnh đã tách nền vào thư mục reference.
    """
    if bg_removed_image is None or bg_removed_image.size == 0:
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.png"
    cv2.imwrite(str(output_path), bg_removed_image)
    logger.info(f"Đã tạo reference template: {output_path}")
    return output_path
