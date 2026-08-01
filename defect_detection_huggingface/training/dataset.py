import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from loguru import logger
import config

def augment_image(img, aug_idx):
    """
    Áp dụng các phép tăng cường dữ liệu khác nhau dựa trên aug_idx.
    V2: Thêm perspective transform, shear, scale, color jitter, cutout.
    aug_idx = 0: Không biến đổi (ảnh gốc)
    """
    if aug_idx == 0:
        return img.copy()

    h, w = img.shape[:2]

    # 1. Flip ngang (đối xứng qua trục đứng)
    if aug_idx == 1:
        return cv2.flip(img, 1)

    # 2. Flip dọc (đối xứng qua trục ngang)
    elif aug_idx == 2:
        return cv2.flip(img, 0)

    # 3. Xoay góc nhỏ nhẹ (từ -10 đến -5 độ)
    elif aug_idx == 3:
        angle = np.random.uniform(-10, -5)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 4. Xoay góc nhỏ nhẹ (từ 5 đến 10 độ)
    elif aug_idx == 4:
        angle = np.random.uniform(5, 10)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 5. Tăng độ sáng
    elif aug_idx == 5:
        return cv2.convertScaleAbs(img, alpha=1.15, beta=15)

    # 6. Giảm độ sáng
    elif aug_idx == 6:
        return cv2.convertScaleAbs(img, alpha=0.85, beta=-15)

    # 7. Làm mờ nhẹ (Gaussian Blur)
    elif aug_idx == 7:
        return cv2.GaussianBlur(img, (3, 3), 0)

    # 8. Thêm nhiễu Gauss (Gaussian Noise)
    elif aug_idx == 8:
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # === AUGMENTATION MỚI V2 ===

    # 9. Perspective Transform — Dịch 4 góc ngẫu nhiên ±15px để simulate thay đổi góc camera
    elif aug_idx == 9:
        margin = 15
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        pts2 = np.float32([
            [np.random.randint(0, margin), np.random.randint(0, margin)],
            [w - np.random.randint(0, margin), np.random.randint(0, margin)],
            [w - np.random.randint(0, margin), h - np.random.randint(0, margin)],
            [np.random.randint(0, margin), h - np.random.randint(0, margin)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 10. Affine Shear — Biến dạng cắt ngẫu nhiên
    elif aug_idx == 10:
        shear_factor = np.random.uniform(-0.1, 0.1)
        M = np.float32([
            [1, shear_factor, 0],
            [0, 1, 0]
        ])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 11. Scale Variations — Phóng to/thu nhỏ ngẫu nhiên 0.7x đến 1.3x
    elif aug_idx == 11:
        scale = np.random.uniform(0.7, 1.3)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # Pad hoặc crop để trở lại kích thước gốc
        if scale > 1.0:
            # Crop trung tâm
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            return resized[start_y:start_y + h, start_x:start_x + w]
        else:
            # Pad bằng reflection
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            return cv2.copyMakeBorder(resized, pad_y, h - new_h - pad_y,
                                       pad_x, w - new_w - pad_x,
                                       cv2.BORDER_REFLECT)

    # 12. Strong Color Jitter — Thay đổi HSV channels mạnh để simulate ánh sáng khác nhau
    elif aug_idx == 12:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Dịch Hue ngẫu nhiên ±15
        hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-15, 15)) % 180
        # Scale Saturation ngẫu nhiên 0.7-1.3
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)
        # Scale Value ngẫu nhiên 0.7-1.3
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.7, 1.3), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 13. Random Cutout — Che 1-3 vùng ngẫu nhiên kích thước 20-40px
    elif aug_idx == 13:
        aug = img.copy()
        n_holes = np.random.randint(1, 4)
        for _ in range(n_holes):
            hole_h = np.random.randint(20, min(40, h // 2))
            hole_w = np.random.randint(20, min(40, w // 2))
            y_start = np.random.randint(0, max(1, h - hole_h))
            x_start = np.random.randint(0, max(1, w - hole_w))
            # Điền bằng giá trị mean của ảnh (thay vì đen)
            mean_color = np.mean(img, axis=(0, 1)).astype(np.uint8)
            aug[y_start:y_start + hole_h, x_start:x_start + hole_w] = mean_color
        return aug

    # Các tổ hợp ngẫu nhiên nâng cao cho chỉ số cao hơn
    else:
        aug = img.copy()
        # Tổ hợp ngẫu nhiên: flip + rotate + brightness + perspective
        if np.random.rand() > 0.5:
            aug = cv2.flip(aug, 1)

        # Xoay nhẹ
        angle = np.random.uniform(-12, 12)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Brightness + contrast
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(-20, 20)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

        # 50% xác suất thêm perspective nhẹ
        if np.random.rand() > 0.5:
            margin = 10
            pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            pts2 = np.float32([
                [np.random.randint(0, margin), np.random.randint(0, margin)],
                [w - np.random.randint(0, margin), np.random.randint(0, margin)],
                [w - np.random.randint(0, margin), h - np.random.randint(0, margin)],
                [np.random.randint(0, margin), h - np.random.randint(0, margin)]
            ])
            M_p = cv2.getPerspectiveTransform(pts1, pts2)
            aug = cv2.warpPerspective(aug, M_p, (w, h), borderMode=cv2.BORDER_REFLECT)

        return aug


class DefectDataset(Dataset):
    """
    Dataset PyTorch tích hợp bộ tiền xử lý (ImageProcessor) của Hugging Face
    và kỹ thuật Data Augmentation động (V2 — bao gồm perspective, shear, scale, cutout).
    """
    def __init__(self, image_paths, labels, hf_image_processor, augment=True, n_augment=15):
        self.image_paths = image_paths
        self.labels = labels
        self.hf_image_processor = hf_image_processor
        self.augment = augment
        self.n_augment = n_augment

        self.num_base_images = len(image_paths)
        if self.augment:
            self.total_samples = self.num_base_images * (self.n_augment + 1)
        else:
            self.total_samples = self.num_base_images

        logger.info(f"Dataset initialized: Base images={self.num_base_images}, "
                    f"Augment={self.augment}, Total samples={self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if self.augment:
            base_idx = idx // (self.n_augment + 1)
            aug_idx = idx % (self.n_augment + 1)
        else:
            base_idx = idx
            aug_idx = 0

        img_path = self.image_paths[base_idx]
        label = self.labels[base_idx]

        # Đọc ảnh gốc bằng OpenCV (BGR)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {img_path}")

        # Áp dụng Data Augmentation động
        img_aug = augment_image(img, aug_idx)

        # Chuyển đổi BGR (OpenCV) -> RGB (PIL Image) theo chuẩn đầu vào của Hugging Face
        img_rgb = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Chạy qua Hugging Face ImageProcessor (chuẩn hóa size, mean, std)
        inputs = self.hf_image_processor(images=pil_img, return_tensors="pt")

        # Hugging Face trả về dạng batch (1, C, H, W) nên cần squeeze chiều đầu tiên
        pixel_values = inputs["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }
