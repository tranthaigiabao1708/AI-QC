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
    
    # 3. Xoay góc nhỏ nhẹ (từ -10 đến 10 độ)
    elif aug_idx == 3:
        angle = np.random.uniform(-10, -5)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif aug_idx == 4:
        angle = np.random.uniform(5, 10)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 4. Tăng/Giảm độ sáng
    elif aug_idx == 5:
        return cv2.convertScaleAbs(img, alpha=1.15, beta=15)
        
    elif aug_idx == 6:
        return cv2.convertScaleAbs(img, alpha=0.85, beta=-15)
        
    # 5. Làm mờ nhẹ (Gaussian Blur)
    elif aug_idx == 7:
        return cv2.GaussianBlur(img, (3, 3), 0)
    
    # 6. Thêm nhiễu Gauss (Gaussian Noise)
    elif aug_idx == 8:
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Các tổ hợp ngẫu nhiên khác cho chỉ số cao hơn
    else:
        # Tổ hợp ngẫu nhiên giữa xoay, dịch chuyển độ sáng và lật
        aug = img.copy()
        if np.random.rand() > 0.5:
            aug = cv2.flip(aug, 1)
        angle = np.random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        factor = np.random.uniform(0.9, 1.1)
        aug = np.clip(aug * factor, 0, 255).astype(np.uint8)
        return aug


class DefectDataset(Dataset):
    """
    Dataset PyTorch tích hợp bộ tiền xử lý (ImageProcessor) của Hugging Face
    và kỹ thuật Data Augmentation động.
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
        # Trả về tensor PyTorch
        inputs = self.hf_image_processor(images=pil_img, return_tensors="pt")
        
        # Hugging Face trả về dạng batch (1, C, H, W) nên cần squeeze chiều đầu tiên
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }
