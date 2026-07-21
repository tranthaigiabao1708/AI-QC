import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from loguru import logger
import config

def augment_image(img, aug_idx):
    """
    Áp dụng phép tăng cường dữ liệu dựa trên chỉ số aug_idx.
    aug_idx = 0: Không biến đổi (ảnh gốc)
    aug_idx = 1: Lật ngang
    aug_idx = 2: Lật dọc
    aug_idx = 3..4: Xoay góc nhỏ (-10 đến 10 độ)
    aug_idx = 5..6: Tăng/giảm độ sáng
    aug_idx = 7: Làm mờ Gaussian Blur
    aug_idx = 8: Thêm nhiễu Gauss
    """
    if aug_idx == 0:
        return img.copy()

    h, w = img.shape[:2]
    
    # =====================================================================
    # TODO: HÃY TỰ VIẾT CÁC PHÉP BIẾN ĐỔI ẢNH DƯỚI ĐÂY
    # =====================================================================
    
    # 1. Flip ngang (Gợi ý: Dùng cv2.flip(img, 1))
    if aug_idx == 1:
        # TODO: Trả về ảnh lật ngang
        pass
    
    # 2. Flip dọc (Gợi ý: Dùng cv2.flip(img, 0))
    elif aug_idx == 2:
        # TODO: Trả về ảnh lật dọc
        pass
    
    # 3. Xoay góc nhỏ (-10 đến -5 độ)
    # Gợi ý: 
    # M = cv2.getRotationMatrix2D((w // 2, h // 2), góc_xoay, 1.0)
    # return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif aug_idx == 3:
        # TODO: Tính ma trận xoay M và trả về ảnh đã xoay nhẹ trái
        pass
        
    elif aug_idx == 4:
        # TODO: Tính ma trận xoay M và trả về ảnh đã xoay nhẹ phải
        pass
    
    # 4. Tăng/Giảm độ sáng (Gợi ý: Dùng cv2.convertScaleAbs(img, alpha=..., beta=...))
    elif aug_idx == 5:
        # TODO: Trả về ảnh sáng hơn
        pass
        
    elif aug_idx == 6:
        # TODO: Trả về ảnh tối hơn
        pass
        
    # 5. Làm mờ nhẹ (Gaussian Blur) (Gợi ý: Dùng cv2.GaussianBlur(img, (3, 3), 0))
    elif aug_idx == 7:
        # TODO: Trả về ảnh mờ nhẹ
        pass
    
    # 6. Thêm nhiễu Gauss ngẫu nhiên
    # Gợi ý:
    # noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    # return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif aug_idx == 8:
        # TODO: Trả về ảnh đã cộng thêm nhiễu Gauss
        pass
    
    # Tổ hợp ngẫu nhiên cho chỉ số cao hơn
    else:
        return img.copy()


class DefectDataset(Dataset):
    """
    Dataset PyTorch quản lý việc đọc dữ liệu nhãn OK/NG,
    áp dụng data augmentation động và chuyển đổi qua Hugging Face ImageProcessor.
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
            
        logger.info(f"Dataset thực hành khởi tạo: Base images={self.num_base_images}, Total={self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # =====================================================================
        # TODO: HÃY TỰ VIẾT LOGIC LẤY ẢNH VÀ XỬ LÝ SANG TENSOR HUGGING FACE
        # =====================================================================
        
        # 1. Tính toán base_idx và aug_idx dựa trên chỉ số idx
        # Gợi ý:
        # Nếu self.augment là True:
        #   base_idx = idx // (self.n_augment + 1)
        #   aug_idx = idx % (self.n_augment + 1)
        # Ngược lại:
        #   base_idx = idx, aug_idx = 0
        base_idx = 0 # TODO: Sửa lại cách tính
        aug_idx = 0  # TODO: Sửa lại cách tính
        
        # 2. Lấy đường dẫn ảnh và nhãn tại vị trí base_idx
        img_path = self.image_paths[base_idx]
        label = self.labels[base_idx]
        
        # 3. Đọc ảnh bằng OpenCV (BGR)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {img_path}")
            
        # 4. Chạy qua hàm augment_image(img, aug_idx) ở trên
        img_aug = img # TODO: Sửa lại để áp dụng augment_image
        
        # 5. Chuyển đổi màu BGR (OpenCV) -> RGB (Chuẩn đầu vào của PIL & Hugging Face)
        # Gợi ý: Dùng cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB) và Image.fromarray(...)
        pil_img = None # TODO: Chuyển đổi ảnh OpenCV sang ảnh PIL RGB
        
        # 6. Chuyển đổi ảnh sang Tensor sử dụng bộ xử lý của Hugging Face
        # Gợi ý:
        # inputs = self.hf_image_processor(images=pil_img, return_tensors="pt")
        # pixel_values = inputs["pixel_values"].squeeze(0)  # Squeeze để bỏ chiều batch index thừa
        pixel_values = None # TODO: Thực hiện qua hf_image_processor
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }
