"""
tools/auto_label_yolo.py
────────────────────────
Phase 1: Tự động tạo YOLO segmentation dataset từ ảnh gốc.

Workflow:
1. Chạy pipeline CV _step1_remove_background trên 14 ảnh gốc (pipeline detect ĐÚNG trên ảnh gốc)
2. Trích xuất fg_mask → chuyển contour thành YOLO polygon format
3. Augment ảnh + mask ĐỒNG THỜI (đảm bảo label luôn khớp)
4. Chia train/val (80/20)
5. Tạo dataset.yaml cho YOLOv8 training

Output: data/yolo_dataset/
"""

import sys
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from preprocessing.image_processor import ImageProcessor

RAW_DIR = PROJECT_DIR / "data" / "raw_images"
DATASET_DIR = PROJECT_DIR / "data" / "yolo_dataset"
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Augmentation settings
AUGMENTATIONS = [
    # (name, function)
    # Mỗi function nhận (img, mask) trả về (aug_img, aug_mask)
]


def rotate_with_mask(img, mask, angle):
    """Xoay ảnh + mask cùng lúc."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    mask_rot = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_rot, mask_rot


def brightness_with_mask(img, mask, factor):
    """Thay đổi brightness, mask giữ nguyên."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), mask.copy()


def contrast_with_mask(img, mask, factor):
    """Thay đổi contrast, mask giữ nguyên."""
    mean = np.mean(img)
    aug = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    return aug, mask.copy()


def flip_with_mask(img, mask):
    """Flip ngang ảnh + mask."""
    return cv2.flip(img, 1), cv2.flip(mask, 1)


def noise_with_mask(img, mask, sigma=20):
    """Thêm noise vào ảnh, mask giữ nguyên."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy, mask.copy()


def blur_with_mask(img, mask, ksize=5):
    """Blur ảnh, mask giữ nguyên."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0), mask.copy()


def mask_to_yolo_polygon(mask, img_h, img_w, min_area=5000):
    """
    Chuyển binary mask → YOLO segmentation polygon format.
    Returns list of polygon strings: "class_id x1 y1 x2 y2 ... xn yn"
    Coordinates normalized to [0, 1].
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Simplify contour để giảm số điểm (YOLO không cần quá chi tiết)
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

        # Normalize coordinates
        points = approx.reshape(-1, 2).astype(float)
        points[:, 0] /= img_w
        points[:, 1] /= img_h

        # Clamp to [0, 1]
        points = np.clip(points, 0.0, 1.0)

        # Format: "0 x1 y1 x2 y2 ..."
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
        polygons.append(f"0 {coords}")

    return polygons


def extract_mask_from_pipeline(img, processor):
    """
    Chạy pipeline CV trên ảnh gốc, chỉ lấy fg_mask.
    Pipeline detect ĐÚNG trên ảnh gốc (chưa augment).
    """
    h, w = img.shape[:2]
    fg_mask, product_contour, conf = processor._step1_remove_background(img, h, w)
    return fg_mask, product_contour, conf


def generate_augmented_pairs(img, mask, stem):
    """Sinh các cặp (ảnh, mask) augmented."""
    pairs = [(f"{stem}", img.copy(), mask.copy())]  # Original

    # Rotations
    for angle in [-10, -5, 5, 10]:
        aug_img, aug_mask = rotate_with_mask(img, mask, angle)
        pairs.append((f"{stem}_rot{angle:+d}", aug_img, aug_mask))

    # Brightness
    for factor, name in [(0.6, "dim"), (0.8, "slightly_dim"), (1.2, "bright"), (1.5, "vbright")]:
        aug_img, aug_mask = brightness_with_mask(img, mask, factor)
        pairs.append((f"{stem}_br_{name}", aug_img, aug_mask))

    # Contrast
    for factor, name in [(0.7, "low_ct"), (1.3, "high_ct")]:
        aug_img, aug_mask = contrast_with_mask(img, mask, factor)
        pairs.append((f"{stem}_ct_{name}", aug_img, aug_mask))

    # Flip
    aug_img, aug_mask = flip_with_mask(img, mask)
    pairs.append((f"{stem}_flipH", aug_img, aug_mask))

    # Noise
    for sigma, name in [(15, "noise_low"), (30, "noise_med")]:
        aug_img, aug_mask = noise_with_mask(img, mask, sigma)
        pairs.append((f"{stem}_noise_{name}", aug_img, aug_mask))

    # Blur
    aug_img, aug_mask = blur_with_mask(img, mask, 7)
    pairs.append((f"{stem}_blur", aug_img, aug_mask))

    # Combos
    aug_img, aug_mask = rotate_with_mask(img, mask, 7)
    aug_img, aug_mask = brightness_with_mask(aug_img, aug_mask, 0.7)
    pairs.append((f"{stem}_combo_rot7_dim", aug_img, aug_mask))

    aug_img, aug_mask = flip_with_mask(img, mask)
    aug_img, aug_mask = contrast_with_mask(aug_img, aug_mask, 1.3)
    pairs.append((f"{stem}_combo_flip_hict", aug_img, aug_mask))

    aug_img, aug_mask = noise_with_mask(img, mask, 20)
    aug_img, aug_mask = blur_with_mask(aug_img, aug_mask, 3)
    pairs.append((f"{stem}_combo_noisy_blur", aug_img, aug_mask))

    return pairs  # ~18 variants per image


def main():
    # Cleanup
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True)

    raw_files = sorted([f for f in RAW_DIR.iterdir() if f.suffix.lower() in EXTS])
    if not raw_files:
        logger.error(f"Không có ảnh trong {RAW_DIR}")
        return

    logger.info(f"Tìm thấy {len(raw_files)} ảnh gốc")

    # Init processor (chỉ dùng step1)
    processor = ImageProcessor(save_vis_steps=False)

    # Chia train/val: giữ 3 ảnh gốc cho val
    random.seed(42)
    val_indices = set(random.sample(range(len(raw_files)), min(3, len(raw_files))))

    all_pairs = []
    val_pairs = []

    for idx, img_path in enumerate(raw_files):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Không đọc được: {img_path.name}")
            continue

        h, w = img.shape[:2]
        fg_mask, contour, conf = extract_mask_from_pipeline(img, processor)

        if fg_mask is None:
            logger.warning(f"Pipeline không detect được: {img_path.name}")
            continue

        # Verify mask chất lượng
        mask_ratio = np.sum(fg_mask > 0) / (h * w)
        if mask_ratio > 0.7 or mask_ratio < 0.02:
            logger.warning(f"Mask nghi ngờ cho {img_path.name}: {mask_ratio:.1%} pixels")
            continue

        logger.info(f"  ✓ {img_path.name} — mask: {mask_ratio:.1%}, conf: {conf:.2f}")

        stem = img_path.stem

        if idx in val_indices:
            # Val: chỉ ảnh gốc, không augment
            val_pairs.append((stem, img, fg_mask))
        else:
            # Train: gốc + augmented
            pairs = generate_augmented_pairs(img, fg_mask, stem)
            all_pairs.extend(pairs)

    # Save train
    for name, aug_img, aug_mask in all_pairs:
        h, w = aug_img.shape[:2]
        polygons = mask_to_yolo_polygon(aug_mask, h, w)

        if not polygons:
            logger.warning(f"Không tạo được polygon cho: {name}")
            continue

        img_path = DATASET_DIR / "images" / "train" / f"{name}.jpg"
        lbl_path = DATASET_DIR / "labels" / "train" / f"{name}.txt"

        cv2.imwrite(str(img_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(lbl_path, "w") as f:
            f.write("\n".join(polygons))

    # Save val
    for name, img, mask in val_pairs:
        h, w = img.shape[:2]
        polygons = mask_to_yolo_polygon(mask, h, w)

        if not polygons:
            continue

        img_path = DATASET_DIR / "images" / "val" / f"{name}.jpg"
        lbl_path = DATASET_DIR / "labels" / "val" / f"{name}.txt"

        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(lbl_path, "w") as f:
            f.write("\n".join(polygons))

    # Create dataset.yaml
    yaml_content = f"""# YOLOv8 Segmentation Dataset
# Auto-generated from pipeline CV detection

path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val

# Class names
names:
  0: product

# Number of classes
nc: 1
"""
    yaml_path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    train_count = len(list((DATASET_DIR / "images" / "train").glob("*.jpg")))
    val_count = len(list((DATASET_DIR / "images" / "val").glob("*.jpg")))

    logger.info("═" * 50)
    logger.info(f"DATASET CREATED")
    logger.info(f"═" * 50)
    logger.info(f"Train: {train_count} ảnh")
    logger.info(f"Val:   {val_count} ảnh")
    logger.info(f"YAML:  {yaml_path}")
    logger.info(f"Path:  {DATASET_DIR}")


if __name__ == "__main__":
    main()
