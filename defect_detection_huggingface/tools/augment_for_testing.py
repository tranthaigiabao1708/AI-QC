"""
tools/augment_for_testing.py
────────────────────────────
Sinh ảnh augmented từ raw_images để stress-test pipeline.
Mỗi ảnh gốc tạo ra nhiều biến thể:
  - Xoay nhẹ (±5°, ±10°, ±15°)
  - Thay đổi độ sáng (tối hơn / sáng hơn)
  - Thay đổi contrast
  - Thêm noise (Gaussian)
  - Blur mờ (mô phỏng camera out-of-focus)
  - Biến dạng phối cảnh nhẹ (perspective warp)
  - Flip ngang
  - Scale (phóng to / thu nhỏ)

Kết quả lưu vào data/augmented_test/
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from loguru import logger

# Setup paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

RAW_DIR = PROJECT_DIR / "data" / "raw_images"
AUG_DIR = PROJECT_DIR / "data" / "augmented_test"
AUG_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def rotate_image(img, angle):
    """Xoay ảnh quanh tâm."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def adjust_brightness(img, factor):
    """Thay đổi độ sáng. factor > 1 = sáng hơn, < 1 = tối hơn."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_contrast(img, factor):
    """Thay đổi contrast. factor > 1 = tăng, < 1 = giảm."""
    mean = np.mean(img)
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def add_gaussian_noise(img, sigma=25):
    """Thêm noise Gaussian."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def apply_blur(img, ksize=7):
    """Mô phỏng camera out-of-focus."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def perspective_warp(img, intensity=0.03):
    """Biến dạng phối cảnh nhẹ."""
    h, w = img.shape[:2]
    d = int(min(h, w) * intensity)

    # Random offsets cho 4 góc
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [np.random.randint(0, d + 1), np.random.randint(0, d + 1)],
        [w - np.random.randint(0, d + 1), np.random.randint(0, d + 1)],
        [w - np.random.randint(0, d + 1), h - np.random.randint(0, d + 1)],
        [np.random.randint(0, d + 1), h - np.random.randint(0, d + 1)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def scale_image(img, scale_factor):
    """Scale ảnh (giữ nguyên kích thước canvas)."""
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if scale_factor > 1.0:
        # Crop center
        cx, cy = new_w // 2, new_h // 2
        x1 = cx - w // 2
        y1 = cy - h // 2
        return resized[y1:y1 + h, x1:x1 + w]
    else:
        # Pad with border replication
        canvas = cv2.copyMakeBorder(
            resized,
            (h - new_h) // 2, h - new_h - (h - new_h) // 2,
            (w - new_w) // 2, w - new_w - (w - new_w) // 2,
            cv2.BORDER_REPLICATE
        )
        return canvas[:h, :w]


def generate_augmentations(img, stem):
    """Sinh tất cả biến thể cho 1 ảnh."""
    augmented = []

    # 1. Xoay nhẹ
    for angle in [-15, -10, -5, 5, 10, 15]:
        aug = rotate_image(img, angle)
        augmented.append((f"{stem}_rot{angle:+d}", aug))

    # 2. Độ sáng
    for factor, name in [(0.5, "dark"), (0.7, "dim"), (1.3, "bright"), (1.6, "vbright")]:
        aug = adjust_brightness(img, factor)
        augmented.append((f"{stem}_br_{name}", aug))

    # 3. Contrast
    for factor, name in [(0.6, "low"), (1.4, "high"), (1.8, "vhigh")]:
        aug = adjust_contrast(img, factor)
        augmented.append((f"{stem}_ct_{name}", aug))

    # 4. Noise
    for sigma, name in [(15, "noise_low"), (35, "noise_med"), (55, "noise_high")]:
        aug = add_gaussian_noise(img, sigma)
        augmented.append((f"{stem}_{name}", aug))

    # 5. Blur
    for ksize, name in [(5, "blur_light"), (11, "blur_med"), (21, "blur_heavy")]:
        aug = apply_blur(img, ksize)
        augmented.append((f"{stem}_{name}", aug))

    # 6. Perspective warp
    for i, intensity in enumerate([0.02, 0.04, 0.06]):
        np.random.seed(42 + i)
        aug = perspective_warp(img, intensity)
        augmented.append((f"{stem}_persp_{i+1}", aug))

    # 7. Flip ngang
    aug = cv2.flip(img, 1)
    augmented.append((f"{stem}_flipH", aug))

    # 8. Scale
    for factor, name in [(0.8, "zoom_out"), (1.2, "zoom_in")]:
        aug = scale_image(img, factor)
        augmented.append((f"{stem}_scale_{name}", aug))

    # 9. Combo: xoay + sáng
    aug = adjust_brightness(rotate_image(img, 8), 0.7)
    augmented.append((f"{stem}_combo_rot8_dim", aug))

    # 10. Combo: noise + blur nhẹ
    aug = apply_blur(add_gaussian_noise(img, 20), 3)
    augmented.append((f"{stem}_combo_noisy_blur", aug))

    # 11. Combo: perspective + contrast cao
    np.random.seed(99)
    aug = adjust_contrast(perspective_warp(img, 0.03), 1.5)
    augmented.append((f"{stem}_combo_persp_hicontrast", aug))

    return augmented


def main():
    raw_files = sorted([f for f in RAW_DIR.iterdir() if f.suffix.lower() in EXTS])

    if not raw_files:
        logger.error(f"Không có ảnh nào trong {RAW_DIR}")
        return

    logger.info(f"Tìm thấy {len(raw_files)} ảnh gốc trong {RAW_DIR}")

    total_generated = 0
    for img_path in raw_files:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Không đọc được: {img_path.name}")
            continue

        stem = img_path.stem
        augmentations = generate_augmentations(img, stem)

        for aug_name, aug_img in augmentations:
            out_path = AUG_DIR / f"{aug_name}.jpg"
            cv2.imwrite(str(out_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            total_generated += 1

        logger.info(f"  {img_path.name} → {len(augmentations)} biến thể")

    logger.info(f"═══ Hoàn tất! Đã sinh {total_generated} ảnh augmented tại: {AUG_DIR} ═══")


if __name__ == "__main__":
    main()
