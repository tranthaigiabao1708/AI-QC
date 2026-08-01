"""
Tạo ảnh reference templates từ ảnh raw hiện có.
Chạy pipeline tách nền → lưu ảnh sản phẩm đã tách nền vào data/reference_templates/
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import config
from preprocessing.image_processor import ImageProcessor

def main():
    raw_dir = config.RAW_DATA_DIR
    ref_dir = Path(__file__).resolve().parent / "data" / "reference_templates"
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    processor = ImageProcessor(save_vis_steps=True)
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0
    
    for img_path in sorted(raw_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        result = processor.process(img)
        if result is None:
            print(f"SKIP: {img_path.name} — pipeline thất bại")
            continue
        
        # Lay anh da tach nen (buoc 1)
        if "01_bg_removed" in result.vis_steps:
            bg_removed = result.vis_steps["01_bg_removed"]
        else:
            bg_removed = cv2.bitwise_and(img, img, mask=result.vis_steps.get("mask", None))
        
        # Lay anh da xoay thang (buoc 3) — tot hon cho matching
        if "03_rotated" in result.vis_steps:
            rotated = result.vis_steps["03_rotated"]
            ref_name = f"ref_{img_path.stem}_rotated.png"
            cv2.imwrite(str(ref_dir / ref_name), rotated)
            print(f"OK: {ref_name} (rotated)")
            count += 1
        
        # Cung luu anh tach nen goc
        ref_name_bg = f"ref_{img_path.stem}_bg_removed.png"
        cv2.imwrite(str(ref_dir / ref_name_bg), bg_removed)
        print(f"OK: {ref_name_bg} (bg_removed)")
        count += 1
    
    print(f"\nCreated {count} reference images at: {ref_dir}")

if __name__ == "__main__":
    main()
