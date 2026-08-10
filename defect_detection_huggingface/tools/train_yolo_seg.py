"""
tools/train_yolo_seg.py
───────────────────────
Phase 2: Train YOLOv8n-seg trên dataset đã auto-label.

Usage:
    python tools/train_yolo_seg.py [--epochs 100] [--imgsz 640] [--batch 4]
"""

import sys
import argparse
from pathlib import Path

from loguru import logger

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "yolo_dataset"
OUTPUT_DIR = PROJECT_DIR / "output" / "yolo_seg_model"


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n-seg")
    parser.add_argument("--epochs", type=int, default=100, help="Số epoch training")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh training")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (nhỏ cho CPU)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu hoặc 0 (GPU)")
    args = parser.parse_args()

    yaml_path = DATASET_DIR / "dataset.yaml"
    if not yaml_path.exists():
        logger.error(f"Dataset chưa tồn tại: {yaml_path}")
        logger.info("Chạy: python tools/auto_label_yolo.py trước!")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics chưa cài. Chạy: pip install ultralytics")
        return

    logger.info("═" * 50)
    logger.info("BẮT ĐẦU TRAINING YOLOv8n-seg")
    logger.info("═" * 50)
    logger.info(f"Dataset:  {yaml_path}")
    logger.info(f"Epochs:   {args.epochs}")
    logger.info(f"ImgSize:  {args.imgsz}")
    logger.info(f"Batch:    {args.batch}")
    logger.info(f"Device:   {args.device}")
    logger.info(f"Output:   {OUTPUT_DIR}")

    # Load pretrained YOLOv8n-seg
    model = YOLO("yolov8n-seg.pt")

    # Train
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        patience=20,        # Early stopping
        save=True,
        save_period=25,      # Save checkpoint mỗi 25 epochs
        plots=True,
        verbose=True,
        # Augmentation tích hợp sẵn của YOLO
        hsv_h=0.015,         # Hue augmentation
        hsv_s=0.5,           # Saturation augmentation
        hsv_v=0.3,           # Value augmentation
        degrees=10,          # Rotation ±10°
        translate=0.1,
        scale=0.3,
        fliplr=0.5,          # Flip ngang 50%
        flipud=0.0,
        mosaic=0.5,          # Mosaic augmentation (giảm vì dataset nhỏ)
        mixup=0.1,
    )

    # Export best model
    best_model_path = OUTPUT_DIR / "train" / "weights" / "best.pt"
    if best_model_path.exists():
        # Copy to output root for easy access
        final_path = PROJECT_DIR / "output" / "yolo_seg_best.pt"
        import shutil
        shutil.copy2(best_model_path, final_path)
        logger.info(f"\n✅ Model tốt nhất đã lưu tại: {final_path}")

        # Cũng export sang ONNX cho inference nhanh
        try:
            best_model = YOLO(str(best_model_path))
            best_model.export(format="onnx", imgsz=args.imgsz)
            logger.info(f"✅ ONNX model đã export")
        except Exception as e:
            logger.warning(f"Không export được ONNX: {e}")
    else:
        logger.error(f"Không tìm thấy best model tại: {best_model_path}")

    logger.info("\n═══ TRAINING HOÀN TẤT ═══")


if __name__ == "__main__":
    main()
