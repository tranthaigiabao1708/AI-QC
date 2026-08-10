"""
tools/test_pipeline_batch.py
────────────────────────────
Chạy pipeline QC trên tất cả ảnh augmented và sinh báo cáo kết quả.
Lưu ảnh prediction + CSV summary.
"""

import sys
import csv
import time
from pathlib import Path

import cv2
from loguru import logger

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import config
from inference import QualityInspector

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    aug_dir = PROJECT_DIR / "data" / "augmented_test"
    pred_dir = config.OUTPUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if not aug_dir.exists():
        logger.error(f"Thư mục augmented chưa tồn tại: {aug_dir}")
        logger.info("Chạy: python tools/augment_for_testing.py trước.")
        return

    aug_files = sorted([f for f in aug_dir.iterdir() if f.suffix.lower() in EXTS])
    if not aug_files:
        logger.error(f"Không có ảnh nào trong {aug_dir}")
        return

    logger.info(f"Tìm thấy {len(aug_files)} ảnh augmented. Bắt đầu chạy pipeline...")

    inspector = QualityInspector()

    # CSV report
    csv_path = config.OUTPUT_DIR / "pipeline_test_report.csv"
    results = []
    errors = []
    t_start = time.time()

    for i, img_path in enumerate(aug_files):
        img = cv2.imread(str(img_path))
        if img is None:
            errors.append({"file": img_path.name, "error": "Không đọc được file"})
            continue

        result = inspector.inspect(img)

        if result["success"]:
            row = {
                "file": img_path.name,
                "decision": result["decision"],
                "confidence": f"{result['confidence']:.1%}",
                "copper_ratio": f"{result['copper_ratio']:.2%}",
                "pipeline_conf": f"{result['pipeline_confidence']:.0%}",
                "status": "OK"
            }
            results.append(row)

            # Lưu prediction image
            out_path = pred_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(out_path), result["vis_img"])
        else:
            row = {
                "file": img_path.name,
                "decision": "FAILED",
                "confidence": "0%",
                "copper_ratio": "0%",
                "pipeline_conf": "0%",
                "status": f"FAIL: {result['error']}"
            }
            results.append(row)
            errors.append({"file": img_path.name, "error": result["error"]})

        # Progress log mỗi 50 ảnh
        if (i + 1) % 50 == 0 or (i + 1) == len(aug_files):
            elapsed = time.time() - t_start
            logger.info(f"  [{i + 1}/{len(aug_files)}] — {elapsed:.1f}s — Lỗi: {len(errors)}")

    elapsed = time.time() - t_start

    # Ghi CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "decision", "confidence", "copper_ratio", "pipeline_conf", "status"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    total = len(results)
    success = sum(1 for r in results if r["status"] == "OK")
    failed = total - success

    logger.info("═" * 60)
    logger.info(f"PIPELINE TEST REPORT")
    logger.info(f"═" * 60)
    logger.info(f"Tổng ảnh test:    {total}")
    logger.info(f"Thành công:       {success} ({success/total*100:.1f}%)")
    logger.info(f"Thất bại:         {failed} ({failed/total*100:.1f}%)")
    logger.info(f"Thời gian:        {elapsed:.1f}s ({elapsed/total:.2f}s/ảnh)")
    logger.info(f"CSV report:       {csv_path}")
    logger.info(f"Predictions:      {pred_dir}")

    if errors:
        logger.warning(f"\n--- {len(errors)} ẢNH LỖI ---")
        for e in errors:
            logger.warning(f"  {e['file']}: {e['error']}")


if __name__ == "__main__":
    main()
