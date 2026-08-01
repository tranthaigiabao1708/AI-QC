"""
live_inspector.py
─────────────────
Module nhận diện live từ camera USB/webcam ở 20fps.
Kiến trúc multi-thread: Camera Thread → Processing Thread → Display Thread.

Sử dụng:
    py live_inspector.py --camera 0 --fps 20 --mode continuous
    py live_inspector.py --camera 0 --fps 20 --mode auto-capture --save-dir ./output/captures
    py live_inspector.py --camera 0 --mode manual-trigger

Phím tắt:
    q     — Thoát
    s     — Chụp screenshot
    Space — Manual trigger (chế độ manual-trigger)
    m     — Chuyển đổi chế độ hoạt động
"""

import os
import sys
import csv
import time
import argparse
import threading
from pathlib import Path
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from loguru import logger

# Thêm đường dẫn gốc vào python path
sys.path.append(str(Path(__file__).resolve().parent))
import config
from inference import QualityInspector


# ═══════════════════════════════════════════════════════════════
# LATEST FRAME BUFFER — Thread-safe, chỉ giữ frame mới nhất
# ═══════════════════════════════════════════════════════════════
class LatestFrameBuffer:
    """Buffer thread-safe chỉ lưu frame mới nhất, tự drop frame cũ."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()

    def put(self, frame):
        """Đẩy frame mới vào buffer (ghi đè frame cũ)."""
        with self._lock:
            self._frame = frame
        self._new_frame_event.set()

    def get(self, timeout=1.0):
        """Lấy frame mới nhất. Chờ tối đa timeout giây."""
        if self._new_frame_event.wait(timeout=timeout):
            with self._lock:
                frame = self._frame
            self._new_frame_event.clear()
            return frame
        return None

    def peek(self):
        """Xem frame hiện tại mà không chờ."""
        with self._lock:
            return self._frame


# ═══════════════════════════════════════════════════════════════
# CAMERA THREAD — Đọc frame liên tục từ camera
# ═══════════════════════════════════════════════════════════════
class CameraThread(threading.Thread):
    """Thread đọc frame liên tục từ camera và đẩy vào buffer."""

    def __init__(self, camera_id, resolution, target_fps, frame_buffer):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.resolution = resolution
        self.target_fps = target_fps
        self.frame_buffer = frame_buffer
        self._stop_event = threading.Event()

    def run(self):
        """Vòng lặp đọc camera."""
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_V4L2)

        if not cap.isOpened():
            # Fallback: thử mở không chỉ định backend
            cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            logger.error(f"Không thể mở camera ID={self.camera_id}")
            return

        # Set resolution và FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera mở thành công: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

        frame_interval = 1.0 / self.target_fps

        while not self._stop_event.is_set():
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                logger.warning("Không đọc được frame từ camera")
                time.sleep(0.01)
                continue

            self.frame_buffer.put(frame)

            # Điều chỉnh thời gian chờ để đạt target FPS
            elapsed = time.perf_counter() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        logger.info("Camera thread đã dừng.")

    def stop(self):
        """Dừng camera thread."""
        self._stop_event.set()


# ═══════════════════════════════════════════════════════════════
# TEMPORAL SMOOTHER — Ổn định kết quả giữa các frame
# ═══════════════════════════════════════════════════════════════
class TemporalSmoother:
    """
    Ổn định kết quả QC giữa các frame liên tiếp.
    Dùng moving average + hysteresis threshold.
    """

    def __init__(self, window_size=7, ok_threshold=0.85, ng_threshold=0.80, min_frames=5):
        self.window_size = window_size
        self.ok_threshold = ok_threshold
        self.ng_threshold = ng_threshold
        self.min_frames = min_frames
        self.history = deque(maxlen=window_size)
        self.current_state = "UNKNOWN"  # "OK", "NG", hoặc "UNKNOWN"
        self._consecutive_ok = 0
        self._consecutive_ng = 0

    def update(self, confidence_ok, confidence_ng, raw_decision):
        """
        Cập nhật kết quả mới và trả về quyết định ổn định.

        Trả về:
            tuple: (smoothed_decision, smoothed_confidence)
        """
        self.history.append({
            "ok": confidence_ok,
            "ng": confidence_ng,
            "raw": raw_decision
        })

        if len(self.history) < 2:
            self.current_state = raw_decision
            conf = confidence_ok if raw_decision == "OK" else confidence_ng
            return raw_decision, conf

        # Tính trung bình confidence qua cửa sổ
        avg_ok = np.mean([h["ok"] for h in self.history])
        avg_ng = np.mean([h["ng"] for h in self.history])

        # Hysteresis logic
        if raw_decision == "OK":
            self._consecutive_ok += 1
            self._consecutive_ng = 0
        else:
            self._consecutive_ng += 1
            self._consecutive_ok = 0

        # Chỉ chuyển trạng thái khi đủ frame liên tục vượt ngưỡng
        if self.current_state != "OK" and self._consecutive_ok >= self.min_frames and avg_ok > self.ok_threshold:
            self.current_state = "OK"
        elif self.current_state != "NG" and self._consecutive_ng >= self.min_frames and avg_ng > self.ng_threshold:
            self.current_state = "NG"

        smoothed_conf = avg_ok if self.current_state == "OK" else avg_ng
        return self.current_state, smoothed_conf

    def reset(self):
        """Reset khi phát hiện sản phẩm mới."""
        self.history.clear()
        self.current_state = "UNKNOWN"
        self._consecutive_ok = 0
        self._consecutive_ng = 0


# ═══════════════════════════════════════════════════════════════
# PRODUCT TRACKER — Theo dõi sản phẩm giữa các frame
# ═══════════════════════════════════════════════════════════════
class ProductTracker:
    """Theo dõi centroid sản phẩm để phát hiện sản phẩm mới/biến mất."""

    def __init__(self, distance_threshold=100, stable_delay=0.5):
        self.distance_threshold = distance_threshold
        self.stable_delay = stable_delay
        self.last_centroid = None
        self.product_appeared_at = None
        self.is_stable = False
        self.no_product_count = 0

    def update(self, centroid):
        """
        Cập nhật vị trí sản phẩm.

        Trả về:
            str: "NEW" nếu sản phẩm mới, "STABLE" nếu ổn định, "GONE" nếu mất, "TRACKING" bình thường
        """
        if centroid is None:
            self.no_product_count += 1
            if self.no_product_count > 10:
                self.last_centroid = None
                self.product_appeared_at = None
                self.is_stable = False
                return "GONE"
            return "GONE"

        self.no_product_count = 0

        if self.last_centroid is None:
            # Sản phẩm xuất hiện lần đầu
            self.last_centroid = centroid
            self.product_appeared_at = time.perf_counter()
            self.is_stable = False
            return "NEW"

        # Tính khoảng cách di chuyển
        dist = np.sqrt((centroid[0] - self.last_centroid[0]) ** 2 +
                       (centroid[1] - self.last_centroid[1]) ** 2)

        if dist > self.distance_threshold:
            # Sản phẩm mới (centroid nhảy xa)
            self.last_centroid = centroid
            self.product_appeared_at = time.perf_counter()
            self.is_stable = False
            return "NEW"

        # Cập nhật centroid
        self.last_centroid = centroid

        # Kiểm tra ổn định
        if not self.is_stable and self.product_appeared_at is not None:
            elapsed = time.perf_counter() - self.product_appeared_at
            if elapsed >= self.stable_delay:
                self.is_stable = True
                return "STABLE"

        return "TRACKING"


# ═══════════════════════════════════════════════════════════════
# OSD RENDERER — Vẽ overlay lên frame hiển thị
# ═══════════════════════════════════════════════════════════════
class OSDRenderer:
    """Vẽ On-Screen Display overlay lên frame video live."""

    @staticmethod
    def render(frame, result, fps, mode, stats, smoother_decision=None, smoother_conf=None):
        """
        Vẽ đầy đủ OSD lên frame.

        Args:
            frame: Frame BGR gốc
            result: Dict kết quả từ QualityInspector.inspect() hoặc None
            fps: FPS hiện tại
            mode: Chế độ hoạt động ("continuous", "auto-capture", "manual-trigger")
            stats: Dict thống kê {"total", "ok", "ng"}
            smoother_decision: Quyết định sau temporal smoothing
            smoother_conf: Confidence sau smoothing
        """
        display = frame.copy()
        h, w = display.shape[:2]

        # === FPS counter — góc trên phải ===
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(display, fps_text, (w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # === Mode indicator — góc trên phải, dưới FPS ===
        mode_text = f"Mode: {mode}"
        cv2.putText(display, mode_text, (w - 250, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        if result is not None and result.get("success"):
            decision = smoother_decision or result["decision"]
            confidence = smoother_conf or result["confidence"]
            is_ok = "OK" in str(decision) and "NG" not in str(decision)

            color = (0, 255, 0) if is_ok else (0, 0, 255)
            bg_color = (0, 80, 0) if is_ok else (0, 0, 80)

            # === Bounding box sản phẩm ===
            if "vis_img" in result:
                # Vẽ oriented bounding box từ outer_rect
                pass  # Bounding box đã được vẽ trong vis_img

            # === QC Result — góc trên trái, nền bán trong suốt ===
            result_text = f"QC: {'OK' if is_ok else 'NG'} ({confidence:.1%})"
            # Vẽ nền bán trong suốt
            text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            overlay_rect = display.copy()
            cv2.rectangle(overlay_rect, (10, 5), (20 + text_size[0], 45 + text_size[1]), bg_color, -1)
            cv2.addWeighted(overlay_rect, 0.6, display, 0.4, 0, display)
            cv2.putText(display, result_text, (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

            # === Confidence bar ===
            bar_x, bar_y = 15, 55
            bar_w, bar_h = 200, 15
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            fill_w = int(bar_w * confidence)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

            # === Copper info — góc dưới trái ===
            copper_ratio = result.get("copper_ratio", 0)
            copper_w = result.get("copper_w", 0)
            copper_h = result.get("copper_h", 0)
            pipe_conf = result.get("pipeline_confidence", 0)
            info_text = f"Copper: {copper_ratio:.1%} | Size: {copper_w}x{copper_h}px | Pipeline: {pipe_conf:.0%}"
            cv2.putText(display, info_text, (15, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        else:
            # Không có sản phẩm
            cv2.putText(display, "Waiting for product...", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2, cv2.LINE_AA)

        # === Statistics counter — góc dưới phải ===
        total = stats.get("total", 0)
        ok = stats.get("ok", 0)
        ng = stats.get("ng", 0)
        rate = (ng / total * 100) if total > 0 else 0
        stats_text = f"Total: {total} | OK: {ok} | NG: {ng} ({rate:.1f}%)"
        text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(display, stats_text, (w - text_size[0] - 15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return display


# ═══════════════════════════════════════════════════════════════
# LIVE INSPECTOR — Module chính điều phối live detection
# ═══════════════════════════════════════════════════════════════
class LiveInspector:
    """
    Hệ thống nhận diện live từ camera ở 20fps.
    Kiến trúc: Camera Thread → Processing → Display.
    """

    MODES = ["continuous", "auto-capture", "manual-trigger"]

    def __init__(self, camera_id=None, target_fps=None, resolution=None,
                 mode="continuous", save_dir=None):
        self.camera_id = camera_id or config.CAMERA_ID
        self.target_fps = target_fps or config.TARGET_FPS
        self.resolution = resolution or config.LIVE_RESOLUTION
        self.mode = mode
        self.save_dir = Path(save_dir) if save_dir else config.NG_CAPTURES_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Khởi tạo components
        logger.info("Đang khởi tạo QualityInspector cho live mode...")
        self.inspector = QualityInspector()

        self.frame_buffer = LatestFrameBuffer()
        self.smoother = TemporalSmoother(
            window_size=config.SMOOTHING_WINDOW,
            ok_threshold=config.HYSTERESIS_OK_THRESHOLD,
            ng_threshold=config.HYSTERESIS_NG_THRESHOLD,
            min_frames=config.HYSTERESIS_MIN_FRAMES,
        )
        self.tracker = ProductTracker(
            distance_threshold=config.NEW_PRODUCT_DISTANCE,
            stable_delay=config.PRODUCT_STABLE_DELAY,
        )
        self.osd = OSDRenderer()

        # Thống kê
        self.stats = {"total": 0, "ok": 0, "ng": 0}

        # Frame counter cho skip logic
        self.frame_count = 0
        self.process_every_n = config.PROCESS_EVERY_N_FRAMES
        self.last_result = None

        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.perf_counter()

        # CSV log
        self.log_path = config.LIVE_LOG_PATH
        self._init_csv_log()

        # Manual trigger flag
        self._manual_trigger = False

        logger.info(f"LiveInspector sẵn sàng: camera={self.camera_id}, fps={self.target_fps}, mode={self.mode}")

    def _init_csv_log(self):
        """Khởi tạo file CSV log nếu chưa có."""
        if not self.log_path.exists():
            with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "decision", "confidence", "copper_ratio",
                                 "copper_w", "copper_h", "pipeline_confidence"])

    def _log_result(self, result):
        """Ghi kết quả vào CSV log."""
        try:
            with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    result.get("decision", ""),
                    f"{result.get('confidence', 0):.4f}",
                    f"{result.get('copper_ratio', 0):.4f}",
                    result.get("copper_w", 0),
                    result.get("copper_h", 0),
                    f"{result.get('pipeline_confidence', 0):.4f}",
                ])
        except Exception as e:
            logger.warning(f"Lỗi ghi CSV log: {e}")

    def _save_ng_capture(self, frame, result):
        """Tự động lưu ảnh sản phẩm NG."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"NG_{ts}.jpg"
            filepath = self.save_dir / filename
            cv2.imwrite(str(filepath), frame)
            logger.info(f"Đã lưu ảnh NG: {filepath}")
        except Exception as e:
            logger.warning(f"Lỗi lưu ảnh NG: {e}")

    def _update_stats(self, decision):
        """Cập nhật thống kê."""
        self.stats["total"] += 1
        if "OK" in str(decision) and "NG" not in str(decision):
            self.stats["ok"] += 1
        else:
            self.stats["ng"] += 1

    def run(self):
        """
        Vòng lặp chính: đọc camera → xử lý → hiển thị.
        Nhấn 'q' để thoát.
        """
        # Khởi động Camera Thread
        cam_thread = CameraThread(self.camera_id, self.resolution, self.target_fps, self.frame_buffer)
        cam_thread.start()

        logger.info("Bắt đầu live inspection. Nhấn 'q' để thoát, 's' screenshot, 'm' đổi mode, Space trigger.")

        window_name = "QC Live Inspector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                # Lấy frame mới nhất
                frame = self.frame_buffer.get(timeout=2.0)
                if frame is None:
                    continue

                self.frame_count += 1

                # Tính FPS
                now = time.perf_counter()
                dt = now - self.last_frame_time
                self.last_frame_time = now
                if dt > 0:
                    self.fps_history.append(1.0 / dt)
                current_fps = np.mean(self.fps_history) if self.fps_history else 0.0

                # === Xử lý theo chế độ ===
                result = None
                smoother_decision = None
                smoother_conf = None

                if self.mode == "continuous":
                    # Chạy full pipeline mỗi N frames, frame giữa dùng cache
                    if self.frame_count % self.process_every_n == 0 or self.last_result is None:
                        result = self.inspector.inspect(frame)
                        self.last_result = result
                    else:
                        # Dùng kết quả cache + chạy detect nhanh để cập nhật centroid
                        fast = self.inspector.inspect_fast(frame)
                        result = self.last_result
                        if result and fast["has_product"]:
                            result = dict(result)  # Copy để không thay đổi cache
                            result["centroid"] = fast["centroid"]

                elif self.mode == "auto-capture":
                    # Chạy detect nhanh mọi frame, full pipeline khi sản phẩm ổn định
                    fast = self.inspector.inspect_fast(frame)
                    centroid = fast["centroid"] if fast["has_product"] else None
                    tracking_state = self.tracker.update(centroid)

                    if tracking_state == "NEW":
                        self.smoother.reset()
                        self.last_result = None

                    if tracking_state == "STABLE" or (tracking_state == "TRACKING" and self.tracker.is_stable):
                        if self.frame_count % self.process_every_n == 0 or self.last_result is None:
                            result = self.inspector.inspect(frame)
                            self.last_result = result

                            # Auto-capture khi ổn định lần đầu
                            if tracking_state == "STABLE" and result and result.get("success"):
                                self._log_result(result)
                                self._update_stats(result["decision"])
                                if "NG" in str(result["decision"]):
                                    self._save_ng_capture(frame, result)
                        else:
                            result = self.last_result
                    else:
                        result = self.last_result

                elif self.mode == "manual-trigger":
                    # Luôn chạy detect nhanh cho preview, full pipeline chỉ khi trigger
                    if self._manual_trigger:
                        result = self.inspector.inspect(frame)
                        self.last_result = result
                        self._manual_trigger = False

                        if result and result.get("success"):
                            self._log_result(result)
                            self._update_stats(result["decision"])
                            if "NG" in str(result["decision"]):
                                self._save_ng_capture(frame, result)
                            logger.info(f"Manual trigger: {result['decision']} ({result['confidence']:.1%})")
                    else:
                        result = self.last_result

                # Temporal smoothing (chỉ cho continuous mode)
                if self.mode == "continuous" and result and result.get("success"):
                    raw = result.get("raw_decision", "NG")
                    conf = result.get("confidence", 0.5)
                    ok_conf = conf if raw == "OK" else (1.0 - conf)
                    ng_conf = conf if raw == "NG" else (1.0 - conf)
                    smoother_decision, smoother_conf = self.smoother.update(ok_conf, ng_conf, raw)

                    # Cập nhật tracking
                    centroid = result.get("centroid")
                    tracking_state = self.tracker.update(centroid)
                    if tracking_state == "NEW":
                        self.smoother.reset()

                # === Render OSD ===
                display = self.osd.render(
                    frame, result, current_fps, self.mode,
                    self.stats, smoother_decision, smoother_conf
                )

                cv2.imshow(window_name, display)

                # === Xử lý phím tắt ===
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    logger.info("Thoát live inspection.")
                    break

                elif key == ord('s'):
                    # Screenshot
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = self.save_dir / f"screenshot_{ts}.jpg"
                    cv2.imwrite(str(screenshot_path), display)
                    logger.info(f"Screenshot đã lưu: {screenshot_path}")

                elif key == ord(' '):
                    # Manual trigger
                    self._manual_trigger = True

                elif key == ord('m'):
                    # Chuyển đổi chế độ
                    current_idx = self.MODES.index(self.mode)
                    self.mode = self.MODES[(current_idx + 1) % len(self.MODES)]
                    logger.info(f"Chuyển sang chế độ: {self.mode}")
                    self.smoother.reset()
                    self.last_result = None

        except KeyboardInterrupt:
            logger.info("Nhận tín hiệu dừng (Ctrl+C)")
        finally:
            cam_thread.stop()
            cam_thread.join(timeout=3.0)
            cv2.destroyAllWindows()

            # In thống kê cuối cùng
            logger.info("=== THỐNG KÊ PHIÊN KIỂM TRA ===")
            logger.info(f"Tổng sản phẩm: {self.stats['total']}")
            logger.info(f"OK: {self.stats['ok']}")
            logger.info(f"NG: {self.stats['ng']}")
            if self.stats['total'] > 0:
                rate = self.stats['ng'] / self.stats['total'] * 100
                logger.info(f"Tỷ lệ lỗi: {rate:.1f}%")


def parse_resolution(s):
    """Parse resolution string 'WIDTHxHEIGHT' → tuple (width, height)."""
    try:
        parts = s.lower().split('x')
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(f"Resolution phải có dạng 'WxH', ví dụ '640x480'. Nhận: {s}")


def main():
    parser = argparse.ArgumentParser(
        description="Nhận diện live chất lượng sản phẩm cos đồng từ camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  py live_inspector.py --camera 0 --fps 20 --mode continuous
  py live_inspector.py --camera 0 --mode auto-capture --save-dir ./output/captures
  py live_inspector.py --camera 1 --resolution 800x600 --mode manual-trigger

Phím tắt khi chạy:
  q     — Thoát
  s     — Chụp screenshot
  Space — Manual trigger (chế độ manual-trigger)
  m     — Chuyển đổi chế độ hoạt động
        """
    )
    parser.add_argument("--camera", type=int, default=None,
                        help=f"ID camera (mặc định: {config.CAMERA_ID})")
    parser.add_argument("--fps", type=int, default=None,
                        help=f"FPS mục tiêu (mặc định: {config.TARGET_FPS})")
    parser.add_argument("--resolution", type=parse_resolution, default=None,
                        help=f"Resolution WxH (mặc định: {config.LIVE_RESOLUTION[0]}x{config.LIVE_RESOLUTION[1]})")
    parser.add_argument("--mode", type=str, choices=LiveInspector.MODES, default="continuous",
                        help="Chế độ hoạt động (mặc định: continuous)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Thư mục lưu ảnh NG/screenshots")

    args = parser.parse_args()

    live = LiveInspector(
        camera_id=args.camera,
        target_fps=args.fps,
        resolution=args.resolution,
        mode=args.mode,
        save_dir=args.save_dir,
    )
    live.run()


if __name__ == "__main__":
    main()
