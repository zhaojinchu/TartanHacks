"""Run live YOLO webcam inference and optionally drive Arduino servos via USB."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import TemporalDecisionEngine
from src.io_utils import CLASS_NAMES, load_decision_config

try:
    import serial  # type: ignore[import-not-found]
    from serial.tools import list_ports  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    serial = None
    list_ports = None


BIN_TO_CHANNEL: dict[str, int] = {
    "bottles": 0,
    "recycle": 0,
    "bottles-cans": 0,
    "compost": 1,
    "compostables": 1,
    "landfill": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO webcam + optional Arduino servo control.")
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--decision_config", type=str, default="configs/decision.yaml")
    parser.add_argument("--window", type=int, default=None, help="Override smoothing window")
    parser.add_argument("--threshold", type=float, default=None, help="Override low-confidence threshold")

    parser.add_argument("--servo_enable", action="store_true", help="Enable Arduino servo control over USB serial.")
    parser.add_argument(
        "--arduino_port",
        type=str,
        default="auto",
        help="Serial port path (example /dev/cu.usbmodem*). Use `auto` to auto-detect.",
    )
    parser.add_argument("--arduino_baud", type=int, default=115200)
    parser.add_argument("--servo_open_seconds", type=float, default=5.0)
    parser.add_argument("--servo_settle_seconds", type=float, default=0.4)
    parser.add_argument("--servo_trigger_interval", type=float, default=1.5)
    parser.add_argument("--servo_min_score", type=float, default=0.60)
    parser.add_argument("--servo_dry_run", action="store_true", help="Log servo commands but do not send serial.")
    return parser.parse_args()


def get_class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def resolve_model_class_names(names: object) -> list[str]:
    if isinstance(names, list):
        return [str(item) for item in names]
    if isinstance(names, dict):
        keys = sorted(int(k) for k in names.keys())
        return [str(names[k]) for k in keys]
    return CLASS_NAMES


def detect_arduino_port() -> str:
    if list_ports is None:
        raise RuntimeError("pyserial is not installed. Install with `pip install pyserial`.")

    candidates: list[str] = []
    for port in list_ports.comports():
        device = str(port.device)
        lower = device.lower()
        if any(tag in lower for tag in ("usbmodem", "usbserial", "ttyacm", "ttyusb")):
            candidates.append(device)

    if not candidates:
        raise RuntimeError("No Arduino serial port detected. Use --arduino_port explicitly.")
    return candidates[0]


class ArduinoServoQueueController:
    """Open/close lids in strict sequence. Never opens two bins at once."""

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        open_seconds: float,
        settle_seconds: float,
        trigger_interval: float,
        min_score: float,
        dry_run: bool = False,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.open_seconds = max(open_seconds, 0.1)
        self.settle_seconds = max(settle_seconds, 0.0)
        self.trigger_interval = max(trigger_interval, 0.0)
        self.min_score = max(min_score, 0.0)
        self.dry_run = dry_run

        self._queue: list[int] = []
        self._pending: set[int] = set()
        self._last_enqueue: dict[int, float] = {}
        self._current: int | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="arduino-servo-queue")
        self._serial = None

    def start(self) -> None:
        if not self.dry_run:
            if serial is None:
                raise RuntimeError("pyserial is not installed. Install with `pip install pyserial`.")
            self._serial = serial.Serial(self.port, self.baudrate, timeout=0.2)
            # Most Arduino boards reset on serial open.
            time.sleep(2.0)
        print(
            "Servo queue enabled: "
            f"port={self.port} baud={self.baudrate} open_seconds={self.open_seconds} "
            f"settle_seconds={self.settle_seconds} trigger_interval={self.trigger_interval} "
            f"min_score={self.min_score} dry_run={self.dry_run}"
        )
        self.close_all()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self.close_all()
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def queue_length(self) -> int:
        with self._lock:
            return len(self._queue) + (1 if self._current is not None else 0)

    def enqueue_from_decision(self, decision: dict[str, object]) -> None:
        reason = str(decision.get("reason", ""))
        if reason != "mapped_from_class":
            return

        score = float(decision.get("score", 0.0))
        if score < self.min_score:
            return

        bin_name = str(decision.get("final_bin", "")).strip().lower()
        channel = BIN_TO_CHANNEL.get(bin_name)
        if channel is None:
            return

        now = time.monotonic()
        with self._lock:
            last = self._last_enqueue.get(channel, 0.0)
            if now - last < self.trigger_interval:
                return
            if channel in self._pending or channel == self._current:
                return

            self._queue.append(channel)
            self._pending.add(channel)
            self._last_enqueue[channel] = now
            print(f"[servo] queued bin={bin_name} channel={channel}")

    def close_all(self) -> None:
        for channel in (0, 1, 2):
            self._send(f"C{channel}")

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            channel: int | None = None
            with self._lock:
                if self._queue:
                    channel = self._queue.pop(0)
                    self._current = channel

            if channel is None:
                time.sleep(0.05)
                continue

            try:
                opened = self._send(f"O{channel}")
                if opened:
                    time.sleep(self.open_seconds)
                self._send(f"C{channel}")
                if self.settle_seconds > 0:
                    time.sleep(self.settle_seconds)
            finally:
                with self._lock:
                    self._pending.discard(channel)
                    if self._current == channel:
                        self._current = None

    def _send(self, command: str) -> bool:
        try:
            if self.dry_run:
                print(f"[servo] dry-run {command}")
                return True
            assert self._serial is not None
            self._serial.write(f"{command}\n".encode("utf-8"))
            self._serial.flush()
            print(f"[servo] sent {command}")
            return True
        except Exception as exc:
            print(f"[servo] send failed {command}: {exc}")
            return False


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    class_to_bin, cfg_threshold, cfg_window = load_decision_config(args.decision_config)
    threshold = args.threshold if args.threshold is not None else cfg_threshold
    window_size = args.window if args.window is not None else cfg_window

    model = YOLO(str(weights))
    model_class_names = resolve_model_class_names(model.names)
    engine = TemporalDecisionEngine(
        class_to_bin=class_to_bin,
        threshold=threshold,
        window_size=window_size,
        class_names=model_class_names,
    )

    servo_controller: ArduinoServoQueueController | None = None
    if args.servo_enable:
        port = args.arduino_port if args.arduino_port != "auto" else detect_arduino_port()
        servo_controller = ArduinoServoQueueController(
            port=port,
            baudrate=args.arduino_baud,
            open_seconds=args.servo_open_seconds,
            settle_seconds=args.servo_settle_seconds,
            trigger_interval=args.servo_trigger_interval,
            min_score=args.servo_min_score,
            dry_run=args.servo_dry_run,
        )
        servo_controller.start()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam index {args.camera_index}")

    fps_ema = 0.0
    alpha = 0.2
    prev_t = time.perf_counter()

    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: webcam frame read failed")
                break

            result = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)[0]
            boxes = result.boxes

            detections: list[tuple[int, float]] = []
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    detections.append((cls_id, conf))

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = f"{get_class_name(model.names, cls_id)} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )

            decision = engine.update(detections)
            if servo_controller is not None:
                servo_controller.enqueue_from_decision(decision)

            now = time.perf_counter()
            dt = max(now - prev_t, 1e-6)
            prev_t = now

            fps = 1.0 / dt
            fps_ema = fps if fps_ema == 0.0 else (1 - alpha) * fps_ema + alpha * fps

            overlay = [
                f"final_bin: {decision['final_bin']}",
                f"top_class: {decision['top_class']} ({decision['score']:.2f})",
                f"reason: {decision['reason']}",
                f"fps: {fps_ema:.1f}",
            ]
            if servo_controller is not None:
                overlay.append(f"servo_queue: {servo_controller.queue_length()}")

            for i, text in enumerate(overlay):
                y = 28 + i * 24
                cv2.putText(
                    frame,
                    text,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (20, 230, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("waste_sorter_webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if servo_controller is not None:
            servo_controller.stop()


if __name__ == "__main__":
    main()
