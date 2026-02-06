"""Real-time Raspberry Pi 5 inference using ONNX Runtime + OpenCV."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import TemporalDecisionEngine
from src.io_utils import CLASS_NAMES, load_decision_config
from src.onnx_detector import ONNXWasteDetector


class OpenCVCamera:
    """OpenCV-based camera source (USB webcams or V4L2 devices)."""

    def __init__(self, camera_index: int, width: int, height: int, fps: float) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {camera_index} with OpenCV")

        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))

    def read(self) -> tuple[bool, Any]:
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()

    def output_meta(self, fallback_width: int, fallback_height: int, fallback_fps: float) -> tuple[int, int, float]:
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or fallback_width)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or fallback_height)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or fallback_fps)
        if fps <= 0:
            fps = fallback_fps
        return width, height, fps


class PiCamera2Source:
    """Picamera2-based source for Raspberry Pi Camera Module 2/3 via libcamera."""

    def __init__(self, camera_index: int, width: int, height: int, fps: float) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 is not installed. Install on Raspberry Pi OS with:\n"
                "  sudo apt update && sudo apt install -y python3-picamera2\n"
                "If using venv, recreate with --system-site-packages."
            ) from exc

        self.width = width
        self.height = height
        self.fps = fps if fps > 0 else 30.0

        self.picam2 = Picamera2(camera_index)
        controls: dict[str, Any] = {}
        if self.fps > 0:
            controls["FrameRate"] = float(self.fps)

        config = self.picam2.create_preview_configuration(
            main={"format": "BGR888", "size": (self.width, self.height)},
            controls=controls,
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.2)

    def read(self) -> tuple[bool, Any]:
        frame = self.picam2.capture_array()
        if frame is None:
            return False, None

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return True, frame

    def release(self) -> None:
        self.picam2.stop()
        self.picam2.close()

    def output_meta(self, fallback_width: int, fallback_height: int, fallback_fps: float) -> tuple[int, int, float]:
        return (
            int(self.width or fallback_width),
            int(self.height or fallback_height),
            float(self.fps or fallback_fps),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-time ONNX inference on Raspberry Pi 5 webcam."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/waste_sorter.onnx"),
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--decision_config",
        type=Path,
        default=Path("configs/decision.yaml"),
        help="Path to decision YAML",
    )
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument(
        "--camera_backend",
        type=str,
        default="auto",
        choices=["auto", "picamera2", "opencv"],
        help="Camera backend. Use `picamera2` for Raspberry Pi Camera Module 2.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera_fps", type=float, default=30.0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--window", type=int, default=None, help="Override smoothing window")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override unknown threshold",
    )
    parser.add_argument("--save_path", type=Path, default=None)
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Disable GUI window (headless mode)",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=30,
        help="Print status every N frames",
    )
    return parser.parse_args()


def open_camera_source(args: argparse.Namespace) -> tuple[Any, str]:
    """Open camera with requested backend and return (source, backend_name)."""
    if args.camera_backend in ("picamera2", "auto"):
        try:
            source = PiCamera2Source(
                camera_index=args.camera_index,
                width=args.width,
                height=args.height,
                fps=args.camera_fps,
            )
            return source, "picamera2"
        except Exception as exc:
            if args.camera_backend == "picamera2":
                raise RuntimeError(f"Failed to open Pi camera with picamera2: {exc}") from exc
            print(f"Warning: picamera2 backend unavailable, falling back to OpenCV. ({exc})")

    try:
        source = OpenCVCamera(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
        )
        return source, "opencv"
    except Exception as exc:
        raise RuntimeError(
            "Failed to open camera with both picamera2 and OpenCV backends.\n"
            "For Raspberry Pi Camera Module 2, install picamera2 and use --camera_backend picamera2."
        ) from exc


def draw_overlay(
    frame,
    detections,
    decision: dict[str, object],
    fps_ema: float,
) -> None:
    """Draw detections, routing decision, and FPS on frame."""
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        class_name = CLASS_NAMES[det.class_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame,
            f"{class_name} {det.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    lines = [
        f"final_bin: {decision['final_bin']}",
        f"top_class: {decision['top_class']} ({decision['score']:.2f})",
        f"reason: {decision['reason']}",
        f"fps: {fps_ema:.1f}",
    ]

    for i, text in enumerate(lines):
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


def main() -> None:
    args = parse_args()

    class_to_bin, cfg_threshold, cfg_window = load_decision_config(args.decision_config)
    threshold = args.threshold if args.threshold is not None else cfg_threshold
    window_size = args.window if args.window is not None else cfg_window

    detector = ONNXWasteDetector(
        model_path=args.model,
        class_names=CLASS_NAMES,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    decision_engine = TemporalDecisionEngine(
        class_to_bin=class_to_bin,
        threshold=threshold,
        window_size=window_size,
        class_names=CLASS_NAMES,
    )

    camera, backend_name = open_camera_source(args)
    print(f"Camera backend: {backend_name}")

    writer = None
    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        out_w, out_h, out_fps = camera.output_meta(
            fallback_width=args.width,
            fallback_height=args.height,
            fallback_fps=args.camera_fps if args.camera_fps > 0 else 20.0,
        )
        if out_fps <= 0:
            out_fps = 20.0

        writer = cv2.VideoWriter(
            str(args.save_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps,
            (out_w, out_h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open writer: {args.save_path}")

    print("Running real-time inference. Press 'q' to quit.")

    frame_count = 0
    fps_ema = 0.0
    fps_alpha = 0.2
    prev_t = time.perf_counter()
    last_decision: dict[str, object] | None = None

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Warning: camera read failed, stopping.")
                break

            detections = detector.predict(frame)
            decision_input = [(det.class_id, det.confidence) for det in detections]
            last_decision = decision_engine.update(decision_input)

            now = time.perf_counter()
            dt = max(now - prev_t, 1e-6)
            prev_t = now
            fps = 1.0 / dt
            fps_ema = fps if fps_ema == 0.0 else ((1 - fps_alpha) * fps_ema + fps_alpha * fps)

            draw_overlay(frame, detections, last_decision, fps_ema)

            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("waste_sorter_rpi5", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if args.print_every > 0 and frame_count % args.print_every == 0:
                print(
                    f"frame={frame_count} fps={fps_ema:.1f} "
                    f"detections={len(detections)} bin={last_decision['final_bin']} "
                    f"class={last_decision['top_class']} score={last_decision['score']:.2f}"
                )
    finally:
        camera.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Processed frames: {frame_count}")
    if last_decision is not None:
        print("Last decision:")
        print(json.dumps(last_decision, indent=2))


if __name__ == "__main__":
    main()
