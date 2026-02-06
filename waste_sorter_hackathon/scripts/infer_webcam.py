"""Run live YOLO inference on webcam with FPS display."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on webcam.")
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--camera_index", type=int, default=0)
    return parser.parse_args()


def get_class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def draw_predictions(frame, result, names: object):
    boxes = result.boxes
    if boxes is None:
        return frame

    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        label = f"{get_class_name(names, cls_id)} {conf:.2f}"

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

    return frame


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam index {args.camera_index}")

    fps_ema = 0.0
    alpha = 0.2
    prev_t = time.perf_counter()

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Warning: webcam frame read failed")
            break

        result = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)[0]
        annotated = draw_predictions(frame, result, model.names)

        now = time.perf_counter()
        dt = max(now - prev_t, 1e-6)
        prev_t = now

        fps = 1.0 / dt
        fps_ema = fps if fps_ema == 0.0 else (1 - alpha) * fps_ema + alpha * fps

        cv2.putText(
            annotated,
            f"FPS: {fps_ema:.1f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 240, 20),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("waste_sorter_webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
