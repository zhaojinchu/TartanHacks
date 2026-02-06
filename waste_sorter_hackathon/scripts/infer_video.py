"""Run YOLO inference on a video and save annotated output."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a video file.")
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--source", required=True, type=str, help="Input video path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save_path", required=True, type=str, help="Output video path")
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
    source = Path(args.source)
    save_path = Path(args.save_path)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Video source not found: {source}")

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video writer: {save_path}")

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)[0]
        annotated = draw_predictions(frame, result, model.names)
        writer.write(annotated)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    writer.release()

    print(f"Done. Processed {frame_count} frames")
    print(f"Saved: {save_path.resolve()}")


if __name__ == "__main__":
    main()
