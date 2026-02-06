"""Demo script: YOLO detections -> temporal smoothing -> final bin routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.decision import TemporalDecisionEngine
from src.io_utils import CLASS_NAMES, load_decision_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run routing demo with temporal smoothing.")
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video path or webcam index string (e.g. '0')",
    )
    parser.add_argument("--decision_config", type=str, default="configs/decision.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--window", type=int, default=None, help="Override smoothing window")
    parser.add_argument(
        "--threshold", type=float, default=None, help="Override low-confidence threshold"
    )
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Disable on-screen window (useful for headless runs)",
    )
    return parser.parse_args()


def get_class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def parse_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    class_to_bin, cfg_threshold, cfg_window = load_decision_config(args.decision_config)
    threshold = args.threshold if args.threshold is not None else cfg_threshold
    window_size = args.window if args.window is not None else cfg_window

    engine = TemporalDecisionEngine(
        class_to_bin=class_to_bin,
        threshold=threshold,
        window_size=window_size,
        class_names=CLASS_NAMES,
    )

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    writer = None
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0

        writer = cv2.VideoWriter(
            str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open output writer: {save_path}")

    last_decision = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)[0]

        detections: list[tuple[int, float]] = []
        boxes = result.boxes
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

        last_decision = engine.update(detections)

        overlay_lines = [
            f"final_bin: {last_decision['final_bin']}",
            f"top_class: {last_decision['top_class']} ({last_decision['score']:.2f})",
            f"reason: {last_decision['reason']}",
        ]

        for i, text in enumerate(overlay_lines):
            y = 28 + i * 26
            cv2.putText(
                frame,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (30, 220, 255),
                2,
                cv2.LINE_AA,
            )

        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("waste_sorter_route_demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(
                f"Processed {frame_idx} frames | final_bin={last_decision['final_bin']} "
                f"score={last_decision['score']:.2f}"
            )

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    if last_decision is None:
        print("No frames processed.")
        return

    print("Last decision:")
    print(json.dumps(last_decision, indent=2))


if __name__ == "__main__":
    main()
