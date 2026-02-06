"""Run ONNX inference on a single image for quick debugging on Raspberry Pi."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import TemporalDecisionEngine
from src.io_utils import load_decision_config, resolve_class_names
from src.onnx_detector import ONNXWasteDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ONNX inference on one image and save an annotated output."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/waste_sorter.onnx"),
        help="Path to ONNX model",
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output annotated image path (default: <image>_pred.jpg)",
    )
    parser.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional path to save detections as JSON",
    )
    parser.add_argument(
        "--class_names_yaml",
        type=Path,
        default=None,
        help="Optional YAML with `names` list/dict (fallback if ONNX metadata missing).",
    )
    parser.add_argument(
        "--decision_config",
        type=Path,
        default=Path("configs/decision.yaml"),
        help="Decision YAML used for final bin mapping",
    )
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--line_width", type=int, default=2)
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not write annotated image to disk",
    )
    return parser.parse_args()


def default_output_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_pred.jpg")


def draw_detections(
    image,
    detections,
    class_names: list[str],
    line_width: int,
):
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        class_name = class_names[det.class_id] if 0 <= det.class_id < len(class_names) else str(det.class_id)
        label = f"{class_name} {det.confidence:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), line_width)
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            line_width,
            cv2.LINE_AA,
        )

    return image


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    fallback_class_names = resolve_class_names(
        class_names=None,
        class_names_yaml=args.class_names_yaml,
    )

    detector = ONNXWasteDetector(
        model_path=args.model,
        class_names=fallback_class_names,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    runtime_class_names = detector.class_names

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    detections = detector.predict(image)

    class_to_bin, threshold, window_size = load_decision_config(
        args.decision_config,
        class_names=runtime_class_names,
    )
    engine = TemporalDecisionEngine(
        class_to_bin=class_to_bin,
        threshold=threshold,
        window_size=window_size,
        class_names=runtime_class_names,
    )
    decision = engine.update([(d.class_id, d.confidence) for d in detections])

    print(f"Image: {args.image}")
    print(f"Resolution: {image.shape[1]}x{image.shape[0]}")
    print(f"Model classes ({len(runtime_class_names)}): {runtime_class_names}")
    print(f"Detections: {len(detections)}")

    detection_rows: list[dict[str, object]] = []
    for det in detections:
        class_name = runtime_class_names[det.class_id] if 0 <= det.class_id < len(runtime_class_names) else str(det.class_id)
        row = {
            "class_id": det.class_id,
            "class_name": class_name,
            "confidence": round(float(det.confidence), 4),
            "box_xyxy": [int(v) for v in det.box_xyxy],
        }
        detection_rows.append(row)

    print("Detections JSON:")
    print(json.dumps(detection_rows, indent=2))
    print("Decision:")
    print(json.dumps(decision, indent=2))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image": str(args.image),
            "detections": detection_rows,
            "decision": decision,
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON: {args.json_out}")

    if not args.no_save:
        out_path = args.out if args.out is not None else default_output_path(args.image)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = draw_detections(
            image=image.copy(),
            detections=detections,
            class_names=runtime_class_names,
            line_width=max(1, args.line_width),
        )
        ok = cv2.imwrite(str(out_path), annotated)
        if not ok:
            raise RuntimeError(f"Failed to write output image: {out_path}")
        print(f"Saved annotated image: {out_path}")


if __name__ == "__main__":
    main()
