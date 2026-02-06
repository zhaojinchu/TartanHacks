"""Export trained YOLO `.pt` weights to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX.")
    parser.add_argument("--weights", required=True, type=str, help="Path to .pt weights")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    export_result = model.export(format="onnx", opset=args.opset, imgsz=args.imgsz)

    if isinstance(export_result, (str, Path)):
        onnx_path = Path(export_result)
    else:
        onnx_path = weights.with_suffix(".onnx")

    if not onnx_path.exists():
        raise RuntimeError(f"ONNX export did not produce file: {onnx_path}")

    print(f"ONNX exported: {onnx_path.resolve()}")


if __name__ == "__main__":
    main()
