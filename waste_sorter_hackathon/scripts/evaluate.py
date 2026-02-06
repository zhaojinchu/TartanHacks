"""Evaluate a trained YOLO model and print key metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model.")
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--data", type=str, default="configs/data.yaml")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--project", type=str, default="runs_hack")
    parser.add_argument("--name", type=str, default="eval")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str | int:
    if device_arg != "auto":
        return device_arg
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    data_path = Path(args.data)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_path}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    save_dir = Path(metrics.save_dir)
    confusion = sorted(save_dir.glob("confusion_matrix*.png"))
    samples = sorted(save_dir.glob("val_batch*.jpg"))

    print("=== Eval Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"mAP50:     {map50:.4f}")
    print(f"mAP50-95:  {map5095:.4f}")
    print(f"Artifacts saved to: {save_dir}")
    print(f"Confusion matrix files: {len(confusion)}")
    print(f"Sample prediction files: {len(samples)}")


if __name__ == "__main__":
    main()
