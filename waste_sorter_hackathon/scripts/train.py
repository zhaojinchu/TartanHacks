"""Train a lightweight YOLO model for waste detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO waste sorter model.")
    parser.add_argument("--data", type=str, default="configs/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=str, default="runs_hack")
    parser.add_argument("--name", type=str, default="baseline")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str | int:
    """Resolve `auto` to mps, cuda:0, or cpu."""
    if device_arg != "auto":
        return device_arg

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return 0

    return "cpu"


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_path}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=device,
        seed=args.seed,
        save=True,
        exist_ok=True,
    )

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    print(f"Training outputs: {save_dir}")
    print(f"best.pt: {best_pt} (exists={best_pt.exists()})")
    print(f"last.pt: {last_pt} (exists={last_pt.exists()})")


if __name__ == "__main__":
    main()
