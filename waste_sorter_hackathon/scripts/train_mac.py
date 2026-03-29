"""Train a higher-capacity YOLO model for Mac (M-series) inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a medium/large YOLO model for Mac-native inference."
    )
    parser.add_argument("--data", type=str, default="configs/data.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m.pt",
        help="Pretrained weights: yolo11m.pt (recommended), yolo11l.pt, yolov8m.pt, yolov8l.pt",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--project", type=str, default="runs_hack")
    parser.add_argument("--name", type=str, default="mac_medium")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs with no mAP improvement)",
    )
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--cos_lr", action="store_true", help="Use cosine LR schedule")
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


def resolve_repo_path(path_arg: str) -> Path:
    """Resolve path relative to repo root unless already absolute."""
    path = Path(path_arg).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def main() -> None:
    args = parse_args()

    data_path = resolve_repo_path(args.data)
    project_dir = resolve_repo_path(args.project)
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_path}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Data config: {data_path}")
    print(f"Project dir: {project_dir}")
    print(f"Image size: {args.imgsz}, Batch: {args.batch}, Epochs: {args.epochs}")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=str(project_dir),
        name=args.name,
        device=device,
        seed=args.seed,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
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
