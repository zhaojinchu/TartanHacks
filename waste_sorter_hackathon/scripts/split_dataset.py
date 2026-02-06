"""Split a flat YOLO dataset into train/val/test folders."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split flat images+labels into YOLO folders.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Flat labeled input directory")
    parser.add_argument(
        "--output_dir", type=Path, default=Path("dataset"), help="Output dataset root"
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_files(input_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        stem = path.stem

        if suffix in IMAGE_EXTS:
            if stem in images:
                raise ValueError(f"Duplicate image stem found: {stem}")
            images[stem] = path
        elif suffix == ".txt" and path.name != "classes.txt":
            if stem in labels:
                raise ValueError(f"Duplicate label stem found: {stem}")
            labels[stem] = path

    return images, labels


def ensure_output_dirs(output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_stems(
    stems: list[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[str], list[str], list[str]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)
    shuffled = stems[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_n = int(total * train_ratio)
    val_n = int(total * val_ratio)
    test_n = total - train_n - val_n

    train_stems = shuffled[:train_n]
    val_stems = shuffled[train_n : train_n + val_n]
    test_stems = shuffled[train_n + val_n : train_n + val_n + test_n]

    return train_stems, val_stems, test_stems


def copy_split(
    stems: list[str], split: str, images: dict[str, Path], labels: dict[str, Path], output_dir: Path
) -> None:
    for stem in stems:
        img_src = images[stem]
        lbl_src = labels[stem]

        img_dst = output_dir / "images" / split / img_src.name
        lbl_dst = output_dir / "labels" / split / lbl_src.name

        shutil.copy2(img_src, img_dst)
        shutil.copy2(lbl_src, lbl_dst)


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    images, labels = collect_files(args.input_dir)

    image_stems = set(images.keys())
    label_stems = set(labels.keys())
    paired_stems = sorted(image_stems & label_stems)

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    if not paired_stems:
        raise RuntimeError("No paired image/label files found")

    ensure_output_dirs(args.output_dir)

    train_stems, val_stems, test_stems = split_stems(
        stems=paired_stems,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    copy_split(train_stems, "train", images, labels, args.output_dir)
    copy_split(val_stems, "val", images, labels, args.output_dir)
    copy_split(test_stems, "test", images, labels, args.output_dir)

    print(f"Paired samples: {len(paired_stems)}")
    print(f"train={len(train_stems)} val={len(val_stems)} test={len(test_stems)}")
    print(f"Output written to: {args.output_dir}")

    if missing_labels:
        print(f"Warning: images missing labels: {len(missing_labels)}")
    if missing_images:
        print(f"Warning: labels missing images: {len(missing_images)}")


if __name__ == "__main__":
    main()
