"""Validate YOLO labels for class range, bbox normalization, and file pairing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check YOLO labels.")
    parser.add_argument("--images_dir", required=True, type=Path)
    parser.add_argument("--labels_dir", required=True, type=Path)
    parser.add_argument("--num_classes", type=int, default=8)
    return parser.parse_args()


def build_index(root: Path, valid_suffixes: set[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in valid_suffixes:
            continue
        if path.name == "classes.txt":
            continue

        key = str(path.relative_to(root).with_suffix("")).replace("\\", "/")
        if key in index:
            raise ValueError(f"Duplicate file key found: {key} in {root}")
        index[key] = path
    return index


def validate_label_file(path: Path, num_classes: int) -> tuple[int, list[str]]:
    errors: list[str] = []
    valid_boxes = 0

    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{path}:{line_num} expected 5 values, got {len(parts)}")
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                errors.append(f"{path}:{line_num} class id is not int: {parts[0]}")
                continue

            if not (0 <= class_id < num_classes):
                errors.append(
                    f"{path}:{line_num} class id {class_id} out of range [0,{num_classes - 1}]"
                )
                continue

            try:
                x, y, w, h = (float(v) for v in parts[1:])
            except ValueError:
                errors.append(f"{path}:{line_num} bbox values must be floats")
                continue

            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                errors.append(
                    f"{path}:{line_num} bbox values must be normalized in [0,1], "
                    f"got {(x, y, w, h)}"
                )
                continue

            valid_boxes += 1

    return valid_boxes, errors


def main() -> None:
    args = parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    if not args.labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {args.labels_dir}")

    image_index = build_index(args.images_dir, IMAGE_EXTS)
    label_index = build_index(args.labels_dir, {".txt"})

    image_keys = set(image_index.keys())
    label_keys = set(label_index.keys())

    missing_labels = sorted(image_keys - label_keys)
    orphan_labels = sorted(label_keys - image_keys)

    total_boxes = 0
    critical_errors: list[str] = []

    for key in sorted(label_keys & image_keys):
        label_path = label_index[key]
        box_count, errors = validate_label_file(label_path, args.num_classes)
        total_boxes += box_count
        critical_errors.extend(errors)

    for key in orphan_labels:
        critical_errors.append(
            f"Label has no matching image: {label_index[key]} (key={key})"
        )

    print("=== Label Sanity Check Report ===")
    print(f"Images found: {len(image_index)}")
    print(f"Labels found: {len(label_index)}")
    print(f"Valid paired files: {len(image_keys & label_keys)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Orphan labels: {len(orphan_labels)}")
    print(f"Valid boxes parsed: {total_boxes}")
    print(f"Critical errors: {len(critical_errors)}")

    if missing_labels:
        print("\nWarnings (first 20):")
        for key in missing_labels[:20]:
            print(f"- Image has no label: {image_index[key]} (key={key})")

    if critical_errors:
        print("\nCritical errors (first 50):")
        for err in critical_errors[:50]:
            print(f"- {err}")
        sys.exit(1)

    print("\nSanity check passed.")


if __name__ == "__main__":
    main()
