"""Remove near-duplicate frames using perceptual hash similarity."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class KeptFrame:
    hash_bits: np.ndarray
    sharpness: float
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate near-identical frames.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Input frames directory")
    parser.add_argument("--output_dir", required=True, type=Path, help="Deduped output directory")
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.92,
        help="Hash similarity threshold in [0,1]",
    )
    return parser.parse_args()


def compute_ahash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return small > float(np.mean(small))


def hash_similarity(hash_a: np.ndarray, hash_b: np.ndarray) -> float:
    return float(np.count_nonzero(hash_a == hash_b) / hash_a.size)


def laplacian_variance(image: np.ndarray) -> float:
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def list_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def resolve_unique_path(output_dir: Path, name: str) -> Path:
    candidate = output_dir / name
    if not candidate.exists():
        return candidate

    stem = Path(name).stem
    suffix = Path(name).suffix
    idx = 1
    while True:
        candidate = output_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def find_best_match(
    hash_bits: np.ndarray, kept_frames: list[KeptFrame], threshold: float
) -> int | None:
    best_idx: int | None = None
    best_sim = -1.0

    for idx, kept in enumerate(kept_frames):
        sim = hash_similarity(hash_bits, kept.hash_bits)
        if sim >= threshold and sim > best_sim:
            best_sim = sim
            best_idx = idx

    return best_idx


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not (0.0 <= args.sim_threshold <= 1.0):
        raise ValueError("--sim_threshold must be in [0,1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.input_dir}")

    kept_frames: list[KeptFrame] = []
    unreadable = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            unreadable += 1
            print(f"Warning: unreadable image skipped: {image_path}")
            continue

        hash_bits = compute_ahash(image)
        sharpness = laplacian_variance(image)

        match_idx = find_best_match(hash_bits, kept_frames, args.sim_threshold)
        if match_idx is None:
            dst = resolve_unique_path(args.output_dir, image_path.name)
            shutil.copy2(image_path, dst)
            kept_frames.append(KeptFrame(hash_bits=hash_bits, sharpness=sharpness, output_path=dst))
            continue

        matched = kept_frames[match_idx]
        if sharpness > matched.sharpness:
            if matched.output_path.exists():
                matched.output_path.unlink()
            dst = resolve_unique_path(args.output_dir, image_path.name)
            shutil.copy2(image_path, dst)
            matched.hash_bits = hash_bits
            matched.sharpness = sharpness
            matched.output_path = dst

    processed = len(image_paths) - unreadable
    kept = len(kept_frames)
    removed = processed - kept

    print(
        "Done. "
        f"processed={processed} kept={kept} removed={removed} unreadable={unreadable}"
    )


if __name__ == "__main__":
    main()
