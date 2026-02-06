"""Extract frames from videos at a target FPS."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory with videos")
    parser.add_argument(
        "--output_dir", required=True, type=Path, help="Where extracted JPG frames go"
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Target extracted FPS")
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=None,
        help="Optional hard cap of frames per video",
    )
    parser.add_argument("--width", type=int, default=None, help="Optional resize width")
    parser.add_argument("--height", type=int, default=None, help="Optional resize height")
    return parser.parse_args()


def find_videos(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float,
    max_frames: int | None,
    width: int | None,
    height: int | None,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if target_fps <= 0:
        raise ValueError("--fps must be > 0")

    if src_fps > 0:
        sample_every = max(int(round(src_fps / target_fps)), 1)
    else:
        sample_every = 1

    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every == 0:
            if width is not None and height is not None:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            output_name = f"{video_path.stem}_{frame_idx:06d}.jpg"
            output_path = output_dir / output_name
            success = cv2.imwrite(str(output_path), frame)
            if not success:
                raise RuntimeError(f"Failed to write frame: {output_path}")

            saved += 1
            if max_frames is not None and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    if (args.width is None) != (args.height is None):
        raise ValueError("Provide both --width and --height, or neither")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(args.input_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {args.input_dir}")

    total_frames = 0
    for video_path in videos:
        count = extract_video_frames(
            video_path=video_path,
            output_dir=args.output_dir,
            target_fps=args.fps,
            max_frames=args.max_frames_per_video,
            width=args.width,
            height=args.height,
        )
        total_frames += count
        print(f"{video_path.name}: extracted {count} frames")

    print(f"Done. Videos={len(videos)} total_frames={total_frames}")


if __name__ == "__main__":
    main()
