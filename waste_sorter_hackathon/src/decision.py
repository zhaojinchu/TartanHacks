"""Temporal decision logic for mapping detections to bins."""

from __future__ import annotations

from collections import deque
from typing import Sequence

from src.io_utils import CLASS_NAMES

Detection = tuple[int, float]
DecisionResult = dict[str, object]


class TemporalDecisionEngine:
    """Smooth frame detections over time and return a single bin decision."""

    def __init__(
        self,
        class_to_bin: dict[str, str],
        threshold: float = 0.60,
        window_size: int = 5,
        class_names: list[str] | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")

        self.class_names = class_names or CLASS_NAMES
        self.class_to_bin = class_to_bin
        self.threshold = threshold
        self.window_size = window_size
        self.history: deque[list[float]] = deque(maxlen=window_size)

    def reset(self) -> None:
        """Clear all history."""
        self.history.clear()

    def update(self, detections: Sequence[Detection]) -> DecisionResult:
        """Update smoothing state with one frame of detections.

        Args:
            detections: List of `(class_id, confidence)` tuples for the frame.

        Returns:
            Dictionary with final bin decision and diagnostic scores.
        """
        frame_scores = [0.0] * len(self.class_names)

        for class_id, confidence in detections:
            if not (0 <= class_id < len(self.class_names)):
                continue
            conf = max(0.0, min(float(confidence), 1.0))
            frame_scores[class_id] = max(frame_scores[class_id], conf)

        self.history.append(frame_scores)

        per_class_scores: dict[str, float] = {}
        history_len = len(self.history)

        for idx, class_name in enumerate(self.class_names):
            avg_score = sum(frame[idx] for frame in self.history) / history_len
            per_class_scores[class_name] = float(avg_score)

        top_class = max(per_class_scores, key=per_class_scores.get)
        top_score = float(per_class_scores[top_class])

        if top_score < self.threshold:
            return {
                "final_bin": "landfill",
                "top_class": top_class,
                "score": top_score,
                "reason": "unknown_low_conf",
                "per_class_scores": per_class_scores,
            }

        final_bin = self.class_to_bin.get(top_class, "landfill")
        return {
            "final_bin": final_bin,
            "top_class": top_class,
            "score": top_score,
            "reason": "mapped_from_class",
            "per_class_scores": per_class_scores,
        }
