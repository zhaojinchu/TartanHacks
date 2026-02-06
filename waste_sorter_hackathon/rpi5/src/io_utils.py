"""Raspberry Pi runtime helpers for config loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CLASS_NAMES: list[str] = [
    "rigid_plastic_container",
]

VALID_BINS: set[str] = {"recycle", "compost", "landfill"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML as dictionary."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    with path_obj.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML dictionary: {path_obj}")

    return data


def build_class_to_bin_map(decision_cfg: dict[str, Any]) -> dict[str, str]:
    """Validate and build class->bin map."""
    bins_cfg = decision_cfg.get("bins")
    if not isinstance(bins_cfg, dict):
        raise ValueError("decision.yaml must contain a `bins` dictionary")

    class_to_bin: dict[str, str] = {}

    for bin_name, classes in bins_cfg.items():
        if bin_name not in VALID_BINS:
            raise ValueError(
                f"Invalid bin `{bin_name}`. Valid bins: {sorted(VALID_BINS)}"
            )

        if not isinstance(classes, list):
            raise ValueError(f"Bin `{bin_name}` must map to a list")

        for class_name in classes:
            if class_name not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class `{class_name}` in decision config. "
                    f"Valid classes: {CLASS_NAMES}"
                )
            if class_name in class_to_bin:
                raise ValueError(f"Duplicate mapping for class `{class_name}`")
            class_to_bin[class_name] = bin_name

    missing = [name for name in CLASS_NAMES if name not in class_to_bin]
    if missing:
        raise ValueError(
            "decision.yaml missing class mappings for: " + ", ".join(missing)
        )

    return class_to_bin


def load_decision_config(path: str | Path) -> tuple[dict[str, str], float, int]:
    """Load class map, unknown threshold, and window size."""
    cfg = load_yaml(path)
    class_to_bin = build_class_to_bin_map(cfg)

    threshold = float(cfg.get("unknown_threshold", 0.60))
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("`unknown_threshold` must be in [0,1]")

    window_size = int(cfg.get("window_size", 5))
    if window_size <= 0:
        raise ValueError("`window_size` must be > 0")

    return class_to_bin, threshold, window_size
