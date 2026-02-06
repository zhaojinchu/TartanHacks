"""Small IO helpers shared across scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CLASS_NAMES: list[str] = [
    "aluminum_can",
    "plastic_bottle",
    "lp_paper_cup",
    "lp_plastic_cup",
    "rigid_plastic_container",
    "straw",
    "utensil",
    "napkin",
]

VALID_BINS: set[str] = {"recycle", "compost", "landfill"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a dict."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML must parse to a dictionary: {yaml_path}")

    return data


def build_class_to_bin_map(decision_cfg: dict[str, Any]) -> dict[str, str]:
    """Build and validate class->bin mapping from decision config."""
    bins_cfg = decision_cfg.get("bins")
    if not isinstance(bins_cfg, dict):
        raise ValueError("decision.yaml must include a `bins` dictionary")

    class_to_bin: dict[str, str] = {}

    for bin_name, class_list in bins_cfg.items():
        if bin_name not in VALID_BINS:
            raise ValueError(
                f"Invalid bin name `{bin_name}`. Valid bins: {sorted(VALID_BINS)}"
            )
        if not isinstance(class_list, list):
            raise ValueError(f"Bin `{bin_name}` must map to a list of classes")

        for class_name in class_list:
            if class_name not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class `{class_name}` in decision.yaml. "
                    f"Valid classes: {CLASS_NAMES}"
                )
            if class_name in class_to_bin:
                raise ValueError(
                    f"Class `{class_name}` is mapped more than once in decision.yaml"
                )
            class_to_bin[class_name] = bin_name

    missing_classes = [c for c in CLASS_NAMES if c not in class_to_bin]
    if missing_classes:
        raise ValueError(
            "decision.yaml is missing class mappings for: "
            + ", ".join(missing_classes)
        )

    return class_to_bin


def load_decision_config(path: str | Path) -> tuple[dict[str, str], float, int]:
    """Load class mapping, unknown threshold, and smoothing window from YAML."""
    cfg = load_yaml(path)
    class_to_bin = build_class_to_bin_map(cfg)

    threshold = float(cfg.get("unknown_threshold", 0.60))
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("`unknown_threshold` must be in [0, 1]")

    window_size = int(cfg.get("window_size", 5))
    if window_size <= 0:
        raise ValueError("`window_size` must be > 0")

    return class_to_bin, threshold, window_size
