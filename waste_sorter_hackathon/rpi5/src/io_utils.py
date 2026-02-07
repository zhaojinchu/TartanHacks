"""Raspberry Pi runtime helpers for config loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CLASS_NAMES: list[str] = [
    "lp_cup_lids",
    "lp_paper_cup",
    "lp_plastic_cup",
    "napkin",
    "rigid_plastic_container",
    "rigid_plastic_lid",
    "small_plastic_container",
    "straw",
    "utensil",
]

VALID_BINS: set[str] = {"bottles", "compost", "landfill"}


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


def resolve_class_names(
    class_names: list[str] | None,
    class_names_yaml: str | Path | None = None,
) -> list[str]:
    """Resolve class names from explicit list, YAML, or default locked list."""
    if class_names:
        resolved = [str(x) for x in class_names]
        if not resolved:
            raise ValueError("Provided class_names is empty")
        return resolved

    if class_names_yaml is not None:
        cfg = load_yaml(class_names_yaml)
        names = cfg.get("names")

        if isinstance(names, list):
            resolved = [str(x) for x in names]
            if not resolved:
                raise ValueError(f"No names found in {class_names_yaml}")
            return resolved

        if isinstance(names, dict):
            try:
                ordered_keys = sorted(names.keys(), key=lambda k: int(k))
            except Exception:
                ordered_keys = sorted(names.keys(), key=str)
            resolved = [str(names[k]) for k in ordered_keys]
            if not resolved:
                raise ValueError(f"No names found in {class_names_yaml}")
            return resolved

        raise ValueError(
            f"{class_names_yaml} must include `names` as list or dict, "
            f"got {type(names).__name__}"
        )

    return DEFAULT_CLASS_NAMES.copy()


def build_class_to_bin_map(
    decision_cfg: dict[str, Any], class_names: list[str]
) -> dict[str, str]:
    """Build class->bin mapping for current runtime class list.

    decision config may contain additional class mappings that are not present in
    the current model class list; those are ignored.
    """
    bins_cfg = decision_cfg.get("bins")
    if not isinstance(bins_cfg, dict):
        raise ValueError("decision.yaml must contain a `bins` dictionary")

    class_set = set(class_names)
    known_reference = set(DEFAULT_CLASS_NAMES) | class_set
    class_to_bin: dict[str, str] = {}

    for bin_name, classes in bins_cfg.items():
        if bin_name not in VALID_BINS:
            raise ValueError(
                f"Invalid bin `{bin_name}`. Valid bins: {sorted(VALID_BINS)}"
            )

        if not isinstance(classes, list):
            raise ValueError(f"Bin `{bin_name}` must map to a list")

        for class_name in classes:
            if class_name not in known_reference:
                raise ValueError(
                    f"Unknown class `{class_name}` in decision config. "
                    f"Known classes include: {sorted(known_reference)}"
                )

            # Ignore classes that are not part of the currently-loaded model.
            if class_name not in class_set:
                continue

            if class_name in class_to_bin:
                raise ValueError(f"Duplicate mapping for class `{class_name}`")
            class_to_bin[class_name] = bin_name

    missing = [name for name in class_names if name not in class_to_bin]
    if missing:
        print(
            "Warning: decision.yaml missing class mappings for "
            + ", ".join(missing)
            + ". Defaulting them to landfill."
        )
        for name in missing:
            class_to_bin[name] = "landfill"

    return class_to_bin


def load_decision_config(
    path: str | Path,
    class_names: list[str] | None = None,
) -> tuple[dict[str, str], float, int]:
    """Load class map, unknown threshold, and window size."""
    cfg = load_yaml(path)
    resolved_names = resolve_class_names(class_names)
    class_to_bin = build_class_to_bin_map(cfg, resolved_names)

    threshold = float(cfg.get("unknown_threshold", 0.60))
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("`unknown_threshold` must be in [0,1]")

    window_size = int(cfg.get("window_size", 5))
    if window_size <= 0:
        raise ValueError("`window_size` must be > 0")

    return class_to_bin, threshold, window_size
