#!/usr/bin/env python3
from __future__ import annotations


def prompt_float(label: str) -> float:
    while True:
        raw = input(f"{label}: ").strip()
        try:
            value = float(raw)
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Please enter a positive number.")


def main() -> None:
    print("Ultrasonic Bin Calibration")
    print("Measure with an empty bin first, then nearly full.")

    empty_distance = prompt_float("Distance when bin is empty (cm)")
    full_distance = prompt_float("Distance when bin is near full target (cm)")

    bin_height = empty_distance - full_distance
    sensor_offset = max(empty_distance - bin_height, 0.0)

    print("\nSuggested config values:")
    print(f"  height_cm: {bin_height:.2f}")
    print(f"  sensor_offset_cm: {sensor_offset:.2f}")
    print("\nAdd these values to config/bins.yaml for the selected bin.")


if __name__ == "__main__":
    main()
