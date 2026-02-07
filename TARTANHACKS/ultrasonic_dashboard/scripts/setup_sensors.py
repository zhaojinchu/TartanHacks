#!/usr/bin/env python3
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.config import load_config
from src.sensors.ultrasonic import SensorReadError, build_reader


def main() -> None:
    config = load_config(ROOT / "config")
    print(f"Loaded {len(config.bins)} bins (mock_mode={config.sensors.mock_mode})")

    for bin_cfg in config.bins:
        print(f"\nTesting {bin_cfg.id} [{bin_cfg.type}] @ {bin_cfg.location}")
        reader = build_reader(bin_cfg, mock_mode=config.sensors.mock_mode)
        samples: list[float] = []

        try:
            for _ in range(5):
                distance = reader.read_distance(samples=3, timeout_s=config.measurement.sensor_timeout_seconds)
                samples.append(distance)
                print(f"  distance_cm={distance:.2f}")
                time.sleep(0.2)

            mean = statistics.mean(samples)
            stdev = statistics.pstdev(samples)
            print(f"  avg={mean:.2f} cm stdev={stdev:.2f} cm")
        except SensorReadError as exc:
            print(f"  ERROR: {exc}")
        finally:
            reader.close()


if __name__ == "__main__":
    main()
