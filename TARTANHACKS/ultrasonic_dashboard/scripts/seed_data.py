#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.config import load_config
from src.models.database import DatabaseManager
from src.models.schemas import BinMeasurement
from src.sensors.data_collector import resolve_status


def generate_series(start: datetime, end: datetime, step_minutes: int, seed: str) -> list[tuple[datetime, float]]:
    rng = random.Random(seed)
    points: list[tuple[datetime, float]] = []
    ts = start
    fullness = rng.uniform(2.0, 18.0)

    while ts <= end:
        if rng.random() < 0.02:
            fullness = rng.uniform(1.0, 10.0)
        else:
            fullness = min(100.0, fullness + rng.uniform(0.0, 1.4))
        points.append((ts, fullness))
        ts += timedelta(minutes=step_minutes)

    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed mock historical measurements")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--step-minutes", type=int, default=30)
    args = parser.parse_args()

    config = load_config(ROOT / "config")
    db = DatabaseManager(config.database.path)
    db.initialize()

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=args.days)

    created = 0
    for bin_cfg in config.bins:
        series = generate_series(start, end, args.step_minutes, seed=f"{bin_cfg.id}:{args.days}")
        for ts, fullness in series:
            trash_height = (fullness / 100.0) * bin_cfg.height_cm
            distance = round(bin_cfg.sensor_offset_cm + bin_cfg.height_cm - trash_height, 2)
            status = resolve_status(
                fullness,
                normal_max=config.thresholds.normal_max,
                warning_max=config.thresholds.warning_max,
                full_min=config.thresholds.full_min,
            )

            db.insert_measurement(
                BinMeasurement(
                    bin_id=bin_cfg.id,
                    bin_type=bin_cfg.type,
                    timestamp=ts,
                    distance_cm=distance,
                    fullness_percent=round(fullness, 2),
                    bin_height_cm=bin_cfg.height_cm,
                    status=status,
                    location=bin_cfg.location,
                ),
                source="seed",
            )
            created += 1

    db.close()
    print(f"Inserted {created} rows into {config.database.path}")


if __name__ == "__main__":
    main()
