#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.config import load_config
from src.models.database import DatabaseManager
from src.models.schemas import BinMeasurement
from src.sensors.data_collector import resolve_status

CAMPUS_TZ = ZoneInfo("America/New_York")

BIN_BASE_RATE_PER_HOUR: dict[str, float] = {
    "recycle": 4.8,
    "compost": 4.0,
    "landfill": 3.5,
}

LOCATION_MULTIPLIER: dict[str, float] = {
    "La Prima 1": 1.18,  # Main queue + pickup
    "La Prima 2": 1.00,  # Seating area
    "La Prima 3": 0.88,  # Side hallway / low traffic
}


def weekday_multiplier(weekday: int) -> float:
    # CMU cafe usage: heaviest Tue-Thu, lower Friday evenings, light weekends.
    return {
        0: 1.03,  # Mon
        1: 1.12,  # Tue
        2: 1.18,  # Wed
        3: 1.14,  # Thu
        4: 0.96,  # Fri
        5: 0.60,  # Sat
        6: 0.48,  # Sun
    }[weekday]


def hourly_multiplier(hour: int) -> float:
    # La Prima pattern: morning rush + lunch spike, quiet overnight.
    if 7 <= hour < 9:
        return 1.55
    if 9 <= hour < 11:
        return 1.18
    if 11 <= hour < 14:
        return 2.10
    if 14 <= hour < 17:
        return 1.35
    if 17 <= hour < 20:
        return 0.95
    return 0.06


def event_multiplier(ts_local: datetime) -> float:
    # Slight extra rush around midday on Tue-Thu at CMU.
    if ts_local.weekday() in {1, 2, 3} and 11 <= ts_local.hour < 14:
        return 1.22
    return 1.0


def should_empty_bin(ts_local: datetime, fullness: float, rng: random.Random) -> bool:
    if fullness >= 96.0:
        return True

    is_closing_window = ts_local.hour == 19 and ts_local.minute >= 20
    if is_closing_window and fullness >= 74.0 and rng.random() < 0.92:
        return True

    return False


def generate_series(
    *,
    start: datetime,
    end: datetime,
    step_minutes: int,
    seed: str,
    bin_type: str,
    location: str,
) -> list[tuple[datetime, float]]:
    rng = random.Random(seed)
    points: list[tuple[datetime, float]] = []
    ts = start
    fullness = rng.uniform(5.0, 18.0)
    step_hours = step_minutes / 60.0

    while ts <= end:
        ts_local = ts.astimezone(CAMPUS_TZ)
        base_rate = BIN_BASE_RATE_PER_HOUR.get(bin_type, 1.5)
        location_factor = LOCATION_MULTIPLIER.get(location, 1.0)
        weekday_factor = weekday_multiplier(ts_local.weekday())
        hour_factor = hourly_multiplier(ts_local.hour)
        rush_factor = event_multiplier(ts_local)

        noise = rng.uniform(-0.08, 0.18)
        increase = base_rate * location_factor * weekday_factor * hour_factor * rush_factor * step_hours
        fullness = min(99.5, max(0.0, fullness + increase + noise))

        if should_empty_bin(ts_local, fullness, rng):
            fullness = rng.uniform(4.0, 13.0)

        points.append((ts, round(fullness, 2)))
        ts += timedelta(minutes=step_minutes)

    return points


def clear_existing_rows(db_path: Path, *, bin_ids: list[str], full_reset: bool) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        if full_reset:
            cur.execute("DELETE FROM measurements")
            cur.execute("DELETE FROM empty_events")
            cur.execute("DELETE FROM prediction_logs")
        elif bin_ids:
            placeholders = ",".join("?" for _ in bin_ids)
            cur.execute(f"DELETE FROM measurements WHERE bin_id IN ({placeholders})", bin_ids)
            cur.execute(f"DELETE FROM empty_events WHERE bin_id IN ({placeholders})", bin_ids)
            cur.execute(f"DELETE FROM prediction_logs WHERE bin_id IN ({placeholders})", bin_ids)
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed realistic La Prima mock measurements")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument(
        "--replace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace existing rows for configured bins before insert (default: true)",
    )
    parser.add_argument(
        "--full-reset",
        action="store_true",
        help="Delete all measurements/prediction logs/empty events before seeding",
    )
    args = parser.parse_args()

    config = load_config(ROOT / "config")
    if args.replace:
        clear_existing_rows(
            config.database.path,
            bin_ids=[item.id for item in config.bins],
            full_reset=args.full_reset,
        )

    db = DatabaseManager(config.database.path)
    db.initialize()

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=args.days)

    created = 0
    for bin_cfg in config.bins:
        series = generate_series(
            start=start,
            end=end,
            step_minutes=args.step_minutes,
            seed=f"{bin_cfg.id}:{args.days}:{bin_cfg.location}",
            bin_type=bin_cfg.type,
            location=bin_cfg.location,
        )
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
    print(f"Inserted {created} La Prima-pattern rows into {config.database.path}")


if __name__ == "__main__":
    main()
