from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import exp
from typing import Any

import numpy as np


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def detect_anomalies(rows: list[dict[str, Any]]) -> list[str]:
    anomalies: list[str] = []
    if not rows:
        return anomalies

    values = [float(row["fullness_percent"]) for row in rows if row.get("fullness_percent") is not None]
    if not values:
        return anomalies

    if any(v < 0 or v > 100 for v in values):
        anomalies.append("impossible_readings")

    if len(values) >= 6 and max(values[-6:]) - min(values[-6:]) < 0.2:
        anomalies.append("stuck_sensor")

    if len(values) >= 2 and values[-1] < values[-2] - 40:
        anomalies.append("sudden_drop_possible_empty")

    return anomalies


def calculate_fill_rate(rows: list[dict[str, Any]], window_hours: int = 24) -> tuple[float | None, float | None]:
    if not rows:
        return None, None

    now = datetime.now(tz=UTC)
    filtered: list[tuple[datetime, float]] = []
    for row in rows:
        fullness = row.get("fullness_percent")
        if fullness is None:
            continue
        ts = _parse_ts(row["timestamp"])
        if now - ts <= timedelta(hours=window_hours):
            filtered.append((ts, float(fullness)))

    filtered.sort(key=lambda item: item[0])
    if len(filtered) < 3:
        return None, None

    weighted_rates: list[tuple[float, float]] = []
    for (prev_ts, prev_fullness), (curr_ts, curr_fullness) in zip(filtered, filtered[1:], strict=False):
        dt_hours = (curr_ts - prev_ts).total_seconds() / 3600.0
        if dt_hours <= 0:
            continue

        delta = curr_fullness - prev_fullness
        if delta < -12.0:
            # Large drop usually means a bin empty event; do not treat it as negative fill.
            continue

        if delta <= 0:
            continue

        rate = min(delta / dt_hours, 35.0)
        age_hours = (now - curr_ts).total_seconds() / 3600.0
        weight = exp(-age_hours / 10.0)
        weighted_rates.append((rate, weight))

    if not weighted_rates:
        return None, None

    weight_sum = sum(weight for _, weight in weighted_rates)
    slope = sum(rate * weight for rate, weight in weighted_rates) / max(weight_sum, 1e-6)
    variance = sum(weight * ((rate - slope) ** 2) for rate, weight in weighted_rates) / max(weight_sum, 1e-6)
    return float(slope), float(variance)


def predict_time_to_target(
    *,
    bin_id: str,
    current_fullness: float | None,
    rows: list[dict[str, Any]],
    target_fullness: float = 90.0,
) -> dict[str, Any]:
    if current_fullness is None:
        return {
            "bin_id": bin_id,
            "target_fullness": target_fullness,
            "current_fullness": None,
            "predicted_full_at": None,
            "hours_to_target": None,
            "fill_rate_per_hour": None,
            "confidence_low_hours": None,
            "confidence_high_hours": None,
            "confidence_score": 0.0,
            "anomalies": ["no_current_reading"],
        }

    anomalies = detect_anomalies(rows)
    rate, variance = calculate_fill_rate(rows)

    if rate is None or rate <= 0:
        return {
            "bin_id": bin_id,
            "target_fullness": target_fullness,
            "current_fullness": current_fullness,
            "predicted_full_at": None,
            "hours_to_target": None,
            "fill_rate_per_hour": rate,
            "confidence_low_hours": None,
            "confidence_high_hours": None,
            "confidence_score": 0.15 if rate is not None else 0.0,
            "anomalies": anomalies + ["not_filling"],
        }

    remaining = max(target_fullness - current_fullness, 0.0)
    hours = remaining / rate if rate > 0 else None
    if hours is None:
        predicted = None
    else:
        predicted = datetime.now(tz=UTC) + timedelta(hours=float(hours))

    sigma = np.sqrt(max(variance, 0.0)) if variance is not None else 0.0
    uncertainty = min(12.0, max(0.5, sigma * 2.0))
    confidence_low = max((hours or 0) - uncertainty, 0.0) if hours is not None else None
    confidence_high = (hours or 0) + uncertainty if hours is not None else None
    confidence_score = float(max(0.1, min(1.0, 1 / (1 + sigma))))

    return {
        "bin_id": bin_id,
        "target_fullness": target_fullness,
        "current_fullness": round(current_fullness, 2),
        "predicted_full_at": predicted,
        "hours_to_target": round(hours, 2) if hours is not None else None,
        "fill_rate_per_hour": round(rate, 3),
        "confidence_low_hours": round(confidence_low, 2) if confidence_low is not None else None,
        "confidence_high_hours": round(confidence_high, 2) if confidence_high is not None else None,
        "confidence_score": round(confidence_score, 3),
        "anomalies": anomalies,
    }


def optimize_schedule(
    predictions: list[dict[str, Any]],
    *,
    bin_locations: dict[str, str],
) -> list[dict[str, Any]]:
    now = datetime.now(tz=UTC)

    candidates: list[dict[str, Any]] = []
    for item in predictions:
        predicted = item.get("predicted_full_at")
        if predicted is None:
            continue

        if isinstance(predicted, str):
            predicted_dt = _parse_ts(predicted)
        else:
            predicted_dt = predicted

        candidates.append(
            {
                "bin_id": item["bin_id"],
                "location": bin_locations.get(item["bin_id"], "unknown"),
                "predicted_full_at": predicted_dt,
                "hours_to_target": (predicted_dt - now).total_seconds() / 3600.0,
            }
        )

    candidates.sort(key=lambda row: row["predicted_full_at"])

    schedule: list[dict[str, Any]] = []
    for idx, item in enumerate(candidates, start=1):
        ts = item["predicted_full_at"]
        window_start = ts - timedelta(minutes=30)
        window_end = ts + timedelta(minutes=30)
        schedule.append(
            {
                "bin_id": item["bin_id"],
                "location": item["location"],
                "priority": idx,
                "predicted_full_at": ts,
                "eta_window": f"{window_start.strftime('%Y-%m-%d %H:%M')} - {window_end.strftime('%H:%M')}",
            }
        )

    return schedule
