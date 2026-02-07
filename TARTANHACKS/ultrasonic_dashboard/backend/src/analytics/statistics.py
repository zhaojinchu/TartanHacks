from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def calculate_fill_cycles(rows: list[dict[str, Any]], *, target_fullness: float = 90.0) -> list[float]:
    if not rows:
        return []

    points = sorted(rows, key=lambda item: item["timestamp"])
    start_ts: datetime | None = None
    previous_fullness: float | None = None
    target_crossed = False
    cycle_hours: list[float] = []

    for row in points:
        fullness = row.get("fullness_percent")
        if fullness is None:
            continue
        ts = _parse_ts(row["timestamp"])

        if start_ts is None:
            start_ts = ts
        elif previous_fullness is not None and fullness < previous_fullness - 20:
            start_ts = ts
            target_crossed = False

        crossed_up = (
            previous_fullness is not None
            and previous_fullness < target_fullness
            and fullness >= target_fullness
        )
        if start_ts is not None and not target_crossed and crossed_up:
            hours = max((ts - start_ts).total_seconds() / 3600.0, 0.0)
            if hours >= 0.25:
                cycle_hours.append(hours)
            target_crossed = True

        if fullness <= target_fullness - 15:
            target_crossed = False

        previous_fullness = fullness

    return cycle_hours


def average_fill_time_by_group(
    rows: list[dict[str, Any]],
    *,
    group_field: str,
    target_fullness: float = 90.0,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_field, "unknown"))
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for key, items in grouped.items():
        cycles = calculate_fill_cycles(items, target_fullness=target_fullness)
        avg = round(sum(cycles) / len(cycles), 2) if cycles else None
        output.append(
            {
                "group_key": key,
                "count_cycles": len(cycles),
                "average_hours_to_target": avg,
            }
        )

    return sorted(output, key=lambda item: item["group_key"])


def fill_rate_trends(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ts = _parse_ts(row["timestamp"])
        day_key = ts.strftime("%Y-%m-%d")
        buckets[(row["bin_id"], day_key)].append(row)

    trends: list[dict[str, Any]] = []
    for (bin_id, day_key), items in sorted(buckets.items()):
        ordered = sorted(items, key=lambda item: item["timestamp"])
        first = next((item for item in ordered if item.get("fullness_percent") is not None), None)
        last = next((item for item in reversed(ordered) if item.get("fullness_percent") is not None), None)
        if not first or not last:
            continue

        start = _parse_ts(first["timestamp"])
        end = _parse_ts(last["timestamp"])
        elapsed_hours = max((end - start).total_seconds() / 3600.0, 0.01)
        delta = float(last["fullness_percent"]) - float(first["fullness_percent"])
        rate = round(max(delta, 0.0) / elapsed_hours, 3)

        trends.append(
            {
                "bin_id": bin_id,
                "timestamp": datetime.fromisoformat(f"{day_key}T00:00:00+00:00").isoformat(),
                "fill_rate_per_hour": rate,
            }
        )

    return trends


def temporal_heatmap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_totals: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bin[str(row["bin_id"])].append(row)

    for items in by_bin.values():
        ordered = sorted(items, key=lambda item: item["timestamp"])
        for prev, curr in zip(ordered, ordered[1:], strict=False):
            p_value = prev.get("fullness_percent")
            c_value = curr.get("fullness_percent")
            if p_value is None or c_value is None:
                continue
            delta = float(c_value) - float(p_value)
            if delta <= 0:
                continue

            ts = _parse_ts(curr["timestamp"])
            key_totals[(ts.weekday(), ts.hour)].append(delta)

    cells: list[dict[str, Any]] = []
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for day in range(7):
        for hour in range(24):
            deltas = key_totals.get((day, hour), [])
            value = round(sum(deltas) / len(deltas), 3) if deltas else 0.0
            cells.append({"x": str(hour), "y": weekday_labels[day], "value": value})
    return cells


def location_heatmap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        fullness = row.get("fullness_percent")
        if fullness is None:
            continue
        grouped[str(row.get("location", "unknown"))].append(float(fullness))

    if not grouped:
        return []

    maximum = max((max(values) for values in grouped.values()), default=1.0)
    output: list[dict[str, Any]] = []
    for location, values in sorted(grouped.items()):
        avg = sum(values) / len(values)
        intensity = round((avg / max(maximum, 1.0)) * 100.0, 2)
        output.append({"x": location, "y": "fullness", "value": intensity})
    return output
