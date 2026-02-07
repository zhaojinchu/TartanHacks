from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from src.analytics.predictions import optimize_schedule, predict_time_to_target
from src.analytics.statistics import (
    average_fill_time_by_group,
    fill_rate_trends,
    location_heatmap,
    temporal_heatmap,
)
from src.models.schemas import BinMeasurement, EmptyBinRequest, ScheduleResponse

router = APIRouter(prefix="/api")


def _to_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _bin_lookup(request: Request) -> dict[str, Any]:
    return {item.id: item for item in request.app.state.config.bins}


def _row_to_measurement(row: dict[str, Any]) -> BinMeasurement:
    return BinMeasurement(
        bin_id=row["bin_id"],
        bin_type=row["bin_type"],
        timestamp=_to_datetime(row["timestamp"]),
        distance_cm=row.get("distance_cm"),
        fullness_percent=row.get("fullness_percent"),
        bin_height_cm=row["bin_height_cm"],
        status=row["status"],
        location=row["location"],
    )


@router.get("/bins")
async def list_bins(request: Request) -> list[dict[str, Any]]:
    db = request.app.state.db
    latest = db.get_latest_by_bin()
    thresholds = request.app.state.config.thresholds

    response: list[dict[str, Any]] = []
    for bin_cfg in request.app.state.config.bins:
        row = latest.get(bin_cfg.id)
        if row:
            payload = _row_to_measurement(row).model_dump(mode="json")
        else:
            payload = BinMeasurement(
                bin_id=bin_cfg.id,
                bin_type=bin_cfg.type,
                timestamp=datetime.now(tz=UTC),
                distance_cm=None,
                fullness_percent=None,
                bin_height_cm=bin_cfg.height_cm,
                status="sensor_error",
                location=bin_cfg.location,
            ).model_dump(mode="json")

        payload["alerts_active"] = (
            payload["fullness_percent"] is not None
            and payload["fullness_percent"] >= thresholds.alert_critical
        )
        response.append(payload)

    return response


@router.get("/bins/{bin_id}")
async def get_bin(bin_id: str, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    lookup = _bin_lookup(request)
    if bin_id not in lookup:
        raise HTTPException(status_code=404, detail="Bin not found")

    row = db.get_latest_for_bin(bin_id)
    if not row:
        raise HTTPException(status_code=404, detail="No measurements available")

    thresholds = request.app.state.config.thresholds
    measurement = _row_to_measurement(row).model_dump(mode="json")
    return {
        "bin": measurement,
        "alerts_active": (
            measurement["fullness_percent"] is not None
            and measurement["fullness_percent"] >= thresholds.alert_critical
        ),
    }


@router.get("/bins/{bin_id}/history")
async def get_history(
    bin_id: str,
    request: Request,
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict[str, Any]:
    db = request.app.state.db
    lookup = _bin_lookup(request)
    if bin_id not in lookup:
        raise HTTPException(status_code=404, detail="Bin not found")

    rows = db.get_history(bin_id=bin_id, start=_to_datetime(start), end=_to_datetime(end), limit=limit)
    measurements = [_row_to_measurement(row).model_dump(mode="json") for row in rows]
    return {"bin_id": bin_id, "count": len(measurements), "items": measurements}


@router.get("/bins/{bin_id}/prediction")
async def get_prediction(
    bin_id: str,
    request: Request,
    target_fullness: float = Query(default=85.0, ge=10, le=100),
) -> dict[str, Any]:
    db = request.app.state.db
    lookup = _bin_lookup(request)
    if bin_id not in lookup:
        raise HTTPException(status_code=404, detail="Bin not found")

    latest = db.get_latest_for_bin(bin_id)
    history = db.get_history(
        bin_id=bin_id,
        start=datetime.now(tz=UTC) - timedelta(days=7),
        limit=5000,
    )

    prediction = predict_time_to_target(
        bin_id=bin_id,
        current_fullness=latest.get("fullness_percent") if latest else None,
        rows=history,
        target_fullness=target_fullness,
    )

    db.log_prediction(
        bin_id=bin_id,
        target_fullness=target_fullness,
        predicted_full_at=prediction["predicted_full_at"],
        confidence_low_hours=prediction["confidence_low_hours"],
        confidence_high_hours=prediction["confidence_high_hours"],
    )

    return prediction


@router.get("/analytics/fill-times")
async def analytics_fill_times(
    request: Request,
    days: int = Query(default=30, ge=1, le=3650),
    target_fullness: float = Query(default=85.0, ge=10, le=100),
) -> dict[str, Any]:
    db = request.app.state.db
    start = datetime.now(tz=UTC) - timedelta(days=days)
    rows = db.get_all_history(start=start)

    return {
        "by_bin_type": average_fill_time_by_group(rows, group_field="bin_type", target_fullness=target_fullness),
        "by_location": average_fill_time_by_group(rows, group_field="location", target_fullness=target_fullness),
        "window_days": days,
    }


@router.get("/analytics/fill-rate-trends")
async def analytics_fill_rate_trends(
    request: Request,
    days: int = Query(default=30, ge=1, le=3650),
) -> list[dict[str, Any]]:
    db = request.app.state.db
    start = datetime.now(tz=UTC) - timedelta(days=days)
    rows = db.get_all_history(start=start)
    return fill_rate_trends(rows)


@router.get("/analytics/heatmap")
async def analytics_heatmap(
    request: Request,
    days: int = Query(default=30, ge=1, le=3650),
    mode: str = Query(default="temporal", pattern="^(temporal|location)$"),
) -> dict[str, Any]:
    db = request.app.state.db
    start = datetime.now(tz=UTC) - timedelta(days=days)
    rows = db.get_all_history(start=start)

    data = temporal_heatmap(rows) if mode == "temporal" else location_heatmap(rows)
    return {"mode": mode, "window_days": days, "cells": data}


@router.get("/schedule/optimize", response_model=ScheduleResponse)
async def schedule_optimize(
    request: Request,
    target_fullness: float = Query(default=85.0, ge=10, le=100),
) -> ScheduleResponse:
    db = request.app.state.db
    bins = request.app.state.config.bins

    predictions: list[dict[str, Any]] = []
    for bin_cfg in bins:
        latest = db.get_latest_for_bin(bin_cfg.id)
        history = db.get_history(
            bin_id=bin_cfg.id,
            start=datetime.now(tz=UTC) - timedelta(days=7),
            limit=5000,
        )
        prediction = predict_time_to_target(
            bin_id=bin_cfg.id,
            current_fullness=latest.get("fullness_percent") if latest else None,
            rows=history,
            target_fullness=target_fullness,
        )
        predictions.append(prediction)

    route = optimize_schedule(
        predictions,
        bin_locations={item.id: item.location for item in bins},
    )
    return ScheduleResponse(
        generated_at=datetime.now(tz=UTC),
        target_fullness=target_fullness,
        route=route,
    )


@router.get("/analytics/prediction-accuracy")
async def prediction_accuracy(
    request: Request,
    bin_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    db = request.app.state.db
    return db.get_prediction_logs(bin_id=bin_id, limit=limit)


@router.post("/bins/{bin_id}/empty")
async def mark_empty(bin_id: str, request: Request, payload: EmptyBinRequest) -> dict[str, Any]:
    db = request.app.state.db
    lookup = _bin_lookup(request)
    bin_cfg = lookup.get(bin_id)
    if not bin_cfg:
        raise HTTPException(status_code=404, detail="Bin not found")

    measurement = BinMeasurement(
        bin_id=bin_cfg.id,
        bin_type=bin_cfg.type,
        timestamp=datetime.now(tz=UTC),
        distance_cm=round(bin_cfg.height_cm + bin_cfg.sensor_offset_cm, 2),
        fullness_percent=0.0,
        bin_height_cm=bin_cfg.height_cm,
        status="normal",
        location=bin_cfg.location,
    )

    db.insert_measurement(measurement, source="manual_empty")
    db.add_empty_event(bin_id=bin_id, reason=payload.reason)

    manager = request.app.state.ws_manager
    await manager.broadcast({"type": "measurement", "data": measurement.model_dump(mode="json")})

    return {"ok": True, "bin_id": bin_id, "reason": payload.reason}
