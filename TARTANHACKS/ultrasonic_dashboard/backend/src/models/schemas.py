from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

BinStatusLiteral = Literal["normal", "almost_full", "full", "sensor_error"]
BinTypeLiteral = Literal["recycle", "compost", "landfill"]


class BinMeasurement(BaseModel):
    bin_id: str
    bin_type: BinTypeLiteral
    timestamp: datetime
    distance_cm: float | None = None
    fullness_percent: float | None = None
    bin_height_cm: float
    status: BinStatusLiteral
    location: str


class BinDetailResponse(BaseModel):
    bin: BinMeasurement
    alerts_active: bool


class BinHistoryResponse(BaseModel):
    bin_id: str
    count: int
    items: list[BinMeasurement]


class FillTimeStat(BaseModel):
    group_key: str
    count_cycles: int
    average_hours_to_85: float | None


class FillRateTrendPoint(BaseModel):
    bin_id: str
    timestamp: datetime
    fill_rate_per_hour: float


class HeatmapCell(BaseModel):
    x: str
    y: str
    value: float


class BinPrediction(BaseModel):
    bin_id: str
    target_fullness: float = Field(default=85.0)
    current_fullness: float | None
    predicted_full_at: datetime | None
    hours_to_target: float | None
    fill_rate_per_hour: float | None
    confidence_low_hours: float | None
    confidence_high_hours: float | None
    confidence_score: float
    anomalies: list[str] = Field(default_factory=list)


class ScheduleItem(BaseModel):
    bin_id: str
    location: str
    priority: int
    predicted_full_at: datetime | None
    eta_window: str


class ScheduleResponse(BaseModel):
    generated_at: datetime
    target_fullness: float
    route: list[ScheduleItem]


class EmptyBinRequest(BaseModel):
    reason: str = "manual_override"
