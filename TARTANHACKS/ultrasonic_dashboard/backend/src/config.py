from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

BinType = Literal["recycle", "compost", "landfill"]
SensorMode = Literal["mock", "gpio", "serial"]


@dataclass(slots=True)
class BinDefinition:
    id: str
    type: BinType
    location: str
    height_cm: float
    sensor_offset_cm: float
    gpio_trigger: int
    gpio_echo: int
    sensor_channel: int | None = None


@dataclass(slots=True)
class ThresholdConfig:
    normal_max: float = 70.0
    warning_max: float = 90.0
    full_min: float = 95.0
    alert_critical: float = 90.0


@dataclass(slots=True)
class MeasurementConfig:
    interval_seconds: int = 10
    samples_per_read: int = 3
    sensor_timeout_seconds: float = 0.04
    auto_start_collector: bool = True


@dataclass(slots=True)
class SensorRuntimeConfig:
    mode: SensorMode = "mock"
    cleanup_gpio_on_exit: bool = True
    serial_port: str = "/dev/ttyACM0"
    serial_baudrate: int = 9600
    serial_timeout_seconds: float = 0.2
    serial_stale_seconds: float = 5.0
    serial_startup_delay_seconds: float = 2.0


@dataclass(slots=True)
class DatabaseConfig:
    path: Path


@dataclass(slots=True)
class DashboardConfig:
    refresh_hint_seconds: int = 2


@dataclass(slots=True)
class AppConfig:
    bins: list[BinDefinition]
    thresholds: ThresholdConfig
    measurement: MeasurementConfig
    sensors: SensorRuntimeConfig
    database: DatabaseConfig
    dashboard: DashboardConfig


class ConfigError(RuntimeError):
    pass


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected mapping in {path}")
    return data


def _resolve_config_dir() -> Path:
    env_dir = os.getenv("ULTRASONIC_CONFIG_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "config").resolve()


def _parse_bins(raw_bins: list[dict[str, Any]]) -> list[BinDefinition]:
    bins: list[BinDefinition] = []
    for item in raw_bins:
        bins.append(
            BinDefinition(
                id=str(item["id"]),
                type=str(item["type"]),  # type: ignore[arg-type]
                location=str(item.get("location", "unknown")),
                height_cm=float(item["height_cm"]),
                sensor_offset_cm=float(item.get("sensor_offset_cm", 0.0)),
                gpio_trigger=int(item["gpio_trigger"]),
                gpio_echo=int(item["gpio_echo"]),
                sensor_channel=int(item["sensor_channel"]) if item.get("sensor_channel") is not None else None,
            )
        )
    return bins


def load_config(config_dir: Path | None = None) -> AppConfig:
    directory = config_dir or _resolve_config_dir()
    if not directory.exists():
        raise ConfigError(f"Config directory not found: {directory}")

    bins_cfg = _read_yaml(directory / "bins.yaml")
    thresholds_cfg = _read_yaml(directory / "thresholds.yaml")
    sensors_cfg = _read_yaml(directory / "sensors.yaml")

    bins_raw = bins_cfg.get("bins", [])
    if not bins_raw:
        raise ConfigError("No bins configured in bins.yaml")

    bins = _parse_bins(bins_raw)
    thresholds = ThresholdConfig(
        normal_max=float(thresholds_cfg.get("normal_max", 70)),
        warning_max=float(thresholds_cfg.get("warning_max", 90)),
        full_min=float(thresholds_cfg.get("full_min", 95)),
        alert_critical=float(thresholds_cfg.get("alert_critical", 90)),
    )

    measurement_raw = bins_cfg.get("measurement", {})
    measurement = MeasurementConfig(
        interval_seconds=int(measurement_raw.get("interval_seconds", 10)),
        samples_per_read=int(measurement_raw.get("samples_per_read", 3)),
        sensor_timeout_seconds=float(measurement_raw.get("sensor_timeout_seconds", 0.04)),
        auto_start_collector=bool(measurement_raw.get("auto_start_collector", True)),
    )

    mode_raw = str(sensors_cfg.get("mode", "")).strip().lower()
    if not mode_raw:
        mode_raw = "mock" if bool(sensors_cfg.get("mock_mode", True)) else "gpio"
    if mode_raw not in {"mock", "gpio", "serial"}:
        raise ConfigError(f"Invalid sensors.mode `{mode_raw}`. Use mock|gpio|serial.")

    serial_cfg = sensors_cfg.get("serial", {})
    if not isinstance(serial_cfg, dict):
        raise ConfigError("sensors.serial must be a dictionary")

    sensor_runtime = SensorRuntimeConfig(
        mode=mode_raw,  # type: ignore[arg-type]
        cleanup_gpio_on_exit=bool(sensors_cfg.get("cleanup_gpio_on_exit", True)),
        serial_port=str(serial_cfg.get("port", "/dev/ttyACM0")),
        serial_baudrate=int(serial_cfg.get("baudrate", 9600)),
        serial_timeout_seconds=float(serial_cfg.get("timeout_seconds", 0.2)),
        serial_stale_seconds=float(serial_cfg.get("stale_seconds", 5.0)),
        serial_startup_delay_seconds=float(serial_cfg.get("startup_delay_seconds", 2.0)),
    )

    db_path_raw = sensors_cfg.get("database", {}).get("path", "./data/ultrasonic.db")
    db_path = Path(db_path_raw)
    if not db_path.is_absolute():
        db_path = (Path(__file__).resolve().parents[2] / db_path).resolve()

    dashboard = DashboardConfig(
        refresh_hint_seconds=int(sensors_cfg.get("dashboard", {}).get("refresh_hint_seconds", 2))
    )

    return AppConfig(
        bins=bins,
        thresholds=thresholds,
        measurement=measurement,
        sensors=sensor_runtime,
        database=DatabaseConfig(path=db_path),
        dashboard=dashboard,
    )
