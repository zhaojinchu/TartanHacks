from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from src.config import AppConfig, BinDefinition
from src.models.database import DatabaseManager
from src.models.schemas import BinMeasurement
from src.sensors.ultrasonic import SensorReadError, UltrasonicReader, build_reader

LOGGER = logging.getLogger(__name__)


def calculate_fullness(distance_cm: float, bin_height_cm: float, sensor_offset_cm: float = 0.0) -> float:
    trash_height = (sensor_offset_cm + bin_height_cm) - distance_cm
    fullness = (trash_height / bin_height_cm) * 100
    return round(max(0.0, min(100.0, fullness)), 2)


def resolve_status(fullness_percent: float | None, *, normal_max: float, warning_max: float, full_min: float) -> str:
    _ = full_min
    if fullness_percent is None:
        return "sensor_error"
    if fullness_percent >= warning_max:
        return "full"
    if fullness_percent >= normal_max:
        return "almost_full"
    return "normal"


class DataCollector:
    def __init__(
        self,
        config: AppConfig,
        db: DatabaseManager,
        *,
        on_measurement: Callable[[BinMeasurement], asyncio.Future[None] | None] | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.on_measurement = on_measurement
        self._running = False
        self._readers: dict[str, UltrasonicReader] = {
            item.id: build_reader(item, mock_mode=self.config.sensors.mock_mode)
            for item in self.config.bins
        }

    async def collect_once(self) -> list[BinMeasurement]:
        results: list[BinMeasurement] = []
        for bin_config in self.config.bins:
            measurement = await self._read_bin(bin_config)
            self.db.insert_measurement(measurement)
            results.append(measurement)

            if self.on_measurement is not None:
                maybe_awaitable = self.on_measurement(measurement)
                if asyncio.iscoroutine(maybe_awaitable):
                    await maybe_awaitable
        return results

    async def _read_bin(self, bin_config: BinDefinition) -> BinMeasurement:
        reader = self._readers[bin_config.id]
        timestamp = datetime.now(tz=UTC)

        try:
            distance = await asyncio.to_thread(
                reader.read_distance,
                self.config.measurement.samples_per_read,
                self.config.measurement.sensor_timeout_seconds,
            )
            fullness = calculate_fullness(distance, bin_config.height_cm, bin_config.sensor_offset_cm)
        except SensorReadError as exc:
            LOGGER.warning("Sensor read failed for %s: %s", bin_config.id, exc)
            distance = None
            fullness = None
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Unexpected sensor failure for %s: %s", bin_config.id, exc)
            distance = None
            fullness = None

        status = resolve_status(
            fullness,
            normal_max=self.config.thresholds.normal_max,
            warning_max=self.config.thresholds.warning_max,
            full_min=self.config.thresholds.full_min,
        )

        return BinMeasurement(
            bin_id=bin_config.id,
            bin_type=bin_config.type,
            timestamp=timestamp,
            distance_cm=distance,
            fullness_percent=fullness,
            bin_height_cm=bin_config.height_cm,
            status=status,
            location=bin_config.location,
        )

    async def run_forever(self) -> None:
        self._running = True
        LOGGER.info("Data collector started with %d bins", len(self.config.bins))

        while self._running:
            await self.collect_once()
            await asyncio.sleep(self.config.measurement.interval_seconds)

    async def stop(self) -> None:
        self._running = False
        for reader in self._readers.values():
            await asyncio.to_thread(reader.close)
