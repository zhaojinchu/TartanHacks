from __future__ import annotations

import random
import time
from dataclasses import dataclass

from src.config import BinDefinition

try:
    import RPi.GPIO as GPIO  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - GPIO unavailable in non-RPi env
    GPIO = None


class SensorReadError(RuntimeError):
    pass


class UltrasonicReader:
    def read_distance(self, samples: int = 3, timeout_s: float = 0.04) -> float:
        raise NotImplementedError

    def close(self) -> None:
        return


@dataclass(slots=True)
class GPIOUltrasonicReader(UltrasonicReader):
    trigger_pin: int
    echo_pin: int

    def __post_init__(self) -> None:
        if GPIO is None:
            raise SensorReadError("RPi.GPIO is not available. Enable mock_mode for development.")

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trigger_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        GPIO.output(self.trigger_pin, GPIO.LOW)
        time.sleep(0.05)

    def _single_read(self, timeout_s: float) -> float:
        assert GPIO is not None

        GPIO.output(self.trigger_pin, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trigger_pin, GPIO.LOW)

        start_wait = time.monotonic()
        while GPIO.input(self.echo_pin) == 0:
            if time.monotonic() - start_wait > timeout_s:
                raise SensorReadError("Timeout waiting for echo start")
        pulse_start = time.monotonic()

        while GPIO.input(self.echo_pin) == 1:
            if time.monotonic() - pulse_start > timeout_s:
                raise SensorReadError("Timeout waiting for echo end")
        pulse_end = time.monotonic()

        pulse_duration = pulse_end - pulse_start
        return round(pulse_duration * 17150, 2)

    def read_distance(self, samples: int = 3, timeout_s: float = 0.04) -> float:
        readings: list[float] = []
        for _ in range(max(samples, 1)):
            readings.append(self._single_read(timeout_s))
            time.sleep(0.03)

        if not readings:
            raise SensorReadError("No sensor readings returned")
        readings.sort()
        return readings[len(readings) // 2]

    def close(self) -> None:
        if GPIO is not None:
            GPIO.cleanup([self.trigger_pin])


class MockUltrasonicReader(UltrasonicReader):
    def __init__(self, bin_config: BinDefinition):
        self.bin = bin_config
        self._rng = random.Random(bin_config.id)
        self._fullness = self._rng.uniform(5.0, 55.0)
        self._ticks = 0

    def read_distance(self, samples: int = 3, timeout_s: float = 0.04) -> float:
        _ = samples, timeout_s
        self._ticks += 1

        if self._ticks % self._rng.randint(140, 220) == 0:
            self._fullness = self._rng.uniform(2.0, 15.0)
        else:
            delta = self._rng.uniform(0.1, 1.8)
            self._fullness = min(99.0, self._fullness + delta)

        noise = self._rng.uniform(-0.8, 0.8)
        trash_height = (self._fullness / 100.0) * self.bin.height_cm
        distance = self.bin.sensor_offset_cm + self.bin.height_cm - trash_height + noise
        return round(max(0.0, distance), 2)


def build_reader(bin_config: BinDefinition, mock_mode: bool) -> UltrasonicReader:
    if mock_mode:
        return MockUltrasonicReader(bin_config)
    return GPIOUltrasonicReader(bin_config.gpio_trigger, bin_config.gpio_echo)
