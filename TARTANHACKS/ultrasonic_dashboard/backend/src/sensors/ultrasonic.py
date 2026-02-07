from __future__ import annotations

import logging
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from src.config import BinDefinition, SensorRuntimeConfig

try:
    import RPi.GPIO as GPIO  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - GPIO unavailable in non-RPi env
    GPIO = None

try:
    import serial  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - pyserial optional in non-serial mode
    serial = None

LOGGER = logging.getLogger(__name__)


class SensorReadError(RuntimeError):
    pass


class UltrasonicReader:
    def read_distance(self, samples: int = 3, timeout_s: float = 0.04) -> float:
        raise NotImplementedError

    def close(self) -> None:
        return


class SerialLineParser:
    _PATTERNS = [
        re.compile(r"Ultrasonic\s+Sensor\s+(\d+)\s+Distance:\s*([-+]?\d*\.?\d+)", re.IGNORECASE),
        re.compile(r"S(\d+)\s*:\s*([-+]?\d*\.?\d+)", re.IGNORECASE),
        re.compile(r"(\d+)\s*,\s*([-+]?\d*\.?\d+)")
    ]

    @classmethod
    def parse(cls, line: str) -> tuple[int, float] | None:
        text = line.strip()
        if not text:
            return None

        for pattern in cls._PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            channel = int(match.group(1))
            distance_cm = float(match.group(2))
            return channel, distance_cm
        return None


class SerialUltrasonicHub:
    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        timeout_s: float,
        stale_s: float,
        startup_delay_s: float,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._timeout_s = timeout_s
        self._stale_s = stale_s
        self._startup_delay_s = startup_delay_s

        self._samples: dict[int, deque[tuple[float, float]]] = {}
        self._samples_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._serial: Any | None = None
        self._connected_once = False

        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="serial-ultrasonic-hub")
        self._thread.start()

    def _connect(self) -> None:
        if serial is None:
            raise SensorReadError("pyserial is not installed. Install `pyserial` for serial sensor mode.")
        if self._serial is not None:
            return

        self._serial = serial.Serial(
            port=self._port,
            baudrate=self._baudrate,
            timeout=self._timeout_s,
        )
        LOGGER.info("Connected to Arduino serial %s @ %s baud", self._port, self._baudrate)

        if not self._connected_once and self._startup_delay_s > 0:
            # Many Arduino boards reset when a serial connection is opened.
            time.sleep(self._startup_delay_s)
            self._connected_once = True

    def _disconnect(self) -> None:
        if self._serial is None:
            return
        try:
            self._serial.close()
        except Exception:
            pass
        self._serial = None

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._serial is None:
                    self._connect()

                line_raw = self._serial.readline()
                if not line_raw:
                    continue

                line = line_raw.decode("utf-8", errors="ignore").strip()
                parsed = SerialLineParser.parse(line)
                if not parsed:
                    continue

                channel, distance = parsed
                now = time.monotonic()
                with self._samples_lock:
                    history = self._samples.setdefault(channel, deque(maxlen=32))
                    history.append((now, distance))
            except Exception as exc:
                LOGGER.warning("Serial read error (%s). Reconnecting in 1s...", exc)
                self._disconnect()
                time.sleep(1.0)

    def read_distance(self, *, sensor_channel: int, samples: int, timeout_s: float) -> float:
        sample_count = max(samples, 1)
        deadline = time.monotonic() + max(timeout_s, self._timeout_s, 0.25)

        while time.monotonic() < deadline:
            with self._samples_lock:
                history = list(self._samples.get(sensor_channel, []))

            if history:
                now = time.monotonic()
                recent_values = [value for ts, value in history if now - ts <= self._stale_s]
                if recent_values:
                    recent_values = recent_values[-sample_count:]
                    recent_values.sort()
                    return float(recent_values[len(recent_values) // 2])

            time.sleep(0.02)

        raise SensorReadError(
            f"No fresh serial reading for sensor channel {sensor_channel} on {self._port}"
        )

    def write_command(self, command: str) -> None:
        cmd = command.strip()
        if not cmd:
            raise SensorReadError("Serial command is empty")
        if self._serial is None:
            self._connect()
        assert self._serial is not None

        with self._write_lock:
            self._serial.write(f"{cmd}\n".encode("utf-8"))
            self._serial.flush()

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._disconnect()


@dataclass(slots=True)
class _HubRef:
    hub: SerialUltrasonicHub
    refs: int


_HUB_POOL_LOCK = threading.Lock()
_HUB_POOL: dict[tuple[str, int], _HubRef] = {}


def _acquire_shared_hub(config: SensorRuntimeConfig) -> tuple[tuple[str, int], SerialUltrasonicHub]:
    key = (config.serial_port, config.serial_baudrate)
    with _HUB_POOL_LOCK:
        if key in _HUB_POOL:
            ref = _HUB_POOL[key]
            ref.refs += 1
            return key, ref.hub

        hub = SerialUltrasonicHub(
            port=config.serial_port,
            baudrate=config.serial_baudrate,
            timeout_s=config.serial_timeout_seconds,
            stale_s=config.serial_stale_seconds,
            startup_delay_s=config.serial_startup_delay_seconds,
        )
        _HUB_POOL[key] = _HubRef(hub=hub, refs=1)
        return key, hub


def _release_shared_hub(key: tuple[str, int]) -> None:
    with _HUB_POOL_LOCK:
        ref = _HUB_POOL.get(key)
        if ref is None:
            return
        ref.refs -= 1
        if ref.refs > 0:
            return
        hub = ref.hub
        del _HUB_POOL[key]
    hub.close()


@dataclass(slots=True)
class GPIOUltrasonicReader(UltrasonicReader):
    trigger_pin: int
    echo_pin: int

    def __post_init__(self) -> None:
        if GPIO is None:
            raise SensorReadError("RPi.GPIO is not available. Use `mode: mock` for development.")

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


class SerialUltrasonicReader(UltrasonicReader):
    def __init__(self, *, sensor_channel: int, runtime: SensorRuntimeConfig):
        self.sensor_channel = sensor_channel
        self._key, self._hub = _acquire_shared_hub(runtime)

    def read_distance(self, samples: int = 3, timeout_s: float = 0.04) -> float:
        return self._hub.read_distance(
            sensor_channel=self.sensor_channel,
            samples=samples,
            timeout_s=max(timeout_s, 0.5),
        )

    def send_command(self, command: str) -> None:
        self._hub.write_command(command)

    def close(self) -> None:
        _release_shared_hub(self._key)


def build_reader(bin_config: BinDefinition, runtime: SensorRuntimeConfig) -> UltrasonicReader:
    if runtime.mode == "mock":
        return MockUltrasonicReader(bin_config)
    if runtime.mode == "gpio":
        return GPIOUltrasonicReader(bin_config.gpio_trigger, bin_config.gpio_echo)
    if runtime.mode == "serial":
        if bin_config.sensor_channel is None:
            raise SensorReadError(
                f"Bin `{bin_config.id}` missing `sensor_channel` for serial mode."
            )
        return SerialUltrasonicReader(sensor_channel=bin_config.sensor_channel, runtime=runtime)
    raise SensorReadError(f"Unsupported sensor mode: {runtime.mode}")
