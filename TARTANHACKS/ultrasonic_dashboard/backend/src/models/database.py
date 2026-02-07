from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.models.schemas import BinMeasurement


class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def initialize(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bin_id TEXT NOT NULL,
                    bin_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    distance_cm REAL,
                    fullness_percent REAL,
                    bin_height_cm REAL NOT NULL,
                    status TEXT NOT NULL,
                    location TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'sensor'
                );

                CREATE INDEX IF NOT EXISTS idx_measurements_bin_time
                    ON measurements(bin_id, timestamp DESC);

                CREATE TABLE IF NOT EXISTS empty_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bin_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    reason TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bin_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    target_fullness REAL NOT NULL,
                    predicted_full_at TEXT,
                    confidence_low_hours REAL,
                    confidence_high_hours REAL
                );
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def insert_measurement(self, measurement: BinMeasurement, source: str = "sensor") -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO measurements (
                    bin_id, bin_type, timestamp, distance_cm, fullness_percent,
                    bin_height_cm, status, location, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    measurement.bin_id,
                    measurement.bin_type,
                    measurement.timestamp.astimezone(UTC).isoformat(),
                    measurement.distance_cm,
                    measurement.fullness_percent,
                    measurement.bin_height_cm,
                    measurement.status,
                    measurement.location,
                    source,
                ),
            )
            self._conn.commit()

    def log_prediction(
        self,
        *,
        bin_id: str,
        target_fullness: float,
        predicted_full_at: datetime | None,
        confidence_low_hours: float | None,
        confidence_high_hours: float | None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO prediction_logs (
                    bin_id, timestamp, target_fullness, predicted_full_at,
                    confidence_low_hours, confidence_high_hours
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    bin_id,
                    datetime.now(tz=UTC).isoformat(),
                    target_fullness,
                    predicted_full_at.astimezone(UTC).isoformat() if predicted_full_at else None,
                    confidence_low_hours,
                    confidence_high_hours,
                ),
            )
            self._conn.commit()

    def add_empty_event(self, *, bin_id: str, reason: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO empty_events (bin_id, timestamp, reason) VALUES (?, ?, ?)",
                (bin_id, datetime.now(tz=UTC).isoformat(), reason),
            )
            self._conn.commit()

    def get_latest_by_bin(self) -> dict[str, dict[str, Any]]:
        query = """
            SELECT m.*
            FROM measurements m
            INNER JOIN (
                SELECT bin_id, MAX(timestamp) AS max_timestamp
                FROM measurements
                GROUP BY bin_id
            ) latest
            ON m.bin_id = latest.bin_id AND m.timestamp = latest.max_timestamp
        """
        with self._lock:
            rows = self._conn.execute(query).fetchall()
        return {str(row["bin_id"]): dict(row) for row in rows}

    def get_latest_for_bin(self, bin_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM measurements WHERE bin_id = ? ORDER BY timestamp DESC LIMIT 1",
                (bin_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_history(
        self,
        *,
        bin_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        clauses = ["bin_id = ?"]
        args: list[Any] = [bin_id]
        if start:
            clauses.append("timestamp >= ?")
            args.append(start.astimezone(UTC).isoformat())
        if end:
            clauses.append("timestamp <= ?")
            args.append(end.astimezone(UTC).isoformat())
        args.append(limit)

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM measurements
                WHERE {' AND '.join(clauses)}
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                args,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_history(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        bin_ids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        if start:
            clauses.append("timestamp >= ?")
            args.append(start.astimezone(UTC).isoformat())
        if end:
            clauses.append("timestamp <= ?")
            args.append(end.astimezone(UTC).isoformat())
        if bin_ids:
            ids = list(bin_ids)
            placeholders = ",".join("?" for _ in ids)
            clauses.append(f"bin_id IN ({placeholders})")
            args.extend(ids)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM measurements {where} ORDER BY timestamp ASC",
                args,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_prediction_logs(
        self,
        *,
        bin_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        where = ""
        args: list[Any] = []
        if bin_id:
            where = "WHERE bin_id = ?"
            args.append(bin_id)
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM prediction_logs
                {where}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                args,
            ).fetchall()
        return [dict(row) for row in rows]
