from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as api_router
from src.api.websocket import ConnectionManager, router as websocket_router
from src.config import load_config
from src.models.database import DatabaseManager
from src.sensors.data_collector import DataCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    db = DatabaseManager(config.database.path)
    db.initialize()

    ws_manager = ConnectionManager()

    async def publish(measurement):
        await ws_manager.broadcast({"type": "measurement", "data": measurement.model_dump(mode="json")})

    collector = DataCollector(config, db, on_measurement=publish)
    collector_task: asyncio.Task[Any] | None = None

    if config.measurement.auto_start_collector:
        collector_task = asyncio.create_task(collector.run_forever())

    app.state.config = config
    app.state.db = db
    app.state.ws_manager = ws_manager
    app.state.collector = collector
    app.state.collector_task = collector_task

    yield

    if collector_task:
        collector_task.cancel()
        with suppress(asyncio.CancelledError):
            await collector_task

    await collector.stop()
    db.close()


app = FastAPI(
    title="Ultrasonic Bin Monitoring API",
    version="1.0.0",
    description="Real-time monitoring and analytics for waste bin fullness",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(websocket_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
