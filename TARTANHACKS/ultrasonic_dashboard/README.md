# Ultrasonic Bin Monitoring Dashboard

A complete FastAPI + React module for real-time bin fullness monitoring with HC-SR04 ultrasonic sensors.

## Location
This implementation is in:
- `TARTANHACKS/ultrasonic_dashboard`

## Features
- Real-time sensor collection from multiple bins (serial, GPIO, or mock mode)
- Fullness calculations with configurable bin dimensions
- SQLite time-series storage
- REST API + WebSocket streaming updates
- Dashboard pages:
  - Real-time monitoring
  - Usage analytics
  - Temporal/location heatmaps
  - Predictive scheduling + route export
- Manual empty override endpoint
- Prediction logs for historical tracking
- Docker Compose deployment
- Raspberry Pi systemd service template

## Folder Structure
```text
TARTANHACKS/ultrasonic_dashboard/
  backend/
    src/
      api/
      sensors/
      models/
      analytics/
      config.py
      main.py
    tests/
    requirements.txt
    Dockerfile
  frontend/
    src/
      components/
      pages/
      services/
    package.json
    Dockerfile
  config/
    bins.yaml
    sensors.yaml
    thresholds.yaml
  scripts/
    setup_sensors.py
    calibrate_bins.py
    seed_data.py
  systemd/
    ultrasonic-dashboard.service
  docker-compose.yml
```

## Quick Start (Local)

### 1. Backend
```bash
cd TARTANHACKS/ultrasonic_dashboard/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API (collector auto-starts):
```bash
ULTRASONIC_CONFIG_DIR=../config uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend
```bash
cd TARTANHACKS/ultrasonic_dashboard/frontend
npm install
npm run dev
```

Dashboard URL:
- `http://localhost:3000`

API docs:
- `http://localhost:8000/docs`

## Docker Deployment
```bash
cd TARTANHACKS/ultrasonic_dashboard
docker compose up --build
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

## Configuration
Edit YAML files in `config/`.

### `config/bins.yaml`
Defines bins and collection timing.

### `config/sensors.yaml`
Runtime behavior and DB path.
- Set `mode: serial` for Arduino-over-USB, `mode: gpio` for direct Pi GPIO, or `mode: mock` for development.

### `config/thresholds.yaml`
Status and alert thresholds.

## API Endpoints
- `GET /api/bins`
- `GET /api/bins/{id}`
- `GET /api/bins/{id}/history`
- `GET /api/bins/{id}/prediction`
- `POST /api/bins/{id}/empty`
- `GET /api/analytics/fill-times`
- `GET /api/analytics/fill-rate-trends`
- `GET /api/analytics/heatmap?mode=temporal|location`
- `GET /api/analytics/prediction-accuracy`
- `GET /api/schedule/optimize`
- `POST /api/arduino/command` (payload `{"command":"O0"}` etc.)
- `WS /ws/bins`

## Scripts
Run from `TARTANHACKS/ultrasonic_dashboard/`:

Test sensor connectivity:
```bash
python scripts/setup_sensors.py
```

Seed historical mock data:
```bash
python scripts/seed_data.py --days 30 --step-minutes 30
```

Calibration helper:
```bash
python scripts/calibrate_bins.py
```

## Raspberry Pi Notes
- HC-SR04 Echo pin is 5V; use a resistor divider before Pi GPIO input.
- For serial mode, wire sensors to Arduino and connect Arduino USB to Pi (`/dev/ttyACM0` by default).
- Set `mode: serial` in `config/sensors.yaml` and map each bin `sensor_channel` in `config/bins.yaml`.
- If using direct Pi GPIO instead, set `mode: gpio` in `config/sensors.yaml`.
- With verbose Arduino serial logs, prefer `115200` baud and slower publish intervals (about 200-500ms).
- Install service:
```bash
sudo cp systemd/ultrasonic-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ultrasonic-dashboard
sudo systemctl start ultrasonic-dashboard
```
