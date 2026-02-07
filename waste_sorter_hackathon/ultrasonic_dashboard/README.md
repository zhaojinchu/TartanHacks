# Ultrasonic Bin Monitoring Dashboard

A real-time web dashboard for monitoring trash bin fullness using ultrasonic distance sensors mounted on Raspberry Pi 5.

## Overview

This module extends the TartanHacks waste sorting system by adding intelligent bin monitoring capabilities:

- **Real-time monitoring**: Live visualization of bin fullness across multiple locations
- **Usage analytics**: Track average fill times and identify usage patterns
- **Heatmap visualizations**: See when and where bins fill up fastest
- **Predictive scheduling**: ML-powered predictions for optimal cleaner routing

## Features

### 1. Real-Time Monitoring
- Live dashboard showing current status of all bins
- Visual indicators (green/yellow/red) based on fullness
- WebSocket-powered instant updates
- Filter by bin type, location, or status
- Automatic alerts when bins reach critical levels

### 2. Analytics Dashboard
- Average time for bins to become full
- Fill rate trends over time
- Historical data visualization
- Usage patterns by day/time
- Exportable reports

### 3. Heatmap Visualizations
- **Temporal heatmap**: Usage intensity by hour and day of week
- **Location heatmap**: Identify high-traffic areas
- Interactive visualizations

### 4. Predictive Scheduling
- Predict when bins will need emptying
- Optimize cleaner routes to minimize trips
- Track prediction accuracy over time
- Export schedules for facility management

## Hardware Requirements

- Raspberry Pi 5 (shares same hardware as waste detection system)
- HC-SR04 Ultrasonic Distance Sensors (one per bin)
- Jumper wires for GPIO connections
- Trash bins with known dimensions

## Quick Start

### 1. Installation

```bash
cd waste_sorter_hackathon/ultrasonic_dashboard

# Install backend dependencies
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
npm install
```

### 2. Configuration

Edit `config/bins.yaml` to define your bins:

```yaml
bins:
  - id: "recycle_1"
    type: "recycle"
    location: "Building_A_Floor_1"
    height_cm: 80
    sensor_offset_cm: 5
    gpio_trigger: 23
    gpio_echo: 24

  - id: "compost_1"
    type: "compost"
    location: "Building_A_Floor_1"
    height_cm: 75
    sensor_offset_cm: 5
    gpio_trigger: 17
    gpio_echo: 27
```

### 3. Calibrate Sensors

```bash
cd scripts
python calibrate_bins.py
```

### 4. Run the System

**Development mode:**

```bash
# Terminal 1: Start backend
cd backend
source .venv/bin/activate
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start sensor collection
python -m src.sensors.data_collector

# Terminal 3: Start frontend
cd frontend
npm run dev
```

**Production mode (Docker):**

```bash
docker-compose up -d
```

Access dashboard at: `http://localhost:3000`

## Project Structure

```
ultrasonic_dashboard/
├── backend/              # FastAPI backend service
│   ├── src/
│   │   ├── api/         # REST API routes
│   │   ├── sensors/     # GPIO sensor reading
│   │   ├── models/      # Data models and DB
│   │   ├── analytics/   # Prediction and stats
│   │   └── main.py      # App entry point
│   ├── tests/
│   └── requirements.txt
│
├── frontend/            # React dashboard
│   ├── src/
│   │   ├── components/  # Reusable UI components
│   │   ├── pages/       # Dashboard pages
│   │   └── services/    # API/WebSocket clients
│   └── package.json
│
├── config/              # Configuration files
│   ├── bins.yaml        # Bin definitions
│   ├── sensors.yaml     # GPIO mappings
│   └── thresholds.yaml  # Alert thresholds
│
├── scripts/             # Utility scripts
│   ├── setup_sensors.py    # Test GPIO connections
│   ├── calibrate_bins.py   # Measure bin dimensions
│   └── seed_data.py        # Generate test data
│
├── docker-compose.yml   # Container orchestration
└── README.md
```

## API Endpoints

### REST API

- `GET /api/bins` - List all bins with current status
- `GET /api/bins/:id` - Get specific bin details
- `GET /api/bins/:id/history` - Historical data
- `GET /api/bins/:id/prediction` - Predicted fill time
- `GET /api/analytics/fill-times` - Average fill times
- `GET /api/analytics/heatmap` - Usage heatmap data
- `GET /api/schedule/optimize` - Optimized cleaning schedule
- `POST /api/bins/:id/empty` - Mark bin as emptied

### WebSocket

- `WS /ws/bins` - Real-time bin updates

## Configuration

### Bin Configuration (`config/bins.yaml`)

```yaml
bins:
  - id: "bin_identifier"
    type: "recycle|compost|landfill"
    location: "Building_X_Floor_Y"
    height_cm: 80           # Bin depth
    sensor_offset_cm: 5     # Sensor to empty bin bottom
    gpio_trigger: 23        # GPIO pin for trigger
    gpio_echo: 24           # GPIO pin for echo

thresholds:
  normal_max: 70            # Green status < 70%
  warning_max: 85           # Yellow status 70-85%
  critical_min: 85          # Red status > 85%

measurement:
  interval_seconds: 60      # How often to measure
  samples_per_read: 5       # Average multiple readings
```

### Alert Thresholds (`config/thresholds.yaml`)

```yaml
alerts:
  almost_full: 85           # Send alert at 85%
  full: 95                  # Critical alert at 95%

notifications:
  email: true
  recipients:
    - facility@example.com
```

## Development

### Testing Without Hardware

Generate mock data for development:

```bash
cd scripts
python seed_data.py --days 30 --bins 5
```

### Running Tests

```bash
cd backend
pytest tests/
```

### API Documentation

Once backend is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Deployment

### Raspberry Pi Setup

1. Clone repository on Pi
2. Install dependencies
3. Configure bins and GPIO pins
4. Set up systemd service for autostart
5. Configure Nginx reverse proxy

Detailed instructions in [docs/deployment.md](docs/deployment.md)

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Integration with Waste Detection System

This module runs independently on the same Raspberry Pi 5 hardware. Future enhancements could correlate:
- Object detection events with fill rate increases
- Types of items detected with bin capacity usage
- Peak detection times with peak fill times

## Troubleshooting

### Sensor Reading Issues

```bash
# Test individual sensor
python scripts/setup_sensors.py --bin-id recycle_1

# Check GPIO permissions
sudo usermod -a -G gpio $USER
```

### WebSocket Connection Issues

- Check firewall settings
- Verify backend is running on correct port
- Check CORS configuration in backend

### Prediction Accuracy

- Ensure sufficient historical data (minimum 7 days)
- Check for sensor anomalies
- Verify bins are being marked as emptied correctly

## Future Enhancements

- [ ] Mobile app for cleaners
- [ ] SMS/Email notifications
- [ ] Multi-building support
- [ ] Cost analysis and reporting
- [ ] QR code bin identification
- [ ] Integration with facility management systems
- [ ] Carbon footprint tracking
- [ ] Voice alerts via speakers

## License

Part of the TartanHacks 2026 waste sorting project.

## Contributors

Built for TartanHacks 2026 - Smart Waste Management Challenge

## Support

For issues or questions, see the main project README or create an issue in the repository.
