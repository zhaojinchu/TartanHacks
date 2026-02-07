# AI Agent Prompt: Ultrasonic Bin Monitoring Dashboard

## Project Context
You are building a web-based dashboard module for the TartanHacks waste sorting system. This system currently uses YOLO object detection on Raspberry Pi 5 to identify and route waste items (aluminum cans, plastic bottles, cups, containers, straws, utensils, napkins) to appropriate bins (recycle, compost, landfill).

## New Module: Ultrasonic Sensor Bin Monitoring

### Hardware Setup
- **Sensors**: HC-SR04 ultrasonic distance sensors (or similar) mounted facing downward into trash bins
- **Measurement**: Distance from sensor (top of bin) to trash surface determines bin fullness
- **Deployment**: Raspberry Pi 5 with multiple ultrasonic sensors connected via GPIO
- **Configuration**: Support for multiple bins with different dimensions and sensor placements

### Core Requirements

#### 1. Real-Time Data Collection (Backend)
Create a Python backend service that:
- Reads distance measurements from multiple HC-SR04 ultrasonic sensors via Raspberry Pi GPIO
- Converts distance to fullness percentage based on configurable bin dimensions
  - Example: If bin depth is 80cm and sensor reads 20cm distance, bin is 75% full
- Timestamps all measurements
- Stores data in a time-series database (SQLite or InfluxDB recommended)
- Exposes real-time data via WebSocket and REST API endpoints
- Handles sensor errors and missing data gracefully

**Key Data Schema**:
```python
{
    "bin_id": "recycle_1",
    "bin_type": "recycle",  # recycle, compost, landfill
    "timestamp": "2026-02-07T10:30:00Z",
    "distance_cm": 25.4,
    "fullness_percent": 68.3,
    "bin_height_cm": 80.0,
    "status": "normal",  # normal, almost_full, full, sensor_error
    "location": "Building_A_Floor_1"
}
```

#### 2. Web Dashboard (Frontend)
Build a responsive React/Vue.js dashboard with the following features:

**Page 1: Real-Time Monitoring**
- Live grid/list view of all bins showing:
  - Bin ID and type (recycle/compost/landfill)
  - Current fullness percentage with visual gauge/progress bar
  - Status indicator (green/yellow/red based on thresholds: <70% green, 70-85% yellow, >85% red)
  - Last update timestamp
  - Location information
- Auto-refresh using WebSocket for real-time updates
- Filter by bin type, location, or status
- Alert notifications when bins reach critical fullness (>85%)

**Page 2: Usage Analytics**
- **Average Fill Time Chart**:
  - Bar/line chart showing average time (in hours) for each bin to go from 0% to 85% full
  - Grouped by bin type and location
  - Configurable time ranges (last 7 days, 30 days, all time)

- **Fill Rate Trends**:
  - Time-series graph showing how quickly bins fill over different days/weeks
  - Identify patterns (e.g., faster filling on weekends, specific days)

**Page 3: Heatmap Visualizations**
- **Temporal Heatmap**:
  - Grid showing bin activity by hour-of-day vs day-of-week
  - Color intensity represents average fullness increase rate
  - Helps identify peak usage times

- **Location Heatmap**:
  - If multiple bins/locations, show which areas have highest waste generation
  - Interactive map or grid visualization

**Page 4: Predictive Scheduling**
- **Bin Change Predictions**:
  - ML-based predictions for when each bin will reach 85% full
  - Display predicted time with confidence interval
  - Use linear regression or simple time-series forecasting based on historical fill rates

- **Optimized Cleaner Schedule**:
  - Suggested collection routes and times based on predictions
  - Minimize cleaner trips by grouping bins that will be full around same time
  - Visual route planner showing bins by priority
  - Export schedule as CSV/PDF

- **Historical Accuracy**:
  - Track prediction accuracy over time
  - Show actual vs predicted fill times

#### 3. Machine Learning / Prediction Engine
Implement predictive models:
- **Fill Rate Calculation**:
  - Sliding window average of fullness increase per hour
  - Account for different rates during different times of day

- **Time-to-Full Prediction**:
  - Linear extrapolation with recent trend weighting
  - Consider day-of-week and time-of-day patterns
  - Confidence intervals based on historical variance

- **Anomaly Detection**:
  - Flag unusual patterns (sudden drops in fullness = bin emptied)
  - Detect sensor malfunctions (impossible readings, stuck values)

#### 4. Backend API Endpoints
Create RESTful API:
```
GET  /api/bins                    # List all bins with current status
GET  /api/bins/:id                # Get specific bin details
GET  /api/bins/:id/history        # Historical data (with date range params)
GET  /api/bins/:id/prediction     # Predicted fill time
GET  /api/analytics/fill-times    # Average fill times by bin type
GET  /api/analytics/heatmap       # Usage heatmap data
GET  /api/schedule/optimize       # Get optimized cleaning schedule
POST /api/bins/:id/empty          # Mark bin as emptied (manual override)
WS   /ws/bins                     # WebSocket for real-time updates
```

#### 5. Configuration & Setup
- **Config file** (YAML/JSON) for:
  - Bin definitions (ID, type, location, dimensions)
  - GPIO pin mappings for each sensor
  - Fullness thresholds for alerts
  - Measurement intervals
  - Dashboard settings

- **Easy deployment**:
  - Docker Compose setup for database + backend + frontend
  - Systemd service for Raspberry Pi autostart
  - Clear installation instructions

### Technical Stack Recommendations

**Backend**:
- Python 3.11+
- FastAPI (REST API + WebSocket support)
- SQLite or InfluxDB (time-series data)
- RPi.GPIO or gpiozero (sensor reading)
- scikit-learn or numpy (predictions)
- Pydantic (data validation)

**Frontend**:
- React with TypeScript or Vue.js 3
- Chart.js or Recharts (visualizations)
- TailwindCSS or Material-UI (styling)
- Socket.io-client or native WebSocket (real-time updates)
- React Router or Vue Router (navigation)

**Deployment**:
- Docker & Docker Compose
- Nginx reverse proxy (frontend serving + API routing)
- Systemd service (sensor reading daemon)

### Folder Structure
```
waste_sorter_hackathon/
  ultrasonic_dashboard/
    backend/
      src/
        api/
          routes.py          # API endpoints
          websocket.py       # WebSocket handlers
        sensors/
          ultrasonic.py      # GPIO sensor reading
          data_collector.py  # Continuous measurement service
        models/
          schemas.py         # Pydantic models
          database.py        # DB connection and queries
        analytics/
          predictions.py     # ML prediction models
          statistics.py      # Analytics calculations
        config.py            # Configuration loading
        main.py              # FastAPI app entry point
      requirements.txt
      Dockerfile

    frontend/
      src/
        components/
          BinCard.tsx        # Individual bin display
          LiveGrid.tsx       # Real-time monitoring view
          Heatmap.tsx        # Heatmap visualizations
          PredictionChart.tsx # Prediction displays
          ScheduleView.tsx   # Optimized schedule
        pages/
          Dashboard.tsx      # Main real-time page
          Analytics.tsx      # Usage analytics page
          Predictions.tsx    # Prediction & scheduling page
        services/
          api.ts             # API client
          websocket.ts       # WebSocket client
        App.tsx
        main.tsx
      package.json
      vite.config.ts
      Dockerfile

    config/
      bins.yaml              # Bin configuration
      sensors.yaml           # GPIO pin mappings
      thresholds.yaml        # Alert thresholds

    scripts/
      setup_sensors.py       # Test GPIO sensor connections
      calibrate_bins.py      # Measure empty/full distances
      seed_data.py           # Generate mock historical data for testing

    docker-compose.yml       # Orchestrate all services
    README.md                # Setup and usage instructions
```

### Development Phases

**Phase 1: Sensor Integration & Data Collection**
- Set up GPIO sensor reading on Raspberry Pi
- Create data collection service with SQLite storage
- Test with mock data if sensors not available yet
- Basic logging and error handling

**Phase 2: Backend API**
- FastAPI REST endpoints for bin data
- WebSocket for real-time updates
- Basic analytics calculations (average fill times)
- Database schema and migrations

**Phase 3: Frontend Dashboard (Core)**
- Real-time monitoring page with live updates
- Bin cards with status indicators
- Basic filtering and search
- Responsive layout

**Phase 4: Analytics & Visualizations**
- Fill time analytics with charts
- Temporal and location heatmaps
- Historical data visualization
- Export capabilities

**Phase 5: Predictions & Scheduling**
- Implement prediction algorithms
- Build scheduling optimizer
- Prediction accuracy tracking
- Schedule export and notifications

**Phase 6: Polish & Deployment**
- Docker containerization
- Production-ready deployment guide
- Performance optimization
- Comprehensive testing

### Specific Implementation Notes

1. **Ultrasonic Sensor Reading (Python)**:
```python
import RPi.GPIO as GPIO
import time

def measure_distance(trigger_pin, echo_pin):
    # Send trigger pulse
    GPIO.output(trigger_pin, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trigger_pin, GPIO.LOW)

    # Measure echo return time
    while GPIO.input(echo_pin) == 0:
        pulse_start = time.time()
    while GPIO.input(echo_pin) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # Speed of sound calculation
    return round(distance, 2)
```

2. **Fullness Calculation**:
```python
def calculate_fullness(distance_cm, bin_height_cm, sensor_offset_cm=5):
    """
    sensor_offset_cm: Distance from sensor to empty bin bottom
    """
    trash_height = (sensor_offset_cm + bin_height_cm) - distance_cm
    fullness_percent = (trash_height / bin_height_cm) * 100
    return max(0, min(100, fullness_percent))  # Clamp 0-100
```

3. **Fill Time Prediction**:
```python
def predict_time_to_full(bin_id, current_fullness, target_fullness=85):
    # Get recent measurements (last 24 hours)
    recent_data = get_recent_measurements(bin_id, hours=24)

    # Calculate fill rate (% per hour)
    fill_rate = calculate_fill_rate(recent_data)

    if fill_rate <= 0:
        return None  # Not filling or being emptied

    percent_remaining = target_fullness - current_fullness
    hours_to_full = percent_remaining / fill_rate

    return {
        "predicted_full_at": datetime.now() + timedelta(hours=hours_to_full),
        "confidence": calculate_confidence(recent_data),
        "fill_rate_per_hour": fill_rate
    }
```

### Testing Requirements
- Unit tests for sensor reading functions
- Integration tests for API endpoints
- Mock data generation for frontend development
- End-to-end tests for WebSocket updates
- Load testing for multiple concurrent sensors
- Sensor failure simulation and recovery

### Success Criteria
- Dashboard loads in <2 seconds
- Real-time updates with <500ms latency
- Predictions accurate within ±2 hours for 80% of cases
- Support at least 10 simultaneous bins
- Mobile-responsive interface
- 99% uptime for data collection service
- Clear documentation for deployment

### Integration with Existing System
This module operates independently but should:
- Use similar configuration patterns as the main waste sorter
- Share the same Raspberry Pi 5 hardware
- Optionally correlate ultrasonic data with object detection events
  - Future enhancement: Track correlation between detected items and fill rate
- Maintain consistent coding style with existing Python codebase
- Follow the same project structure conventions

### Deliverables
1. Fully functional backend service with sensor integration
2. Web dashboard with all specified features
3. Docker Compose setup for easy deployment
4. README with setup, configuration, and usage instructions
5. Sample configuration files
6. Mock data generation script for testing without hardware
7. API documentation (auto-generated with FastAPI/Swagger)

### Extra Credit Features (Optional)
- Email/SMS alerts when bins are almost full
- Multi-tenant support for different buildings/locations
- Historical data export and reporting
- Mobile app (React Native) for cleaners
- QR code scanning to mark bins as emptied
- Cost analysis (correlate waste volume with collection costs)
- Carbon footprint tracking based on waste type and volume
- Integration with existing waste detection system to track what types of items fill bins fastest

---

## Getting Started
Begin by:
1. Creating the folder structure
2. Setting up basic sensor reading with test script
3. Implementing data collection service with SQLite
4. Building REST API with FastAPI
5. Creating React dashboard skeleton
6. Iterating through each phase

Focus on getting real-time monitoring working first, then add analytics and predictions.
