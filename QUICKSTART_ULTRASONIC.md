# Ultrasonic Dashboard - Quick Start Guide

## 🎯 What You Need to Know

This module adds intelligent bin monitoring to your TartanHacks waste sorter using ultrasonic sensors.

### Hardware Required
- **HC-SR04 Ultrasonic Sensors** (one per bin)
- **Raspberry Pi 5** (same one running your waste detection)
- **Jumper wires** for GPIO connections
- **Voltage divider** (1kΩ + 2kΩ resistors) for echo pin protection

### What Was Created

1. **Complete AI Prompt** → [ULTRASONIC_DASHBOARD_PROMPT.md](ULTRASONIC_DASHBOARD_PROMPT.md)
   - Give this to any AI agent to build the entire system
   
2. **Project Structure** → `waste_sorter_hackathon/ultrasonic_dashboard/`
   - Ready-to-use configuration files
   - Sensor testing script
   - Docker deployment setup
   
3. **Documentation** → [ULTRASONIC_DASHBOARD_SUMMARY.md](ULTRASONIC_DASHBOARD_SUMMARY.md)
   - Complete feature list
   - Technical details
   - Development guide

---

## 🚀 Build It with AI (Fastest Way)

### Step 1: Open the AI Prompt
```bash
open ULTRASONIC_DASHBOARD_PROMPT.md
```

### Step 2: Copy Everything
Select all the text in that file and copy it.

### Step 3: Give to AI Agent
Paste the prompt into:
- **Claude** (recommended - you're already here!)
- **ChatGPT o1/4**
- **Any coding AI assistant**

### Step 4: Ask It to Build
Say something like:
> "Please build this ultrasonic bin monitoring dashboard following the specifications in the prompt. Start with Phase 1 (sensor integration) and work through all phases."

The AI will generate all the backend and frontend code for you!

---

## 🛠️ Manual Development Path

### 1. Configure Your Bins

Edit `waste_sorter_hackathon/ultrasonic_dashboard/config/bins.yaml`:

```yaml
bins:
  - id: "recycle_1"
    type: "recycle"
    location: "Your_Location"
    height_cm: 80              # Measure your bin!
    sensor_offset_cm: 5
    gpio_trigger: 23           # Your GPIO pins
    gpio_echo: 24
```

### 2. Wire Up Sensors

For each HC-SR04 sensor:
```
Sensor Pin → Raspberry Pi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VCC        → 5V (Pin 2 or 4)
GND        → Ground (Pin 6, 9, 14, etc.)
Trig       → GPIO 23 (or your chosen pin)
Echo       → GPIO 24 via voltage divider ⚠️
```

**⚠️ Important**: Echo outputs 5V but Pi expects 3.3V. Use resistor divider:
- Echo → 1kΩ resistor → GPIO pin
- Junction → 2kΩ resistor → Ground

### 3. Test Sensors

```bash
cd waste_sorter_hackathon/ultrasonic_dashboard
python scripts/setup_sensors.py
```

This will test all configured sensors and show:
- Distance readings
- Measurement stability
- Connection health

### 4. Build Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Now implement the modules:
# - src/sensors/ultrasonic.py (GPIO reading)
# - src/sensors/data_collector.py (continuous monitoring)
# - src/api/routes.py (REST endpoints)
# - src/models/schemas.py (data models)
# - src/analytics/predictions.py (ML)
```

### 5. Build Frontend

```bash
cd frontend
npm install

# Now implement the components:
# - src/components/BinCard.tsx
# - src/pages/Dashboard.tsx
# - src/pages/Analytics.tsx
# - src/pages/Predictions.tsx
# - src/services/api.ts
# - src/services/websocket.ts
```

### 6. Deploy

```bash
cd waste_sorter_hackathon/ultrasonic_dashboard
docker-compose up -d
```

Access at: `http://localhost:3000`

---

## 📊 What You'll Get

### Real-Time Dashboard
- See all bins at a glance
- Green/yellow/red status indicators
- Live updates via WebSocket
- Alerts when bins are almost full

### Analytics
- Average fill times by bin type
- Usage trends over time
- Peak usage identification
- Exportable reports

### Heatmaps
- When bins fill up (hour/day patterns)
- Where bins fill fastest (location)
- Visual pattern recognition

### Predictions
- When each bin will be full
- Optimized cleaner schedules
- Route planning
- Cost optimization

---

## 🔧 Configuration Files

### bins.yaml - Define Your Bins
```yaml
bins:
  - id: "unique_bin_id"
    type: "recycle|compost|landfill"
    location: "Building_Floor_Area"
    height_cm: 80
    gpio_trigger: 23
    gpio_echo: 24
```

### thresholds.yaml - Set Alerts
```yaml
alert_levels:
  normal: {max: 70, color: "green"}
  warning: {min: 70, max: 85, color: "yellow"}
  critical: {min: 85, color: "red"}
```

---

## 🎓 How It Works

```
┌─────────────────┐
│ Ultrasonic      │ Measures distance to trash
│ Sensor (HC-SR04)│
└────────┬────────┘
         │ GPIO
         ▼
┌─────────────────┐
│ Raspberry Pi 5  │ Converts to fullness %
│ Python Service  │
└────────┬────────┘
         │ WebSocket + REST
         ▼
┌─────────────────┐
│ Web Dashboard   │ Real-time visualization
│ (React)         │ Analytics & Predictions
└─────────────────┘
```

**Distance to Fullness Calculation:**
```python
trash_height = (sensor_offset + bin_height) - measured_distance
fullness_percent = (trash_height / bin_height) * 100
```

---

## 📦 Technology Stack

**Backend:**
- FastAPI (API server)
- RPi.GPIO (sensor reading)
- SQLite/InfluxDB (time-series data)
- scikit-learn (predictions)
- WebSocket (real-time)

**Frontend:**
- React + TypeScript
- Recharts (charts)
- TailwindCSS (styling)
- Socket.io (WebSocket)

**Deployment:**
- Docker + Docker Compose
- Nginx (reverse proxy)
- Systemd (autostart)

---

## 🐛 Troubleshooting

### No sensor readings?
```bash
# Check GPIO permissions
sudo usermod -a -G gpio $USER

# Test specific sensor
python scripts/setup_sensors.py --trigger 23 --echo 24
```

### Erratic readings?
- Check wiring connections
- Verify voltage divider on echo pin
- Ensure nothing blocking sensor
- Try increasing samples_per_read in config

### WebSocket not connecting?
- Check backend is running: `curl http://localhost:8000/api/bins`
- Verify CORS settings
- Check firewall rules

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `ULTRASONIC_DASHBOARD_PROMPT.md` | Complete AI agent prompt |
| `ULTRASONIC_DASHBOARD_SUMMARY.md` | Project overview & guide |
| `ultrasonic_dashboard/README.md` | Module documentation |
| `config/bins.yaml` | Bin configuration |
| `config/thresholds.yaml` | Alert settings |
| `scripts/setup_sensors.py` | Sensor testing tool |

---

## 🎯 Quick Commands

```bash
# Test sensors
python scripts/setup_sensors.py

# Run backend (dev)
cd backend && uvicorn src.main:app --reload

# Run frontend (dev)
cd frontend && npm run dev

# Deploy all (production)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all
docker-compose down
```

---

## 💡 Pro Tips

1. **Start Simple**: Get one sensor working before adding multiple bins
2. **Test Often**: Use `setup_sensors.py` frequently to verify hardware
3. **Mock Data**: Use the seed_data.py script (to be created) for frontend development without hardware
4. **Iterative Development**: Build real-time monitoring first, then add analytics and predictions
5. **AI Assistance**: Use the comprehensive prompt to get help building any component

---

## 🤝 Integration with Waste Detector

This module:
- ✅ Runs on same Raspberry Pi 5
- ✅ Independent operation (won't interfere)
- ✅ Uses similar config patterns
- 🔮 Future: Correlate object detection with fill rates

---

## 🎉 You're Ready!

Pick your path:
- **Fast**: Use AI agent with the prompt → Get code in minutes
- **Learn**: Build manually following the structure → Full control

Either way, you'll have a powerful bin monitoring system!

Questions? Check the README files or the comprehensive prompt for details.

**Happy building! 🚀**
