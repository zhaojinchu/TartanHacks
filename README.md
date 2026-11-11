# ReBin

A smart bin attachment for restaurant waste, built in 24 hours at TartanHacks in February 2026,
where it won Best Hardware. A camera on a Raspberry Pi 5 classifies each item with a YOLOv8n model
and opens the lid of the right bin, and ultrasonic sensors in the bins feed a dashboard that
predicts when each one will be full.

## How it works

- The Pi runs the model through ONNX Runtime on its CPU, averages each class's score over the last
  5 frames, and maps the top class to bottles, compost or landfill. A top score under 0.60 goes to
  landfill.
- With `--servo_enable`, each confident decision opens that bin's lid through an Arduino, by way
  of the dashboard backend.
- The dashboard (FastAPI and React) reads HC-SR04 sensors, stores readings in SQLite, predicts from
  each bin's fill rate when it reaches a target fullness (90% by default), and orders the pickups
  by that time.
- The model has 9 classes (cups, lids, containers, napkins, straws, utensils) and was trained on
  about 1,550 video frames labelled in Roboflow, 5,764 images after augmentation.

## Run it

On the Pi, from `waste_sorter_hackathon/`:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install -r rpi5/requirements.txt
python rpi5/scripts/run_realtime.py \
  --model rpi5/models/waste_sorter.onnx \
  --decision_config rpi5/configs/decision.yaml \
  --camera_backend picamera2 \
  --no_display
```

The dashboard, from `TARTANHACKS/ultrasonic_dashboard/`, is `docker compose up --build`, then
`http://localhost:3000`.

`waste_sorter_hackathon/README.md` covers training, `waste_sorter_hackathon/rpi5/README.md` the
Pi setup, and `TARTANHACKS/ultrasonic_dashboard/README.md` the sensor wiring and configuration.
