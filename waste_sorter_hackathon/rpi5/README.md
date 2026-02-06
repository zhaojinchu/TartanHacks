# Raspberry Pi 5 Runtime (Real-Time Inference)

This folder is a self-contained runtime for running your exported ONNX waste sorter model on Raspberry Pi 5.

It uses:
- `onnxruntime` for inference (CPU)
- `opencv-python` for webcam/video I/O
- temporal smoothing + threshold routing logic (same behavior as training project)

## Folder Structure
```text
rpi5/
  configs/
    decision.yaml
  models/
    waste_sorter.onnx      # place exported model here
  scripts/
    run_realtime.py
  src/
    io_utils.py
    decision.py
    onnx_detector.py
  requirements.txt
```

## 1) Export model on your Mac
From the main project root (`waste_sorter_hackathon/`):

```bash
python scripts/export_onnx.py \
  --weights runs_hack/baseline/weights/best.pt \
  --opset 12 \
  --imgsz 512
```

Copy the generated `.onnx` file into:
- `rpi5/models/waste_sorter.onnx`

## 2) Setup on Raspberry Pi 5
On the Pi:

```bash
cd rpi5
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Raspberry Pi Camera Module 2 (libcamera stack), install:
```bash
sudo apt update
sudo apt install -y python3-picamera2
```

Quick camera sanity test:
```bash
libcamera-hello -t 3000
```

## 3) Run real-time webcam inference
```bash
python scripts/run_realtime.py \
  --model models/waste_sorter.onnx \
  --decision_config configs/decision.yaml \
  --camera_backend picamera2 \
  --camera_index 0 \
  --imgsz 512 \
  --conf 0.25 \
  --iou 0.45
```

Press `q` to quit.

## 4) View live feed in browser (HTTP stream)
Start runtime with MJPEG stream enabled:
```bash
python scripts/run_realtime.py \
  --model models/waste_sorter.onnx \
  --decision_config configs/decision.yaml \
  --camera_backend picamera2 \
  --http_stream \
  --http_host 0.0.0.0 \
  --http_port 8080 \
  --no_display
```

Open in browser:
- `http://<PI_IP>:8080/`

Direct stream URL:
- `http://<PI_IP>:8080/stream`

## Useful options
- Save annotated output video:
```bash
python scripts/run_realtime.py --model models/waste_sorter.onnx --save_path runs/rpi5_demo.mp4
```

- Headless mode (no window):
```bash
python scripts/run_realtime.py --model models/waste_sorter.onnx --camera_backend picamera2 --no_display
```

- Tune routing quickly:
```bash
python scripts/run_realtime.py --model models/waste_sorter.onnx --camera_backend picamera2 --threshold 0.65 --window 7
```

## Decision Output Behavior
Per frame, detections are smoothed over a rolling window.

If top smoothed class score is below threshold (`unknown_threshold`), output becomes:
- `final_bin = landfill`
- `reason = unknown_low_conf`

Otherwise top class maps to bin (`recycle`, `compost`, `landfill`) via `configs/decision.yaml`.

## Locked classes (must match training/export)
0 `aluminum_can`
1 `plastic_bottle`
2 `lp_paper_cup`
3 `lp_plastic_cup`
4 `rigid_plastic_container`
5 `straw`
6 `utensil`
7 `napkin`

## Notes for Pi performance
- Start with `--imgsz 512`; reduce to `384` if FPS is too low.
- Keep webcam resolution moderate (`1280x720` default in script).
- Increase `--conf` (for example `0.35`) if too many noisy detections appear.
- For Camera Module 2, use `--camera_backend picamera2` (recommended).
- HTTP stream adds JPEG encoding overhead; lower resolution first if FPS drops.

## Campus Wi-Fi Note (CMU-Secure / WPA2-Enterprise)
It may work directly at `http://<PI_IP>:8080/` if client-to-client traffic is allowed on your network segment.

If direct access is blocked, use SSH tunnel (reliable):
```bash
# Run on your laptop:
ssh -L 8080:localhost:8080 zhaojin@<PI_IP>
```

Then open locally:
- `http://localhost:8080/`

This avoids needing open inbound ports from campus network to your Pi.
