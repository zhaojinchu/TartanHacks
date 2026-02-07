# Raspberry Pi 5 Runtime (Real-Time Inference)

This folder is a self-contained runtime for running your exported ONNX waste sorter model on Raspberry Pi 5.

It uses:
- `onnxruntime` for inference (CPU)
- `opencv-python` for webcam/video I/O
- temporal smoothing + threshold routing logic (same behavior as training project)
- class names auto-loaded from ONNX metadata (with YAML fallback)

## Folder Structure
```text
rpi5/
  configs/
    decision.yaml
  models/
    waste_sorter.onnx      # place exported model here
  scripts/
    run_realtime.py
    infer_image.py
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
rpicam-hello -t 3000
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

## 3.5) Run single-image inference (debug)
No manual preprocessing is needed. You can pass any dataset image directly (JPG/PNG/JPEG).
The script applies the same letterbox + normalization preprocessing internally.

Example with any labeled dataset image:
```bash
python scripts/infer_image.py \
  --model models/waste_sorter.onnx \
  --image ../dataset/train/images/<your_image>.jpg \
  --decision_config configs/decision.yaml \
  --conf 0.10 \
  --iou 0.45
```

This prints detections + decision JSON and saves an annotated image next to the input as:
- `<your_image>_pred.jpg`

Optionally save JSON:
```bash
python scripts/infer_image.py \
  --model models/waste_sorter.onnx \
  --image ../dataset/train/images/<your_image>.jpg \
  --json_out runs/infer_image_result.json
```

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

## 5) HTTPS stream (TLS)
Use HTTPS if you want encrypted browser transport.

Self-signed cert (auto-generate):
```bash
python scripts/run_realtime.py \
  --model models/waste_sorter.onnx \
  --decision_config configs/decision.yaml \
  --camera_backend picamera2 \
  --https_stream \
  --http_host 0.0.0.0 \
  --http_port 8443 \
  --tls_self_signed \
  --no_display
```

Then open:
- `https://<PI_IP>:8443/`

Browser warning is expected for self-signed certs.

Use your own cert/key:
```bash
python scripts/run_realtime.py \
  --model models/waste_sorter.onnx \
  --camera_backend picamera2 \
  --https_stream \
  --http_port 8443 \
  --tls_cert /path/to/cert.pem \
  --tls_key /path/to/key.pem \
  --no_display
```

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

- If ONNX metadata is missing names, provide a names YAML:
```bash
python scripts/run_realtime.py \
  --model models/waste_sorter.onnx \
  --class_names_yaml ../configs/data.yaml \
  --camera_backend picamera2
```

- Lower CPU load (recommended for first run stability):
```bash
python scripts/run_realtime.py \
  --model models/waste_sorter.onnx \
  --camera_backend picamera2 \
  --width 960 --height 540 \
  --imgsz 384 \
  --camera_fps 15 \
  --http_stream --http_quality 65 \
  --ort_intra_threads 2 --ort_inter_threads 1 \
  --no_display
```

## Servo auto-control (single-bin queue)
If your ultrasonic backend is running and Arduino commands are available at
`POST http://localhost:8000/api/arduino/command`, you can auto-open lids from
vision decisions with strict one-at-a-time sequencing:

```bash
python scripts/run_realtime.py \
  --model models/waste_sorter.onnx \
  --decision_config configs/decision.yaml \
  --camera_backend picamera2 \
  --no_display \
  --servo_enable \
  --servo_api_base http://localhost:8000 \
  --servo_open_seconds 5 \
  --servo_settle_seconds 0.4 \
  --servo_trigger_interval 1.5
```

Behavior:
- only one bin opens at a time
- each open command holds for 5 seconds, then closes
- additional recognized bins are queued and handled serially

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

## Crash / Red LED Debug (Important)
If Pi hard-resets or appears to crash during inference, most common causes are:
- power instability / undervoltage
- thermal stress
- aggressive CPU load (model + JPEG stream + high camera resolution)

Run with health telemetry enabled:
```bash
python scripts/run_realtime.py \
  --model models/waste_sorter.onnx \
  --camera_backend picamera2 \
  --no_display \
  --width 960 --height 540 \
  --imgsz 384 \
  --camera_fps 15 \
  --ort_intra_threads 2 --ort_inter_threads 1 \
  --health_every 30 \
  --health_log runs/health.jsonl
```

To isolate stream overhead, compare three modes for 5-10 minutes each:
1. No stream: no `--http_stream` / `--https_stream`
2. HTTP stream: `--http_stream`
3. HTTPS stream: `--https_stream --tls_self_signed`

Watch:
- `fps`
- `enc_ms` (JPEG encode time)
- `clients`
- `throttled`

Lower stream load if needed:
- reduce `--width/--height`
- reduce `--camera_fps`
- reduce `--http_quality` (for example `60`)

## No Detections Debug
If runtime shows `detections=0` continuously:
1. Lower threshold first:
```bash
python scripts/run_realtime.py --model models/waste_sorter.onnx --camera_backend picamera2 --conf 0.10 --iou 0.45 --no_display
```
2. Verify class names detected at startup (`Classes (...)` and `Runtime classes:` log lines).
3. Ensure object is large and centered in frame.
4. Confirm ONNX model matches your current training stage:
   - 1-class prototype model detects only one class
   - 8-class model required for full sorter behavior

## Pi Camera Blue Tint Fix
If whites look blue or skin tones look wrong, test these in order:

1) Try BGR camera format:
```bash
python scripts/run_realtime.py \
  --model models/waste_sorter.onnx \
  --camera_backend picamera2 \
  --picam_format BGR888 \
  --no_display
```

2) Keep AWB enabled and avoid pointing directly at bright lights.

3) If color is still unstable, set manual colour gains:
```bash
python scripts/run_realtime.py \
  --model models/waste_sorter.onnx \
  --camera_backend picamera2 \
  --picam_format BGR888 \
  --no-awb \
  --colour_gains 1.6 1.2 \
  --no_display
```

Note: very bright overhead light in frame can skew auto white balance and exposure.

Check for throttling/undervoltage after reboot:
```bash
vcgencmd get_throttled
journalctl -b -1 -e | tail -n 200
```

Hardware checks:
- Use official Raspberry Pi 5 27W USB-C PSU (or known-good 5V/5A PD supply).
- Avoid weak USB power banks/cables.
- Add cooling (fan/heatsink) for sustained inference workloads.

## Campus Wi-Fi Note (CMU-Secure / WPA2-Enterprise)
It may work directly at `http://<PI_IP>:8080/` if client-to-client traffic is allowed on your network segment.

If direct access is blocked, use SSH tunnel (reliable):
```bash
# Run on your laptop:
ssh -L 8080:localhost:8080 zhaojin@<PI_IP>
```

Then open locally:
- `http://localhost:8080/`

For HTTPS stream on port 8443:
```bash
ssh -L 8443:localhost:8443 zhaojin@<PI_IP>
```
Then open:
- `https://localhost:8443/`

This avoids needing open inbound ports from campus network to your Pi.
