# Waste Sorter Hackathon

A small, practical YOLO project for a 24-hour prototype.

You can:
- train a lightweight detector on MacBook (`mps` auto-selected when available)
- run inference on video or webcam
- smooth noisy frame-by-frame detections into one routing decision
- export ONNX for Raspberry Pi 5

## How This Project Works
Think of the system as 4 simple stages.

1. Data prep
- `extract_frames.py` pulls frames from raw videos
- `dedupe_frames.py` removes near-duplicates and keeps the sharper frame
- You label the deduped frames in Roboflow and export YOLO labels

2. Dataset checks + split
- `split_dataset.py` creates `train/val/test`
- `sanity_check_labels.py` catches bad labels early (class range, normalized boxes, pairing)

3. Model training + evaluation
- `train.py` runs Ultralytics YOLO with strong defaults
- `evaluate.py` prints core metrics and saves confusion matrix + prediction samples

4. Runtime decisions
- `infer_video.py` and `infer_webcam.py` run detector inference
- `route_demo.py` sends detections to `src/decision.py`
- `TemporalDecisionEngine` smooths over recent frames and maps top class to a bin
- If confidence stays too low, it falls back to `landfill` with reason `unknown_low_conf`

## Locked Class Contract (Do Not Change Order)
Your labels must use this exact order:

0 `aluminum_can`
1 `plastic_bottle`
2 `lp_paper_cup`
3 `lp_plastic_cup`
4 `rigid_plastic_container`
5 `straw`
6 `utensil`
7 `napkin`

Rules:
- no `food_scrap` class
- no `other_unknown` class
- unknown handling happens in decision logic via threshold, not training labels

## Project Layout
```text
waste_sorter_hackathon/
  configs/
    data.yaml          # YOLO dataset config + locked class names
    decision.yaml      # class->bin mapping + unknown threshold + window size
  scripts/
    extract_frames.py
    dedupe_frames.py
    split_dataset.py
    sanity_check_labels.py
    train.py
    evaluate.py
    export_onnx.py
    infer_video.py
    infer_webcam.py
    route_demo.py
  src/
    io_utils.py        # YAML + decision config loaders
    decision.py        # TemporalDecisionEngine
```

## Quick Setup (Mac)
Run from project root (`waste_sorter_hackathon/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## End-to-End Commands

### 1) Extract frames
```bash
python scripts/extract_frames.py \
  --input_dir data/raw_videos \
  --output_dir data/frames \
  --fps 2
```

Optional resize and cap:
```bash
python scripts/extract_frames.py \
  --input_dir data/raw_videos \
  --output_dir data/frames \
  --fps 2 \
  --width 1280 --height 720 \
  --max_frames_per_video 400
```

### 2) Deduplicate frames
```bash
python scripts/dedupe_frames.py \
  --input_dir data/frames \
  --output_dir data/frames_deduped \
  --sim_threshold 0.92
```

### 3) Label in Roboflow
- Upload `data/frames_deduped`
- Annotate with the exact 8 classes above, in that order
- Export YOLO format
- Put flat exports (image + matching `.txt`) in `data/flat_labeled`

### 4) Split train/val/test
```bash
python scripts/split_dataset.py \
  --input_dir data/flat_labeled \
  --output_dir dataset \
  --seed 42
```

### 5) Sanity check labels
```bash
python scripts/sanity_check_labels.py \
  --images_dir dataset/images \
  --labels_dir dataset/labels \
  --num_classes 8
```

### 6) Train model
```bash
python scripts/train.py \
  --data configs/data.yaml \
  --model yolov8n.pt \
  --imgsz 512 \
  --epochs 30 \
  --batch 16 \
  --project runs_hack \
  --name baseline \
  --device auto \
  --seed 42
```

Notes:
- `--device auto` chooses `mps`, then CUDA `0`, then `cpu`
- try `--model yolo11n.pt` if you want to test newer tiny weights

### 7) Evaluate
```bash
python scripts/evaluate.py \
  --weights runs_hack/baseline/weights/best.pt \
  --data configs/data.yaml
```

### 8) Export ONNX
```bash
python scripts/export_onnx.py \
  --weights runs_hack/baseline/weights/best.pt \
  --opset 12 \
  --imgsz 512
```

### 9) Inference on video
```bash
python scripts/infer_video.py \
  --weights runs_hack/baseline/weights/best.pt \
  --source data/test.mp4 \
  --conf 0.25 \
  --iou 0.45 \
  --save_path runs_hack/infer_test.mp4
```

### 10) Inference on webcam
```bash
python waste_sorter_hackathon/scripts/infer_webcam.py \
  --weights runs_hack/baseline/weights/best.pt \
  --conf 0.25 \
  --iou 0.45
```

Press `q` to quit.

### 11) Routing demo (smoothed final bin)
```bash
python scripts/route_demo.py \
  --weights runs_hack/baseline/weights/best.pt \
  --source data/test.mp4 \
  --decision_config configs/decision.yaml \
  --conf 0.25 \
  --iou 0.45 \
  --save_path runs_hack/route_demo.mp4
```

## Decision Logic (Simple)
Configured in `configs/decision.yaml`:
- `window_size`: smoothing window (default `5`)
- `unknown_threshold`: fallback threshold (default `0.60`)
- class-to-bin mapping (`recycle`, `compost`, `landfill`)

Runtime behavior per frame:
1. Collect `(class_id, confidence)` detections.
2. For each class, keep max confidence for that frame.
3. Average class scores across last `N` frames.
4. If top class score `< threshold`, output:
   - `final_bin = landfill`
   - `reason = unknown_low_conf`
5. Else map class to bin from config.

Returned structure:
```python
{
  "final_bin": "...",
  "top_class": "...",
  "score": 0.0,
  "reason": "mapped_from_class | unknown_low_conf",
  "per_class_scores": {"aluminum_can": 0.12, ...}
}
```

## Makefile Shortcuts
```bash
make setup
make extract
make dedupe
make split
make check
make train
make eval
make export
make infer_video
```

## Troubleshooting

### MPS not detected on Apple Silicon
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```
If `False`, reinstall PyTorch in this venv.

### Label check fails
Run:
```bash
python scripts/sanity_check_labels.py --images_dir dataset/images --labels_dir dataset/labels
```
Fix class IDs outside `[0..7]`, bad normalized values, or missing image/label pairs.

### Roboflow class order mismatch
- `configs/data.yaml` must match annotation order exactly.
- If class order differs, fix labels/export and retrain.

### Numpy conflict (`numpy 2.x` vs sklearn/statsmodels)
This repo pins `numpy<2.0`.
If your env already has `numpy 2.x`:
```bash
pip install --upgrade --force-reinstall "numpy<2.0"
pip install -r requirements.txt
```

## First Goal for a Hackathon Demo
If you only want to get to first model quickly:
1. Extract + dedupe frames.
2. Label 200-500 good images in Roboflow.
3. Split + sanity check.
4. Train 20-30 epochs with `yolov8n.pt`.
5. Run `route_demo.py` on a short test video and tune threshold in `configs/decision.yaml`.

## Raspberry Pi 5 Deployment
Use the dedicated runtime in `rpi5/`:
- ONNX Runtime + OpenCV real-time inference script
- Pi-specific requirements and decision config
- step-by-step setup and run instructions in `rpi5/README.md`

## Common Commands (Copy-Paste)
Run from repo root (`waste_sorter_hackathon/`) unless noted.

```bash
# 1) Train (Mac)
python3.11 scripts/train.py \
  --data configs/data.yaml \
  --model yolov8n.pt \
  --imgsz 512 \
  --epochs 50 \
  --batch 32 \
  --project runs_hack \
  --name baseline \
  --device auto \
  --seed 42

# 2) Export ONNX (Mac)
python3.11 scripts/export_onnx.py \
  --weights runs_hack/baseline/weights/best.pt \
  --opset 12 \
  --imgsz 512

# 3) Copy ONNX to Pi (from Mac)
scp runs_hack/baseline/weights/best.onnx \
  zhaojin@<PI_IP>:~/Projects/TartanHacks/waste_sorter_hackathon/rpi5/models/waste_sorter.onnx

cp /Users/zhaojin/Projects/TartanHacks/waste_sorter_hackathon/runs_hack/baseline/weights/best.onnx rpi5/models/waste_sorter.onnx

# 4) Pi setup (on Raspberry Pi)
cd ~/Projects/TartanHacks/waste_sorter_hackathon
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r rpi5/requirements.txt

# 5) Pi real-time inference (headless, recommended)
python rpi5/scripts/run_realtime.py \
  --model rpi5/models/waste_sorter.onnx \
  --decision_config rpi5/configs/decision.yaml \
  --camera_backend picamera2 \
  --no_display \
  --conf 0.10 \
  --iou 0.45 \
  --width 960 --height 540 \
  --camera_fps 15

# 6) Pi HTTPS stream mode (headless)
python rpi5/scripts/run_realtime.py \
  --model rpi5/models/waste_sorter.onnx \
  --decision_config rpi5/configs/decision.yaml \
  --camera_backend picamera2 \
  --https_stream \
  --http_host 0.0.0.0 \
  --http_port 8443 \
  --tls_self_signed \
  --no_display

# 7) SSH tunnel for HTTPS stream (run on laptop)
ssh -L 8443:localhost:8443 zhaojin@<PI_IP>
# then open https://localhost:8443/

# 8) Pi single-image test (any dataset image, no manual preprocessing needed)
python rpi5/scripts/infer_image.py \
  --model rpi5/models/waste_sorter.onnx \
  --image dataset/train/images/<your_image>.jpg \
  --decision_config rpi5/configs/decision.yaml \
  --conf 0.10 \
  --iou 0.45
```
