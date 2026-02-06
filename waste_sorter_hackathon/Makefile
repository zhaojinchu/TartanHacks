PY ?= python3

.PHONY: setup extract dedupe split check train eval export infer_video

setup:
	$(PY) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

extract:
	$(PY) scripts/extract_frames.py --input_dir data/raw_videos --output_dir data/frames

dedupe:
	$(PY) scripts/dedupe_frames.py --input_dir data/frames --output_dir data/frames_deduped

split:
	$(PY) scripts/split_dataset.py --input_dir data/flat_labeled --output_dir dataset

check:
	$(PY) scripts/sanity_check_labels.py --images_dir dataset/images --labels_dir dataset/labels

train:
	$(PY) scripts/train.py --data configs/data.yaml --model yolov8n.pt --epochs 30 --batch 16

eval:
	$(PY) scripts/evaluate.py --weights runs_hack/baseline/weights/best.pt --data configs/data.yaml

export:
	$(PY) scripts/export_onnx.py --weights runs_hack/baseline/weights/best.pt --imgsz 512

infer_video:
	$(PY) scripts/infer_video.py --weights runs_hack/baseline/weights/best.pt --source data/test.mp4 --save_path runs_hack/infer_test.mp4
