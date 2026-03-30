#!/usr/bin/env python3
"""REBIN Waste Sorting Game – tabling event app.

State machine:
  scanning -> detected -> player_chose -> reveal -> result -> scanning
"""

from __future__ import annotations

import base64
import csv
import sys
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.decision import TemporalDecisionEngine
from src.io_utils import CLASS_NAMES, load_decision_config
from ultralytics import YOLO

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
WEIGHTS_DEFAULT = str(ROOT / "runs_hack/v2_9class/weights/best.pt")
DECISION_CFG    = str(ROOT / "configs/decision.yaml")
CAMERA_INDEX    = 0
MODEL_CONF      = 0.35
MODEL_IOU       = 0.45
LOCK_IN_FRAMES  = 10   # consecutive confident frames before auto-freeze

RAFFLE_CSV  = ROOT / "game/raffle_entries.csv"

BIN_COLORS = {"bottles": "#1565C0", "compost": "#2E7D32", "landfill": "#BF360C"}
BIN_LABELS = {"bottles": "Bottles / Recycling", "compost": "Compost", "landfill": "Landfill / Trash"}
BIN_ICONS  = {"bottles": "♻️", "compost": "🌱", "landfill": "🗑️"}

# ── Shared game state ──────────────────────────────────────────────────────────
_lock = threading.Lock()
_game: dict = {
    "phase": "scanning",   # scanning | detected | player_chose | reveal | result
    "frozen_jpeg": None,   # raw bytes of frozen annotated frame
    "ai_bin": None,
    "ai_class": None,
    "ai_score": 0.0,
    "player_bin": None,
    "outcome": None,       # prize | raffle | none
    "consecutive_hits": 0,
}

_frame_lock = threading.Lock()
_latest_jpeg: bytes | None = None


# ── Camera / inference loop ────────────────────────────────────────────────────

def _draw_box(frame, x1: int, y1: int, x2: int, y2: int, label: str) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 0), 2, cv2.LINE_AA)


def camera_loop(weights: str, camera_idx: int) -> None:
    global _latest_jpeg

    model = YOLO(weights)
    class_to_bin, threshold, window_size = load_decision_config(DECISION_CFG)
    engine = TemporalDecisionEngine(
        class_to_bin=class_to_bin,
        threshold=threshold,
        window_size=window_size,
        class_names=CLASS_NAMES,
    )

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_idx}")
        return

    print(f"[INFO] Camera {camera_idx} opened. Model loaded from {weights}")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        with _lock:
            phase = _game["phase"]

        # Only run inference during scanning; other phases use frozen frame
        if phase != "scanning":
            time.sleep(0.03)
            continue

        result = model.predict(frame, conf=MODEL_CONF, iou=MODEL_IOU, verbose=False)[0]

        detections: list[tuple[int, float]] = []
        annotated = frame.copy()

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf   = float(box.conf.item())
                detections.append((cls_id, conf))
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                _draw_box(annotated, x1, y1, x2, y2, f"{name} {conf:.2f}")

        decision = engine.update(detections)

        with _lock:
            if decision["reason"] == "mapped_from_class":
                _game["consecutive_hits"] = min(_game["consecutive_hits"] + 1, LOCK_IN_FRAMES)
                if _game["consecutive_hits"] >= LOCK_IN_FRAMES and _game["phase"] == "scanning":
                    _, buf = cv2.imencode(".jpg", annotated)
                    _game["frozen_jpeg"] = buf.tobytes()
                    _game["ai_bin"]   = decision["final_bin"]
                    _game["ai_class"] = decision["top_class"]
                    _game["ai_score"] = float(decision["score"])
                    _game["phase"]    = "detected"
                    _game["consecutive_hits"] = 0
                    engine.reset()
            else:
                _game["consecutive_hits"] = max(0, _game["consecutive_hits"] - 1)

        _, buf = cv2.imencode(".jpg", annotated)
        with _frame_lock:
            _latest_jpeg = buf.tobytes()

    cap.release()


def _mjpeg_gen():
    while True:
        with _frame_lock:
            frame = _latest_jpeg
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.033)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           bin_colors=BIN_COLORS,
                           bin_labels=BIN_LABELS,
                           bin_icons=BIN_ICONS)


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/state")
def get_state():
    with _lock:
        g = dict(_game)
        raw = g.pop("frozen_jpeg", None)
        g["frozen_frame_b64"] = base64.b64encode(raw).decode() if raw else None
        g["detection_progress"] = int(g["consecutive_hits"] / LOCK_IN_FRAMES * 100)
    return jsonify(g)


@app.route("/player_answer", methods=["POST"])
def player_answer():
    data = request.get_json(force=True)
    chosen = data.get("bin")
    if chosen not in BIN_LABELS:
        return jsonify({"error": "invalid bin"}), 400
    with _lock:
        if _game["phase"] != "detected":
            return jsonify({"error": "wrong phase"}), 400
        _game["player_bin"] = chosen
        _game["phase"] = "player_chose"
    return jsonify({"ok": True})


@app.route("/reveal_ai", methods=["POST"])
def reveal_ai():
    with _lock:
        if _game["phase"] != "player_chose":
            return jsonify({"error": "wrong phase"}), 400
        _game["phase"] = "reveal"
    return jsonify({"ok": True})


@app.route("/set_result", methods=["POST"])
def set_result():
    data = request.get_json(force=True)
    outcome = data.get("outcome")
    if outcome not in ("prize", "raffle", "none"):
        return jsonify({"error": "invalid outcome"}), 400
    with _lock:
        if _game["phase"] != "reveal":
            return jsonify({"error": "wrong phase"}), 400
        _game["outcome"] = outcome
        _game["phase"]   = "result"
    return jsonify({"ok": True})


@app.route("/submit_raffle", methods=["POST"])
def submit_raffle():
    data  = request.get_json(force=True)
    name  = str(data.get("name",  "")).strip()
    email = str(data.get("email", "")).strip()
    if not name:
        return jsonify({"error": "name required"}), 400

    with _lock:
        outcome = _game.get("outcome")

    if outcome not in ("prize", "raffle"):
        return jsonify({"error": "not a winner"}), 400

    write_header = not RAFFLE_CSV.exists()
    with RAFFLE_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "name", "email", "outcome"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, email, outcome])

    return jsonify({"ok": True})


@app.route("/next_round", methods=["POST"])
def next_round():
    with _lock:
        _game.update({
            "phase": "scanning",
            "frozen_jpeg": None,
            "ai_bin": None,
            "ai_class": None,
            "ai_score": 0.0,
            "player_bin": None,
            "outcome": None,
            "consecutive_hits": 0,
        })
    return jsonify({"ok": True})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="REBIN tabling game")
    p.add_argument("--weights", default=WEIGHTS_DEFAULT)
    p.add_argument("--camera",  type=int, default=CAMERA_INDEX)
    p.add_argument("--port",    type=int, default=5000)
    args = p.parse_args()

    t = threading.Thread(target=camera_loop, args=(args.weights, args.camera), daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
