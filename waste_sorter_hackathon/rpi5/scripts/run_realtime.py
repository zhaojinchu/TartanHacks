"""Real-time Raspberry Pi 5 inference using ONNX Runtime + OpenCV."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import subprocess
import sys
import threading
import time
from http import server
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import TemporalDecisionEngine
from src.io_utils import load_decision_config, resolve_class_names
from src.onnx_detector import ONNXWasteDetector


class MjpegStreamServer:
    """Very small MJPEG HTTP server for browser viewing."""

    def __init__(
        self,
        host: str,
        port: int,
        use_tls: bool = False,
        certfile: Path | None = None,
        keyfile: Path | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.use_tls = use_tls
        self._frame_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._active_clients = 0
        self._total_clients = 0
        self._disconnects = 0
        self._errors = 0

        parent = self

        class Handler(server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path in ("/", "/index.html"):
                    body = (
                        "<html><head><title>Waste Sorter Stream</title></head>"
                        "<body style='margin:0;background:#111;'>"
                        "<img src='/stream' style='width:100%;height:auto;'/>"
                        "</body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/stream":
                    parent._on_client_connect()
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame",
                    )
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Pragma", "no-cache")
                    self.end_headers()

                    try:
                        while True:
                            frame = parent.get_frame()
                            if frame is None:
                                time.sleep(0.03)
                                continue

                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                            time.sleep(0.01)
                    except (BrokenPipeError, ConnectionResetError):
                        parent._on_client_disconnect(disconnected=True)
                        return
                    except Exception:
                        parent._on_client_disconnect(error=True)
                        return
                    parent._on_client_disconnect(disconnected=True)
                    return

                self.send_response(404)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = server.ThreadingHTTPServer((host, port), Handler)
        if use_tls:
            if certfile is None or keyfile is None:
                raise ValueError("TLS enabled but certfile/keyfile not provided")
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile=str(certfile), keyfile=str(keyfile))
            self._server.socket = context.wrap_socket(self._server.socket, server_side=True)

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def update_frame(self, frame_jpeg: bytes) -> None:
        with self._lock:
            self._frame_jpeg = frame_jpeg

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._frame_jpeg

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1.0)

    def _on_client_connect(self) -> None:
        with self._stats_lock:
            self._active_clients += 1
            self._total_clients += 1

    def _on_client_disconnect(self, disconnected: bool = False, error: bool = False) -> None:
        with self._stats_lock:
            self._active_clients = max(self._active_clients - 1, 0)
            if disconnected:
                self._disconnects += 1
            if error:
                self._errors += 1

    def get_stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {
                "active_clients": self._active_clients,
                "total_clients": self._total_clients,
                "disconnects": self._disconnects,
                "errors": self._errors,
            }


class OpenCVCamera:
    """OpenCV-based camera source (USB webcams or V4L2 devices)."""

    def __init__(self, camera_index: int, width: int, height: int, fps: float) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {camera_index} with OpenCV")

        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))

    def read(self) -> tuple[bool, Any]:
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()

    def output_meta(self, fallback_width: int, fallback_height: int, fallback_fps: float) -> tuple[int, int, float]:
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or fallback_width)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or fallback_height)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or fallback_fps)
        if fps <= 0:
            fps = fallback_fps
        return width, height, fps


class PiCamera2Source:
    """Picamera2-based source for Raspberry Pi Camera Module 2/3 via libcamera."""

    def __init__(self, camera_index: int, width: int, height: int, fps: float) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 is not installed. Install on Raspberry Pi OS with:\n"
                "  sudo apt update && sudo apt install -y python3-picamera2\n"
                "If using venv, recreate with --system-site-packages."
            ) from exc

        self.width = width
        self.height = height
        self.fps = fps if fps > 0 else 30.0

        self.picam2 = Picamera2(camera_index)
        controls: dict[str, Any] = {}
        if self.fps > 0:
            controls["FrameRate"] = float(self.fps)

        config = self.picam2.create_preview_configuration(
            main={"format": "BGR888", "size": (self.width, self.height)},
            controls=controls,
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.2)

    def read(self) -> tuple[bool, Any]:
        frame = self.picam2.capture_array()
        if frame is None:
            return False, None

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return True, frame

    def release(self) -> None:
        self.picam2.stop()
        self.picam2.close()

    def output_meta(self, fallback_width: int, fallback_height: int, fallback_fps: float) -> tuple[int, int, float]:
        return (
            int(self.width or fallback_width),
            int(self.height or fallback_height),
            float(self.fps or fallback_fps),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-time ONNX inference on Raspberry Pi 5 webcam."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/waste_sorter.onnx"),
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--decision_config",
        type=Path,
        default=Path("configs/decision.yaml"),
        help="Path to decision YAML",
    )
    parser.add_argument(
        "--class_names_yaml",
        type=Path,
        default=None,
        help="Optional YAML with `names` list/dict (fallback if ONNX metadata missing).",
    )
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument(
        "--camera_backend",
        type=str,
        default="auto",
        choices=["auto", "picamera2", "opencv"],
        help="Camera backend. Use `picamera2` for Raspberry Pi Camera Module 2.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera_fps", type=float, default=30.0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--ort_intra_threads",
        type=int,
        default=2,
        help="ONNX Runtime intra-op CPU threads (0=default).",
    )
    parser.add_argument(
        "--ort_inter_threads",
        type=int,
        default=1,
        help="ONNX Runtime inter-op CPU threads (0=default).",
    )
    parser.add_argument("--window", type=int, default=None, help="Override smoothing window")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override unknown threshold",
    )
    parser.add_argument("--save_path", type=Path, default=None)
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Disable GUI window (headless mode)",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=30,
        help="Print status every N frames",
    )
    parser.add_argument(
        "--http_stream",
        action="store_true",
        help="Enable browser stream at http://host:port/ (MJPEG).",
    )
    parser.add_argument(
        "--https_stream",
        action="store_true",
        help="Enable browser stream over HTTPS (TLS).",
    )
    parser.add_argument("--http_host", type=str, default="0.0.0.0")
    parser.add_argument("--http_port", type=int, default=8080)
    parser.add_argument(
        "--tls_cert",
        type=Path,
        default=Path("rpi5/certs/cert.pem"),
        help="TLS certificate path (PEM) for --https_stream.",
    )
    parser.add_argument(
        "--tls_key",
        type=Path,
        default=Path("rpi5/certs/key.pem"),
        help="TLS private key path (PEM) for --https_stream.",
    )
    parser.add_argument(
        "--tls_self_signed",
        action="store_true",
        help="Auto-generate self-signed TLS cert/key if missing.",
    )
    parser.add_argument(
        "--http_quality",
        type=int,
        default=80,
        help="JPEG quality [1..100] for HTTP stream.",
    )
    parser.add_argument(
        "--health_every",
        type=int,
        default=30,
        help="Print runtime health stats every N frames (0 to disable).",
    )
    parser.add_argument(
        "--health_log",
        type=Path,
        default=None,
        help="Optional path to append health stats as JSONL.",
    )
    return parser.parse_args()


def open_camera_source(args: argparse.Namespace) -> tuple[Any, str]:
    """Open camera with requested backend and return (source, backend_name)."""
    if args.camera_backend in ("picamera2", "auto"):
        try:
            source = PiCamera2Source(
                camera_index=args.camera_index,
                width=args.width,
                height=args.height,
                fps=args.camera_fps,
            )
            return source, "picamera2"
        except Exception as exc:
            if args.camera_backend == "picamera2":
                raise RuntimeError(f"Failed to open Pi camera with picamera2: {exc}") from exc
            print(f"Warning: picamera2 backend unavailable, falling back to OpenCV. ({exc})")

    try:
        source = OpenCVCamera(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
        )
        return source, "opencv"
    except Exception as exc:
        raise RuntimeError(
            "Failed to open camera with both picamera2 and OpenCV backends.\n"
            "For Raspberry Pi Camera Module 2, install picamera2 and use --camera_backend picamera2."
        ) from exc


def read_vcgencmd(key: str) -> str | None:
    """Read vcgencmd output if available, else return None."""
    try:
        result = subprocess.run(
            ["vcgencmd", key],
            capture_output=True,
            text=True,
            check=True,
            timeout=1.0,
        )
    except Exception:
        return None
    return result.stdout.strip()


def ensure_tls_material(
    cert_path: Path,
    key_path: Path,
    allow_self_signed: bool,
    host: str,
) -> tuple[Path, Path]:
    """Ensure TLS certificate and key exist for HTTPS streaming."""
    cert_path = cert_path.expanduser().resolve()
    key_path = key_path.expanduser().resolve()

    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    if not allow_self_signed:
        raise FileNotFoundError(
            "HTTPS requested but TLS cert/key not found.\n"
            f"Expected cert: {cert_path}\n"
            f"Expected key:  {key_path}\n"
            "Provide --tls_cert/--tls_key or use --tls_self_signed."
        )

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    cn = host if host not in ("0.0.0.0", "::") else "localhost"
    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-keyout",
        str(key_path),
        "-out",
        str(cert_path),
        "-days",
        "365",
        "-nodes",
        "-subj",
        f"/CN={cn}",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15.0)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "openssl not found. Install openssl or provide existing --tls_cert and --tls_key."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to generate self-signed TLS cert with openssl:\n"
            f"{exc.stderr}"
        ) from exc

    try:
        os.chmod(key_path, 0o600)
    except Exception:
        pass

    print(f"Generated self-signed cert: {cert_path}")
    print(f"Generated TLS key: {key_path}")
    return cert_path, key_path


def get_temp_c() -> float | None:
    """Read CPU temperature in Celsius."""
    thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        if thermal_path.exists():
            raw = thermal_path.read_text(encoding="utf-8").strip()
            return float(raw) / 1000.0
    except Exception:
        pass

    out = read_vcgencmd("measure_temp")
    if out and "temp=" in out:
        try:
            return float(out.split("temp=")[1].split("'")[0])
        except Exception:
            return None
    return None


def get_mem_available_mb() -> float | None:
    """Read available memory from /proc/meminfo."""
    meminfo = Path("/proc/meminfo")
    try:
        if not meminfo.exists():
            return None
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    kb = float(parts[1])
                    return kb / 1024.0
    except Exception:
        return None
    return None


def get_uptime_s() -> float | None:
    """Read system uptime in seconds."""
    uptime_path = Path("/proc/uptime")
    try:
        if not uptime_path.exists():
            return None
        first = uptime_path.read_text(encoding="utf-8").split()[0]
        return float(first)
    except Exception:
        return None


def get_health_snapshot() -> dict[str, object]:
    """Collect lightweight health metrics for debugging crashes/resets."""
    temp_c = get_temp_c()
    mem_mb = get_mem_available_mb()
    throttled = read_vcgencmd("get_throttled")
    uptime_s = get_uptime_s()

    snapshot: dict[str, object] = {
        "temp_c": round(temp_c, 2) if temp_c is not None else None,
        "mem_available_mb": round(mem_mb, 1) if mem_mb is not None else None,
        "throttled": throttled,
        "uptime_s": round(uptime_s, 1) if uptime_s is not None else None,
    }
    return snapshot


def draw_overlay(
    frame,
    detections,
    class_names: list[str],
    decision: dict[str, object],
    fps_ema: float,
) -> None:
    """Draw detections, routing decision, and FPS on frame."""
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        if 0 <= det.class_id < len(class_names):
            class_name = class_names[det.class_id]
        else:
            class_name = str(det.class_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame,
            f"{class_name} {det.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    lines = [
        f"final_bin: {decision['final_bin']}",
        f"top_class: {decision['top_class']} ({decision['score']:.2f})",
        f"reason: {decision['reason']}",
        f"fps: {fps_ema:.1f}",
    ]

    for i, text in enumerate(lines):
        y = 28 + i * 24
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (20, 230, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    if not (1 <= args.http_quality <= 100):
        raise ValueError("--http_quality must be in [1,100]")
    if args.ort_intra_threads < 0 or args.ort_inter_threads < 0:
        raise ValueError("--ort_intra_threads and --ort_inter_threads must be >= 0")
    if args.http_stream and args.https_stream:
        raise ValueError("Use only one of --http_stream or --https_stream")

    fallback_class_names = resolve_class_names(
        class_names=None,
        class_names_yaml=args.class_names_yaml,
    )

    detector = ONNXWasteDetector(
        model_path=args.model,
        class_names=fallback_class_names,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        intra_op_threads=args.ort_intra_threads,
        inter_op_threads=args.ort_inter_threads,
    )
    runtime_class_names = detector.class_names

    class_to_bin, cfg_threshold, cfg_window = load_decision_config(
        args.decision_config,
        class_names=runtime_class_names,
    )
    threshold = args.threshold if args.threshold is not None else cfg_threshold
    window_size = args.window if args.window is not None else cfg_window

    print(f"Runtime classes: {runtime_class_names}")

    decision_engine = TemporalDecisionEngine(
        class_to_bin=class_to_bin,
        threshold=threshold,
        window_size=window_size,
        class_names=runtime_class_names,
    )

    camera, backend_name = open_camera_source(args)
    print(f"Camera backend: {backend_name}")

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    display_enabled = not args.no_display and has_display
    if not args.no_display and not has_display:
        print("No GUI display detected. Running in headless mode (display disabled).")

    stream_server = None
    if args.http_stream or args.https_stream:
        use_tls = bool(args.https_stream)
        cert_path = None
        key_path = None
        if use_tls:
            cert_path, key_path = ensure_tls_material(
                cert_path=args.tls_cert,
                key_path=args.tls_key,
                allow_self_signed=args.tls_self_signed,
                host=args.http_host,
            )

        stream_server = MjpegStreamServer(
            host=args.http_host,
            port=args.http_port,
            use_tls=use_tls,
            certfile=cert_path,
            keyfile=key_path,
        )
        stream_server.start()
        scheme = "https" if use_tls else "http"
        print(f"Stream URL: {scheme}://{args.http_host}:{args.http_port}/")

    writer = None
    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        out_w, out_h, out_fps = camera.output_meta(
            fallback_width=args.width,
            fallback_height=args.height,
            fallback_fps=args.camera_fps if args.camera_fps > 0 else 20.0,
        )
        if out_fps <= 0:
            out_fps = 20.0

        writer = cv2.VideoWriter(
            str(args.save_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps,
            (out_w, out_h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open writer: {args.save_path}")

    health_log_file = None
    if args.health_log is not None:
        args.health_log.parent.mkdir(parents=True, exist_ok=True)
        health_log_file = args.health_log.open("a", encoding="utf-8")

    print("Running real-time inference. Press 'q' to quit.")

    frame_count = 0
    fps_ema = 0.0
    fps_alpha = 0.2
    encode_ms_ema = 0.0
    encode_alpha = 0.2
    prev_t = time.perf_counter()
    last_decision: dict[str, object] | None = None

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Warning: camera read failed, stopping.")
                break

            detections = detector.predict(frame)
            decision_input = [(det.class_id, det.confidence) for det in detections]
            last_decision = decision_engine.update(decision_input)

            now = time.perf_counter()
            dt = max(now - prev_t, 1e-6)
            prev_t = now
            fps = 1.0 / dt
            fps_ema = fps if fps_ema == 0.0 else ((1 - fps_alpha) * fps_ema + fps_alpha * fps)

            draw_overlay(frame, detections, runtime_class_names, last_decision, fps_ema)

            if stream_server is not None:
                enc_t0 = time.perf_counter()
                ok_jpg, jpg = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(args.http_quality)],
                )
                enc_ms = (time.perf_counter() - enc_t0) * 1000.0
                encode_ms_ema = (
                    enc_ms
                    if encode_ms_ema == 0.0
                    else (1 - encode_alpha) * encode_ms_ema + encode_alpha * enc_ms
                )
                if ok_jpg:
                    stream_server.update_frame(jpg.tobytes())

            if writer is not None:
                writer.write(frame)

            if display_enabled:
                cv2.imshow("waste_sorter_rpi5", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if args.print_every > 0 and frame_count % args.print_every == 0:
                msg = (
                    f"frame={frame_count} fps={fps_ema:.1f} "
                    f"detections={len(detections)} bin={last_decision['final_bin']} "
                    f"class={last_decision['top_class']} score={last_decision['score']:.2f}"
                )
                if stream_server is not None:
                    stream_stats = stream_server.get_stats()
                    msg += (
                        " "
                        f"clients={stream_stats['active_clients']} "
                        f"enc_ms={encode_ms_ema:.1f} "
                        f"stream_err={stream_stats['errors']}"
                    )
                if args.health_every > 0 and frame_count % args.health_every == 0:
                    health = get_health_snapshot()
                    msg += (
                        " "
                        f"temp_c={health['temp_c']} "
                        f"mem_avail_mb={health['mem_available_mb']} "
                        f"throttled={health['throttled']}"
                    )
                    if health_log_file is not None:
                        health_row = {
                            "time": time.time(),
                            "frame": frame_count,
                            "fps": round(fps_ema, 2),
                            "detections": len(detections),
                            "final_bin": last_decision["final_bin"],
                            "top_class": last_decision["top_class"],
                            "score": round(float(last_decision["score"]), 4),
                            "health": health,
                        }
                        if stream_server is not None:
                            health_row["stream"] = {
                                "encode_ms_ema": round(encode_ms_ema, 3),
                                **stream_server.get_stats(),
                            }
                        health_log_file.write(json.dumps(health_row) + "\n")
                        health_log_file.flush()
                print(msg)
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping...")
    finally:
        camera.release()
        if writer is not None:
            writer.release()
        if health_log_file is not None:
            health_log_file.close()
        if stream_server is not None:
            stream_server.stop()
        if display_enabled:
            cv2.destroyAllWindows()

    print(f"Processed frames: {frame_count}")
    if last_decision is not None:
        print("Last decision:")
        print(json.dumps(last_decision, indent=2))


if __name__ == "__main__":
    main()
