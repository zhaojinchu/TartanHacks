from pathlib import Path

from fastapi.testclient import TestClient


def _write_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    (config_dir / "bins.yaml").write_text(
        """
bins:
  - id: recycle_1
    type: recycle
    location: Test_Lab
    height_cm: 80
    sensor_offset_cm: 5
    gpio_trigger: 23
    gpio_echo: 24
measurement:
  interval_seconds: 999
  auto_start_collector: false
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (config_dir / "thresholds.yaml").write_text(
        """
normal_max: 70
warning_max: 85
full_min: 95
alert_critical: 85
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (config_dir / "sensors.yaml").write_text(
        f"""
mock_mode: true
database:
  path: {str((tmp_path / "test.db").resolve())}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return config_dir


def test_health_endpoint(monkeypatch, tmp_path: Path) -> None:
    config_dir = _write_config(tmp_path)
    monkeypatch.setenv("ULTRASONIC_CONFIG_DIR", str(config_dir))

    from src.main import app

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_bins_endpoint(monkeypatch, tmp_path: Path) -> None:
    config_dir = _write_config(tmp_path)
    monkeypatch.setenv("ULTRASONIC_CONFIG_DIR", str(config_dir))

    from src.main import app

    with TestClient(app) as client:
        response = client.get("/api/bins")
        assert response.status_code == 200
        payload = response.json()
        assert isinstance(payload, list)
        assert payload[0]["bin_id"] == "recycle_1"
