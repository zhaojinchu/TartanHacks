from src.sensors.data_collector import calculate_fullness, resolve_status


def test_calculate_fullness_midpoint() -> None:
    fullness = calculate_fullness(distance_cm=45.0, bin_height_cm=80.0, sensor_offset_cm=5.0)
    assert fullness == 50.0


def test_calculate_fullness_clamped() -> None:
    assert calculate_fullness(distance_cm=999.0, bin_height_cm=80.0, sensor_offset_cm=5.0) == 0.0
    assert calculate_fullness(distance_cm=0.0, bin_height_cm=80.0, sensor_offset_cm=5.0) == 100.0


def test_status_resolution() -> None:
    assert resolve_status(60, normal_max=70, warning_max=85, full_min=95) == "normal"
    assert resolve_status(78, normal_max=70, warning_max=85, full_min=95) == "almost_full"
    assert resolve_status(88, normal_max=70, warning_max=85, full_min=95) == "full"
    assert resolve_status(None, normal_max=70, warning_max=85, full_min=95) == "sensor_error"
