from src.sensors.ultrasonic import SerialLineParser


def test_parse_arduino_verbose_line() -> None:
    parsed = SerialLineParser.parse("Ultrasonic Sensor 2 Distance: 34.5")
    assert parsed == (2, 34.5)


def test_parse_compact_line() -> None:
    parsed = SerialLineParser.parse("S3:17.25")
    assert parsed == (3, 17.25)


def test_parse_invalid_line_returns_none() -> None:
    assert SerialLineParser.parse("Unknown command: O0") is None
