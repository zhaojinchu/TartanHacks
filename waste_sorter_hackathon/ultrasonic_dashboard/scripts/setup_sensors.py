#!/usr/bin/env python3
"""
Test GPIO sensor connections and verify ultrasonic sensors are working.
Run this script to debug sensor hardware before running the main application.
"""

import argparse
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    print("⚠️  RPi.GPIO not available. Running in simulation mode.")
    GPIO_AVAILABLE = False


def measure_distance(trigger_pin: int, echo_pin: int, timeout: float = 0.03) -> float:
    """
    Measure distance using HC-SR04 ultrasonic sensor.

    Args:
        trigger_pin: GPIO pin number for trigger
        echo_pin: GPIO pin number for echo
        timeout: Maximum time to wait for echo (seconds)

    Returns:
        Distance in centimeters, or -1 on error
    """
    if not GPIO_AVAILABLE:
        # Simulate reading for testing
        import random
        return round(random.uniform(10, 80), 2)

    try:
        # Send 10μs trigger pulse
        GPIO.output(trigger_pin, GPIO.HIGH)
        time.sleep(0.00001)  # 10 microseconds
        GPIO.output(trigger_pin, GPIO.LOW)

        # Wait for echo to start
        pulse_start = time.time()
        timeout_start = time.time()
        while GPIO.input(echo_pin) == 0:
            pulse_start = time.time()
            if pulse_start - timeout_start > timeout:
                print(f"  ⚠️  Timeout waiting for echo start")
                return -1

        # Wait for echo to end
        pulse_end = time.time()
        timeout_start = time.time()
        while GPIO.input(echo_pin) == 1:
            pulse_end = time.time()
            if pulse_end - timeout_start > timeout:
                print(f"  ⚠️  Timeout waiting for echo end")
                return -1

        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        # Speed of sound: 34300 cm/s, divide by 2 for round trip
        distance = (pulse_duration * 34300) / 2

        return round(distance, 2)

    except Exception as e:
        print(f"  ❌ Error measuring distance: {e}")
        return -1


def test_sensor(bin_id: str, trigger_pin: int, echo_pin: int, samples: int = 10):
    """Test a single ultrasonic sensor."""

    print(f"\n{'='*60}")
    print(f"Testing Sensor: {bin_id}")
    print(f"  Trigger Pin: GPIO {trigger_pin}")
    print(f"  Echo Pin: GPIO {echo_pin}")
    print(f"{'='*60}\n")

    if GPIO_AVAILABLE:
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(trigger_pin, GPIO.OUT)
        GPIO.setup(echo_pin, GPIO.IN)
        GPIO.output(trigger_pin, GPIO.LOW)
        print("⏳ Waiting for sensor to settle...")
        time.sleep(2)

    # Take multiple measurements
    distances = []
    print(f"\nTaking {samples} measurements:\n")

    for i in range(samples):
        distance = measure_distance(trigger_pin, echo_pin)

        if distance > 0:
            distances.append(distance)
            print(f"  [{i+1:2d}] {distance:6.2f} cm  {'✓' if distance < 400 else '⚠️  Out of range'}")
        else:
            print(f"  [{i+1:2d}] ERROR")

        time.sleep(0.5)

    # Calculate statistics
    if distances:
        avg = sum(distances) / len(distances)
        min_dist = min(distances)
        max_dist = max(distances)
        variance = sum((d - avg) ** 2 for d in distances) / len(distances)
        std_dev = variance ** 0.5

        print(f"\n{'─'*60}")
        print(f"Statistics:")
        print(f"  Average:   {avg:6.2f} cm")
        print(f"  Min:       {min_dist:6.2f} cm")
        print(f"  Max:       {max_dist:6.2f} cm")
        print(f"  Std Dev:   {std_dev:6.2f} cm")
        print(f"  Variance:  {variance:6.2f}")
        print(f"  Success:   {len(distances)}/{samples} ({len(distances)/samples*100:.0f}%)")

        # Health check
        print(f"\n{'─'*60}")
        if std_dev < 2:
            print("✅ Sensor readings are STABLE")
        elif std_dev < 5:
            print("⚠️  Sensor readings have MODERATE variation")
        else:
            print("❌ Sensor readings are UNSTABLE - check connections")

        if len(distances) == samples:
            print("✅ All measurements successful")
        else:
            print(f"⚠️  {samples - len(distances)} measurements failed")
    else:
        print("\n❌ NO VALID MEASUREMENTS - Check:")
        print("   - GPIO pin numbers are correct")
        print("   - Sensor is powered (VCC=5V, GND)")
        print("   - Wiring connections are secure")
        print("   - No obstacles blocking sensor")


def test_all_sensors_from_config():
    """Load bins.yaml and test all configured sensors."""
    import yaml

    config_path = Path(__file__).parent.parent / "config" / "bins.yaml"

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    bins = config.get('bins', [])

    if not bins:
        print("❌ No bins configured in bins.yaml")
        return

    print(f"\n🔍 Found {len(bins)} bins in configuration\n")

    for bin_config in bins:
        if not bin_config.get('enabled', True):
            print(f"⏭️  Skipping disabled bin: {bin_config['id']}")
            continue

        test_sensor(
            bin_id=bin_config['id'],
            trigger_pin=bin_config['gpio_trigger'],
            echo_pin=bin_config['gpio_echo'],
            samples=5
        )

        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(
        description="Test ultrasonic sensors for bin monitoring system"
    )
    parser.add_argument(
        "--bin-id",
        help="Specific bin ID to test (otherwise test all from config)"
    )
    parser.add_argument(
        "--trigger",
        type=int,
        help="GPIO trigger pin number"
    )
    parser.add_argument(
        "--echo",
        type=int,
        help="GPIO echo pin number"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of measurements to take (default: 10)"
    )

    args = parser.parse_args()

    try:
        if args.trigger and args.echo:
            # Test specific sensor
            bin_id = args.bin_id or f"Sensor_GPIO{args.trigger}/{args.echo}"
            test_sensor(bin_id, args.trigger, args.echo, args.samples)
        else:
            # Test all sensors from config
            test_all_sensors_from_config()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")

    finally:
        if GPIO_AVAILABLE:
            GPIO.cleanup()
            print("\n🧹 GPIO cleaned up\n")


if __name__ == "__main__":
    main()
