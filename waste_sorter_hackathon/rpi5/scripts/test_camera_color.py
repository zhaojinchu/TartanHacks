#!/usr/bin/env python3
"""Quick test to diagnose Pi Camera 2 color issues."""

import time
from picamera2 import Picamera2
from libcamera import controls
import cv2

print("Testing Pi Camera 2 color modes...")
print("=" * 60)

picam2 = Picamera2()

# Test different AWB modes using libcamera enums
awb_modes = [
    ("Auto", controls.AwbModeEnum.Auto),
    ("Tungsten", controls.AwbModeEnum.Tungsten),
    ("Fluorescent", controls.AwbModeEnum.Fluorescent),
    ("Indoor", controls.AwbModeEnum.Indoor),
    ("Daylight", controls.AwbModeEnum.Daylight),
    ("Cloudy", controls.AwbModeEnum.Cloudy),
]

for mode_name, mode_value in awb_modes:
    print(f"\n--- Testing AWB mode: {mode_name} ---")

    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)},
        controls={"AwbMode": mode_value}
    )

    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Let AWB settle

    # Capture frame
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Save test image
    filename = f"/tmp/test_awb_{mode_name.lower()}.jpg"
    cv2.imwrite(filename, frame_bgr)
    print(f"Saved: {filename}")

    # Calculate average RGB values
    avg_b = frame[:, :, 2].mean()
    avg_g = frame[:, :, 1].mean()
    avg_r = frame[:, :, 0].mean()

    print(f"Average RGB: R={avg_r:.1f} G={avg_g:.1f} B={avg_b:.1f}")
    if avg_b > 0:
        print(f"R/B ratio: {avg_r/avg_b:.2f} (should be ~1.0-1.2 for neutral)")
    else:
        print("R/B ratio: N/A (blue channel is 0)")

    picam2.stop()
    time.sleep(0.5)

picam2.close()
print("\n" + "=" * 60)
print("Test complete! Check /tmp/test_awb_*.jpg images")
print("Look for the AWB mode where R/B ratio is closest to 1.0-1.2")
