#!/usr/bin/env python3
"""Quick test to diagnose Pi Camera 2 color issues."""

import time
from picamera2 import Picamera2
import cv2

print("Testing Pi Camera 2 color modes...")
print("=" * 60)

picam2 = Picamera2()

# Test different AWB modes
awb_modes = ["auto", "tungsten", "fluorescent", "indoor", "daylight", "cloudy"]

for mode in awb_modes:
    print(f"\n--- Testing AWB mode: {mode} ---")

    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)},
        controls={"AwbMode": mode}
    )

    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Let AWB settle

    # Capture frame
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Save test image
    filename = f"/tmp/test_awb_{mode}.jpg"
    cv2.imwrite(filename, frame_bgr)
    print(f"Saved: {filename}")

    # Calculate average RGB values
    avg_b = frame[:, :, 2].mean()
    avg_g = frame[:, :, 1].mean()
    avg_r = frame[:, :, 0].mean()

    print(f"Average RGB: R={avg_r:.1f} G={avg_g:.1f} B={avg_b:.1f}")
    print(f"R/B ratio: {avg_r/avg_b:.2f} (should be ~1.0-1.2 for neutral)")

    picam2.stop()
    time.sleep(0.5)

picam2.close()
print("\n" + "=" * 60)
print("Test complete! Check /tmp/test_awb_*.jpg images")
print("Look for the AWB mode where R/B ratio is closest to 1.0-1.2")
