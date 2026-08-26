"""
Standalone manual test script for POST /detect — no test framework, no
pytest, just a plain script you run directly to eyeball the result.

Usage:
    python tests/test_detect_manual.py                    # generates a
                                                            # synthetic test
                                                            # image
    python tests/test_detect_manual.py path/to/photo.jpg   # uses a real
                                                            # image instead —
                                                            # do this to
                                                            # actually verify
                                                            # detection
                                                            # accuracy, not
                                                            # just that the
                                                            # pipeline runs

Requires the service to already be running (see README.md "Running the
service").
"""
import json
import sys
from io import BytesIO
from pathlib import Path

import requests

SERVICE_URL = "http://localhost:8000"


def make_synthetic_test_image() -> bytes:
    """A plain generated image so the pipeline can be exercised with zero
    external dependencies. This proves the mechanics (upload -> decode ->
    YOLO inference -> structured JSON) work end-to-end, but a synthetic
    drawn image has no real person/phone in it, so don't expect
    personCount/phoneDetected to be meaningful here — point this script at
    a real photo (see the argv[1] path below) to check actual accuracy."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 480), color=(40, 44, 52))
    draw = ImageDraw.Draw(img)
    draw.rectangle([200, 100, 440, 380], outline=(200, 200, 200), width=4)
    draw.ellipse([260, 140, 380, 260], outline=(200, 200, 200), width=4)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def main():
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if not image_path.exists():
            print(f"File not found: {image_path}")
            sys.exit(1)
        image_bytes = image_path.read_bytes()
        filename = image_path.name
        print(f"Using real image: {image_path}")
    else:
        image_bytes = make_synthetic_test_image()
        filename = "synthetic_test.jpg"
        print("No image path given — using a generated synthetic test image.")
        print("(This only proves the pipeline runs end-to-end. Pass a real")
        print(" photo path as argv[1] to verify actual detection accuracy.)")

    print(f"\nChecking {SERVICE_URL}/health ...")
    try:
        health = requests.get(f"{SERVICE_URL}/health", timeout=5)
        print(json.dumps(health.json(), indent=2))
    except requests.exceptions.ConnectionError:
        print(f"Could not reach {SERVICE_URL} — is the service running? See README.md.")
        sys.exit(1)

    print(f"\nPOSTing frame to {SERVICE_URL}/detect ...")
    response = requests.post(
        f"{SERVICE_URL}/detect",
        files={"file": (filename, image_bytes, "image/jpeg")},
        timeout=30,
    )

    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
