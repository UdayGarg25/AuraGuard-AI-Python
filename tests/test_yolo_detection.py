"""
Standalone YOLO detection test.

This does NOT start the FastAPI server and does NOT create a second YOLO
implementation — it imports and calls app/services/vision_service.py
directly, the exact same functions app/main.py and app/api/detect.py already
use. This only verifies the already-implemented vision service works.

USAGE:
    1. Edit IMAGE_PATH below to point at a real image on your computer.
    2. Run from the project root:
           python tests/test_yolo_detection.py
"""
import sys
from pathlib import Path

# Make the `app` package importable when this file is run directly. Python
# only adds this script's own folder (tests/) to sys.path by default, not
# the project root `app` actually lives under — without this, `from app...`
# would fail with ModuleNotFoundError regardless of which directory you run
# the command from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# EDIT THIS — paste the path to the image you want to test
#
# Windows paths contain backslashes, which Python normally treats as escape
# characters (e.g. \t becomes a tab). Use an r"..." string like below, or
# switch the backslashes to forward slashes ("C:/Users/...") — either
# avoids that. Plain "C:\Users\..." will fail to even parse.
# ============================================================
IMAGE_PATH = r"C:\Users\HP\OneDrive\Desktop\AuraGuard-AI-Python\tests\test2.webp"
# ============================================================


def print_banner(title: str) -> None:
    print("=" * 40)
    print(title.center(40))
    print("=" * 40)


def print_results(result: dict) -> None:
    detections = result["detections"]
    objects = result["objects"]

    print("Detection Results:")
    print(f"- Person count: {detections['personCount']}")
    print(f"- Phone detected: {'YES' if detections['phoneDetected'] else 'NO'}")
    if detections["phoneConfidence"] is not None:
        print(f"- Phone confidence: {detections['phoneConfidence'] * 100:.1f}%")
    print()

    print("Detected Objects:")
    if not objects:
        print("No supported objects detected.")
    else:
        for obj in objects:
            print(f"- {obj['class']} | confidence: {obj['confidence'] * 100:.1f}%")
    print()

    print(f"Model used: {result['model']}")
    print(f"Processing time: {result['processingTimeMs']} ms")


def main() -> None:
    print_banner("AURAGUARD YOLO TEST")
    print()

    # --- Case: IMAGE_PATH was never changed from the placeholder -----------
    if not IMAGE_PATH.strip() or IMAGE_PATH == "PASTE_YOUR_IMAGE_PATH_HERE":
        print("ERROR: IMAGE_PATH has not been set.")
        print("Open tests/test_yolo_detection.py and change this line:")
        print('    IMAGE_PATH = "PASTE_YOUR_IMAGE_PATH_HERE"')
        print("to the path of a real image on your computer, then run this script again.")
        return

    image_path = Path(IMAGE_PATH)

    # --- Case: file does not exist ------------------------------------------
    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}")
        print("Check the path in IMAGE_PATH and try again.")
        return

    if not image_path.is_file():
        print(f"ERROR: Path exists but is not a file: {image_path}")
        return

    print(f"Image: {image_path}")
    print()

    # --- Import the EXISTING vision service (no new/second YOLO here) ------
    try:
        from app.services import vision_service
    except ImportError as exc:
        print("ERROR: Could not import the vision service.")
        print(f"Details: {exc}")
        print("Make sure dependencies are installed and you're running this from")
        print("the project's virtual environment (activate it, then re-run).")
        return

    # --- Case: YOLO model cannot be loaded ------------------------------------
    print("Loading YOLO model (same model/config the running service uses)...")
    try:
        vision_service.load_models()
    except Exception as exc:  # noqa: BLE001 - want a readable message, not a raw traceback
        print("ERROR: YOLO model could not be loaded.")
        print(f"Details: {exc}")
        return

    if not vision_service.is_ready():
        print("ERROR: Model did not finish loading.")
        return

    # --- Read the image file --------------------------------------------------
    try:
        raw_bytes = image_path.read_bytes()
    except Exception as exc:  # noqa: BLE001
        print("ERROR: Could not read the image file.")
        print(f"Details: {exc}")
        return

    # --- Run the EXISTING detection logic (reused, not reimplemented) --------
    print("Running detection...")
    print()
    try:
        result = vision_service.run_detection(raw_bytes)
    except vision_service.VisionServiceError as exc:
        # Covers "invalid/corrupt image" and "detection throws an exception" —
        # vision_service.py already turns both into this clean error type.
        print(f"ERROR: Detection failed — {exc.message}")
        return
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        print("ERROR: An unexpected error occurred during detection.")
        print(f"Details: {exc}")
        return

    print_results(result)
    print()
    print_banner("TEST COMPLETE")


if __name__ == "__main__":
    main()
