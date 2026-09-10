"""
Camera-covered / visibility detection — an image-quality heuristic, NOT a
YOLO object-detection class.

There is no "covered camera" object for any detector to learn — a hand, a
lens cap, a sticky note, or a laptop lid can all obstruct a camera and none
of them share a visual class. This is a stateless signal-analysis function
over the same decoded frame YOLO and the explicit-content detector already
use, not a trained model.

Deliberately NOT `if brightness < threshold: covered = True` — a dark room,
night-time use, or normal camera exposure would all false-positive under
that rule. Instead this combines two independent signals:

  - brightness: mean grayscale pixel value (0-255)
  - "flatness": pixel-value standard deviation AND Laplacian variance (a
    standard blur/edge-detail measure) both being very low, meaning the
    frame is close to a single uniform color with no real detail anywhere

A physically obstructed lens (finger, cap, tape, palm pressed to the
camera) produces a FLAT frame regardless of its brightness. A dark room
does NOT — dim real content still has visible noise/edges. Combining both
signals is what lets this tell "very dark but real" apart from "blocked."

Confidence values here are fixed, conservative bands describing how
strongly the combined signals point one way — not a learned probability.
Ambiguous cases return "uncertain" rather than guessing "covered".
"""
import logging

import cv2
import numpy as np

from app.core.config import get_settings

logger = logging.getLogger("[VISION]")

settings = get_settings()


def analyze(image_array: np.ndarray) -> dict:
    """Runs the visibility heuristic on an already-decoded RGB frame.

    Never raises — a bad frame degrades to "uncertain", it never takes down
    /detect for YOLO or explicit-content. Returns:
        {"status": "covered" | "clear" | "uncertain", "confidence": float}
    """
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        mean_brightness = float(gray.mean())
        stddev = float(gray.std())
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        is_flat = stddev < settings.camera_visibility_flat_stddev_threshold and (
            laplacian_var < settings.camera_visibility_flat_detail_threshold
        )
        is_dark = mean_brightness < settings.camera_visibility_dark_threshold

        if is_flat and is_dark:
            # Could be a covered lens in a dark room, or could just be a
            # genuinely dark/empty room — the two look identical on these
            # signals alone. Conservative choice: say so, don't guess.
            status, confidence = "uncertain", 0.35
        elif is_flat and not is_dark:
            # No real detail AND not dark — the clearest signature of
            # something opaque pressed against or very close to the lens.
            status, confidence = "covered", 0.70
        elif is_dark and not is_flat:
            # Dim, but real detail/sensor noise is present — a genuinely
            # visible (if dark) scene, not an obstruction.
            status, confidence = "clear", 0.55
        else:
            status, confidence = "clear", 0.85

        logger.info(f"[Camera] status={status} confidence={confidence}")
        logger.info(
            f"[Camera] brightness={round(mean_brightness, 1)} flat={is_flat} "
            f"(stddev={round(stddev, 1)}, detail={round(laplacian_var, 1)})"
        )

        return {"status": status, "confidence": confidence}
    except Exception as exc:  # noqa: BLE001 - a malformed frame must degrade, not crash /detect
        logger.error(f"[Camera] status=uncertain reason=analysis_failed ({exc.__class__.__name__})")
        return {"status": "uncertain", "confidence": 0.0}
