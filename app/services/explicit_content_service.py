"""
Explicit-content detection — a separate, purpose-built classifier.

YOLOv8n (app/services/vision_service.py) is a general-purpose COCO object
detector. It has no NSFW/nudity class and was never trained to recognize
one — using it for this would be exactly the "pretend the model supports a
class it wasn't trained for" mistake this project explicitly rejects
elsewhere (see the face-detection comments in vision_service.py).

Model: NudeNet (https://pypi.org/project/nudenet/), a small ONNX object
detector purpose-built for this task. Chosen over opennsfw2 (the other
common local option) because opennsfw2 requires TensorFlow — an entire
second deep-learning framework alongside the PyTorch stack Ultralytics
already needs, for one classifier. NudeNet runs on `onnxruntime`, which was
already present in this environment as an Ultralytics transitive
dependency, so it adds no new heavy runtime.

Runs fully locally: the ONNX checkpoint (320n.onnx, a few MB) ships inside
the `nudenet` pip package itself. No download step, no network call at
runtime, no third-party API, no API key.

Privacy: `NudeDetector.detect()` is called on the same in-memory numpy
array vision_service already decoded — nothing is written to disk, and
nothing about the image is logged, only aggregate counts/labels.
"""
import logging
from typing import Optional

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger("[VISION]")

settings = get_settings()

# The model's own label set includes several non-explicit classes (faces,
# and "*_COVERED" variants meaning clothed) — only true nudity classes count
# as "explicit" for a meeting-safety context. Feet/armpits/belly are
# deliberately excluded too: NudeNet reports these as "*_EXPOSED" whenever
# they're simply visible (bare feet, a t-shirt showing belly), which is
# completely normal on a video call and not sexually explicit.
_EXPLICIT_LABELS = frozenset(
    {
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "ANUS_EXPOSED",
    }
)

_detector: Optional["object"] = None
_load_error: Optional[str] = None


def load_model() -> None:
    """Called once from the FastAPI startup lifespan, alongside YOLO. Wrapped
    defensively — if this fails, the rest of the vision service (YOLO,
    camera-visibility) must keep working; explicit-content just reports
    'not_available' instead of crashing the process."""
    global _detector, _load_error

    if not settings.explicit_content_enabled:
        logger.info("Explicit-content detection disabled via config — skipping model load")
        return

    try:
        from nudenet import NudeDetector

        _detector = NudeDetector()
        logger.info("Explicit-content model loaded (NudeNet, local ONNX checkpoint)")
    except Exception as exc:  # noqa: BLE001 - a missing/broken dependency must not take down the service
        _load_error = f"{exc.__class__.__name__}: {exc}"
        logger.error(f"Explicit-content model failed to load — feature will report 'not_available': {_load_error}")
        _detector = None


def is_ready() -> bool:
    return _detector is not None


def analyze(image_array: np.ndarray) -> dict:
    """Runs the explicit-content classifier on an already-decoded RGB frame.

    Never raises — any failure here must not take down /detect for YOLO or
    camera-visibility. Returns the exact shape documented in the API:
        {"status": ..., "flagged": bool, "confidence": float | None}
    """
    if not settings.explicit_content_enabled:
        logger.info("[Explicit] status=not_available reason=disabled_via_config")
        return {"status": "not_available", "flagged": False, "confidence": None}

    if not is_ready():
        reason = _load_error or "model_not_loaded"
        logger.info(f"[Explicit] status=not_available reason={reason}")
        return {"status": "not_available", "flagged": False, "confidence": None}

    logger.info("[Explicit] Running NudeNet...")

    try:
        # NudeDetector accepts a BGR numpy array (like cv2.imread) or a path.
        # vision_service decodes frames as RGB (via PIL), so convert channel
        # order here rather than asking vision_service to change its shared
        # decode step for this one consumer.
        bgr_array = image_array[:, :, ::-1]
        detections = _detector.detect(bgr_array)

        threshold = settings.explicit_content_confidence_threshold
        # TEMP DEBUG: raw NudeNet counts only (no image data) to tell apart
        # "detected nothing" from "detected something below threshold".
        raw_max = max((d.get("score", 0.0) for d in detections), default=0.0)
        logger.info(
            f"[Explicit][debug] modelLoaded={is_ready()} rawDetections={len(detections)} "
            f"rawMaxScore={round(raw_max, 4)} threshold={threshold}"
        )
        matches = [
            d for d in detections if d.get("class") in _EXPLICIT_LABELS and d.get("score", 0.0) >= threshold
        ]

        flagged = len(matches) > 0
        confidence = round(max((d["score"] for d in matches), default=0.0), 4)

        # A concise summary only — raw NudeNet detections (which include
        # benign classes like faces) are never dumped, and no image content
        # is ever logged, only counts/labels already known to be explicit.
        if flagged:
            logger.info("[Explicit] Explicit content detected")
        else:
            logger.info("[Explicit] No explicit content detected")
        logger.info(f"[Explicit] status={'detected' if flagged else 'not_detected'} flagged={flagged} confidence={confidence}")

        return {
            "status": "detected" if flagged else "not_detected",
            "flagged": flagged,
            "confidence": confidence,
        }
    except Exception as exc:  # noqa: BLE001 - a bad frame must degrade, not crash /detect
        logger.error(f"[Explicit] status=error ({exc.__class__.__name__})")
        return {"status": "error", "flagged": False, "confidence": None}
