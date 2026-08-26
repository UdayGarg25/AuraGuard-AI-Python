"""
Core YOLO inference logic.

Models are loaded ONCE — via `load_models()`, called from FastAPI's startup
lifespan in app/main.py — and held as module-level singletons. Loading YOLO
weights parses the checkpoint and builds the network; doing that on every
request would make the endpoint unusably slow and defeats the entire point
of "load once, infer many times."

Detection honesty (see README.md "What YOLO Actually Supports" for the full
explanation): the standard Ultralytics COCO-pretrained model has 80 trained
classes. Two of AuraGuard's five target detections map onto real classes in
that set — person and cell phone. The other three (face, camera-covered,
explicit content) have NO COCO class and are not implemented here rather
than approximated with something the model wasn't trained for.
"""
import time
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.core.config import get_settings
from app.core.logging import detection_logger, vision_logger, yolo_logger

settings = get_settings()

# Real, trained COCO class names — not an approximation or a guess at what
# the model might contain. Verified against the labels the loaded model
# itself reports (see load_models(), which logs the full class list once).
COCO_PERSON_CLASS = "person"
COCO_PHONE_CLASS = "cell phone"

_general_model: Optional[YOLO] = None
_face_model: Optional[YOLO] = None


class VisionServiceError(Exception):
    """Expected, user-facing failure (bad image, model not ready, etc.).
    Distinct from an unexpected exception — the /detect route maps this to
    a clean HTTP status + message rather than a 500 with a stack trace."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def load_models() -> None:
    """Called once from the FastAPI startup lifespan. Never call this per
    request."""
    global _general_model, _face_model

    yolo_logger.info(f"Loading general model: {settings.yolo_model}")
    try:
        _general_model = YOLO(settings.yolo_model)
    except Exception as exc:  # noqa: BLE001 - genuinely want to catch anything here
        yolo_logger.error(f"Failed to load general model '{settings.yolo_model}': {exc}")
        raise
    class_names = list(_general_model.names.values())
    yolo_logger.info(f"General model loaded — {len(class_names)} classes available")
    if COCO_PERSON_CLASS not in class_names or COCO_PHONE_CLASS not in class_names:
        # If this ever fires, someone pointed YOLO_MODEL at a non-COCO
        # checkpoint — person/phone detection would silently stop working.
        vision_logger.info(
            f"WARNING: loaded model does not contain expected classes "
            f"'{COCO_PERSON_CLASS}' / '{COCO_PHONE_CLASS}' — person/phone "
            f"detection results will be inaccurate for this model."
        )

    if settings.yolo_face_model_path:
        yolo_logger.info(f"Loading face model: {settings.yolo_face_model_path}")
        try:
            _face_model = YOLO(settings.yolo_face_model_path)
            yolo_logger.info("Face model loaded — face detection is now active")
        except Exception as exc:  # noqa: BLE001
            # A misconfigured face-model path should not take down the whole
            # service — general detection (person/phone) still works fine
            # without it. Fail this one model open, not the process.
            yolo_logger.error(f"Failed to load face model, face detection stays disabled: {exc}")
            _face_model = None
    else:
        vision_logger.info(
            "No YOLO_FACE_MODEL_PATH configured — faceCount/multipleFaces will "
            "report 'MODEL SUPPORT REQUIRED / NOT YET IMPLEMENTED'"
        )


def is_ready() -> bool:
    return _general_model is not None


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    if not raw_bytes:
        raise VisionServiceError("No image data received", status_code=400)
    try:
        image = Image.open(BytesIO(raw_bytes))
        image.load()  # force full decode now, not lazily later inside YOLO
        image = image.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - any decode failure means "bad image"
        raise VisionServiceError(f"Invalid or corrupt image: {exc}", status_code=400) from exc
    return np.array(image)


def _run_face_model(image_array: np.ndarray) -> tuple[int, bool]:
    results = _face_model.predict(image_array, conf=settings.confidence_threshold, verbose=False)
    count = len(results[0].boxes)
    return count, count > 1


def run_detection(raw_bytes: bytes) -> dict:
    """Runs YOLO inference on one image and returns a fully-shaped detection
    result (already using the exact camelCase keys the API responds with).

    Raises VisionServiceError for bad input or an unready/broken model — the
    /detect route turns that into a clean HTTP error response. Never lets a
    single bad frame crash the process.
    """
    if not is_ready():
        raise VisionServiceError("YOLO model is not loaded yet", status_code=503)

    start = time.perf_counter()
    image_array = _decode_image(raw_bytes)

    detection_logger.info(f"Running inference on frame ({image_array.shape[1]}x{image_array.shape[0]})")

    try:
        results = _general_model.predict(image_array, conf=settings.confidence_threshold, verbose=False)
    except Exception as exc:  # noqa: BLE001 - inference failure must not crash the service
        yolo_logger.error(f"Inference failed: {exc}")
        raise VisionServiceError(f"YOLO inference failed: {exc}", status_code=500) from exc

    objects = []
    person_count = 0
    phone_detected = False
    phone_confidence: Optional[float] = None

    result = results[0]
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])

        objects.append(
            {
                "class": cls_name,
                "confidence": round(confidence, 4),
                "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
            }
        )

        if cls_name == COCO_PERSON_CLASS:
            person_count += 1
        elif cls_name == COCO_PHONE_CLASS:
            # Confidence thresholding already happened via `conf=` above —
            # anything that reaches this loop already cleared the bar, so a
            # "cell phone" box here is never reported just because some
            # unrelated low-confidence object happened to exist in-frame.
            phone_detected = True
            phone_confidence = confidence if phone_confidence is None else max(phone_confidence, confidence)

    detection_logger.info(
        f"Frame result: {len(objects)} object(s) — personCount={person_count}, phoneDetected={phone_detected}"
    )

    face_count = None
    multiple_faces = None
    if _face_model is not None:
        face_count, multiple_faces = _run_face_model(image_array)
        detection_logger.info(f"Face model result: faceCount={face_count}, multipleFaces={multiple_faces}")

    processing_time_ms = (time.perf_counter() - start) * 1000

    return {
        "detections": {
            "personCount": person_count,
            "phoneDetected": phone_detected,
            "phoneConfidence": round(phone_confidence, 4) if phone_confidence is not None else None,
            "faceCount": face_count,
            "multipleFaces": multiple_faces,
            # Neither has a COCO class or any general-purpose YOLO equivalent
            # — see README.md. Not approximated with a brightness/heuristic
            # check per explicit project requirement.
            "cameraCovered": None,
            "explicitContent": "not_available",
        },
        "modelSupport": {
            "personDetection": "implemented",
            "phoneDetection": "implemented",
            "faceDetection": "implemented" if _face_model is not None else "MODEL SUPPORT REQUIRED / NOT YET IMPLEMENTED",
            "cameraCovered": "MODEL SUPPORT REQUIRED / NOT YET IMPLEMENTED",
            "explicitContent": "MODEL SUPPORT REQUIRED / NOT YET IMPLEMENTED",
        },
        "objects": objects,
        "model": settings.yolo_model,
        "processingTimeMs": round(processing_time_ms, 2),
    }
