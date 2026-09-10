"""
Centralized, typed configuration for the Vision Service.

Mirrors the pattern the Node backend already uses (src/config/env.js):
one place reads environment variables, and nothing else touches
`os.environ` directly.
"""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Server -----------------------------------------------------------
    host: str = "0.0.0.0"
    # 8000 matches the Node backend's default AI_SERVICE_BASE_URL
    # (http://localhost:8000, see AuraGuard-AI-Backend/src/config/env.js)
    # so no configuration change is needed on either side to point them
    # at each other later.
    port: int = 8000
    log_level: str = "info"

    # --- General-purpose YOLO model ----------------------------------------
    # The standard Ultralytics COCO-pretrained model. Auto-downloaded from
    # Ultralytics' own servers on first use if not already cached locally.
    # Genuinely supports (real COCO classes, not an approximation):
    #   - person detection  (class "person")
    #   - phone detection   (class "cell phone")
    yolo_model: str = "models/yolov8n.pt"
    confidence_threshold: float = 0.5

    # --- Optional, pluggable face-detection model --------------------------
    # A standard COCO model has NO "face" class — only "person" (a whole-body
    # box). Reporting person-count as face-count would be exactly the kind of
    # "pretend the model supports a class it wasn't trained for" this project
    # explicitly rejected. Leave this unset and /detect honestly reports
    # faceCount as unsupported; point it at a real face-trained YOLO .pt file
    # (e.g. a YOLOv8 model trained on WIDER FACE) to activate genuine face
    # counting — same detection-model code path as the general model, just
    # different trained weights.
    yolo_face_model_path: Optional[str] = None

    # --- Explicit-content detection ----------------------------------------
    # YOLOv8n has no NSFW-related class and was never trained for this task —
    # using it here would be exactly the "pretend the model supports what it
    # wasn't trained for" mistake this project explicitly rejects. This uses
    # a separate, purpose-built local classifier (NudeNet, see
    # app/services/explicit_content_service.py) instead. Runs fully offline —
    # its checkpoint ships inside the `nudenet` pip package itself, no
    # separate download and no third-party API call.
    #
    # A kill-switch, not just a threshold: explicit-content flags are
    # sensitive enough that an operator should be able to disable this
    # detector entirely via one env var without touching code.
    explicit_content_enabled: bool = True
    # Higher than YOLO's general confidence_threshold (0.5) on purpose — a
    # false positive here is more consequential than an extra "cell phone"
    # box, so this defaults more conservative.
    explicit_content_confidence_threshold: float = 0.6

    # --- Camera-covered / visibility detection ------------------------------
    # NOT a YOLO class — there is no "covered camera" object to detect.
    # This is a heuristic image-quality/visibility analysis (see
    # app/services/camera_visibility_service.py) run on the same decoded
    # frame, using multiple signals together so a merely dark room is never
    # confused with a physically blocked lens.
    #
    # Mean grayscale brightness (0-255) below this is "dark" — on its own,
    # NOT sufficient evidence of a covered camera (a dark room is normal).
    camera_visibility_dark_threshold: float = 20.0
    # Pixel-value standard deviation below this means the frame is close to
    # a single flat, uniform color — the signature of something pressed
    # directly against the lens, at ANY brightness level.
    camera_visibility_flat_stddev_threshold: float = 10.0
    # Laplacian variance (a standard blur/edge-detail measure) below this
    # means almost no real detail/edges are visible anywhere in the frame.
    camera_visibility_flat_detail_threshold: float = 15.0

    # --- Request limits -----------------------------------------------------
    max_image_size_mb: float = 10.0

    # --- Documented cadence for a FUTURE caller -----------------------------
    # This service is stateless and per-request — it does not pull frames
    # from LiveKit or run any loop itself. Whatever calls /detect repeatedly
    # (a future Node worker or frame extractor) should read this value to
    # decide how often to call. It exists here only so the recommended
    # cadence is documented and configurable in one place, not enforced.
    recommended_frame_sample_interval_ms: int = 2000


@lru_cache
def get_settings() -> Settings:
    return Settings()
