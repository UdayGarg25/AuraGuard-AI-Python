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
