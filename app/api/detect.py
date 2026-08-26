"""
POST /detect — the Phase 3 raw-detection endpoint.

Accepts a single image as multipart/form-data (the simplest thing to test
manually with curl -F or Postman — see README.md "Testing /detect"). Returns
structured YOLO detections. Makes no moderation decision and never removes
or flags a participant — that is explicitly the Violation Engine's job, in a
later phase, not this endpoint's.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.logging import vision_logger
from app.services import vision_service

router = APIRouter()
settings = get_settings()


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    vision_logger.info(f"Received frame: {file.filename} ({file.content_type})")

    raw_bytes = await file.read()

    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > settings.max_image_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({size_mb:.1f}MB > {settings.max_image_size_mb}MB limit)",
        )

    try:
        result = vision_service.run_detection(raw_bytes)
    except vision_service.VisionServiceError as exc:
        vision_logger.error(f"Detection failed: {exc.message}")
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return {
        "success": True,
        **result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
