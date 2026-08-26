"""
GET /health — plain liveness check, no auth.

Deliberately uses this exact unversioned path because the Node backend's
existing AI-service client already expects it here (see
AuraGuard-AI-Backend/src/config/aiService.js `endpoints.health` and
src/docs/ai-service-contract.md) — this one endpoint is already
wire-compatible with the rest of the system with zero extra work.
"""
from fastapi import APIRouter

from app.core.config import get_settings
from app.services import vision_service

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health():
    return {
        "status": "ok" if vision_service.is_ready() else "loading",
        "modelLoaded": vision_service.is_ready(),
        "model": settings.yolo_model,
    }
