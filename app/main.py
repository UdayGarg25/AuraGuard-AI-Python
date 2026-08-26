"""
AuraGuard Vision Service — FastAPI application.

Phase 3 scope: YOLO-based object detection only (POST /detect). This
service does NOT make moderation decisions and never removes or flags a
participant on its own — it returns raw, structured detections. The future
Violation Engine (a separate, later phase) is what interprets these results
and decides severity/action; the host always remains the one who decides to
remove a participant. See README.md and PHASE_3_NOTES.md for the full
architecture.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import detect, health
from app.core.config import get_settings
from app.core.logging import vision_logger
from app.services import vision_service

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    vision_logger.info("Starting AuraGuard Vision Service...")
    vision_service.load_models()
    vision_logger.info(f"Ready — listening for /detect requests (general model: {settings.yolo_model})")
    yield
    vision_logger.info("Shutting down AuraGuard Vision Service")


app = FastAPI(
    title="AuraGuard Vision Service",
    description="YOLO-based frame detection for AuraGuard meetings (Phase 3).",
    version="0.1.0",
    lifespan=lifespan,
)

# Permissive CORS for local development/testing. The intended caller in
# production is the Node backend (server-to-server), not a browser directly
# — tighten this once that wiring exists.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(detect.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # A single bad frame or an unexpected bug must never crash the whole
    # service — always respond with clean JSON instead of dying.
    vision_logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"success": False, "error": "Internal vision service error"})
