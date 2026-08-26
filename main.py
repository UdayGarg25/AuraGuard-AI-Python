"""AuraGuard AI — Python service entrypoint.

Only mounts the speech router (Phase 4). The video/vision pipeline is owned
by a different developer and lives in its own module — when it's ready, it
gets its own router mounted here the same way, without touching this file's
existing lines. Nothing in this file or in speech/ imports or depends on
any vision/video code.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from speech.router import router as speech_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

app = FastAPI(title="AuraGuard AI - Python Service", version="0.1.0")

# Required for the frontend (a different origin — Vite on :5173) to call
# /api/v1/analyze/audio directly from the browser. Not an auth mechanism —
# see config.py#cors_origin_list.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(speech_router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok", "service": "auraguard-ai-python"}
