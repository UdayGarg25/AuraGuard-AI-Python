"""
Lightweight response models used for API documentation and error shaping.

The main /detect success response is intentionally returned as a plain,
already-camelCase dict from the service layer rather than forced through a
strict Pydantic schema with alias gymnastics — this is a first-phase
prototype endpoint (see PHASE_3_NOTES.md) and the shape is still expected to
evolve once this connects to LiveKit frame sampling. These models cover the
parts worth typing now: errors and health, which are stable.
"""
from typing import Optional

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str  # "ok" | "loading"
    modelLoaded: bool
    model: str
