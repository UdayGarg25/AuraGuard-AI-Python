"""HTTP surface for the speech pipeline. Matches the endpoint paths already
documented in the Node backend's speech contract
(AuraGuard-AI-Backend/src/docs/ai-service-contract.md):
POST /api/v1/analyze/audio and POST /api/v1/analyze/text.
"""

import base64
import logging

from fastapi import APIRouter, HTTPException

from speech import pipeline
from speech.schemas import AnalyzeAudioRequest, AnalyzeTextRequest, SpeechDetectionResult

logger = logging.getLogger("speech.router")

router = APIRouter(prefix="/analyze", tags=["speech"])


@router.post("/audio", response_model=SpeechDetectionResult)
def analyze_audio(payload: AnalyzeAudioRequest) -> SpeechDetectionResult:
    try:
        audio_bytes = base64.b64decode(payload.audio)
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {error}") from error

    logger.info(
        "Analyzing audio chunk meeting=%s user=%s bytes=%d",
        payload.meetingId,
        payload.userId,
        len(audio_bytes),
    )
    return pipeline.process_audio(audio_bytes)


@router.post("/text", response_model=SpeechDetectionResult)
def analyze_text(payload: AnalyzeTextRequest) -> SpeechDetectionResult:
    """Text-only variant — also useful for moderating chat/caption text
    directly, without going through STT at all."""
    logger.info("Analyzing text meeting=%s user=%s", payload.meetingId, payload.userId)
    return pipeline.process_text(payload.text)
