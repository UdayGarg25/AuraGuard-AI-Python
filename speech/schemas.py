"""Pydantic models for the speech pipeline. Keeping these separate from the
pipeline logic means the shape of a "speech detection result" is defined in
exactly one place, whether it's produced by a unit test, the CLI, or the
FastAPI router.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

Severity = Literal["none", "low", "medium", "high", "critical"]

# Mirrors the Node backend's VIOLATION_TYPES enum (src/constants/domain.js)
# so results returned here map directly onto it without a second lossy
# translation step downstream.
ViolationType = Literal[
    "abusive_language",
    "harassment",
    "violence",
    "hate_speech",
    "self_harm",
    "other",
]


class KeywordMatchResult(BaseModel):
    """Output of the local/manual keyword filter — the lightweight
    trigger stage, never the final decision."""

    matched: bool
    category: Optional[str] = None
    matched_terms: list[str] = Field(default_factory=list)


class GeminiAnalysis(BaseModel):
    """Output of the Gemini contextual-analysis stage."""

    flagged: bool
    type: Optional[ViolationType] = None
    severity: Severity = "none"
    confidence: float = 0.0
    reason: str = ""


class SpeechDetectionResult(BaseModel):
    """The speech pipeline's final, structured output. This is what gets
    handed to the future Violation Engine — the speech service itself never
    takes moderation action.
    """

    text: str
    keyword_matched: bool
    matched_category: Optional[str] = None
    llm_analyzed: bool
    flagged: bool
    type: Optional[ViolationType] = None
    severity: Severity = "none"
    confidence: float = 0.0
    reason: Optional[str] = None


class AnalyzeAudioRequest(BaseModel):
    """Matches the Node backend's AI Service contract
    (ai-service-contract.md POST /api/v1/analyze/audio): base64 raw audio,
    plus caller-supplied context that gets echoed back for correlation."""

    meetingId: str
    userId: str
    audio: str  # base64-encoded audio chunk


class AnalyzeTextRequest(BaseModel):
    """Matches POST /api/v1/analyze/text — also handy for testing the
    filter/Gemini stages without needing an audio file at all."""

    meetingId: str = "test-meeting"
    userId: str = "test-user"
    text: str
