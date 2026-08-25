"""Orchestrates the full speech pipeline:

    audio bytes -> STT -> text -> keyword filter -> (suspicious?) -> Gemini -> result

This is the ONLY module that should be imported by the router/CLI/tests to
run the pipeline end-to-end — it's the single place the "don't call Gemini
unless suspicious" rule is enforced, so that invariant can't be
accidentally bypassed by a caller wiring the stages together differently.
"""

import logging

from speech import gemini_service, keyword_filter, stt_service
from speech.schemas import SpeechDetectionResult

logger = logging.getLogger("speech.pipeline")


def process_text(text: str) -> SpeechDetectionResult:
    """Runs the filter -> (conditional) Gemini stages on already-transcribed
    text. Exposed separately from process_audio() so text can be tested
    directly (see scripts/test_speech_cli.py and tests/), and so a future
    chat/caption text source could reuse this without going through STT."""

    keyword_result = keyword_filter.check_text(text)

    if not keyword_result.matched:
        # The whole point of the two-stage design: normal speech never
        # reaches Gemini.
        return SpeechDetectionResult(
            text=text,
            keyword_matched=False,
            matched_category=None,
            llm_analyzed=False,
            flagged=False,
            type=None,
            severity="none",
            confidence=0.0,
            reason=None,
        )

    logger.info("Keyword filter matched category=%s — escalating to Gemini", keyword_result.category)
    gemini_result = gemini_service.analyze_with_gemini(text, keyword_result.category)

    return SpeechDetectionResult(
        text=text,
        keyword_matched=True,
        matched_category=keyword_result.category,
        llm_analyzed=True,
        flagged=gemini_result.flagged,
        type=gemini_result.type,
        severity=gemini_result.severity,
        confidence=gemini_result.confidence,
        reason=gemini_result.reason,
    )


def process_audio(audio_bytes: bytes) -> SpeechDetectionResult:
    """Full pipeline entry point for a raw audio chunk."""
    text = stt_service.transcribe_audio_bytes(audio_bytes)

    if not text:
        return SpeechDetectionResult(
            text="",
            keyword_matched=False,
            matched_category=None,
            llm_analyzed=False,
            flagged=False,
            type=None,
            severity="none",
            confidence=0.0,
            reason=None,
        )

    return process_text(text)
