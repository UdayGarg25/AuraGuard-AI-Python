"""Gemini 2.5 Flash Lite — Step 5/6 of the pipeline: contextual analysis.

Only ever called for text that already tripped the local keyword filter
(speech/keyword_filter.py). Never called for every chunk/sentence — that's
the whole point of the two-stage design (see speech/README.md).
"""

import json
import logging
import re

import google.generativeai as genai

from config import get_settings
from speech.schemas import GeminiAnalysis, ViolationType

logger = logging.getLogger("speech.gemini")

# Gemini's own vocabulary for "type" is intentionally more specific than the
# Node backend's VIOLATION_TYPES enum — this maps one to the other so the
# result plugs straight into the existing contract without a second lossy
# translation downstream. Unrecognized types fall back to "other".
_TYPE_MAP: dict[str, ViolationType] = {
    "threat": "violence",
    "violent_intent": "violence",
    "violence": "violence",
    "abuse": "abusive_language",
    "toxic": "abusive_language",
    "profanity": "abusive_language",
    "harassment": "harassment",
    "hate_speech": "hate_speech",
    "self_harm": "self_harm",
}

_PROMPT_TEMPLATE = """You are a content-safety classifier for a video-conferencing platform.

You will be given a short transcript of live speech that was already flagged by a keyword
filter as POTENTIALLY containing abuse, threats, toxic speech, harassment, or violent intent.
Your job is to judge the actual INTENT AND CONTEXT — the keyword filter is a blunt trigger and
is frequently wrong (e.g. "that movie had a violent scene" contains "violent" but is harmless).

Analyze the transcript below and respond with ONLY a single JSON object, no other text, no
markdown code fences, matching exactly this schema:

{{
  "flagged": <true|false>,
  "type": <one of "threat", "violent_intent", "abuse", "toxic", "harassment", "hate_speech",
           "self_harm", or null if not flagged>,
  "severity": <one of "none", "low", "medium", "high", "critical">,
  "confidence": <float 0.0 to 1.0>,
  "reason": <one short sentence explaining your judgment>
}}

Examples:
- Transcript: "I will kill you" (keyword matched: threatening_phrases)
  -> {{"flagged": true, "type": "threat", "severity": "high", "confidence": 0.92, "reason": "The speaker appears to make a direct threat."}}
- Transcript: "That movie had a really violent scene last night"
  -> {{"flagged": false, "type": null, "severity": "none", "confidence": 0.9, "reason": "The phrase is being used in a non-threatening, descriptive context."}}

Transcript to analyze:
\"\"\"{text}\"\"\"

Matched keyword category (for context only, do not treat as the verdict): {category}
"""


def _extract_json(raw: str) -> dict:
    """Gemini sometimes wraps JSON in ```json fences despite instructions
    not to — strip those defensively before parsing."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def _fallback_result(reason: str) -> GeminiAnalysis:
    # Fails closed-but-cautious: if Gemini is unreachable or returns
    # something unparseable, we can't confirm safety, so we surface a
    # low-confidence flag for human review rather than silently dropping it
    # (the keyword filter already told us this text was suspicious).
    return GeminiAnalysis(flagged=True, type="other", severity="low", confidence=0.0, reason=reason)


def analyze_with_gemini(text: str, matched_category: str | None) -> GeminiAnalysis:
    settings = get_settings()

    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY not set — skipping contextual analysis")
        return _fallback_result("Gemini API key not configured; flagged for manual review.")

    try:
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)

        prompt = _PROMPT_TEMPLATE.format(text=text, category=matched_category or "unknown")
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
        )

        parsed = _extract_json(response.text)
        raw_type = parsed.get("type")
        mapped_type: ViolationType | None = _TYPE_MAP.get(raw_type, "other") if raw_type else None

        return GeminiAnalysis(
            flagged=bool(parsed.get("flagged", False)),
            type=mapped_type,
            severity=parsed.get("severity", "none"),
            confidence=float(parsed.get("confidence", 0.0)),
            reason=parsed.get("reason", ""),
        )
    except Exception as error:  # noqa: BLE001 — a bad/unreachable LLM call must never crash the pipeline
        logger.error("Gemini analysis failed: %s", error)
        return _fallback_result(f"Gemini call failed ({error.__class__.__name__}); flagged for manual review.")
