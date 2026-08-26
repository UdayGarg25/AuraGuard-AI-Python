"""Speech-to-Text — Step 3 of the pipeline.

STT technology: **faster-whisper** (CTranslate2 reimplementation of OpenAI
Whisper).

Why:
- Runs entirely locally — no API key, no network call, no per-request cost.
  Fits "the current local development setup" directly: `pip install` and go.
- Meaningfully faster than stock openai-whisper on CPU, which matters since
  we're transcribing short chunks close to real time (see
  AUDIO_CHUNK_INTERVAL_MS in config.py / speech/README.md).
- Same underlying Whisper model quality/multilingual support, actively
  maintained, well documented.
- Keeping STT local and free also matches the architecture's intent:
  Gemini is reserved for contextual analysis of already-suspicious text,
  not burned on every chunk's transcription too.

How audio reaches it: this module accepts raw audio bytes (any container
ffmpeg/faster-whisper can decode — WAV is the expected/documented format
from the audio producer, see speech/README.md) and transcribes them
in-memory via a temp file (faster-whisper's transcribe() needs a
file path or file-like object, not raw bytes).

What it returns: a plain string — the transcribed text for that chunk.
Empty string for silence/no speech detected, which the keyword filter and
Gemini stage both treat as "not suspicious" and skip cheaply.
"""

import logging
import tempfile
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from config import get_settings

logger = logging.getLogger("speech.stt")


@lru_cache
def _get_model() -> WhisperModel:
    settings = get_settings()
    logger.info(
        "Loading faster-whisper model=%s device=%s compute_type=%s (first call only)",
        settings.whisper_model_size,
        settings.whisper_device,
        settings.whisper_compute_type,
    )
    return WhisperModel(
        settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )


def transcribe_audio_bytes(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Transcribes one audio chunk. Loads the Whisper model lazily on first
    call and reuses it for every subsequent chunk (loading it per-request
    would dominate latency)."""

    if not audio_bytes:
        return ""

    model = _get_model()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        segments, _info = model.transcribe(str(tmp_path), beam_size=1, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as error:  # noqa: BLE001 — a malformed/truncated/corrupt
        # chunk (real-world possibility with live network audio) must never
        # crash the request. Treated the same as "no speech detected" rather
        # than surfaced as a 500, consistent with how silence is handled.
        logger.warning("Failed to decode/transcribe audio chunk (%s) — treating as no speech", error)
        return ""
    finally:
        tmp_path.unlink(missing_ok=True)
