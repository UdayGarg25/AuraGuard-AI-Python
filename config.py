"""Centralized, typed environment configuration — the rest of the codebase
never touches os.environ directly. Mirrors the pattern used in the Node
backend's src/config/env.js for consistency across the AuraGuard repos.
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000

    # Gemini — contextual analysis only, never the primary STT.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"

    # Speech-to-text (faster-whisper, local, no API key required).
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # Reference value for the audio producer's chunking window. The speech
    # service doesn't chunk audio itself (that happens upstream, close to
    # the LiveKit track) — this is exposed so producers/tests can read the
    # agreed-upon interval from one place instead of hardcoding it twice.
    audio_chunk_interval_ms: int = 5000

    # Browser origins allowed to call this service directly (the frontend's
    # Vite dev server calls /api/v1/analyze/audio straight from the browser,
    # a different origin than this API, so CORS is required for that request
    # to succeed at all — not an auth mechanism, just transport-level).
    # Vite auto-bumps to 5174/5175 if 5173 is already taken by another
    # local project, so the default covers that range rather than assuming
    # a single fixed port. Override via CORS_ORIGINS in .env if needed.
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5174,http://127.0.0.1:5174,"
        "http://localhost:5175,http://127.0.0.1:5175"
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
