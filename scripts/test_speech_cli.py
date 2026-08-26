"""Local test harness for the speech pipeline — run it directly, no server
needed.

Usage:
    uv run python scripts/test_speech_cli.py --text "I will kill you"
    uv run python scripts/test_speech_cli.py --audio path/to/chunk.wav

Prints every stage: transcribed text, keyword match + category, whether
Gemini was called, Gemini's result, and the final structured result — per
the Phase 4 testing requirements.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speech import keyword_filter, stt_service  # noqa: E402
from speech.pipeline import process_text  # noqa: E402


def run(text: str | None, audio_path: str | None) -> None:
    if audio_path:
        print(f"[1/4] Transcribing audio file: {audio_path}")
        audio_bytes = Path(audio_path).read_bytes()
        text = stt_service.transcribe_audio_bytes(audio_bytes, suffix=Path(audio_path).suffix)
        print(f"      Transcribed text: {text!r}\n")
    else:
        print(f"[1/4] Using provided text: {text!r}\n")

    print("[2/4] Running local keyword/phrase check...")
    keyword_result = keyword_filter.check_text(text)
    print(f"      matched={keyword_result.matched} category={keyword_result.category} terms={keyword_result.matched_terms}\n")

    if not keyword_result.matched:
        print("[3/4] No suspicious keyword — Gemini will NOT be called.")
        print("[4/4] Final result:")
    else:
        print("[3/4] Suspicious content detected — calling Gemini for context analysis...")
        print("[4/4] Final result (after Gemini):")

    result = process_text(text)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the AuraGuard speech pipeline locally")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Raw text to run through the pipeline")
    group.add_argument("--audio", help="Path to a local audio file (wav/mp3/etc.)")
    args = parser.parse_args()

    run(text=args.text, audio_path=args.audio)
