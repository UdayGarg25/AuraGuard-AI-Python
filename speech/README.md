# AuraGuard AI — Speech Pipeline (Phase 4)

Fully isolated from the video/vision pipeline. Nothing in this folder imports, calls, or
depends on any camera/LiveKit-video/YOLO/vision-service code, and nothing outside `speech/`
was modified to build this.

## Architecture

```
Audio chunk (bytes)
      │
      ▼
┌─────────────┐
│ STT         │  speech/stt_service.py — faster-whisper, runs locally
│ (audio→text)│
└─────┬───────┘
      ▼
┌─────────────────────┐
│ Keyword/phrase check │  speech/keyword_filter.py + speech/data/prohibited_words.json
│ (local, no API call) │
└─────┬────────────────┘
      │
   suspicious?
   ┌──┴──┐
   NO     YES
   │       │
 result   ┌────────────────────┐
          │ Gemini 2.5 Flash   │  speech/gemini_service.py
          │ Lite (context)     │
          └─────────┬──────────┘
                     ▼
              SpeechDetectionResult
                     │
                     ▼
         (future) Violation Engine
```

Orchestrated by `speech/pipeline.py` — the only module that wires the stages together, so the
"never call Gemini on non-suspicious text" rule lives in exactly one place.

## STT: faster-whisper

- **What**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a CTranslate2
  reimplementation of OpenAI Whisper.
- **Why**: runs entirely locally — no API key, no network call, no per-request cost, fits "the
  current local development setup" out of the box. Meaningfully faster than stock
  `openai-whisper` on CPU, which matters for near-real-time chunk transcription. Keeping STT
  free/local also means Gemini stays reserved for the one job it's actually needed for
  (contextual judgment), not burned on every chunk's transcription too.
- **How audio reaches it**: `speech/stt_service.py#transcribe_audio_bytes(audio_bytes)` writes
  the bytes to a temp file and calls `WhisperModel.transcribe()`. The model is loaded once
  (lazy singleton) and reused across requests.
- **Expected input format**: any container ffmpeg/faster-whisper can decode. **WAV
  (16-bit PCM) is the documented/expected format** from the audio producer (the frontend or
  Node relay slicing the live LiveKit audio track) — see "Audio chunking" below.
- **Returns**: a plain string — the transcript. Empty string for silence, which short-circuits
  the rest of the pipeline (no keyword check, no Gemini call). Also returns empty string (never
  raises) if the chunk can't be decoded at all — e.g. truncated/corrupted audio from a flaky
  network — logged as a warning rather than a request-crashing exception.
- **Config**: `WHISPER_MODEL_SIZE` (default `base`), `WHISPER_DEVICE` (default `cpu`),
  `WHISPER_COMPUTE_TYPE` (default `int8`) — see `.env.example`.

## Audio chunking

`AUDIO_CHUNK_INTERVAL_MS` (default **5000ms**, `.env` / `config.py`) is the window size the
**audio producer** (whatever captures the existing LiveKit audio track — frontend or Node
relay) should use when slicing live audio before sending a chunk to `POST /api/v1/analyze/audio`.

5 seconds was chosen as the default because it's long enough to usually capture a full
sentence (short-word chunking fragments STT accuracy badly) while still short enough to keep
detection latency reasonable for a live safety feature. Tune it based on real STT accuracy
once wired to live audio — it's a single env var, not hardcoded.

This service does not do the chunking itself (it has no access to the live LiveKit track) —
it just documents and expects this contract from whatever calls it. Whoever wires up the
producer side is responsible for:

- one in-flight transcription request per chunk at a time (don't fire a new chunk before the
  previous one's request resolves),
- clearing the chunking timer/interval when the user leaves the meeting,
- never opening a second/duplicate audio capture.

## Manual keyword/phrase filter (Step 4)

`speech/keyword_filter.py`, backed by **two complementary sources**:

1. **`speech/data/prohibited_words.json`** — hand-curated `threatening_phrases`,
   `violent_words`, `harassment_terms`, and mild non-swear `abusive_words`. These are
   contextual *phrases* ("I will kill you", "I know where you live"), not swear words, so a
   generic profanity dictionary would never contain them. The JSON file is the only place
   this list lives — update it freely, no code changes needed. Phrases match as substrings;
   single words match on word boundaries (so "class" never matches "ass").
2. **[`better_profanity`](https://github.com/snguyenthanh/better_profanity)'s built-in
   dictionary** (thousands of words, `profanity.load_censor_words()`) — covers broad
   profanity/abusive language comprehensively instead of hand-typing (and inevitably
   under-covering) a curse-word list. A match here is reported under category `"profanity"`;
   the specific matched word(s) are recovered by diffing the text against
   `profanity.censor(text)`.

Custom phrase matches take priority over a plain profanity match when both are present
(`CATEGORY_ORDER` in `keyword_filter.py`) — a direct threat should never be filed under the
same bucket as casual swearing.

**This is a trigger, not a verdict.** A match from either source only means "send this to
Gemini for context" — see `speech/pipeline.py#process_text`. Non-matching text never reaches
Gemini at all.

## Gemini 2.5 Flash Lite (Steps 5–6)

`speech/gemini_service.py`. Called **only** when the keyword filter matched. The prompt
(`_PROMPT_TEMPLATE`) explicitly tells Gemini the keyword filter is a blunt trigger and asks it
to judge actual intent/context, with the "violent movie scene" vs. "I will kill you" contrast
as few-shot examples. Response is constrained to JSON
(`response_mime_type: application/json`) and defensively parsed (strips stray markdown fences
if Gemini adds them anyway).

Gemini's `type` vocabulary (`threat`, `violent_intent`, `abuse`, `toxic`, `harassment`,
`hate_speech`, `self_harm`) is mapped onto the Node backend's existing `VIOLATION_TYPES` enum
(`abusive_language`, `harassment`, `violence`, `hate_speech`, `self_harm`, `other`) so the
result plugs directly into the already-established
`AuraGuard-AI-Backend/src/docs/ai-service-contract.md` contract — see `_TYPE_MAP` in
`gemini_service.py`.

**Failure handling**: if `GEMINI_API_KEY` is unset, or the Gemini call throws/returns
unparseable output, the pipeline does not crash and does not silently drop the text — it
returns a low-confidence `flagged=true` result (the keyword filter already said this text was
suspicious; a broken LLM call shouldn't make that signal disappear). See
`_fallback_result()`.

### Environment setup

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=<PASTE_API_KEY_HERE>
GEMINI_MODEL=gemini-2.5-flash-lite
```

Never hardcode the real key — `config.py` reads it from the environment via
`pydantic-settings`, and `.env` is gitignored.

## Speech Detection Result (Step 7)

```json
{
  "text": "I will kill you if you do that again",
  "keyword_matched": true,
  "matched_category": "threatening_phrases",
  "llm_analyzed": true,
  "flagged": true,
  "type": "violence",
  "severity": "high",
  "confidence": 0.92,
  "reason": "The speaker appears to make a direct threat."
}
```

The speech service stops here. It never mutes, removes, bans, or ends a meeting — those
decisions belong to the future Violation Engine, which consumes this result (see the
"Speech Result → Violation Engine" arrow in the architecture diagram above).

## Testing locally

**Unit tests** (offline, Gemini mocked — no API key needed):

```bash
uv run pytest tests/ -v
```

Covers all 4 required scenarios: normal sentence (Gemini never called), abusive word
(Gemini called), threatening sentence (flagged high severity), suspicious word in harmless
context (Gemini overrides the keyword trigger to `flagged: false`).

**CLI harness** (exercises the real pipeline, including a live Gemini call if
`GEMINI_API_KEY` is set):

```bash
uv run python scripts/test_speech_cli.py --text "I will kill you"
uv run python scripts/test_speech_cli.py --text "That movie had a violent scene"
uv run python scripts/test_speech_cli.py --audio path/to/chunk.wav
```

Prints every stage: transcribed text → keyword match + category → whether Gemini was called →
Gemini's raw result → final structured result.

**HTTP** (once the server's running — `uv run uvicorn main:app --reload --port 8001`; the vision
service already uses port 8000, so both can run at the same time):

```bash
curl -X POST http://localhost:8001/api/v1/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"meetingId":"m1","userId":"u1","text":"I will kill you"}'
```

## What was NOT touched

No file outside `speech/` (plus `main.py`, `config.py`, and the repo-root project files, which
didn't exist before this phase) was modified. No camera/`getUserMedia()` video call, no YOLO
code, no vision service, no frame sampling, no existing LiveKit `Room` implementation — none
of it exists in this repo yet, and nothing here assumes or depends on it. This service only
consumes audio bytes handed to it over HTTP; it has no opinion about how those bytes were
captured.
