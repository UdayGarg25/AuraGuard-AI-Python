# Phase 3 — Architecture Decisions & Honest Limitations

This is the reasoning behind the choices in this phase — written so a future
session (or a future you) doesn't have to re-derive them.

## Why this repo, not a nested `ai-services/vision-service/` folder

The Node backend already has a fully-built (if not-yet-called) client
expecting an external repo called `AuraGuard-AI-Service` (FastAPI + Python):
`src/clients/aiService.client.js`, `src/config/aiService.js`, and a written
contract at `src/docs/ai-service-contract.md`. `AuraGuard-AI-Python` — this
repo — is that service. It is not a monorepo host for multiple AI
sub-services; it *is* the one AI service, with vision as its first
capability module. Nesting a duplicate `ai-services/vision-service/`
subfolder inside it would have created a second, disconnected structure for
no reason.

## Why `POST /detect` is separate from `POST /api/v1/analyze/frame`

The Node backend's existing, documented contract
(`ai-service-contract.md`) already specifies `POST /api/v1/analyze/frame`
— but its response shape is `{ flagged, type, severity, confidence, details }`.
That is a **violation-decision** shape: `type` must be one of
`VIOLATION_TYPES` (nudity, violence, hate_speech, ...), `severity` one of
`VIOLATION_SEVERITY` (low/medium/high/critical). Implementing that endpoint
*now* would mean inventing the violation-decision logic this phase was
explicitly told to defer — e.g., deciding what violation `type` and
`severity` a detected phone should map to, with no specification for that
mapping given.

So this phase built `/detect` instead: a new, lower-level endpoint that
returns raw detections with no interpretation attached. `/api/v1/analyze/frame`
remains for the future Violation Engine phase, which would implement it by
calling the exact same `vision_service.run_detection()` this phase built,
then adding a translation layer on top. Nothing here needs to change for
that to happen — it's a new function, not a rewrite.

## Why person detection isn't called "face detection"

Standard YOLO COCO-pretrained models are trained on 80 classes defined by
the COCO dataset. `person` — a whole-body bounding box — is one of them.
`face` is not. Calling `personCount` "face count" would be exactly the kind
of "pretend the model detects something it wasn't trained for" this task
explicitly ruled out. They're reported as clearly separate, honestly-named
things: `personCount` (real, always on) and `faceCount` (real only if a
dedicated face model is configured, otherwise explicitly marked
unsupported).

## Why camera-covered has no heuristic fallback

The obvious naive approach — "if the frame is mostly black/uniform, the
camera is covered" — was explicitly ruled out by the task requirements, and
for good reason: it's not YOLO-based, and it's unreliable in ways that would
generate real false positives (a dark room, someone wearing a dark shirt
that fills the frame, bad lighting, a low-light camera sensor). Rather than
ship something that *looks* like a real feature but silently misfires, this
field is left as `MODEL SUPPORT REQUIRED / NOT YET IMPLEMENTED`. A real
implementation would most plausibly need either:

- A small custom-trained binary classifier (not a YOLO detection model —
  YOLO detects objects/regions, this is a whole-frame classification
  question), or
- A different signal entirely: e.g., detecting that the LiveKit video track
  itself reports very low variance/motion over time, which is a track-level
  signal LiveKit could expose, not something a single frame's object
  detection reveals.

Either approach is a deliberate future decision, not something to guess at
now.

## Why explicit-content detection isn't wired to any model

No COCO class covers this, and reasonable YOLO-based NSFW-detection models
that do exist online are third-party community-trained checkpoints of
uncertain provenance, training-data licensing, and accuracy. Silently
downloading and wiring one up mid-implementation — without you having
chosen and vetted the specific model — isn't a call this phase should make
unilaterally. The service is structured so plugging one in later (the same
way `YOLO_FACE_MODEL_PATH` works) is a small, additive change, not a
rearchitecture. Until then, `/detect` always returns
`explicitContent: "not_available"`, exactly as specified.

## Why a face model wasn't downloaded and wired up automatically

A real YOLO-architecture face-detection checkpoint (e.g., trained on WIDER
FACE) would satisfy "YOLO must be the only detection model" and would be a
genuinely correct, working addition — the integration code for it
(`YOLO_FACE_MODEL_PATH` → `_run_face_model()`) is already written and
tested via the same code path the general model uses. What's missing is
sourcing a specific, trustworthy weights file — that means picking a
specific third-party checkpoint from a specific source, which affects
supply-chain trust (`.pt` files can be more than plain weights) and model
accuracy/licensing. That's a decision better made deliberately by you,
pointing `YOLO_FACE_MODEL_PATH` at a checkpoint you've chosen, than one made
silently by grabbing the first plausible-looking download mid-session.

## Why frame-sampling cadence lives in config but isn't enforced here

`/detect` is a stateless, synchronous, one-frame-in-one-result-out endpoint.
It has no concept of "the last frame I saw" or "how often I should be
called" — that's a property of whatever calls it repeatedly (a future
LiveKit frame sampler), not of the detector itself.
`RECOMMENDED_FRAME_SAMPLE_INTERVAL_MS` exists in `.env`/`config.py` purely
as a documented, discoverable default for that future caller to read — it
is not read or enforced anywhere in this service's own request handling.

## What was actually verified, not just written

Writing code that *should* work and confirming it *does* work are different
things. This phase:

1. Installed the real dependencies (FastAPI, Ultralytics, PyTorch, Pillow)
   into an actual virtual environment.
2. Started the actual service and confirmed the YOLO model loads
   successfully (auto-downloaded from Ultralytics).
3. Sent a real image to `POST /detect` and confirmed a structurally correct
   JSON response comes back, with real inference having run (not a
   hardcoded/mocked response).

See the end of this session's summary for the exact commands run and their
output. If you're reading this later and haven't re-run it yourself, treat
"verified" as meaning "verified once, on one machine, on one date" — not an
ongoing guarantee.
