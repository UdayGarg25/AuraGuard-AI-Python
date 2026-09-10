"""
Three prefixed loggers, matching the debugging convention already used
across the rest of AuraGuard ([ROOM], [LIVEKIT], [VIDEO], [AUTH BACKEND] in
the Node backend and frontend — see AURAGUARD_DEVELOPER_INTERVIEW_DOCUMENTATION.md).

  [VISION]    — service lifecycle (startup, model loading, config) and the
                per-frame explicit-content / camera-visibility results
  [YOLO]      — model loading/inference specifics
  [DETECTION] — per-request pipeline narrative (frame received -> each
                detector's result -> completed) and YOLO's own result line

One line per detector per frame — not per internal function call — so this
stays readable once real 3-second frame-sampling traffic starts arriving.

Level is read from Settings.log_level (LOG_LEVEL in .env, already declared
in app/core/config.py but previously unused) rather than a separate debug
flag — set LOG_LEVEL=warning to silence all of this per-frame chatter
without touching any code.
"""
import logging
import sys

from app.core.config import get_settings

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def _configure_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(_LEVELS.get(get_settings().log_level.lower(), logging.INFO))
    logger.propagate = False
    return logger


vision_logger = _configure_logger("[VISION]")
yolo_logger = _configure_logger("[YOLO]")
detection_logger = _configure_logger("[DETECTION]")
