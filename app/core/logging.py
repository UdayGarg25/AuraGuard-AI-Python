"""
Three prefixed loggers, matching the debugging convention already used
across the rest of AuraGuard ([ROOM], [LIVEKIT], [VIDEO], [AUTH BACKEND] in
the Node backend and frontend — see AURAGUARD_DEVELOPER_INTERVIEW_DOCUMENTATION.md).

  [VISION]    — service lifecycle (startup, model loading, config)
  [YOLO]      — model loading/inference specifics
  [DETECTION] — per-request detection results

Deliberately terse and INFO-level only — this logs one line per request,
not per frame internals, so it stays readable once real frame-sampling
traffic starts arriving.
"""
import logging
import sys


def _configure_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


vision_logger = _configure_logger("[VISION]")
yolo_logger = _configure_logger("[YOLO]")
detection_logger = _configure_logger("[DETECTION]")
