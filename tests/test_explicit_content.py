"""Explicit-content detector tests.

Separates the failure layers:
  - model not loaded           -> test_model_loads_and_blank_frame_is_not_detected
  - model runs, finds nothing  -> test_blank_frame_returns_not_detected
  - threshold filtering        -> test_threshold_and_label_filtering (fake detector)
  - response mapping           -> test_detect_endpoint_maps_explicit_result (fake detector)

No real explicit imagery is used or needed. The fixture is a flat grey frame.
"""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.services import explicit_content_service as svc

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "explicit_blank.jpg")


class FakeDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, _image):
        return self._detections


@pytest.fixture
def blank_frame():
    return np.full((320, 320, 3), 128, dtype=np.uint8)


@pytest.fixture
def real_model():
    svc.load_model()
    if not svc.is_ready():
        pytest.skip("NudeNet model did not load in this environment")
    yield
    svc._detector = None


def test_model_loads_and_blank_frame_is_not_detected(real_model, blank_frame):
    result = svc.analyze(blank_frame)
    assert result == {"status": "not_detected", "flagged": False, "confidence": 0.0}


@pytest.mark.parametrize(
    "detections, expected",
    [
        (
            [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.82, "box": [0, 0, 1, 1]}],
            {"status": "detected", "flagged": True, "confidence": 0.82},
        ),
        (
            [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.45, "box": [0, 0, 1, 1]}],
            {"status": "not_detected", "flagged": False, "confidence": 0.0},
        ),
        (
            [{"class": "FACE_FEMALE", "score": 0.95, "box": [0, 0, 1, 1]}],
            {"status": "not_detected", "flagged": False, "confidence": 0.0},
        ),
        (
            [{"class": "FEMALE_BREAST_COVERED", "score": 0.9, "box": [0, 0, 1, 1]}],
            {"status": "not_detected", "flagged": False, "confidence": 0.0},
        ),
    ],
    ids=["above-threshold", "below-threshold", "benign-face", "covered-label"],
)
def test_threshold_and_label_filtering(monkeypatch, blank_frame, detections, expected):
    monkeypatch.setattr(svc, "_detector", FakeDetector(detections))
    assert svc.analyze(blank_frame) == expected


def test_detect_endpoint_maps_explicit_result(monkeypatch):
    from app.main import app

    # App startup calls load_model(), which would replace the fake with the
    # real NudeNet detector. Stub it so this test checks only the mapping.
    monkeypatch.setattr(svc, "load_model", lambda: None)
    monkeypatch.setattr(
        svc,
        "_detector",
        FakeDetector([{"class": "MALE_GENITALIA_EXPOSED", "score": 0.9, "box": [0, 0, 1, 1]}]),
    )
    with open(FIXTURE, "rb") as fh:
        with TestClient(app) as client:
            resp = client.post("/detect", files={"file": ("explicit_blank.jpg", fh, "image/jpeg")})
    assert resp.status_code == 200
    explicit = resp.json()["detections"]["explicitContent"]
    assert explicit == {"status": "detected", "flagged": True, "confidence": 0.9}
