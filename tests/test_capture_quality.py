import cv2
import numpy as np
import pytest

from yolo_conveyor.capture import QualityPolicy, measure_quality


def test_synthetic_sharp_and_blurred_scores_are_deterministic():
    sharp = np.zeros((64, 64), dtype=np.uint8)
    sharp[16:48, 16:48] = 255
    blurred = cv2.GaussianBlur(sharp, (9, 9), 2)

    sharp_metrics = measure_quality(sharp)
    blurred_metrics = measure_quality(blurred)
    assert sharp_metrics == measure_quality(sharp)
    assert blurred_metrics == measure_quality(blurred)
    assert sharp_metrics.blur_score > blurred_metrics.blur_score > 0
    assert sharp_metrics.brightness == pytest.approx(63.75)
    assert blurred_metrics.brightness == pytest.approx(63.75, abs=0.5)

    policy = QualityPolicy(min_blur_score=(sharp_metrics.blur_score + blurred_metrics.blur_score) / 2)
    assert policy.evaluate(sharp).warnings == ()
    assert policy.evaluate(blurred).warnings == ("blur",)
    assert policy.evaluate(blurred).accepted
    assert not QualityPolicy(min_blur_score=policy.min_blur_score, reject=True).evaluate(blurred).accepted


def test_dark_bright_and_bgr_brightness():
    dark = np.full((12, 12), 10, dtype=np.uint8)
    bright = np.full((12, 12), 240, dtype=np.uint8)
    assert measure_quality(dark).brightness == 10.0
    assert measure_quality(bright).brightness == 240.0
    assert measure_quality(dark).blur_score == measure_quality(bright).blur_score == 0
    bgr = np.full((12, 12, 3), (0, 0, 255), dtype=np.uint8)
    assert measure_quality(bgr).brightness == float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())

    warn = QualityPolicy(min_brightness=20, max_brightness=220)
    reject = QualityPolicy(min_brightness=20, max_brightness=220, reject=True)
    assert warn.evaluate(dark).warnings == ("dark",)
    assert warn.evaluate(bright).warnings == ("bright",)
    assert warn.evaluate(dark).accepted and warn.evaluate(bright).accepted
    assert not reject.evaluate(dark).accepted and not reject.evaluate(bright).accepted
    assert reject.evaluate(np.full((12, 12), 128, dtype=np.uint8)).accepted


@pytest.mark.parametrize("frame", [np.empty((0, 4), dtype=np.uint8),
                                  np.zeros((4, 4, 4), dtype=np.uint8),
                                  np.zeros((4, 4), dtype=np.float32)])
def test_invalid_frames(frame):
    with pytest.raises(ValueError):
        measure_quality(frame)


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        QualityPolicy(min_blur_score=float("nan"))
    with pytest.raises(ValueError):
        QualityPolicy(min_brightness=220, max_brightness=20)
