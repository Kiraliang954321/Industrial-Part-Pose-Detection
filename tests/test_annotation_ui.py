import cv2
import numpy as np
import pytest

from yolo_conveyor.annotation import ui
from yolo_conveyor.annotation.ui import (PointState, canonical_corners, display_size,
                                          original_point)
from yolo_conveyor.annotation.contract import validate_corners


def test_display_mapping_and_true_rotated_rectangle():
    assert display_size(800, 400, 200) == (200, 100)
    assert display_size(100, 80, 200) == (100, 80)
    assert original_point(25, 20, 800, 400, 200, 100) == (100, 80)
    rotated = cv2.boxPoints(((400, 200), (280, 100), 27))
    # Four approximate perimeter clicks; minAreaRect always emits a rectangle.
    state = PointState(800, 400, 200, 100)
    for x, y in rotated:
        state.click(round(float(x) / 4), round(float(y) / 4))
    corners = state.corners()
    assert validate_corners(corners)
    pixels = np.asarray(corners) * [800, 400]
    edges = np.roll(pixels, -1, axis=0) - pixels
    assert abs(float(np.dot(edges[0], edges[1]))) < 0.01
    assert np.allclose(np.mean(pixels, axis=0), (400, 200), atol=4)
    assert np.allclose(np.mean(pixels[:, 0]), 400, atol=4)


def test_invalid_clicks_and_key_state():
    state = PointState(100, 100, 50, 50)
    state.click(50, 20)  # outside image
    assert state.points == []
    for point in ((5, 5), (30, 5), (30, 20)):
        state.click(*point)
    assert state.key(13) is None
    state.click(5, 20)
    assert state.key(32) == "save"
    state.key(ord("u"))
    assert len(state.points) == 3 and state.key(13) is None
    state.key(ord("r"))
    assert state.points == []
    assert state.key(ord("s")) == "skip" and state.points == []
    assert state.key(ord("x")) == "exclude" and state.points == []
    assert state.key(27) == "quit"
    with pytest.raises(ValueError):
        canonical_corners([(1, 1)] * 4, 100, 100)
    with pytest.raises(ValueError):
        canonical_corners([(0, 0), (99, 0), (99, 99), (0, 100)], 100, 100)


def test_mock_window_header_offset_and_save(monkeypatch):
    callbacks = []
    frames = []
    monkeypatch.setattr(ui.cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(ui.cv2, "destroyWindow", lambda *a: None)
    monkeypatch.setattr(ui.cv2, "setMouseCallback", lambda window, cb: callbacks.append(cb))
    monkeypatch.setattr(ui.cv2, "imshow", lambda window, frame: frames.append(frame))
    monkeypatch.setattr(ui.cv2, "getWindowProperty", lambda *a: 1)
    presses = iter([0, 0, 0, 0, 13])

    def key(delay):
        step = len(frames)
        if step <= 4:
            x, y = [(5, 5), (25, 5), (25, 15), (5, 15)][step - 1]
            callbacks[0](ui.cv2.EVENT_LBUTTONDOWN, x, y + 110, 0, None)
        return next(presses)

    monkeypatch.setattr(ui.cv2, "waitKeyEx", key)
    action, corners = ui.review_image(np.zeros((80, 120, 3), dtype=np.uint8),
                                       session_id="s", part_id="gear", face="front",
                                       class_name="gear_front", index=1, total=1,
                                       existing=False, max_window=60)
    assert action == "save" and validate_corners(corners)
    assert frames[0].shape[0] == 150  # 110-pixel header plus 40-pixel scaled image.
    assert np.allclose(np.mean(np.asarray(corners) * [120, 80], axis=0), (30, 20), atol=0.01)
