import cv2
import pytest

from yolo_conveyor.capture.cli import build_parser, parse_source
from yolo_conveyor.capture.collector import open_source, run_collector
from yolo_conveyor.capture import collector as module


@pytest.mark.parametrize(("text", "result"), [
    ("0", 0), ("002", 2), ("42", 42), ("-1", "-1"), ("+1", "+1"),
    ("camera.mp4", "camera.mp4"), ("rtsp://host/live", "rtsp://host/live"),
    ("1.0", "1.0"),
])
def test_source_parsing(text, result):
    assert parse_source(text) == result


def test_cli_defaults_and_policy():
    args = build_parser().parse_args([])
    assert parse_source(args.source) == 0
    assert str(args.output_root).replace("\\", "/") == "data/captures"
    assert args.quality_policy == "warn" and args.auto_interval == 0


@pytest.mark.parametrize(("source", "backend", "uses_dshow"), [
    (0, "dshow", True), (3, "auto", False), ("video.mp4", "dshow", False),
    ("rtsp://host/live", "dshow", False),
])
def test_backend_selection_and_positive_capture_properties(monkeypatch, source, backend, uses_dshow):
    calls = []

    class FakeCamera:
        def isOpened(self):
            return True

        def set(self, key, value):
            calls.append((key, value))

    def video_capture(*args):
        calls.append(args)
        return FakeCamera()

    monkeypatch.setattr(module.cv2, "VideoCapture", video_capture)
    open_source(source, backend, width=640, height=0, fps=-1)
    assert calls[0] == ((source, cv2.CAP_DSHOW) if uses_dshow else (source,))
    assert calls[1:] == [(cv2.CAP_PROP_FRAME_WIDTH, 640)]


def test_open_failure_releases_and_run_destroys_windows(monkeypatch, tmp_path):
    class Closed:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    cap = Closed()
    destroyed = []
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda *args: cap)
    monkeypatch.setattr(module.cv2, "destroyAllWindows", lambda: destroyed.append(True))
    with pytest.raises(RuntimeError, match="cannot open"):
        run_collector(None, "bad.mp4")
    assert cap.released and destroyed
