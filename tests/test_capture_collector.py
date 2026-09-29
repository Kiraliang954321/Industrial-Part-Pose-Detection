import json

import cv2
import numpy as np
import pytest

from yolo_conveyor.capture import CaptureCollector, CaptureSession, DuplicateGuard, QualityPolicy
from yolo_conveyor.capture import collector as module


def frame():
    return np.tile(np.arange(9, dtype=np.uint8), (8, 1))


def setup(tmp_path, *, reject=False):
    session = CaptureSession.create(tmp_path, source="0", backend="auto", config={})
    collector = CaptureCollector(session, quality=QualityPolicy(min_blur_score=100, reject=reject),
                                 duplicates=DuplicateGuard(max_distance=0))
    return collector, session


def rows(session):
    return [json.loads(line) for line in (session.path / "manifest.jsonl").read_text().splitlines()]


def test_warn_only_accepts_jpeg_and_manifest_records_metrics_hash(tmp_path):
    collector, session = setup(tmp_path)
    outcome = collector.attempt(frame())
    assert outcome.accepted and outcome.warnings == ("blur",)
    assert outcome.saved_path == "front/000001.jpg"
    assert (session.path / outcome.saved_path).read_bytes()[:2] == b"\xff\xd8"
    assert rows(session)[0]["saved_path"] == outcome.saved_path
    assert rows(session)[0]["hash"] == outcome.hash_value
    assert rows(session)[0]["quality_metrics"]["brightness"] >= 0
    assert collector.duplicates.history("front")


def test_reject_policy_records_warning_but_does_not_write(tmp_path):
    collector, session = setup(tmp_path, reject=True)
    outcome = collector.attempt(frame())
    assert not outcome.accepted and outcome.rejection_reasons == ("quality",)
    assert rows(session)[0]["saved_path"] is None
    assert rows(session)[0]["warnings"] == ["blur"]
    assert collector.duplicates.history("front") == ()
    assert list((session.path / "front").iterdir()) == []


def test_duplicates_per_side_and_monotonic_names(tmp_path):
    collector, session = setup(tmp_path)
    first = collector.attempt(frame())
    rejected = collector.attempt(frame(), "auto")
    assert not rejected.accepted and rejected.rejection_reasons == ("duplicate",)
    assert rejected.nearest_duplicate_distance == 0
    assert rows(session)[1]["mode"] == "auto" and rows(session)[1]["hash"] == first.hash_value
    collector.label = "back"
    other = collector.attempt(frame())
    assert other.accepted and other.saved_path == "back/000001.jpg"
    assert len(collector.duplicates.history("front")) == 1
    assert len(collector.duplicates.history("back")) == 1
    collector.label = "front"
    changed = np.tile(np.arange(9, dtype=np.uint8)[::-1], (8, 1))
    second = collector.attempt(changed)
    assert second.accepted and second.saved_path == "front/000002.jpg"


def test_write_failure_records_rejection_without_poisoning_history(tmp_path, monkeypatch):
    collector, session = setup(tmp_path)
    monkeypatch.setattr(module.cv2, "imwrite", lambda *_: False)
    failure = collector.attempt(frame())
    assert not failure.accepted and failure.rejection_reasons == ("image_write_failed",)
    assert rows(session)[0]["saved_path"] is None
    assert rows(session)[0]["hash"] == failure.hash_value
    assert collector.duplicates.history("front") == ()
    assert collector.counts["front"] == 0
    assert list((session.path / "front").iterdir()) == []
    monkeypatch.setattr(module.cv2, "imwrite", lambda path, _: open(path, "wb").write(b"jpeg") > 0)
    assert collector.attempt(frame()).saved_path == "front/000001.jpg"


def test_run_controls_auto_and_cleanup(tmp_path, monkeypatch):
    collector, session = setup(tmp_path)

    class Camera:
        def __init__(self):
            self.released = False

        def read(self):
            return True, frame()

        def release(self):
            self.released = True

    camera = Camera()
    destroyed = []
    keys = iter([ord("b"), 32, ord("a"), -1, ord("q")])
    times = iter([0, 0, 2, 2])
    monkeypatch.setattr(module, "open_source", lambda *a, **kw: camera)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(module.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(module.cv2, "waitKey", lambda _: next(keys))
    monkeypatch.setattr(module.cv2, "destroyAllWindows", lambda: destroyed.append(True))
    module.run_collector(collector, 0, auto_interval=1)
    assert camera.released and destroyed
    assert [(r["label"], r["mode"]) for r in rows(session)] == [("back", "manual"), ("back", "auto")]


def test_repeated_read_failure_releases_and_destroys(tmp_path, monkeypatch):
    collector, _ = setup(tmp_path)

    class Camera:
        released = False

        def read(self):
            return False, None

        def release(self):
            self.released = True

    camera = Camera()
    destroyed = []
    monkeypatch.setattr(module, "open_source", lambda *a, **kw: camera)
    monkeypatch.setattr(module.cv2, "destroyAllWindows", lambda: destroyed.append(True))
    with pytest.raises(RuntimeError, match="repeated frame-read failure"):
        module.run_collector(collector, 0, max_read_failures=2)
    assert camera.released and destroyed
