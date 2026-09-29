import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from yolo_conveyor.annotation import AnnotationSession
from yolo_conveyor.annotation import cli
from yolo_conveyor.annotation.contract import sha256_file
from yolo_conveyor.capture import CaptureSession


def source_session(tmp_path, session_id="sample"):
    capture = CaptureSession.create(tmp_path / "raw", part_id="gear", session_id=session_id,
                                    source="test", backend="test", config={})
    image = capture.path / "front" / "a.jpg"
    assert cv2.imwrite(str(image), np.zeros((100, 200, 3), dtype=np.uint8))
    capture.record_attempt(label="front", mode="manual", accepted=True,
                           saved_path="front/a.jpg", quality_metrics={})
    capture.record_attempt(label="back", mode="manual", accepted=False,
                           saved_path=None, quality_metrics={})
    return capture.path, image


def test_default_workspace_resume_and_explicit_revisit(tmp_path, monkeypatch):
    source, image = source_session(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls = []

    def review(frame, **kw):
        calls.append(kw)
        return "save", [[0.1, 0.1], [0.6, 0.1], [0.6, 0.4], [0.1, 0.4]]

    monkeypatch.setattr(cli, "review_image", review)
    assert cli.run(source) == 0
    store = AnnotationSession(Path("data/annotations/sample"))
    record = store.read_record("front/a.jpg")
    assert (record["part_id"], record["face"], record["class_name"]) == ("gear", "front", "gear_front")
    assert (record["image_width"], record["image_height"], record["source_image_sha256"]) == (200, 100, sha256_file(image))
    assert calls[0]["existing"] is False and calls[0]["total"] == 1
    record_file = store.record_file("front/a.jpg")
    before = record_file.read_bytes()
    assert cli.run(source) == 0
    assert record_file.read_bytes() == before and len(calls) == 1
    cli.run(source, revisit=True)
    assert calls[-1]["existing"] is True and len(calls) == 2


def test_workspace_mismatch_and_source_drift_fail_closed(tmp_path, monkeypatch):
    source, _ = source_session(tmp_path, "one")
    other, _ = source_session(tmp_path, "two")
    store = AnnotationSession.create(tmp_path / "review", source)
    monkeypatch.setattr(cli, "review_image", lambda *a, **k: pytest.fail("GUI should not open"))
    with pytest.raises(ValueError, match="does not match"):
        cli.run(other, workspace=store.path)
    (source / "manifest.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        cli.run(source, workspace=store.path)


@pytest.mark.parametrize("action", ["skip", "quit", "exclude"])
def test_no_fabricated_annotation_for_non_save_actions(tmp_path, monkeypatch, action):
    source, _ = source_session(tmp_path)
    workspace = tmp_path / "review"
    monkeypatch.setattr(cli, "review_image", lambda *a, **k: (action, None))
    assert cli.run(source, workspace=workspace) == 0
    store = AnnotationSession(workspace)
    path = store.record_file("front/a.jpg")
    if action == "exclude":
        assert store.read_record("front/a.jpg")["reason"] == "operator_excluded"
        assert "corners" not in store.read_record("front/a.jpg")
    else:
        assert not path.exists()


def test_missing_corners_cannot_be_saved(tmp_path, monkeypatch):
    source, _ = source_session(tmp_path)
    workspace = tmp_path / "review"
    monkeypatch.setattr(cli, "review_image", lambda *a, **k: ("save", None))
    with pytest.raises(ValueError, match="invalid review action"):
        cli.run(source, workspace=workspace)
    assert not AnnotationSession(workspace).record_file("front/a.jpg").exists()


def test_existing_record_unchanged_on_skip_quit_and_invalid_record_fails(tmp_path, monkeypatch):
    source, _ = source_session(tmp_path)
    store = AnnotationSession.create(tmp_path / "review", source)
    for action in ("skip", "quit"):
        monkeypatch.setattr(cli, "review_image", lambda *a, **k: ("exclude", None))
        cli.run(source, workspace=store.path)
        before = store.record_file("front/a.jpg").read_bytes()
        monkeypatch.setattr(cli, "review_image", lambda *a, **k: (action, None))
        cli.run(source, workspace=store.path, revisit=True)
        assert store.record_file("front/a.jpg").read_bytes() == before
    store.record_file("front/a.jpg").write_text(json.dumps({"oops": True}))
    with pytest.raises(ValueError):
        cli.run(source, workspace=store.path)
