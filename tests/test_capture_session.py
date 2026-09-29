import json

import pytest

from yolo_conveyor.capture import CaptureSession, QualityMetrics


def _session(tmp_path, session_id="pilot-01"):
    return CaptureSession.create(
        tmp_path, part_id="screw_A-1", session_id=session_id, source="phone/iVCam", backend="ivcam",
        config={"device": "téléphone", "fps": 30},
    )


def _attempt(session, **changes):
    values = dict(label="front", mode="manual", accepted=True,
                  saved_path="front/frame-001.jpg",
                  quality_metrics={"blur_score": 12.0, "brightness": 100.0})
    values.update(changes)
    session.record_attempt(**values)


def test_initial_layout_and_raw_capture_metadata(tmp_path):
    session = _session(tmp_path)
    assert session.path == tmp_path / "pilot-01"
    assert {p.name for p in session.path.iterdir()} == {
        "front", "back", "session.json", "manifest.jsonl",
    }
    assert (session.path / "front").is_dir()
    assert (session.path / "back").is_dir()
    assert (session.path / "manifest.jsonl").read_bytes() == b""
    metadata = json.loads((session.path / "session.json").read_text(encoding="utf-8"))
    assert metadata == {
        "schema_version": 2, "session_id": "pilot-01", "part_id": "screw_A-1",
        "state": "raw_capture", "labels": ["front", "back"],
        "face_labels": ["front", "back"], "manifest_label_semantics": "face_label",
        "class_names": {"front": "screw_A-1_front", "back": "screw_A-1_back"},
        "source": "phone/iVCam", "backend": "ivcam",
        "config": {"device": "téléphone", "fps": 30},
        "started_at": metadata["started_at"], "obb_labeled": False, "train_ready": False,
    }
    assert metadata["started_at"].endswith("+00:00")
    assert b"t\xc3\xa9l\xc3\xa9phone" in (session.path / "session.json").read_bytes()
    with pytest.raises(FileExistsError):
        _session(tmp_path)
    generated = CaptureSession.create(tmp_path, part_id="screw_A-1", source="phone", backend="ivcam", config={})
    assert generated.path.parent == tmp_path
    assert generated.path != session.path


def test_quality_metrics_dataclass_is_serialized(tmp_path):
    session = _session(tmp_path)
    _attempt(session, quality_metrics=QualityMetrics(12.0, 100.0))
    row = json.loads((session.path / "manifest.jsonl").read_text(encoding="utf-8"))
    assert row["quality_metrics"] == {"blur_score": 12.0, "brightness": 100.0}


def test_append_only_jsonl_attempts(tmp_path):
    session = _session(tmp_path)
    _attempt(session, warnings=["low light"], hash_value="00ff",
             nearest_duplicate_distance=3, timestamp="2025-01-01T00:00:00+00:00")
    manifest = session.path / "manifest.jsonl"
    original = manifest.read_bytes()
    assert original.endswith(b"\n")
    _attempt(session, label="back", mode="auto", accepted=False, saved_path=None,
             rejection_reasons=["duplicate"], hash_value="00ff",
             nearest_duplicate_distance=0)
    assert manifest.read_bytes().startswith(original)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0] == {
        "timestamp": "2025-01-01T00:00:00+00:00", "label": "front", "mode": "manual",
        "accepted": True, "saved_path": "front/frame-001.jpg",
        "quality_metrics": {"blur_score": 12.0, "brightness": 100.0},
        "warnings": ["low light"], "rejection_reasons": [], "hash": "00ff",
        "nearest_duplicate_distance": 3,
    }
    assert rows[1]["label"] == "back"
    assert rows[1]["mode"] == "auto"
    assert rows[1]["accepted"] is False
    assert rows[1]["saved_path"] is None
    assert rows[1]["rejection_reasons"] == ["duplicate"]
    assert rows[1]["nearest_duplicate_distance"] == 0
    assert rows[1]["timestamp"]


@pytest.mark.parametrize("saved_path", [
    None, "", "../front/image.jpg", "front/../back/image.jpg", "front/./image.jpg",
    "/tmp/image.jpg", "C:\\outside\\image.jpg", "\\\\server\\share\\file.jpg",
    "front\\image.jpg", "back/image.jpg", "front", "front/link/image.jpg",
])
def test_invalid_accepted_paths_do_not_append(tmp_path, saved_path):
    session = _session(tmp_path)
    # Existing symlink traversal must not allow paths outside the session.
    if saved_path == "front/link/image.jpg":
        try:
            (session.path / "front" / "link").symlink_to(tmp_path, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("directory symlinks unavailable")
    with pytest.raises(ValueError):
        _attempt(session, saved_path=saved_path)
    assert (session.path / "manifest.jsonl").read_bytes() == b""


@pytest.mark.parametrize("changes", [
    {"label": "side"}, {"mode": "burst"},
    {"accepted": False, "saved_path": "front/image.jpg"},
    {"quality_metrics": {"bad": float("nan")}},
    {"nearest_duplicate_distance": -1},
])
def test_invalid_attempt_does_not_append(tmp_path, changes):
    session = _session(tmp_path)
    with pytest.raises(ValueError):
        _attempt(session, **changes)
    assert (session.path / "manifest.jsonl").read_bytes() == b""


def test_part_id_is_required_before_creating_root(tmp_path):
    root = tmp_path / "not-created"
    with pytest.raises(TypeError, match="part_id"):
        CaptureSession.create(root, source="phone", backend="auto", config={})
    assert not root.exists()


@pytest.mark.parametrize("part_id", [None, "", "../escape", "a/b", "a\\b", ".", "-a",
                                    "a b", "C:bad", "é", "a\nb", 123])
def test_unsafe_part_ids_are_rejected_before_filesystem_mutation(tmp_path, part_id):
    root = tmp_path / "not-created"
    with pytest.raises(ValueError, match="part_id"):
        CaptureSession.create(root, part_id=part_id, session_id="pilot-01",
                              source="phone", backend="auto", config={})
    assert not root.exists()


@pytest.mark.parametrize("session_id", ["../escape", "a/b", "a\\b", ".", "..", "C:bad", ""])
def test_invalid_session_ids_are_rejected(tmp_path, session_id):
    with pytest.raises(ValueError):
        _session(tmp_path, session_id=session_id)
    assert list(tmp_path.iterdir()) == []
