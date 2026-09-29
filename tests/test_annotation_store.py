import json
from pathlib import Path

import pytest

from yolo_conveyor.annotation import AnnotationSession
from yolo_conveyor.annotation.contract import sha256_file
from yolo_conveyor.capture import CaptureSession


def setup_workspace(tmp_path):
    source = CaptureSession.create(tmp_path / "raw", part_id="gear", session_id="sample",
                                   source="test", backend="test", config={})
    image = source.path / "front" / "a.jpg"
    # Minimal JPEG frame header with 100 x 100 dimensions (pixels are not decoded).
    image.write_bytes(bytes.fromhex("ffd8ffc00011080064006403011100021100031100ffd9"))
    source.record_attempt(label="front", mode="manual", accepted=True,
                          saved_path="front/a.jpg", quality_metrics={})
    source.record_attempt(label="back", mode="auto", accepted=False,
                          saved_path=None, quality_metrics={})
    workspace = tmp_path / "review" / "sample"
    originals = {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in
                 (source.path / "session.json", source.path / "manifest.jsonl", image)}
    store = AnnotationSession.create(workspace, source.path)
    assert {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in originals} == originals
    base = dict(schema_version=1, status="annotated", part_id="gear", face="front",
                class_name="gear_front", source_path="front/a.jpg",
                source_image_sha256=sha256_file(image), image_width=100, image_height=100,
                corners=[[0, 0], [0.8, 0], [0.8, 0.4], [0, 0.4]])
    return source.path, store, base


def test_creation_and_round_trip_atomic_replacement(tmp_path, monkeypatch):
    source, store, base = setup_workspace(tmp_path)
    meta = store.metadata()
    assert meta["source_session_sha256"] == sha256_file(source / "session.json")
    assert meta["source_manifest_sha256"] == sha256_file(source / "manifest.jsonl")
    assert meta["source_session_id"] == "sample"
    path = store.write_record(base)
    assert path == store.path / "records/front/a.json"
    assert store.read_record("front/a.jpg") == base
    excluded = {k: v for k, v in base.items() if k != "corners"}
    excluded.update(status="excluded", reason="unreadable")
    store.write_record(excluded)
    assert store.read_record("front/a.jpg") == excluded
    assert list(path.parent.iterdir()) == [path]
    import yolo_conveyor.annotation.store as module
    def fail_replace(*args):
        raise OSError("replace failed")
    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.write_record(base)
    assert store.read_record("front/a.jpg") == excluded
    assert list(path.parent.iterdir()) == [path]
    with pytest.raises(FileExistsError):
        AnnotationSession.create(store.path, source)


def test_unsafe_and_unaccepted_records(tmp_path):
    source, store, base = setup_workspace(tmp_path)
    for path in ("../front/a.jpg", "front/../a.jpg", "back/a.jpg", "front/b.jpg"):
        with pytest.raises(ValueError):
            store.write_record({**base, "source_path": path})
    with pytest.raises(ValueError):
        store.write_record({**base, "source_image_sha256": "0" * 64})
    with pytest.raises(ValueError):
        store.write_record({**base, "class_name": "wrong"})
    with pytest.raises(ValueError, match="dimensions"):
        store.write_record({**base, "image_width": 200})


def test_nested_accepted_path_mirrors_relative_stem(tmp_path):
    source = CaptureSession.create(tmp_path / "raw", part_id="gear", session_id="nested",
                                   source="test", backend="test", config={})
    image = source.path / "back/sub/a.jpeg"
    image.parent.mkdir()
    image.write_bytes(bytes.fromhex("ffd8ffc00011080064006403011100021100031100ffd9"))
    source.record_attempt(label="back", mode="manual", accepted=True,
                          saved_path="back/sub/a.jpeg", quality_metrics={})
    store = AnnotationSession.create(tmp_path / "workspace", source.path)
    record = dict(schema_version=1, status="excluded", part_id="gear", face="back",
                  class_name="gear_back", source_path="back/sub/a.jpeg",
                  source_image_sha256=sha256_file(image), image_width=100, image_height=100,
                  reason="hidden")
    assert store.write_record(record) == store.path / "records/back/sub/a.json"
    assert store.read_record("back/sub/a.jpeg") == record


def test_creation_rejects_invalid_source_and_workspace_inside_source(tmp_path):
    source, _, _ = setup_workspace(tmp_path)
    with pytest.raises(ValueError):
        AnnotationSession.create(source / "review", source)
    meta = json.loads((source / "session.json").read_text())
    meta["schema_version"] = 1
    (source / "session.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        AnnotationSession.create(tmp_path / "invalid", source)
    assert not (tmp_path / "invalid").exists()
