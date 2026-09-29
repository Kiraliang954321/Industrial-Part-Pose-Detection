import json
import subprocess
import sys
from pathlib import Path

import pytest

from yolo_conveyor.annotation import audit_annotations
from test_annotation_store import setup_workspace


@pytest.mark.parametrize("change,phrase", [
    ("session", "session.json hash drift"),
    ("manifest", "manifest.jsonl hash drift"),
    ("image", "source image hash drift"),
    ("face", "invalid review record"),
    ("part", "record identity/face/class mismatch"),
    ("class", "invalid review record"),
    ("path", "invalid review record"),
    ("geometry", "invalid review record"),
    ("metadata", "annotation metadata inconsistent"),
    ("dimensions", "source image dimensions mismatch"),
])
def test_qa_drift_and_corruption(tmp_path, change, phrase):
    source, store, record = setup_workspace(tmp_path)
    store.write_record(record)
    target = store.record_file("front/a.jpg")
    if change == "session":
        (source / "session.json").write_bytes((source / "session.json").read_bytes() + b" ")
    elif change == "manifest":
        with (source / "manifest.jsonl").open("ab") as stream:
            stream.write(b" ")
    elif change == "image":
        (source / "front/a.jpg").write_bytes(b"drift")
    elif change == "metadata":
        meta = store.metadata()
        meta["part_id"] = "other"
        (store.path / "annotation.json").write_text(json.dumps(meta))
    else:
        changed = store.read_record("front/a.jpg")
        if change == "face":
            changed.update(face="back", class_name="gear_back")
        elif change == "part":
            changed.update(part_id="other", class_name="other_front")
        elif change == "class":
            changed["class_name"] = "bad"
        elif change == "path":
            changed["source_path"] = "../bad.jpg"
        elif change == "geometry":
            changed["corners"] = [[0, 0]] * 4
        elif change == "dimensions":
            changed["image_width"] = 200
        target.write_text(json.dumps(changed))
    assert any(phrase in error for error in audit_annotations(store.path).errors)


def test_missing_excluded_and_near_square(tmp_path):
    _, store, record = setup_workspace(tmp_path)
    result = audit_annotations(store.path)
    assert (result.expected, result.missing, result.annotated) == (1, 1, 0)
    assert result.errors
    store.write_record({**record, "corners": [[0, 0], [1, 0], [1, 1], [0, 1]]})
    result = audit_annotations(store.path)
    assert (result.expected, result.annotated, result.angle_unevaluable) == (1, 1, 1)
    assert not result.errors and result.warnings
    excluded = {k: v for k, v in record.items() if k != "corners"}
    excluded.update(status="excluded", reason="occluded")
    store.write_record(excluded)
    result = audit_annotations(store.path)
    assert (result.excluded, result.angle_unevaluable) == (1, 0)
    assert not result.errors


def test_cli_json_and_exit_code(tmp_path):
    _, store, record = setup_workspace(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/qa_annotations.py"
    def run():
        return subprocess.run([sys.executable, str(script), str(store.path)], capture_output=True, text=True)
    missing = run()
    assert missing.returncode == 1 and json.loads(missing.stdout)["missing"] == 1
    store.write_record(record)
    clean = run()
    assert clean.returncode == 0
    assert json.loads(clean.stdout)["errors"] == []
    assert "warnings" in json.loads(clean.stdout)


def test_unsafe_source_symlink_and_unexpected_review(tmp_path):
    source, store, record = setup_workspace(tmp_path)
    store.write_record(record)
    stray = store.path / "records/back/extra.json"
    stray.parent.mkdir()
    stray.write_text(json.dumps(record))
    assert any("unexpected" in error for error in audit_annotations(store.path).errors)
    stray.unlink()
    image = source / "front/a.jpg"
    image.unlink()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"other")
    try:
        image.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert any("unsafe" in error for error in audit_annotations(store.path).errors)


def test_unsafe_record_symlink(tmp_path):
    _, store, record = setup_workspace(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(record))
    directory = store.path / "records/front"
    directory.mkdir()
    try:
        (directory / "a.json").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert any("unsafe" in error for error in audit_annotations(store.path).errors)
