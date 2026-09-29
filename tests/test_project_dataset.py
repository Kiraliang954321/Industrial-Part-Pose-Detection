import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from yolo_conveyor.annotation import AnnotationSession
from yolo_conveyor.annotation.contract import sha256_file
from yolo_conveyor.capture import CaptureSession
from yolo_conveyor.datasets.project_v1 import build_project_dataset

ROOT = Path(__file__).resolve().parents[1]
JPEG = bytes.fromhex("ffd8ffc00011080064006403011100021100031100ffd9")
CORNERS = [[0, 0], [0.8, 0], [0.8, 0.4], [0, 0.4]]


def session(root, session_id, part="gear", back=True, exclude=False):
    source = CaptureSession.create(root / "raw" / session_id, part_id=part, session_id=session_id,
                                   source="test", backend="test", config={})
    entries = [("front", "front/deep/a.jpeg", "annotated")]
    if back:
        entries.append(("back", "back/b.jpg", "annotated"))
    if exclude:
        entries.append(("front", "front/reject.jpg", "excluded"))
    for face, name, _ in entries:
        path = source.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(JPEG)
        source.record_attempt(label=face, mode="manual", accepted=True,
                              saved_path=name, quality_metrics={})
    store = AnnotationSession.create(root / "reviews" / session_id, source.path)
    for face, name, status in entries:
        record = dict(schema_version=1, status=status, part_id=part, face=face,
                      class_name=f"{part}_{face}", source_path=name,
                      source_image_sha256=sha256_file(source.path / name),
                      image_width=100, image_height=100)
        record.update(corners=CORNERS if face == "front" else [[0, 0], [1, 0], [1, 1], [0, 1]]) if status == "annotated" else record.update(reason="bad")
        store.write_record(record)
    return store, source.path


def setup(tmp_path):
    pairs = [session(tmp_path, f"s{i}", exclude=i == 1) for i in range(4)]
    plan = tmp_path / "plan.json"
    splits = {split: [f"s{i}"] for i, split in enumerate(("train", "val", "test", "final_holdout"))}
    plan.write_text(json.dumps({"schema_version": 2, "splits": splits}))
    return pairs, plan, splits


def test_assembly_and_dataset_audit(tmp_path):
    pairs, plan, splits = setup(tmp_path)
    output = tmp_path / "built"
    before = {(p, p.read_bytes()) for store, source in pairs for p in
              [*source.rglob("*.jpg"), *source.rglob("*.jpeg"), *store.path.rglob("*.json")]}
    manifest = build_project_dataset([store.path for store, _ in reversed(pairs)], plan, output)
    assert manifest == json.loads((output / "dataset_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["splits"] == splits
    assert manifest["final_holdout"] == {"path": "images/final_holdout",
                                         "labels_path": "labels/final_holdout",
                                         "purpose": "untuned final validation"}
    assert manifest["names"] == {"0": "gear_front", "1": "gear_back"}
    assert manifest["training_gate"] == "closed"
    assert manifest["counts"] == {s: {"images": 2, "faces": {"front": 1, "back": 1}} for s in splits}
    assert manifest["excluded_counts"] == {"train": 0, "val": 1, "test": 0, "final_holdout": 0}
    assert len(manifest["inputs"]) == 4
    assert all(len(entry["source_session_sha256"]) == 64 and
               len(entry["annotation_sha256"]) == 64 and entry["record_sha256"]
               for entry in manifest["inputs"])
    yaml = (output / "data.yaml").read_text()
    assert yaml == ('path: .\ntrain: images/train\nval: images/val\ntest: images/test\n'
                    'names:\n  0: "gear_front"\n  1: "gear_back"\n')
    assert "final_holdout" not in yaml
    assert not list(output.rglob("reject*"))
    for i, split in enumerate(splits):
        assert (output / f"images/{split}/s{i}/front/deep/a.jpeg").read_bytes() == JPEG
        assert (output / f"images/{split}/s{i}/back/b.jpg").read_bytes() == JPEG
        assert (output / f"labels/{split}/s{i}/front/deep/a.txt").read_text() == "0 0 0 0.80000000000000004 0 0.80000000000000004 0.40000000000000002 0 0.40000000000000002\n"
        assert (output / f"labels/{split}/s{i}/back/b.txt").read_text().startswith("1 ")
    assert {(p, p.read_bytes()) for p, _ in before} == before
    result = subprocess.run([sys.executable, str(ROOT / "scripts/audit_dataset.py"), str(output),
                             "--classes", "2"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["image_count"] == 8
    assert json.loads(result.stdout)["label_count"] == 8
    with pytest.raises(FileExistsError):
        build_project_dataset([store.path for store, _ in pairs], plan, output)
    assert subprocess.run([sys.executable, str(ROOT / "scripts/build_project_dataset.py"),
                           "--help"], capture_output=True).returncode == 0
    cli = subprocess.run([sys.executable, str(ROOT / "scripts/build_project_dataset.py"),
                          *(arg for store, _ in pairs for arg in ("--annotation", str(store.path))),
                          "--split-plan", str(plan), "--output", str(tmp_path / "cli")],
                         capture_output=True, text=True)
    assert cli.returncode == 0, cli.stderr
    assert (tmp_path / "cli/dataset_manifest.json").is_file()
    assert {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()} == {
        p.relative_to(tmp_path / "cli"): p.read_bytes() for p in (tmp_path / "cli").rglob("*") if p.is_file()}


@pytest.mark.parametrize("change", ["duplicate", "unknown", "unsafe", "empty", "extra", "version",
                                    "v1", "three_way", "missing_holdout", "empty_holdout"])
def test_invalid_plan_preflight(tmp_path, change):
    pairs, plan, splits = setup(tmp_path)
    if change == "duplicate":
        splits["test"].append("s0")
    elif change == "unknown":
        splits["test"] = ["alien"]
    elif change == "unsafe":
        splits["test"] = ["../s2"]
    elif change == "empty":
        splits["test"] = []
    elif change == "extra":
        splits["other"] = ["s2"]
    elif change == "three_way":
        splits.pop("final_holdout")
    elif change == "missing_holdout":
        splits.pop("final_holdout")
    elif change == "empty_holdout":
        splits["final_holdout"] = []
    payload = {"schema_version": 1 if change == "v1" else True if change == "version" else 2,
               "splits": splits}
    plan.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        build_project_dataset([store.path for store, _ in pairs], plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_unassigned_preflight(tmp_path):
    pairs, plan, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="unassigned"):
        build_project_dataset([s.path for s, _ in pairs] + [session(tmp_path, "s4")[0].path],
                              plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["mixed", "duplicate", "drift", "missing", "coverage", "overlap", "source_overlap", "occupied", "dangling"])
def test_invalid_inputs_preflight(tmp_path, change):
    pairs, plan, _ = setup(tmp_path)
    workspaces = [s.path for s, _ in pairs]
    output = tmp_path / "out"
    if change == "mixed":
        workspaces[2] = session(tmp_path / "other", "s2", part="bolt")[0].path
    elif change == "duplicate":
        workspaces.append(workspaces[0])
    elif change == "drift":
        (pairs[1][1] / "front/deep/a.jpeg").write_bytes(b"bad")
    elif change == "missing":
        pairs[1][0].record_file("front/deep/a.jpeg").unlink()
    elif change == "coverage":
        record = pairs[2][0].read_record("back/b.jpg")
        record.pop("corners")
        record.update(status="excluded", reason="hidden")
        pairs[2][0].write_record(record)
    elif change == "overlap":
        output = pairs[0][0].path / "nested"
    elif change == "source_overlap":
        output = pairs[0][1] / "nested"
    elif change == "occupied":
        output.write_text("do not overwrite")
    elif change == "dangling":
        try:
            output.symlink_to(tmp_path / "not-existing")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
    with pytest.raises((ValueError, FileExistsError)):
        build_project_dataset(workspaces, plan, output)
    if change == "occupied":
        assert output.read_text() == "do not overwrite"
    elif change == "dangling":
        assert os.path.lexists(output)
    else:
        assert not output.exists()


@pytest.mark.parametrize("split_index", range(4))
def test_each_split_requires_annotated_front_and_back(tmp_path, split_index):
    pairs, plan, _ = setup(tmp_path)
    record = pairs[split_index][0].read_record("back/b.jpg")
    record.pop("corners")
    record.update(status="excluded", reason="hidden")
    pairs[split_index][0].write_record(record)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="lacks annotated front/back coverage"):
        build_project_dataset([s.path for s, _ in pairs], plan, output)
    assert not output.exists()


def test_copy_fallback_and_visible_write_failure(tmp_path, monkeypatch):
    pairs, plan, _ = setup(tmp_path)
    import yolo_conveyor.datasets.project_v1 as module
    def no_link(*args):
        raise OSError("no links")
    monkeypatch.setattr(module.os, "link", no_link)
    output = tmp_path / "copy"
    module.build_project_dataset([s.path for s, _ in pairs], plan, output)
    assert (output / "images/train/s0/front/deep/a.jpeg").read_bytes() == JPEG
    def no_copy(*args):
        raise OSError("disk full")
    monkeypatch.setattr(module.shutil, "copyfile", no_copy)
    with pytest.raises(OSError, match="disk full"):
        module.build_project_dataset([s.path for s, _ in pairs], plan, tmp_path / "partial")
    assert (tmp_path / "partial").is_dir()
