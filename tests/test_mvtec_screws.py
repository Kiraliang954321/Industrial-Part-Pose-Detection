import json
import math
import os
from pathlib import Path

import pytest

from scripts.audit_dataset import audit_dataset
from scripts.prepare_mvtec_screws import main
from yolo_conveyor.datasets import convert_mvtec_screws
from yolo_conveyor.datasets.mvtec_screws import bbox_corners


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "raw"
    (root / "images").mkdir(parents=True)
    categories = [{"id": 12, "name": "screw"}, {"id": 3, "name": "nut"}]
    for split, category_id in (("train", 12), ("val", 3), ("test", None)):
        (root / "images" / f"{split}.jpg").write_bytes(split.encode())
        data = {"images": [{"id": 1, "file_name": f"{split}.jpg", "width": 100, "height": 100}],
                "categories": categories,
                "annotations": [] if category_id is None else
                [{"image_id": 1, "category_id": category_id, "bbox": [50, 40, 20, 10, 0]}]}
        (root / f"mvtec_screws_{split}.json").write_text(json.dumps(data), encoding="utf-8")
    return root


def test_geometry_phi_zero_and_half_pi():
    assert bbox_corners([50, 40, 20, 10, 0], 100, 100) == pytest.approx(
        (.3, .45, .5, .45, .5, .55, .3, .55))
    assert bbox_corners([50, 40, 20, 10, math.pi / 2], 100, 100) == pytest.approx(
        (.35, .6, .35, .4, .45, .4, .45, .6))


@pytest.mark.parametrize("bbox", [
    [50, 40, 20, 10], [50, 40, 0, 10, 0], [50, 40, -20, 10, 0],
    [50, 40, 20, 10, float("nan")], [50, 40, 20, 10, float("inf")],
    [50, 40, float("inf"), 10, 0], [50, 4, 20, 10, 0],
    [50, 40, 1e308, 10, 0], [50, 1.7e308, 1.7e308, 10, 0],
    [50, 40, 20, 10, "0"],
])
def test_invalid_geometry_rejected(bbox):
    with pytest.raises(ValueError):
        bbox_corners(bbox, 100, 100)


def test_split_mapping_audit_and_source_unchanged(source, tmp_path):
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "processed"
    manifest = convert_mvtec_screws(source, output)
    assert manifest["category_id_map"] == {"3": 0, "12": 1}
    assert manifest["names"] == ["nut", "screw"]
    assert [manifest["splits"][s]["annotations"] for s in ("train", "val", "test")] == [1, 1, 0]
    assert (output / "labels/train/train.txt").read_text().startswith("1 ")
    assert (output / "labels/val/val.txt").read_text().startswith("0 ")
    assert (output / "labels/test/test.txt").read_text() == ""
    for split in ("train", "val", "test"):
        assert (output / "images" / split / f"{split}.jpg").read_bytes() == split.encode()
    assert audit_dataset(output, class_count=2).errors == ()
    assert "train: images/train" in (output / "data.yaml").read_text()
    assert json.loads((output / "conversion_manifest.json").read_text()) == manifest
    assert before == {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        convert_mvtec_screws(source, output)


def test_hardlinks(source, tmp_path):
    output = tmp_path / "linked"
    convert_mvtec_screws(source, output)
    assert os.path.samefile(source / "images/train.jpg", output / "images/train/train.jpg")


def test_copy_fallback(source, tmp_path, monkeypatch):
    def cannot_link(*args):
        raise OSError("cross-device link")

    monkeypatch.setattr("yolo_conveyor.datasets.mvtec_screws.os.link", cannot_link)
    output = tmp_path / "copied"
    convert_mvtec_screws(source, output)
    assert (output / "images/train/train.jpg").read_bytes() == b"train"
    assert not os.path.samefile(source / "images/train.jpg", output / "images/train/train.jpg")
    assert audit_dataset(output, 2).errors == ()


def test_cli(source, tmp_path, capsys):
    assert main([str(source), str(tmp_path / "cli")]) == 0
    assert "train: 1 images, 1 annotations" in capsys.readouterr().out


def test_conversion_rejects_invalid_before_writing(source, tmp_path):
    split = source / "mvtec_screws_train.json"
    data = json.loads(split.read_text())
    data["annotations"][0]["bbox"] = [50, 2, 20, 10, 0]
    split.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="corner"):
        convert_mvtec_screws(source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_nonfinite_json_and_unknown_category_rejected(source, tmp_path):
    split = source / "mvtec_screws_train.json"
    data = json.loads(split.read_text())
    data["annotations"][0]["category_id"] = 99
    split.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown"):
        convert_mvtec_screws(source, tmp_path / "unknown")
    data["annotations"][0]["category_id"] = 12
    data["annotations"][0]["bbox"][0] = float("nan")
    split.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="non-finite"):
        convert_mvtec_screws(source, tmp_path / "nonfinite")
    assert not (tmp_path / "unknown").exists()
    assert not (tmp_path / "nonfinite").exists()


def test_traversal_rejected(source, tmp_path):
    split = source / "mvtec_screws_train.json"
    data = json.loads(split.read_text())
    data["images"][0]["file_name"] = "../outside.jpg"
    split.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source"):
        convert_mvtec_screws(source, tmp_path / "invalid")
