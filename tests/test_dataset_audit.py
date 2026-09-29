from pathlib import Path

from scripts.audit_dataset import audit_dataset, main, validate_row

VALID = "0 0 0 1 0 1 1 0 1\n"


def make_pair(root: Path, stem: str = "train/frame") -> tuple[Path, Path]:
    image = root / "images" / f"{stem}.jpg"
    label = root / "labels" / f"{stem}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image bytes are not decoded")
    label.write_text(VALID, encoding="utf-8")
    return image, label


def test_valid_pair_and_read_only(tmp_path, capsys):
    image, label = make_pair(tmp_path)
    before = (image.read_bytes(), label.read_bytes(), image.stat().st_mtime_ns, label.stat().st_mtime_ns)
    assert audit_dataset(tmp_path, class_count=2).errors == ()
    assert main([str(tmp_path), "--classes", "2"]) == 0
    assert '"errors": []' in capsys.readouterr().out
    assert before == (image.read_bytes(), label.read_bytes(), image.stat().st_mtime_ns, label.stat().st_mtime_ns)


def test_invalid_rows(tmp_path):
    _, label = make_pair(tmp_path)
    label.write_text("1 0 0 1 0 1 1 0\n"  # wrong length
                     "front 0 0 1 0 1 1 0 1\n"  # nonnumeric class
                     "0 a 0 1 0 1 1 0 1\n"  # nonnumeric coordinate
                     "0 0 0 1 0 1 1 0 1.1\n"  # out of range
                     "0 0 0 1 0 1 1 0 nan\n"  # nonfinite
                     "2 0 0 1 0 1 1 0 1\n", encoding="utf-8")
    result = audit_dataset(tmp_path, class_count=2)
    assert len(result.errors) == 6
    assert result.errors[0].startswith("labels/train/frame.txt:1:")
    assert "out of range" in result.errors[-1]
    assert main([str(tmp_path), "--classes", "2"]) == 1
    assert validate_row("1.0 0 0 1 0 1 1 0 1") is not None


def test_missing_pairs_and_negative_images(tmp_path):
    _, label = make_pair(tmp_path, "a")
    (tmp_path / "images" / "b.png").write_bytes(b"")
    (tmp_path / "labels" / "c.txt").write_text("", encoding="utf-8")
    label.write_text("", encoding="utf-8")  # empty file is a valid negative frame
    result = audit_dataset(tmp_path)
    assert result.errors == ("missing label: b.txt", "missing image: c")
    assert result.image_count == 2 and result.label_count == 2
