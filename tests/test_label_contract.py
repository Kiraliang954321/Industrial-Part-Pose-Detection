import pytest

from yolo_conveyor.label_contract import FACE_LABELS, canonical_class_name, validate_part_id


def test_face_labels_and_canonical_names():
    assert FACE_LABELS == ("front", "back")
    for part_id in ("A", "7", "Screw_02-A"):
        assert validate_part_id(part_id) == part_id
        assert canonical_class_name(part_id, "front") == f"{part_id}_front"
        assert canonical_class_name(part_id, "back") == f"{part_id}_back"


@pytest.mark.parametrize("part_id", [None, 1, "", "_part", "-part", ".", "..",
                                    "part/name", "part\\name", "part name", "é", "a\nb"])
def test_invalid_part_id(part_id):
    with pytest.raises(ValueError, match="part_id"):
        validate_part_id(part_id)
    with pytest.raises(ValueError, match="part_id"):
        canonical_class_name(part_id, "front")


@pytest.mark.parametrize("face", ["side", "FRONT", "", None])
def test_invalid_face(face):
    with pytest.raises(ValueError, match="face"):
        canonical_class_name("part_1", face)
