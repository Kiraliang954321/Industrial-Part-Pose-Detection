import pytest

from yolo_conveyor.annotation import angle_degrees, validate_corners, validate_record
from yolo_conveyor.annotation.contract import safe_source_path

BOX = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.4], [0.1, 0.4]]


def test_both_windings_and_pixel_angle():
    for box in (BOX, list(reversed(BOX))):
        assert len(validate_corners(box)) == 4
        assert angle_degrees(box, 100, 100) == 0
    # normalized edge x=.4 y=.6, but pixel edge x=40 y=120.
    tilted = [[0.1, 0.1], [0.5, 0.7], [0.4, 0.75], [0, 0.15]]
    assert 70 < angle_degrees(tilted, 100, 200) < 73
    assert angle_degrees([[0, 0], [1, 0], [1, 1], [0, 1]], 100, 100) is None


@pytest.mark.parametrize("box", [
    [], BOX[:3], [[0, 0], [0, 0], [1, 1], [1, 0]],
    [[0, 0], [1, 1], [1, 0], [0, 1]],  # crossing
    [[0, 0], [1, 0], [0.5, 0.2], [0, 1]],  # concave
    [[0, 0], [0.5, 0], [1, 0], [0, 1]],  # collinear
    [[0, 0], [2, 0], [1, 1], [0, 1]],
    [[0, 0], [float('nan'), 0], [1, 1], [0, 1]],
])
def test_malformed_geometry_rejected(box):
    with pytest.raises(ValueError):
        validate_corners(box)


@pytest.mark.parametrize("name", ["../front/a.jpg", "front/../a.jpg", "C:/a.jpg",
                                      "front\\a.jpg", "back/a.jpg", "front/a.png"])
def test_unsafe_source_paths(name):
    with pytest.raises(ValueError):
        safe_source_path(name, "front")


def test_record_shape_and_exclusion():
    base = dict(schema_version=1, status="annotated", part_id="widget", face="front",
                class_name="widget_front", source_path="front/a.jpg", source_image_sha256="a" * 64,
                image_width=100, image_height=200, corners=BOX)
    validate_record(base)
    with pytest.raises(ValueError):
        validate_record({**base, "class_id": 0})
    excluded = {k: v for k, v in base.items() if k != "corners"}
    excluded.update(status="excluded", reason="obscured")
    validate_record(excluded)
    with pytest.raises(ValueError):
        validate_record({**excluded, "corners": []})
    with pytest.raises(ValueError):
        validate_record({**excluded, "reason": " "})
