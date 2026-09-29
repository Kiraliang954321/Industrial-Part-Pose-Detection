import pytest

from yolo_conveyor.geometry import angle_error, long_axis_angle, normalize_angle, point_in_polygon


def test_normalization_and_wraparound():
    assert normalize_angle(180) == 0
    assert normalize_angle(-1) == 179
    assert normalize_angle(541) == 1
    assert angle_error(179, 1) == pytest.approx(2)
    assert angle_error(-1, 181) == pytest.approx(2)
    with pytest.raises(ValueError):
        normalize_angle(float("nan"))


def test_long_axis_independent_of_start_and_width_height_swap():
    box = ((0, 0), (1, 0), (1, 4), (0, 4))
    assert long_axis_angle(box) == pytest.approx(90)
    assert long_axis_angle(box[2:] + box[:2]) == pytest.approx(90)
    assert long_axis_angle(((0, 0), (-4, 1), (-4.25, 0), (-0.25, -1))) is not None
    assert long_axis_angle(((0, 0), (2, 0), (2, 2), (0, 2))) is None
    assert long_axis_angle(((0, 0), (0, 0), (0, 0), (0, 0))) is None


def test_polygon_boundary_inside():
    polygon = ((0, 0), (2, 0), (2, 2), (0, 2))
    assert point_in_polygon((0, 1), polygon)
    assert point_in_polygon((1, 1), polygon)
    assert not point_in_polygon((3, 1), polygon)
