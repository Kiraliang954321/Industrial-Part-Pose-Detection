from yolo_conveyor.decision import Outcome, decide

ZONE = ((0, 0), (10, 0), (10, 10), (0, 10))


def test_outside_zone_pending_even_with_bad_face():
    result = decide((12, 5), ZONE, "back", "front", 90, 0, 5)
    assert result.outcome == Outcome.PENDING
    assert result.reason == "outside_zone"


def test_wrong_face_failure():
    assert decide((5, 5), ZONE, "back", "front", 0, 0, 5).outcome == Outcome.WRONG_FACE


def test_wrong_orientation_failure():
    assert decide((5, 5), ZONE, "front", "front", 20, 0, 5).outcome == Outcome.WRONG_ORIENTATION


def test_pass_with_modulo_180_wraparound():
    assert decide((5, 5), ZONE, "front", "front", 179, 1, 2).outcome == Outcome.PASS


def test_unknown_orientation_remains_pending():
    assert decide((5, 5), ZONE, "front", "front", None, 0, 5).outcome == Outcome.PENDING
