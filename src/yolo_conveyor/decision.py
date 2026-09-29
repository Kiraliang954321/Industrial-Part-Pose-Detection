"""Pure, single-candidate decision calculation; not a tracking or control signal."""

from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence

from .geometry import angle_error, point_in_polygon
from .types import Point


class Outcome(str, Enum):
    PENDING = "pending"
    WRONG_FACE = "wrong_face"
    WRONG_ORIENTATION = "wrong_orientation"
    PASS = "pass"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    reason: str


def decide(center: Point, zone: Sequence[Point], face: str | None,
           expected_face: str, angle: float | None, target_angle: float,
           max_angle_error: float) -> Decision:
    """Evaluate one supplied candidate; missing evidence remains pending.

    This is a pure calculation for offline inspection only, not a production event.
    """
    if not 0 <= max_angle_error <= 90:
        raise ValueError("max_angle_error must be in [0, 90]")
    if not point_in_polygon(center, zone):
        return Decision(Outcome.PENDING, "outside_zone")
    if face is None:
        return Decision(Outcome.PENDING, "unknown_face")
    if face != expected_face:
        return Decision(Outcome.WRONG_FACE, "wrong_face")
    if angle is None:
        return Decision(Outcome.PENDING, "unknown_orientation")
    if angle_error(angle, target_angle) > max_angle_error:
        return Decision(Outcome.WRONG_ORIENTATION, "wrong_orientation")
    return Decision(Outcome.PASS, "matches")
