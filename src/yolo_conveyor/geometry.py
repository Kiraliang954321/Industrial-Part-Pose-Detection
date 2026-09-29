"""Undirected long-axis geometry in image coordinates (x right, y down)."""

import math
from collections.abc import Sequence

from .types import Point


def normalize_angle(angle: float) -> float:
    """Normalize an axial angle to [0, 180) degrees."""
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    return angle % 180.0


def angle_error(predicted: float, expected: float) -> float:
    """Smallest axial difference, in [0, 90] degrees."""
    difference = abs(normalize_angle(predicted) - normalize_angle(expected))
    return min(difference, 180.0 - difference)


def long_axis_angle(corners: Sequence[Point], square_tolerance: float = 0.05) -> float | None:
    """Angle from ordered OBB corners; None when near-square/degenerate.

    The four points must follow the box perimeter, in either direction.
    square_tolerance is the minimum relative difference of the two edge lengths.
    """
    if len(corners) != 4 or not 0 <= square_tolerance < 1:
        raise ValueError("expected four corners and square_tolerance in [0, 1)")
    edges = []
    for i in range(4):
        x, y = corners[i]
        next_x, next_y = corners[(i + 1) % 4]
        if not all(math.isfinite(v) for v in (x, y, next_x, next_y)):
            raise ValueError("corners must be finite")
        dx, dy = next_x - x, next_y - y
        edges.append((math.hypot(dx, dy), dx, dy))
    # Average opposite sides to avoid dependence on corner starting position.
    lengths = ((edges[0][0] + edges[2][0]) / 2,
               (edges[1][0] + edges[3][0]) / 2)
    longest = max(lengths)
    if longest == 0 or (longest - min(lengths)) / longest <= square_tolerance:
        return None
    index = 0 if lengths[0] > lengths[1] else 1
    if edges[index][0] == 0:
        return None
    return normalize_angle(math.degrees(math.atan2(edges[index][2], edges[index][1])))


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Even-odd containment; points on polygon boundary count as inside."""
    if len(polygon) < 3:
        raise ValueError("polygon needs at least three vertices")
    x, y = point
    if not all(math.isfinite(v) for p in (point, *polygon) for v in p):
        raise ValueError("coordinates must be finite")
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-9 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside
