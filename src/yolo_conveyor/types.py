"""Plain image-coordinate data contracts, independent of inference libraries."""

from dataclasses import dataclass

Point = tuple[float, float]
Corners = tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class Detection:
    class_index: int
    confidence: float
    corners: Corners

    @property
    def center(self) -> Point:
        return (sum(p[0] for p in self.corners) / 4,
                sum(p[1] for p in self.corners) / 4)
