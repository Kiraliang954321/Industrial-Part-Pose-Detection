"""Single-piece, directed multi-frame decision. No ground truth or simulator input."""

from dataclasses import dataclass, asdict
import math
import re

from yolo_conveyor.orientation.geometry import directed_error


@dataclass(frozen=True)
class LineRecipe:
    target_heading_deg: float = 0.0
    tolerance_deg: float = 10.0
    guard_deg: float = 2.0
    frame_budget: int = 8
    min_valid_frames: int = 5
    min_valid_fraction: float = 0.75
    max_spread_deg: float = 5.0
    max_center_spread_m: float = 0.002
    observation_seconds: float = 0.7

    def __post_init__(self):
        numeric = [self.target_heading_deg, self.tolerance_deg, self.guard_deg, self.min_valid_fraction,
                   self.max_spread_deg, self.max_center_spread_m, self.observation_seconds]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError("recipe values must be finite")
        if (type(self.frame_budget) is not int or type(self.min_valid_frames) is not int or
                not 1 <= self.min_valid_frames <= self.frame_budget <= 100 or
                not 0 <= self.guard_deg < self.tolerance_deg < 180 or
                self.tolerance_deg + self.guard_deg > 180 or not 0 < self.min_valid_fraction <= 1 or
                not 0 < self.max_spread_deg < 90 or self.max_center_spread_m <= 0 or self.observation_seconds < 0):
            raise ValueError("invalid recipe bounds")

    def record(self):
        return asdict(self)


@dataclass(frozen=True)
class FrameEvidence:
    frame_id: int
    time_s: float
    heading_deg: float | None
    center_xy_m: tuple[float, float] | None
    rejection_reason: str | None = None

    def __post_init__(self):
        if type(self.frame_id) is not int or self.frame_id < 0 or not math.isfinite(self.time_s):
            raise ValueError("invalid frame identity/time")
        if self.rejection_reason is None:
            if self.heading_deg is None or not math.isfinite(self.heading_deg) or self.center_xy_m is None:
                raise ValueError("valid evidence needs direction and center")
            if len(self.center_xy_m) != 2 or not all(math.isfinite(v) for v in self.center_xy_m):
                raise ValueError("invalid frame center")


class InspectionEvent:
    """Observe one PLC-assigned job; finish emits at most one latched decision."""
    def __init__(self, event_id: str, start_s: float, recipe: LineRecipe):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", event_id) or not math.isfinite(start_s):
            raise ValueError("invalid event identity/start")
        self.event_id, self.start_s, self.recipe = event_id, start_s, recipe
        self.frames = []
        self.seen = set()
        self.duplicate_frames = 0
        self.late_frames = 0
        self.final = None

    def observe(self, evidence: FrameEvidence):
        if self.final is not None:
            return False
        if evidence.frame_id in self.seen:
            self.duplicate_frames += 1
            return False
        if evidence.time_s < self.start_s - 1e-9 or (self.frames and evidence.time_s < self.frames[-1].time_s - 1e-9):
            raise ValueError("frame timestamps must be monotonic")
        if evidence.time_s > self.start_s + self.recipe.observation_seconds + 1e-6:
            self.late_frames += 1
            return False
        if len(self.frames) >= self.recipe.frame_budget:
            return False
        self.seen.add(evidence.frame_id)
        self.frames.append(evidence)
        return True

    def finish(self, now_s: float):
        if self.final is not None:
            return None
        if not math.isfinite(now_s) or now_s < self.start_s - 1e-9 or (self.frames and now_s < self.frames[-1].time_s - 1e-9):
            raise ValueError("invalid finish timestamp")
        now_s = max(now_s, self.start_s, self.frames[-1].time_s if self.frames else self.start_s)
        r = self.recipe
        if len(self.frames) < r.frame_budget and now_s < self.start_s + r.observation_seconds - 1e-6:
            return None
        valid = [f for f in self.frames if f.rejection_reason is None]
        mean_angle = None; center = None; spread = None; center_spread = None
        reason, outcome = "insufficient_valid_frames", "UNCERTAIN"
        # Missing frames remain in the denominator instead of disappearing from the window.
        if len(valid) >= r.min_valid_frames and len(valid) / r.frame_budget >= r.min_valid_fraction:
            x = sum(math.cos(math.radians(f.heading_deg)) for f in valid)
            y = sum(math.sin(math.radians(f.heading_deg)) for f in valid)
            if math.hypot(x, y) / len(valid) < 1e-6:
                reason = "opposed_directions"
            else:
                mean_angle = math.degrees(math.atan2(y, x)) % 360
                center = tuple(sum(f.center_xy_m[i] for f in valid) / len(valid) for i in (0, 1))
                spread = max(directed_error(f.heading_deg, mean_angle) for f in valid)
                center_spread = max(math.dist(f.center_xy_m, center) for f in valid)
                if spread > r.max_spread_deg:
                    reason = "inconsistent_direction"
                elif center_spread > r.max_center_spread_m:
                    reason = "part_not_stationary"
                else:
                    error = directed_error(mean_angle, r.target_heading_deg)
                    if error <= r.tolerance_deg - r.guard_deg:
                        outcome, reason = "PASS", "stable_conforming_orientation"
                    elif error >= r.tolerance_deg + r.guard_deg:
                        outcome, reason = "FAIL_ORIENTATION", "stable_nonconforming_orientation"
                    else:
                        reason = "orientation_boundary_guard"
        self.final = {"schema_version": 1, "event_id": self.event_id, "outcome": outcome, "reason": reason,
                      "started_at_sim_s": self.start_s, "decided_at_sim_s": now_s,
                      "latency_sim_s": now_s - self.start_s, "frame_count": len(self.frames),
                      "valid_frame_count": len(valid), "expected_frame_count": r.frame_budget,
                      "duplicate_frame_count": self.duplicate_frames, "late_frame_count": self.late_frames,
                      "heading_world_deg": mean_angle, "center_xy_world_m": center,
                      "heading_spread_deg": spread, "center_spread_m": center_spread,
                      "recipe": r.record()}
        return self.final


def single_frame_decision(event_id, evidence, recipe):
    one = LineRecipe(target_heading_deg=recipe.target_heading_deg, tolerance_deg=recipe.tolerance_deg,
                     guard_deg=recipe.guard_deg, frame_budget=1, min_valid_frames=1, min_valid_fraction=1,
                     max_spread_deg=recipe.max_spread_deg, max_center_spread_m=recipe.max_center_spread_m,
                     observation_seconds=0)
    event = InspectionEvent(event_id, evidence.time_s, one)
    event.observe(evidence)
    return event.finish(evidence.time_s)
