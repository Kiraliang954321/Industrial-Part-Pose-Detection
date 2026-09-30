"""Interactive raw JPEG collector; labels are chosen by the operator, never inferred."""

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from .duplicates import DuplicateGuard, dhash, hamming_distance
from .quality import QualityMetrics, QualityPolicy
from .session import CaptureSession


@dataclass(frozen=True)
class AttemptResult:
    accepted: bool
    saved_path: str | None
    warnings: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    metrics: QualityMetrics
    hash_value: str
    nearest_duplicate_distance: int | None


class CaptureCollector:
    """Keep counters and duplicate histories local to a new capture session."""

    def __init__(self, session: CaptureSession, *, quality: QualityPolicy | None = None,
                 duplicates: DuplicateGuard | None = None) -> None:
        self.session = session
        self.quality = quality if quality is not None else QualityPolicy()
        self.duplicates = duplicates if duplicates is not None else DuplicateGuard()
        self.label = "front"
        self.counts = {"front": 0, "back": 0}

    def attempt(self, frame: np.ndarray, mode: str = "manual") -> AttemptResult:
        """Record one attempt. Only successful JPEG writes enter duplicate history."""
        if mode not in ("manual", "auto"):
            raise ValueError("mode must be manual or auto")
        label = self.label
        assessment = self.quality.evaluate(frame)
        value = dhash(frame, self.duplicates.hash_size)
        history = self.duplicates.history(label)
        nearest = min((hamming_distance(value, old) for old in history), default=None)
        reasons = []
        if not assessment.accepted:
            reasons.append("quality")
        if nearest is not None and nearest <= self.duplicates.max_distance:
            reasons.append("duplicate")
        relative = None
        if not reasons:
            # The session directory was exclusively created. Skip any existing name rather
            # than silently overwriting a file, including files placed there externally.
            number = self.counts[label] + 1
            while (self.session.path / label / f"{number:06d}.jpg").exists():
                number += 1
            relative = f"{label}/{number:06d}.jpg"
            target = self.session.path / relative
            try:
                if not cv2.imwrite(str(target), frame):
                    raise OSError("JPEG write returned false")
            except (OSError, cv2.error):
                # OpenCV may have left a partial file on a failed write.
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    pass  # A cleanup failure must not hide the rejected attempt.
                relative = None
                reasons.append("image_write_failed")
            else:
                self.counts[label] += 1
                self.duplicates.check_and_add_hash(value, label)
        accepted = not reasons
        result = AttemptResult(accepted, relative, assessment.warnings, tuple(reasons),
                               assessment.metrics, f"{value:0{self.duplicates.hash_size ** 2 // 4}x}", nearest)
        self.session.record_attempt(
            label=label, mode=mode, accepted=accepted, saved_path=relative,
            quality_metrics=assessment.metrics, warnings=result.warnings,
            rejection_reasons=result.rejection_reasons, hash_value=result.hash_value,
            nearest_duplicate_distance=nearest,
        )
        return result


def open_source(source: int | str, backend: str = "auto", *, width: float | None = None,
                height: float | None = None, fps: float | None = None):
    """Open camera index or video path; DirectShow applies only to camera indices."""
    if backend not in ("auto", "dshow"):
        raise ValueError("backend must be auto or dshow")
    if isinstance(source, int) and not isinstance(source, bool) and source >= 0 and backend == "dshow":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open capture source: {source}")
    for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, width), (cv2.CAP_PROP_FRAME_HEIGHT, height),
                        (cv2.CAP_PROP_FPS, fps)):
        if value is not None and value > 0:
            cap.set(prop, value)
    return cap


def _overlay(frame: np.ndarray, *, source: int | str, collector: CaptureCollector,
             auto: bool, metrics: QualityMetrics, warnings: tuple[str, ...],
             status: str | None = None) -> np.ndarray:
    preview = frame.copy()
    lines = [
        f"Source: {source}  Face: {collector.label.upper()}  Auto: {'ON' if auto else 'OFF'}",
        f"Accepted  front: {collector.counts['front']}  back: {collector.counts['back']}",
        f"Blur: {metrics.blur_score:.1f}  Brightness: {metrics.brightness:.1f}",
        f"Warnings: {', '.join(warnings) if warnings else 'none'}",
        "F front  B back  Space capture  A auto  Q/Esc quit",
    ]
    if status is not None:
        lines.append(status)
    for index, line in enumerate(lines):
        y = 25 + index * 25
        cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2, cv2.LINE_AA)
    return preview


def run_collector(collector: CaptureCollector, source: int | str, *, backend: str = "auto",
                  width: float | None = None, height: float | None = None,
                  fps: float | None = None, auto_interval: float = 0,
                  max_read_failures: int = 10) -> None:
    """Preview live frames until quit; raise RuntimeError for source/read failures."""
    if not math.isfinite(auto_interval) or auto_interval < 0 or max_read_failures < 1:
        raise ValueError("auto_interval must be finite and nonnegative; max_read_failures positive")
    cap = None
    auto = False
    deadline = None
    failures = 0
    status = None
    status_until = 0.0
    try:
        cap = open_source(source, backend, width=width, height=height, fps=fps)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                failures += 1
                if failures >= max_read_failures:
                    raise RuntimeError(f"capture source {source}: repeated frame-read failure")
                continue
            failures = 0
            assessment = collector.quality.evaluate(frame)
            if status is not None and time.monotonic() >= status_until:
                status = None
            cv2.imshow("Phone capture", _overlay(frame, source=source, collector=collector,
                        auto=auto, metrics=assessment.metrics, warnings=assessment.warnings,
                        status=status))
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), ord("Q"), 27):
                break
            result = None
            if key in (ord("f"), ord("F")):
                collector.label = "front"
            elif key in (ord("b"), ord("B")):
                collector.label = "back"
            elif key in (ord("a"), ord("A")) and auto_interval > 0:
                auto = not auto
                deadline = time.monotonic() + auto_interval if auto else None
            elif key == 32:
                result = collector.attempt(frame, "manual")
            if auto and deadline is not None and time.monotonic() >= deadline:
                result = collector.attempt(frame, "auto")
                deadline = time.monotonic() + auto_interval
            if result is not None:
                status = (f"SAVED {result.saved_path}" if result.accepted else
                          f"REJECTED: {', '.join(result.rejection_reasons)}")
                status_until = time.monotonic() + 1.5
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
