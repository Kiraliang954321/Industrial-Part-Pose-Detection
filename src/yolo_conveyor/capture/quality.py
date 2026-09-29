"""Frame quality measurement and configurable warn/reject policy."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


def grayscale(frame: np.ndarray) -> np.ndarray:
    """Return a uint8 grayscale image from grayscale or BGR input."""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("frame must be a uint8 numpy array")
    if frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("frame must have nonzero width and height")
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError("frame must be grayscale (H, W) or BGR (H, W, 3)")


@dataclass(frozen=True)
class QualityMetrics:
    blur_score: float
    brightness: float


def measure_quality(frame: np.ndarray) -> QualityMetrics:
    """Measure Laplacian variance (higher is sharper) and mean grayscale level."""
    gray = grayscale(frame)
    return QualityMetrics(
        blur_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        brightness=float(gray.mean()),
    )


@dataclass(frozen=True)
class QualityResult:
    metrics: QualityMetrics
    warnings: tuple[str, ...]
    accepted: bool


@dataclass(frozen=True)
class QualityPolicy:
    """Warn by default; set reject=True to reject frames failing a threshold."""

    min_blur_score: float = 0.0
    min_brightness: float = 0.0
    max_brightness: float = 255.0
    reject: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_blur_score) or self.min_blur_score < 0:
            raise ValueError("min_blur_score must be finite and nonnegative")
        if not (math.isfinite(self.min_brightness) and math.isfinite(self.max_brightness)
                and 0 <= self.min_brightness <= self.max_brightness <= 255):
            raise ValueError("brightness thresholds must be ordered within [0, 255]")

    def evaluate(self, frame: np.ndarray) -> QualityResult:
        metrics = measure_quality(frame)
        warnings = []
        if metrics.blur_score < self.min_blur_score:
            warnings.append("blur")
        if metrics.brightness < self.min_brightness:
            warnings.append("dark")
        if metrics.brightness > self.max_brightness:
            warnings.append("bright")
        return QualityResult(metrics, tuple(warnings), not (self.reject and warnings))
