"""Reusable capture assessment, duplicate detection and session metadata."""

from .collector import AttemptResult, CaptureCollector, open_source, run_collector
from .duplicates import DuplicateGuard, dhash, hamming_distance
from .quality import QualityMetrics, QualityPolicy, QualityResult, measure_quality
from .session import CaptureSession

__all__ = [
    "AttemptResult",
    "CaptureCollector",
    "CaptureSession",
    "DuplicateGuard",
    "QualityMetrics",
    "QualityPolicy",
    "QualityResult",
    "dhash",
    "hamming_distance",
    "measure_quality",
    "open_source",
    "run_collector",
]
