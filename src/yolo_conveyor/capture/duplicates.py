"""Deterministic difference hashes and bounded per-side duplicate detection."""

from collections import deque

import cv2
import numpy as np

from .quality import grayscale


def dhash(frame: np.ndarray, hash_size: int = 8) -> int:
    """Row-major difference hash; first comparison occupies the highest bit.

    Pixels are compared left > right (ties produce zero). Input is grayscale or BGR.
    """
    if isinstance(hash_size, bool) or not isinstance(hash_size, int) or hash_size < 1:
        raise ValueError("hash_size must be a positive integer")
    gray = grayscale(frame)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (resized[:, :-1] > resized[:, 1:]).flat
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    """Count differing bits in two nonnegative integer hashes."""
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (left, right)):
        raise ValueError("hashes must be nonnegative integers")
    return (left ^ right).bit_count()


class DuplicateGuard:
    """Track nonduplicate hashes in separate, bounded front/back histories.

    check_and_add returns True for matches within max_distance of a stored hash;
    duplicates are not inserted. Only accepted hashes displace old history entries.
    """

    def __init__(self, max_distance: int = 5, history_size: int = 50, hash_size: int = 8) -> None:
        for name, value in (("max_distance", max_distance), ("history_size", history_size),
                            ("hash_size", hash_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if history_size < 1 or hash_size < 1 or max_distance > hash_size * hash_size:
            raise ValueError("history_size and hash_size must be positive; max_distance must fit hash")
        self.max_distance = max_distance
        self.history_size = history_size
        self.hash_size = hash_size
        self._histories: dict[str, deque[int]] = {
            "front": deque(maxlen=history_size),
            "back": deque(maxlen=history_size),
        }

    def history(self, label: str) -> tuple[int, ...]:
        """Return a read-only snapshot, oldest first."""
        return tuple(self._history(label))

    def _history(self, label: str) -> deque[int]:
        if label not in self._histories:
            raise ValueError("label must be 'front' or 'back'")
        return self._histories[label]

    def check_and_add_hash(self, value: int, label: str) -> bool:
        """Return duplicate status; store hash only when it is not a duplicate."""
        history = self._history(label)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << (self.hash_size ** 2)):
            raise ValueError("hash must fit the configured hash_size")
        if any(hamming_distance(value, old) <= self.max_distance for old in history):
            return True
        history.append(value)
        return False

    def check_and_add(self, frame: np.ndarray, label: str) -> bool:
        """Hash a frame and return whether it matches a recent hash for its side."""
        return self.check_and_add_hash(dhash(frame, self.hash_size), label)
