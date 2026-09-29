import numpy as np
import pytest

from yolo_conveyor.capture import DuplicateGuard, dhash, hamming_distance


def test_dhash_deterministic_and_explicit_bit_order():
    descending = np.tile(np.arange(9, dtype=np.uint8)[::-1], (8, 1))
    flat = np.zeros((8, 9), dtype=np.uint8)
    assert dhash(descending) == (1 << 64) - 1
    assert dhash(descending) == dhash(np.repeat(descending[:, :, None], 3, axis=2))
    assert dhash(flat) == 0
    assert hamming_distance(dhash(descending), dhash(descending)) == 0
    assert hamming_distance(0, 0b10101) == 3
    assert hamming_distance(0b10101, 0) == 3


def test_threshold_is_inclusive_and_frame_path():
    guard = DuplicateGuard(max_distance=1)
    assert not guard.check_and_add_hash(0, "front")
    assert guard.check_and_add_hash(1, "front")
    assert guard.history("front") == (0,)  # duplicates do not displace history
    assert not guard.check_and_add_hash(3, "front")
    frame = np.zeros((8, 9), dtype=np.uint8)
    assert guard.check_and_add(frame, "front")
    assert guard.history("front") == (0, 3)


def test_bounded_eviction_and_isolated_sides():
    guard = DuplicateGuard(max_distance=0, history_size=2)
    assert not guard.check_and_add_hash(0, "front")
    assert not guard.check_and_add_hash(0, "back")
    assert not guard.check_and_add_hash(1, "front")
    assert not guard.check_and_add_hash(2, "front")
    assert guard.history("front") == (1, 2)
    assert guard.history("back") == (0,)
    assert not guard.check_and_add_hash(0, "front")
    assert guard.check_and_add_hash(0, "back")
    assert guard.history("front") == (2, 0)
    assert guard.history("back") == (0,)


def test_invalid_guard_inputs():
    guard = DuplicateGuard()
    with pytest.raises(ValueError):
        guard.check_and_add_hash(0, "other")
    with pytest.raises(ValueError):
        guard.check_and_add_hash(-1, "front")
    with pytest.raises(ValueError):
        guard.check_and_add_hash(1 << 64, "front")
    with pytest.raises(ValueError):
        DuplicateGuard(history_size=0)
    with pytest.raises(ValueError):
        dhash(np.zeros((8, 9), dtype=np.uint8), hash_size=0)
