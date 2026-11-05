"""Source inventory for L1 without altering M4's frozen source set."""
from pathlib import Path
from yolo_conveyor.orientation.common import source_hashes, sha256


def line_sources():
    root = Path(__file__).resolve().parents[3]
    result = source_hashes()
    files = list(Path(__file__).parent.glob("*.py"))
    files += list((root / "scripts").glob("*indexed_line*.py"))
    for path in files:
        result[path.relative_to(root).as_posix()] = sha256(path)
    return result
