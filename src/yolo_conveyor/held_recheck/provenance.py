"""H2 inventory includes immutable H1 dependencies and its own modules only."""
from pathlib import Path
import shutil
from yolo_conveyor.held.provenance import held_sources
from yolo_conveyor.orientation.common import sha256


def sources():
    root = Path(__file__).resolve().parents[3]
    result = held_sources()
    for path in [*Path(__file__).parent.glob('*.py'), *(root/'scripts').glob('*held_recheck*.py')]:
        result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def snapshot(output):
    root = Path(__file__).resolve().parents[3]
    hashes = sources()
    for relative, expected in hashes.items():
        target = Path(output)/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/relative, target)
        if sha256(target) != expected:
            raise ValueError('source changed during snapshot')
    return hashes
