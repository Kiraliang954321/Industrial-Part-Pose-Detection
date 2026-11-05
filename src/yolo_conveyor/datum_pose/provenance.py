"""Independent A2 inventory, retaining immutable A1 and robot dependencies."""
from pathlib import Path
import shutil
from yolo_conveyor.annular.provenance import sources as a1_sources
from yolo_conveyor.orientation.common import sha256


def sources():
    root=Path(__file__).resolve().parents[3];result=a1_sources()
    # Held camera/triangulation helpers are explicit dependencies, outside the A1 inventory.
    for p in [*Path(__file__).parent.glob('*.py'),*(root/'scripts').glob('*datum_pose*.py'),
              root/'src/yolo_conveyor/held/scene.py',root/'src/yolo_conveyor/held/geometry.py']:
        result[p.relative_to(root).as_posix()]=sha256(p)
    return result


def snapshot(out):
    root=Path(__file__).resolve().parents[3];hashes=sources()
    for relative,digest in hashes.items():
        p=Path(out)/'source'/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/relative,p)
        if sha256(p)!=digest:raise ValueError('source changed during snapshot')
    return hashes
