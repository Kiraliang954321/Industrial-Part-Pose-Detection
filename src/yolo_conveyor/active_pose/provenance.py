from pathlib import Path
import shutil
from yolo_conveyor.pose_cell.provenance import sources as baseline_sources
from yolo_conveyor.orientation.common import sha256


def sources():
    root=Path(__file__).resolve().parents[3];result=baseline_sources()
    for p in [*Path(__file__).parent.glob('*.py'),*(root/'scripts').glob('*active_pose*.py')]:
        result[p.relative_to(root).as_posix()]=sha256(p)
    return result


def snapshot(out):
    root=Path(__file__).resolve().parents[3];hashes=sources()
    for rel,h in hashes.items():
        p=Path(out)/'source'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,p)
        if sha256(p)!=h:raise ValueError('source snapshot drift')
    return hashes
