"""Independent H1 source inventory; adding H1 does not change M4/L1 source sets."""
from pathlib import Path
import shutil
from yolo_conveyor.orientation.common import source_hashes,sha256


def held_sources():
    root=Path(__file__).resolve().parents[3];result=source_hashes()
    for p in [*Path(__file__).parent.glob("*.py"),*(root/"scripts").glob("*held_inspection*.py")]:
        result[p.relative_to(root).as_posix()]=sha256(p)
    return result


def snapshot_sources(output):
    root=Path(__file__).resolve().parents[3];hashes=held_sources()
    for relative,expected in hashes.items():
        target=Path(output)/"source"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/relative,target)
        if sha256(target)!=expected:raise ValueError("source changed during snapshot")
    return hashes
