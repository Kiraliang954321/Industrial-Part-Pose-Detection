"""A1 source snapshot includes frozen robot helpers, but does not alter prior inventories."""
from pathlib import Path
import shutil
from yolo_conveyor.orientation.common import source_hashes,sha256


def sources():
    root=Path(__file__).resolve().parents[3];result=source_hashes()
    for path in [*Path(__file__).parent.glob('*.py'),*(root/'scripts').glob('*annular*.py')]:result[path.relative_to(root).as_posix()]=sha256(path)
    return result


def snapshot(output):
    root=Path(__file__).resolve().parents[3];hashes=sources()
    for relative,digest in hashes.items():
        target=Path(output)/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/relative,target)
        if sha256(target)!=digest:raise ValueError('source changed during snapshot')
    return hashes
