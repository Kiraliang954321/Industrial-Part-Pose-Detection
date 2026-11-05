"""Run the explicitly scoped L1 single-piece indexing cell simulation."""
from pathlib import Path
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from yolo_conveyor.line.runner import main  # noqa: E402

if __name__=="__main__":raise SystemExit(main())
