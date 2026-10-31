"""Generate physically held, two-camera development data with grouped splits."""
from pathlib import Path
import os,sys
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from yolo_conveyor.held.data import main  # noqa: E402
if __name__=='__main__':raise SystemExit(main())
