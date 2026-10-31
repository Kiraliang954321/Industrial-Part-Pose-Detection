"""Fine-tune the held-view model without modifying the frozen M3 training route."""
from pathlib import Path
import os,sys
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from yolo_conveyor.held.training import main  # noqa: E402
if __name__=='__main__':raise SystemExit(main())
