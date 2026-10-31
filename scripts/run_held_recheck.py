"""Run H2 initial inspection and pre-insertion relative-pose recheck."""
import os,sys
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from yolo_conveyor.held_recheck.runtime import main
if __name__=='__main__':raise SystemExit(main())
