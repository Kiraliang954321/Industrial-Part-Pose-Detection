"""Fit and verify H2 residual calibration without changing frozen H1."""
import os,sys
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from yolo_conveyor.held_recheck.development import main
if __name__=='__main__':raise SystemExit(main())
