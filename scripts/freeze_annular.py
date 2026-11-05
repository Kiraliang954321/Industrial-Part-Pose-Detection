"""Freeze A1 bearing/gear development inputs and independent job assignment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from yolo_conveyor.annular.protocol import main
if __name__=='__main__':raise SystemExit(main())
