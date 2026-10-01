"""120 warmup and 600 measured completed draws; run without --background."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.gpu-deps'),str(ROOT/'examples'),str(ROOT/'scripts')]
from connected_viewport import run
JOB=run(benchmark=True)
