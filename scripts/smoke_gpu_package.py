"""Parent owns wheel extraction so Windows DLLs are unlocked before cleanup."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('package',type=Path)
parser.add_argument('--blender',default='C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')
args=parser.parse_args()
scratch=ROOT/'.superpowers'; scratch.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='gpu-package-smoke-',dir=scratch) as directory:
    wheels=list((args.package/'wheels').glob('*.whl'))
    if not wheels: raise RuntimeError('GPU package contains no wheels')
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive: archive.extractall(directory)
    result=subprocess.run([args.blender,'--background','--factory-startup','--python-exit-code','1',
        '--python',str(ROOT/'scripts'/'smoke_test_blender.py'),'--','--package',str(args.package.resolve()),
        '--gpu','--wheel-env',directory],cwd=ROOT)
raise SystemExit(result.returncode)
