"""Stage an extension from canonical sources without modifying an existing directory."""
import argparse
from pathlib import Path
import shutil
import hashlib

ROOT = Path(__file__).resolve().parents[1]


WHEEL_NAME = 'warp_lang-1.17.0-py3-none-win_amd64.whl'
WHEEL_HASH = 'ca35c82242a7553046f09023bbe56750b5c3d9af72625bd1a905a56d3189771d'


def build_extension(stage_dir: Path, wheel_dir: Path | None = None) -> Path:
    wheel = None
    if wheel_dir is not None:
        wheel = Path(wheel_dir) / WHEEL_NAME
        with wheel.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != WHEEL_HASH:
            raise ValueError('Warp wheel hash does not match requirements-gpu.txt')
    stage_dir = Path(stage_dir).resolve()
    stage_dir.mkdir(parents=True, exist_ok=False)
    for path in sorted((ROOT / 'flumen').rglob('*.py')):
        target = stage_dir / path.relative_to(ROOT / 'flumen')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(ROOT / 'flumen' / 'blender_manifest.toml', stage_dir)
    shutil.copy2(ROOT / 'LICENSE', stage_dir)
    shutil.copy2(ROOT / 'THIRD_PARTY_NOTICES.md', stage_dir)
    if wheel is not None:
        (stage_dir / 'wheels').mkdir()
        shutil.copy2(wheel, stage_dir / 'wheels' / wheel.name)
        manifest = stage_dir / 'blender_manifest.toml'
        with manifest.open('a', encoding='utf8') as stream:
            stream.write(f'\nplatforms = ["windows-x64"]\nwheels = ["./wheels/{wheel.name}"]\n')
    return stage_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage_dir', type=Path)
    parser.add_argument('--wheel-dir', type=Path)
    args = parser.parse_args()
    print(build_extension(args.stage_dir, args.wheel_dir))
