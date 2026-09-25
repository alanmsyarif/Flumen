"""Stage an extension from canonical sources without modifying an existing directory."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def build_extension(stage_dir: Path) -> Path:
    stage_dir = Path(stage_dir).resolve()
    stage_dir.mkdir(parents=True, exist_ok=False)
    for path in sorted((ROOT / 'flumen').glob('*.py')):
        shutil.copy2(path, stage_dir / path.name)
    shutil.copy2(ROOT / 'flumen' / 'blender_manifest.toml', stage_dir)
    shutil.copy2(ROOT / 'LICENSE', stage_dir)
    return stage_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage_dir', type=Path)
    print(build_extension(parser.parse_args().stage_dir))
