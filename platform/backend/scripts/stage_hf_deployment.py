"""Prepare one complete HF deployment tree without local secrets or databases."""
from pathlib import Path
import shutil


def stage_deployment(repo: Path, destination: Path) -> None:
    backend = repo / 'platform' / 'backend'
    def ignore(directory, names):
        return [name for name in names if (
            (Path(directory) / name).is_symlink()
            or name in {'.venv', '__pycache__', '.pytest_cache', '.git', 'hf-space', 'backups', 'data', 'tests'}
            or name.startswith('.venv') or name == '.benchmarks'
            or name.startswith('.env') or name.startswith('.coverage')
            or name.endswith(('.pyc', '.pkl', '.pem', '.key'))
            or '.db' in name or '.sqlite' in name
        )]
    shutil.copytree(backend, destination, ignore=ignore)
    shutil.copytree(repo / 'curriculum', destination / 'curriculum', ignore=ignore)
    shutil.copyfile(backend / 'hf-space' / 'Dockerfile', destination / 'Dockerfile')
    shutil.copyfile(backend / 'hf-space' / 'README.md', destination / 'README.md')
