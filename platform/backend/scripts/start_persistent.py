"""HF entry point: refuse initialization outside a durable data mount."""
import os
from pathlib import Path
import sys


def validate_storage(database_path: Path, storage_root: Path = Path('/data')) -> None:
    root = storage_root.resolve()
    if not os.path.ismount(root):
        raise RuntimeError('Persistent /data mount required; see storage operations runbook')
    target = database_path.resolve()
    if not target.is_relative_to(root) or target == root:
        raise RuntimeError('DATABASE_PATH must be a file inside persistent /data')


if __name__ == '__main__':
    # Make application imports independent of invocation from scripts/.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.config import settings
    validate_storage(settings.database_path)
    os.execvp('uvicorn', ['uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '7860'])
