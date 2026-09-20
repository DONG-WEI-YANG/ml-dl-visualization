"""Consistent SQLite backup or restore to a NEW path, never overwrite live data."""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3


def copy_database(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if source == destination:
        raise ValueError('Source and destination must differ')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of existing databases.
    with destination.open('xb'):
        pass
    try:
        with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as src:
            with closing(sqlite3.connect(destination)) as dst:
                src.backup(dst)
                if dst.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                    raise RuntimeError('Backup integrity check failed')
                if dst.execute('PRAGMA foreign_key_check').fetchall():
                    raise RuntimeError('Backup foreign-key check failed')
    except Exception:
        destination.unlink(missing_ok=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    copy_database(args.source, args.destination)
    print('Copy complete; SQLite integrity and foreign keys verified.')
