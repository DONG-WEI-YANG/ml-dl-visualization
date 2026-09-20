import pytest

from scripts.stage_hf_deployment import stage_deployment
from scripts.start_persistent import validate_storage


def test_stage_complete_tree_without_nested_private_files(tmp_path):
    repo = tmp_path / 'repo'
    backend = repo / 'platform/backend'
    (backend / 'hf-space').mkdir(parents=True)
    (backend / 'app/nested').mkdir(parents=True)
    (backend / 'data').mkdir()
    (repo / 'curriculum/week-01').mkdir(parents=True)
    (backend / 'Dockerfile').write_text('wrong generic Dockerfile')
    (backend / 'README.md').write_text('generic readme')
    (backend / 'hf-space/Dockerfile').write_text('correct persistent HF Dockerfile')
    (backend / 'hf-space/README.md').write_text('HF metadata')
    (backend / 'app/main.py').write_text('code')
    for name in ['.env', '.env.production', 'private.db', 'private.db-wal', 'private.sqlite3', 'secret.key']:
        (backend / 'app/nested' / name).write_text('private')
    (backend / 'data/passwords.csv').write_text('private')
    (repo / 'curriculum/week-01/lecture.md').write_text('lecture')
    staged = tmp_path / 'staged'
    stage_deployment(repo, staged)
    assert (staged / 'Dockerfile').read_text() == 'correct persistent HF Dockerfile'
    assert (staged / 'README.md').read_text() == 'HF metadata'
    assert (staged / 'curriculum/week-01/lecture.md').read_text() == 'lecture'
    assert not list((staged / 'app/nested').iterdir())
    assert not (staged / 'data').exists()


def test_persistent_guard_rejects_missing_mount(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.start_persistent.os.path.ismount', lambda _: False)
    with pytest.raises(RuntimeError, match='mount required'):
        validate_storage(tmp_path / 'app.db', tmp_path)


def test_persistent_guard_rejects_effective_path_outside_mount(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.start_persistent.os.path.ismount', lambda _: True)
    mount = tmp_path / 'data'
    with pytest.raises(RuntimeError, match='inside persistent'):
        validate_storage(tmp_path / 'ephemeral/app.db', mount)
    with pytest.raises(RuntimeError, match='inside persistent'):
        validate_storage(mount, mount)
    validate_storage(mount / 'app.db', mount)
