"""Downloaded lecture HTML must be self-contained.

The download route sends HTML as an attachment, so students open it from
disk (file://). Root-relative image URLs like /api/curriculum/... resolve to
the local drive root there and the figures silently disappear.
"""
import base64
import re

from app.api import curriculum_routes
from app.api.curriculum_routes import _md_to_html

# Smallest valid PNG (1x1 transparent pixel)
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _setup_week(tmp_path, monkeypatch, md: str):
    week_dir = tmp_path / "week-05"
    (week_dir / "figures").mkdir(parents=True)
    (week_dir / "figures" / "roc.png").write_bytes(PNG_1X1)
    (week_dir / "lecture.md").write_text(md, encoding="utf-8")
    monkeypatch.setattr(curriculum_routes, "CURRICULUM_DIR", tmp_path)
    return week_dir / "lecture.md"


def test_relative_figure_is_embedded_as_data_uri(tmp_path, monkeypatch):
    md_path = _setup_week(tmp_path, monkeypatch, "![ROC 曲線](figures/roc.png)\n")
    out = _md_to_html(md_path, "t", week_id=5)

    srcs = re.findall(r'<img src="([^"]+)"', out)
    assert len(srcs) == 1
    assert srcs[0].startswith("data:image/png;base64,")
    assert base64.b64decode(srcs[0].split(",", 1)[1]) == PNG_1X1
    assert "/api/curriculum/" not in out


def test_missing_figure_falls_back_to_api_url(tmp_path, monkeypatch):
    md_path = _setup_week(tmp_path, monkeypatch, "![x](figures/nope.png)\n")
    out = _md_to_html(md_path, "t", week_id=5)
    assert 'src="/api/curriculum/week/5/assets/figures/nope.png"' in out


def test_path_traversal_is_not_embedded(tmp_path, monkeypatch):
    (tmp_path / "secret.png").write_bytes(PNG_1X1)
    md_path = _setup_week(tmp_path, monkeypatch, "![x](../secret.png)\n")
    out = _md_to_html(md_path, "t", week_id=5)
    assert "data:image" not in out
