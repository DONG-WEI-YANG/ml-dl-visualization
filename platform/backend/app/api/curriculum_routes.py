from pathlib import Path
from urllib.parse import quote
import html
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter(prefix="/api/curriculum", tags=["curriculum"])

import os

# In local dev: parents[4] = ml-dl-visualization root
# In Docker container (/app/app/api/...): parents list is too short
# Use env var override or try to resolve gracefully
_CURRICULUM_ENV = os.environ.get("CURRICULUM_DIR")
if _CURRICULUM_ENV:
    CURRICULUM_DIR = Path(_CURRICULUM_ENV)
else:
    try:
        CURRICULUM_DIR = Path(__file__).resolve().parents[4] / "curriculum"
    except IndexError:
        CURRICULUM_DIR = Path("/app/curriculum")  # fallback in container

# Primary format, fallback format
FILE_MAP = {
    "lecture": {
        "formats": [("lecture.pdf", "application/pdf"), ("lecture.md", "text/markdown")],
        "label": "講義",
    },
    "slides": {
        "formats": [("slides.pdf", "application/pdf"), ("slides.md", "text/markdown")],
        "label": "投影片",
    },
    "notebook": {
        "formats": [("notebook.ipynb", "application/x-ipynb+json")],
        "label": "Notebook",
    },
    "assignment": {
        "formats": [("assignment.pdf", "application/pdf"), ("assignment.md", "text/markdown")],
        "label": "作業",
    },
    "rubric": {
        "formats": [("rubric.md", "text/markdown")],
        "label": "評分標準",
    },
    "teacher-guide": {
        "formats": [("teacher-guide.md", "text/markdown")],
        "label": "教師手冊",
    },
}


def _inline(text: str) -> str:
    """Apply inline markdown (bold, inline code) to a text fragment."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'`(.+?)`', r'<code class="inline">\1</code>', text)
    return text


def _parse_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip('|').split('|')]


def _is_separator_row(line: str) -> bool:
    s = line.strip()
    if not (s.startswith('|') and s.endswith('|')):
        return False
    cells = _parse_row(line)
    return len(cells) >= 1 and all(re.match(r'^:?-+:?$', c) for c in cells)


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return len(s) > 1 and s.startswith('|') and s.endswith('|')


def _md_to_html(md_path: Path, title: str) -> str:
    """Convert markdown to a styled printable HTML page."""
    content = md_path.read_text(encoding="utf-8")
    escaped = html.escape(content)
    lines = escaped.split("\n")
    html_lines: list[str] = []
    in_code = False
    in_table = False
    pending_header: list[str] | None = None

    def flush_pending_header() -> None:
        nonlocal pending_header
        if pending_header is not None:
            html_lines.append(f"<p>| {' | '.join(pending_header)} |</p>")
            pending_header = None

    def close_table() -> None:
        nonlocal in_table
        if in_table:
            html_lines.append("</tbody></table>")
            in_table = False

    for line in lines:
        # Code fence toggling
        if line.startswith("```"):
            close_table()
            flush_pending_header()
            if in_code:
                html_lines.append("</code></pre>")
                in_code = False
            else:
                html_lines.append("<pre><code>")
                in_code = True
            continue
        if in_code:
            html_lines.append(line)
            continue

        # Table state machine
        if pending_header is not None:
            if _is_separator_row(line):
                html_lines.append('<table class="md-table">')
                header_cells = "".join(f"<th>{_inline(c)}</th>" for c in pending_header)
                html_lines.append(f"<thead><tr>{header_cells}</tr></thead><tbody>")
                in_table = True
                pending_header = None
                continue
            if line.strip() == "":
                continue  # tolerate blank line between header and separator
            flush_pending_header()  # was not a real table; fall through

        if in_table:
            if _is_table_row(line):
                cells = _parse_row(line)
                row = "".join(f"<td>{_inline(c)}</td>" for c in cells)
                html_lines.append(f"<tr>{row}</tr>")
                continue
            if line.strip() == "":
                continue  # tolerate blank lines inside a table
            close_table()  # non-table line ends the table; fall through

        if _is_table_row(line):
            pending_header = _parse_row(line)
            continue

        if line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{_inline(line[2:])}</li>")
        elif line.strip() == "":
            html_lines.append("<br/>")
        else:
            html_lines.append(f"<p>{_inline(line)}</p>")

    close_table()
    flush_pending_header()
    if in_code:
        html_lines.append("</code></pre>")

    body = "\n".join(html_lines)
    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: "Microsoft JhengHei", "Noto Sans TC", sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.8; color: #333; }}
  h1 {{ color: #1e40af; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }}
  h2 {{ color: #1e3a5f; margin-top: 24px; }}
  h3 {{ color: #374151; }}
  pre {{ background: #f8f9fa; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; overflow-x: auto; }}
  code {{ font-family: "Consolas", monospace; font-size: 14px; }}
  code.inline {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-size: 13px; }}
  li {{ margin-left: 20px; }}
  table.md-table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }}
  table.md-table th, table.md-table td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; vertical-align: top; }}
  table.md-table thead {{ background: #eff6ff; }}
  table.md-table thead th {{ color: #1e3a5f; font-weight: 600; }}
  table.md-table tbody tr:nth-child(even) {{ background: #fafafa; }}
  mjx-container {{ overflow-x: auto; overflow-y: hidden; }}
  @media print {{ body {{ margin: 20px; }} }}
</style>
<script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
      displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
      processEscapes: true
    }},
    options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }}
  }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
</head>
<body>{body}</body>
</html>"""


@router.get("/week/{week_id}/{file_type}")
async def download_curriculum_file(week_id: int, file_type: str):
    if week_id < 1 or week_id > 18:
        raise HTTPException(404, "Invalid week")
    if file_type not in FILE_MAP:
        raise HTTPException(404, f"Unknown file type: {file_type}")

    entry = FILE_MAP[file_type]
    week_dir = CURRICULUM_DIR / f"week-{week_id:02d}"

    # Try each format in priority order
    for filename, media_type in entry["formats"]:
        file_path = week_dir / filename
        if file_path.exists():
            if media_type == "text/markdown" and file_type in ("lecture", "slides", "assignment"):
                # Convert markdown to styled HTML for better printability
                title = f"第 {week_id} 週 - {entry['label']}"
                html_content = _md_to_html(file_path, title)
                ascii_name = f"week-{week_id:02d}-{file_type}.html"
                utf8_name = quote(f"week-{week_id:02d}-{entry['label']}.html")
                return HTMLResponse(
                    content=html_content,
                    headers={
                        "Content-Disposition": (
                            f'attachment; filename="{ascii_name}"; '
                            f"filename*=UTF-8''{utf8_name}"
                        )
                    },
                )
            return FileResponse(
                path=file_path,
                media_type=media_type,
                filename=f"week-{week_id:02d}-{filename}",
            )

    raise HTTPException(404, f"File not found for week {week_id} {file_type}")


@router.get("/week/{week_id}")
async def list_curriculum_files(week_id: int):
    if week_id < 1 or week_id > 18:
        raise HTTPException(404, "Invalid week")

    week_dir = CURRICULUM_DIR / f"week-{week_id:02d}"
    available = {}
    for key, entry in FILE_MAP.items():
        found = False
        actual_format = None
        size = 0
        for filename, media_type in entry["formats"]:
            path = week_dir / filename
            if path.exists():
                found = True
                actual_format = media_type
                size = path.stat().st_size
                break
        available[key] = {
            "exists": found,
            "size": size,
            "format": actual_format,
            "label": entry["label"],
            "url": f"/api/curriculum/week/{week_id}/{key}",
        }
    return {"week": week_id, "files": available}
