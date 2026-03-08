"""Parse curriculum markdown files into chunks with metadata."""

import re
from pathlib import Path

CURRICULUM_DIR = Path(__file__).parent.parent.parent.parent.parent / "curriculum"

# Files worth indexing per week (skip rubric / teacher-guide for student-facing RAG)
INDEXABLE_FILES = ["lecture.md", "slides.md", "assignment.md"]


def split_by_heading(text: str, max_chunk: int = 1000) -> list[str]:
    """Split markdown text by ## headings, further split if chunk too long."""
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chunk:
            chunks.append(section)
        else:
            # Split long sections by paragraphs
            paragraphs = section.split("\n\n")
            current = ""
            for para in paragraphs:
                if len(current) + len(para) + 2 > max_chunk and current:
                    chunks.append(current.strip())
                    current = para
                else:
                    current = current + "\n\n" + para if current else para
            if current.strip():
                chunks.append(current.strip())
    return chunks


def load_curriculum_chunks() -> list[dict]:
    """Load all curriculum files and return chunks with metadata."""
    all_chunks = []

    for week_dir in sorted(CURRICULUM_DIR.glob("week-*")):
        week_match = re.search(r"week-(\d+)", week_dir.name)
        if not week_match:
            continue
        week_num = int(week_match.group(1))

        for filename in INDEXABLE_FILES:
            filepath = week_dir / filename
            if not filepath.exists():
                continue
            text = filepath.read_text(encoding="utf-8")
            file_type = filename.replace(".md", "")
            chunks = split_by_heading(text)

            for i, chunk in enumerate(chunks):
                # Extract first heading as title
                title_match = re.match(r"^#+ (.+)", chunk)
                title = title_match.group(1).strip() if title_match else f"第{week_num}週 {file_type} 段落{i+1}"

                all_chunks.append({
                    "id": f"w{week_num:02d}_{file_type}_{i:03d}",
                    "content": chunk,
                    "metadata": {
                        "week": week_num,
                        "file_type": file_type,
                        "title": title,
                        "source": str(filepath.relative_to(CURRICULUM_DIR)),
                    },
                })

    # Also index the syllabus
    syllabus_path = CURRICULUM_DIR / "syllabus.md"
    if syllabus_path.exists():
        text = syllabus_path.read_text(encoding="utf-8")
        chunks = split_by_heading(text)
        for i, chunk in enumerate(chunks):
            title_match = re.match(r"^#+ (.+)", chunk)
            title = title_match.group(1).strip() if title_match else f"課程大綱 段落{i+1}"
            all_chunks.append({
                "id": f"syllabus_{i:03d}",
                "content": chunk,
                "metadata": {
                    "week": 0,
                    "file_type": "syllabus",
                    "title": title,
                    "source": "syllabus.md",
                },
            })

    return all_chunks
