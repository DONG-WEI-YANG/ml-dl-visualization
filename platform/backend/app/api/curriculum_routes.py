from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/curriculum", tags=["curriculum"])

CURRICULUM_DIR = Path(__file__).resolve().parents[4] / "curriculum"

FILE_MAP = {
    "lecture": ("lecture.md", "text/markdown"),
    "slides": ("slides.md", "text/markdown"),
    "notebook": ("notebook.ipynb", "application/x-ipynb+json"),
    "assignment": ("assignment.md", "text/markdown"),
    "rubric": ("rubric.md", "text/markdown"),
    "teacher-guide": ("teacher-guide.md", "text/markdown"),
}


@router.get("/week/{week_id}/{file_type}")
async def download_curriculum_file(week_id: int, file_type: str):
    if week_id < 1 or week_id > 18:
        raise HTTPException(404, "Invalid week")
    if file_type not in FILE_MAP:
        raise HTTPException(404, f"Unknown file type: {file_type}")

    filename, media_type = FILE_MAP[file_type]
    week_dir = CURRICULUM_DIR / f"week-{week_id:02d}"
    file_path = week_dir / filename

    if not file_path.exists():
        raise HTTPException(404, f"File not found: week-{week_id:02d}/{filename}")

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=f"week-{week_id:02d}-{filename}",
    )


@router.get("/week/{week_id}")
async def list_curriculum_files(week_id: int):
    if week_id < 1 or week_id > 18:
        raise HTTPException(404, "Invalid week")

    week_dir = CURRICULUM_DIR / f"week-{week_id:02d}"
    available = {}
    for key, (filename, _) in FILE_MAP.items():
        path = week_dir / filename
        available[key] = {
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
            "url": f"/api/curriculum/week/{week_id}/{key}",
        }
    return {"week": week_id, "files": available}
