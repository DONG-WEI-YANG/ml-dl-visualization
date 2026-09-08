from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.analytics.models import LearningEvent
from app.analytics.tracker import record_event, get_student_analytics, get_class_summary, get_roster
from app.auth.dependencies import get_current_user
from app.db import db_connection

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.post("/events")
async def create_event(event: LearningEvent, user: dict = Depends(get_current_user)):
    bound_event = event.model_copy(update={"student_id": str(user["id"])})
    event_id = record_event(bound_event)
    return {"id": event_id, "status": "recorded", "student_id": bound_event.student_id}


def _can_read_student(user: dict, student_id: str) -> bool:
    if user["role"] == "admin":
        return True
    if user["role"] == "student":
        return str(user["id"]) == student_id
    if user["role"] != "teacher" or not student_id.isdigit():
        return False
    with db_connection() as conn:
        assigned = conn.execute(
            "SELECT 1 FROM teacher_students WHERE teacher_id = ? AND student_id = ?",
            (user["id"], int(student_id)),
        ).fetchone()
    return assigned is not None


@router.get("/students/{student_id}")
async def student_analytics(student_id: str, semester: str | None = Query(None), user: dict = Depends(get_current_user)):
    if not _can_read_student(user, student_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="無權查看此學生資料")
    return get_student_analytics(student_id, semester)


@router.get("/summary")
async def class_summary(
    semester: str | None = Query(None),
    class_name: str | None = Query(None),
    academic_year: str | None = Query(None),
    user: dict = Depends(get_current_user),
):
    if user["role"] not in ("teacher", "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要教師或管理員權限")
    teacher_id = user["id"] if user["role"] == "teacher" else None
    return get_class_summary(semester, teacher_id=teacher_id, class_name=class_name, academic_year=academic_year)


@router.get('/roster')
async def roster(semester: str | None = Query(None), class_name: str | None = Query(None),
                 academic_year: str | None = Query(None), user: dict = Depends(get_current_user)):
    if user['role'] not in ('teacher', 'admin'):
        raise HTTPException(status_code=403, detail='需要教師或管理員權限')
    return get_roster(semester, user['id'] if user['role'] == 'teacher' else None, class_name, academic_year)
