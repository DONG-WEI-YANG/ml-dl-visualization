from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from app.analytics.models import LearningEvent
from app.analytics.tracker import record_event, get_student_analytics, get_class_summary, get_roster
from app.auth.dependencies import get_current_user, require_teacher_or_admin
from app.db import db_connection
from app.audit import log_audit

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.post("/events")
async def create_event(event: LearningEvent, user: dict = Depends(get_current_user)):
    if event.event_type != 'viz_interaction':
        raise HTTPException(status_code=403, detail='測驗、作業評分與助教紀錄由伺服器產生')
    if event.score is not None or event.timestamp is not None:
        raise HTTPException(status_code=422, detail='互動事件不可包含成績或自行指定時間')
    bound_event = event.model_copy(update={"student_id": str(user["id"])})
    event_id = record_event(bound_event)
    return {"id": event_id, "status": "recorded", "student_id": bound_event.student_id}


class AssignmentGrade(BaseModel):
    student_id: str = Field(pattern=r'^\d+$', max_length=20)
    week: int = Field(ge=1, le=18)
    score: float = Field(ge=0, le=100, allow_inf_nan=False)

    @field_validator('student_id')
    @classmethod
    def canonical_student_id(cls, value):
        number = int(value)
        if not 1 <= number <= 9223372036854775807:
            raise ValueError('Invalid student ID')
        return str(number)


@router.post('/assignments/grade')
async def grade_assignment(grade: AssignmentGrade, request: Request,
                           user: dict = Depends(require_teacher_or_admin)):
    if not _can_read_student(user, grade.student_id):
        raise HTTPException(status_code=403, detail='只能評分已指派的學生')
    with db_connection() as conn:
        target = conn.execute("SELECT id FROM users WHERE id = ? AND role = 'student' "
                              "AND is_active = 1 AND deleted_at IS NULL", (grade.student_id,)).fetchone()
    if not target:
        raise HTTPException(status_code=404, detail='學生不存在或已停用')
    event_id = record_event(LearningEvent(student_id=grade.student_id, week=grade.week,
        event_type='assignment', score=grade.score,
        metadata={'graded_by': user['id'], 'source': 'teacher_grading'}))
    log_audit('assignment.grade', actor=user, target_type='learning_event', target_id=event_id,
              detail=grade.model_dump(), ip=request.client.host if request.client else '')
    return {'id': event_id, 'status': 'recorded', 'student_id': grade.student_id}


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
