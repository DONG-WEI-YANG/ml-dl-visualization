from fastapi import APIRouter, Query
from app.analytics.models import LearningEvent
from app.analytics.tracker import record_event, get_student_analytics, get_class_summary

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.post("/events")
async def create_event(event: LearningEvent):
    event_id = record_event(event)
    return {"id": event_id, "status": "recorded"}


@router.get("/students/{student_id}")
async def student_analytics(student_id: str):
    return get_student_analytics(student_id)


@router.get("/summary")
async def class_summary(semester: str | None = Query(None)):
    return get_class_summary(semester)
