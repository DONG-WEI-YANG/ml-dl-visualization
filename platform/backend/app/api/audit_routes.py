"""Admin audit log query and CSV export."""
import csv
import io

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.auth.dependencies import require_admin
from app.db import get_db

router = APIRouter(prefix="/api/admin", tags=["Audit"])

_COLUMNS = ["id", "timestamp", "actor_id", "actor_username", "actor_role",
            "action", "target_type", "target_id", "detail", "ip"]


def _build_where(actor_id, action, action_prefix, date_from, date_to):
    conditions, params = [], []
    if actor_id is not None:
        conditions.append("actor_id = ?")
        params.append(actor_id)
    if action:
        conditions.append("action = ?")
        params.append(action)
    if action_prefix:
        conditions.append("action LIKE ?")
        params.append(action_prefix + "%")
    if date_from:
        conditions.append("timestamp >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("timestamp <= ?")
        params.append(date_to)
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    return where, params


@router.get("/audit-logs")
async def list_audit_logs(
    actor_id: int | None = Query(None),
    action: str | None = Query(None),
    action_prefix: str | None = Query(None),
    date_from: str | None = Query(None, alias="from"),
    date_to: str | None = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    admin: dict = Depends(require_admin),
):
    where, params = _build_where(actor_id, action, action_prefix, date_from, date_to)
    conn = get_db()
    total = conn.execute(f"SELECT COUNT(*) AS c FROM audit_logs{where}", params).fetchone()["c"]
    rows = conn.execute(
        f"SELECT * FROM audit_logs{where} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [page_size, (page - 1) * page_size],
    ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows], "total": total, "page": page, "page_size": page_size}


def _safe_cell(v):
    """Prefix values that Excel/Sheets would interpret as a formula so a
    malicious value (e.g. attacker-controlled username) can't execute as one."""
    s = "" if v is None else str(v)
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        s = "'" + s
    return s


@router.get("/audit-logs/export")
async def export_audit_logs(
    actor_id: int | None = Query(None),
    action: str | None = Query(None),
    action_prefix: str | None = Query(None),
    date_from: str | None = Query(None, alias="from"),
    date_to: str | None = Query(None, alias="to"),
    admin: dict = Depends(require_admin),
):
    where, params = _build_where(actor_id, action, action_prefix, date_from, date_to)
    conn = get_db()
    rows = conn.execute(
        f"SELECT * FROM audit_logs{where} ORDER BY id DESC LIMIT 10000", params
    ).fetchall()
    conn.close()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_COLUMNS)
    for r in rows:
        writer.writerow([_safe_cell(r[c]) for c in _COLUMNS])
    # UTF-8 BOM so Excel opens Chinese content correctly.
    data = "﻿" + buf.getvalue()
    return StreamingResponse(
        iter([data]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=audit-logs.csv"},
    )
