"""Audit logging helper — explicit calls from routes; failures never break the caller."""
import json
import logging

from app.db import get_db

logger = logging.getLogger(__name__)


def log_audit(
    action: str,
    *,
    actor: dict | None = None,
    target_type: str = "",
    target_id: str | int = "",
    detail: dict | None = None,
    ip: str = "",
) -> None:
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO audit_logs (actor_id, actor_username, actor_role, action, "
            "target_type, target_id, detail, ip) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                actor["id"] if actor else None,
                actor["username"] if actor else "",
                actor["role"] if actor else "",
                action,
                target_type,
                str(target_id),
                json.dumps(detail or {}, ensure_ascii=False),
                ip,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:  # noqa: BLE001 — audit must never break the main operation
        logger.warning("audit log write failed for %s: %s", action, e)
