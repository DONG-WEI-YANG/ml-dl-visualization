from dataclasses import dataclass, field
import time


@dataclass
class ReadinessState:
    status: str = "starting"
    database: str = "connected"
    rag: str = "pending"
    started_at: float = field(default_factory=time.monotonic)

    def snapshot(self) -> dict:
        return {
            "status": self.status,
            "database": self.database,
            "rag": self.rag,
            "uptime_seconds": round(time.monotonic() - self.started_at, 3),
        }


readiness = ReadinessState()
