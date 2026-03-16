import logging
import time
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.llm_routes import router as llm_router
from app.api.model_routes import router as model_router
from app.api.analytics_routes import router as analytics_router
from app.api.auth_routes import router as auth_router
from app.api.admin_routes import router as admin_router
from app.api.rag_routes import router as rag_router
from app.api.quiz_routes import router as quiz_router
from app.api.curriculum_routes import router as curriculum_router
from app.config import settings
from app.db import init_db, get_db
from app.rag.store import init_rag_tables

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML/DL 視覺化教學平台", version="0.3.0")

# ── CORS (environment-driven) ──
cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Simple rate limiting middleware ──
_rate_limits: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 60  # requests per window
RATE_WINDOW = 60  # seconds


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    # Clean old entries
    _rate_limits[client_ip] = [t for t in _rate_limits[client_ip] if now - t < RATE_WINDOW]
    if len(_rate_limits[client_ip]) >= RATE_LIMIT:
        return JSONResponse({"detail": "Too many requests"}, status_code=429)
    _rate_limits[client_ip].append(now)
    return await call_next(request)


# ── Request logging middleware ──
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info("%s %s %d %.0fms", request.method, request.url.path, response.status_code, duration)
    return response


from contextlib import asynccontextmanager


def _auto_ingest_curriculum():
    """Auto-ingest curriculum if RAG is empty (e.g. after container restart)."""
    try:
        from app.rag.store import get_stats
        stats = get_stats()
        if stats.get("curriculum_chunks", 0) > 0:
            logger.info("RAG already has %d curriculum chunks, skipping auto-ingest", stats["curriculum_chunks"])
            return
        from app.rag.chunker import load_curriculum_chunks
        chunks = load_curriculum_chunks()
        if not chunks:
            logger.info("No local curriculum files found, skipping auto-ingest")
            return
        from app.rag.store import ingest_chunks
        count = ingest_chunks(chunks)
        logger.info("Auto-ingested %d curriculum chunks into RAG", count)
    except Exception as e:
        logger.warning("Auto-ingest failed (non-fatal): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_rag_tables()
    # Auto-ingest curriculum if RAG is empty
    _auto_ingest_curriculum()
    # Start daily web enrichment background task
    from app.rag.web_enricher import start_daily_enrichment
    start_daily_enrichment()
    logger.info("ML/DL Visualization Platform started (CORS origins: %s)", cors_origins)
    yield


app.router.lifespan_context = lifespan

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(llm_router)
app.include_router(model_router)
app.include_router(analytics_router)
app.include_router(rag_router)
app.include_router(quiz_router)
app.include_router(curriculum_router)


@app.get("/health")
async def health():
    """Health check with database connectivity verification."""
    try:
        conn = get_db()
        conn.execute("SELECT 1")
        conn.close()
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return JSONResponse(
            {"status": "error", "database": "disconnected", "detail": str(e)},
            status_code=503,
        )
