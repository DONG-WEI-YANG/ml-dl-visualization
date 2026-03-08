from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from app.auth.dependencies import require_admin
from app.rag.chunker import load_curriculum_chunks
from app.rag.store import ingest_chunks, search_fts, get_stats, init_rag_tables
from app.rag.retriever import retrieve_context

router = APIRouter(prefix="/api/rag", tags=["RAG"])


class SearchRequest(BaseModel):
    query: str
    week: int | None = None
    top_k: int = 5


@router.post("/ingest")
async def ingest_curriculum(admin: dict = Depends(require_admin)):
    """Parse and index all curriculum files. Admin only."""
    init_rag_tables()
    chunks = load_curriculum_chunks()
    count = ingest_chunks(chunks)
    return {"status": "ok", "chunks_indexed": count}


@router.get("/stats")
async def rag_stats():
    """Get RAG index statistics."""
    return get_stats()


@router.post("/search")
async def search(req: SearchRequest):
    """Search curriculum content."""
    results = search_fts(req.query, week=req.week, top_k=req.top_k)
    return {"results": results, "count": len(results)}


@router.post("/context")
async def get_context(req: SearchRequest):
    """Get formatted RAG context for a query (what the LLM sees)."""
    context = retrieve_context(req.query, week=req.week or 1, top_k=req.top_k)
    return {"context": context}
