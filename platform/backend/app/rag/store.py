"""SQLite FTS5 vector store for curriculum retrieval.

Uses SQLite Full-Text Search for keyword-based retrieval (zero external deps).
Optionally supports embedding-based semantic search via chromadb if installed.
"""

import hashlib
import sqlite3
import json
import re
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "app.db"

# Regex to match CJK Unified Ideographs (common Chinese/Japanese/Korean characters)
_CJK_RE = re.compile(r"([\u4e00-\u9fff\u3400-\u4dbf])")


def _space_cjk(text: str) -> str:
    """Insert spaces around each CJK character so unicode61 tokenizer indexes them individually."""
    return _CJK_RE.sub(r" \1 ", text)


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_rag_tables():
    conn = get_db()
    # Use standalone FTS table (not content-synced) so we can store
    # CJK-spaced text for proper tokenization while keeping original
    # content in rag_chunks.
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            week INTEGER NOT NULL,
            file_type TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS rag_fts USING fts5(
            chunk_id,
            content,
            title,
            tokenize='unicode61'
        );

        CREATE TABLE IF NOT EXISTS rag_content_hashes (
            content_hash TEXT PRIMARY KEY,
            chunk_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS rag_enrichment_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source TEXT NOT NULL,
            chunks_found INTEGER NOT NULL DEFAULT 0,
            chunks_added INTEGER NOT NULL DEFAULT 0,
            chunks_skipped INTEGER NOT NULL DEFAULT 0,
            topics_searched TEXT NOT NULL DEFAULT ''
        );
    """)
    conn.commit()
    conn.close()


def ingest_chunks(chunks: list[dict]) -> int:
    """Insert or replace chunks into the store. Returns count."""
    conn = get_db()
    # Clear existing data
    conn.execute("DELETE FROM rag_chunks")
    conn.execute("DELETE FROM rag_fts")
    for chunk in chunks:
        meta = chunk["metadata"]
        conn.execute(
            "INSERT OR REPLACE INTO rag_chunks (id, content, week, file_type, title, source) VALUES (?, ?, ?, ?, ?, ?)",
            (chunk["id"], chunk["content"], meta["week"], meta["file_type"], meta["title"], meta["source"]),
        )
        # Insert CJK-spaced version into FTS for proper Chinese tokenization
        conn.execute(
            "INSERT INTO rag_fts (chunk_id, content, title) VALUES (?, ?, ?)",
            (chunk["id"], _space_cjk(chunk["content"]), _space_cjk(meta["title"])),
        )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
    conn.close()
    return count


def _tokenize_query(query: str) -> str:
    """Tokenize query for FTS5 with CJK-spaced index.

    Since the index has spaces between CJK chars, we space them in the query
    too. Each CJK character becomes a separate token; English words stay intact.
    We join with OR for broad matching.
    """
    # Space out CJK characters (matches how content was indexed)
    spaced = _space_cjk(query)
    # Remove special chars (keep alphanumeric + CJK + spaces)
    clean = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', spaced).strip()
    if not clean:
        return ""

    tokens = clean.split()
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in tokens:
        low = t.lower()
        if low not in seen:
            seen.add(low)
            unique.append(t)

    return " OR ".join(unique) if unique else ""


def search_fts(query: str, week: int | None = None, top_k: int = 5) -> list[dict]:
    """Full-text search using SQLite FTS5 with BM25 ranking."""
    conn = get_db()

    fts_query = _tokenize_query(query)
    if not fts_query:
        conn.close()
        return []

    try:
        if week:
            rows = conn.execute(
                """SELECT c.id, c.content, c.week, c.file_type, c.title, c.source,
                          f.rank as score
                   FROM rag_fts f
                   JOIN rag_chunks c ON f.chunk_id = c.id
                   WHERE rag_fts MATCH ? AND c.week = ?
                   ORDER BY f.rank
                   LIMIT ?""",
                (fts_query, week, top_k),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT c.id, c.content, c.week, c.file_type, c.title, c.source,
                          f.rank as score
                   FROM rag_fts f
                   JOIN rag_chunks c ON f.chunk_id = c.id
                   WHERE rag_fts MATCH ?
                   ORDER BY f.rank
                   LIMIT ?""",
                (fts_query, top_k),
            ).fetchall()
    except Exception:
        rows = []

    conn.close()

    return [
        {
            "id": r["id"],
            "content": r["content"],
            "week": r["week"],
            "file_type": r["file_type"],
            "title": r["title"],
            "source": r["source"],
            "score": r["score"],
        }
        for r in rows
    ]


def get_chunks_by_week(week: int) -> list[dict]:
    """Get all chunks for a specific week (for context injection)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM rag_chunks WHERE week = ? ORDER BY id",
        (week,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
    by_week = conn.execute(
        "SELECT week, COUNT(*) as cnt FROM rag_chunks GROUP BY week ORDER BY week"
    ).fetchall()
    conn.close()
    return {
        "total_chunks": total,
        "by_week": [{"week": r["week"], "count": r["cnt"]} for r in by_week],
    }


def _content_hash(text: str) -> str:
    """SHA256 hash of normalized text for deduplication."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def append_chunks(chunks: list[dict]) -> dict:
    """Append new chunks, skipping duplicates. Returns {added, skipped}."""
    conn = get_db()
    added = 0
    skipped = 0
    for chunk in chunks:
        meta = chunk["metadata"]
        h = _content_hash(chunk["content"])
        # Check if content already exists
        existing = conn.execute(
            "SELECT 1 FROM rag_content_hashes WHERE content_hash = ?", (h,)
        ).fetchone()
        if existing:
            skipped += 1
            continue
        # Insert chunk
        conn.execute(
            "INSERT OR REPLACE INTO rag_chunks (id, content, week, file_type, title, source) VALUES (?, ?, ?, ?, ?, ?)",
            (chunk["id"], chunk["content"], meta["week"], meta["file_type"], meta["title"], meta["source"]),
        )
        conn.execute(
            "INSERT INTO rag_fts (chunk_id, content, title) VALUES (?, ?, ?)",
            (chunk["id"], _space_cjk(chunk["content"]), _space_cjk(meta["title"])),
        )
        conn.execute(
            "INSERT OR IGNORE INTO rag_content_hashes (content_hash, chunk_id) VALUES (?, ?)",
            (h, chunk["id"]),
        )
        added += 1
    conn.commit()
    conn.close()
    return {"added": added, "skipped": skipped}


def log_enrichment(source: str, chunks_found: int, chunks_added: int, chunks_skipped: int, topics: str):
    conn = get_db()
    conn.execute(
        "INSERT INTO rag_enrichment_log (source, chunks_found, chunks_added, chunks_skipped, topics_searched) VALUES (?, ?, ?, ?, ?)",
        (source, chunks_found, chunks_added, chunks_skipped, topics),
    )
    conn.commit()
    conn.close()


def get_enrichment_history(limit: int = 20) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM rag_enrichment_log ORDER BY run_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
