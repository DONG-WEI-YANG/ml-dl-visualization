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


# File types that come from local curriculum (vs web enrichment)
CURRICULUM_FILE_TYPES = {"lecture", "slides", "assignment", "syllabus"}


def ingest_chunks(chunks: list[dict]) -> int:
    """Replace curriculum chunks while preserving web-enrichment data. Returns curriculum count."""
    conn = get_db()

    # Only delete curriculum chunks — leave web enrichment (web-zh, web-en) intact
    curriculum_ids = [
        r[0] for r in conn.execute(
            "SELECT id FROM rag_chunks WHERE file_type IN ('lecture','slides','assignment','syllabus')"
        ).fetchall()
    ]
    if curriculum_ids:
        placeholders = ",".join("?" * len(curriculum_ids))
        conn.execute(f"DELETE FROM rag_chunks WHERE id IN ({placeholders})", curriculum_ids)
        conn.execute(f"DELETE FROM rag_fts WHERE chunk_id IN ({placeholders})", curriculum_ids)
        conn.execute(
            f"DELETE FROM rag_content_hashes WHERE chunk_id IN ({placeholders})", curriculum_ids
        )

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
    # Breakdown: curriculum vs web enrichment
    curriculum_count = conn.execute(
        "SELECT COUNT(*) FROM rag_chunks WHERE file_type IN ('lecture','slides','assignment','syllabus')"
    ).fetchone()[0]
    web_count = conn.execute(
        "SELECT COUNT(*) FROM rag_chunks WHERE file_type IN ('web-zh','web-en')"
    ).fetchone()[0]
    conn.close()
    return {
        "total_chunks": total,
        "curriculum_chunks": curriculum_count,
        "web_chunks": web_count,
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


def cleanup_garbage_chunks() -> dict:
    """Remove garbage web-enrichment chunks from the RAG store.

    Targets: Wikipedia redirects, disambiguation, Simplified Chinese remnants,
    and chunks with too much LaTeX/math noise.
    """
    try:
        from opencc import OpenCC
        occ = OpenCC("s2twp")
    except ImportError:
        occ = None

    # Garbage indicators
    garbage_texts = [
        "簡繁重定向", "本重定向用來", "請勿使用管道連結",
        "消歧義", "消歧义", "本條目存在以下問題",
        "Template:", "分類:",
    ]
    # Simplified-only character pattern
    simplified_re = re.compile(
        r"[么个们仅从优体儿关兴养减几则创办务动劳单卫发变响团园圆坏块场声处备够头夸奋奖妇学宁宝实审宽将对导专岁岛帅师帮干并广庆库应废开异弃张弹归当录忆志忧怀态总恋惊愿懒戏户扑执扩扫扬护报拟拥择挡挤挥损换据携操收斗断无时显晓暂术机杀杂权条来杨极构档梦检样桥欢毁毕汇汉汤济温满灭灯灵烂烧热爱状独狭猎环现电疗盐监盖矫矿硕础确码种稳窃竞笔笼简粮纠纤纪纯纸线练组细经绘结给统继绩绪续维综绿编缘缝缩网罗翘职联聪肤肠脑腾艰艺节荣获蓝虑虽装观览觉规视证评词译试话说请读课调谢资赋赏赔赖赚赛赞趋跃踪轨轮软轴轻载辅辆辈辉辑输辩边达迁过运近还这进远连迟适选递通遗邻释里钟钢钱铁银链锁错键长门闭问闲间闻阀阅阳阴阶阻际陆陈险随隐难雾静韩顶顿预领频颜风饭饮饰馆驰驱验骗鱼鸟鸡龙龟]"
    )

    conn = get_db()
    # Get all web-enrichment chunks
    rows = conn.execute(
        "SELECT id, content FROM rag_chunks WHERE file_type IN ('web-zh', 'web-en')"
    ).fetchall()

    removed = 0
    cleaned = 0
    ids_to_remove = []

    for row in rows:
        chunk_id = row["id"]
        content = row["content"]

        # Check for garbage
        is_garbage = False
        for gt in garbage_texts:
            if gt in content:
                is_garbage = True
                break

        if is_garbage or len(content.strip()) < 30:
            ids_to_remove.append(chunk_id)
            removed += 1
            continue

        # Clean Simplified Chinese
        new_content = content
        if occ is not None:
            try:
                new_content = occ.convert(new_content)
            except Exception:
                pass
        new_content = simplified_re.sub("", new_content)

        # Clean LaTeX artifacts
        new_content = re.sub(r"\\(?:displaystyle|alpha|beta|gamma|delta|sigma|theta|lambda|nabla|partial|hat|vec|frac|sqrt|sum|prod|int|lim|operatorname|left|right|cdot|times)\b", "", new_content)
        new_content = re.sub(r"[{}]", "", new_content)
        new_content = re.sub(r"\n{3,}", "\n\n", new_content)
        new_content = re.sub(r"  +", " ", new_content).strip()

        if len(new_content) < 30:
            ids_to_remove.append(chunk_id)
            removed += 1
            continue

        if new_content != content:
            conn.execute("UPDATE rag_chunks SET content = ? WHERE id = ?", (new_content, chunk_id))
            # Update FTS index
            conn.execute("DELETE FROM rag_fts WHERE chunk_id = ?", (chunk_id,))
            conn.execute(
                "INSERT INTO rag_fts (chunk_id, content, title) VALUES (?, ?, ?)",
                (chunk_id, _space_cjk(new_content), _space_cjk(chunk_id)),
            )
            cleaned += 1

    # Remove garbage chunks
    if ids_to_remove:
        placeholders = ",".join("?" * len(ids_to_remove))
        conn.execute(f"DELETE FROM rag_chunks WHERE id IN ({placeholders})", ids_to_remove)
        conn.execute(f"DELETE FROM rag_fts WHERE chunk_id IN ({placeholders})", ids_to_remove)
        conn.execute(f"DELETE FROM rag_content_hashes WHERE chunk_id IN ({placeholders})", ids_to_remove)

    conn.commit()
    conn.close()
    return {"removed": removed, "cleaned": cleaned, "total_processed": len(rows)}
