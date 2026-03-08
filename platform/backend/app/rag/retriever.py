"""High-level retriever: combines FTS search with week-based context."""

from .store import search_fts, get_chunks_by_week

MAX_CONTEXT_CHARS = 4000  # Limit injected context size


def retrieve_context(query: str, week: int, top_k: int = 5) -> str:
    """Retrieve relevant curriculum content for a student query.

    Strategy:
    1. Search FTS with week filter first (most relevant)
    2. If not enough results, search across all weeks
    3. Build a formatted context string for the LLM
    """
    # First: search within the current week
    results = search_fts(query, week=week, top_k=top_k)

    # If fewer than 2 results from current week, broaden search
    if len(results) < 2:
        broad_results = search_fts(query, week=None, top_k=top_k)
        # Add non-duplicate broad results
        seen_ids = {r["id"] for r in results}
        for r in broad_results:
            if r["id"] not in seen_ids:
                results.append(r)
                if len(results) >= top_k:
                    break

    if not results:
        return ""

    # Build context string
    parts = []
    total_chars = 0
    for r in results:
        entry = f"[來源：第{r['week']}週 {r['file_type']} — {r['title']}]\n{r['content']}"
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            break
        parts.append(entry)
        total_chars += len(entry)

    return "\n\n---\n\n".join(parts)


def get_week_summary(week: int) -> str:
    """Get a summary of all content for a specific week."""
    chunks = get_chunks_by_week(week)
    if not chunks:
        return ""

    # Take first chunk from each file type (usually the heading/intro)
    seen_types = set()
    parts = []
    for c in chunks:
        ft = c["file_type"]
        if ft not in seen_types:
            seen_types.add(ft)
            # Take first 500 chars as summary
            parts.append(f"[{ft}] {c['content'][:500]}")
    return "\n\n".join(parts)
