#!/usr/bin/env python3
"""Bulk-ingest local curriculum into a remote (HF Spaces) RAG backend.

Usage:
    python scripts/ingest_to_cloud.py                          # uses default HF Spaces URL
    python scripts/ingest_to_cloud.py --api https://custom.url # custom API
    python scripts/ingest_to_cloud.py --dry-run                # preview without sending

The script:
  1. Loads curriculum markdown files from ../curriculum/
  2. Chunks them (same logic as app.rag.chunker)
  3. Authenticates as admin via /api/auth/login
  4. POSTs chunks to /api/rag/ingest/bulk
"""

import argparse
import getpass
import json
import sys
from pathlib import Path

import requests

# Add parent directory to path so we can import the chunker
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.rag.chunker import load_curriculum_chunks  # noqa: E402

DEFAULT_API = "https://kevin19830331-ml-dl-viz-api.hf.space"


def login(api_base: str, username: str, password: str) -> str:
    """Authenticate and return JWT token."""
    resp = requests.post(
        f"{api_base}/api/auth/login",
        json={"username": username, "password": password},
        timeout=15,
    )
    if resp.status_code != 200:
        print(f"Login failed ({resp.status_code}): {resp.text}")
        sys.exit(1)
    return resp.json()["access_token"]


def bulk_ingest(api_base: str, token: str, chunks: list[dict]) -> dict:
    """POST chunks to /api/rag/ingest/bulk."""
    resp = requests.post(
        f"{api_base}/api/rag/ingest/bulk",
        json={"chunks": chunks},
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    if resp.status_code != 200:
        print(f"Ingest failed ({resp.status_code}): {resp.text}")
        sys.exit(1)
    return resp.json()


def get_stats(api_base: str) -> dict:
    """GET current RAG stats."""
    resp = requests.get(f"{api_base}/api/rag/stats", timeout=10)
    return resp.json() if resp.status_code == 200 else {}


def main():
    parser = argparse.ArgumentParser(description="Bulk-ingest curriculum to cloud RAG")
    parser.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default: {DEFAULT_API})")
    parser.add_argument("--username", "-u", default="admin", help="Admin username (default: admin)")
    parser.add_argument("--password", "-p", default="", help="Admin password (prompted if not given)")
    parser.add_argument("--dry-run", action="store_true", help="Preview chunks without sending")
    args = parser.parse_args()

    api_base = args.api.rstrip("/")

    # 1. Load and chunk curriculum
    print("Loading curriculum from local files...")
    chunks = load_curriculum_chunks()
    print(f"  Loaded {len(chunks)} chunks from curriculum/")

    if not chunks:
        print("No chunks found. Check that curriculum/ directory has week-*/lecture.md etc.")
        sys.exit(1)

    # Show summary by week
    by_week: dict[int, int] = {}
    for c in chunks:
        w = c["metadata"]["week"]
        by_week[w] = by_week.get(w, 0) + 1
    for w in sorted(by_week):
        label = f"Week {w:02d}" if w > 0 else "Syllabus"
        print(f"    {label}: {by_week[w]} chunks")

    if args.dry_run:
        print("\n[Dry run] Would send these chunks to:", api_base)
        return

    # 2. Show current cloud stats
    print(f"\nChecking current RAG stats on {api_base}...")
    stats = get_stats(api_base)
    if stats:
        print(f"  Currently indexed: {stats.get('total_chunks', 0)} chunks")

    # 3. Authenticate
    password = args.password or getpass.getpass(f"Password for '{args.username}': ")
    print("Logging in...")
    token = login(api_base, args.username, password)
    print("  Authenticated!")

    # 4. Bulk ingest
    print(f"Sending {len(chunks)} chunks to {api_base}/api/rag/ingest/bulk ...")
    result = bulk_ingest(api_base, token, chunks)
    print(f"  Done! Indexed: {result.get('chunks_indexed', '?')} chunks")

    # 5. Verify
    stats_after = get_stats(api_base)
    if stats_after:
        print(f"  RAG stats after ingest: {stats_after.get('total_chunks', 0)} total chunks")

    print("\nCurriculum successfully ingested to cloud RAG!")
    print("Web enrichment will add Wikipedia content automatically (daily).")


if __name__ == "__main__":
    main()
