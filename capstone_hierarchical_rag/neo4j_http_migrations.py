#!/usr/bin/env python3
"""
HTTP-based Neo4j migration using Query API (Aura/HTTP endpoint).
- Uses env: NEO4J_QUERY_API_URL, NEO4J_USERNAME, NEO4J_PASSWORD
- Creates constraints/indexes idempotently (IF NOT EXISTS)
- No deletes; safe to re-run
"""
from __future__ import annotations

import os
import base64
import json
from typing import List, Tuple

try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

# Load env from hierarchical first, then root
load_dotenv("capstone_hierarchical_rag/.env", override=False)
load_dotenv(".env", override=False)

API = os.getenv("NEO4J_QUERY_API_URL", "")
USER = os.getenv("NEO4J_USERNAME", "")
PASS = os.getenv("NEO4J_PASSWORD", "")
if not (API and USER and PASS):
    raise SystemExit("Missing NEO4J_QUERY_API_URL/NEO4J_USERNAME/NEO4J_PASSWORD")

LABELS_WITH_UNIQUE_IDS: List[str] = [
    "Meeting",
    "AgendaItem",
    "Person",
    "Topic",
    "Resolution",
    "Motion",
    "Department",
    "Contract",
    "Attachment",
    "Concept",
    "TranscriptSegment",
]

ID_PROPERTY_BY_LABEL = {
    "Meeting": "meeting_id",
    "AgendaItem": "item_id",
    "Person": "person_id",
    "Topic": "topic_id",
    "Resolution": "resolution_id",
    "Motion": "motion_id",
    "Department": "department_id",
    "Contract": "contract_id",
    "Attachment": "attachment_id",
    "Concept": "slug",
    "TranscriptSegment": "chunk_id",
}

PROPERTY_INDEXES: List[Tuple[str, str]] = [
    ("Meeting", "meeting_date"),
    ("Meeting", "meeting_type"),
    ("Meeting", "title"),
    ("AgendaItem", "title"),
    ("AgendaItem", "description"),
    ("Concept", "name"),
]

def _headers() -> dict:
    token = base64.b64encode(f"{USER}:{PASS}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json", "Accept": "application/json"}

async def _post(stmt: str, params: dict | None = None) -> None:
    if httpx is None:
        raise RuntimeError("httpx required for HTTP migrations")
    payload = {"statement": stmt, "parameters": params or {}}
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(API, json=payload, headers=_headers())
        if resp.status_code not in (200, 202):
            raise RuntimeError(f"Neo4j HTTP error {resp.status_code}: {resp.text[:200]}")

async def main() -> int:
    # Constraints
    for label in LABELS_WITH_UNIQUE_IDS:
        prop = ID_PROPERTY_BY_LABEL[label]
        stmt = f"CREATE CONSTRAINT {label.lower()}_{prop}_unique IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.{prop} IS UNIQUE"
        await _post(stmt)
    # Indexes
    for label, prop in PROPERTY_INDEXES:
        stmt = f"CREATE INDEX idx_{label.lower()}_{prop} IF NOT EXISTS FOR (n:`{label}`) ON (n.{prop})"
        await _post(stmt)
    print("✅ Neo4j HTTP migration complete")
    return 0

if __name__ == "__main__":
    import asyncio
    raise SystemExit(asyncio.run(main())) 