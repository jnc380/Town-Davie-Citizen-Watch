#!/usr/bin/env python3
"""
Backfill Meeting.url in Neo4j from local YouTube metadata files.
- Scans downloads/town_meetings_youtube/*.info.json
- Derives meeting_id from filename (strip .info in stem)
- Extracts URL from common fields (url, webpage_url, original_url)
- Updates Neo4j: MATCH (m:Meeting) WHERE m.meeting_id = $id OR m.id = $id SET m.url = $url
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple

from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv  # type: ignore
    # Try multiple .env locations
    for p in [Path(".env"), Path("capstone/.env"), Path(__file__).resolve().parent / ".env"]:
        if p.exists():
            load_dotenv(str(p))
            break
except Exception:
    pass

DOWNLOAD_DIR = Path(os.getenv("YOUTUBE_DOWNLOAD_DIR", "downloads/town_meetings_youtube"))
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
    print("Missing Neo4j environment variables (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)")
    sys.exit(1)

if not DOWNLOAD_DIR.exists():
    print(f"YouTube download directory not found: {DOWNLOAD_DIR}")
    sys.exit(1)


def extract_url(meta: Dict) -> str:
    for k in ("url", "webpage_url", "original_url"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def collect_mappings() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for path in sorted(DOWNLOAD_DIR.glob("*.info.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                info = json.load(f)
            # filename like <meeting_id>.info.json → stem is '<meeting_id>.info'
            meeting_id = path.stem.replace(".info", "")
            url = extract_url(info)
            if not url:
                continue
            mapping[meeting_id] = url
        except Exception as e:
            print(f"Skip {path}: {e}")
    return mapping


def backfill(driver, mapping: Dict[str, str]) -> Tuple[int, int]:
    updated = 0
    missing = 0
    with driver.session(database=NEO4J_DATABASE) as session:
        for mid, url in mapping.items():
            try:
                res = session.run(
                    """
                    MATCH (m:Meeting)
                    WHERE m.meeting_id = $id OR m.id = $id
                    SET m.url = $url
                    RETURN coalesce(m.meeting_id, m.id) as id
                    """,
                    {"id": mid, "url": url},
                ).single()
                if res and res.get("id"):
                    updated += 1
                else:
                    missing += 1
            except Exception as e:
                print(f"Failed to update {mid}: {e}")
    return updated, missing


if __name__ == "__main__":
    mapping = collect_mappings()
    if not mapping:
        print("No YouTube mappings found.")
        sys.exit(0)
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        ok, miss = backfill(drv, mapping)
        print(f"Updated {ok} meetings with URLs; {miss} meeting_ids not found in graph.")
    finally:
        drv.close() 