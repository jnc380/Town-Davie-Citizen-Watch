#!/usr/bin/env python3
"""
HTTP-based Neo4j loader for concepts/taxonomy/links via Query API.
Env: NEO4J_QUERY_API_URL, NEO4J_USERNAME, NEO4J_PASSWORD
Inputs default to files in capstone_hierarchical_rag/ (override with flags)
"""
from __future__ import annotations

import os
import base64
import json
import csv
from dataclasses import dataclass
from typing import List, Dict, Optional

try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

@dataclass
class Concept:
    slug: str
    name: str
    category: Optional[str] = None

@dataclass
class TaxonomyEdge:
    parent: str
    child: str

@dataclass
class AgendaConceptLink:
    agenda_id: str
    slug: str
    weight: float = 1.0

# Load env from hierarchical first
load_dotenv("capstone_hierarchical_rag/.env", override=False)
load_dotenv(".env", override=False)

API = os.getenv("NEO4J_QUERY_API_URL", "")
USER = os.getenv("NEO4J_USERNAME", "")
PASS = os.getenv("NEO4J_PASSWORD", "")
if not (API and USER and PASS):
    raise SystemExit("Missing NEO4J_QUERY_API_URL/NEO4J_USERNAME/NEO4J_PASSWORD")

def _headers() -> dict:
    token = base64.b64encode(f"{USER}:{PASS}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json", "Accept": "application/json"}

async def _post(stmt: str, params: dict | None = None) -> None:
    if httpx is None:
        raise RuntimeError("httpx required for HTTP loading")
    payload = {"statement": stmt, "parameters": params or {}}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(API, json=payload, headers=_headers())
        if resp.status_code not in (200, 202):
            raise RuntimeError(f"Neo4j HTTP error {resp.status_code}: {resp.text[:200]}")

async def upsert_concepts(rows: List[Concept]) -> None:
    if not rows:
        return
    stmt = (
        "UNWIND $rows AS c "
        "MERGE (x:Concept {slug:c.slug}) "
        "ON CREATE SET x.name=c.name, x.category=c.category "
        "ON MATCH  SET x.name=coalesce(c.name,x.name), x.category=coalesce(c.category,x.category)"
    )
    await _post(stmt, {"rows": [c.__dict__ for c in rows]})

async def upsert_taxonomy(edges: List[TaxonomyEdge]) -> None:
    if not edges:
        return
    stmt = (
        "UNWIND $rows AS r "
        "MATCH (p:Concept {slug:r.parent}), (ch:Concept {slug:r.child}) "
        "MERGE (p)-[:PARENT_OF]->(ch)"
    )
    await _post(stmt, {"rows": [e.__dict__ for e in edges]})

async def upsert_agenda_links(links: List[AgendaConceptLink]) -> None:
    if not links:
        return
    stmt = (
        "UNWIND $rows AS r "
        "MATCH (a:AgendaItem {item_id:r.agenda_id}) "
        "MATCH (c:Concept {slug:r.slug}) "
        "MERGE (a)-[rel:HAS_CONCEPT]->(c) "
        "SET rel.weight = coalesce(r.weight, 1.0)"
    )
    await _post(stmt, {"rows": [l.__dict__ for l in links]})


def read_concepts(path: str) -> List[Concept]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[Concept] = []
    for row in data:
        slug = (row.get("slug") or "").strip()
        name = (row.get("name") or "").strip()
        category = row.get("category")
        if slug and name:
            out.append(Concept(slug=slug, name=name, category=category))
    return out


def read_taxonomy(path: str) -> List[TaxonomyEdge]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[TaxonomyEdge] = []
    for row in data:
        parent = (row.get("parent") or "").strip()
        child = (row.get("child") or "").strip()
        if parent and child:
            out.append(TaxonomyEdge(parent=parent, child=child))
    return out


def read_agenda_links(path: str) -> List[AgendaConceptLink]:
    if not path or not os.path.isfile(path):
        return []
    out: List[AgendaConceptLink] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ag = (row.get("agenda_id") or "").strip()
            slug = (row.get("concept_slug") or "").strip()
            weight_s = (row.get("weight") or "").strip()
            if not ag or not slug:
                continue
            try:
                w = float(weight_s) if weight_s else 1.0
            except Exception:
                w = 1.0
            out.append(AgendaConceptLink(agenda_id=ag, slug=slug, weight=w))
    return out

async def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Neo4j HTTP loader for Concepts")
    parser.add_argument("--concepts", type=str, default="capstone_hierarchical_rag/concepts_seed.json")
    parser.add_argument("--taxonomy", type=str, default="capstone_hierarchical_rag/concepts_taxonomy.json")
    parser.add_argument("--agenda-links", type=str, default="capstone_hierarchical_rag/agenda_concept_links.tsv")
    args = parser.parse_args()

    concepts = read_concepts(args.concepts)
    taxonomy = read_taxonomy(args.taxonomy)
    agenda_links = read_agenda_links(args.agenda_links)

    await upsert_concepts(concepts)
    await upsert_taxonomy(taxonomy)
    await upsert_agenda_links(agenda_links)
    print(f"Loaded via HTTP: concepts={len(concepts)} taxonomy_edges={len(taxonomy)} agenda_links={len(agenda_links)}")
    return 0

if __name__ == "__main__":
    import asyncio
    raise SystemExit(asyncio.run(main())) 