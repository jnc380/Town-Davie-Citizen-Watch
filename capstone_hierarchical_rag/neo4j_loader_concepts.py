#!/usr/bin/env python3
"""
Idempotent Neo4j loader for concepts, taxonomy, and weighted links (hierarchical).
- Reads from concepts_seed.json, concepts_taxonomy.json, agenda_concept_links.tsv in this folder (or via flags)
- Uses MERGE only; safe to re-run. No deletes.
- Deduplicates HAS_CONCEPT relationships, backfills relationship key (rk), and enforces uniqueness.
"""
from __future__ import annotations

import os
import json
import csv
from dataclasses import dataclass
from typing import List, Dict, Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv

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

class Neo4jConceptLoader:
    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database or os.getenv("NEO4J_DATABASE", "neo4j")

    def close(self) -> None:
        self._driver.close()

    def _run(self, cypher: str, params: Optional[Dict] = None) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(cypher, params or {})

    def ensure_schema(self) -> None:
        self._run("CREATE CONSTRAINT concept_slug_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.slug IS UNIQUE")
        self._run("CREATE INDEX idx_concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)")

    def _dedupe_by_node_pair(self) -> None:
        # Remove duplicate HAS_CONCEPT relationships for identical (a,c) pairs
        self._run(
            """
            MATCH (a:AgendaItem)-[r:HAS_CONCEPT]->(c:Concept)
            WITH a, c, collect(r) AS rels
            WHERE size(rels) > 1
            FOREACH(r IN tail(rels) | DELETE r)
            """
        )

    def _backfill_rk(self) -> None:
        # Set deterministic relationship key rk on HAS_CONCEPT relationships
        self._run(
            """
            MATCH (a:AgendaItem)-[r:HAS_CONCEPT]->(c:Concept)
            WHERE r.rk IS NULL
            SET r.rk = toString(a.item_id) + '|' + toString(c.slug)
            """
        )

    def _dedupe_by_rk(self) -> None:
        # Remove duplicates by rk across all HAS_CONCEPT relationships (handles duplicate AgendaItem nodes)
        self._run(
            """
            MATCH ()-[r:HAS_CONCEPT]->()
            WITH r.rk AS rk, collect(r) AS rels
            WHERE rk IS NOT NULL AND size(rels) > 1
            FOREACH (x IN tail(rels) | DELETE x)
            """
        )

    def _ensure_rel_unique_constraint(self) -> None:
        # Enforce uniqueness on HAS_CONCEPT.rk
        self._run("CREATE CONSTRAINT has_concept_rk_unique IF NOT EXISTS FOR ()-[r:HAS_CONCEPT]-() REQUIRE r.rk IS UNIQUE")

    def upsert_concepts(self, concepts: List[Concept]) -> int:
        if not concepts:
            return 0
        cy = (
            "UNWIND $rows AS c "
            "MERGE (x:Concept {slug:c.slug}) "
            "ON CREATE SET x.name=c.name, x.category=c.category "
            "ON MATCH  SET x.name=coalesce(c.name,x.name), x.category=coalesce(c.category,x.category)"
        )
        rows = [c.__dict__ for c in concepts]
        self._run(cy, {"rows": rows})
        return len(rows)

    def upsert_taxonomy(self, edges: List[TaxonomyEdge]) -> int:
        if not edges:
            return 0
        cy = (
            "UNWIND $rows AS r "
            "MATCH (p:Concept {slug:r.parent}), (ch:Concept {slug:r.child}) "
            "MERGE (p)-[:PARENT_OF]->(ch)"
        )
        rows = [e.__dict__ for e in edges]
        self._run(cy, {"rows": rows})
        return len(rows)

    def upsert_agenda_links(self, links: List[AgendaConceptLink]) -> int:
        if not links:
            return 0
        # Pre-step: dedupe and backfill
        self._dedupe_by_node_pair()
        self._backfill_rk()
        self._dedupe_by_rk()
        self._ensure_rel_unique_constraint()
        cy = (
            "UNWIND $rows AS r "
            "MATCH (c:Concept {slug:r.slug}) "
            "MATCH (a:AgendaItem {item_id:r.agenda_id}) "
            "WITH collect(a) AS as_, c, r "
            "WITH head(as_) AS a, c, r, toString(r.agenda_id) + '|' + toString(r.slug) AS rk "
            "OPTIONAL MATCH ()-[existing:HAS_CONCEPT {rk: rk}]->() "
            "FOREACH (_ IN CASE WHEN existing IS NULL THEN [1] ELSE [] END | "
            "  MERGE (a)-[rel:HAS_CONCEPT {rk: rk}]->(c) "
            "  SET rel.weight = coalesce(r.weight, 1.0) "
            ") "
            "FOREACH (_ IN CASE WHEN existing IS NOT NULL THEN [1] ELSE [] END | "
            "  SET existing.weight = coalesce(r.weight, existing.weight) "
            ")"
        )
        rows = [l.__dict__ for l in links]
        self._run(cy, {"rows": rows})
        return len(rows)


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
            agenda_id = (row.get("agenda_id") or "").strip()
            slug = (row.get("concept_slug") or "").strip()
            weight_s = (row.get("weight") or "").strip()
            if not agenda_id or not slug:
                continue
            try:
                weight = float(weight_s) if weight_s else 1.0
            except Exception:
                weight = 1.0
            out.append(AgendaConceptLink(agenda_id=agenda_id, slug=slug, weight=weight))
    return out


def main(argv: List[str]) -> int:
    import argparse

    # Load env from hierarchical .env first
    load_dotenv("capstone_hierarchical_rag/.env", override=False)
    load_dotenv(".env", override=False)

    parser = argparse.ArgumentParser(description="Neo4j loader for Concepts (hierarchical)")
    parser.add_argument("--concepts", type=str, default="capstone_hierarchical_rag/concepts_seed.json")
    parser.add_argument("--taxonomy", type=str, default="capstone_hierarchical_rag/concepts_taxonomy.json")
    parser.add_argument("--agenda-links", type=str, default="capstone_hierarchical_rag/agenda_concept_links.tsv")
    parser.add_argument("--ensure-schema", action="store_true")

    args = parser.parse_args(argv)

    uri = os.getenv("NEO4J_URI", "")
    user = os.getenv("NEO4J_USERNAME", "")
    pw = os.getenv("NEO4J_PASSWORD", "")
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    if not (uri and user and pw):
        print("NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD are required")
        return 2

    loader = Neo4jConceptLoader(uri, user, pw, database=db)
    try:
        if args.ensure_schema:
            loader.ensure_schema()
        concepts = read_concepts(args.concepts)
        taxonomy = read_taxonomy(args.taxonomy)
        agenda_links = read_agenda_links(args.agenda_links)

        c = loader.upsert_concepts(concepts)
        t = loader.upsert_taxonomy(taxonomy)
        a = loader.upsert_agenda_links(agenda_links)
        print(f"Loaded: concepts={c} taxonomy_edges={t} agenda_links={a}")
        return 0
    finally:
        loader.close()


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:])) 