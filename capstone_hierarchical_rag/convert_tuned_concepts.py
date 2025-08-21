#!/usr/bin/env python3
"""
Convert tuned_concept_results.json into loader-ready files for Neo4j.
- Input: path to tuned_concept_results.json (in cleanup_files/concept_extraction)
- Outputs (in --outdir):
  - concepts_seed.json
  - concepts_taxonomy.json
  - agenda_concept_links.tsv
This script does NOT connect to Neo4j.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import argparse


def slugify(name: str) -> str:
    s = (name or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum(): out.append(ch)
        elif ch in [" ", "_", "-", "/"]: out.append("-")
        else: out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def compute_weight(freq: int) -> float:
    try:
        w = 1.0 + (max(0, int(freq)) / 50.0)
        return round(min(3.0, w), 2)
    except Exception:
        return 1.0


def convert(tuned: Dict[str, Any], min_freq: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], List[Tuple[str, str, float]]]:
    concepts_seed: List[Dict[str, Any]] = []
    taxonomy_edges: List[Dict[str, str]] = []
    agenda_links: List[Tuple[str, str, float]] = []

    category_seen: Set[str] = set()
    concept_seen: Set[str] = set()

    items: List[Dict[str, Any]] = tuned.get("meaningful_concepts") or []
    for item in items:
        name = (item.get("name") or "").strip()
        category = (item.get("category") or "").strip() or "uncategorized"
        freq = int(item.get("frequency") or 0)
        if min_freq and freq < min_freq: continue
        if not name: continue
        c_slug = slugify(name)
        cat_slug = slugify(category)
        if cat_slug and cat_slug not in category_seen:
            concepts_seed.append({"slug": cat_slug, "name": category.title(), "category": None})
            category_seen.add(cat_slug)
        if c_slug not in concept_seen:
            concepts_seed.append({"slug": c_slug, "name": name, "category": category or None})
            concept_seen.add(c_slug)
        taxonomy_edges.append({"parent": cat_slug, "child": c_slug})
        weight = compute_weight(freq)
        for ag in (item.get("agenda_items") or []):
            ag_id = (ag or "").strip()
            if not ag_id: continue
            agenda_links.append((ag_id, c_slug, weight))
    return concepts_seed, taxonomy_edges, agenda_links


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert tuned_concept_results.json to Neo4j loader files")
    parser.add_argument("--input", required=True, help="Path to tuned_concept_results.json")
    parser.add_argument("--outdir", default="capstone_hierarchical_rag", help="Output directory for loader files")
    parser.add_argument("--min-frequency", type=int, default=0, help="Drop concepts with frequency below this value")
    args = parser.parse_args()

    src = Path(args.input).resolve()
    if not src.exists():
        print(f"Input file not found: {src}")
        return 2

    with src.open("r", encoding="utf-8") as f:
        tuned = json.load(f)
    seed, tax, links = convert(tuned, min_freq=args.min_frequency)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "concepts_seed.json").open("w", encoding="utf-8") as f:
        json.dump(seed, f, ensure_ascii=False, indent=2)
    with (outdir / "concepts_taxonomy.json").open("w", encoding="utf-8") as f:
        json.dump(tax, f, ensure_ascii=False, indent=2)
    with (outdir / "agenda_concept_links.tsv").open("w", encoding="utf-8") as f:
        f.write("agenda_id\tconcept_slug\tweight\n")
        for ag_id, slug, w in links:
            f.write(f"{ag_id}\t{slug}\t{w}\n")

    print(f"Wrote {len(seed)} concepts, {len(tax)} taxonomy edges, {len(links)} agenda links to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main()) 