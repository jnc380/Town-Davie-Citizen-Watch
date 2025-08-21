import os
import re
from typing import List, Optional, Tuple
from neo4j import GraphDatabase
from dotenv import load_dotenv

"""
Backfill/enrichment script for Neo4j.
- Derives basic nodes and relationships from existing (Meeting)-[:HAS_AGENDA_ITEM]->(AgendaItem)
- Creates:
  - (AgendaItem)-[:RESULTED_IN]->(Resolution) when likely resolution/outcome detected
  - (AgendaItem)-[:LINKS_TO]->(Contract) when a vendor/contract-like phrase detected
  - (Meeting)-[:DISCUSSED]->(Topic) for lightweight topics extracted from title/description
- Idempotent via MERGE; safe to re-run.
- Does NOT fabricate motions/votes.
"""

AMOUNT_PATTERNS = [
    re.compile(r"\$\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?", re.I),
    re.compile(r"\$\d+(?:\.\d+)?\s*(?:m|million|b|billion)\b", re.I),
    re.compile(r"\$\d+(?:\.\d+)?\s*[\-–]\s*\$?\d+(?:\.\d+)?\s*(?:m|million|b|billion)?", re.I),
]

OUTCOME_KEYWORDS = [
    (re.compile(r"\bapproved\b|motion to approve|approve[d]?", re.I), "Approved"),
    (re.compile(r"\badopted\b", re.I), "Adopted"),
    (re.compile(r"\bpassed?\b|passes", re.I), "Passed"),
    (re.compile(r"\bdenied\b", re.I), "Denied"),
    (re.compile(r"\bfailed\b", re.I), "Failed"),
    (re.compile(r"\btabled\b", re.I), "Tabled"),
    (re.compile(r"\bwithdrawn\b", re.I), "Withdrawn"),
    (re.compile(r"\bcontinued\b|\bpostponed\b", re.I), "Continued"),
]

VENDOR_PATTERNS = [
    re.compile(r"(?:agreement|contract|services|purchase|psa|engagement)\s+with\s+([A-Z][^,;\n]{2,60})", re.I),
    re.compile(r"award\s+to\s+([A-Z][^,;\n]{2,60})", re.I),
    re.compile(r"with\s+([A-Z][A-Za-z&\.'\- ]{2,60})\s+(?:for|to)", re.I),
]

TOPIC_TERMS = [
    "CRA", "Beautification", "Budget", "Transfer", "Interlocal", "Agreement",
    "Stormwater", "Fire", "Police", "Parks", "Roadway", "Pine Island", "University Drive",
    "Stirling", "Davie Road", "True-Up", "Assessment", "Rate", "Resolution", "Contract",
]

def extract_amounts(text: str) -> List[str]:
    if not text:
        return []
    vals: List[str] = []
    for p in AMOUNT_PATTERNS:
        vals.extend(m.group(0) for m in p.finditer(text))
    seen = set()
    out: List[str] = []
    for v in vals:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out

def extract_outcome(text: str) -> Optional[str]:
    if not text:
        return None
    for rx, label in OUTCOME_KEYWORDS:
        if rx.search(text):
            return label
    return None

def extract_vendor(text: str) -> Optional[str]:
    if not text:
        return None
    for rx in VENDOR_PATTERNS:
        m = rx.search(text)
        if m:
            name = m.group(1).strip()
            if len(name) > 2 and "town of davie" not in name.lower():
                return name
    return None

def extract_topics(text: str) -> List[str]:
    topics: List[str] = []
    if not text:
        return topics
    for term in TOPIC_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text, flags=re.I):
            topics.append(term)
    # Collapse to unique, case-insensitive
    uniq = []
    seen = set()
    for t in topics:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


def run(limit: Optional[int] = None):
    load_dotenv("capstone/.env", override=True)
    uri = os.getenv("NEO4J_URI"); user = os.getenv("NEO4J_USERNAME"); pw = os.getenv("NEO4J_PASSWORD"); db = os.getenv("NEO4J_DATABASE", "neo4j")
    assert uri and user and pw, "Missing Neo4j env"
    drv = GraphDatabase.driver(uri, auth=(user, pw))
    with drv.session(database=db) as s:
        lim = "" if not limit else f"LIMIT {int(limit)}"
        q = f"""
        MATCH (m:Meeting)-[:HAS_AGENDA_ITEM]->(ai:AgendaItem)
        RETURN m.meeting_id AS mid, m.title AS mtitle, m.meeting_date AS mdate,
               ai.item_id AS item_id, ai.title AS title, ai.description AS description
        {lim}
        """
        data = s.run(q).data()
        for row in data:
            mid = row["mid"]; mtitle = row.get("mtitle") or ""; mdate = row.get("mdate") or ""
            item_id = row["item_id"]; title = row.get("title") or ""; desc = row.get("description") or ""
            text = f"{title}\n{desc}"
            amounts = extract_amounts(text)
            outcome = extract_outcome(text)
            vendor = extract_vendor(text)
            topics = extract_topics(text)

            # Topics: (Meeting)-[:DISCUSSED]->(Topic)
            for topic in topics:
                s.run(
                    """
                    MERGE (t:Topic {topic_id: $tid})
                    ON CREATE SET t.name = $tname
                    WITH t
                    MATCH (m:Meeting {meeting_id: $mid})
                    MERGE (m)-[:DISCUSSED]->(t)
                    """,
                    {"tid": topic.lower(), "tname": topic, "mid": mid}
                )

            # Resolution: if outcome or resolution keyword present
            if outcome or re.search(r"\bresolution\b|\badopt\w*\b", text, flags=re.I):
                s.run(
                    """
                    MERGE (r:Resolution {resolution_id: $rid})
                    ON CREATE SET r.title = $title
                    SET r.outcome = coalesce($outcome, r.outcome)
                    SET r.amount = coalesce($amount, r.amount)
                    WITH r
                    MATCH (ai:AgendaItem {item_id: $item_id})
                    MERGE (ai)-[:RESULTED_IN]->(r)
                    """,
                    {
                        "rid": f"res_{item_id}",
                        "title": title[:400],
                        "outcome": outcome,
                        "amount": amounts[0] if amounts else None,
                        "item_id": item_id,
                    }
                )

            # Contract: if vendor pattern present
            if vendor:
                s.run(
                    """
                    MERGE (c:Contract {contract_id: $cid})
                    ON CREATE SET c.vendor = $vendor
                    WITH c
                    MATCH (ai:AgendaItem {item_id: $item_id})
                    MERGE (ai)-[:LINKS_TO]->(c)
                    """,
                    {"cid": f"contract_{item_id}", "vendor": vendor[:200], "item_id": item_id}
                )
    drv.close()
    print("✅ Backfill complete")

if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(lim) 