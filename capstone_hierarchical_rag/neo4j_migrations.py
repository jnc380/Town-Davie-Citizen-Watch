import os
from typing import List
from neo4j import GraphDatabase
from dotenv import load_dotenv

"""
Neo4j schema migration for capstone project.
- Creates unique constraints on ids for core and new labels
- Creates helpful property indexes (no full-text)
- Handles existing non-unique indexes blocking constraint creation
- Does NOT create any data or relationships; safe to run multiple times
"""

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

PROPERTY_INDEXES = [
    ("Meeting", "meeting_date"),
    ("Meeting", "meeting_type"),
    ("Meeting", "title"),
    ("AgendaItem", "title"),
    ("AgendaItem", "description"),
    ("Concept", "name"),
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

def _ensure_unique_constraint(session, label: str, prop: str) -> None:
    # 1) Check for duplicate ids; if duplicates, do NOT attempt unique constraint
    dup_cnt = session.run(
        f"""
        MATCH (n:`{label}`)
        WITH n.{prop} AS id, count(*) AS c
        WHERE id IS NOT NULL AND c > 1
        RETURN count(*) AS num
        """
    ).single()
    num_dup = dup_cnt["num"] if dup_cnt else 0
    if num_dup and num_dup > 0:
        # Ensure a non-unique index exists for performance and exit
        session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.{prop})")
        return
    # 2) Drop any non-unique index on this label.property that is not owned by a constraint
    idx_rows = session.run(
        """
        SHOW INDEXES YIELD name, entityType, labelsOrTypes, properties, owningConstraint
        WHERE entityType = 'NODE' AND labelsOrTypes = [$label] AND properties = [$prop]
        RETURN name, owningConstraint
        """,
        {"label": label, "prop": prop},
    ).data()
    for row in idx_rows:
        if not row.get("owningConstraint"):
            idx_name = row["name"]
            try:
                session.run(f"DROP INDEX `{idx_name}`")
            except Exception:
                pass
    # 3) Create the unique constraint idempotently
    session.run(
        f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.{prop} IS UNIQUE"
    )


def run():
    # Load env from hierarchical capstone first, then fallback
    load_dotenv("capstone_hierarchical_rag/.env", override=False)
    load_dotenv(".env", override=False)
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pw = os.getenv("NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    assert uri and user and pw, "Missing Neo4j env vars"

    drv = GraphDatabase.driver(uri, auth=(user, pw))
    with drv.session(database=db) as s:
        # Unique constraints for ids (robust handling of existing non-unique indexes)
        for label in LABELS_WITH_UNIQUE_IDS:
            id_prop = ID_PROPERTY_BY_LABEL[label]
            _ensure_unique_constraint(s, label, id_prop)
        # Property indexes
        for label, prop in PROPERTY_INDEXES:
            s.run(f"CREATE INDEX idx_{label.lower()}_{prop} IF NOT EXISTS FOR (n:`{label}`) ON (n.{prop})")
    drv.close()
    print("✅ Neo4j schema migration applied: constraints and indexes ensured")

if __name__ == "__main__":
    run() 