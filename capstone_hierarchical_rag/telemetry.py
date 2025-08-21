import os
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv  # type: ignore
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False

try:
    import psycopg2
    _HAS_PG = True
except Exception:
    _HAS_PG = False

_DB_CONN = None

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  last_activity_at TIMESTAMPTZ DEFAULT NOW(),
  hashed_ip TEXT,
  user_agent TEXT
);

CREATE TABLE IF NOT EXISTS conversation_events (
  id BIGSERIAL PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  turn_index INT NOT NULL,
  request_id TEXT,
  trace_id TEXT,
  role TEXT NOT NULL, -- 'user' | 'assistant'
  question TEXT,
  answer TEXT,
  citations JSONB,
  vector_count INT,
  graph_count INT,
  -- timings
  dense_ms INT, sparse_ms INT, graph_ms INT, fuse_ms INT, rerank_ms INT, final_llm_ms INT, total_ms INT,
  -- models/config
  model TEXT, rerank_model TEXT, embedding_model TEXT,
  dataset_version TEXT, collection_name TEXT, schema_hash TEXT,
  vector_params JSONB,
  retrieved_ids JSONB,
  scores JSONB,
  -- cost/usage
  token_prompt INT, token_completion INT, token_total INT, cost_usd NUMERIC(12,6),
  -- abuse/security
  rate_limit_status TEXT, captcha_score REAL, blocked_reason TEXT,
  validation_flags JSONB,
  -- privacy
  pii_redaction_applied BOOLEAN, pii_fields_found INT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON conversation_events(session_id, turn_index DESC);
CREATE INDEX IF NOT EXISTS idx_turns_created ON conversation_events(created_at);
"""


def _looks_like_supabase_host(host: Optional[str]) -> bool:
    if not host:
        return False
    return "supabase.co" in host or "supabase" in host


def _ensure_sslmode_require(dsn: str) -> str:
    # Add sslmode=require if not present; needed for Supabase
    if "sslmode=" in dsn:
        return dsn
    delimiter = "&" if "?" in dsn else "?"
    return f"{dsn}{delimiter}sslmode=require"


def _build_dsn_from_env() -> Optional[str]:
    """Construct a Postgres DSN from various possible env var schemes.
    Supports POSTGRES_*, DATABASE_URL, and PG* conventions. Enforces SSL for Supabase.
    """
    # Prefer explicit DSN-style vars first
    dsn = (
        os.getenv("POSTGRES_DSN")
        or os.getenv("DATABASE_URL")
        or os.getenv("SUPABASE_DB_URL")
    )
    if dsn:
        # If the DSN points to Supabase, ensure sslmode=require
        host = None
        try:
            # crude host extraction without importing urllib
            after_at = dsn.split("@", 1)[1] if "@" in dsn else dsn
            host_port_db = after_at.split("/", 1)[0]
            host = host_port_db.split(":")[0]
        except Exception:
            host = None
        if _looks_like_supabase_host(host):
            dsn = _ensure_sslmode_require(dsn)
        return dsn

    # Next, assemble from SUPABASE_* or POSTGRES_* or PG* variables
    host = (
        os.getenv("SUPABASE_HOST")
        or os.getenv("POSTGRES_HOST")
        or os.getenv("PGHOST")
    )
    port = (
        os.getenv("SUPABASE_PORT")
        or os.getenv("POSTGRES_PORT")
        or os.getenv("PGPORT")
        or "5432"
    )
    db = (
        os.getenv("SUPABASE_DBNAME")
        or os.getenv("POSTGRES_DB")
        or os.getenv("POSTGRES_DATABASE")
        or os.getenv("PGDATABASE")
    )
    user = (
        os.getenv("SUPABASE_USER")
        or os.getenv("POSTGRES_USER")
        or os.getenv("POSTGRES_USERNAME")
        or os.getenv("PGUSER")
    )
    password = (
        os.getenv("SUPABASE_PASSWORD")
        or os.getenv("POSTGRES_PASSWORD")
        or os.getenv("PGPASSWORD")
    )

    if host and db and user and password:
        dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        # Enforce SSL if environment requires it or host looks like Supabase
        if os.getenv("PGSSLMODE") == "require" or _looks_like_supabase_host(host):
            dsn = _ensure_sslmode_require(dsn)
        return dsn

    return None


def init_if_configured() -> None:
    dsn = _build_dsn_from_env()
    if not dsn or not _HAS_PG:
        return
    global _DB_CONN
    if _DB_CONN is not None:
        return
    try:
        # psycopg2 reads sslmode from DSN; for Supabase we appended sslmode=require
        _DB_CONN = psycopg2.connect(dsn)
        _DB_CONN.autocommit = True
        with _DB_CONN.cursor() as cur:
            cur.execute(SCHEMA_SQL)
    except Exception as e:
        # Do not block app start if DSN is invalid; fallback to file logging
        _DB_CONN = None
        print(f"[telemetry] Skipping Postgres init: {e}")


def upsert_session(session_id: str, hashed_ip: Optional[str], user_agent: Optional[str]) -> None:
    if not _HAS_PG or _DB_CONN is None:
        return
    try:
        with _DB_CONN.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (session_id, hashed_ip, user_agent)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id)
                DO UPDATE SET last_activity_at = NOW(), hashed_ip = EXCLUDED.hashed_ip, user_agent = EXCLUDED.user_agent
                """,
                (session_id, hashed_ip, user_agent),
            )
    except Exception as e:
        print(f"[telemetry] upsert_session failed: {e}")


def record_event(event: Dict[str, Any]) -> None:
    """Persist an event to Postgres if available; else append to local JSONL."""
    if _HAS_PG and _DB_CONN is not None:
        try:
            with _DB_CONN.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_events (
                        session_id, turn_index, request_id, trace_id, role, question, answer, citations,
                        vector_count, graph_count,
                        dense_ms, sparse_ms, graph_ms, fuse_ms, rerank_ms, final_llm_ms, total_ms,
                        model, rerank_model, embedding_model,
                        dataset_version, collection_name, schema_hash,
                        vector_params, retrieved_ids, scores,
                        token_prompt, token_completion, token_total, cost_usd,
                        rate_limit_status, captcha_score, blocked_reason,
                        validation_flags,
                        pii_redaction_applied, pii_fields_found
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,
                              %s,%s,
                              %s,%s,%s,%s,%s,%s,%s,
                              %s,%s,%s,
                              %s,%s,%s,
                              %s,%s,%s,
                              %s,%s,%s,%s,
                              %s,%s,%s,
                              %s,
                              %s,%s)
                    """,
                    (
                        event.get("session_id"),
                        event.get("turn_index", 0),
                        event.get("request_id"),
                        event.get("trace_id"),
                        event.get("role"),
                        event.get("question"),
                        event.get("answer"),
                        json.dumps(event.get("citations") or []),
                        event.get("vector_count"),
                        event.get("graph_count"),
                        # timings
                        event.get("dense_ms"), event.get("sparse_ms"), event.get("graph_ms"), event.get("fuse_ms"), event.get("rerank_ms"), event.get("final_llm_ms"), event.get("total_ms"),
                        # models/config
                        event.get("model"), event.get("rerank_model"), event.get("embedding_model"),
                        event.get("dataset_version"), event.get("collection_name"), event.get("schema_hash"),
                        json.dumps(event.get("vector_params") or {}),
                        json.dumps(event.get("retrieved_ids") or {}),
                        json.dumps(event.get("scores") or {}),
                        # cost/usage
                        event.get("token_prompt"), event.get("token_completion"), event.get("token_total"), event.get("cost_usd"),
                        # abuse/security
                        event.get("rate_limit_status"), event.get("captcha_score"), event.get("blocked_reason"),
                        json.dumps(event.get("validation_flags") or {}),
                        # privacy
                        event.get("pii_redaction_applied"), event.get("pii_fields_found"),
                    ),
                )
            return
        except Exception as e:
            print(f"[telemetry] record_event failed: {e}")
    # Fallback to file (ensure directory exists, avoid duplicate path segments)
    base = Path(__file__).resolve().parent
    out_dir = base / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "conversation_events.jsonl"
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[telemetry] write file failed: {e}")


def is_connected() -> bool:
    try:
        return _DB_CONN is not None
    except Exception:
        return False


def get_status() -> Dict[str, Any]:
    try:
        return {"postgres_connected": bool(_DB_CONN is not None)}
    except Exception:
        return {"postgres_connected": False} 