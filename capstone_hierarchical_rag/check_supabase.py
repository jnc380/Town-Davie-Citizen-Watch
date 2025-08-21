import json
import sys
from pathlib import Path
import os

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

# Load environment variables from possible locations
try:
    load_dotenv()
    repo_root_env = Path(__file__).resolve().parents[1] / ".env"
    capstone_env = Path(__file__).resolve().parent / ".env"
    if repo_root_env.exists():
        load_dotenv(dotenv_path=repo_root_env, override=False)
    if capstone_env.exists():
        load_dotenv(dotenv_path=capstone_env, override=False)
except Exception:
    pass

# Ensure capstone is importable
sys.path.append(str(Path(__file__).resolve().parent))

import telemetry  # type: ignore


def _looks_like_supabase_host(host: str | None) -> bool:
    return bool(host) and ("supabase.co" in host or "supabase" in host)


def _ensure_sslmode_require(dsn: str) -> str:
    return dsn if "sslmode=" in dsn else (dsn + ("&" if "?" in dsn else "?") + "sslmode=require")


def _get_env(keys: list[str]) -> str | None:
    for k in keys:
        v = os.getenv(k)
        if v:
            return v
    return None


def _build_dsn_from_env_including_supabase() -> str | None:
    dsn = _get_env(["POSTGRES_DSN", "DATABASE_URL", "SUPABASE_DB_URL"])
    if dsn:
        # best effort SSL
        try:
            after_at = dsn.split("@", 1)[1] if "@" in dsn else dsn
            host_port_db = after_at.split("/", 1)[0]
            host = host_port_db.split(":")[0]
        except Exception:
            host = None
        if _looks_like_supabase_host(host):
            dsn = _ensure_sslmode_require(dsn)
        return dsn

    # Try SUPABASE_* split
    host = _get_env(["SUPABASE_HOST", "POSTGRES_HOST", "PGHOST"])
    port = _get_env(["SUPABASE_PORT", "POSTGRES_PORT", "PGPORT"]) or "5432"
    db = _get_env(["SUPABASE_DBNAME", "POSTGRES_DB", "POSTGRES_DATABASE", "PGDATABASE"]) or "postgres"
    user = _get_env(["SUPABASE_USER", "POSTGRES_USER", "PGUSER"]) or "postgres"
    password = _get_env(["SUPABASE_PASSWORD", "POSTGRES_PASSWORD", "PGPASSWORD"]) or ""

    if host and user and db:
        dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        if _looks_like_supabase_host(host) or os.getenv("PGSSLMODE") == "require":
            dsn = _ensure_sslmode_require(dsn)
        return dsn

    return None


# Initialize telemetry if possible; otherwise build a direct connection
telemetry.init_if_configured()
status = telemetry.get_status()
conn = telemetry._DB_CONN if status.get("postgres_connected") else None  # type: ignore

if conn is None:
    dsn = _build_dsn_from_env_including_supabase()
    if not dsn:
        print(json.dumps({"error": "No database configuration found in env"}, indent=2))
        sys.exit(1)
    try:
        import psycopg2  # type: ignore
        conn = psycopg2.connect(dsn)
    except Exception as e:
        print(json.dumps({"error": f"Failed to connect via DSN: {e}"}, indent=2))
        sys.exit(1)

# Ensure schema exists (create if missing)
try:
    with conn.cursor() as cur:  # type: ignore
        cur.execute(telemetry.SCHEMA_SQL)  # type: ignore
    conn.commit()  # type: ignore
except Exception:
    pass


def fetch_all(sql: str, params: tuple | None = None):
    with conn.cursor() as cur:  # type: ignore
        cur.execute(sql, params or ())
        if cur.description is None:
            return []
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]


out = {}
try:
    out["sessions_count"] = fetch_all("SELECT COUNT(*) AS count FROM sessions")[0]["count"]
    out["conversation_events_count"] = fetch_all("SELECT COUNT(*) AS count FROM conversation_events")[0]["count"]
    out["sessions_sample"] = fetch_all(
        "SELECT session_id, created_at, last_activity_at FROM sessions ORDER BY created_at DESC LIMIT 5"
    )
    out["conversation_events_sample"] = fetch_all(
        "SELECT id, session_id, turn_index, role, created_at FROM conversation_events ORDER BY created_at DESC LIMIT 5"
    )
except Exception as e:
    out["error"] = str(e)

print(json.dumps(out, default=str, indent=2)) 