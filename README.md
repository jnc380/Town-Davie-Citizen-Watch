# Town of Davie Citizen Watch (Capstone)

This is a curated subset of the capstone project intended for public hosting on Vercel. It excludes private data, large assets, and any secrets.

## Included
- `api/index.py`: ASGI entry for Vercel
- `capstone/hybrid_rag_system.py`: FastAPI server (light mode compatible)
- `capstone/telemetry.py`: Telemetry to Supabase Postgres (file fallback locally)
- `capstone/vercel.json`: Vercel configuration
- `capstone/requirements-vercel.txt`: Minimal dependencies for Vercel free tier
- `capstone/agenda_processes/agenda_scraper.py`: Agenda scraper
- `capstone/youtube_processes/youtube_hybrid_downloader.py`: YouTube scraper
- `capstone/youtube_url_mapper.py`: URL mapping helpers
- `capstone/templates/index.html`: Basic UI

## Excluded (on purpose)
- Any `.env` or credentials
- Local datasets, downloads, `tmp/`, `eval/`, videos, and other large assets
- Experimental or heavy code paths (e.g., TF‑IDF) are disabled by `LIGHT_DEPLOYMENT=true`

## Environment Variables (set in Vercel)
- `OPENAI_API_KEY`
- `MILVUS_URI`, `MILVUS_TOKEN`
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
- Supabase Postgres (choose one):
  - `DATABASE_URL` (must include `sslmode=require`), or
  - `SUPABASE_USER`, `SUPABASE_PASSWORD`, `SUPABASE_HOST`, `SUPABASE_PORT`, `SUPABASE_DBNAME`
- `LIGHT_DEPLOYMENT=true`

## Local Run
```
PYTHONPATH=capstone uvicorn api.index:app --host 0.0.0.0 --port 8000
```
- Health: `GET /api/health`
- Telemetry: `GET /api/telemetry/status`
- Search: `POST /api/search` with JSON: `{ "query": "...", "top_k": 5 }`

## Deploy to Vercel
1. Configure env vars in Vercel project (no secrets in repo)
2. Deploy via Vercel dashboard or CLI
3. Verify `/api/health` and `/api/telemetry/status`

## Safe Contribution Workflow
- Branch: create a feature branch (e.g., `feat/xxx`)
- Keep secrets local; never commit `.env`
- Add or modify files that are safe and essential
- If adding a new script, ensure it has no secrets and minimal dependencies
- Test locally, then open a PR

## Security
- No credentials in code or config
- Use Vercel Env for secrets; rotate regularly
- Rate limiting and basic validation are enabled 