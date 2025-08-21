#!/usr/bin/env python3
"""
Dump up to 20,000 rows of id and meeting_date from a Zilliz (Milvus) collection using REST API.
Writes a TSV file with columns: id \t meeting_date (raw) \t meeting_date_normalized.
"""
import os
import json
import re
from datetime import datetime
from pathlib import Path

try:
	from dotenv import load_dotenv  # type: ignore
except Exception:
	def load_dotenv(*args, **kwargs):
		return False

# Load .env from repo root and capstone dir
repo_root = Path(__file__).resolve().parents[1] / ".env"
capstone_env = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=repo_root, override=False)
load_dotenv(dotenv_path=capstone_env, override=False)
load_dotenv()

try:
	import httpx  # type: ignore
except Exception as e:
	raise SystemExit("httpx is required to run this script. pip install httpx python-dotenv")

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION = os.getenv("MILVUS_COLLECTION", "capstone_hybrid_rag")
LIMIT = int(os.getenv("DATE_DUMP_LIMIT", "20000"))
OUT_PATH = Path(__file__).resolve().parent / "meeting_dates_20k.txt"

if not MILVUS_URI or not MILVUS_TOKEN:
	raise SystemExit("MILVUS_URI and MILVUS_TOKEN env vars are required")

QUERY_URL = f"{MILVUS_URI.rstrip('/')}/v2/vectordb/entities/query"
HEADERS = {"Authorization": f"Bearer {MILVUS_TOKEN}", "Content-Type": "application/json"}

# Same normalizer as server (simplified copy)
def _normalize_date_string(raw: str | None) -> str | None:
	if not raw:
		return None
	s = str(raw).strip()
	if not s:
		return None
	# ISO
	m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
	if m:
		return m.group(1)
	# Month name day, year
	m2 = re.search(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", s)
	if m2:
		name = m2.group(1).lower(); day = int(m2.group(2)); year = int(m2.group(3))
		months = {'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,'july':7,'august':8,'september':9,'october':10,'november':11,'december':12}
		if name in months:
			return datetime(year, months[name], day).date().isoformat()
	# MM/DD/YYYY
	m3 = re.search(r"(\d{1,2})\/(\d{1,2})\/(\d{4})", s)
	if m3:
		mm = int(m3.group(1)); dd = int(m3.group(2)); yy = int(m3.group(3))
		return datetime(yy, mm, dd).date().isoformat()
	# YYYY/MM/DD
	m4 = re.search(r"(\d{4})\/(\d{1,2})\/(\d{1,2})", s)
	if m4:
		yy = int(m4.group(1)); mm = int(m4.group(2)); dd = int(m4.group(3))
		return datetime(yy, mm, dd).date().isoformat()
	# Fallback: return raw
	return s

# Extract rows from Zilliz shapes
def _extract_rows(body: object) -> list[dict]:
	if body is None:
		return []
	if isinstance(body, list):
		return [r for r in body if isinstance(r, dict)]
	if isinstance(body, dict):
		data = body.get("data")
		if isinstance(data, list):
			return [r for r in data if isinstance(r, dict)]
		if isinstance(data, dict):
			# Rows-oriented
			rows = data.get("rows") or data.get("data") or data.get("entities")
			if isinstance(rows, list):
				return [r for r in rows if isinstance(r, dict)]
			# Columns-oriented
			cols = data.get("columns")
			if isinstance(cols, list) and cols:
				# Expect a list of { name: str, values: list }
				names: list[str] = []
				values: list[list] = []
				for c in cols:
					name = c.get("name")
					vals = c.get("values")
					if isinstance(name, str) and isinstance(vals, list):
						names.append(name)
						values.append(vals)
				if not names or not values:
					return []
				nrows = max((len(v) for v in values), default=0)
				rows_out: list[dict] = []
				for i in range(nrows):
					row: dict = {}
					for j, nm in enumerate(names):
						arr = values[j]
						row[nm] = arr[i] if i < len(arr) else None
					rows_out.append(row)
				return rows_out
		# Top-level fallbacks
		for k in ("rows", "entities", "results"):
			v = body.get(k)
			if isinstance(v, list):
				return [r for r in v if isinstance(r, dict)]
	return []

async def main() -> int:
	MAX_WINDOW = 16384  # Zilliz REST window constraint: offset+limit <= 16384
	PAGE = 1000
	acc: list[dict] = []
	offset = 0
	async with httpx.AsyncClient(timeout=30) as client:
		while offset < MAX_WINDOW and len(acc) < LIMIT:
			page_limit = min(PAGE, LIMIT - len(acc), MAX_WINDOW - offset)
			payload = {
				"collectionName": COLLECTION,
				"outputFields": ["id", "chunk_id", "meeting_date"],
				"limit": page_limit,
				"offset": offset,
			}
			resp = await client.post(QUERY_URL, json=payload, headers=HEADERS)
			# Write last response raw for inspection
			body = {}
			try:
				resp.raise_for_status()
				body = resp.json() or {}
			except Exception:
				body = {"code": getattr(resp, 'status_code', 'n/a'), "message": resp.text[:500]}
			raw_path = Path(__file__).resolve().parent / "meeting_dates_20k_raw.json"
			with open(raw_path, "w", encoding="utf-8") as rf:
				rf.write(json.dumps(body)[:2_000_000])
			rows = _extract_rows(body)
			if not rows:
				break
			acc.extend(rows)
			offset += page_limit
	# Write TSV
	with open(OUT_PATH, "w", encoding="utf-8") as f:
		f.write("id\tchunk_id\tmeeting_date_raw\tmeeting_date_normalized\n")
		for r in acc[:LIMIT]:
			rid = str(r.get("id") or "")
			cid = str(r.get("chunk_id") or "")
			md = r.get("meeting_date")
			norm = _normalize_date_string(md)
			f.write(f"{rid}\t{cid}\t{str(md) if md is not None else ''}\t{norm or ''}\n")
	print(f"Wrote {len(acc[:LIMIT])} rows to {OUT_PATH}")
	if len(acc) < LIMIT:
		print(f"Note: Zilliz window constraint limited results to {len(acc)} (max window {MAX_WINDOW}).")
	return 0

if __name__ == "__main__":
	import asyncio
	raise SystemExit(asyncio.run(main())) 