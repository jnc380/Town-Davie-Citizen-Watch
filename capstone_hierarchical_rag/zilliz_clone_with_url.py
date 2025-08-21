#!/usr/bin/env python3
import os
import sys
import json
import time
import re
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

try:
	import httpx  # type: ignore
except Exception as e:  # pragma: no cover
	httpx = None

# Load .env similar to server
try:
	from dotenv import load_dotenv  # type: ignore
except Exception:
	load_dotenv = None

if load_dotenv is not None:
	for env_path in [
		os.path.join(os.getcwd(), ".env"),
		os.path.join(os.path.dirname(__file__), ".env"),
		os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
	]:
		if os.path.exists(env_path):
			load_dotenv(dotenv_path=env_path, override=False)
			break

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


def _host_only(uri: str) -> str:
	try:
		without_scheme = uri.split("://", 1)[-1]
		return without_scheme.split("/", 1)[0]
	except Exception:
		return uri


def _parse_json_maybe(val: Any) -> Optional[Dict[str, Any]]:
	if isinstance(val, dict):
		return val
	if isinstance(val, str):
		v = val.strip()
		if not v:
			return None
		try:
			return json.loads(v)
		except Exception:
			return None
	return None


def _extract_last_int(text: str) -> Optional[int]:
	try:
		nums = re.findall(r"(\d+)", text or "")
		if not nums:
			return None
		return int(nums[-1])
	except Exception:
		return None


def _build_novus_url(meeting_id: Optional[str], item_id: Optional[int]) -> Optional[str]:
	if not meeting_id or not item_id:
		return None
	return f"https://davie.novusagenda.com/agendapublic/CoverSheet.aspx?ItemID={item_id}&MeetingID={meeting_id}"


def _build_youtube_url(youtube_id: Optional[str], start_time: Optional[int]) -> Optional[str]:
	if not youtube_id:
		return None
	if start_time and start_time > 0:
		return f"https://youtu.be/{youtube_id}?t={start_time}"
	return f"https://youtu.be/{youtube_id}"


def _derive_url(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
	"""Return (url, source_type). Tries explicit fields first, then metadata, then heuristics."""
	# Direct fields first
	direct_url = row.get("url") or row.get("video_url") or row.get("document_url") or row.get("agenda_url")
	if isinstance(direct_url, list) and direct_url:
		direct_url = direct_url[0]
	if isinstance(direct_url, str) and direct_url.strip():
		return direct_url.strip(), row.get("source_type") or row.get("chunk_type") or "document"

	metadata = _parse_json_maybe(row.get("metadata")) or {}
	# Metadata URL variants
	for k in ("url", "video_url", "document_url", "agenda_url", "minutes_url"):
		v = metadata.get(k)
		if isinstance(v, list) and v:
			v = v[0]
		if isinstance(v, str) and v.strip():
			return v.strip(), row.get("source_type") or metadata.get("source_type") or row.get("chunk_type") or "document"

	# YouTube from fields or metadata
	youtube_id = (
		row.get("youtube_id")
		or metadata.get("youtube_id")
		or metadata.get("yt_id")
		or metadata.get("video_id")
	)
	# start_time can be int or string
	start_time_val = row.get("start_time")
	if start_time_val is None:
		start_time_val = metadata.get("start_time")
	try:
		start_time_int = int(start_time_val) if start_time_val is not None else None
	except Exception:
		start_time_int = None

	yt_url = _build_youtube_url(youtube_id, start_time_int)
	if yt_url:
		return yt_url, "video"

	# Novus agenda heuristic from meeting_id + chunk_id/item_id
	meeting_id = row.get("meeting_id") or (metadata.get("meeting_id") if metadata else None)
	chunk_id = row.get("chunk_id") or (metadata.get("chunk_id") if metadata else None)
	item_id = metadata.get("item_id") if metadata else None
	if not item_id:
		item_id = _extract_last_int(chunk_id or "")
	try:
		item_id_int = int(item_id) if item_id is not None else None
	except Exception:
		item_id_int = None

	novus = _build_novus_url(str(meeting_id) if meeting_id is not None else None, item_id_int)
	if novus:
		# If transcript segment with time also exists, prefer video source_type
		source_type = row.get("source_type") or row.get("chunk_type") or metadata.get("source_type") or "document"
		return novus, source_type

	# Nothing found
	return None, row.get("source_type") or row.get("chunk_type") or metadata.get("source_type") or None


def _headers(token: str) -> Dict[str, str]:
	return {
		"Authorization": f"Bearer {token}",
		"Content-Type": "application/json",
	}


def create_collection(client: httpx.Client, base_uri: str, token: str, collection: str) -> None:
	"""Create target collection with URL and video fields."""
	url = f"{base_uri.rstrip('/')}/v2/vectordb/collections/create"
	schema = {
		"collectionName": collection,
		"schema": {
			"autoId": True,
			"fields": [
				{"name": "id", "dataType": "Int64", "isPrimary": True, "autoId": True},
				{"name": "chunk_id", "dataType": "VarChar", "maxLength": 512},
				{"name": "content", "dataType": "VarChar", "maxLength": 65535},
				{"name": "url", "dataType": "VarChar", "maxLength": 1024},
				{"name": "source_type", "dataType": "VarChar", "maxLength": 32},
				{"name": "meeting_id", "dataType": "VarChar", "maxLength": 64},
				{"name": "meeting_type", "dataType": "VarChar", "maxLength": 128},
				{"name": "meeting_date", "dataType": "VarChar", "maxLength": 32},
				{"name": "chunk_type", "dataType": "VarChar", "maxLength": 64},
				{"name": "start_time", "dataType": "Int64"},
				{"name": "duration", "dataType": "Int64"},
				{"name": "youtube_id", "dataType": "VarChar", "maxLength": 64},
				{"name": "metadata", "dataType": "VarChar", "maxLength": 4096},
				{"name": "embedding", "dataType": "FloatVector", "typeParams": {"dim": 1536}},
				{"name": "sparse_embedding", "dataType": "SparseFloatVector"},
			],
		},
	}
	resp = client.post(url, headers=_headers(token), json=schema, timeout=60)
	if resp.status_code != 200:
		# If already exists, log and continue
		try:
			data = resp.json()
		except Exception:
			data = {"message": resp.text}
		msg = (data.get("message") or data.get("msg") or "").lower()
		if "already exists" in msg:
			logger.info(f"Collection '{collection}' already exists; continuing")
			return
		raise RuntimeError(f"Failed to create collection: {resp.status_code} {resp.text}")
	logger.info(f"Created collection '{collection}'")


def describe_fields(client: httpx.Client, base_uri: str, token: str, collection: str) -> List[str]:
	url = f"{base_uri.rstrip('/')}/v2/vectordb/collections/describe"
	resp = client.post(url, headers=_headers(token), json={"collectionName": collection}, timeout=30)
	resp.raise_for_status()
	root = resp.json()
	data = root.get("data") if isinstance(root, dict) else None
	node = data if isinstance(data, dict) else root
	fields: List[str] = []
	for f in (node.get("schema", {}) or {}).get("fields", []) or []:
		name = f.get("name")
		if name:
			fields.append(name)
	for f in node.get("fields", []) or []:
		name = f.get("name")
		if name and name not in fields:
			fields.append(name)
	return fields


def query_rows(
	client: httpx.Client,
	base_uri: str,
	token: str,
	collection: str,
	output_fields: List[str],
	limit: int,
	offset: int,
) -> List[Dict[str, Any]]:
	url = f"{base_uri.rstrip('/')}/v2/vectordb/entities/query"
	payload = {
		"collectionName": collection,
		"filter": "",
		"outputFields": output_fields,
		"limit": limit,
		"offset": offset,
	}
	resp = client.post(url, headers=_headers(token), json=payload, timeout=120)
	resp.raise_for_status()
	root = resp.json()
	if isinstance(root, list):
		return root
	data = root.get("data")
	if isinstance(data, list):
		return data
	if isinstance(data, dict):
		rows = data.get("rows")
		if isinstance(rows, list):
			return rows
		# Some responses return {data: {columns:[...]}}, handle minimal
		columns = data.get("columns")
		if isinstance(columns, list):
			# Not supported here
			return []
	return []


def insert_rows(
	client: httpx.Client,
	base_uri: str,
	token: str,
	collection: str,
	rows: List[Dict[str, Any]],
) -> None:
	if not rows:
		return
	url = f"{base_uri.rstrip('/')}/v2/vectordb/entities/insert"
	# Try rows-based payload first
	payload_rows = {
		"collectionName": collection,
		"data": {"rows": rows},
	}
	resp = client.post(url, headers=_headers(token), json=payload_rows, timeout=120)
	if resp.status_code == 200:
		return
	# Fallback to columns-based
	columns: Dict[str, List[Any]] = {}
	for row in rows:
		for k, v in row.items():
			columns.setdefault(k, []).append(v)
	payload_cols = {
		"collectionName": collection,
		"data": columns,
	}
	resp2 = client.post(url, headers=_headers(token), json=payload_cols, timeout=120)
	resp2.raise_for_status()


def embed_texts(client: httpx.Client, api_key: str, texts: List[str]) -> List[List[float]]:
	if not texts:
		return []
	url = "https://api.openai.com/v1/embeddings"
	body = {"model": EMBED_MODEL, "input": texts}
	headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
	resp = client.post(url, headers=headers, json=body, timeout=60)
	resp.raise_for_status()
	data = resp.json()
	vecs = [item.get("embedding") for item in data.get("data", [])]
	return vecs


def main() -> None:
	parser = argparse.ArgumentParser(description="Clone Zilliz collection into a new one with URL column populated.")
	parser.add_argument("--source", default=os.getenv("MILVUS_COLLECTION", "capstone_hybrid_rag"))
	parser.add_argument("--target", default="TOWN_OF_DAVIE_RAG_URL")
	parser.add_argument("--batch", type=int, default=10)
	parser.add_argument("--limit", type=int, default=1000000, help="Max rows to copy (safety)")
	args = parser.parse_args()

	base_uri = os.getenv("MILVUS_URI")
	token = os.getenv("MILVUS_TOKEN")
	if not base_uri or not token:
		logger.error("MILVUS_URI and MILVUS_TOKEN env vars are required")
		sys.exit(1)
	if not OPENAI_API_KEY:
		logger.error("OPENAI_API_KEY is required to re-embed content")
		sys.exit(1)

	if httpx is None:
		logger.error("httpx is required to run this script")
		sys.exit(1)

	with httpx.Client(timeout=60) as client:
		logger.info(f"Source collection: {args.source}")
		logger.info(f"Target collection: {args.target}")
		# Create target
		create_collection(client, base_uri, token, args.target)
		target_fields = set(describe_fields(client, base_uri, token, args.target))
		logger.info(f"Target fields: {sorted(target_fields)}")

		# Determine output fields to fetch from source
		source_fields = describe_fields(client, base_uri, token, args.source)
		logger.info(f"Source fields: {source_fields}")
		want_fields = [
			"chunk_id",
			"content",
			"meeting_id",
			"meeting_type",
			"meeting_date",
			"chunk_type",
			"start_time",
			"duration",
			"youtube_id",
			"metadata",
			"url",
			"video_url",
			"document_url",
			"agenda_url",
		]
		ofields = [f for f in want_fields if f in source_fields]
		logger.info(f"Querying source with outputFields={ofields}")

		copied = 0
		offset = 0
		page_size = 100
		pending_rows: List[Dict[str, Any]] = []
		pending_texts: List[str] = []

		while copied < args.limit:
			rows = query_rows(client, base_uri, token, args.source, ofields, limit=page_size, offset=offset)
			if not rows:
				logger.info("No more rows from source.")
				break
			logger.info(f"Fetched {len(rows)} rows at offset {offset}")
			offset += len(rows)

			for r in rows:
				# Derive URL and source_type
				url_val, source_type = _derive_url(r)
				if not url_val:
					url_val = ""
				metadata = _parse_json_maybe(r.get("metadata")) or {}
				# Prepare row without embedding yet
				out: Dict[str, Any] = {}
				def put(name: str, value: Any) -> None:
					if name in target_fields:
						out[name] = value

				put("chunk_id", r.get("chunk_id") or "")
				content_text = r.get("content") or ""
				put("content", content_text)
				put("url", url_val)
				put("source_type", source_type or (r.get("chunk_type") or ""))
				put("meeting_id", str(r.get("meeting_id") or ""))
				put("meeting_type", r.get("meeting_type") or "")
				put("meeting_date", r.get("meeting_date") or "")
				put("chunk_type", r.get("chunk_type") or "")
				# times
				try:
					put("start_time", int(r.get("start_time") or 0))
				except Exception:
					put("start_time", 0)
				try:
					put("duration", int(r.get("duration") or 0))
				except Exception:
					put("duration", 0)
				# youtube id
				yt_id = r.get("youtube_id") or metadata.get("youtube_id") or metadata.get("yt_id") or metadata.get("video_id") or ""
				put("youtube_id", yt_id)
				# metadata raw
				raw_meta = r.get("metadata")
				if isinstance(raw_meta, (dict, list)):
					raw_meta_str = json.dumps(raw_meta, ensure_ascii=False)
				elif isinstance(raw_meta, str):
					raw_meta_str = raw_meta
				else:
					raw_meta_str = json.dumps({}, ensure_ascii=False)
				put("metadata", raw_meta_str)

				pending_rows.append(out)
				pending_texts.append(content_text)

				if len(pending_rows) >= args.batch:
					# embed and insert
					embs = embed_texts(client, OPENAI_API_KEY, pending_texts)
					if len(embs) != len(pending_rows):
						logger.warning("Embedding count mismatch; skipping this batch")
						pending_rows.clear(); pending_texts.clear()
					else:
						for i, emb in enumerate(embs):
							pending_rows[i]["embedding"] = emb
						insert_rows(client, base_uri, token, args.target, pending_rows)
						copied += len(pending_rows)
						logger.info(f"Inserted {len(pending_rows)} rows. Total copied={copied}")
						pending_rows.clear(); pending_texts.clear()
			# backoff a bit
			time.sleep(0.2)

		if pending_rows:
			embs = embed_texts(client, OPENAI_API_KEY, pending_texts)
			if len(embs) == len(pending_rows):
				for i, emb in enumerate(embs):
					pending_rows[i]["embedding"] = emb
				insert_rows(client, base_uri, token, args.target, pending_rows)
				copied += len(pending_rows)
				logger.info(f"Inserted final {len(pending_rows)} rows. Total copied={copied}")

		logger.info("Clone completed.")


if __name__ == "__main__":
	main() 