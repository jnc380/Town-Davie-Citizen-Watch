#!/usr/bin/env python3
"""
Milvus-Only RAG System using Zilliz Cloud REST API (URLs-enabled fork)
Uses vector search (dense) and sparse search (BM25) for comprehensive government transparency Q&A
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import asyncio

# OpenAI imports
from openai import AsyncOpenAI

# HTTP client
try:
	import httpx  # type: ignore
except Exception:
	httpx = None  # type: ignore

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
try:
	from dotenv import load_dotenv  # type: ignore
except Exception:
	def load_dotenv(*args, **kwargs):
		return False

# Try loading from current working directory first
load_dotenv()
# Then try repo root and capstone directory explicitly
repo_root_env = Path(__file__).resolve().parents[1] / ".env"
capstone_env = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=repo_root_env, override=False)
load_dotenv(dotenv_path=capstone_env, override=False)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
try:
	logger.setLevel(getattr(logging, _log_level, logging.INFO))
except Exception:
	logger.setLevel(logging.INFO)

# Helpers
def _host_only(url: Optional[str]) -> str:
	if not url:
		return ""
	try:
		return url.split("://", 1)[-1].split("/", 1)[0]
	except Exception:
		return ""

def _extract_last_int(text: Optional[str]) -> Optional[int]:
	if not text:
		return None
	try:
		import re as _re
		nums = _re.findall(r"(\d+)", text)
		return int(nums[-1]) if nums else None
	except Exception:
		return None

def _build_novus_url(meeting_id: Optional[str], item_id: Optional[int]) -> Optional[str]:
	try:
		if not meeting_id or not item_id:
			return None
		return f"https://davie.novusagenda.com/agendapublic/CoverSheet.aspx?ItemID={item_id}&MeetingID={meeting_id}"
	except Exception:
		return None

def _build_youtube_url(youtube_id: Optional[str], start_time: Optional[int]) -> Optional[str]:
	if not youtube_id:
		return None
	try:
		st = int(start_time) if start_time is not None else 0
		return f"https://youtu.be/{youtube_id}?t={st}" if st > 0 else f"https://youtu.be/{youtube_id}"
	except Exception:
		return f"https://youtu.be/{youtube_id}"

def _present(flag: Optional[str]) -> str:
	return "set" if flag else "unset"

def _normalize_date_string(raw: Optional[str]) -> Optional[str]:
	"""Normalize various date strings to YYYY-MM-DD if possible.

	Supports:
	- ISO: YYYY-MM-DD
	- Month name: 'July 23, 2025'
	- MM/DD/YYYY and YYYY/MM/DD
	Also extracts a recognizable date substring from noisy strings.
	"""
	if not raw:
		return None
	s = str(raw).strip()
	if not s:
		return None
	try:
		# Direct ISO yyyy-mm-dd
		m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
		if m:
			return m.group(1)
		# Month name day, year
		m2 = re.search(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", s)
		if m2:
			month_name = m2.group(1).lower()
			day = int(m2.group(2))
			year = int(m2.group(3))
			months = {
				'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
				'july':7,'august':8,'september':9,'october':10,'november':11,'december':12
			}
			if month_name in months:
				return datetime(year, months[month_name], day).date().isoformat()
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
		# Fallback: try parsing whole string
		for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
			try:
				return datetime.strptime(s, fmt).date().isoformat()
			except Exception:
				continue
		return s
	except Exception:
		return s

def _is_iso_ymd(value: Optional[str]) -> bool:
	"""Return True only if value is exactly YYYY-MM-DD (10 chars)."""
	if not value:
		return False
	s = str(value).strip()
	return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s))

def _extract_rows_from_query_response(body: Any) -> List[Dict[str, Any]]:
	"""Handle Zilliz REST response shapes: list, {'data':[...]} or {'data':{'rows':[...]}}, etc."""
	if body is None:
		return []
	if isinstance(body, list):
		return [r for r in body if isinstance(r, dict)]
	if isinstance(body, dict):
		data = body.get("data")
		if isinstance(data, list):
			return [r for r in data if isinstance(r, dict)]
		if isinstance(data, dict):
			rows = data.get("rows") or data.get("data") or data.get("entities")
			if isinstance(rows, list):
				return [r for r in rows if isinstance(r, dict)]
			# Some responses embed at top-level without 'data'
		for key in ("rows", "entities", "results"):
			val = body.get(key)
			if isinstance(val, list):
				return [r for r in val if isinstance(r, dict)]
	return []

@dataclass
class SearchResult:
	chunk_id: str
	content: str
	document_type: str
	meeting_id: str
	meeting_date: str
	meeting_type: str
	hierarchy: str
	section_header: str
	metadata: Dict[str, Any]
	score: float
	search_type: str  # "dense" or "sparse"

class MilvusOnlyRAGSystem:
	def __init__(self):
		# Initialize OpenAI client
		self.openai_api_key = os.getenv("OPENAI_API_KEY")
		if not self.openai_api_key:
			raise ValueError("OPENAI_API_KEY environment variable is required")
		self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
		
		# Milvus/Zilliz configuration
		self.milvus_uri = os.getenv("MILVUS_URI")
		self.milvus_token = os.getenv("MILVUS_TOKEN")
		self.milvus_collection = os.getenv("MILVUS_COLLECTION", "capstone_hybrid_rag")
		if not self.milvus_uri or not self.milvus_token:
			raise ValueError("MILVUS_URI and MILVUS_TOKEN environment variables are required")
		
		# Search configuration
		self.dense_search_limit = int(os.getenv("DENSE_SEARCH_LIMIT", "20"))
		self.sparse_search_limit = int(os.getenv("SPARSE_SEARCH_LIMIT", "20"))
		self.final_result_limit = int(os.getenv("FINAL_RESULT_LIMIT", "10"))
		
		# Reranking configuration
		self.enable_reranking = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
		self.reranking_model = os.getenv("RERANKING_MODEL", "gpt-4o-mini")
		
		# Cache
		self._zilliz_fields_cache: Dict[str, List[str]] = {}
		self._cached_min_date: Optional[str] = None
		self._cached_max_date: Optional[str] = None
		self._date_cache_updated_at: Optional[str] = None
		
		logger.info(f"MilvusOnlyRAGSystem initialized with collection: {self.milvus_collection}")
		logger.info(f"Search limits: dense={self.dense_search_limit}, sparse={self.sparse_search_limit}, final={self.final_result_limit}")
		logger.info(f"Reranking enabled: {self.enable_reranking}")

	async def _zilliz_describe_fields(self, milvus_uri: str, milvus_token: str, collection_name: str) -> List[str]:
		try:
			cache_key = f"{_host_only(milvus_uri)}::{collection_name}"
			if cache_key in self._zilliz_fields_cache:
				return self._zilliz_fields_cache[cache_key]
			url = f"{milvus_uri.rstrip('/')}/v2/vectordb/collections/describe"
			payload = {"collectionName": collection_name}
			if httpx is None:
				return []
			async with httpx.AsyncClient(timeout=15) as client:
				resp = await client.post(url, json=payload, headers={"Authorization": f"Bearer {milvus_token}", "Content-Type": "application/json"})
				resp.raise_for_status()
				body = resp.json() or {}
			fields: List[str] = []
			try:
				data = body.get("data", {}) or {}
				for f in (data.get("schema", {}) or {}).get("fields", []) or []:
					name = f.get("name")
					if isinstance(name, str):
						fields.append(name)
				for f in data.get("fields", []) or []:
					name = f.get("name")
					if isinstance(name, str) and name not in fields:
						fields.append(name)
			except Exception:
				pass
			self._zilliz_fields_cache[cache_key] = fields
			logger.info(f"Zilliz available fields for {collection_name}: {fields}")
			return fields
		except Exception as e:
			logger.warning(f"Unable to describe Zilliz collection '{collection_name}': {e}")
			return []

	async def _zilliz_search(self, query_embedding: List[float], top_k: int, expr: Optional[str] = None) -> List[Dict[str, Any]]:
		milvus_uri = os.getenv("MILVUS_URI")
		milvus_token = os.getenv("MILVUS_TOKEN")
		if not milvus_uri or not milvus_token or not query_embedding:
			return []
		try:
			search_url = f"{milvus_uri.rstrip('/')}/v2/vectordb/entities/search"
			headers = {"Authorization": f"Bearer {milvus_token}", "Content-Type": "application/json"}
			vector_field = "embedding"
			text_field = "content"
			meta_field = "metadata"
			meet_id_field = "meeting_id"
			meet_date_field = "meeting_date"
			meet_type_field = "meeting_type"
			chunk_id_field = "chunk_id"
			title_field = "chunk_id"
			requested_fields = [
				text_field, meta_field, meet_id_field, meet_date_field, meet_type_field, chunk_id_field, title_field,
				"url", "source_type", "youtube_id", "start_time", "duration"
			]
			available = await self._zilliz_describe_fields(milvus_uri, milvus_token, self.milvus_collection)
			ofields = [f for f in requested_fields if f and f in (available or [])]
			logger.info(f"Zilliz ofields for {self.milvus_collection}: {ofields}")
			async def _do_search(use_filter: bool) -> List[Dict[str, Any]]:
				payload = {"collectionName": self.milvus_collection, "data": [query_embedding], "limit": int(top_k), "outputFields": ofields, "annsField": vector_field}
				if use_filter and expr:
					payload["filter"] = expr
				if httpx is None:
					return []
				async with httpx.AsyncClient(timeout=20) as client:
					resp = await client.post(search_url, json=payload, headers=headers)
					resp.raise_for_status()
					result = resp.json() or {}
				data = result.get("data") or []
				parsed: List[Dict[str, Any]] = []
				for hit in data:
					text = hit.get(text_field) or hit.get("text") or hit.get("content") or ""
					md = hit.get(meta_field) or {}
					if isinstance(md, str):
						try:
							md = json.loads(md)
						except Exception:
							md = {}
					# Promote selected fields into metadata for downstream URL building
					for k in ("youtube_id", "start_time", "duration", "source_type", "agenda_url"):
						v = hit.get(k)
						if v is not None and k not in md:
							md[k] = v
					resolved_url = hit.get("url") or md.get("url") or md.get("agenda_url")
					parsed.append({
						"chunk_id": hit.get(chunk_id_field) or hit.get("id") or "",
						"content": text,
						"meeting_id": str(hit.get(meet_id_field) or md.get("meeting_id") or ""),
						"meeting_date": str(hit.get(meet_date_field) or md.get("meeting_date") or ""),
						"meeting_type": str(hit.get(meet_type_field) or md.get("meeting_type") or ""),
						"metadata": {**md, "url": resolved_url} if resolved_url else md,
						"score": float(hit.get("distance") or 0.0),
					})
				return parsed
			dense_hits = await _do_search(use_filter=False)
			return dense_hits
		except Exception as e:
			logger.error(f"Zilliz search failed: {e}")
			return []

	async def get_video_meeting_date_range(self) -> Tuple[Optional[str], Optional[str]]:
		"""Compute min/max meeting_date for items that look like video/transcript entries.

		Heuristic: filter where start_time is not null (present for video segments). Uses Zilliz entities/query.
		Returns ISO-like YYYY-MM-DD strings or (None, None) if unavailable.
		"""
		milvus_uri = os.getenv("MILVUS_URI")
		milvus_token = os.getenv("MILVUS_TOKEN")
		if httpx is None or not milvus_uri or not milvus_token:
			return (None, None)
		query_url = f"{milvus_uri.rstrip('/')}/v2/vectordb/entities/query"
		headers = {"Authorization": f"Bearer {milvus_token}", "Content-Type": "application/json"}
		# Only request fields we need
		output_fields = ["meeting_date", "start_time"]
		all_dates: List[str] = []
		# Pull up to 20,000 rows for broader coverage
		payload = {"collectionName": self.milvus_collection, "filter": "start_time != null", "outputFields": output_fields, "limit": 20000}
		try:
			async with httpx.AsyncClient(timeout=20) as client:
				resp = await client.post(query_url, json=payload, headers=headers)
				resp.raise_for_status()
				body = resp.json() or {}
			data = _extract_rows_from_query_response(body)
			for row in data:
				md = row.get("meeting_date")
				# normalize first, then enforce strict YYYY-MM-DD
				norm = _normalize_date_string(md)
				if _is_iso_ymd(norm):
					all_dates.append(str(norm).strip())
			if not all_dates:
				return (None, None)
			return (min(all_dates), max(all_dates))
		except Exception as e:
			logger.info(f"Video date range query failed or unsupported: {e}")
			return (None, None)

	async def get_collection_meeting_date_range(self) -> Tuple[Optional[str], Optional[str]]:
		"""Compute min/max meeting_date over the entire collection (no filter)."""
		milvus_uri = os.getenv("MILVUS_URI")
		milvus_token = os.getenv("MILVUS_TOKEN")
		if httpx is None or not milvus_uri or not milvus_token:
			return (None, None)
		query_url = f"{milvus_uri.rstrip('/')}/v2/vectordb/entities/query"
		headers = {"Authorization": f"Bearer {milvus_token}", "Content-Type": "application/json"}
		output_fields = ["meeting_date"]
		all_dates: List[str] = []
		payload = {"collectionName": self.milvus_collection, "outputFields": output_fields, "limit": 20000}
		try:
			async with httpx.AsyncClient(timeout=20) as client:
				resp = await client.post(query_url, json=payload, headers=headers)
				resp.raise_for_status()
				body = resp.json() or {}
			data = _extract_rows_from_query_response(body)
			for row in data:
				md = row.get("meeting_date")
				# normalize first, then enforce strict YYYY-MM-DD
				norm = _normalize_date_string(md)
				if _is_iso_ymd(norm):
					all_dates.append(str(norm).strip())
			if not all_dates:
				return (None, None)
			return (min(all_dates), max(all_dates))
		except Exception as e:
			logger.info(f"Collection date range query failed or unsupported: {e}")
			return (None, None)

	async def refresh_collection_date_range(self) -> Tuple[Optional[str], Optional[str]]:
		"""Refresh cached collection date range; fallback to video-only if needed."""
		cmin, cmax = await self.get_collection_meeting_date_range()
		if not (cmin and cmax):
			vmin, vmax = await self.get_video_meeting_date_range()
			cmin, cmax = (vmin, vmax)
		if cmin and cmax:
			self._cached_min_date = cmin
			self._cached_max_date = cmax
			self._date_cache_updated_at = datetime.now().isoformat()
		return (self._cached_min_date, self._cached_max_date)

	async def scan_collection_meeting_dates(self, limit: int = 20000) -> List[str]:
		"""Return all normalized, strict YYYY-MM-DD meeting_date values up to 'limit'."""
		milvus_uri = os.getenv("MILVUS_URI"); milvus_token = os.getenv("MILVUS_TOKEN")
		if httpx is None or not milvus_uri or not milvus_token:
			return []
		query_url = f"{milvus_uri.rstrip('/')}/v2/vectordb/entities/query"
		headers = {"Authorization": f"Bearer {milvus_token}", "Content-Type": "application/json"}
		payload = {"collectionName": self.milvus_collection, "outputFields": ["meeting_date"], "limit": int(limit)}
		try:
			async with httpx.AsyncClient(timeout=30) as client:
				resp = await client.post(query_url, json=payload, headers=headers)
				resp.raise_for_status()
				body = resp.json() or {}
			rows = _extract_rows_from_query_response(body)
			out: List[str] = []
			for r in rows:
				norm = _normalize_date_string(r.get("meeting_date"))
				if _is_iso_ymd(norm):
					out.append(str(norm).strip())
			out.sort()
			return out
		except Exception as e:
			logger.info(f"scan_collection_meeting_dates failed: {e}")
			return []

	async def search_milvus_dense(self, query: str, limit: int = 20) -> List[SearchResult]:
		try:
			embedding_response = await self.openai_client.embeddings.create(model="text-embedding-3-small", input=query)
			query_embedding = embedding_response.data[0].embedding
			results = await self._zilliz_search(query_embedding, limit)
			out: List[SearchResult] = []
			for r in results:
				md = r.get("metadata") or {}
				if isinstance(md, str):
					try:
						md = json.loads(md)
					except Exception:
						md = {}
				if "url" not in md and r.get("url"):
					md["url"] = r.get("url")
				out.append(SearchResult(
					chunk_id=r.get("chunk_id", ""),
					content=r.get("content", ""),
					document_type=r.get("type", ""),
					meeting_id=str(r.get("meeting_id", "")),
					meeting_date=r.get("meeting_date", ""),
					meeting_type=r.get("meeting_type", ""),
					hierarchy=md.get("hierarchy", ""),
					section_header=md.get("section_header", ""),
					metadata=md,
					score=r.get("score", 0.0),
					search_type="dense"
				))
			return out
		except Exception as e:
			logger.error(f"Error in dense search: {e}")
			return []

	async def search_milvus_sparse(self, query: str, limit: int = 20) -> List[SearchResult]:
		try:
			embedding_response = await self.openai_client.embeddings.create(model="text-embedding-3-small", input=query)
			query_embedding = embedding_response.data[0].embedding
			results = await self._zilliz_search(query_embedding, limit * 2)
			out: List[SearchResult] = []
			terms = [t for t in query.lower().split() if len(t) > 2]
			for r in results:
				content_low = (r.get("content", "") or "").lower()
				score = sum(content_low.count(t) for t in terms)
				md = r.get("metadata") or {}
				if isinstance(md, str):
					try:
						md = json.loads(md)
					except Exception:
						md = {}
				if "url" not in md and r.get("url"):
					md["url"] = r.get("url")
				out.append(SearchResult(
					chunk_id=r.get("chunk_id", ""),
					content=r.get("content", ""),
					document_type=r.get("type", ""),
					meeting_id=str(r.get("meeting_id", "")),
					meeting_date=r.get("meeting_date", ""),
					meeting_type=r.get("meeting_type", ""),
					hierarchy=md.get("hierarchy", ""),
					section_header=md.get("section_header", ""),
					metadata=md,
					score=score,
					search_type="sparse"
				))
			# Sort
			out.sort(key=lambda x: x.score, reverse=True)
			return out[:limit]
		except Exception as e:
			logger.error(f"Error in sparse search: {e}")
			return []

	async def hybrid_search(self, query: str, limit: int = None) -> List[SearchResult]:
		try:
			dense_limit = limit or self.dense_search_limit
			sparse_limit = limit or self.sparse_search_limit
			dense_results = await self.search_milvus_dense(query, dense_limit)
			sparse_results = await self.search_milvus_sparse(query, sparse_limit)
			logger.info(f"Dense search returned {len(dense_results)} results")
			logger.info(f"Sparse search returned {len(sparse_results)} results")
			combined: Dict[str, SearchResult] = {}
			for r in dense_results:
				combined[r.chunk_id] = r
			for r in sparse_results:
				combined[r.chunk_id] = r
			results = list(combined.values())
			results.sort(key=lambda x: x.score, reverse=True)
			logger.info(f"Combined search returned {len(results)} unique results")
			if self.enable_reranking and results:
				try:
					return await self._rerank_with_gpt(query, results, self.final_result_limit)
				except Exception as e:
					logger.warning(f"GPT reranking failed: {e}")
					return results[:self.final_result_limit]
			return results[:self.final_result_limit]
		except Exception as e:
			logger.error(f"Error in hybrid search: {e}")
			return []

	async def _rerank_with_gpt(self, query: str, candidates: List[SearchResult], top_k: int) -> List[SearchResult]:
		if not candidates:
			return []
		items = candidates[: min(len(candidates), max(top_k * 3, 12))]
		prompt_items = []
		for c in items:
			preview = (c.content or "").replace("\n", " ").replace("\r", " ").replace("\\", " ").replace('"', "'")
			if len(preview) > 200:
				preview = preview[:200] + "..."
			prompt_items.append({"id": c.chunk_id, "meeting": c.meeting_id, "type": c.document_type, "date": c.meeting_date, "text": preview})
		system = "You are a retrieval reranker. Rank items by usefulness. Return JSON {'ranking': [ids...]} only."
		user = json.dumps({"query": query, "candidates": prompt_items})
		try:
			resp = await self.openai_client.chat.completions.create(model=self.reranking_model, messages=[{"role":"system","content":system},{"role":"user","content":user}], temperature=0.0, response_format={"type":"json_object"}, max_tokens=200)
			content = resp.choices[0].message.content or "{}"
			parsed = json.loads(content)
		except Exception:
			resp = await self.openai_client.chat.completions.create(model=self.reranking_model, messages=[{"role":"system","content":system},{"role":"user","content":user}], temperature=0.0, max_tokens=200)
			content = resp.choices[0].message.content or "{}"
			try:
				parsed = json.loads(content)
			except Exception:
				start = content.find("{"); end = content.rfind("}")
				parsed = json.loads(content[start:end+1]) if start>=0 and end>start else {"ranking": []}
		order = parsed.get("ranking", [])
		id_to_item = {c.chunk_id: c for c in items}
		ranked = [id_to_item[i] for i in order if i in id_to_item]
		seen = set(order)
		for c in items:
			if c.chunk_id not in seen:
				ranked.append(c)
		return ranked[:top_k]

	async def _explain_and_rerank_sources(self, question: str, final_answer: str, sources: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
		debug_payload = {"question": question, "final_answer": final_answer, "sources": sources[: max(1, int(top_k))]}
		try:
			if not sources:
				return {"items": [], "debug": debug_payload}
			# Hard-coded summarizer schema (Option C)
			schema_C = "{\n  \"name\": \"explain_source_relevance\",\n  \"description\": \"Return ONLY sources that directly support the answer with a single concise justification grounded in the excerpt.\",\n  \"parameters\": {\n    \"type\": \"object\",\n    \"properties\": {\n      \"items\": {\n        \"type\": \"array\",\n        \"items\": {\n          \"type\": \"object\",\n          \"properties\": {\n            \"id\": { \"type\": \"string\", \"description\": \"Echo back exactly.\" },\n            \"justification\": {\n              \"type\": \"string\",\n              \"maxLength\": 120,\n              \"description\": \"Single sentence stating the exact fact from the excerpt that supports the answer. Avoid generic phrasing.\"\n            },\n            \"score\": { \"type\": \"integer\", \"minimum\": 0, \"maximum\": 5 }\n          },\n          \"required\": [\"id\", \"justification\", \"score\"],\n          \"additionalProperties\": false\n        },\n        \"minItems\": 0\n      }\n    },\n    \"required\": [\"items\"],\n    \"additionalProperties\": false\n  }\n}"
			sys_msg = (
				"You are a civic information assistant. Your job is to justify the final answer using the provided source excerpts. "
				"Be SPECIFIC and tie each item to the answer with concrete details (dates, ordinance/resolution numbers, dollar amounts, approvals, locations, votes). "
				"STRICTLY AVOID generic phrases like 'Relevant source' or 'Supports the answer'. If an excerpt does not directly support the answer, omit it (or give score 0–1). "
				"Return ONLY JSON matching this schema (echo back ids exactly):\n\n" + schema_C
			)
			user_payload = {
				"question": question,
				"final_answer": final_answer,
				"sources": [{"id": s.get("id"), "excerpt": s.get("excerpt", "")} for s in sources],
				"top_k": max(1, int(top_k)),
			}
			# Dump exact request payload for review
			try:
				from datetime import datetime
				with open("/Users/johncross/RAG-CLASS/capstone_hierarchical_rag/summarizer_io.log", "a", encoding="utf-8") as f:
					f.write(json.dumps({
						"ts": datetime.utcnow().isoformat() + "Z",
						"event": "summarizer_request",
						"system": sys_msg,
						"user": user_payload,
					}, ensure_ascii=False) + "\n")
			except Exception:
				pass
			messages = [{"role":"system","content":sys_msg},{"role":"user","content":json.dumps(user_payload)}]
			try:
				resp = await self.openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.1, response_format={"type":"json_object"}, max_tokens=500)
				content = resp.choices[0].message.content or "{}"
				logger.info(f"LLM explain/rerank response: {content[:500]}")
				# Dump raw response
				try:
					from datetime import datetime
					with open("/Users/johncross/RAG-CLASS/capstone_hierarchical_rag/summarizer_io.log", "a", encoding="utf-8") as f:
						f.write(json.dumps({
							"ts": datetime.utcnow().isoformat() + "Z",
							"event": "summarizer_response_raw",
							"raw": content,
						}, ensure_ascii=False) + "\n")
				except Exception:
					pass
				data = json.loads(content)
			except Exception:
				resp = await self.openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.1, max_tokens=500)
				txt = resp.choices[0].message.content or ""
				logger.info(f"LLM explain/rerank fallback response: {txt[:500]}")
				# Dump fallback raw response
				try:
					from datetime import datetime
					with open("/Users/johncross/RAG-CLASS/capstone_hierarchical_rag/summarizer_io.log", "a", encoding="utf-8") as f:
						f.write(json.dumps({
							"ts": datetime.utcnow().isoformat() + "Z",
							"event": "summarizer_response_fallback_raw",
							"raw": txt,
						}, ensure_ascii=False) + "\n")
				except Exception:
					pass
				# Try to parse the fallback text as JSON (handle function-style wrapper too)
				parsed: Dict[str, Any] = {}
				try:
					parsed = json.loads(txt)
				except Exception:
					start = txt.find("{"); end = txt.rfind("}")
					try:
						parsed = json.loads(txt[start:end+1]) if start >= 0 and end > start else {}
					except Exception:
						parsed = {}
				data = parsed if isinstance(parsed, dict) else {}
			# Accept both top-level {"items": [...]} and function-style {"name":..., "parameters": {"items": [...]}}
			items = []
			if isinstance(data, dict):
				if isinstance(data.get("items"), list):
					items = data.get("items", [])
				elif isinstance(data.get("parameters"), dict):
					params = data.get("parameters", {})
					if isinstance(params.get("items"), list):
						items = params.get("items", [])
					elif isinstance(params.get("properties"), dict) and isinstance(params.get("properties", {}).get("items"), list):
						items = params.get("properties", {}).get("items", [])
			# Normalize fields and enforce id presence; map chunk_id→id; map reason/justification→summary/why
			normalized: List[Dict[str, Any]] = []
			for it in items if isinstance(items, list) else []:
				if not isinstance(it, dict):
					continue
				idv = it.get("id") or it.get("chunk_id") or it.get("source_id")
				if not idv:
					continue
				reason_v = it.get("reason") or it.get("justification")
				summary_v = it.get("summary") or reason_v or "Relevant source."
				why_v = it.get("why") or reason_v or "Supports the answer."
				quote_v = it.get("evidence_quote") or it.get("quote") or ""
				score_v = it.get("score", 3)
				try:
					score_v = int(score_v)
				except Exception:
					score_v = 3
				normalized.append({"id": idv, "summary": summary_v, "why": why_v, "evidence_quote": quote_v, "score": score_v})
			items = normalized
			# Dump normalized parsed items
			try:
				from datetime import datetime
				with open("/Users/johncross/RAG-CLASS/capstone_hierarchical_rag/summarizer_io.log", "a", encoding="utf-8") as f:
					f.write(json.dumps({
						"ts": datetime.utcnow().isoformat() + "Z",
						"event": "summarizer_items_parsed",
						"items": items,
					}, ensure_ascii=False) + "\n")
			except Exception:
				pass
			if not items:
				items = [{"id": s.get("id"), "summary": "Relevant source.", "why": "Supports the answer.", "evidence_quote": (s.get("excerpt", "")[:100] + "...") if len(s.get("excerpt", ""))>100 else s.get("excerpt", ""), "score": 3} for s in sources[:top_k]]
			items.sort(key=lambda x: x.get("score", 0), reverse=True)
			return {"items": items[:top_k], "debug": debug_payload}
		except Exception as e:
			logger.warning(f"explain_and_rerank failed: {e}")
			fallback = [{"id": s.get("id"), "summary": "This source contains relevant information.", "why": "It provides details about the topic discussed.", "evidence_quote": (s.get("excerpt", "")[:100] + "...") if len(s.get("excerpt", ""))>100 else s.get("excerpt", ""), "score": 3} for s in sources[:top_k]]
			return {"items": fallback, "debug": debug_payload}

	async def generate_answer(self, query: str, search_results: List[SearchResult]) -> Dict[str, Any]:
		if not search_results:
			return {"answer": "I couldn't find any relevant information to answer your question. Please try rephrasing your query.", "sources": [], "confidence": 0.0}
		context_parts: List[str] = []
		sources_min: List[Dict[str, Any]] = []
		sources_by_id: Dict[str, Dict[str, Any]] = {}
		def _select_relevant_snippet(text: str, question: str, max_chars: int = 1200) -> str:
			if not text:
				return ""
			q_words = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", question) if len(w) >= 3]
			if not q_words:
				return text[:max_chars]
			sentences = re.split(r"(?<=[\.!?])\s+", text)
			matches: List[int] = []
			for idx, s in enumerate(sentences):
				low = s.lower()
				if any(w in low for w in q_words):
					matches.append(idx)
			if not matches:
				return text[:max_chars]
			# Build a window around matched sentences until max_chars
			selected: List[str] = []
			total = 0
			used = set()
			for mid in matches:
				for off in range(0, 3):  # mid, mid+1, mid+2 for a small context window
					j = mid + off
					if 0 <= j < len(sentences) and j not in used:
						piece = sentences[j]
						if total + len(piece) + 1 > max_chars:
							break
						selected.append(piece)
						total += len(piece) + 1
						used.add(j)
					if total >= max_chars:
						break
				if total >= max_chars:
					break
			return " ".join(selected) if selected else text[:max_chars]
		for i, r in enumerate(search_results[:5]):
			context_parts.append(f"Source {i+1}:\n{r.content}")
			# Build robust URL with fallbacks
			md = r.metadata or {}
			yt_id = md.get("youtube_id")
			try:
				st_val = int(md.get("start_time") or 0)
			except Exception:
				st_val = 0
			yt_url = _build_youtube_url(yt_id, st_val)
			item_id_guess = _extract_last_int(r.chunk_id)
			novus_url = _build_novus_url(r.meeting_id, item_id_guess)
			final_url = md.get("url") or md.get("agenda_url") or yt_url or novus_url
			# Build focused excerpt for LLM summarizer
			excerpt_for_llm = _select_relevant_snippet(r.content, query, max_chars=1200)
			# per-request lookup map
			sources_by_id[r.chunk_id] = {
				"chunk_id": r.chunk_id,
				"meeting_id": r.meeting_id,
				"meeting_date": r.meeting_date,
				"meeting_type": r.meeting_type,
				"document_type": r.document_type,
				"hierarchy": r.hierarchy,
				"section_header": r.section_header,
				"score": r.score,
				"search_type": r.search_type,
				"excerpt": excerpt_for_llm,
				"url": final_url,
			}
			sources_min.append({"id": r.chunk_id, "excerpt": sources_by_id[r.chunk_id]["excerpt"]})
		context = "\n\n".join(context_parts)
		prompt = f"""You are a helpful assistant for the Town of Davie government transparency system.
Answer the user's question based on the provided context from official meeting records and documents.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context. If the context doesn't contain enough information to answer the question, say so. Include specific details from the sources when possible.

Answer:"""
		resp = await self.openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"You are a helpful assistant for government transparency."},{"role":"user","content":prompt}], max_tokens=1000, temperature=0.3)
		answer = resp.choices[0].message.content
		avg_score = sum(r.score for r in search_results[:3]) / min(len(search_results), 3)
		confidence = min(avg_score, 1.0)
		try:
			logger.info("Applying explain and rerank to sources")
			explained = await self._explain_and_rerank_sources(query, answer, sources_min)
			items = explained.get("items") or []
			final_sources: List[Dict[str, Any]] = []
			for it in items:
				idv = it.get("id")
				if not idv or idv not in sources_by_id:
					continue
				base = sources_by_id[idv]
				final_sources.append({
					"chunk_id": idv,
					"url": base.get("url"),
					"meeting_type": base.get("meeting_type"),
					"meeting_date": base.get("meeting_date"),
					"meeting_id": base.get("meeting_id"),
					"summary": it.get("summary"),
					"why": it.get("why"),
					"evidence_quote": it.get("evidence_quote"),
					"score": it.get("score", 0),
				})
			# If LLM produced nothing useful, fall back to top-K base sources
			if not final_sources:
				for sid, base in list(sources_by_id.items())[: max(1, int(5))]:
					final_sources.append({
						"chunk_id": sid,
						"url": base.get("url"),
						"meeting_type": base.get("meeting_type"),
						"meeting_date": base.get("meeting_date"),
						"meeting_id": base.get("meeting_id"),
						"summary": "Relevant source.",
						"why": "Supports the answer.",
						"evidence_quote": base.get("excerpt", "")[:100],
						"score": 3,
					})
			sources_payload = final_sources
		except Exception as e:
			logger.warning(f"Explain and rerank failed: {e}")
		return {"answer": answer, "sources": sources_payload, "confidence": confidence, "search_results_count": len(search_results), "reranking_applied": True}

	async def process_query(self, query: str) -> Dict[str, Any]:
		logger.info(f"Processing query: {query}")
		results = await self.hybrid_search(query)
		answer = await self.generate_answer(query, results)
		return {"query": query, "answer": answer["answer"], "sources": answer["sources"], "confidence": answer["confidence"], "search_results_count": answer.get("search_results_count", 0), "timestamp": datetime.now().isoformat(), "search_types_used": list({r.search_type for r in results})}

# API models
class QueryRequest(BaseModel):
	query: str = Field(...)
	max_results: Optional[int] = Field(10)

class QueryResponse(BaseModel):
	query: str
	answer: str
	sources: List[Dict[str, Any]]
	confidence: float
	search_results_count: int
	timestamp: str
	search_types_used: List[str]

# App
rag_system = MilvusOnlyRAGSystem()
app = FastAPI(title="Town of Davie RAG System (Milvus Only - URLs Enabled)", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def _startup_refresh_dates():
	# Prime cache, then refresh every 6 hours
	try:
		await rag_system.refresh_collection_date_range()
	except Exception as e:
		logger.info(f"Initial date range refresh failed: {e}")
	async def _loop():
		while True:
			try:
				await rag_system.refresh_collection_date_range()
			except Exception as e:
				logger.info(f"Periodic date range refresh failed: {e}")
			await asyncio.sleep(6*60*60)
	asyncio.create_task(_loop())

@app.get("/", response_class=HTMLResponse)
async def root(_: Request):
	# Use cached dates if available; if missing, trigger a refresh once.
	min_date = rag_system._cached_min_date
	max_date = rag_system._cached_max_date
	if not (min_date and max_date):
		try:
			min_date, max_date = await rag_system.refresh_collection_date_range()
		except Exception:
			min_date, max_date = (None, None)
	min_date = min_date or os.getenv("MIN_RECORD_DATE_IN_MILVUS", "Unknown")
	max_date = max_date or os.getenv("MAX_RECORD_DATE_IN_MILVUS", "Unknown")
	html = """
	<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"/><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/><title>Town of Davie Citizen Watch</title><link href=\"https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css\" rel=\"stylesheet\"></head><body class=\"bg-gray-50\"><div class=\"max-w-4xl mx-auto p-4\"><div class=\"rounded-lg shadow bg-white\"><div class=\"px-6 py-4 border-b flex items-center justify-between\"><div><h1 class=\"text-xl font-semibold\">Town of Davie Citizen Watch</h1><p class=\"text-sm text-gray-500\">Ask about meetings, agenda items, decisions, and more.</p></div><a href=\"/api/health\" class=\"text-sm text-blue-600 underline\">Health</a></div><div class=\"px-6 py-4\"><div id=\"chat-messages\" class=\"space-y-3\"><div class=\"bg-blue-50 rounded p-3\"><div class=\"font-medium mb-1\">Welcome</div><div class=\"text-sm\">Try: <span class=\"italic\">What did the council decide about Pine Island Road?</span></div><div class=\"text-xs text-blue-800 mt-2\">This site is currently using public records from <span class=\"font-semibold\">%MIN_DATE%</span> to <span class=\"font-semibold\">%MAX_DATE%</span>.</div></div></div></div><div class=\"px-6 py-4 border-t\"><form id=\"chat-form\" class=\"flex items-center gap-2\" onsubmit=\"event.preventDefault(); sendMessage();\"><input id=\"message-input\" class=\"flex-1 border rounded px-3 py-2\" placeholder=\"Ask a question...\" /><button class=\"bg-blue-600 text-white px-4 py-2 rounded\">Send</button></form></div></div><div class=\"mt-6 text-xs text-gray-500\">Disclaimer: This site summarizes public Town of Davie records using automated tools and may contain errors or omissions. For official and legally binding information, please refer directly to the Town’s published documents or contact the Town Clerk’s office.</div></div><script>async function sendMessage(){const input=document.getElementById('message-input');const msg=(input.value||'').trim();if(!msg)return;input.value='';const box=document.getElementById('chat-messages');const u=document.createElement('div');u.className='bg-green-50 rounded p-3';u.textContent=msg;box.appendChild(u);const t=document.createElement('div');t.className='text-gray-600';t.textContent='Thinking...';box.appendChild(t);box.scrollTop=box.scrollHeight;try{const r=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:msg,max_results:10})});const data=await r.json();t.remove();const ai=document.createElement('div');ai.className='bg-gray-50 rounded p-3 whitespace-pre-wrap';ai.textContent=(data&&data.answer)||'No answer';box.appendChild(ai);const src=(data&&data.sources)||[];if(src.length){const wrap=document.createElement('div');wrap.className='mt-2 text-sm text-gray-700 bg-white border rounded p-3';wrap.innerHTML='<div class="font-semibold mb-1">Sources used to answer your question</div>'+src.map((b,i)=>{const md=(b.meeting_date||'').toString().trim();const tail=/^\d{4}-\d{2}-\d{2}$/.test(md)?md:'';const summary=(b.summary||'').trim();const why=(b.why||'').trim();const s1=summary? (summary.endsWith('.')?summary:summary+'.'):'';const s2=why? (why.endsWith('.')?why:why+'.'):'';const blurb=(s1+' '+s2).trim();const url=b.url||'#';const esc=blurb.replace(/</g,'&lt;').replace(/>/g,'&gt;');return `<div class="mb-2">${i+1}. <a class="text-blue-700 underline" href="${url}" target="_blank" rel="noopener">Click Here</a> <span class="text-gray-500">${tail?('('+tail+')'):''}</span><div class="text-gray-600">${esc}</div></div>`}).join('');box.appendChild(wrap);}box.scrollTop=box.scrollHeight;}catch(e){t.textContent='Error: '+e.message}}</script></body></html>
"""
	html = html.replace('%MIN_DATE%', min_date).replace('%MAX_DATE%', max_date)
	return HTMLResponse(content=html, status_code=200)


@app.post("/api/query", response_model=QueryResponse)
async def query(request: Request, query_request: QueryRequest):
	try:
		result = await rag_system.process_query(query_request.query)
		return result
	except Exception as e:
		logger.error(f"Error in query endpoint: {e}")
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
	return {"status": "healthy", "timestamp": datetime.now().isoformat(), "milvus_configured": bool(rag_system.milvus_uri), "openai_configured": bool(rag_system.openai_api_key)}

@app.get("/api/search")
async def search_only(request: Request, q: str, limit: int = 10):
	try:
		search_results = await rag_system.hybrid_search(q, limit)
		return {"query": q, "results": [{"chunk_id": r.chunk_id, "content": r.content[:500] + "..." if len(r.content) > 500 else r.content, "document_type": r.document_type, "meeting_id": r.meeting_id, "meeting_date": r.meeting_date, "meeting_type": r.meeting_type, "hierarchy": r.hierarchy, "section_header": r.section_header, "score": r.score, "search_type": r.search_type, "url": r.metadata.get("url")} for r in search_results], "count": len(search_results)}
	except Exception as e:
		logger.error(f"Error in search endpoint: {e}")
		raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
	import uvicorn
	port = int(os.getenv("PORT", "8000"))
	uvicorn.run(app, host="0.0.0.0", port=port) 