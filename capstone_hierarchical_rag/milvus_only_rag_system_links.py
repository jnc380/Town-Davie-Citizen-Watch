#!/usr/bin/env python3
"""
Milvus-Only RAG System using Zilliz Cloud REST API (Vercel-compatible)
Uses vector search (dense) and sparse search (BM25) for comprehensive government transparency Q&A
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
from pathlib import Path
import re
import time
import uuid
from collections import deque, defaultdict
import hashlib
import base64

# OpenAI imports
from openai import AsyncOpenAI

# HTTP client for remote services (used in LIGHT_DEPLOYMENT)
try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

# Try loading from current working directory first
loaded = load_dotenv()
# Then try repo root and capstone directory explicitly
repo_root_env = Path(__file__).resolve().parents[1] / ".env"
capstone_env = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=repo_root_env, override=False)
load_dotenv(dotenv_path=capstone_env, override=False)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Allow overriding log level via env
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
try:
    logger.setLevel(getattr(logging, _log_level, logging.INFO))
except Exception:
    logger.setLevel(logging.INFO)

# Safe logging helpers
def _host_only(url: Optional[str]) -> str:
    if not url:
        return ""
    try:
        after_scheme = url.split("://", 1)[-1]
        return after_scheme.split("/", 1)[0]
    except Exception:
        return ""

def _present(flag: Optional[str]) -> str:
    return "set" if flag else "unset"

@dataclass
class SearchResult:
    """Represents a search result from Milvus"""
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
    """
    Milvus-only RAG system using Zilliz Cloud REST API
    for comprehensive government transparency Q&A
    """
    
    def __init__(self):
        """Initialize connections and clients"""
        # Load environment variables from .env if available
        possible_envs = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent / ".env",
            Path(__file__).resolve().parents[1] / ".env"
        ]
        
        for env_path in possible_envs:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)
                break
        
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
        # Source explain toggle (force on)
        self.enable_source_explain = True

        # Cache for Zilliz describe-fields results
        self._zilliz_fields_cache: Dict[str, List[str]] = {}

        # In-memory video URL backfill (optional)
        self._video_by_chunk_id: Dict[str, Dict[str, Any]] = {}
        self._video_by_meeting_id: Dict[str, Dict[str, Any]] = {}
        self._load_video_lookup()
        
        logger.info(f"MilvusOnlyRAGSystem initialized with collection: {self.milvus_collection}")
        logger.info(f"Search limits: dense={self.dense_search_limit}, sparse={self.sparse_search_limit}, final={self.final_result_limit}")
        logger.info(f"Reranking enabled: {self.enable_reranking}")

    def _expand_query(self, query: str) -> tuple[str, list[str]]:
        """Return an expanded query string and a list of boost terms for sparse scoring.
        Keeps original query intact while appending domain synonyms when helpful.
        """
        q_lower = (query or "").lower()
        boost_terms: list[str] = []
        if "shed" in q_lower or "sheds" in q_lower:
            boost_terms.extend([
                "accessory structure",
                "accessory structures",
                "accessory building",
                "accessory buildings",
                "storage shed",
                "outbuilding",
                "outbuildings",
                "accessory unit",
                "accessory units",
            ])
        if not boost_terms:
            return query, []
        expanded = query.strip() + "; " + ", ".join(boost_terms)
        return expanded, boost_terms

    async def _zilliz_describe_fields(self, milvus_uri: str, milvus_token: str, collection_name: str) -> List[str]:
        """Describe collection fields using Zilliz Cloud REST API"""
        try:
            cache_key = f"{_host_only(milvus_uri)}::{collection_name}"
            if cache_key in self._zilliz_fields_cache:
                return self._zilliz_fields_cache[cache_key]
            
            url = f"{milvus_uri.rstrip('/')}/v2/vectordb/collections/describe"
            payload = {"collectionName": collection_name}
            
            if httpx is None:
                return []
            
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    url, 
                    json=payload, 
                    headers={"Authorization": f"Bearer {milvus_token}", "Content-Type": "application/json"}
                )
                resp.raise_for_status()
                body = resp.json() or {}
            
            fields: List[str] = []
            try:
                data = body.get("data", {}) or {}
                schema_fields = (data.get("schema", {}) or {}).get("fields", []) or []
                flat_fields = data.get("fields", []) or []
                for f in (schema_fields + flat_fields):
                    name = f.get("name")
                    if isinstance(name, str):
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
        """Search Zilliz/Milvus over HTTPS using the VectorDB REST API"""
        milvus_uri = os.getenv("MILVUS_URI")
        milvus_token = os.getenv("MILVUS_TOKEN")
        
        if not milvus_uri or not milvus_token or not query_embedding:
            logger.debug(
                f"_zilliz_search skipped: uri={_present(milvus_uri)}, token={_present(milvus_token)}, emb={'set' if query_embedding else 'unset'}"
            )
            return []
        
        try:
            def _resolve_url_from_metadata(md: Optional[Dict[str, Any]]) -> str:
                """Best-effort URL resolver from metadata.
                Tries multiple keys and constructs YouTube timestamp links when possible.
                Returns empty string if nothing can be resolved."""
                if not md or not isinstance(md, dict):
                    return ""
                # Common direct keys
                for k in ["url", "agenda_url", "document_url", "source_url", "video_url", "pdf_url", "link", "href"]:
                    val = md.get(k)
                    if isinstance(val, str) and val.strip():
                        base = val.strip()
                        # Append timestamp if applicable later
                        return base
                # Links array variants
                links = md.get("links") or md.get("urls") or md.get("references")
                if isinstance(links, list) and links:
                    for v in links:
                        if isinstance(v, str) and v.strip():
                            return v.strip()
                        if isinstance(v, dict):
                            for kk in ["url", "href", "link"]:
                                vv = v.get(kk)
                                if isinstance(vv, str) and vv.strip():
                                    return vv.strip()
                # YouTube id construction
                yt_id = md.get("youtube_id") or md.get("yt_id") or md.get("video_id")
                if isinstance(yt_id, str) and yt_id.strip():
                    return f"https://youtu.be/{yt_id.strip()}"
                return ""

            search_url = f"{milvus_uri.rstrip('/')}/v2/vectordb/entities/search"
            headers = {
                "Authorization": f"Bearer {milvus_token}",
                "Content-Type": "application/json",
            }
            
            # Hardcoded field mapping for serverless simplicity
            vector_field = "embedding"
            text_field = "content"
            # Field names in Zilliz (as shown by describe):
            # ['id','content','chunk_id','meeting_id','meeting_type','meeting_date','chunk_type','start_time','duration','metadata','embedding','sparse_embedding']
            meta_field = "metadata"
            meet_id_field = "meeting_id"
            meet_date_field = "meeting_date"
            meet_type_field = "meeting_type"
            chunk_id_field = "chunk_id"
            title_field = "chunk_id"
            
            # Request common fields plus potential video/time fields if present in schema
            requested_fields = [
                text_field,
                meta_field,
                meet_id_field,
                meet_date_field,
                meet_type_field,
                chunk_id_field,
                title_field,
                "start_time",
                "duration",
                "video_url",
                "url",
                "agenda_url",
                "youtube_id",
                "yt_id",
                "video_id",
            ]
            
            # Filter output fields to only those that exist
            available = await self._zilliz_describe_fields(milvus_uri, milvus_token, self.milvus_collection)
            ofields = [f for f in requested_fields if f and f in (available or [])]
            
            async def _do_search(use_filter: bool) -> Tuple[List[Dict[str, Any]], List[Any]]:
                payload = {
                    "collectionName": self.milvus_collection,
                    "data": [query_embedding],
                    "limit": int(top_k),
                    "outputFields": ofields,
                    "annsField": vector_field,
                }
                if use_filter and expr:
                    payload["filter"] = expr
                
                logger.info(
                    f"Zilliz search host={_host_only(milvus_uri)} collection={self.milvus_collection} field={vector_field} k={top_k} expr={'set' if (use_filter and expr) else 'unset'} ofields={ofields}"
                )
                
                if httpx is None:
                    logger.warning("httpx not available; cannot call Zilliz HTTP API")
                    return [], []
                
                ids_local: List[Any] = []
                
                async def _call_once() -> Dict[str, Any]:
                    async with httpx.AsyncClient(timeout=20) as client:
                        resp = await client.post(search_url, json=payload, headers=headers)
                        logger.debug(f"Zilliz status={resp.status_code}")
                        resp.raise_for_status()
                        return resp.json() or {}
                
                try:
                    result_inner = await _call_once()
                except Exception as _e:
                    logger.warning(f"Zilliz call error: {_e}")
                    return [], []
                
                try:
                    code = result_inner.get("code")
                    msg = result_inner.get("message")
                    if code is not None and code != 0:
                        logger.error(f"Zilliz API error code={code} message={msg}")
                        return [], []
                    if "data" not in result_inner:
                        logger.warning(f"Zilliz response missing 'data'; message={msg}")
                        return [], []
                except Exception:
                    pass
                
                parsed: List[Dict[str, Any]] = []
                for hit in (result_inner.get("data") or []):
                    if "id" in hit:
                        ids_local.append(hit.get("id"))
                    
                    text = (hit.get(text_field) if text_field else None) or hit.get("text") or hit.get("content") or ""
                    md = (hit.get(meta_field) if meta_field else None) or hit.get("metadata")
                    
                    if isinstance(md, str):
                        try:
                            md = json.loads(md)
                        except Exception:
                            pass
                    
                    # Compute a safe, clickable url without affecting ranking
                    base_url = _resolve_url_from_metadata(md)
                    if not base_url:
                        for k in ("video_url", "url", "agenda_url"):
                            v = hit.get(k)
                            if isinstance(v, str) and v.strip():
                                base_url = v.strip()
                                break
                    # Backfill via in-memory lookup if needed
                    if not base_url:
                        fb_url, fb_st = self._backfill_video_url(md, hit)
                        if fb_url:
                            base_url = fb_url
                            if fb_st is not None and fb_st >= 0:
                                # We'll apply timestamp below
                                hit["start_time"] = fb_st

                    yt = base_url if ("youtube.com" in base_url or "youtu.be" in base_url) else ""
                    st = None
                    try:
                        st = (md or {}).get("start_time")
                        if st is None:
                            st = hit.get("start_time")
                        st = int(float(st)) if st is not None else None
                    except Exception:
                        st = None
                    if yt and st and st > 0:
                        sep = "&" if "?" in yt else "?"
                        direct_url = f"{yt}{sep}t={st}s"
                    else:
                        direct_url = base_url or ""
                    
                    item = {
                        "title": (hit.get(title_field) if title_field else None) or (md or {}).get("title") or (hit.get(chunk_id_field) if chunk_id_field else None) or hit.get("chunk_id") or "Result",
                        "content": text,
                        "meeting_id": (hit.get(meet_id_field) if meet_id_field else None) or (md or {}).get("meeting_id"),
                        "meeting_date": (hit.get(meet_date_field) if meet_date_field else None) or (md or {}).get("meeting_date"),
                        "meeting_type": (hit.get(meet_type_field) if meet_type_field else None) or (md or {}).get("meeting_type"),
                        "chunk_id": (hit.get(chunk_id_field) if chunk_id_field else None) or (md or {}).get("chunk_id"),
                        "type": (md or {}).get("type") or "document",
                        "url": direct_url or None,
                        "score": hit.get("distance") or hit.get("score"),
                    }
                    parsed.append(item)
                
                return parsed, ids_local
            
            # First attempt: with filter if provided
            hits, ids = await _do_search(use_filter=True)
            if not hits and expr:
                logger.info("Zilliz: no hits with filter; retrying without filter")
                hits, ids = await _do_search(use_filter=False)
            
            hits_len = len(hits)
            logger.info(f"Zilliz hits={hits_len}")
            
            if hits_len:
                prev = (hits[0].get("content") or "")[:120].replace("\n", " ")
                logger.debug(f"Zilliz top title='{hits[0].get('title')}' preview='{prev}'")
            
            return hits
            
        except Exception as e:
            logger.error(f"Zilliz search error: {e}")
            return []

    async def search_milvus_dense(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Perform dense vector search using Milvus"""
        try:
            # Generate embedding for query (no hardcoded expansions)
            embedding_response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = embedding_response.data[0].embedding
            
            # Perform vector search
            results = await self._zilliz_search(query_embedding, limit)
            
            search_results = []
            for result in results:
                try:
                    metadata = result.get("metadata", {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except Exception:
                            metadata = {}
                    # Propagate flat url field into metadata for downstream URL generation
                    if not isinstance(metadata, dict) or metadata is None:
                        metadata = {}
                    if (not metadata.get("url")) and result.get("url"):
                        metadata["url"] = result.get("url")
                    # Propagate video/time related returned fields into metadata for reliable link building
                    for k in ("start_time", "duration", "video_url", "agenda_url", "youtube_id", "yt_id", "video_id"):
                        if (k not in metadata or metadata.get(k) in (None, "")) and result.get(k) is not None:
                            metadata[k] = result.get(k)
                    
                    search_results.append(SearchResult(
                        chunk_id=result.get("chunk_id", ""),
                        content=result.get("content", ""),
                        document_type=result.get("type", ""),
                        meeting_id=result.get("meeting_id", ""),
                        meeting_date=result.get("meeting_date", ""),
                        meeting_type=result.get("meeting_type", ""),
                        hierarchy=metadata.get("hierarchy", ""),
                        section_header=metadata.get("section_header", ""),
                        metadata=metadata,
                        score=result.get("score", 0.0),
                        search_type="dense"
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing search result: {e}")
                    continue
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in dense search: {e}")
            return []

    def _extract_prf_terms(self, results: List[SearchResult], max_terms: int = 8) -> List[str]:
        """Extract frequent informative terms from top results for pseudo-relevance feedback.
        Very simple tokenizer: lowercase, split on non-letters, drop short/common tokens.
        """
        try:
            import re
            from collections import Counter
            text = " ".join((r.content or "") for r in results[:5])
            tokens = re.split(r"[^a-zA-Z]+", text.lower())
            # Minimal stoplist to keep this lightweight and maintainable
            stop = {"the","and","or","of","to","a","in","for","on","is","are","was","were","with","by","at","as","that","this","it","be","an","from","we","you","they","i","he","she","but","not","have","has","had","their","there","which","will","can"}
            cand = [t for t in tokens if len(t) > 3 and t not in stop]
            counts = Counter(cand)
            return [w for w,_ in counts.most_common(max_terms)]
        except Exception:
            return []

    async def search_milvus_sparse(self, query: str, limit: int = 20, prf_terms: Optional[List[str]] = None) -> List[SearchResult]:
        """Perform sparse search using text-based approach over vector candidates"""
        try:
            # Generate embedding for query (used to fetch a larger candidate pool)
            embedding_response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = embedding_response.data[0].embedding
            
            # Fetch larger pool to allow effective sparse re-scoring
            pool = max(limit * 6, 60)
            results = await self._zilliz_search(query_embedding, pool)
            
            # Build term list purely from the query plus dynamic PRF terms
            base_terms = [t for t in query.lower().split() if len(t) > 1]
            prf_terms = prf_terms or []
            query_terms = base_terms + [t for t in prf_terms if t not in base_terms]
            
            # Calculate simple term-frequency score and map back to SearchResult
            search_results = []
            for result in results:
                try:
                    content = (result.get("content", "") or "").lower()
                    tf_score = sum(content.count(term) for term in query_terms) if query_terms else 0
                    metadata = result.get("metadata", {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except Exception:
                            metadata = {}
                    # Propagate flat url field into metadata for downstream URL generation
                    if not isinstance(metadata, dict) or metadata is None:
                        metadata = {}
                    if (not metadata.get("url")) and result.get("url"):
                        metadata["url"] = result.get("url")
                    # Propagate video/time related returned fields into metadata for reliable link building
                    for k in ("start_time", "duration", "video_url", "agenda_url", "youtube_id", "yt_id", "video_id"):
                        if (k not in metadata or metadata.get(k) in (None, "")) and result.get(k) is not None:
                            metadata[k] = result.get(k)
                    
                    search_results.append(SearchResult(
                        chunk_id=result.get("chunk_id", ""),
                        content=result.get("content", ""),
                        document_type=result.get("type", ""),
                        meeting_id=result.get("meeting_id", ""),
                        meeting_date=result.get("meeting_date", ""),
                        meeting_type=result.get("meeting_type", ""),
                        hierarchy=metadata.get("hierarchy", ""),
                        section_header=metadata.get("section_header", ""),
                        metadata=metadata,
                        score=float(tf_score),
                        search_type="sparse"
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing sparse search result: {e}")
                    continue
            
            # Sort by score descending and keep top 'limit'
            search_results.sort(key=lambda x: x.score, reverse=True)
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in sparse search: {e}")
            return []

    async def hybrid_search(self, query: str, limit: int = None) -> List[SearchResult]:
        """Perform hybrid search combining dense and sparse results"""
        try:
            # Use provided limit or default limits
            dense_limit = limit or self.dense_search_limit
            sparse_limit = limit or self.sparse_search_limit
            
            # First run dense to get candidates for PRF
            dense_results = await self.search_milvus_dense(query, dense_limit)
            prf_terms = self._extract_prf_terms(dense_results)
            # Then run sparse with dynamic PRF terms
            sparse_results = await self.search_milvus_sparse(query, sparse_limit, prf_terms=prf_terms)
            
            logger.info(f"Dense search returned {len(dense_results)} results")
            logger.info(f"Sparse search returned {len(sparse_results)} results")
            
            # Combine and deduplicate results
            all_results = {}
            
            # Add dense results
            for result in dense_results:
                all_results[result.chunk_id] = result
            
            # Add sparse results (sparse results get priority if duplicate)
            for result in sparse_results:
                all_results[result.chunk_id] = result
            
            # Convert back to list and sort by score
            combined_results = list(all_results.values())
            combined_results.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Combined search returned {len(combined_results)} unique results")
            
            # Apply reranking if enabled
            if self.enable_reranking and combined_results:
                try:
                    logger.info(f"Applying GPT reranking to {len(combined_results)} results")
                    reranked_results = await self._rerank_with_gpt(query, combined_results, self.final_result_limit)
                    return reranked_results
                except Exception as e:
                    logger.warning(f"GPT reranking failed, using original results: {e}")
                    return combined_results[:self.final_result_limit]
            else:
                return combined_results[:self.final_result_limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    async def _rerank_with_gpt(self, query: str, candidates: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Use GPT to rerank a list of candidates. Returns top_k items."""
        if not candidates:
            return []
        
        # Build a compact list of chunks
        items = candidates[: min(len(candidates), max(top_k * 3, 12))]
        prompt_items = []
        
        for i, c in enumerate(items, 1):
            preview = (c.content or "")
            preview = preview.replace("\n", " ").replace("\r", " ")
            # guard against unescaped quotes/backslashes to reduce JSON parse errors
            preview = preview.replace("\\", " ")
            preview = preview.replace('"', "'")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            
            prompt_items.append({
                "id": c.chunk_id,
                "meeting": c.meeting_id,
                "type": c.document_type,
                "date": c.meeting_date,
                "text": preview
            })
        
        system = (
            "You are a retrieval reranker. Rank items by how useful they are to answer the user's query. "
            "Return only valid JSON with a single key 'ranking' as an array of item ids in best-first order."
        )
        user = json.dumps({"query": query, "candidates": prompt_items})
        
        try:
            async def _call():
                return await self.openai_client.chat.completions.create(
                    model=self.reranking_model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=200
                )
            
            try:
                resp = await _call()
                content = resp.choices[0].message.content or "{}"
                try:
                    parsed = json.loads(content)
                except Exception:
                    # Try to extract JSON object substring
                    start = content.find("{")
                    end = content.rfind("}")
                    if start >= 0 and end > start:
                        parsed = json.loads(content[start:end+1])
                    else:
                        raise
            except Exception:
                # One retry without response_format
                resp = await self.openai_client.chat.completions.create(
                    model=self.reranking_model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.0,
                    max_tokens=200
                )
                content = resp.choices[0].message.content or "{}"
                try:
                    parsed = json.loads(content)
                except Exception:
                    start = content.find("{")
                    end = content.rfind("}")
                    if start >= 0 and end > start:
                        parsed = json.loads(content[start:end+1])
                    else:
                        parsed = {"ranking": []}
            
            order = parsed.get("ranking", [])
            id_to_item = {c.chunk_id: c for c in items}
            ranked = [id_to_item[i] for i in order if i in id_to_item]
            
            # Append any missing to fill
            seen = set(order)
            for c in items:
                if c.chunk_id not in seen:
                    ranked.append(c)
            
            return ranked[:top_k]
            
        except Exception as e:
            logger.warning(f"GPT rerank failed, falling back to original scores: {e}")
            # Sort by score and return top_k
            items_sorted = sorted(items, key=lambda x: x.score, reverse=True)
            return items_sorted[:top_k]

    async def _explain_and_rerank_sources(self, question: str, final_answer: str, sources: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Use GPT to explain why each source is relevant and rerank them."""
        debug_payload = {"question": question, "final_answer": final_answer, "sources": sources[: max(1, int(top_k))]}
        
        try:
            if not sources:
                return {"items": [], "debug": debug_payload}
            
            logger.info(f"explain_and_rerank sources count: {len(sources)}")
            
            # Answer-aware prompt: extract answer facts, score sources by support, return citizen-friendly blurbs with evidence
            sys_msg = (
                "You are a civic information assistant. Explain why each source is relevant to the final answer and score it. "
                "Return ONLY JSON that strictly conforms to the following schema. Do not include any extra keys or text.\n\n"
                "SCHEMA:\n"
                "{\n"
                "  \"name\": \"explain_source_relevance\",\n"
                "  \"description\": \"Explain why each source is relevant to the civic QA answer and score it.\",\n"
                "  \"parameters\": {\n"
                "    \"type\": \"object\",\n"
                "    \"properties\": {\n"
                "      \"items\": {\n"
                "        \"type\": \"array\",\n"
                "        \"description\": \"List of relevant sources with explanations and scores\",\n"
                "        \"items\": {\n"
                "          \"type\": \"object\",\n"
                "          \"properties\": {\n"
                "            \"chunk_id\": { \"type\": \"string\", \"description\": \"Original chunk_id if provided\" },\n"
                "            \"url\": { \"type\": \"string\", \"description\": \"URL of the source document\" },\n"
                "            \"meeting_type\": { \"type\": \"string\", \"description\": \"Type of meeting (e.g., Town Council, Planning & Zoning)\" },\n"
                "            \"meeting_date\": { \"type\": \"string\", \"description\": \"Date of the meeting\" },\n"
                "            \"meeting_id\": { \"type\": \"string\", \"description\": \"Meeting ID from the source system\" },\n"
                "            \"summary\": { \"type\": \"string\", \"description\": \"Concise, citizen-friendly explanation (under 100 chars) of what action/detail this source shows\" },\n"
                "            \"why\": { \"type\": \"string\", \"description\": \"Explanation of why this action/detail matters for the question, in plain English\" },\n"
                "            \"evidence_quote\": { \"type\": \"string\", \"description\": \"Short verbatim quote from the source containing the supporting detail\" },\n"
                "            \"score\": { \"type\": \"number\", \"description\": \"Score 0–5 of how directly this source supports the question\" }\n"
                "          },\n"
                "          \"required\": [\n"
                "            \"chunk_id\",\n"
                "            \"url\",\n"
                "            \"summary\",\n"
                "            \"why\",\n"
                "            \"evidence_quote\",\n"
                "            \"score\"\n"
                "          ]\n"
                "        }\n"
                "      }\n"
                "    },\n"
                "    \"required\": [\"items\"]\n"
                "  }\n"
                "}\n\n"
                "Instructions: Only include truly relevant sources (irrelevant sources should be omitted or receive very low scores). "
                "Keep 'summary' under 100 characters, and include a short verbatim evidence_quote from the provided source text."
            )
            
            user_payload = {
                "question": question,
                "final_answer": final_answer,
                "sources": [{"chunk_id": s.get("chunk_id"), "url": s.get("url"), "excerpt": s.get("excerpt", ""), "meeting_type": s.get("meeting_type"), "meeting_date": s.get("meeting_date")} for s in sources],
                "top_k": max(1, int(top_k)),
            }
            
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": json.dumps(user_payload)}
            ]
            
            try:
                # Prefer broadly-supported json_object format
                resp = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    max_tokens=400,
                )
                content = resp.choices[0].message.content or "{}"
                logger.info(f"LLM explain/rerank response: {content[:500]}")
                data = json.loads(content)
            except Exception:
                # Fallback: content-specific items, no generic phrasing
                items = []
                def _mk_item(s: Dict[str, Any]) -> Dict[str, Any]:
                    excerpt = (s.get("excerpt") or "").strip()
                    # Use first sentence or first 100 chars as summary
                    summ = excerpt.split(". ")[0][:100]
                    why_terms = []
                    try:
                        q_low = (question or "").lower()
                        cand = [w for w in re.split(r"[^a-zA-Z]+", q_low) if len(w) > 3]
                        for w in cand:
                            if w in (excerpt or "").lower():
                                why_terms.append(w)
                                if len(why_terms) >= 3:
                                    break
                    except Exception:
                        pass
                    why_text = ("Mentions: " + ", ".join(why_terms)) if why_terms else "Directly discusses the topic asked."
                    return {
                        "chunk_id": s.get("chunk_id") or "",
                        "url": s.get("url") or "",
                        "meeting_type": s.get("meeting_type") or "",
                        "meeting_date": s.get("meeting_date") or "",
                        "meeting_id": s.get("meeting_id") or "",
                        "summary": summ,
                        "why": why_text,
                        "evidence_quote": excerpt[:160] + ("..." if len(excerpt) > 160 else ""),
                            "score": 3
                    }
                for s in sources[: max(1, int(top_k))]:
                    items.append(_mk_item(s))
                data = {"items": items}
            
            items = data.get("items", [])
            # Sort by score descending
            items.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return {"items": items[:top_k], "debug": debug_payload}
            
        except Exception as e:
            logger.warning(f"explain_and_rerank failed: {e}")
            # Fallback: content-specific items, no generic phrasing
            fallback_items = []
            for s in sources[:top_k]:
                excerpt = (s.get("excerpt") or "").strip()
                summ = excerpt.split(". ")[0][:100]
                why_terms = []
                try:
                    q_low = (question or "").lower()
                    cand = [w for w in re.split(r"[^a-zA-Z]+", q_low) if len(w) > 3]
                    for w in cand:
                        if w in (excerpt or "").lower():
                            why_terms.append(w)
                            if len(why_terms) >= 3:
                                break
                except Exception:
                    pass
                why_text = ("Mentions: " + ", ".join(why_terms)) if why_terms else "Directly discusses the topic asked."
                fallback_items.append({
                    "chunk_id": s.get("chunk_id") or "",
                    "url": s.get("url") or "",
                    "meeting_type": s.get("meeting_type") or "",
                    "meeting_date": s.get("meeting_date") or "",
                    "meeting_id": s.get("meeting_id") or "",
                    "summary": summ,
                    "why": why_text,
                    "evidence_quote": excerpt[:160] + ("..." if len(excerpt) > 160 else ""),
                    "score": 3
                })
            return {"items": fallback_items, "debug": debug_payload}

    async def generate_answer(self, query: str, search_results: List[SearchResult]) -> Dict[str, Any]:
        """Generate answer using OpenAI based on search results"""
        try:
            if not search_results:
                return {
                    "answer": "I couldn't find any relevant information to answer your question. Please try rephrasing your query.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Prepare context from search results
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results[:5]):  # Use top 5 results
                context_parts.append(f"Source {i+1}:\n{result.content}")
                # Derive a stable URL from metadata (and timestamp if present)
                md = result.metadata or {}
                # Reuse resolver to get base URL
                def _resolve_url_from_metadata_local(md_local: Optional[Dict[str, Any]]) -> str:
                    if not md_local or not isinstance(md_local, dict):
                        return ""
                    for k in ["url", "agenda_url", "document_url", "source_url", "video_url", "pdf_url", "link", "href"]:
                        val = md_local.get(k)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
                    links = md_local.get("links") or md_local.get("urls") or md_local.get("references")
                    if isinstance(links, list) and links:
                        for v in links:
                            if isinstance(v, str) and v.strip():
                                return v.strip()
                            if isinstance(v, dict):
                                for kk in ["url", "href", "link"]:
                                    vv = v.get(kk)
                                    if isinstance(vv, str) and vv.strip():
                                        return vv.strip()
                    yt_id = md_local.get("youtube_id") or md_local.get("yt_id") or md_local.get("video_id")
                    if isinstance(yt_id, str) and yt_id.strip():
                        return f"https://youtu.be/{yt_id.strip()}"
                    return ""

                base_url = _resolve_url_from_metadata_local(md)
                # Backfill if missing
                if not base_url:
                    fb_url, fb_st = self._backfill_video_url(md, None)
                    if fb_url:
                        base_url = fb_url
                        if fb_st is not None and fb_st >= 0:
                            md["start_time"] = fb_st
                start_time = md.get("start_time")
                direct_url = base_url
                try:
                    if base_url and ("youtube.com" in base_url or "youtu.be" in base_url) and start_time:
                        st = int(float(start_time))
                        if st > 0:
                            sep = "&" if "?" in base_url else "?"
                            direct_url = f"{base_url}{sep}t={st}s"
                except Exception:
                    pass
                
                # Guarantee a clickable fallback if still missing
                if not (direct_url or base_url):
                    # Internal debug fallback to inspect the chunk if external URL unavailable
                    fallback = f"/api/search?q={result.chunk_id or result.meeting_id or result.meeting_date}"
                    direct_url = fallback
                    base_url = fallback
                
                sources.append({
                    "chunk_id": result.chunk_id,
                    "meeting_id": result.meeting_id,
                    "meeting_date": result.meeting_date,
                    "meeting_type": result.meeting_type,
                    "document_type": result.document_type,
                    "hierarchy": result.hierarchy,
                    "section_header": result.section_header,
                    "score": result.score,
                    "search_type": result.search_type,
                    "excerpt": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                    "url": direct_url or base_url or None
                })
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            prompt = f"""You are a helpful assistant for the Town of Davie government transparency system. 
            Answer the user's question based on the provided context from official meeting records and documents.
            
            Context:
            {context}
            
            Question: {query}
            
            Please provide a comprehensive answer based on the context. If the context doesn't contain enough information to answer the question, say so. Include specific details from the sources when possible.
            
            Answer:"""
            
            # Generate response
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for government transparency."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            # Calculate confidence based on search scores
            avg_score = sum(r.score for r in search_results[:3]) / min(len(search_results), 3)
            confidence = min(avg_score, 1.0)
            
            # Apply explain and rerank to sources (optional)
            if self.enable_source_explain:
            try:
                logger.info("Applying explain and rerank to sources")
                    # Build a quick map from chunk_id to url/title for backfilling
                    by_id = {s.get("chunk_id"): {"url": s.get("url"), "meeting_type": s.get("meeting_type"), "meeting_date": s.get("meeting_date")} for s in sources}
                explained_sources = await self._explain_and_rerank_sources(query, answer, sources)
                    enriched = explained_sources.get("items")
                    # Guard: if reranker returns empty/null, keep original sources
                    if not enriched:
                        # Fall back to original top-5 sources unmodified
                        pass
                    else:
                        # Prepare additional fallbacks
                        by_mt_date = {}
                        for s in sources:
                            key = (s.get("meeting_type"), s.get("meeting_date"))
                            if key not in by_mt_date:
                                by_mt_date[key] = []
                            if s.get("url"):
                                by_mt_date[key].append(s.get("url"))
                        original_urls = [s.get("url") for s in sources]
 
                        # Backfill URL from original sources when missing (chunk_id → meeting/date → positional)
                        fixed = []
                        for idx, item in enumerate(enriched):
                            if not isinstance(item, dict):
                                fixed.append(item)
                                continue
                            cid = item.get("chunk_id")
                            url = (item.get("url") or "").strip()
                            if not url and cid and cid in by_id and (by_id[cid] or {}).get("url"):
                                item["url"] = by_id[cid]["url"]
                                fixed.append(item)
                                continue
                            if not item.get("url"):
                                key = (item.get("meeting_type"), item.get("meeting_date"))
                                if key in by_mt_date and by_mt_date[key]:
                                    item["url"] = by_mt_date[key][0]
                            if not item.get("url") and idx < len(original_urls) and original_urls[idx]:
                                item["url"] = original_urls[idx]
                            # Final guaranteed fallback
                            if not item.get("url"):
                                item["url"] = f"/api/search?q={cid or (item.get('meeting_id') or item.get('meeting_date') or 'source')}"
                            fixed.append(item)
                        sources = fixed
            except Exception as e:
                logger.warning(f"Explain and rerank failed: {e}")
            else:
                # No LLM explain: attach basic summaries so UI isn't blank
                for s in sources:
                    if not s.get("summary"):
                        s["summary"] = (s.get("excerpt") or "").split("\n")[0][:140]
                    if not s.get("why"):
                        s["why"] = "Relevant to the query based on content overlap."

            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "search_results_count": len(search_results),
                "reranking_applied": True
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "I encountered an error while generating the answer. Please try again.",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Main method to process a query end-to-end"""
        try:
            logger.info(f"Processing query: {query}")
            
            # Perform hybrid search
            search_results = await self.hybrid_search(query)
            
            # Generate answer
            answer_data = await self.generate_answer(query, search_results)
            
            # Add search metadata
            result = {
                "query": query,
                "answer": answer_data["answer"],
                "sources": answer_data["sources"],
                "confidence": answer_data["confidence"],
                "search_results_count": answer_data.get("search_results_count", 0),
                "timestamp": datetime.now().isoformat(),
                "search_types_used": list(set(r.search_type for r in search_results))
            }
            
            logger.info(f"Query processed successfully. Found {len(search_results)} results.")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "answer": "I encountered an error while processing your query. Please try again.",
                "sources": [],
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _load_video_lookup(self) -> None:
        """Load optional video URL backfill from JSON. Structure examples:
        {
          "by_chunk_id": {"<chunk_id>": {"video_url": "https://youtu.be/ID", "start_time": 123}},
          "by_meeting_id": {"<meeting_id>": {"video_url": "https://youtu.be/ID"}}
        }
        """
        try:
            import json as _json
            from pathlib import Path as _Path
            cfg = os.getenv("VIDEO_LOOKUP_JSON")
            candidates = []
            if cfg and cfg.strip():
                candidates.append(_Path(cfg.strip()))
            # Default local file inside repo if present
            candidates.append(_Path(__file__).resolve().parent / "video_lookup.json")
            for p in candidates:
                if p and p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        data = _json.load(f)
                    self._video_by_chunk_id = dict(data.get("by_chunk_id", {}) or {})
                    self._video_by_meeting_id = dict(data.get("by_meeting_id", {}) or {})
                    logger.info(f"Loaded video backfill: chunks={len(self._video_by_chunk_id)} meetings={len(self._video_by_meeting_id)}")
                    return
        except Exception as e:
            logger.warning(f"Video lookup load skipped: {e}")

    def _backfill_video_url(self, md: Optional[Dict[str, Any]] = None, hit: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[int]]:
        """Return (video_url, start_time) from in-memory lookup using chunk_id or meeting_id."""
        try:
            cid = None
            mid = None
            if hit:
                cid = hit.get("chunk_id") or (hit.get("metadata") or {}).get("chunk_id")
                mid = hit.get("meeting_id") or (hit.get("metadata") or {}).get("meeting_id")
            if md and not cid:
                cid = md.get("chunk_id")
            if md and not mid:
                mid = md.get("meeting_id")
            # Prefer chunk-level
            if cid and cid in self._video_by_chunk_id:
                v = self._video_by_chunk_id.get(cid) or {}
                return str(v.get("video_url") or ""), (int(v.get("start_time")) if v.get("start_time") is not None else None)
            # Then meeting-level
            if mid and mid in self._video_by_meeting_id:
                v = self._video_by_meeting_id.get(mid) or {}
                return str(v.get("video_url") or ""), (int(v.get("start_time")) if v.get("start_time") is not None else None)
        except Exception:
            pass
        return "", None

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask about Town of Davie meetings")
    max_results: Optional[int] = Field(10, description="Maximum number of results to return")

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    search_results_count: int
    timestamp: str
    search_types_used: List[str]

# Initialize RAG system
rag_system = MilvusOnlyRAGSystem()

# FastAPI app
app = FastAPI(
    title="Town of Davie RAG System (Milvus Only - Corrected)",
    description="RAG system for Town of Davie government transparency using Milvus vector search",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main interface"""
    # Embedded minimal HTML UI for the Town of Davie Citizen Watch
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Town of Davie Citizen Watch</title>
      <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
      <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
      <style>
        .chat-container { height: 60vh; overflow-y: auto; }
        .message { animation: fadeIn .2s ease-in; }
        @keyframes fadeIn { from {opacity:0;transform:translateY(6px)} to {opacity:1;transform:translateY(0)} }
        .loader { width:20px;height:20px;border:3px solid #e5e7eb;border-top-color:#3b82f6;border-radius:50%;animation:spin 1s linear infinite }
        @keyframes spin { to { transform: rotate(360deg) } }
      </style>
    </head>
    <body class="bg-gray-50">
      <div class="max-w-4xl mx-auto p-4">
        <div class="rounded-lg shadow bg-white">
          <div class="px-6 py-4 border-b flex items-center justify-between">
            <div>
              <h1 class="text-xl font-semibold">Town of Davie Citizen Watch</h1>
              <p class="text-sm text-gray-500">Ask about meetings, agenda items, decisions, and more.</p>
            </div>
            <a href="/api/health" class="text-sm text-blue-600 underline">Health</a>
          </div>
          <div class="px-6 py-4">
            <div id="chat-messages" class="chat-container space-y-3">
              <div class="message bg-blue-50 rounded p-3 text-gray-800">
                <div class="font-medium mb-1"><i class="fa-solid fa-circle-info mr-1"></i>Welcome</div>
                <div class="text-sm">Type a question like: <span class="italic">"What did the council decide about Pine Island Road?"</span></div>
              </div>
            </div>
          </div>
          <div class="px-6 py-4 border-t">
            <form id="chat-form" class="flex items-center gap-2" onsubmit="event.preventDefault(); sendMessage();">
              <input id="message-input" class="flex-1 border rounded px-3 py-2 focus:outline-none focus:ring focus:border-blue-300" placeholder="Ask a question..." />
              <button class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition"><i class="fa-solid fa-paper-plane"></i> Send</button>
            </form>
          </div>
        </div>
      </div>
      <script>
        function renderSources(box, list){
          if(!Array.isArray(list) || !list.length) return;
          const wrap = document.createElement('div');
          wrap.className = 'mt-2 text-sm text-gray-700';
          wrap.innerHTML = '<div class="font-semibold mb-1">Sources</div>' +
            list.map((c,i)=>{
              const title = (c.title||c.url||'Source');
              const url = c.url||'#';
              const tail = [c.meeting_id, c.meeting_date].filter(Boolean).join(' • ');
              return `<div class="truncate">${i+1}. <a class="text-blue-700 underline" href="${url}" target="_blank" rel="noopener">${title}</a> <span class="text-gray-500">${tail?('('+tail+')'):''}</span></div>`
            }).join('');
          box.appendChild(wrap);
        }
        async function sendMessage(){
          const input = document.getElementById('message-input');
          const msg = (input.value||'').trim();
          if(!msg) return;
          input.value='';
          const box = document.getElementById('chat-messages');
          const user = document.createElement('div');
          user.className='message bg-green-50 rounded p-3';
          user.textContent=msg; box.appendChild(user);
          const thinking = document.createElement('div'); thinking.className='message flex items-center gap-2 text-gray-600'; thinking.innerHTML='<span class="loader"></span><span>Thinking...</span>'; box.appendChild(thinking);
          box.scrollTop = box.scrollHeight;
          try{
            const r = await fetch('/api/query', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query: msg, max_results: 10})});
            const data = await r.json();
            thinking.remove();
            const ai = document.createElement('div'); ai.className='message bg-gray-50 rounded p-3 whitespace-pre-wrap'; ai.textContent=(data && data.answer) || 'No answer'; box.appendChild(ai);
            
            // Handle sources from our corrected API format
            const sources = (data && data.sources) || [];
            if (sources.length){
              const wrap = document.createElement('div');
              wrap.className='mt-2 text-sm text-gray-700 bg-white border rounded p-3';
              wrap.innerHTML = '<div class="font-semibold mb-1">Sources used to answer your question</div>' +
                sources.map((b,i)=>{
                  const mt = (b.meeting_type||'').toString();
                  const md = (b.meeting_date||'').toString();
                  const tail = [mt, md].filter(Boolean).join(' • ');
                  const summary = (b.summary||'').trim();
                  const why = (b.why||'').trim();
                  const s1 = summary ? (summary.endsWith('.')? summary : summary + '.') : '';
                  const s2 = why ? (why.endsWith('.')? why : why + '.') : '';
                  const blurb = (s1 + ' ' + s2).trim();
                  const url = b.url||'#';
                  const blurbEsc = blurb.replace(/</g,'&lt;').replace(/>/g,'&gt;');
                  
                  return `<div class="mb-2">${i+1}. <a class="text-blue-700 underline" href="${url}" target="_blank" rel="noopener"><i class="fa-solid fa-arrow-up-right-from-square mr-1"></i>Click</a> <span class="text-gray-500">${tail?('('+tail+')'):''}</span><div class="text-gray-600">${blurbEsc}</div></div>`
                }).join('');
              box.appendChild(wrap);
            }
            box.scrollTop = box.scrollHeight;
          }catch(e){ 
            thinking.innerHTML='<span class="text-red-600">Error contacting API: ' + e.message + '</span>'; 
          }
        }
      </script>
    </body></html>
    """
    return HTMLResponse(content=html, status_code=200)

@app.post("/api/query", response_model=QueryResponse)
async def query(request: Request, query_request: QueryRequest):
    """Process a query using the RAG system"""
    try:
        result = await rag_system.process_query(query_request.query)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "milvus_configured": bool(rag_system.milvus_uri),
        "openai_configured": bool(rag_system.openai_api_key)
    }

@app.get("/api/search")
async def search_only(request: Request, q: str, limit: int = 10):
    """Search only endpoint (no answer generation)"""
    try:
        search_results = await rag_system.hybrid_search(q, limit)
        return {
            "query": q,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                    "document_type": r.document_type,
                    "meeting_id": r.meeting_id,
                    "meeting_date": r.meeting_date,
                    "meeting_type": r.meeting_type,
                    "hierarchy": r.hierarchy,
                    "section_header": r.section_header,
                    "score": r.score,
                    "search_type": r.search_type,
                    "metadata": r.metadata
                }
                for r in search_results
            ],
            "count": len(search_results)
        }
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 