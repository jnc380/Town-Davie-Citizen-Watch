#!/usr/bin/env python3
"""
Simplified Milvus-Only RAG System for Testing
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

# HTTP client for remote services
try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

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

# Log runtime config (safe)
try:
    _milvus_uri = os.getenv("MILVUS_URI")
    logger.info(
        "Runtime config: LOG_LEVEL=%s MILVUS_URI_HOST=%s MILVUS_TOKEN=%s",
        _log_level,
        _host_only(_milvus_uri),
        _present(os.getenv("MILVUS_TOKEN")),
    )
except Exception:
    pass

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
    Milvus-only RAG system using dense and sparse search
    for comprehensive government transparency Q&A
    """
    
    def __init__(self):
        """Initialize connections and clients"""
        # Load environment variables from .env if available
        possible_envs = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent / ".env",
            Path(__file__).resolve().parent.parent / ".env",
        ]
        for env_path in possible_envs:
            if env_path.exists():
                try:
                    from dotenv import load_dotenv  # type: ignore
                    load_dotenv(dotenv_path=str(env_path))
                except Exception:
                    pass
                break

        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Milvus/Minerva config
        self.milvus_uri = os.getenv("MILVUS_URI", "")
        self.milvus_token = os.getenv("MILVUS_TOKEN", "")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "TOWN_OF_DAVIE_RAG")
        # Override to use the correct collection name
        self.collection_name = "TOWN_OF_DAVIE_RAG"
        try:
            logger.info(f"Milvus collection configured: {self.collection_name}")
        except Exception:
            pass

        # Year filter for scope guardrails
        self.year_filter = os.getenv("YEAR_FILTER", "2025")

        # Initialize clients
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        
        # Search configuration
        self.dense_search_limit = int(os.getenv("DENSE_SEARCH_LIMIT", "20"))
        self.sparse_search_limit = int(os.getenv("SPARSE_SEARCH_LIMIT", "20"))
        self.final_result_limit = int(os.getenv("FINAL_RESULT_LIMIT", "10"))
        
        # Reranking configuration
        self.enable_reranking: bool = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
        self.reranking_model: str = os.getenv("RERANKING_MODEL", "gpt-4o-mini")
        self.bm25_rerank: bool = os.getenv("BM25_RERANK", "false").lower() == "true"
        
        # Initialize HTTP client for remote Milvus
        if httpx:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        else:
            self.http_client = None

    async def search_milvus_dense(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Perform dense vector search using Milvus"""
        try:
            return await self._search_milvus_http_dense(query, limit)
        except Exception as e:
            logger.error(f"Error in dense search: {e}")
            return []

    async def search_milvus_sparse(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Perform sparse search using BM25"""
        try:
            return await self._search_milvus_http_sparse(query, limit)
        except Exception as e:
            logger.error(f"Error in sparse search: {e}")
            return []

    async def _search_milvus_http_dense(self, query: str, limit: int) -> List[SearchResult]:
        """Search Milvus via HTTP API for dense search"""
        if not self.http_client or not self.milvus_uri:
            logger.warning("HTTP client or Milvus URI not available")
            return []
        
        try:
            # Generate embedding for query
            embedding_response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = embedding_response.data[0].embedding
            
            # Prepare search request
            search_data = {
                "collection_name": self.collection_name,
                "data": [query_embedding],
                "anns_field": "embedding",
                "param": {
                    "metric_type": "COSINE",
                    "params": {"nprobe": 10}
                },
                "limit": limit,
                "output_fields": [
                    "id", "content", "document_type", "meeting_id", 
                    "meeting_date", "meeting_type", "hierarchy", 
                    "section_header", "metadata_json"
                ]
            }
            
            # Make HTTP request to Milvus
            headers = {
                "Authorization": f"Bearer {self.milvus_token}",
                "Content-Type": "application/json"
            }
            
            response = await self.http_client.post(
                f"{self.milvus_uri}/v1/vector/search",
                json=search_data,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Milvus HTTP search failed: {response.status_code}")
                return []
            
            results = response.json()
            search_results = []
            
            for hit in results.get("results", []):
                try:
                    metadata = json.loads(hit.get("metadata_json", "{}"))
                    search_results.append(SearchResult(
                        chunk_id=hit.get("id", ""),
                        content=hit.get("content", ""),
                        document_type=hit.get("document_type", ""),
                        meeting_id=hit.get("meeting_id", ""),
                        meeting_date=hit.get("meeting_date", ""),
                        meeting_type=hit.get("meeting_type", ""),
                        hierarchy=hit.get("hierarchy", ""),
                        section_header=hit.get("section_header", ""),
                        metadata=metadata,
                        score=hit.get("score", 0.0),
                        search_type="dense"
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing search result: {e}")
                    continue
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in HTTP dense search: {e}")
            return []

    async def _search_milvus_http_sparse(self, query: str, limit: int) -> List[SearchResult]:
        """Search Milvus via HTTP API for sparse search (BM25)"""
        if not self.http_client or not self.milvus_uri:
            logger.warning("HTTP client or Milvus URI not available")
            return []
        
        try:
            # For sparse search, we'll use a simple text search approach
            # In a real implementation, you might use Milvus's built-in BM25 or implement it client-side
            
            # Prepare search request for text search
            search_data = {
                "collection_name": self.collection_name,
                "data": [query],
                "anns_field": "content",  # Search in content field
                "param": {
                    "metric_type": "BM25",
                    "params": {}
                },
                "limit": limit,
                "output_fields": [
                    "id", "content", "document_type", "meeting_id", 
                    "meeting_date", "meeting_type", "hierarchy", 
                    "section_header", "metadata_json"
                ]
            }
            
            # Make HTTP request to Milvus
            headers = {
                "Authorization": f"Bearer {self.milvus_token}",
                "Content-Type": "application/json"
            }
            
            response = await self.http_client.post(
                f"{self.milvus_uri}/v1/vector/search",
                json=search_data,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Milvus HTTP sparse search failed: {response.status_code}")
                return []
            
            results = response.json()
            search_results = []
            
            for hit in results.get("results", []):
                try:
                    metadata = json.loads(hit.get("metadata_json", "{}"))
                    search_results.append(SearchResult(
                        chunk_id=hit.get("id", ""),
                        content=hit.get("content", ""),
                        document_type=hit.get("document_type", ""),
                        meeting_id=hit.get("meeting_id", ""),
                        meeting_date=hit.get("meeting_date", ""),
                        meeting_type=hit.get("meeting_type", ""),
                        hierarchy=hit.get("hierarchy", ""),
                        section_header=hit.get("section_header", ""),
                        metadata=metadata,
                        score=hit.get("score", 0.0),
                        search_type="sparse"
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing sparse search result: {e}")
                    continue
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in HTTP sparse search: {e}")
            return []

    async def hybrid_search(self, query: str, limit: int = None) -> List[SearchResult]:
        """Perform hybrid search combining dense and sparse results"""
        try:
            # Use provided limit or default limits
            dense_limit = limit or self.dense_search_limit
            sparse_limit = limit or self.sparse_search_limit
            
            # Perform both dense and sparse searches
            dense_results = await self.search_milvus_dense(query, dense_limit)
            sparse_results = await self.search_milvus_sparse(query, sparse_limit)
            
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
            
            # Return top results
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
                "You are a civic information assistant. Your job is to explain why each source is relevant to the specific question asked."
                " CRITICAL: Only include sources that are actually relevant to the question. "
                "If a source is about a completely different topic (e.g., horse sculptures when asked about Pine Island Road), "
                "do not include it in your response or give it a very low score (0-1). "
                "DO NOT repeat the full resolution title. Instead, write concise, citizen-friendly explanations."
                " For each relevant source, identify the specific action or detail and explain:"
                " (1) What specific action this source shows (e.g., 'This meeting approved the interlocal agreement with Broward County' or 'This document specifies the roadway segment from SW 36 Street to Nova Drive')"
                " (2) Why this specific action matters for the question (e.g., 'This shows the council's approval of the project' or 'This confirms the exact location boundaries')"
                " Include a short verbatim quote from the source that contains the supporting detail. Score 0-5 based on how directly the source supports the question."
                " Be specific about: approval decisions, exact locations, dollar amounts, dates, and specific actions mentioned in the source."
                " Write in simple language that citizens can understand. Keep summaries under 100 characters."
                " Output must be valid JSON with this shape only: {\"items\":[{\"url\":string,\"meeting_type\":string,\"meeting_date\":string,\"meeting_id\":string,\"summary\":string,\"why\":string,\"evidence_quote\":string,\"score\":number}...]}."
            )
            
            user_payload = {
                "question": question,
                "final_answer": final_answer,
                "sources": [{"url": s.get("url"), "excerpt": s.get("excerpt", ""), "meeting_type": s.get("meeting_type"), "meeting_date": s.get("meeting_date")} for s in sources],
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
                # fallback without json mode
                resp = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=400,
                )
                txt = resp.choices[0].message.content or ""
                logger.info(f"LLM explain/rerank fallback response: {txt[:500]}")
                
                # naive parse: pick lines containing http
                items = []
                for s in txt.splitlines():
                    if "http" in s:
                        items.append({"url": s.strip(), "summary": "This source contributes details used in the answer.", "why": "It matters because it documents the council action.", "score": 3})
                
                # If the model didn't echo URLs, synthesize items from provided sources
                if not items:
                    for s in sources[: max(1, int(top_k))]:
                        items.append({
                            "url": s.get("url"),
                            "meeting_type": s.get("meeting_type"),
                            "meeting_date": s.get("meeting_date"),
                            "meeting_id": s.get("meeting_id"),
                            "summary": "This meeting document contains specific details about the topic.",
                            "why": "It matters because it shows the exact council decision and details.",
                            "evidence_quote": s.get("excerpt", "")[:100] + "..." if len(s.get("excerpt", "")) > 100 else s.get("excerpt", ""),
                            "score": 3
                        })
                data = {"items": items}
            
            items = data.get("items", [])
            # Sort by score descending
            items.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return {"items": items[:top_k], "debug": debug_payload}
            
        except Exception as e:
            logger.warning(f"explain_and_rerank failed: {e}")
            # Fallback: return sources with basic info
            fallback_items = []
            for s in sources[:top_k]:
                fallback_items.append({
                    "url": s.get("url"),
                    "meeting_type": s.get("meeting_type"),
                    "meeting_date": s.get("meeting_date"),
                    "meeting_id": s.get("meeting_id"),
                    "summary": "This source contains relevant information.",
                    "why": "It provides details about the topic discussed.",
                    "evidence_quote": s.get("excerpt", "")[:100] + "..." if len(s.get("excerpt", "")) > 100 else s.get("excerpt", ""),
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
                sources.append({
                    "chunk_id": result.chunk_id,
                    "meeting_id": result.meeting_id,
                    "meeting_date": result.meeting_date,
                    "meeting_type": result.meeting_type,
                    "document_type": result.document_type,
                    "hierarchy": result.hierarchy,
                    "section_header": result.section_header,
                    "score": result.score,
                    "search_type": result.search_type
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
            
            # Apply explain and rerank to sources
            try:
                logger.info("Applying explain and rerank to sources")
                explained_sources = await self._explain_and_rerank_sources(query, answer, sources)
                sources = explained_sources.get("items", sources)
            except Exception as e:
                logger.warning(f"Explain and rerank failed: {e}")
            
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
    title="Town of Davie RAG System (Milvus Only - Simple)",
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

# Static files and templates (commented out for testing)
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main interface"""
    return {"message": "Milvus-only RAG system is running"}

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
        search_results = await rag_system.hybrid_search(q)
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
                    "search_type": r.search_type
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