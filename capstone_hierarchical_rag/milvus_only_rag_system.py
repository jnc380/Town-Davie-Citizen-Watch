#!/usr/bin/env python3
"""
Milvus-Only RAG System for Capstone Project
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

# Serverless-safe mode: always skip local Milvus on Vercel
LIGHT_DEPLOYMENT = True
BM25_RERANK = os.getenv("BM25_RERANK", "false").lower() == "true"

# OpenAI imports
from openai import AsyncOpenAI

# Milvus/Minerva imports (removed for Vercel serverless)
connections = None  # type: ignore
Collection = None  # type: ignore
utility = None  # type: ignore

# HTTP client for remote services (used in LIGHT_DEPLOYMENT)
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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

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

# Optional pure-Python BM25
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    BM25Okapi = None  # type: ignore

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

        # Milvus/Minerva config (fallbacks to known cloud defaults if env not set)
        self.milvus_uri = os.getenv("MILVUS_URI", "")
        self.milvus_token = os.getenv("MILVUS_TOKEN", "")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "TOWN_OF_DAVIE_RAG")
        try:
            logger.info(f"Milvus collection configured: {self.collection_name}")
        except Exception:
            pass

        # Year filter for scope guardrails
        self.year_filter = os.getenv("YEAR_FILTER", "2025")

        # Initialize clients
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        self.milvus_collection = None
        
        # Reranking/config toggles
        self.enable_reranking: bool = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
        self.reranking_model: str = os.getenv("RERANKING_MODEL", "gpt-4o-mini")
        # JSON schema toggle for summarizer
        self.use_json_schema: bool = os.getenv("USE_JSON_SCHEMA", "true").lower() == "true"
        
        # Search configuration
        self.dense_search_limit = int(os.getenv("DENSE_SEARCH_LIMIT", "20"))
        self.sparse_search_limit = int(os.getenv("SPARSE_SEARCH_LIMIT", "20"))
        self.final_result_limit = int(os.getenv("FINAL_RESULT_LIMIT", "10"))
        
        # Initialize HTTP client for remote Milvus
        if LIGHT_DEPLOYMENT and httpx:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        else:
            self.http_client = None

    async def search_milvus_dense(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Perform dense vector search using Milvus"""
        try:
            if LIGHT_DEPLOYMENT and self.http_client:
                return await self._search_milvus_http_dense(query, limit)
            else:
                return await self._search_milvus_local_dense(query, limit)
        except Exception as e:
            logger.error(f"Error in dense search: {e}")
            return []

    async def search_milvus_sparse(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Perform sparse search using BM25"""
        try:
            if LIGHT_DEPLOYMENT and self.http_client:
                return await self._search_milvus_http_sparse(query, limit)
            else:
                return await self._search_milvus_local_sparse(query, limit)
        except Exception as e:
            logger.error(f"Error in sparse search: {e}")
            return []

    async def _search_milvus_http_dense(self, query: str, limit: int) -> List[SearchResult]:
        """Search Milvus via HTTP API for dense search"""
        if not self.http_client or not self.milvus_uri:
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

    async def _search_milvus_local_dense(self, query: str, limit: int) -> List[SearchResult]:
        """Search Milvus locally for dense search (not used in LIGHT_DEPLOYMENT)"""
        # This would be used if running locally with pymilvus
        logger.warning("Local dense search not implemented for LIGHT_DEPLOYMENT")
        return []

    async def _search_milvus_local_sparse(self, query: str, limit: int) -> List[SearchResult]:
        """Search Milvus locally for sparse search (not used in LIGHT_DEPLOYMENT)"""
        # This would be used if running locally with pymilvus
        logger.warning("Local sparse search not implemented for LIGHT_DEPLOYMENT")
        return []

    async def hybrid_search(self, query: str) -> List[SearchResult]:
        """Perform hybrid search combining dense and sparse results"""
        try:
            # Perform both dense and sparse searches
            dense_results = await self.search_milvus_dense(query, self.dense_search_limit)
            sparse_results = await self.search_milvus_sparse(query, self.sparse_search_limit)
            
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
            
            # Return top results
            return combined_results[:self.final_result_limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

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
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "search_results_count": len(search_results)
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
    title="Town of Davie RAG System (Milvus Only)",
    description="RAG system for Town of Davie government transparency using Milvus vector search",
    version="1.0.0"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/query", response_model=QueryResponse)
@limiter.limit("10/minute")
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
@limiter.limit("20/minute")
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