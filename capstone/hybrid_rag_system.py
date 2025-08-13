#!/usr/bin/env python3
"""
Hybrid RAG System for Capstone Project
Combines vector search (Minerva/Zilliz) with graph relationships (Neo4j)
for comprehensive government transparency Q&A
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
from pathlib import Path
# import numpy as np  # Avoid heavy dep on Vercel; use standard lists where possible
# from sklearn.feature_extraction.text import TfidfVectorizer  # Removed; guarded import later if needed
import re
import time
import uuid
from collections import deque, defaultdict
from capstone.telemetry import init_if_configured as telemetry_init, record_event, upsert_session
from capstone.telemetry import get_status as telemetry_status
import hashlib
import base64

# Serverless-safe mode: always skip local Milvus/Neo4j on Vercel
LIGHT_DEPLOYMENT = True
BM25_RERANK = os.getenv("BM25_RERANK", "false").lower() == "true"

# OpenAI imports
from openai import AsyncOpenAI

# Milvus/Minerva imports (removed for Vercel serverless)
connections = None  # type: ignore
Collection = None  # type: ignore
utility = None  # type: ignore

# Neo4j imports (removed for Vercel serverless)
GraphDatabase = None  # type: ignore

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
    _neo4j_api = os.getenv("NEO4J_QUERY_API_URL")
    logger.info(
        "Runtime config: LOG_LEVEL=%s MILVUS_URI_HOST=%s MILVUS_TOKEN=%s NEO4J_QUERY_API_HOST=%s NEO4J_USER=%s",
        _log_level,
        _host_only(_milvus_uri),
        _present(os.getenv("MILVUS_TOKEN")),
        _host_only(_neo4j_api),
        _present(os.getenv("NEO4J_USERNAME")),
    )
except Exception:
    pass

# Flags defined above for early conditional imports
# Optional pure-Python BM25
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    BM25Okapi = None  # type: ignore

@dataclass
class MeetingData:
    """Represents a meeting with all its associated data"""
    meeting_id: str
    meeting_type: str
    meeting_date: str
    title: str
    transcript_chunks: List[Dict[str, Any]]
    agenda_items: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@dataclass
class GraphEntity:
    """Represents an entity in the knowledge graph"""
    entity_id: str
    entity_type: str
    properties: Dict[str, Any]
    labels: List[str]

@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph"""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]

class HybridRAGSystem:
    """
    Hybrid RAG system combining vector search with graph relationships
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
        self.collection_name = os.getenv("MILVUS_COLLECTION", "capstone_hybrid_rag")
        try:
            logger.info(f"Milvus collection configured: {self.collection_name}")
        except Exception:
            pass
        
        # Neo4j config (env only)
        self.neo4j_uri = os.getenv("NEO4J_URI", "")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "")

        # Year filter for scope guardrails (aligned with DataProcessor YEAR_FILTER)
        self.year_filter = os.getenv("YEAR_FILTER", "2025")

        # Initialize clients
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        self.neo4j_driver = None
        self.milvus_collection = None
        
        # Reranking/config toggles
        self.enable_reranking: bool = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
        self.reranking_model: str = os.getenv("RERANKING_MODEL", "gpt-4o-mini")
        # JSON schema toggle for summarizer
        self.use_json_schema: bool = os.getenv("USE_JSON_SCHEMA", "true").lower() == "true"
        # Timestamp enrichment for agenda citations
        self.enable_timestamp_enrichment: bool = os.getenv("TRANSCRIPT_TIMESTAMP_ENRICHMENT", "true").lower() == "true"
        try:
            self.max_timestamp_enriched: int = max(0, int(os.getenv("MAX_TIMESTAMP_ENRICHED", "3")))
        except Exception:
            self.max_timestamp_enriched = 3
        
        # Sparse vectorizer (TF-IDF)
        self.sparse_vector_dim: int = 1000
        self.sparse_vectorizer: Optional[TfidfVectorizer] = None  # type: ignore
        if LIGHT_DEPLOYMENT:
            logger.info("ℹ️ LIGHT_DEPLOYMENT enabled: skipping TF-IDF sparse vectorizer initialization")
        
        # Initialize connections (removed for Vercel serverless)
        logger.info("ℹ️ Serverless-safe: skipping Milvus/Neo4j connections")
        self.milvus_collection = None
        self.neo4j_driver = None
        
        logger.info("✅ Hybrid RAG system initialized successfully")
 
    # -----------------------------
    # Helpers: dates and routing
    # -----------------------------
    def _normalize_date_str(self, date_str: str) -> Optional[str]:
        """Normalize various date formats to YYYY-MM-DD if possible."""
        if not date_str:
            return None
        cleaned = (
            date_str.replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
        )
        fmts = ["%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%Y-%m-%d"]
        for fmt in fmts:
            try:
                dt = datetime.strptime(cleaned, fmt)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                continue
        return None

    def _classify_query(self, query: str) -> Dict[str, Any]:
        """Route queries: detect meeting_type hints and date constraints; set alpha."""
        q = query.lower()
        meeting_type = None
        if any(k in q for k in ["cra"]):
            meeting_type = "CRA Meeting"
        elif any(k in q for k in ["workshop"]):
            meeting_type = "Workshop Meeting"
        elif any(k in q for k in ["budget hearing"]):
            meeting_type = "Budget Hearing"
        elif any(k in q for k in ["regular", "council meeting", "town council"]):
            meeting_type = "Regular Council Meeting"

        # Date extraction (very simple): e.g., July 23, 2025; May 2025; 2025
        date_regexes = [
            r"([A-Za-z]+\s+\d{1,2},\s*\d{4})",
            r"([A-Za-z]+\s+\d{4})",
            r"(\d{4})"
        ]
        date_norm = None
        month_year = None
        year = None
        month_prefix = None
        year_prefix = None
        for rx in date_regexes:
            m = re.search(rx, query)
            if m:
                captured = m.group(1)
                # Try full normalization to YYYY-MM-DD; else store month/year or year
                date_norm = self._normalize_date_str(captured)
                if not date_norm:
                    # Month Year pattern like May 2025
                    mm = re.match(r"([A-Za-z]+)\s+(\d{4})", captured)
                    if mm:
                        month_year = f"{mm.group(1)} {mm.group(2)}"
                        # build YYYY-MM prefix
                        try:
                            dt = datetime.strptime(month_year, "%B %Y")
                        except Exception:
                            try:
                                dt = datetime.strptime(month_year, "%b %Y")
                            except Exception:
                                dt = None
                        if dt:
                            month_prefix = dt.strftime("%Y-%m")
                    elif re.match(r"\d{4}$", captured):
                        year = captured
                        year_prefix = captured
                break

        # Alpha routing
        alpha = 0.65
        # Finance/records: lean to sparse/graph
        if any(k in q for k in [
            "itemid","meetingid","ordinance","resolution","bid","rfp","itb","contract","section",
            "budget","amendment","true-up","true up","loan","interlocal","acquisition","purchase","cdbg","grant","cip"
        ]):
            alpha = 0.35
        if any(k in q for k in ["define", "exact phrase", "quote", "#"]):
            alpha = 0.25

        return {
            "meeting_type": meeting_type,
            "date_norm": date_norm,
            "month_year": month_year,
            "year": year,
            "month_prefix": month_prefix,
            "year_prefix": year_prefix,
            "alpha": alpha
        }

    def fit_sparse_vectorizer(self, corpus_texts: List[str]) -> None:
        # Sparse vectorizer removed for Vercel serverless
        self.sparse_vectorizer = None
        return

    
    def _setup_milvus(self):
        """Milvus setup removed on Vercel serverless."""
        self.milvus_collection = None
        return
    
    def _create_milvus_collection(self):
        """Milvus collection creation removed on Vercel serverless."""
        return
    
    def _setup_neo4j(self):
        """Neo4j setup removed on Vercel serverless."""
        self.neo4j_driver = None
        return
     
    def _ensure_unique_constraint(self, session, label: str, property_name: str) -> None:
        """Ensure a unique constraint exists for the given label.property.
        If a non-unique index blocks creation and there are no duplicates, drop the index and retry.
        If duplicates exist, ensure a non-unique index and log a warning.
        """
        # Check for duplicates first to avoid spurious constraint errors
        try:
            dup_result = session.run(
                f"""
                MATCH (n:`{label}`)
                WITH n.{property_name} AS id, count(*) AS c
                WHERE id IS NOT NULL AND c > 1
                RETURN count(*) AS num
                """
            ).single()
            num_dup_groups = dup_result["num"] if dup_result else 0
        except Exception as e:
            logger.warning(
                f"⚠️ Unable to check duplicates for {label}.{property_name}: {e}. Ensuring non-unique index instead."
            )
            num_dup_groups = 1  # force fallback path

        if num_dup_groups > 0:
            # Duplicates present: do NOT attempt unique constraint; ensure non-unique index only
            try:
                session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.{property_name})"
                )
                logger.info(
                    f"ℹ️ Using non-unique index for {label}.{property_name} (duplicates present)."
                )
            except Exception:
                pass
            return

        # No duplicates: drop any non-unique index on this property, then create the unique constraint
        try:
            idx_rows = session.run(
                """
                SHOW INDEXES YIELD name, entityType, labelsOrTypes, properties, owningConstraint
                WHERE entityType = 'NODE' AND labelsOrTypes = [$label] AND properties = [$prop]
                RETURN name, owningConstraint
                """,
                {"label": label, "prop": property_name},
            ).data()

            for row in idx_rows:
                if not row.get("owningConstraint"):
                    idx_name = row["name"]
                    try:
                        session.run(f"DROP INDEX `{idx_name}`")
                        logger.info(
                            f"🧹 Dropped non-unique index {idx_name} on {label}.{property_name}"
                        )
                    except Exception as drop_err:
                        logger.warning(
                            f"⚠️ Failed to drop index {idx_name} on {label}.{property_name}: {drop_err}"
                        )

            # Create unique constraint idempotently
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.{property_name} IS UNIQUE"
            )
            logger.info(f"✅ Ensured unique constraint on {label}.{property_name}")
        except Exception as e:
            logger.warning(
                f"⚠️ Could not finalize unique constraint on {label}.{property_name}: {e}. Falling back to non-unique index."
            )
            try:
                session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.{property_name})"
                )
            except Exception:
                pass
    
    def _initialize_graph_schema(self):
        """Initialize the Neo4j graph schema with constraints and indexes"""
        with self.neo4j_driver.session() as session:
            # Create constraints and indexes
            constraints = [
                "CREATE CONSTRAINT meeting_id IF NOT EXISTS FOR (m:Meeting) REQUIRE m.id IS UNIQUE",
                "CREATE CONSTRAINT council_member_id IF NOT EXISTS FOR (cm:CouncilMember) REQUIRE cm.id IS UNIQUE",
                "CREATE CONSTRAINT agenda_item_id IF NOT EXISTS FOR (ai:AgendaItem) REQUIRE ai.id IS UNIQUE",
                "CREATE CONSTRAINT resolution_id IF NOT EXISTS FOR (r:Resolution) REQUIRE r.id IS UNIQUE",
            ]
             
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")
             
            logger.info("✅ Neo4j graph schema initialized")
     
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def _text_to_sparse_vector(self, text: str):
        if LIGHT_DEPLOYMENT or not self.sparse_vectorizer:
            return None
        try:
            vec = self.sparse_vectorizer.transform([text])  # sparse matrix 1 x dim
            return vec
        except Exception:
            return None
    
    def process_meeting_data(self, meeting_data: MeetingData) -> Tuple[List[Dict], List[GraphEntity], List[GraphRelationship]]:
        """
        Process meeting data into vector chunks and graph entities/relationships
        
        Returns:
            Tuple of (vector_chunks, graph_entities, graph_relationships)
        """
        vector_chunks = []
        graph_entities = []
        graph_relationships = []
        
        # Create meeting entity
        meeting_date_norm = self._normalize_date_str(meeting_data.meeting_date) or meeting_data.meeting_date
        meeting_entity = GraphEntity(
            entity_id=meeting_data.meeting_id,
            entity_type="Meeting",
            properties={
                "title": meeting_data.title,
                "meeting_type": meeting_data.meeting_type,
                "meeting_date": meeting_date_norm,
                "url": meeting_data.metadata.get("url", ""),
                "description": meeting_data.metadata.get("description", "")
            },
            labels=["Meeting", meeting_data.meeting_type.replace(" ", "_")]
        )
        graph_entities.append(meeting_entity)
        
        # Process transcript chunks
        for i, chunk in enumerate(meeting_data.transcript_chunks):
            chunk_id = f"{meeting_data.meeting_id}_transcript_{i}"
            
            # Create vector chunk
            vector_chunk = {
                "content": chunk["text"],
                "chunk_id": chunk_id,
                "meeting_id": meeting_data.meeting_id,
                "meeting_type": meeting_data.meeting_type,
                "meeting_date": meeting_date_norm,
                "chunk_type": "youtube_transcript",
                "start_time": chunk.get("start", 0),
                "duration": chunk.get("duration", 0),
                "metadata": {
                    "source": "transcript",
                    "chunk_index": i,
                    "total_chunks": len(meeting_data.transcript_chunks),
                    "url": meeting_data.metadata.get("url", ""),
                    "meeting_date_raw": meeting_data.meeting_date
                }
            }
            vector_chunks.append(vector_chunk)
        
        # Process agenda items
        for item in meeting_data.agenda_items:
            item_id = f"{meeting_data.meeting_id}_agenda_{item.get('item_id', 'unknown')}"
            
            # Create agenda item entity
            agenda_entity = GraphEntity(
                entity_id=item_id,
                entity_type="AgendaItem",
                properties={
                    "title": item.get("text", ""),
                    "item_id": item.get("item_id", ""),
                    "meeting_id": meeting_data.meeting_id,
                    "description": item.get("description", ""),
                    "url": item.get("url", "")
                },
                labels=["AgendaItem"]
            )
            graph_entities.append(agenda_entity)
            
            # Create relationship between meeting and agenda item
            relationship = GraphRelationship(
                source_id=meeting_data.meeting_id,
                target_id=item_id,
                relationship_type="HAS_AGENDA_ITEM",
                properties={
                    "order": item.get("order", 0)
                }
            )
            graph_relationships.append(relationship)
            
            # Add agenda item to vector chunks if it has content
            if item.get("text"):
                vector_chunk = {
                    "content": item["text"],
                    "chunk_id": item_id,
                    "meeting_id": meeting_data.meeting_id,
                    "meeting_type": meeting_data.meeting_type,
                    "meeting_date": meeting_date_norm,
                    "chunk_type": "agenda_item",
                    "start_time": 0,
                    "duration": 0,
                    "metadata": {
                        "source": "agenda",
                        "item_id": item.get("item_id", ""),
                        "item_type": item.get("type", "unknown"),
                        "url": item.get("url", ""),
                        "meeting_date_raw": meeting_data.meeting_date
                    }
                }
                vector_chunks.append(vector_chunk)
        
        return vector_chunks, graph_entities, graph_relationships
    
    async def load_meeting_data(self, meeting_data: MeetingData):
        """Load meeting data into both vector and graph databases"""
        try:
            # Process the data
            vector_chunks, graph_entities, graph_relationships = self.process_meeting_data(meeting_data)
            
            # Get embeddings for vector chunks
            dense_embeddings: List[List[float]] = []
            sparse_embeddings: List[Optional[List[float]]] = []
            for chunk in vector_chunks:
                dense = await self.get_embedding(chunk["content"])
                dense_embeddings.append(dense)
                sparse = self._text_to_sparse_vector(chunk["content"]) if self.sparse_vectorizer else None
                sparse_embeddings.append(sparse)
            
            # Insert into Milvus
            if vector_chunks and dense_embeddings:
                self._insert_into_milvus(vector_chunks, dense_embeddings, sparse_embeddings)
            
            # Insert into Neo4j
            if graph_entities or graph_relationships:
                self._insert_into_neo4j(graph_entities, graph_relationships)
            
            logger.info(f"✅ Loaded meeting {meeting_data.meeting_id} with {len(vector_chunks)} chunks and {len(graph_entities)} entities")
            
        except Exception as e:
            logger.error(f"❌ Failed to load meeting data: {e}")
            raise
    
    def _insert_into_milvus(self, chunks: List[Dict], dense_embeddings: List[List[float]], sparse_embeddings: List[Optional[List[float]]]):
        """Insert chunks and embeddings (dense + sparse) into Milvus"""
        try:
            num = len(chunks)
            # Prepare all columns in schema order (excluding auto_id primary key)
            contents = [chunk["content"] for chunk in chunks]
            chunk_ids = [chunk["chunk_id"] for chunk in chunks]
            meeting_ids = [chunk["meeting_id"] for chunk in chunks]
            meeting_types = [chunk["meeting_type"] for chunk in chunks]
            meeting_dates = [chunk["meeting_date"] for chunk in chunks]
            chunk_types = [chunk["chunk_type"] for chunk in chunks]
            start_times = [float(chunk.get("start_time", 0.0)) for chunk in chunks]
            durations = [float(chunk.get("duration", 0.0)) for chunk in chunks]
            metadatas = [chunk.get("metadata", {}) for chunk in chunks]
            
            # Dense embeddings
            dense_vecs = dense_embeddings
            
            # Sparse embeddings: ensure present for all rows
            zero_sparse = [0.0] * self.sparse_vector_dim
            sparse_vecs = [
                (se if (se is not None and len(se) == self.sparse_vector_dim) else zero_sparse)
                for se in sparse_embeddings
            ]
            
            # Assemble column-ordered data
            column_data = [
                contents,
                chunk_ids,
                meeting_ids,
                meeting_types,
                meeting_dates,
                chunk_types,
                start_times,
                durations,
                metadatas,
                dense_vecs,
                sparse_vecs,
            ]
            
            self.milvus_collection.insert(column_data)
            self.milvus_collection.flush()
            
            logger.info(f"✅ Inserted {num} chunks into Milvus")
            
        except Exception as e:
            logger.error(f"❌ Failed to insert into Milvus: {e}")
            raise
    
    def _insert_into_neo4j(self, entities: List[GraphEntity], relationships: List[GraphRelationship]):
        """Insert entities and relationships into Neo4j"""
        try:
            with self.neo4j_driver.session() as session:
                # Insert entities
                for entity in entities:
                    labels_str = ":".join(entity.labels)
                    properties_str = ", ".join([f"{k}: ${k}" for k in entity.properties.keys()])
                    
                    query = f"""
                    MERGE (e:{labels_str} {{id: $entity_id}})
                    SET e += {{{properties_str}}}
                    """
                    
                    params = {"entity_id": entity.entity_id, **entity.properties}
                    session.run(query, params)
                
                # Insert relationships
                for rel in relationships:
                    query = """
                    MATCH (source {id: $source_id})
                    MATCH (target {id: $target_id})
                    MERGE (source)-[r:""" + rel.relationship_type + """]->(target)
                    SET r += $properties
                    """
                    
                    params = {
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "properties": rel.properties
                    }
                    session.run(query, params)
                
                logger.info(f"✅ Inserted {len(entities)} entities and {len(relationships)} relationships into Neo4j")
                
        except Exception as e:
            logger.error(f"❌ Failed to insert into Neo4j: {e}")
            raise
    
    def _dense_search(self, query_embedding: List[float], top_k: int, expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform dense vector search in Milvus"""
        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 128}
            }
            
            results = self.milvus_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k * 3,
                expr=expr,
                output_fields=["content", "chunk_id", "meeting_id", "meeting_type", "meeting_date", "chunk_type", "start_time", "duration", "metadata"]
            )
            
            out = []
            for hits in results:
                for rank, hit in enumerate(hits):
                    meta_val = hit.entity.get("metadata")
                    if isinstance(meta_val, str):
                        try:
                            meta = json.loads(meta_val)
                        except Exception:
                            meta = {}
                    elif meta_val is None:
                        meta = {}
                    else:
                        meta = meta_val
                    out.append({
                        "content": hit.entity.get("content"),
                        "chunk_id": hit.entity.get("chunk_id"),
                        "meeting_id": hit.entity.get("meeting_id"),
                        "meeting_type": hit.entity.get("meeting_type"),
                        "meeting_date": hit.entity.get("meeting_date"),
                        "chunk_type": hit.entity.get("chunk_type"),
                        "start_time": hit.entity.get("start_time"),
                        "duration": hit.entity.get("duration"),
                        "metadata": meta,
                        "score": hit.score,
                        "rank": rank
                    })
            return out
            
        except Exception as e:
            logger.error(f"❌ Failed to perform dense search: {e}")
            return []
    
    def _sparse_search(self, query_text: str, top_k: int, expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform sparse (TF-IDF) vector search in Milvus if available"""
        if not self.sparse_vectorizer:
            return []
        try:
            vec = self._text_to_sparse_vector(query_text)
            if vec is None:
                return []
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            results = self.milvus_collection.search(
                data=[vec],
                anns_field="sparse_embedding",
                param=search_params,
                limit=top_k * 3,
                expr=expr,
                output_fields=["content", "chunk_id", "meeting_id", "meeting_type", "meeting_date", "chunk_type", "start_time", "duration", "metadata"]
            )
            out = []
            for hits in results:
                for rank, hit in enumerate(hits):
                    meta_val = hit.entity.get("metadata")
                    if isinstance(meta_val, str):
                        try:
                            meta = json.loads(meta_val)
                        except Exception:
                            meta = {}
                    elif meta_val is None:
                        meta = {}
                    else:
                        meta = meta_val
                    out.append({
                        "content": hit.entity.get("content"),
                        "chunk_id": hit.entity.get("chunk_id"),
                        "meeting_id": hit.entity.get("meeting_id"),
                        "meeting_type": hit.entity.get("meeting_type"),
                        "meeting_date": hit.entity.get("meeting_date"),
                        "chunk_type": hit.entity.get("chunk_type"),
                        "start_time": hit.entity.get("start_time"),
                        "duration": hit.entity.get("duration"),
                        "metadata": meta,
                        "score": hit.score,
                        "rank": rank
                    })
            return out
        except Exception as e:
            logger.error(f"❌ Failed to perform sparse search: {e}")
            return []
    
    def _fuse_dense_sparse(self, dense: List[Dict], sparse: List[Dict], top_k: int, alpha: float = 0.5) -> List[Dict]:
        """Fuse dense and sparse results using simple RRF and weighted sum."""
        # Build maps: chunk_id -> best rank and score
        rrf_k = 60.0
        scores: Dict[str, float] = {}
        meta: Dict[str, Dict] = {}
        
        for lst, weight in ((dense, alpha), (sparse, 1 - alpha)):
            for item in lst:
                cid = item.get("chunk_id")
                rrf = 1.0 / (rrf_k + item.get("rank", 0) + 1)
                scores[cid] = scores.get(cid, 0.0) + weight * rrf
                if cid not in meta:
                    meta[cid] = item
        
        # Rank by fused score
        combined = [
            {**meta[cid], "fused_score": s}
            for cid, s in scores.items()
        ]
        # Small bonus for agenda-aligned transcript segments
        for it in combined:
            if (it.get("chunk_type") or "") == "youtube_agenda_segment":
                it["fused_score"] = float(it.get("fused_score", 0)) + 0.05
        combined.sort(key=lambda x: x.get("fused_score", 0), reverse=True)
        return combined[:top_k]
    
    async def hybrid_search(self, query: str, top_k: int = 10, prior_context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Perform hybrid search combining vector similarity with graph relationships
        """
        try:
            # Route query
            route = self._classify_query(query)
            alpha = route["alpha"]

            # Scope guardrail: if query targets a different year than our index, return explicit message
            def _out_of_scope() -> Dict[str, Any]:
                msg = (
                    f"No indexed data for the requested time period under the current scope (YEAR_FILTER={self.year_filter}). "
                    f"Please adjust the year or broaden the query."
                )
                return {
                    "query": query,
                    "vector_results": [],
                    "graph_results": [],
                    "combined_results": {
                        "answer": msg,
                        "source_citations": [],
                        "vector_sources": [],
                        "graph_sources": [],
                        "total_sources": 0,
                    },
                    "search_metadata": {
                        "dense_count": 0,
                        "sparse_count": 0,
                        "vector_count": 0,
                        "graph_count": 0,
                        "alpha": alpha,
                        "filters": "",
                        "total_results": 0,
                    },
                }

            # If a specific year was detected and it doesn't match our scope, short-circuit
            yp = route.get("year_prefix")
            dn = route.get("date_norm")
            if (yp and yp != self.year_filter) or (dn and not dn.startswith(self.year_filter)):
                return _out_of_scope()

            # Build Milvus scalar filter expression if possible
            filters = []
            if route.get("meeting_type"):
                mt = route["meeting_type"].replace("'", "\\'")
                filters.append(f"meeting_type == '{mt}'")
            if route.get("date_norm"):
                dn = route["date_norm"]
                filters.append(f"meeting_date == '{dn}'")
            # Note: for month/year or year-only we skip expr and rely on fusion + rerank
            expr = " and ".join(filters) if filters else None

            # If LIGHT_DEPLOYMENT, use remote services and skip local heavy clients
            if LIGHT_DEPLOYMENT:
                # Get query embedding once (still needed for answer synthesis but remote may not need it)
                try:
                    query_embedding = await asyncio.wait_for(self.get_embedding(query), timeout=15)
                except Exception:
                    query_embedding = await self.get_embedding(query)

                # Remote calls
                vector_results = await self._zilliz_search(query_embedding, top_k * 2, expr)
                # Prefer Neo4j Query API if configured; else leave graph empty
                graph_results = await self._neo4j_query_api_search(query, top_k)

                # Fuse (treat remote vector as dense; no sparse in light mode)
                fused_results = self._fuse_dense_sparse(vector_results, [], top_k * 2, alpha=alpha)
                # BM25 rerank in light mode
                fused_results = self._bm25_rerank(query, fused_results, top_k)

                # Build combined outputs
                combined = await self._synthesize_answer(query, fused_results, graph_results)
                return {
                    "query": query,
                    "vector_results": vector_results,
                    "graph_results": graph_results,
                    "combined_results": combined,
                    "search_metadata": {
                        "dense_count": len(vector_results),
                        "sparse_count": 0,
                        "vector_count": len(vector_results),
                        "graph_count": len(graph_results),
                        "alpha": alpha,
                        "filters": expr or "",
                        "total_results": len(vector_results) + len(graph_results),
                    },
                }

            # Normal (non-light) path continues below

            # Get query embedding with timeout
            try:
                query_embedding = await asyncio.wait_for(self.get_embedding(query), timeout=15)
            except Exception:
                query_embedding = await self.get_embedding(query)

            # Dense and sparse searches concurrently (offload sync drivers)
            dense_task = asyncio.to_thread(self._dense_search, query_embedding, top_k, expr)
            sparse_task = asyncio.to_thread(self._sparse_search, query, top_k, expr)
            dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

            fused_results = self._fuse_dense_sparse(dense_results, sparse_results, top_k * 2, alpha=alpha)

            # Graph search task (sync → thread)
            graph_task = asyncio.to_thread(self._graph_search, query, top_k, route)

            # Optional GPT rerank
            if self.enable_reranking:
                try:
                    rerank_task = self._rerank_with_gpt(query, fused_results, top_k)
                    reranked, graph_results = await asyncio.gather(rerank_task, graph_task)
                except Exception:
                    reranked = fused_results[:top_k]
                    graph_results = await graph_task
            else:
                # If GPT rerank is off, optionally rerank with BM25
                bm25_candidates = fused_results
                reranked = self._bm25_rerank(query, bm25_candidates, top_k)
                graph_results = await graph_task

            # If nothing retrieved, return a scoped no-data message to avoid hallucinations
            if not reranked and not graph_results:
                return _out_of_scope()

            # Drop vector results that only match agenda boilerplate (addresses, headers)
            try:
                ql = (query or '').lower()
                sig_tokens = [t for t in re.findall(r"[a-z0-9]+", ql) if len(t) > 2]
                # simple stoplist
                stops = {"the","and","for","with","from","that","this","what","when","where","who","did","were","was","are","is","to","of","in","on","at","a","an","be","about"}
                sig_tokens = [t for t in sig_tokens if t not in stops]
                bigrams = [f"{a} {b}" for a,b in zip(sig_tokens, sig_tokens[1:])]
                trigram = " ".join(sig_tokens[-3:]) if len(sig_tokens) >= 3 else None
                def _vec_ok(v: Dict[str, Any]) -> bool:
                    text = self._strip_boilerplate(v.get('content') or '').lower()
                    if not text:
                        return False
                    cond_all = all(t in text for t in sig_tokens) if sig_tokens else True
                    cond_bg = any(bg in text for bg in bigrams) if bigrams else False
                    cond_tri = (trigram in text) if trigram else False
                    # Prefer phrase; allow all-token fallback
                    return cond_tri or cond_bg or cond_all
                filtered = [v for v in reranked if _vec_ok(v)]
                if filtered:
                    reranked = filtered[:top_k]
            except Exception:
                pass

            # Graph→Vector narrowing: if vector recall is low but we have graph meetings, fetch transcript chunks in those meetings
            try:
                if len(reranked) < top_k and graph_results:
                    graph_meeting_ids = []
                    for g in graph_results:
                        mid = (g.get("meeting") or {}).get("meeting_id")
                        if mid and str(mid) not in graph_meeting_ids:
                            graph_meeting_ids.append(str(mid))
                    if graph_meeting_ids:
                        in_list = ", ".join([f"'{m}'" for m in graph_meeting_ids])
                        expr_graph = f"meeting_id in [{in_list}]"
                        dense_g = await asyncio.to_thread(self._dense_search, query_embedding, top_k, expr_graph)
                        sparse_g = await asyncio.to_thread(self._sparse_search, query, top_k, expr_graph)
                        fused_g = self._fuse_dense_sparse(dense_g, sparse_g, top_k, alpha=alpha)
                        # merge dedup by chunk_id, prefer existing order, then fused_g order
                        seen = {x.get("chunk_id") for x in reranked}
                        for x in fused_g:
                            cid = x.get("chunk_id")
                            if cid not in seen:
                                reranked.append(x)
                                seen.add(cid)
                        reranked = reranked[: top_k]
            except Exception:
                pass

            # Combine and analyze results
            combined = await self._combine_results(query, reranked, graph_results, prior_context)

            return {
                "query": query,
                "vector_results": reranked,
                "graph_results": graph_results,
                "combined_results": combined,
                "search_metadata": {
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "vector_count": len(reranked),
                    "graph_count": len(graph_results),
                    "alpha": alpha,
                    "filters": expr or "",
                    "total_results": len(reranked) + len(graph_results)
                }
            }

        except Exception as e:
            logger.error(f"❌ Failed to perform hybrid search: {e}")
            raise
 
    def _graph_search(self, query: str, top_k: int, route: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform graph search in Neo4j"""
        try:
            with self.neo4j_driver.session() as session:
                # Tokenize query into keywords for flexible matching (drop common stopwords)
                stopwords = {
                    "the","and","for","with","from","that","this","what","when","where","who",
                    "did","were","was","are","is","to","of","in","on","at","a","an","be",
                    "council","meeting","regular","board","agency","town","about"
                }
                tokens = [t for t in re.split(r"\W+", query.lower()) if t and len(t) > 2 and t not in stopwords]
                tokens = list(dict.fromkeys(tokens))  # de-duplicate preserving order
 
                # Detect explicit ItemID/MeetingID in query
                m_item = re.search(r"item\s*id\s*[:#\-]?\s*(\d+)|itemid\s*[:#\-]?\s*(\d+)", query, flags=re.I)
                m_meeting = re.search(r"meeting\s*id\s*[:#\-]?\s*(\d+)|meetingid\s*[:#\-]?\s*(\d+)", query, flags=re.I)
                explicit_item_id = (m_item.group(1) or m_item.group(2)) if m_item else None
                explicit_meeting_id = (m_meeting.group(1) or m_meeting.group(2)) if m_meeting else None
 
                # Phrase-aware filter: use the tail 2–3 significant words (common subject at end of question)
                words = re.findall(r"[a-z0-9]+", query.lower())
                sig_words = [w for w in words if len(w) > 2 and w not in stopwords]
                phrase = None
                if len(sig_words) >= 2:
                    tail = sig_words[-3:] if len(sig_words) >= 3 else sig_words[-2:]
                    phrase = " ".join(tail)
 
                token_params = {f"t{i}": tok for i, tok in enumerate(tokens)}
                token_match = " OR ".join([
                    f"toLower(ai.title) CONTAINS ${k} OR toLower(ai.description) CONTAINS ${k} OR toLower(m.title) CONTAINS ${k}"
                    for k in token_params.keys()
                ]) or "true"
 
                where_clauses = [f"({token_match})"]
                params: Dict[str, Any] = {**token_params, "top_k": top_k}
                if phrase and len(phrase.split()) >= 2:
                    where_clauses.append("toLower(coalesce(ai.title,'') + ' ' + coalesce(ai.description,'')) CONTAINS $phrase")
                    params["phrase"] = phrase
                if route:
                    if route.get("meeting_type"):
                        where_clauses.append("m.meeting_type = $mt")
                        params["mt"] = route["meeting_type"]
                    if route.get("date_norm"):
                        where_clauses.append("m.meeting_date = $dn")
                        params["dn"] = route["date_norm"]
                    elif route.get("month_prefix"):
                        where_clauses.append("m.meeting_date STARTS WITH $mp")
                        params["mp"] = route["month_prefix"]
                    elif route.get("year_prefix"):
                        where_clauses.append("m.meeting_date STARTS WITH $yp")
                        params["yp"] = route["year_prefix"]
                if explicit_item_id:
                    where_clauses.append("ai.item_id = $eid")
                    params["eid"] = explicit_item_id
                if explicit_meeting_id:
                    where_clauses.append("m.meeting_id = $emid")
                    params["emid"] = explicit_meeting_id
                # Join clauses: first is the token ORs, followed by AND filters
                where_str = where_clauses[0]
                if len(where_clauses) > 1:
                    where_str = f"({where_str}) AND " + " AND ".join(where_clauses[1:])
 
                cypher_query = f"""
                MATCH (m:Meeting)-[:HAS_AGENDA_ITEM]->(ai:AgendaItem)
                WHERE {where_str}
                OPTIONAL MATCH (ai)-[:RESULTED_IN]->(res:Resolution)
                OPTIONAL MATCH (ai)-[:LINKS_TO]->(con:Contract)
                OPTIONAL MATCH (ai)-[:LINKS_TO]->(att:Attachment)
                OPTIONAL MATCH (m)-[:DISCUSSED]->(t:Topic)
                OPTIONAL MATCH (ai)-[:SAME_TOPIC_AS]->(ai2:AgendaItem)
                OPTIONAL MATCH (ai)<-[:ON_ITEM]-(motion:Motion)
                OPTIONAL MATCH (p:Person)-[v:VOTED_ON]->(motion)
                WITH m, ai,
                     collect(DISTINCT res) as resolutions,
                     collect(DISTINCT con) as contracts,
                     collect(DISTINCT att) as attachments,
                     collect(DISTINCT t) as topics,
                     collect(DISTINCT motion) as motions,
                     collect(DISTINCT {{person: coalesce(p.name, p.full_name, p.person_id), vote: v.vote}}) as votes,
                     size([x IN $tokens WHERE toLower(ai.title) CONTAINS x OR toLower(ai.description) CONTAINS x]) as hits,
                     CASE WHEN any(x IN $tokens WHERE ai.title CONTAINS '$' OR ai.description CONTAINS '$') THEN 1 ELSE 0 END as has_amount,
                     size([x IN $tokens WHERE toLower(m.title) CONTAINS x]) as meeting_hits,
                     0 as topic_hits
                WHERE hits > 0
                RETURN m, ai, resolutions, contracts, attachments, topics, votes, hits, has_amount, meeting_hits, topic_hits,
                       size(resolutions) as res_count,
                       size(contracts) as contract_count,
                       size(attachments) as attachment_count,
                       size(motions) as motion_count
                LIMIT $top_k
                """
                result = session.run(cypher_query, {**params, "tokens": list(tokens)})
                graph_results = []
                for record in result:
                    meeting = record["m"]
                    agenda_item = record["ai"]
                    ai_dict = dict(agenda_item)
                    # safe getters
                    def _safe(k, default=None):
                        try:
                            return record.get(k, default)
                        except Exception:
                            return default
                    hits = _safe("hits", 0) or 0
                    has_amount = _safe("has_amount", 0) or 0
                    meeting_hits = _safe("meeting_hits", 0) or 0
                    topic_hits = _safe("topic_hits", 0) or 0
                    res_list = [dict(r) for r in (_safe("resolutions", []) or [])]
                    con_list = [dict(r) for r in (_safe("contracts", []) or [])]
                    att_list = [dict(r) for r in (_safe("attachments", []) or [])]
                    votes_list = _safe("votes", []) or []
                    motion_count = int(_safe("motion_count", 0) or 0)

                    title_l = (ai_dict.get('title','') or '').lower()
                    desc_l = (ai_dict.get('description','') or '').lower()
                    vendor_terms = ["agreement","purchase","psa","contract","amendment","rfp","bid","itb"]
                    has_vendor = 1 if any((t in title_l) or (t in desc_l) for t in vendor_terms) else 0
                    yes_votes = 0
                    try:
                        yes_votes = sum(1 for v in votes_list if str(v.get('vote','')).lower() in ("yes","yea","approve","approved","pass","passed"))
                    except Exception:
                        yes_votes = 0
                    score = int(hits) + int(meeting_hits) + int(topic_hits) + (2 if has_amount else 0) + has_vendor + (2 if len(res_list)>0 else 0) + (1 if len(con_list)>0 else 0) + (2 if motion_count>0 else 0) + (1 if yes_votes>0 else 0)

                    # augment context/notes
                    reason_extra = []
                    if motion_count:
                        reason_extra.append(f"motions: {motion_count}")
                    if votes_list:
                        reason_extra.append(f"votes: {len(votes_list)}")

                    graph_results.append({
                        "meeting": dict(meeting),
                        "agenda_item": ai_dict,
                        "relationship": "HAS_AGENDA_ITEM",
                        "linked": {
                            "resolutions": res_list,
                            "contracts": con_list,
                            "attachments": att_list,
                            "topics": [dict(t) for t in (record.get("topics", []) or [])],
                            "votes": votes_list,
                            "motion_count": motion_count,
                            "reason_extra": "; ".join(reason_extra) if reason_extra else None,
                        },
                        "score": score,
                        "hits": int(hits)
                    })
                graph_results.sort(key=lambda r: r.get("score", 0), reverse=True)
                return graph_results
        except Exception as e:
            logger.error(f"❌ Failed to perform graph search: {e}")
            return []

    async def _rerank_with_gpt(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Use GPT to rerank a list of candidates. Returns top_k items."""
        if not candidates:
            return []
        # Build a compact list of chunks
        items = candidates[: min(len(candidates), max(top_k * 3, 12))]
        prompt_items = []
        for i, c in enumerate(items, 1):
            preview = (c.get("content", "") or "")
            preview = preview.replace("\n", " ").replace("\r", " ")
            # guard against unescaped quotes/backslashes to reduce JSON parse errors
            preview = preview.replace("\\", " ")
            preview = preview.replace('"', "'")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            prompt_items.append({
                "id": c.get("chunk_id", f"cand_{i}"),
                "meeting": c.get("meeting_id", ""),
                "type": c.get("chunk_type", ""),
                "date": c.get("meeting_date", ""),
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
            id_to_item = {c.get("chunk_id"): c for c in items}
            ranked = [id_to_item[i] for i in order if i in id_to_item]
            # Append any missing to fill
            seen = set(order)
            for c in items:
                cid = c.get("chunk_id")
                if cid not in seen:
                    ranked.append(c)
            return ranked[:top_k]
        except Exception as e:
            logger.warning(f"GPT rerank failed, falling back to fused scores: {e}")
            items_sorted = sorted(items, key=lambda x: x.get("fused_score", 0), reverse=True)
            return items_sorted[:top_k]
 
    async def _final_summarize_with_gpt(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        year_scope: Optional[str] = None,
        prior_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate concise, grounded 1–2 paragraph answer with citations from retrieved context.
        Returns dict with keys: answer_paragraph (str), citations (list[str]).
        """
        try:
            # Prepare compact context
            def _diverse_vectors(candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
                selected: List[Dict[str, Any]] = []
                per_meeting_cap = 2
                meeting_counts: Dict[str, int] = {}
                for c in candidates:
                    mid = str(c.get("meeting_id") or (c.get("metadata") or {}).get("meeting_id") or "")
                    if mid:
                        if meeting_counts.get(mid, 0) >= per_meeting_cap:
                            continue
                    # avoid transcript segments that are too close in time
                    ok = True
                    if c.get("chunk_type") == "youtube_transcript" and selected:
                        st = float(c.get("start_time") or 0)
                        for s in selected:
                            if s.get("chunk_type") == "youtube_transcript" and str(s.get("meeting_id")) == mid:
                                st2 = float(s.get("start_time") or 0)
                                if abs(st - st2) < 60:
                                    ok = False
                                    break
                    if not ok:
                        continue
                    selected.append(c)
                    if mid:
                        meeting_counts[mid] = meeting_counts.get(mid, 0) + 1
                    if len(selected) >= k:
                        break
                return selected

            kv = getattr(self, "context_vector_snippets", 6)
            kg = getattr(self, "context_agenda_items", 6)
            # Consider first 2x candidates to improve diversity window
            vec_pool = vector_results[: max(kv * 2, 8)]
            vec_diverse = _diverse_vectors(vec_pool, kv)
            # Attempt neighbor inclusion for transcripts if room remains
            if len(vec_diverse) < kv:
                need = kv - len(vec_diverse)
                have_ids = {v.get("chunk_id") for v in vec_diverse}
                # simple neighbor: same meeting, adjacent chunk_index if available in pool
                for v in vec_pool:
                    if need <= 0:
                        break
                    if v.get("chunk_type") != "youtube_transcript":
                        continue
                    cid = v.get("chunk_id")
                    if cid in have_ids:
                        continue
                    mid = v.get("meeting_id") or (v.get("metadata") or {}).get("meeting_id")
                    idx = (v.get("metadata") or {}).get("chunk_index")
                    if mid and isinstance(idx, int):
                        # look for someone already selected with adjacent index
                        for s in vec_diverse:
                            if s.get("chunk_type") == "youtube_transcript" and (s.get("meeting_id") or (s.get("metadata") or {}).get("meeting_id")) == mid:
                                sidx = (s.get("metadata") or {}).get("chunk_index")
                                if isinstance(sidx, int) and abs(sidx - idx) == 1:
                                    vec_diverse.append(v)
                                    have_ids.add(cid)
                                    need -= 1
                                    break

            snippets = []
            for v in vec_diverse:
                text = (v.get("content") or "").strip()
                if len(text) > 250:
                    text = text[:250] + "..."
                url = v.get("url") or v.get("metadata", {}).get("url") or ""
                snippets.append({
                    "text": text,
                    "url": url,
                    "date": v.get("meeting_date", ""),
                    "type": v.get("chunk_type", "")
                })
            agenda = []
            # Lightweight amount/outcome extraction for summary context
            def _amts(t: str) -> List[str]:
                if not t:
                    return []
                vals = re.findall(r"\$\d[\d,]*(?:\.\d{1,2})?|\$\d+(?:\.\d+)?\s*(?:m|million)\b", t, flags=re.IGNORECASE)
                # dedupe order preserving
                out, seen = [], set()
                for a in vals:
                    k = a.lower()
                    if k not in seen:
                        seen.add(k)
                        out.append(a)
                return out
            def _outcome(t: str) -> Optional[str]:
                if not t:
                    return None
                tl = t.lower()
                for k,v in [("approved","Approved"),("adopted","Adopted"),("passed","Passed"),("denied","Denied"),("tabled","Tabled")]:
                    if k in tl:
                        return v
                return None
            for g in graph_results[:kg]:
                ai = g.get("agenda_item", {})
                desc = ai.get("description", "") or ""
                title = ai.get("title", "")[:160]
                url = ai.get("url", "")
                amounts = _amts(title + " " + desc)
                outcome = _outcome(desc)
                agenda.append({
                    "title": title,
                    "url": url,
                    "amounts": amounts,
                    "outcome": outcome,
                })

            system = (
                "You are a civic QA assistant summarizing LOCAL GOVERNMENT agendas and meeting transcripts for CITIZENS. "
                "Audience: average resident. Data: provided CONTEXT ONLY (agenda cover sheets, minutes/transcripts). No outside knowledge. "
                "If the question is out of scope for the current index/year, return exactly: 'No indexed data for the current scope.' "
                "Style: 1–2 short paragraphs, plain English, citizen‑friendly, no boilerplate, no long quoted text. "
                "Ground strictly in the context; mention amounts/outcomes only if present; keep it concise and factual. "
                "Jurisdiction: restrict to the Town of Davie content only; ignore references to other cities or counties unless the context explicitly ties them to a Davie action. "
                "Output JSON: answer_paragraph (string), citations (array of urls)."
            )
            # Compact prior context (last few turns)
            convo = []
            if prior_context:
                for t in prior_context[-3:]:
                    convo.append({
                        "question": t.get("question", ""),
                        "answer": (t.get("answer", "") or "")[:400],
                        "citations": t.get("citations", [])
                    })

            user_payload = {
                "query": query,
                "year_scope": year_scope,
                "context_snippets": snippets,
                "agenda_items": agenda,
                "conversation_context": convo or [],
            }
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload)}
            ]
            # JSON schema for structured output
            schema = {
                "name": "summarize_civic_context",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer_paragraph": {
                            "type": "string",
                            "description": "Return only 1–2 short paragraphs in plain English, grounded strictly in the provided context. No lists, no HTML, no raw pasted source text. Mention amounts/outcomes/dates/companies/vendors/council members when applicable. If out of scope for the current YEAR_FILTER, return exactly: 'No indexed data for the current scope.'"
                        },
                        "citations": {
                            "type": "array",
                            "description": "URLs to agenda cover sheets, minutes, or transcripts that support the answer. Jurisdiction restricted to Town of Davie.",
                            "items": {"type": "string", "format": "uri"},
                            "minItems": 0
                        }
                    },
                    "required": ["answer_paragraph"],
                    "additionalProperties": False
                }
            }
            def _response_format():
                if getattr(self, "use_json_schema", False):
                    return {"type": "json_schema", "json_schema": schema}
                return {"type": "json_object"}
            try:
                resp = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2,
                    response_format=_response_format(),
                    max_tokens=getattr(self, "final_max_tokens", 220),
                )
                content = resp.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as e:
                logger.warning(f"Final synthesis JSON mode failed: {e}")
                # Retry without json mode
                resp = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=getattr(self, "final_max_tokens", 220),
                )
                content = resp.choices[0].message.content or ""
                # Build best-effort paragraph from content
                para = " ".join([ln.strip() for ln in content.splitlines() if ln.strip()])
                # Trim to ~2 short paragraphs
                return {"answer_paragraph": para[:900], "citations": re.findall(r"https?://\S+", content)[:5]}
        except Exception as e:
            logger.warning(f"Final synthesis failed: {e}")
            return None

    async def _combine_results(self, query: str, vector_results: List[Dict], graph_results: List[Dict], prior_context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Combine and analyze vector and graph results with source citations"""
        try:
            # Prepare source citations with links
            source_citations = []
 
            # Build a meeting_id -> youtube_url map from graph results when available
            meeting_youtube: Dict[str, str] = {}
            meeting_any_url: Dict[str, str] = {}
            for g in graph_results:
                mt = (g.get('meeting') or {})
                mid = str(mt.get('meeting_id', '') or '')
                murl = (mt.get('url') or '').strip()
                if not mid:
                    continue
                if murl:
                    meeting_any_url[mid] = murl
                    if 'youtube.com' in murl or 'youtu.be' in murl:
                        meeting_youtube[mid] = murl
 
            # Process vector results first
            for i, result in enumerate(vector_results[:5]):
                try:
                    url = result.get('url') or result.get('metadata', {}).get('url', '')
                    if not url:
                        meeting_id = result.get('meeting_id') or result.get('metadata', {}).get('meeting_id', '')
                        start_s = int(float(result.get('start_time', 0)))
                        # Prefer meeting youtube url derived from graph if available; else mapper; else metadata url; else leave empty
                        base_url = meeting_youtube.get(str(meeting_id)) or self._resolve_youtube_base(meeting_id, result.get('meeting_date')) or (result.get('metadata', {}) or {}).get('url') or ""
                        if base_url:
                            if 'youtu' in base_url.lower() and start_s > 0:
                                sep = '&' if '?' in base_url else '?'
                                url = f"{base_url}{sep}t={start_s}s"
                            else:
                                url = base_url
                    # Build concise one-sentence summary
                    meeting_date = result.get('meeting_date', '')
                    ts_h = self._format_seconds(result.get('start_time')) if result.get('start_time') is not None else None
                    if ts_h and meeting_date:
                        summary = f"Transcript segment from {meeting_date} at {ts_h} relevant to your question."
                    elif meeting_date:
                        summary = f"Transcript segment from {meeting_date} relevant to your question."
                    else:
                        summary = "Transcript segment relevant to your question."
                    # For YouTube chunks with a timestamp, add why this time
                    ts_reason = None
                    if result.get('chunk_type') == 'youtube_transcript' and result.get('start_time') is not None:
                        ts_reason = "Top-ranked segment by semantic similarity for your query"
 
                    citation = {
                        "type": result.get('chunk_type', 'document'),
                        "title": f"{result.get('meeting_type', 'Council Meeting')} - {result.get('meeting_date', '')}",
                        "url": url,
                        "summary": summary,
                        "relevance_score": result.get('fused_score', 0),
                        "timestamp": self._format_seconds(result.get('start_time')) if result.get('start_time') is not None else None,
                        "duration": self._format_seconds(result.get('duration')) if result.get('duration') is not None else None,
                        "reason": None,
                        "timestamp_reason": ts_reason,
                        "meeting_date": result.get('meeting_date', ''),
                        "meeting_id": result.get('meeting_id') or result.get('metadata', {}).get('meeting_id', ''),
                        "item_id": (result.get('metadata') or {}).get('item_id', '')
                    }
                    # If we have an agenda link for this meeting in graph, include it as auxiliary
                    mid_for_aux = str(citation.get('meeting_id') or '')
                    aux_agenda = None
                    if mid_for_aux and mid_for_aux in meeting_any_url and 'youtube' not in (meeting_any_url[mid_for_aux] or ''):
                        aux_agenda = meeting_any_url[mid_for_aux]
                    if aux_agenda:
                        citation['agenda_url'] = aux_agenda
                    # Final URL fallback to agenda if primary url is empty
                    if not citation.get('url') and citation.get('agenda_url'):
                        citation['url'] = citation['agenda_url']
                    # Skip citations that still lack any usable link
                    if not (citation.get('url') or citation.get('agenda_url')):
                        continue
                    source_citations.append(citation)
                except Exception:
                    continue
 
            # Process graph results (agenda items)
            def _extract_amounts(text: str) -> List[str]:
                try:
                    t = text or ""
                    patterns = [
                        r"\$\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?",
                        r"\$\d+(?:\.\d+)?\s*(?:m|million|b|billion)\b",
                        r"\b\d+(?:\.\d+)?%\b",
                        r"\$\d+(?:\.\d+)?\s*[\-–]\s*\$?\d+(?:\.\d+)?\s*(?:m|million|b|billion)?"
                    ]
                    results: List[str] = []
                    for p in patterns:
                        results.extend(re.findall(p, t, flags=re.IGNORECASE))
                    seen = set()
                    ordered: List[str] = []
                    for r_ in results:
                        key = r_.lower()
                        if key not in seen:
                            seen.add(key)
                            ordered.append(r_)
                    return ordered
                except Exception:
                    return []
 
            def _extract_outcome(text: str) -> Optional[str]:
                try:
                    if not text:
                        return None
                    t = text.lower()
                    outcome_map = [
                        ("approved", "Approved"),
                        ("adopted", "Adopted"),
                        ("passed", "Passed"),
                        ("denied", "Denied"),
                        ("failed", "Failed"),
                        ("tabled", "Tabled"),
                        ("withdrawn", "Withdrawn"),
                        ("continued", "Continued"),
                        ("postponed", "Postponed"),
                    ]
                    for k, v in outcome_map:
                        if k in t:
                            return v
                    if "motion to approve" in t:
                        return "Approved"
                    if "motion passed" in t or "passes" in t:
                        return "Passed"
                    return None
                except Exception:
                    return None
 
            def _extract_parties(title: str, desc: str) -> Optional[str]:
                try:
                    text = f"{title} {desc}" if desc else title
                    # Known public entities
                    entities = [
                        "Broward County",
                        "Town of Davie",
                        "Community Redevelopment Agency",
                        "Davie CRA",
                        "FDOT",
                        "Broward MPO",
                    ]
                    found = []
                    for e in entities:
                        if e.lower() in text.lower():
                            # normalize CRA names
                            norm = "Town of Davie CRA" if e in ["Community Redevelopment Agency", "Davie CRA"] else e
                            if norm not in found:
                                found.append(norm)
                    # Generic patterns
                    m_between = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:[\.;,\n]|$)", text, flags=re.IGNORECASE)
                    if m_between:
                        a = re.sub(r"\s+", " ", m_between.group(1)).strip()
                        b = re.sub(r"\s+", " ", m_between.group(2)).strip()
                        if a and a not in found:
                            found.append(a)
                        if b and b not in found:
                            found.append(b)
                    if found:
                        # cap to 3 entities for brevity
                        return " & ".join(found[:3])
                    return None
                except Exception:
                    return None
 
            def _extract_vendor(title: str, desc: str) -> Optional[str]:
                try:
                    text = f"{title} {desc}" if desc else title
                    patterns = [
                        r"(?:agreement|contract|services|purchase|psa|engagement)\s+with\s+([A-Z][^,;\n]{2,60})",
                        r"award\s+to\s+([A-Z][^,;\n]{2,60})",
                        r"with\s+([A-Z][A-Za-z&\.'\- ]{2,60})\s+(?:for|to)"
                    ]
                    for p in patterns:
                        m = re.search(p, text, flags=re.IGNORECASE)
                        if m:
                            name = m.group(1).strip()
                            # Avoid overly generic captures
                            if len(name) > 2 and "Town of Davie".lower() not in name.lower():
                                return name
                    return None
                except Exception:
                    return None
 
            def _extract_audit_threshold(title: str, desc: str) -> Optional[str]:
                try:
                    text = f"{title} {desc}" if desc else title
                    if re.search(r"audit|threshold", text, flags=re.IGNORECASE):
                        m = re.search(r"\$\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?", text)
                        if m:
                            return m.group(0)
                    return None
                except Exception:
                    return None
 
            agenda_bullets: List[str] = []
            for i, result in enumerate(graph_results[:5]):
                # Skip graph items that did not match tokens
                if int(result.get('hits', 0) or 0) <= 0:
                    continue
                agenda_item = result.get('agenda_item', {})
                meeting = result.get('meeting', {})
 
                raw_title = (agenda_item.get('title', '') or '').strip()
                title = raw_title[:140] + ('...' if len(raw_title) > 140 else '')
                item_id = agenda_item.get('item_id', '')
                url = agenda_item.get('url', '')
                desc = agenda_item.get('description', '') or ''
                amounts = _extract_amounts(" ".join([title, desc]))
                outcome = _extract_outcome(desc)
                parties = _extract_parties(title, desc)
                vendor = _extract_vendor(title, desc)
                audit_amt = _extract_audit_threshold(title, desc)
 
                # Build reason note for agenda items
                reason_parts: List[str] = ["Matches your keywords in agenda title/description"]
                if amounts:
                    reason_parts.append(f"includes amounts: {', '.join(sorted(set(amounts)))}")
                if outcome:
                    reason_parts.append(f"outcome noted: {outcome}")
                if parties:
                    reason_parts.append(f"parties: {parties}")
                if vendor:
                    reason_parts.append(f"vendor: {vendor}")
                if audit_amt:
                    reason_parts.append(f"audit threshold: {audit_amt}")
                reason_note = "; ".join(reason_parts)
 
                # Short context note for bullets/citations
                context_bits: List[str] = []
                if parties:
                    context_bits.append(f"Parties: {parties}")
                if vendor:
                    context_bits.append(f"Vendor: {vendor}")
                if audit_amt:
                    context_bits.append(f"Audit: {audit_amt}")
                context_note = "; ".join(context_bits) if context_bits else None

                # Build related links from nested entities
                related_links: List[str] = []
                for att in (result.get('linked', {}).get('attachments') or []):
                    u = (att.get('url') or '').strip()
                    if u:
                        related_links.append(u)
                for con in (result.get('linked', {}).get('contracts') or []):
                    u = (con.get('url') or '').strip()
                    if u:
                        related_links.append(u)
                for res in (result.get('linked', {}).get('resolutions') or []):
                    u = (res.get('url') or '').strip()
                    if u:
                        related_links.append(u)

                # Build base URL preference chain for agenda item
                base_url = meeting_youtube.get(str(meeting.get('meeting_id', ''))) or ''
                if not base_url:
                    # Try resolve via mapper/neo4j
                    base_url = self._resolve_youtube_base(meeting.get('meeting_id', ''), meeting.get('meeting_date', '')) or ''
                agenda_link = agenda_item.get('url', '')
                final_url = base_url or agenda_link or meeting_any_url.get(str(meeting.get('meeting_id','')), '')

                source_citations.append({
                    "type": "agenda_item",
                    "title": title or 'Agenda Item',
                    # Prefer resolved YouTube; else agenda; else meeting-level url
                    "url": final_url,
                    "meeting_date": meeting.get('meeting_date', ''),
                    "meeting_type": meeting.get('meeting_type', 'Council Meeting'),
                    "item_id": agenda_item.get('item_id', ''),
                    "meeting_id": meeting.get('meeting_id', ''),
                    "agenda_url": agenda_link,
                    "related_links": related_links[:5] or None,
                    "summary": reason_note,
                    "relevance_score": result.get('score', 0),
                    "reason": None,
                    "context_note": context_note
                })
 
                bullet = f"- {title}"
                if item_id:
                    bullet += f" (ItemID {item_id})"
                if amounts:
                    bullet += f" [Amounts: {', '.join(sorted(set(amounts)))}]"
                if outcome:
                    bullet += f" [Outcome: {outcome}]"
                if context_note:
                    bullet += f" [{context_note}]"
                if url:
                    bullet += f" — {url}"
                if title:
                    agenda_bullets.append(bullet)
 
            # Add plain-language definitions when finance/government terms are detected
            glossary_notes: List[str] = []
            ql = (query or '').lower()
            if any(k in ql for k in ["interlocal", "ila"]):
                glossary_notes.append("Interlocal agreement: a formal contract between government entities (e.g., Town, County) to cooperate on funding or responsibilities.")
            if "true-up" in ql or "true ups" in ql or "trueups" in ql:
                glossary_notes.append("Fund balance true-up: a year-end adjustment that reconciles estimated to actual balances; can trigger transfers or contributions to align accounts.")
            if "cra" in ql:
                glossary_notes.append("CRA (Community Redevelopment Agency): a special district focused on redevelopment; often involves non-TIF extensions and targeted funding.")
            if "tif" in ql:
                glossary_notes.append("TIF (Tax Increment Financing): uses future tax revenue increases from a district to fund current improvements in that area.")
            if any(k in ql for k in ["assessment", "rate resolution"]):
                glossary_notes.append("Assessment rate resolution: sets annual rates for a service (e.g., stormwater, fire); a preliminary rate is noticed, then finalized at a hearing.")
 
            # Final LLM synthesis for concise paragraphs (grounded)
            year_scope = str(getattr(self, "year_filter", "")) or None
            final_json = await self._final_summarize_with_gpt(query, vector_results, graph_results, year_scope, prior_context)
            # If the model returned too few citations, try one augmentation pass with larger context
            try:
                cits = (final_json or {}).get("citations") or []
                if len(cits) < 3:
                    # Expand pools and retry once
                    more_vec = vector_results[: max(len(vector_results), getattr(self, "context_vector_snippets", 6) + 4)]
                    more_graph = graph_results[: max(len(graph_results), getattr(self, "context_agenda_items", 6) + 3)]
                    final_json2 = await self._final_summarize_with_gpt(query, more_vec, more_graph, year_scope, prior_context)
                    if final_json2 and len((final_json2.get("citations") or [])) >= len(cits):
                        final_json = final_json2
            except Exception:
                pass
 
            # Fallback deterministic text if LLM fails (still concise)
            if not final_json or not isinstance(final_json, dict):
                # Build 1 short paragraph from top agenda items
                pieces = []
                for c in source_citations[:2]:
                    t = c.get("title", "")
                    m = c.get("meeting_date", "")
                    pieces.append(f"{t} ({m})")
                fallback = "; ".join(pieces)[:600]
                return {
                    "answer": fallback or "No indexed data for the current scope.",
                    "source_citations": source_citations,
                    "answer_bullets": [],
                }
 
            # Prefer the model's paragraph answer
            answer_text = (final_json.get("answer_paragraph") or "").strip()
            # Unwrap accidental JSON-in-string from model
            if answer_text.startswith("{") and '"answer_paragraph"' in answer_text:
                try:
                    inner = json.loads(answer_text)
                    if isinstance(inner, dict):
                        candidate = (inner.get("answer_paragraph") or "").strip()
                        if candidate:
                            answer_text = candidate
                except Exception:
                    pass
 
            # Query-aware filtering of citations for relevancy
            def _sig_tokens(q: str) -> List[str]:
                stops = {"the","and","for","with","from","that","this","what","when","where","who","which","did","were","was","are","is","to","of","in","on","at","a","an","be","road","roads"}
                toks = re.findall(r"[a-zA-Z0-9]+", (q or "").lower())
                return [t for t in toks if len(t) > 2 and t not in stops]
 
            sig = _sig_tokens(query)
            sig2 = sig[:]
            # Require at least two tokens to activate strict filter
            use_strict = len(sig2) >= 2
            # Build adjacent bigram phrases to capture core query phrases like "pine island"
            bigrams: List[str] = [f"{a} {b}" for a, b in zip(sig2, sig2[1:])]
            trigram: Optional[str] = " ".join(sig2[-3:]) if len(sig2) >= 3 else None
 
            meets_ok: set = set()
            if use_strict:
                for v in vector_results:
                    text = (v.get("content") or "").lower()
                    cond_all = all(t in text for t in sig2)
                    cond_tri = (trigram in text) if trigram else False
                    cond_bigram = any(bg in text for bg in bigrams) if bigrams else False
                    if cond_all or cond_tri or cond_bigram:
                        mid = v.get("meeting_id") or (v.get("metadata") or {}).get("meeting_id")
                        if mid:
                            meets_ok.add(str(mid))
 
            agenda_ok: set = set()
            if use_strict:
                for g in graph_results:
                    ai = g.get("agenda_item", {})
                    title = (ai.get("title") or "").lower()
                    desc = (ai.get("description") or "").lower()
                    blob = f"{title} {desc}"
                    cond_all = all(t in blob for t in sig2)
                    cond_tri = (trigram in blob) if trigram else False
                    cond_bigram = any(bg in blob for bg in bigrams) if bigrams else False
                    if cond_all or cond_tri or cond_bigram:
                        mid = (g.get("meeting") or {}).get("meeting_id")
                        iid = ai.get("item_id")
                        if mid and iid:
                            agenda_ok.add((str(mid), str(iid)))
 
            def _matches_query(cite: Dict[str, Any]) -> bool:
                if not use_strict:
                    return True
                mid = str(cite.get("meeting_id") or "")
                if cite.get("type") == "agenda_item":
                    iid = str(cite.get("item_id") or "")
                    # If we have graph-confirmed agenda items, require match; otherwise allow agenda items with links
                    if agenda_ok:
                        if (mid, iid) in agenda_ok:
                            return True
                    else:
                        if cite.get("url") or cite.get("agenda_url"):
                            return True
                # Non-agenda citations (e.g., transcript) can pass via meeting-level match
                if mid in meets_ok:
                    return True
                return False
 
            # If the model returned a citations array, filter to those urls and apply query-match constraint
            model_cites = set(final_json.get("citations") or [])
            if model_cites:
                filtered = [c for c in source_citations if (c.get("url") in model_cites or c.get("agenda_url") in model_cites) and _matches_query(c)]
                # Fallback: only vector citations that match query
                if not filtered and use_strict:
                    filtered = [c for c in source_citations if c.get("type") != "agenda_item" and _matches_query(c)]
                if not filtered:
                    filtered = source_citations[:3]
            else:
                # No model-provided cites → prefer vector citations that match query
                if use_strict:
                    # If graph provided no confirmations, allow agenda items with links; else prefer non-agenda that match
                    if not agenda_ok:
                        filtered = [c for c in source_citations if (c.get('url') or c.get('agenda_url'))][:5]
                    else:
                        vec_only = [c for c in source_citations if c.get("type") != "agenda_item" and _matches_query(c)]
                        filtered = vec_only if vec_only else [c for c in source_citations if _matches_query(c)]
                    if not filtered:
                        filtered = source_citations[:5]
                else:
                    filtered = source_citations
 
            # Drop any citations without a usable link to avoid broken anchors on the client
            filtered = [c for c in filtered if (c.get('url') or c.get('agenda_url'))]

            # Final fallback: if still empty, synthesize from top vector results to ensure non-empty citations list
            if not filtered:
                synthesized: List[Dict[str, Any]] = []
                for v in vector_results[:3]:
                    try:
                        meeting_id = v.get('meeting_id') or (v.get('metadata') or {}).get('meeting_id', '')
                        start_s = int(float(v.get('start_time', 0) or 0))
                        base_url = self._resolve_youtube_base(meeting_id, v.get('meeting_date')) or ''
                        url = ''
                        if base_url:
                            sep = '&' if '?' in base_url else '?'
                            url = f"{base_url}{sep}t={start_s}s" if start_s > 0 else base_url
                        # fallback to any meeting url seen from graph
                        if not url and meeting_id and meeting_id in meeting_any_url:
                            url = meeting_any_url.get(meeting_id, '')
                        # fallback to agenda metadata url from vector result
                        if not url:
                            meta_u = (v.get('metadata') or {}).get('url')
                            if isinstance(meta_u, str) and meta_u:
                                url = meta_u
                        # Only add synthesized citation if we actually found a URL
                        if not url:
                            continue
                        synthesized.append({
                            "type": v.get('chunk_type', 'document'),
                            "title": f"{v.get('meeting_type','Council Meeting')} - {v.get('meeting_date','')}",
                            "url": url,
                            "summary": "Transcript segment relevant to your question.",
                            "meeting_date": v.get('meeting_date',''),
                            "meeting_id": meeting_id,
                        })
                    except Exception:
                        continue
                if synthesized:
                    filtered = synthesized

            # Enrich agenda-item citations with transcript timestamps when available (limited for latency)
            try:
                if getattr(self, "enable_timestamp_enrichment", True) and self.max_timestamp_enriched > 0:
                    # Build quick lookup of transcript chunks by meeting
                    transcripts_by_meeting: Dict[str, List[Dict[str, Any]]] = {}
                    for v in vector_results:
                        if (v.get('chunk_type') or '') != 'youtube_transcript':
                            continue
                        mid = str(v.get('meeting_id') or (v.get('metadata') or {}).get('meeting_id') or '')
                        if not mid:
                            continue
                        transcripts_by_meeting.setdefault(mid, []).append(v)
                    # Simple term extraction from agenda title/desc
                    def _terms(text: str) -> List[str]:
                        if not text:
                            return []
                        stops = {"the","and","for","with","from","that","this","what","when","where","who","which","did","were","was","are","is","to","of","in","on","at","a","an","be"}
                        toks = re.findall(r"[a-zA-Z0-9]+", text.lower())
                        return [t for t in toks if len(t) > 2 and t not in stops]
                    def _score(text_l: str, key_terms: List[str], vendor: Optional[str], amounts: List[str]) -> float:
                        s = 0.0
                        for t in key_terms:
                            if t in text_l:
                                s += 1.0
                        if vendor and vendor.lower() in text_l:
                            s += 2.0
                        for a in amounts:
                            if a.lower() in text_l:
                                s += 1.5
                        for kw in ("motion", "approve", "approved", "award", "awarded", "contract", "bid", "rfp"):
                            if kw in text_l:
                                s += 0.5
                        return s
                    enriched = 0
                    for c in filtered:
                        if enriched >= self.max_timestamp_enriched:
                            break
                        if (c.get('type') or '') != 'agenda_item':
                            continue
                        mid = str(c.get('meeting_id') or '')
                        if not mid:
                            continue
                        agenda_title = (c.get('title') or '')
                        # We do not have description in citation; use graph_results to find it
                        desc_lookup = ''
                        for g in graph_results:
                            m = (g.get('meeting') or {})
                            ai = g.get('agenda_item', {})
                            if str(m.get('meeting_id') or '') == mid and (ai.get('item_id') or '') == (c.get('item_id') or ''):
                                desc_lookup = ai.get('description') or ''
                                break
                        amounts_here = _extract_amounts(agenda_title + ' ' + (desc_lookup or '')) if ' _extract_amounts' or True else []
                        vendor_here = _extract_vendor(agenda_title, desc_lookup) if ' _extract_vendor' or True else None
                        key_terms = _terms(agenda_title + ' ' + (desc_lookup or ''))
                        chunks = transcripts_by_meeting.get(mid) or []
                        best = None
                        best_score = 0.0
                        for v in chunks:
                            text_l = (v.get('content') or '').lower()
                            sc = _score(text_l, key_terms, vendor_here, [a.lower() for a in (amounts_here or [])])
                            if sc > best_score:
                                best = v
                                best_score = sc
                        if best and best_score >= 2.0:
                            start_s = int(float(best.get('start_time') or 0))
                            if start_s > 0:
                                base = c.get('url') or ''
                                # If current url is not YouTube, try resolve from meeting
                                if 'youtube' not in base:
                                    base = self._resolve_youtube_base(mid, c.get('meeting_date')) or base
                                if base:
                                    if 'youtu' in base.lower():
                                        sep = '&' if '?' in base else '?'
                                        c['url'] = f"{base}{sep}t={start_s}s"
                                    else:
                                        c['url'] = base
                                    c['timestamp'] = self._format_seconds(start_s)
                                    # Build more precise reason
                                    reason_bits: List[str] = []
                                    if vendor_here and vendor_here.lower() in (best.get('content') or '').lower():
                                        reason_bits.append(f"vendor match: {vendor_here}")
                                    amt_hits = [a for a in (amounts_here or []) if a.lower() in (best.get('content') or '').lower()]
                                    if amt_hits:
                                        reason_bits.append("amounts: " + ", ".join(amt_hits[:3]))
                                    term_hits = [t for t in key_terms if t in (best.get('content') or '').lower()]
                                    if term_hits:
                                        reason_bits.append("terms: " + ", ".join(term_hits[:4]))
                                    if any(k in (best.get('content') or '').lower() for k in ("motion","approve","approved","award","awarded","contract","bid","rfp")):
                                        reason_bits.append("mentions motion/approval")
                                    c['timestamp_reason'] = "Best matching transcript segment (" + "; ".join(reason_bits) + ")" if reason_bits else "Best matching transcript segment for this agenda item"
                                    enriched += 1
                        # If no decent match from already-retrieved transcripts, do a quick per-meeting dense search
                        if (best is None or best_score < 2.0):
                            try:
                                enrich_query = " ".join(key_terms[:8])
                                if vendor_here:
                                    enrich_query = f"{vendor_here} {enrich_query}".strip()
                                if amounts_here:
                                    enrich_query = f"{enrich_query} {' '.join(amounts_here[:2])}".strip()
                                                                # embed and search within meeting
                                emb = await self.get_embedding(enrich_query)
                                expr_graph = f"meeting_id == '{mid}' and chunk_type == 'youtube_transcript'"
                                dense_hits = await asyncio.to_thread(self._dense_search, emb, 4, expr_graph)
                                # pick best hit by simple token overlap
                                for dh in dense_hits:
                                    text_l = (dh.get('content') or '').lower()
                                    sc = _score(text_l, key_terms, vendor_here, [a.lower() for a in (amounts_here or [])])
                                    if sc > best_score:
                                        best = dh
                                        best_score = sc
                            except Exception:
                                pass
            except Exception:
                pass

            # Deduplicate citations while preserving distinct items and timestamps
            try:
                deduped: List[Dict[str, Any]] = []
                seen_agenda: set = set()            # (meeting_id, item_id)
                seen_transcript_ids: set = set()    # chunk_id when present
                seen_urls: set = set()              # fallback exact url match
                for c in filtered:
                    ctype = c.get('type')
                    url = c.get('url') or c.get('agenda_url')
                    mid = str(c.get('meeting_id') or '')
                    iid = str(c.get('item_id') or '')
                    # Prefer keeping distinct agenda items (by item_id)
                    if ctype == 'agenda_item' and mid and iid:
                        key = (mid, iid)
                        if key in seen_agenda:
                            continue
                        seen_agenda.add(key)
                        deduped.append(c)
                        continue
                    # For transcript citations, dedupe by unique chunk_id when available
                    cid = c.get('chunk_id')
                    if cid:
                        if cid in seen_transcript_ids:
                            continue
                        seen_transcript_ids.add(cid)
                        deduped.append(c)
                        continue
                    # Fallback: drop exact duplicate URLs
                    if url and url in seen_urls:
                        continue
                    if url:
                        seen_urls.add(url)
                    deduped.append(c)
                filtered = deduped
            except Exception:
                pass
 
            # Ensure we return a clean string answer
            if isinstance(answer_text, str) and answer_text.strip().startswith("{") and '"answer_paragraph"' in answer_text:
                try:
                    inner_obj = json.loads(answer_text)
                    if isinstance(inner_obj, dict):
                        answer_text2 = (inner_obj.get("answer_paragraph") or "").strip()
                        if answer_text2:
                            answer_text = answer_text2
                except Exception:
                    pass
 
            return {
                "answer": answer_text or "No indexed data for the current scope.",
                "source_citations": filtered,
                "answer_bullets": [],
            }
 
        except Exception as e:
            logger.error(f"❌ Failed to combine results: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your query. Please try again.",
                "source_citations": [],
                "vector_sources": vector_results[:5],
                "graph_sources": graph_results[:5],
                "total_sources": len(vector_results) + len(graph_results)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            # Milvus stats
            milvus_stats = {}
            if self.milvus_collection:
                milvus_stats = {
                    "collection_name": self.collection_name,
                    "entity_count": self.milvus_collection.num_entities
                }
            
            # Neo4j stats
            neo4j_stats = {}
            if self.neo4j_driver:
                with self.neo4j_driver.session() as session:
                    # Get node counts
                    node_counts = session.run("""
                        MATCH (n)
                        RETURN labels(n) as labels, count(n) as count
                        """)
                    
                    neo4j_stats = {
                        "node_counts": {",".join(record["labels"]): record["count"] for record in node_counts}
                    }
            
            return {
                "milvus": milvus_stats,
                "neo4j": neo4j_stats,
                "status": "healthy"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get system stats: {e}")
            return {"status": "error", "message": str(e)}
    
    def _get_youtube_url(self, meeting_id: str) -> Optional[str]:
        """Get YouTube URL for a meeting ID from stored metadata"""
        try:
            # Use the YouTube URL mapper to get actual URLs
            return get_youtube_url(meeting_id)
        except Exception as e:
            logger.error(f"Error getting YouTube URL for {meeting_id}: {e}")
            return None
    
    def _resolve_youtube_base(self, meeting_id: Optional[str], meeting_date: Optional[str]) -> Optional[str]:
        """Resolve a YouTube base URL by mapper, then Neo4j via meeting_id, then by meeting_date."""
        try:
            # 1) Mapper by meeting_id
            if meeting_id:
                url = self._get_youtube_url(meeting_id)
                if url:
                    return url
            # 2) Neo4j by meeting_id
            if self.neo4j_driver and meeting_id:
                with self.neo4j_driver.session() as session:
                    rec = session.run("MATCH (m:Meeting {meeting_id: $mid}) RETURN m.url AS url", {"mid": meeting_id}).single()
                    if rec and rec.get("url"):
                        return rec.get("url")
            # 3) Neo4j by meeting_date (exact string)
            if self.neo4j_driver and meeting_date:
                with self.neo4j_driver.session() as session:
                    rec = session.run("MATCH (m:Meeting {meeting_date: $d}) WHERE m.url IS NOT NULL RETURN m.url AS url LIMIT 1", {"d": meeting_date}).single()
                    if rec and rec.get("url"):
                        return rec.get("url")
        except Exception:
            pass
        return None
    
    def close(self):
        """Clean up connections"""
        try:
            if self.neo4j_driver:
                self.neo4j_driver.close()
            logger.info("✅ Connections closed")
        except Exception as e:
            logger.error(f"❌ Error closing connections: {e}")

    def _strip_boilerplate(self, text: Optional[str]) -> str:
        """Remove recurring agenda header/venue boilerplate so it does not affect retrieval.
        Conservative: only removes standalone header lines and obvious venue lines.
        """
        if not text:
            return ""
        t = text
        patterns = [
            r"(?mi)^\s*Town of Davie\s*$",
            r"(?mi)^\s*Town Hall Address:.*$",
            r"(?mi)^\s*Telephone:.*$",
            r"(?mi)^\s*Town Council Meeting\s*$",
            r"(?mi)^\s*REGULAR MEETING\s*$",
            r"(?mi)^\s*Location:.*$",
            r"(?mi)^\s*Cypress Room\s*$",
            r"(?mi)^\s*\d{2,5}\s+S?\s*Pine\s+Island\s+Rd.*$",
            r"(?mi)^\s*Davie,\s*FL\b.*$",
        ]
        for p in patterns:
            t = re.sub(p, "", t)
        # collapse excessive blank lines
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

    def _simple_tokenize(self, text: str) -> List[str]:
        t = (text or "").lower()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        return [tok for tok in t.split() if tok]

    def _bm25_rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not BM25_RERANK or BM25Okapi is None or not candidates:
            return candidates[:top_k]
        try:
            corpus_tokens = [self._simple_tokenize(c.get("content") or c.get("text") or "") for c in candidates]
            bm25 = BM25Okapi(corpus_tokens)
            query_tokens = self._simple_tokenize(query)
            scores = bm25.get_scores(query_tokens)
            for cand, s in zip(candidates, scores):
                cand["bm25_score"] = float(s)
            candidates.sort(key=lambda x: x.get("bm25_score", 0.0), reverse=True)
            return candidates[:top_k]
        except Exception:
            return candidates[:top_k]

    async def _synthesize_answer(self, query: str, candidates: List[Dict[str, Any]], graph_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Produce a combined result list and attach a synthesized answer summary to the first element.
        Returns the candidates list (possibly truncated to top_k elsewhere) and relies on the caller to build
        the final response. Lightweight for Vercel.
        """
        # Ensure consistent structure
        combined: List[Dict[str, Any]] = []
        for c in (candidates or [])[:10]:
            combined.append({
                "title": c.get("title") or c.get("chunk_id") or "Result",
                "content": c.get("content") or c.get("text") or "",
                "meeting_id": c.get("meeting_id"),
                "meeting_date": c.get("meeting_date"),
                "type": c.get("type") or c.get("chunk_type") or "document",
                "url": c.get("url") or c.get("agenda_url"),
                "score": c.get("score") or c.get("dense_score") or c.get("sparse_score"),
            })
        # Append graph references lightly
        for g in (graph_results or [])[:5]:
            combined.append({
                "title": g.get("title") or g.get("entity") or "Graph Entity",
                "content": g.get("summary") or g.get("description") or "",
                "type": "graph",
                "score": g.get("score"),
            })
        return combined

    _zilliz_fields_cache: Dict[str, List[str]] = {}

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

    # Helper: fetch entities by ids to hydrate missing fields
    async def _zilliz_get_entities_by_ids(
        self,
        milvus_uri: str,
        milvus_token: str,
        collection_name: str,
        ids: List[Any],
        output_fields: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        try:
            if httpx is None or not ids:
                return {}
            url = f"{milvus_uri.rstrip('/')}/v2/vectordb/entities/get"
            payload = {
                "collectionName": collection_name,
                "ids": ids,
                "outputFields": output_fields,
            }
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(url, json=payload, headers={"Authorization": f"Bearer {milvus_token}", "Content-Type": "application/json"})
                resp.raise_for_status()
                body = resp.json() or {}
            data = body.get("data") or []
            out: Dict[str, Dict[str, Any]] = {}
            for ent in data:
                ent_id = ent.get("id")
                if ent_id is not None:
                    out[str(ent_id)] = ent
            logger.info(f"Zilliz get entities: fetched={len(out)}")
            return out
        except Exception as e:
            logger.warning(f"Zilliz get entities error: {e}")
            return {}

    async def _zilliz_search(self, query_embedding: List[float], top_k: int, expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search Zilliz/Milvus over HTTPS using the VectorDB REST API.
        Requires MILVUS_URI, MILVUS_TOKEN, and self.collection_name.
        """
        milvus_uri = os.getenv("MILVUS_URI")
        milvus_token = os.getenv("MILVUS_TOKEN")
        if not milvus_uri or not milvus_token or not query_embedding:
            logger.debug(
                f"_zilliz_search skipped: uri={_present(milvus_uri)}, token={_present(milvus_token)}, emb={'set' if query_embedding else 'unset'}"
            )
            return []
        try:
            search_url = f"{milvus_uri.rstrip('/')}/v2/vectordb/entities/search"
            headers = {
                "Authorization": "Bearer ****",  # redacted
                "Content-Type": "application/json",
            }
            vector_field = os.getenv("MILVUS_VECTOR_FIELD", "embedding")
            text_field = os.getenv("MILVUS_TEXT_FIELD", "content")
            meta_field = os.getenv("MILVUS_METADATA_FIELD", "metadata")
            meet_id_field = os.getenv("MILVUS_MEETING_ID_FIELD", "meeting_id")
            meet_date_field = os.getenv("MILVUS_MEETING_DATE_FIELD", "meeting_date")
            chunk_id_field = os.getenv("MILVUS_CHUNK_ID_FIELD", "chunk_id")
            title_field = os.getenv("MILVUS_TITLE_FIELD", "chunk_id")
            requested_fields = [text_field, meta_field, meet_id_field, meet_date_field, chunk_id_field, title_field]
            # Filter output fields to only those that exist to avoid "field ... not exist"
            available = await self._zilliz_describe_fields(milvus_uri, milvus_token, self.collection_name)
            ofields = [f for f in requested_fields if f and f in (available or [])]
            async def _do_search(use_filter: bool) -> Tuple[List[Dict[str, Any]], List[Any]]:
                payload = {
                    "collectionName": self.collection_name,
                    "data": [query_embedding],
                    "limit": int(top_k),
                    "outputFields": ofields,
                    "annsField": vector_field,
                }
                if use_filter and expr:
                    payload["filter"] = expr
                logger.info(
                    f"Zilliz search host={_host_only(milvus_uri)} collection={self.collection_name} field={vector_field} k={top_k} expr={'set' if (use_filter and expr) else 'unset'} ofields={ofields}"
                )
                if httpx is None:
                    logger.warning("httpx not available; cannot call Zilliz HTTP API")
                    return [], []
                ids_local: List[Any] = []
                async def _call_once() -> Dict[str, Any]:
                    async with httpx.AsyncClient(timeout=20) as client:
                        resp = await client.post(search_url, json=payload, headers={"Authorization": f"Bearer {milvus_token}", "Content-Type": "application/json"})
                        logger.debug(f"Zilliz status={resp.status_code}")
                        resp.raise_for_status()
                        return resp.json() or {}
                # Since we are already in an async function, simply await
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
                    item = {
                        "title": (hit.get(title_field) if title_field else None) or (md or {}).get("title") or (hit.get(chunk_id_field) if chunk_id_field else None) or hit.get("chunk_id") or "Result",
                        "content": text,
                        "meeting_id": (hit.get(meet_id_field) if meet_id_field else None) or (md or {}).get("meeting_id"),
                        "meeting_date": (hit.get(meet_date_field) if meet_date_field else None) or (md or {}).get("meeting_date"),
                        "chunk_id": (hit.get(chunk_id_field) if chunk_id_field else None) or (md or {}).get("chunk_id"),
                        "type": (md or {}).get("type") or "document",
                        "url": (md or {}).get("url") or (md or {}).get("agenda_url"),
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
            # Fallback: hydrate by ids if we have no content in any hit
            if hits_len and not any((h.get("content") or "").strip() for h in hits):
                # choose a fallback text field if not present in ofields
                fallback_text = None
                prefs = [os.getenv("MILVUS_TEXT_FIELD"), "text", "content", "body", "document", "chunk", "chunk_text"]
                for p in prefs:
                    if p and p in (available or []):
                        fallback_text = p
                        break
                hydrate_fields = [f for f in {fallback_text, meta_field, meet_id_field, meet_date_field, chunk_id_field, title_field} if f and ((available is None) or (f in (available or [])))]
                if ids and hydrate_fields:
                    logger.info(f"Zilliz fallback hydration by ids: count={len(ids)} fields={hydrate_fields}")
                    ent_map = await self._zilliz_get_entities_by_ids(milvus_uri, milvus_token, self.collection_name, ids, hydrate_fields)
                    # merge
                    for i, h in enumerate(hits):
                        key = str(ids[i]) if i < len(ids) else None
                        ent = ent_map.get(key) if key else None
                        if ent:
                            txt = (ent.get(fallback_text) if fallback_text else None) or h.get("content") or ""
                            md2 = (ent.get(meta_field) if meta_field else None) or {}
                            if isinstance(md2, str):
                                try:
                                    md2 = json.loads(md2)
                                except Exception:
                                    pass
                            h["content"] = txt
                            h["title"] = h.get("title") or (ent.get(title_field) if title_field else None) or (md2 or {}).get("title") or h.get("title")
                            h["meeting_id"] = h.get("meeting_id") or ent.get(meet_id_field) or (md2 or {}).get("meeting_id")
                            h["meeting_date"] = h.get("meeting_date") or ent.get(meet_date_field) or (md2 or {}).get("meeting_date")
                            h["chunk_id"] = h.get("chunk_id") or ent.get(chunk_id_field) or (md2 or {}).get("chunk_id")
                elif ids and not hydrate_fields:
                    # attempt hydration with no outputFields to let server decide defaults
                    logger.info("Zilliz fallback hydration by ids with no explicit fields")
                    ent_map = await self._zilliz_get_entities_by_ids(milvus_uri, milvus_token, self.collection_name, ids, [])
                    for i, h in enumerate(hits):
                        key = str(ids[i]) if i < len(ids) else None
                        ent = ent_map.get(key) if key else None
                        if ent:
                            txt = ent.get("text") or ent.get("content") or h.get("content") or ""
                            h["content"] = txt
            if hits_len:
                prev = (hits[0].get("content") or "")[:120].replace("\n", " ")
                logger.debug(f"Zilliz top title='{hits[0].get('title')}' preview='{prev}'")
            return hits
        except Exception as e:
            logger.error(f"Zilliz search error: {e}")
            return []

    async def _neo4j_query_api_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Query Neo4j over HTTPS using the Query API. Requires NEO4J_QUERY_API_URL, NEO4J_USERNAME, NEO4J_PASSWORD.
        This uses a simple CONTAINS match over common AgendaItem fields as a lightweight fallback.
        """
        api_url = os.getenv("NEO4J_QUERY_API_URL")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        if not api_url or not username or not password or httpx is None:
            logger.debug(
                f"neo4j query skipped: url={_present(api_url)} user={_present(username)} pass={_present(password)} httpx={'set' if httpx else 'unset'}"
            )
            return []
        try:
            logger.info(f"Neo4j Query API host={_host_only(api_url)} k={top_k}")
            # Basic auth header
            token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("utf-8")
            headers = {
                "Authorization": f"Basic {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            statement = (
                "MATCH (a:AgendaItem) "
                "WHERE toLower(a.title) CONTAINS toLower($q) "
                "   OR toLower(a.description) CONTAINS toLower($q) "
                "   OR toLower(a.summary) CONTAINS toLower($q) "
                "RETURN a.title AS title, a.description AS content, a.meeting_id AS meeting_id, "
                "       a.meeting_date AS meeting_date, a.item_id AS item_id "
                "LIMIT $k"
            )
            payload = {"statement": statement, "parameters": {"q": query_text, "k": int(top_k)}}
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(api_url, json=payload, headers=headers)
                logger.debug(f"Neo4j status={resp.status_code}")
                if resp.status_code not in (200, 202):
                    logger.error(f"Neo4j non-success status={resp.status_code}")
                    return []
                body = resp.json() or {}
            logger.debug(f"Neo4j body keys={list(body.keys())}")
            # Parse results
            rows: List[List[Any]] = []
            if isinstance(body.get("data"), dict):
                rows = body["data"].get("values") or []
            elif isinstance(body.get("data"), list):
                rows = body["data"]
            logger.info(f"Neo4j rows={len(rows)}")
            results: List[Dict[str, Any]] = []
            for row in rows:
                try:
                    title = row[0]
                    content = row[1]
                    meeting_id = row[2]
                    meeting_date = row[3]
                    item_id = row[4]
                except Exception:
                    title = (row or {}).get("title")
                    content = (row or {}).get("content")
                    meeting_id = (row or {}).get("meeting_id")
                    meeting_date = (row or {}).get("meeting_date")
                    item_id = (row or {}).get("item_id")
                results.append({
                    "title": title or "Agenda Item",
                    "content": content or "",
                    "meeting_id": meeting_id,
                    "meeting_date": meeting_date,
                    "item_id": item_id,
                    "type": "agenda_item",
                })
            if results:
                prev = (results[0].get("content") or "")[:120].replace("\n", " ")
                logger.debug(f"Neo4j top title='{results[0].get('title')}' preview='{prev}'")
            logger.info(f"Neo4j results={len(results)}")
            return results
        except Exception as e:
            logger.error(f"Neo4j query error: {e}")
            return []

# Pydantic models for API
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    top_k: int = Field(default=10, description="Number of results to return", ge=1, le=100)
    concise_mode: bool = Field(default=True, description="If true, prefer concise bullets in response")
    session_id: Optional[str] = Field(default=None, description="Session identifier for conversational context")

class SearchResponse(BaseModel):
    answer: str = Field(..., description="Combined answer from hybrid search")
    source_citations: List[Dict[str, Any]] = Field(default=[], description="Top relevant source citations with links")
    vector_sources: List[Dict[str, Any]] = Field(default=[], description="Vector search results")
    graph_sources: List[Dict[str, Any]] = Field(default=[], description="Graph search results")
    total_sources: int = Field(..., description="Total number of sources found")
    answer_bullets: List[Dict[str, Any]] = Field(default=[], description="Bullets for the answer, if available")

# Initialize the hybrid RAG system
rag_system = HybridRAGSystem()

# Simple in-memory conversation store: session_id -> deque of turns (maxlen=10)
CONVERSATIONS: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
telemetry_init()
try:
    ts = telemetry_status()
    if ts.get("postgres_connected"):
        logger.info("✅ Telemetry connected to Postgres")
    else:
        logger.info("ℹ️ Telemetry not connected to Postgres (using file fallback)")
except Exception as e:
    logger.info(f"ℹ️ Telemetry status unavailable: {e}")

# FastAPI app
app = FastAPI(
    title="Capstone Hybrid RAG System",
    description="Government transparency Q&A system combining vector and graph search",
    version="1.0.0"
)

# Mount static files and templates conditionally to avoid import-time failures in headless runs
try:
    if os.path.isdir("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
    else:
        logger.warning("Static directory not found; skipping static mount")
except Exception as e:
    logger.warning(f"Skipping static mount due to error: {e}")

try:
    templates = None
    # Probe multiple likely template locations
    candidate_templates = [
        "capstone/templates",
        "templates",
        "/var/task/capstone/templates",
        "/var/task/templates",
    ]
    for tp in candidate_templates:
        if os.path.isdir(tp):
            templates = Jinja2Templates(directory=tp)
            break
    if templates is None:
        logger.warning("Templates directory not found; index route will return a basic response")
except Exception as e:
    templates = None
    logger.warning(f"Skipping templates setup due to error: {e}")

# CORS configuration
try:
    allowed_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
    cors_origins = ["*"] if allowed_origins_env.strip() == "*" else [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception as e:
    logger.warning(f"Skipping CORS setup due to error: {e}")

# Rate limiting setup
try:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
except Exception as e:
    logger.warning(f"Skipping rate limiting setup due to error: {e}")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main application page"""
    if templates is None:
        # Embedded minimal HTML fallback so the UI still renders on Vercel
        html = """
        <!DOCTYPE html>
        <html><head><meta charset=\"utf-8\"/><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>
        <title>Town of Davie Citizen Watch</title>
        <link href=\"https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css\" rel=\"stylesheet\">
        </head>
        <body class=\"bg-gray-50\">
          <div class=\"max-w-3xl mx-auto mt-10 p-6 bg-white shadow rounded\">
            <h1 class=\"text-2xl font-bold mb-2\">Town of Davie Citizen Watch</h1>
            <p class=\"text-gray-600 mb-6\">Service is running. Default UI loaded because no templates directory was found.</p>
            <form id=\"chat-form\" class=\"flex space-x-2\" onsubmit=\"event.preventDefault(); sendMessage();\">
              <input id=\"message-input\" class=\"flex-1 border rounded px-3 py-2\" placeholder=\"Ask a question...\"/>
              <button class=\"bg-blue-600 text-white px-4 py-2 rounded\">Send</button>
            </form>
            <div id=\"chat-messages\" class=\"mt-4 space-y-3\"></div>
          </div>
          <script>
            async function sendMessage(){
              const input = document.getElementById('message-input');
              const msg = (input.value||'').trim();
              if(!msg) return;
              input.value='';
              const box = document.getElementById('chat-messages');
              const user = document.createElement('div'); user.className='p-3 bg-green-50 rounded'; user.textContent=msg; box.appendChild(user);
              try{
                const r = await fetch('/api/search', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query: msg, top_k: 10})});
                const data = await r.json();
                const ai = document.createElement('div'); ai.className='p-3 bg-blue-50 rounded'; ai.textContent=(data && data.answer) || 'No answer'; box.appendChild(ai);
              }catch(e){ const er=document.createElement('div'); er.className='p-3 bg-red-50 rounded'; er.textContent='Error contacting API'; box.appendChild(er); }
              box.scrollTop = box.scrollHeight;
            }
          </script>
        </body></html>
        """
        return HTMLResponse(content=html, status_code=200)
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search", response_model=SearchResponse)
# Apply a conservative default limit; configurable via RATE_LIMIT_SEARCH env var not implemented for simplicity
async def search(req: Request, request: SearchRequest):
    """Perform hybrid search with optional conversational context tracking."""
    try:
        start_t = time.time()
        # Basic input sanitation beyond Pydantic constraints
        q = (request.query or "").strip()
        if not q:
            raise HTTPException(status_code=422, detail="Query must not be empty")
        top_k = request.top_k
        try:
            max_top_k = int(os.getenv("MAX_TOP_K", "100"))
        except Exception:
            max_top_k = 100
        if top_k > max_top_k:
            top_k = max_top_k
        session_id = request.session_id or str(uuid.uuid4())

        # Track session
        client_ip = req.client.host if req and req.client else None
        user_agent = req.headers.get("user-agent") if req and req.headers else None
        hashed_ip = None
        try:
            if client_ip:
                salt = os.getenv("IP_HASH_SALT", "davie_salt")
                hashed_ip = hashlib.sha256(f"{salt}:{client_ip}".encode("utf-8")).hexdigest()
        except Exception:
            hashed_ip = None
        try:
            upsert_session(session_id, hashed_ip, user_agent)
        except Exception as e:
            logger.warning(f"telemetry upsert_session failed: {e}")

        # Prepare prior context summary (last up to 3 turns compact)
        prior_turns = list(CONVERSATIONS.get(session_id, deque()))
        prior_context = prior_turns[-3:]
 
        # Simple follow-up heuristic
        ql = q.lower()
        is_follow_up = (len(prior_turns) > 0) and (len(q) <= 120) and any(w in ql for w in ["that", "this", "it", "they", "the project", "discussion", "those"])
 
        # Extract prior meeting_ids and item_ids from citations
        prior_meeting_ids: List[str] = []
        prior_item_ids: List[str] = []
        for turn in reversed(prior_context):
            for c in turn.get("citations", []) or []:
                mid = c.get("meeting_id")
                if mid and mid not in prior_meeting_ids:
                    prior_meeting_ids.append(str(mid))
                iid = c.get("item_id")
                if iid and str(iid) not in prior_item_ids:
                    prior_item_ids.append(str(iid))
 
        logger.info(f"/api/search start q_len={len(q)} top_k={top_k}")
        # Run hybrid search
        t_hybrid_start = time.time()
        results = await rag_system.hybrid_search(q, top_k, prior_context=prior_context)
        t_hybrid_ms = int((time.time() - t_hybrid_start) * 1000)
        logger.info(f"/api/search hybrid_ms={t_hybrid_ms} vec={len(results.get('vector_results') or [])} graph={len(results.get('graph_results') or [])}")
 
        combined = results.get("combined_results")
        # Normalize combined: may be list (light mode) or dict (full mode)
        if isinstance(combined, list):
            top_texts = [c.get("content") for c in combined if isinstance(c, dict) and c.get("content")]
            answer_text = "\n\n".join(top_texts[:2]) or "I couldn't find relevant information."
            citations = []
            for c in combined[:5]:
                if isinstance(c, dict):
                    citations.append({
                        "title": c.get("title"),
                        "url": c.get("url"),
                        "meeting_id": c.get("meeting_id"),
                        "meeting_date": c.get("meeting_date"),
                        "type": c.get("type", "document"),
                    })
            combined = {"answer": answer_text, "source_citations": citations}
        elif not isinstance(combined, dict):
            combined = {}
        vector_sources = combined.get("vector_sources") or results.get("vector_results") or []
        graph_sources = combined.get("graph_sources") or results.get("graph_results") or []
        total_sources = combined.get("total_sources") or (len(vector_sources) + len(graph_sources))

        # No-results fallback
        if not vector_sources and not graph_sources:
            logger.info("/api/search no-results: vector=0 graph=0; returning fallback message")
            combined.setdefault("answer", "I couldn't find relevant information for your question. Please try rephrasing or be more specific (names, dates, locations).")
            combined.setdefault("source_citations", [])
            total_sources = 0

        # Final safety: unwrap accidental JSON-in-string in answer
        try:
            ans = combined.get("answer")
            if isinstance(ans, str) and ans.strip().startswith("{") and '"answer_paragraph"' in ans:
                inner = json.loads(ans)
                if isinstance(inner, dict):
                    para = (inner.get("answer_paragraph") or "").strip()
                    if para:
                        combined["answer"] = para
        except Exception:
            pass

        # Follow-up fallback: if low recall, augment query with prior turn to re-run retrieval
        if is_follow_up and total_sources < max(3, top_k // 3):
            try:
                last_q = prior_context[-1]["question"] if prior_context else ""
                aug_q = (last_q + " " + q).strip() if last_q else q
                results2 = await rag_system.hybrid_search(aug_q, top_k, prior_context=prior_context)
                combined2 = results2.get("combined_results", {})
                vector2 = combined2.get("vector_sources") or results2.get("vector_results") or []
                graph2 = combined2.get("graph_sources") or results2.get("graph_results") or []
                total2 = combined2.get("total_sources") or (len(vector2) + len(graph2))
                if total2 > total_sources:
                    results = results2
                    combined = combined2
                    vector_sources = vector2
                    graph_sources = graph2
                    total_sources = total2
            except Exception:
                pass
 
        # Save turn in memory
        CONVERSATIONS[session_id].append({
            "question": q,
            "answer": combined.get("answer", ""),
            "citations": combined.get("source_citations", []),
            "vector_count": len(vector_sources),
            "graph_count": len(graph_sources),
        })

        # Telemetry lineage/perf/cost (best-effort)
        model = os.getenv("FINAL_LLM_MODEL", "gpt-4o-mini")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        rerank_model = os.getenv("RERANKING_MODEL", "gpt-4o-mini")
        dataset_version = os.getenv("DATASET_VERSION", "v1")
        collection_name = getattr(rag_system, "collection_name", "")
        schema_hash = os.getenv("SCHEMA_HASH", "")
        # timings are coarse for now
        event = {
            "session_id": session_id,
            "turn_index": len(CONVERSATIONS[session_id]),
            "role": "assistant",
            "question": q,
            "answer": combined.get("answer", ""),
            "citations": combined.get("source_citations", []),
            "vector_count": len(vector_sources),
            "graph_count": len(graph_sources),
            "dense_ms": None, "sparse_ms": None, "graph_ms": None, "fuse_ms": None,
            "rerank_ms": None, "final_llm_ms": None, "total_ms": int((time.time() - start_t) * 1000),
            "model": model, "rerank_model": rerank_model, "embedding_model": embedding_model,
            "dataset_version": dataset_version, "collection_name": collection_name, "schema_hash": schema_hash,
            "vector_params": {"k": top_k, "alpha": getattr(rag_system, "alpha", None), "multiplier": os.getenv("INITIAL_SEARCH_MULTIPLIER", "5")},
            "retrieved_ids": {
                "vector": [s.get("chunk_id") for s in vector_sources[:10]],
                "graph": [s.get("agenda_item", {}).get("item_id") for s in graph_sources[:10]]
            },
            "scores": {
                "vector_top": [s.get("fused_score") for s in vector_sources[:5] if s.get("fused_score") is not None]
            },
            "token_prompt": None, "token_completion": None, "token_total": None, "cost_usd": None,
            "rate_limit_status": "allowed", "captcha_score": None, "blocked_reason": None,
            "validation_flags": {"prompt_injection": False},
            "pii_redaction_applied": False, "pii_fields_found": 0,
        }
        try:
            record_event(event)
        except Exception as e:
            logger.warning(f"telemetry record_event failed: {e}")

        # Include session_id in a header for the client to persist
        headers = {"X-Session-Id": session_id}
        return JSONResponse(
            content=SearchResponse(
                answer=combined.get("answer", ""),
                source_citations=combined.get("source_citations", []),
                vector_sources=vector_sources,
                graph_sources=graph_sources,
                total_sources=total_sources,
                answer_bullets=combined.get("answer_bullets") or []
            ).model_dump(),
            headers=headers
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/api/search error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    return rag_system.get_system_stats()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/telemetry/status")
async def telemetry_status_endpoint():
    try:
        ts = telemetry_status()
        resp = {"postgres_connected": bool(ts.get("postgres_connected"))}
        # If connected, try a quick count to confirm access
        if resp["postgres_connected"]:
            try:
                # Use direct connection to avoid importing heavy libs
                from capstone.telemetry import _DB_CONN  # type: ignore
                with _DB_CONN.cursor() as cur:  # type: ignore
                    cur.execute("SELECT COUNT(*) FROM sessions")
                    s_count = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM conversation_events")
                    e_count = cur.fetchone()[0]
                resp.update({"sessions": s_count, "events": e_count})
            except Exception:
                pass
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Remote service endpoints
VECTOR_API_URL = os.getenv("VECTOR_API_URL")
GRAPH_API_URL = os.getenv("GRAPH_API_URL")

async def _remote_vector_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    if httpx is None or not VECTOR_API_URL:
        return []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(VECTOR_API_URL, json={"query": query, "top_k": top_k})
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
    except Exception:
        return []

async def _remote_graph_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    if httpx is None or not GRAPH_API_URL:
        return []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(GRAPH_API_URL, json={"query": query, "top_k": top_k})
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
    except Exception:
        return []

# YouTube URL helper (safe import)
try:
    from capstone.youtube_url_mapper import get_youtube_url  # type: ignore
except Exception:
    try:
        from youtube_url_mapper import get_youtube_url  # type: ignore
    except Exception:
        def get_youtube_url(meeting_id: str):
            return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 