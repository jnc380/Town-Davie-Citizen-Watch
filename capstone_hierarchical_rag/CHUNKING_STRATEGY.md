# Chunking Strategy for Hybrid RAG System

## 🎯 Overview

The hybrid RAG system processes multiple file types with different chunking strategies optimized for each content type. This document explains the chunking approaches for YouTube transcripts, agenda documents, and other government data sources.

## 📊 File Types and Chunking Strategies

### 1. YouTube Transcripts (JSON Format)

**File Structure:**
```json
[
  {
    "text": "call the meeting to order where",
    "start": 231.879,
    "duration": 5.041
  },
  {
    "text": "everybody please rise for the Pledge of",
    "start": 233.28,
    "duration": 3.64
  }
]
```

**Chunking Strategy: Semantic Boundary Chunking**
- **Chunk Size**: 1000 characters (configurable)
- **Overlap**: 200 characters for context preservation
- **Boundary Detection**: Natural speech pauses and topic transitions
- **Temporal Preservation**: Maintains start time and duration for video linking

**Implementation:**
```python
def process_transcript_chunks(self, transcript_data: List[Dict], chunk_size: int = 1000):
    chunks = []
    current_chunk = []
    current_text = ""
    
    for segment in transcript_data:
        text = segment.get("text", "").strip()
        if not text or text == "[Music]" or text == "e":
            continue
        
        current_chunk.append(segment)
        current_text += " " + text
        
        # Create chunk when size limit is reached
        if len(current_text) >= chunk_size:
            chunks.append({
                "text": current_text.strip(),
                "segments": current_chunk,
                "start": current_chunk[0].get("start", 0),
                "duration": sum(seg.get("duration", 0) for seg in current_chunk)
            })
            current_chunk = []
            current_text = ""
    
    return chunks
```

**Benefits:**
- ✅ Preserves temporal context for video timestamps
- ✅ Maintains speaker continuity
- ✅ Removes noise (music, silence, filler words)
- ✅ Semantic coherence within chunks

### 2. Agenda Documents (PDF/JSON Metadata)

**File Structure:**
```json
{
  "pdf_links": [
    {
      "url": "https://davie.novusagenda.com/agendapublic/CoverSheet.aspx?ItemID=8332",
      "text": "AGREEMENT- A RESOLUTION OF THE TOWN OF DAVIE...",
      "item_id": "8332",
      "meeting_id": "578"
    }
  ]
}
```

**Chunking Strategy: Hierarchical Item-Based Chunking**
- **Primary Chunk**: Each agenda item as a separate chunk
- **Secondary Chunk**: Detailed PDF content when available
- **Metadata Preservation**: Item IDs, meeting IDs, and relationships
- **Cross-Reference Linking**: Links between related agenda items

**Implementation:**
```python
def extract_agenda_items(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    agenda_items = []
    pdf_links = metadata.get("pdf_links", [])
    
    for i, link in enumerate(pdf_links):
        agenda_items.append({
            "item_id": link.get("item_id", f"item_{i}"),
            "text": link.get("text", ""),
            "url": link.get("url", ""),
            "order": i,
            "type": "agenda_item",
            "meeting_id": link.get("meeting_id", "")
        })
    
    return agenda_items
```

**Benefits:**
- ✅ Preserves hierarchical structure
- ✅ Maintains item relationships
- ✅ Enables cross-referencing between meetings
- ✅ Supports resolution tracking over time

### 3. PDF Documents (Agenda Attachments)

**File Structure:** Binary PDF files with nested structure
```
item_8332_578_nested_1.pdf
item_8332_578_nested_2.pdf
item_8332_578_nested_3.pdf
```

**Chunking Strategy: Multi-Level PDF Chunking**
- **Level 1**: Document-level chunks (entire PDF)
- **Level 2**: Page-level chunks (individual pages)
- **Level 3**: Section-level chunks (logical document sections)
- **Level 4**: Paragraph-level chunks (detailed content)

**Implementation:**
```python
def process_pdf_chunks(self, pdf_files: List[str], chunk_level: str = "page"):
    chunks = []
    
    for pdf_file in pdf_files:
        if chunk_level == "document":
            # Process entire document
            chunks.append(self._extract_document_text(pdf_file))
        elif chunk_level == "page":
            # Process page by page
            pages = self._extract_pages(pdf_file)
            for page_num, page_text in enumerate(pages):
                chunks.append({
                    "text": page_text,
                    "source": pdf_file,
                    "page": page_num,
                    "chunk_type": "pdf_page"
                })
        elif chunk_level == "section":
            # Process by logical sections
            sections = self._extract_sections(pdf_file)
            for section in sections:
                chunks.append({
                    "text": section["text"],
                    "source": pdf_file,
                    "section": section["title"],
                    "chunk_type": "pdf_section"
                })
    
    return chunks
```

**Benefits:**
- ✅ Handles complex document structures
- ✅ Preserves document hierarchy
- ✅ Enables fine-grained search
- ✅ Maintains page-level citations

## 🔄 Chunking Pipeline

### Step 1: Content Type Detection
```python
def detect_content_type(self, file_path: str) -> str:
    """Detect the type of content for appropriate chunking"""
    if file_path.endswith("_transcript.json"):
        return "youtube_transcript"
    elif file_path.endswith("meeting_metadata.json"):
        return "agenda_metadata"
    elif file_path.endswith(".pdf"):
        return "pdf_document"
    else:
        return "unknown"
```

### Step 2: Content-Specific Processing
```python
def process_content_by_type(self, content: Any, content_type: str) -> List[Dict]:
    """Apply appropriate chunking strategy based on content type"""
    if content_type == "youtube_transcript":
        return self.process_transcript_chunks(content)
    elif content_type == "agenda_metadata":
        return self.extract_agenda_items(content)
    elif content_type == "pdf_document":
        return self.process_pdf_chunks(content)
    else:
        return self.default_chunking(content)
```

### Step 3: Metadata Enrichment
```python
def enrich_chunks_with_metadata(self, chunks: List[Dict], source_info: Dict) -> List[Dict]:
    """Add metadata to chunks for better search and retrieval"""
    enriched_chunks = []
    
    for chunk in chunks:
        enriched_chunk = {
            **chunk,
            "source_type": source_info.get("type"),
            "meeting_id": source_info.get("meeting_id"),
            "meeting_date": source_info.get("meeting_date"),
            "meeting_type": source_info.get("meeting_type"),
            "chunk_id": f"{source_info.get('meeting_id')}_{chunk.get('chunk_type', 'unknown')}_{len(enriched_chunks)}"
        }
        enriched_chunks.append(enriched_chunk)
    
    return enriched_chunks
```

## 📈 Chunking Optimization

### Performance Considerations

1. **Chunk Size Optimization**
   - **Small chunks (500-1000 chars)**: Better precision, more granular search
   - **Large chunks (1000-2000 chars)**: Better context, fewer chunks to process
   - **Hybrid approach**: Variable chunk sizes based on content type

2. **Overlap Strategy**
   - **Transcripts**: 200-300 character overlap for speech continuity
   - **Documents**: 100-200 character overlap for context preservation
   - **Agendas**: No overlap (discrete items)

3. **Semantic Boundary Detection**
   - **Sentence boundaries**: Natural language processing
   - **Topic transitions**: Keyword and context analysis
   - **Speaker changes**: Transcript-specific markers

### Quality Assurance

1. **Content Validation**
   ```python
   def validate_chunk_quality(self, chunk: Dict) -> bool:
       """Validate chunk quality before storage"""
       text = chunk.get("text", "")
       
       # Check minimum length
       if len(text) < 50:
           return False
       
       # Check for noise
       noise_patterns = ["[Music]", "e", "...", "---"]
       if any(pattern in text for pattern in noise_patterns):
           return False
       
       # Check for meaningful content
       if len(text.split()) < 10:
           return False
       
       return True
   ```

2. **Duplicate Detection**
   ```python
   def detect_duplicates(self, chunks: List[Dict]) -> List[Dict]:
       """Remove duplicate or near-duplicate chunks"""
       unique_chunks = []
       seen_texts = set()
       
       for chunk in chunks:
           text_hash = hashlib.md5(chunk["text"].encode()).hexdigest()
           if text_hash not in seen_texts:
               unique_chunks.append(chunk)
               seen_texts.add(text_hash)
       
       return unique_chunks
   ```

## 🎯 Chunking Strategy Summary

| File Type | Chunking Strategy | Chunk Size | Overlap | Special Features |
|-----------|------------------|------------|---------|------------------|
| **YouTube Transcripts** | Semantic Boundary | 1000 chars | 200 chars | Temporal preservation |
| **Agenda Metadata** | Item-Based | Variable | None | Hierarchical structure |
| **PDF Documents** | Multi-Level | Variable | 100-200 chars | Document hierarchy |
| **Meeting Info** | Metadata-Only | N/A | N/A | Cross-referencing |

## 🔧 Implementation in Hybrid RAG

### Vector Store Integration
```python
def store_chunks_in_vector_db(self, chunks: List[Dict]):
    """Store processed chunks in Milvus vector database"""
    embeddings = []
    metadata = []
    
    for chunk in chunks:
        # Generate embedding
        embedding = await self.get_embedding(chunk["text"])
        embeddings.append(embedding)
        
        # Prepare metadata
        metadata.append({
            "content": chunk["text"],
            "chunk_id": chunk["chunk_id"],
            "meeting_id": chunk["meeting_id"],
            "meeting_type": chunk["meeting_type"],
            "chunk_type": chunk["chunk_type"],
            "start_time": chunk.get("start_time", 0),
            "duration": chunk.get("duration", 0),
            "metadata": json.dumps(chunk.get("metadata", {}))
        })
    
    # Insert into Milvus
    self._insert_into_milvus(metadata, embeddings)
```

### Graph Database Integration
```python
def store_relationships_in_graph_db(self, chunks: List[Dict]):
    """Store chunk relationships in Neo4j graph database"""
    entities = []
    relationships = []
    
    for chunk in chunks:
        # Create chunk entity
        chunk_entity = GraphEntity(
            entity_id=chunk["chunk_id"],
            entity_type="Chunk",
            properties={
                "text": chunk["text"][:200],  # Truncated for display
                "chunk_type": chunk["chunk_type"],
                "meeting_id": chunk["meeting_id"],
                "meeting_date": chunk["meeting_date"]
            },
            labels=["Chunk", chunk["chunk_type"]]
        )
        entities.append(chunk_entity)
        
        # Create relationship to meeting
        relationship = GraphRelationship(
            source_id=chunk["meeting_id"],
            target_id=chunk["chunk_id"],
            relationship_type="CONTAINS_CHUNK",
            properties={"order": chunk.get("order", 0)}
        )
        relationships.append(relationship)
    
    # Insert into Neo4j
    self._insert_into_neo4j(entities, relationships)
```

## 📊 Expected Results

With this chunking strategy, the system will create:

- **~500-1000 transcript chunks** from 52 YouTube meetings
- **~200-300 agenda item chunks** from meeting agendas
- **~100-200 PDF document chunks** from agenda attachments
- **Total: ~800-1500 chunks** for comprehensive search

This provides excellent coverage for both semantic search (vector) and relationship search (graph) while maintaining the temporal and hierarchical context essential for government transparency applications.

---

**Key Benefits:**
- ✅ **Content-Aware Chunking**: Different strategies for different file types
- ✅ **Context Preservation**: Maintains temporal and hierarchical relationships
- ✅ **Quality Control**: Validates chunks for noise and duplicates
- ✅ **Scalable Architecture**: Handles large volumes of government data
- ✅ **Hybrid Optimization**: Optimized for both vector and graph search


## 🧠 Vector Store Strategy: Dense vs Sparse (Minerva/Zilliz)

### Goals
- Maximize recall for policy/legal queries (keyword-heavy)
- Maximize precision for narrative/summarization (semantic-heavy)
- Enable hybrid ranking with low-latency retrieval

### Index Design (Single Collection, Dual Fields)
- `content` (VarChar): Full text
- `dense_vector` (FloatVector, dim 1536): OpenAI `text-embedding-3-large`
- `sparse_vector` (FloatVector, dim 1000): TF-IDF/Okapi BM25-style sparse embedding (aligned with week_2 approach)
- `metadata` (JSON): `{meeting_id, meeting_type, meeting_date, chunk_type, start_time, duration, url, item_id}`

### Milvus Indexes
- Dense: `HNSW`, metric `COSINE`, params `{M: 32, efConstruction: 200}`
- Sparse: `IVF_FLAT` (or `HNSW` if fixed-length), metric `COSINE`, params `{nlist: 128}`

### Retrieval Policy
- Compute both scores per chunk: `dense_score` and `sparse_score`
- Hybrid score = `alpha * dense_score + (1 - alpha) * sparse_score` where:
  - `alpha = 0.35` for keyword/short queries (detected via heuristics: many proper nouns, ordinance numbers, item IDs)
  - `alpha = 0.65` for semantic/summary queries
  - Add query-time switch for explicit modes: `mode=semantic|keyword|hybrid`
- Fetch `k * 3` candidates per index → reciprocal-rank-fusion (RRF) → top-k → GPT rerank

### Sparse Vector Construction
- Use `TfidfVectorizer(max_features=1000, ngram_range=(1,2))`
- Fit vectorizer per domain snapshot; version and persist the vocabulary in MinIO/JSON
- Normalize vectors to unit length for cosine

### Query-Time Heuristics (Routing)
- If query has ordinance numbers, `ItemID`, `MeetingID`, dates, or exact phrases: boost sparse
- If query asks “why/how/summary/trend/compare”: boost dense
- If query includes named entities (people/streets/projects) and timeframe: hybrid with temporal filter

### Filters
- Scalar filter first: restrict by `meeting_type`, date ranges, agenda `item_id` if present
- Then run hybrid vector search on the filtered subset for speed and relevance


## 🕸️ Graph RAG Strategy (Neo4j)

### Goals
- Resolve entity-centric and temporal questions that require relationships
- Provide provenance and verifiable chains (who → discussed → what → when → decided → where stored)

### Core Schema (Nodes)
- `Meeting{id, title, date, type, url}`
- `AgendaItem{id, title, item_id, url, order}`
- `Topic{id, name}`
- `Person{id, name, role}`
- `Resolution{id, number, status}`
- `Contract{id, vendor, amount}`
- `Department{id, name}`

### Relationships
- `(Meeting)-[:HAS_AGENDA_ITEM]->(AgendaItem)`
- `(Meeting)-[:DISCUSSED]->(Topic)`
- `(Person)-[:SPOKE_IN]->(Meeting)` with `{start, end}`
- `(AgendaItem)-[:RESULTED_IN]->(Resolution)`
- `(Resolution)-[:FUNDS]->(Department)`
- `(AgendaItem)-[:LINKS_TO]->(Contract)`
- `(AgendaItem)-[:SAME_TOPIC_AS]->(AgendaItem)` (semantic linking)

### Ingestion
- From transcripts: extract speakers, topics, time windows; upsert `(Person)`, `(Topic)`, `(Meeting)`; add `SPOKE_IN`, `DISCUSSED`
- From agenda metadata: upsert `(AgendaItem)`; connect to `(Meeting)` via `HAS_AGENDA_ITEM`
- From attachments: parse vendors/amounts to create `(Contract)` and link via `LINKS_TO`

### Indexing (Neo4j)
- Constraints: unique on `id` for all node labels
- Fulltext indexes: `CALL db.index.fulltext.createNodeIndex('agendaText', ['AgendaItem'], ['title'])`
- Optional: APOC text-search helpers for keyword queries

### Retrieval Patterns (examples)
- Agenda by topic:
```cypher
CALL db.index.fulltext.queryNodes('agendaText', $topic) YIELD node, score
MATCH (m:Meeting)-[:HAS_AGENDA_ITEM]->(node)
RETURN m, node, score ORDER BY score DESC LIMIT $k;
```
- Timeline:
```cypher
MATCH (m:Meeting)-[:HAS_AGENDA_ITEM]->(a:AgendaItem)
WHERE toLower(a.title) CONTAINS toLower($topic)
RETURN m.date AS date, collect({meeting:m.title, item:a.title, url:a.url}) AS entries
ORDER BY date ASC;
```
- Who discussed what:
```cypher
MATCH (p:Person)-[s:SPOKE_IN]->(m:Meeting)-[:DISCUSSED]->(t:Topic)
WHERE toLower(t.name) CONTAINS toLower($topic)
RETURN p.name, m.title, s.start AS t_start ORDER BY t_start ASC LIMIT $k;
```

### Graph + Vector Fusion
- Use graph query to shortlist relevant `(Meeting, AgendaItem)` pairs (k=50)
- Feed shortlisted `meeting_id`s as a filter to vector store
- Retrieve top-k transcript chunks with timestamps
- GPT synthesis with inline citations (top 10 links)

### When to Prefer Graph vs Vector
- Prefer Graph for: “who voted”, “what passed”, “across meetings”, “timeline”, “by department/vendor”
- Prefer Vector for: “summarize discussion”, “what concerns were raised”, “long-form context”
- Hybrid for: “how did the Pine Island Road project evolve and what was approved?”


## ⚖️ End-to-End Retrieval Policy (Town GPT)

1. Classify query: `relation | temporal | semantic | hybrid`
2. Apply scalar filters (date, meeting type, item id) if present
3. If `relation/temporal` → run Graph-first, then Vector on candidates
4. If `semantic` → run Hybrid vector (dense+sparse) with rerank
5. Fuse results via Reciprocal Rank Fusion + GPT reranker
6. Return answer + top 10 citations:
   - YouTube links with `&t=<start_seconds>s`
   - Agenda/attachment URLs (page/section note if available)


## ⏱️ Performance & Cost Controls
- Cache embeddings per `chunk_id` and TF-IDF vectors
- Cache frequent graph queries by `(topic, date_bucket)`
- Use pagination for large timelines
- Track latency budgets: Graph (<= 300ms), Vector search (<= 400ms), Rerank (<= 1.2s)


## ✅ Testing & Evaluation
- Golden queries: create 30 civic questions (10 semantic, 10 relational, 10 temporal)
- Measure: HIT@k for citations, factual correctness with human review
- A/B test `alpha` for hybrid scoring; log query → strategy → outcomes
- Regression: ensure new meetings/agenda items don’t degrade recall 