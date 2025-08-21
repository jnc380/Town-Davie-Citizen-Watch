## Problem I set out to solve

I wanted to understand how my town’s government operates and provide some level of oversight without spending hundreds of hours watching council meetings and manually combing through documents. I built the Town of Davie Council Meeting GPT so residents can quickly get grounded answers on major topics discussed at the highest levels, with links back to the source records.

## Scope and selected data sources

- The 4-week build focuses on 2025 meetings to keep the scope realistic.
- There is a wealth of additional public data (budgets, committee agendas, monthly financial statements, procurement/bid data) that could be integrated later. 
  - Town Council meeting YouTube recordings and transcripts
  - Meeting agendas and agenda items
  - Supporting documents attached to agenda items (when available)
 
## Use at least two different data sources

- YouTube meeting transcripts (video/audio-derived text)
- Official agendas and agenda items
- Supporting documentation for agenda items

## Use cases and the AI value

- Lower the barrier for residents to access government records and discussions
- Provide concise, sourced answers without sitting through hours of video or reading long packets
- Summarize cross-meeting conversations about the same topic and surface the most relevant sources

## Realtime API behavior (high level)

- Query Neo4j first for graph “topic” hints; return relevant Milvus chunk IDs and related meeting context
- Perform both dense and sparse Milvus searches for additional context
- Deduplicate and rerank context with an LLM (Reranker)
- Pass top-N to an LLM that answers the user’s question (Answerer)
- Send the question, answer, and top context to a final LLM (Source Summarizer) that explains why each source supports the answer and returns URLs
- Wrap responses in logging and basic IP monitoring

## Tech stack and justification

- Milvus/Zilliz (Vector DB): High-scale, low-latency vector search with serverless Zilliz endpoint; supports dense and sparse retrieval and stores rich metadata for filtering.
- Neo4j (Graph DB): Ideal for modeling and traversing relationships between concepts, agenda items, meetings, and documents; enables “topic-first” graph hints that guide retrieval.
- FastAPI: Modern async Python web framework with great performance and typing support; fits well with async OpenAI and I/O-bound vector/graph calls.
- OpenAI (LLMs + embeddings): Robust embeddings for dense search and strong reasoning for reranking, answer generation, and source justification; response_format JSON helps keep outputs structured.
- Vercel (serverless API entrypoint via `api/index.py`): Simple deploys and automatic scaling for the online experience; pairs well with FastAPI using a lazy ASGI loader.
- Python tooling: Rich ecosystem for scraping, parsing, async I/O, and testing; keeps the entire stack in one language for ingestion and serving.
- Supabase/Postgres (telemetry): Centralized, SSL-enforced logging storage (with file fallback) for sessions and events; supports hashed IP storage for privacy-conscious monitoring.

## Data cleaning steps 

- URL normalization and backfill
  - Normalized YouTube watch URLs and mapped them to meetings (`youtube_url_mapper.py`, `backfill_meeting_urls.py`).
  - Corrected/standardized document and packet URLs when possible and deduped duplicates.
- Agenda scraping and parsing
  - Scraped agendas and extracted item structures, headers, and supporting links (`agenda_scraper.py`).
  - Cleaned HTML/PDF artifacts, removed boilerplate, and preserved important section headers.
- Transcript ingestion and segmentation
  - Downloaded transcripts/audio and standardized timestamps and speaker text (`youtube_hybrid_downloader.py`).
  - Segmented transcripts using meeting-aware heuristics LLM call to align with agenda boundaries, creating start and end dates from the transcripts (`transcript_segmenter.py`).
- Alignment of transcripts to agenda items
  - Analyzed agenda structure to guide matching (`youtube_agenda_alignment/dynamic_agenda_analyzer.py`).
- Chunk generation and metadata normalization
  - Generated retrieval-ready chunks (titles, hierarchy, headers, excerpts) and enforced consistent metadata keys (`chunk_generator.py`, `enhanced_data_processor.py`, `data_processor.py`).
  - Deduplicated overlapping chunks, collapsed near-repeats, and ensured clean, URL-safe identifiers.
- Concept extraction and enrichment
  - Ran LLM-based concept extraction over agendas/transcripts to find common subjects and hot topics.
  - Loaded tuned concepts and relationships into Neo4j, ensuring consistent slugs, types, and links to agenda items and meetings (`neo4j_*_loader*_concepts.py`).
- Embeddings and storage
  - Computed OpenAI embeddings for dense retrieval, ensured dimensional consistency, and validated payload shapes.
  - Stored chunks and metadata in Milvus/Zilliz (`MILVUS_COLLECTION=capstone_hybrid_rag`) and verified search/filter behavior.


## Security and observability

- Rate limiting and request validation via FastAPI middlewares (and conservative defaults)
- Session logging and events in Supabase/Postgres using SSL-required connections; file-based fallback is enabled in restricted environments (`telemetry.py`)
- IP privacy: I hash IPs with a salt before storage to facilitate abuse monitoring without keeping raw addresses
- Structured logging of LLM steps (rerank, answer, summarizer) and counts/timings for dense/sparse/graph retrieval 