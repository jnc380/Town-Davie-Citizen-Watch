# Data Quality Summary (Code-Backed)

This one‑pager summarizes what the code already does today to ensure quality across the four data sources and at retrieval/LLM time. Each item references the implementing file and lines.

## 1) YouTube Transcripts (Meeting Recordings)
- Requires transcript before saving metadata (prevents empty entries)
  - `capstone_hierarchical_rag/youtube_processes/youtube_hybrid_downloader.py` L121–144, L206–214
- Stops on suspected IP blocking to avoid corrupt/partial runs
  - `youtube_hybrid_downloader.py` L137–144
- Skips already processed videos (idempotency)
  - `youtube_hybrid_downloader.py` L174–186, L189–199

## 2) Agendas & Agenda Items (Scraping & Structure)
- HTTP status enforcement and robust discovery of embedded document links from CoverSheet pages
  - `capstone_hierarchical_rag/agenda_processes/agenda_scraper.py` L126–133, L139–157, L159–171
- Validates agenda folder structure and required metadata
  - `capstone_hierarchical_rag/enhanced_data_processor.py` L198–213
  - Raises if missing `meeting_metadata.json`; logs counts of expected HTML/PDF files

## 3) Supporting Documents (PDFs/Attachments)
- Text extraction validation (length, spaces, error tokens, basic meaningfulness)
  - `capstone_hierarchical_rag/nested_document_processor.py` L194–214 (`validate_extraction`)
- Content quality flag for quick triage
  - `capstone_hierarchical_rag/integrate_nested_chunks.py` L89 (`content_quality` = high/low)
- Summary stats logged at batch end (good/poor)
  - `nested_document_processor.py` L557–570, L601

## 4) Post‑Vectorization, Graph, and Retrieval Quality
- Strict date normalization before usage (ISO YYYY‑MM‑DD only)
  - `capstone_hierarchical_rag/milvus_only_rag_system_corrected_urls.py` L344–347 (and L374–377)
- Neo4j concept link deduplication prior to/while loading
  - `capstone_hierarchical_rag/neo4j_loader_concepts.py` L50–72, L114–117
- Combined dense+sparse result deduplication
  - `capstone_hierarchical_rag/milvus_only_rag_system.py` L369–376 (also present in variants)
- LLM quality gates during serving:
  - Reranking to select the most useful chunks for the question
    - `capstone_hierarchical_rag/milvus_neo4j_hybrid_system.py` L646–677 (`_rerank_with_gpt`)
  - Source justification to keep only directly supportive citations with concise rationale
    - `capstone_hierarchical_rag/milvus_neo4j_hybrid_system.py` L679–713 (`_explain_and_rerank_sources`)

## Why this matters
- Prevents empty/low‑quality inputs from entering the corpus (transcript gate, extraction validation)
- Ensures agenda bundles are complete and consistently structured (folder validation, file counts)
- Reduces noise and duplication in retrieval and graph relationships (dedup at multiple layers)
- Improves final answer quality and transparency (LLM rerank + per‑source explanation with URLs) 