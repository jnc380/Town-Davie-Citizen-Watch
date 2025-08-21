# Key Modules (capstone_hierarchical_rag)

This is a curated list of the most impactful Python modules that power the system. It focuses on value-add work: YouTube and agenda processing, alignment, chunking, concept extraction, and the online API with its 3-LLM flow.

## Online API and Retrieval/LLM Orchestration
- `api/index.py`: Vercel entrypoint that loads the FastAPI app from `milvus_neo4j_hybrid_system.py`.
- `milvus_neo4j_hybrid_system.py`: Production API (FastAPI) serving hybrid RAG.
  - Milvus/Zilliz dense + sparse retrieval with optional Neo4j graph hints.
  - LLM #1 reranker: `_rerank_with_gpt()` ranks retrieved chunks by usefulness to the query.
  - LLM #2 answerer: builds a context window from top chunks and answers user questions.
  - LLM #3 source explainer: `_explain_and_rerank_sources()` filters/summarizes top sources and explains how each supports the answer.
- `hybrid_rag_system.py`: Larger/earlier variant with similar hybrid logic and UI demo; useful reference for light/serverless mode.
- `milvus_pymilvus_rag.py`, `milvus_only_rag_system*.py`: Reference/experimental retrieval implementations and tests.

## Offline Ingestion: YouTube (Video/Transcript)
- `youtube_processes/youtube_hybrid_downloader.py`: Unified downloader for YouTube assets (audio/transcripts) feeding transcript processing.
- `youtube_url_mapper.py`: Utilities to map and normalize known YouTube URLs to meetings.

## Offline Ingestion: Agendas (Documents/HTML/PDF)
- `agenda_processes/agenda_scraper.py`: Robust scraper/parser for agenda packets and meeting metadata producing structured agenda text and sections.

## Alignment: Combining YouTube + Agenda
- `youtube_agenda_alignment/transcript_segmenter.py`: Transcript segmentation with meeting structure awareness; basis for agenda-item boundary detection.
- `youtube_agenda_alignment/dynamic_agenda_analyzer.py`: Analyzes agenda structure to guide alignment.
- `youtube_agenda_alignment/enhanced_transcript_aligner.py` and `enhanced_transcript_aligner_v2.py`: Enhanced alignment of transcript segments to agenda items; core of improved alignment quality.
- `youtube_agenda_alignment/enhanced_agenda_extractor.py`: Extracts structured agenda items suitable for alignment.
- `youtube_agenda_alignment/complete_agenda_structure_builder.py`: Builds complete agenda hierarchies for downstream chunking.
- Runners and reports (for completeness):
  - `run_final_alignment.py`, `fix_alignment_and_generate_chunks.py`, `extract_real_alignments.py`, `extract_simple_alignments.py`, `show_*`, `test_*` files help execute, validate, and analyze alignment results.

## Chunking and Data Processing
- `chunk_generator.py`: Generates retrieval-ready chunks from aligned data (with headers, hierarchy, and metadata).
- `data_processor.py`, `enhanced_data_processor.py`: End-to-end processing pipeline that integrates alignments, normalizes metadata, and prepares Milvus/graph payloads.
- `integrate_nested_chunks.py`, `nested_document_processor.py`: Creates hierarchical/nested chunks and merges them into the final dataset.

## Concept Extraction and Graph ("Common Subjects")
- `cleanup_files/concept_extraction/llm_concept_extractor.py`, `concept_extractor.py`, `enhanced_concept_extractor.py`: LLM-based concept identification and tuning (common subjects across agendas/transcripts).
- `cleanup_files/concept_extraction/neo4j_concept_loader.py`: Loads extracted concepts/links into Neo4j.
- `neo4j_http_loader_concepts.py`, `neo4j_loader_concepts.py`, `neo4j_http_migrations.py`, `neo4j_migrations.py`: Graph loading and migrations for concept taxonomy and relationships.
- `convert_tuned_concepts.py`: Tools to convert tuned concept sets for loading.

## Backfill, Debugging, and Tests
- `backfill_meeting_urls.py`: Backfills missing meeting URL metadata.
- `debug_milvus_search.py`, `fix_milvus_api.py`: Troubleshooting Milvus queries and API compatibility.
- `eval/`: End-to-end test harnesses and diagnostics (`test_system.py`, `test_hybrid_queries.py`, plus dataset-specific tests).
- `test_*` in root: Retrieval, reranking, and API behavior tests.

## Notable Docs in This Folder
- `RAG_RETRIEVAL_EXPLANATION.md`: Rationale and details for the hybrid retrieval strategy.
- `CHUNKING_STRATEGY.md`: Chunking approaches and trade-offs.
- `IMPLEMENTATION_SUMMARY.md`, `PROCESSING_SUMMARY.md`, `CLEANUP_SUMMARY.md`, `CLEANUP_SUMMARY_2025.md`: Narrative summaries of processing, cleanup, and implementation details. 