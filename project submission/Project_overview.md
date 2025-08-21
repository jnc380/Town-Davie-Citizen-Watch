# Webpage Link https://town-davie-citizen-watch.vercel.app/

## Project GIT Link https://github.com/jnc380/Town-Davie-Citizen-Watch/tree/main 


## Problem I set out to solve

I wanted to understand how my town’s government operates and provide some level of oversight without spending hundreds of hours watching council meetings and manually combing through documents. I built the Town of Davie Council Meeting GPT so residents can quickly get grounded answers on major topics impacting my community, with links back to the source records.  


## Scope and selected data sources

- The 4-week build focuses on 2025 meetings to keep the scope realistic.
- There is a wealth of additional public data (budgets, committee agendas, monthly financial statements, procurement/bid data) that could be integrated later. 
- Scrape transcripts, agendas, agenda items, and agenda items supporting documents and store it into Milvus
- Create relationships of common topics (find common topics in recent data by scanning youtube transacripts and agendas) in NEO4J, to allow for a broad range of different documents to be linked to a "hot" topic in the town, across multiple meeting dates. 
- Combine links to those datasets, then load them into NEO4J graph and the context into MILVUS. 
- Real time api response, will query neo4j first to see if any larger topic conversations are covered and provide the relevant Milvus chunk id back to Milvus for context
- Will flow through multiple nodes and relationships of Topics, to Agenda Items, then to agenda items supporting documents
- Sparse and dense searchs will also be performed by Milvus every time to create additional context
- Context will be deduped and sent to a ranker LLM that will pass through N amount of ranked records to a answer LLM - Once the answer LLM is complete, the answer, source context and question is passed to a source summarizer LLM 
- The source summarizer llm provides reasoning on why this supporting document is relevant to the answer and provide that and a link to the end user to click on to review the document/youtube themselves
- Youtube videos will have the start time that aligns to the agendas topic 


## Challenges I Faced

- Honeslty everything, I am not the best python coder. Close to master level at database design and SQL, so this was a challenge
- It took me almost 2 weeks to figure out how to scrape everything from the web and make sure I have enought metadata to link back to the source when serving up to the llm
- The design side of this was pretty easy, but I have never deployed a front end before, so the learning curve on Vercel was high


## Future Enhancements

- Integrate budgets, laws and procurement bids
- Expand beyond 2025 data
- Make the UI more robust with a way to share feedback and rate responses
    - Would allow for an easy tracking mechanism to track see the quality of the responses
    - With rated responses can start auto-testing different prompts using DSPY and test if quality of resposnes improves
- Feedback loop of hot topics based on queries logged into supabase to automatically load into NEO4J topic nodes
 
## Use at least two different data sources

- YouTube meeting transcripts (video/audio-derived text) https://www.youtube.com/watch?v=giwu0UqBw-o&t=4024s 
- Agendas https://davie.novusagenda.com/agendapublic/MeetingView.aspx?MeetingID=589&MinutesMeetingID=-1&doctype=Agenda
- Agenda items https://davie.novusagenda.com/agendapublic/CoverSheet.aspx?ItemID=8659&MeetingID=589 
- Supporting documentation for agenda items https://davie.novusagenda.com/agendapublic/AttachmentViewer.ashx?AttachmentID=35790&ItemID=8659


## Use cases and the AI value

- Lower the barrier for residents to access government records and discussions
- Provide concise, sourced answers without sitting through hours of video or reading long packets
- Summarize cross-meeting conversations about the same topic and surface the most relevant sources
  # Technical AI use cases utilized in project
    - Using LLM ranker to provide the highest relevance context for the quesiton asked to the LLM Summarizer
    - The LLM sumarizer takes the highest rated context and provides an answer to the question
    - The LLM Source summarizer takes the context used, the questiona and answer, then provides 1-2 sentences of why the source is relevant to the answer
    - For the offline procesing of transcripts I used an LLM to match up when agenda items were starting and stopping in the transcript. Allowing me to create context windows relevant to a specific agenda item, allowing the LLM Summarizer to have both the discussion and the official government documentaiton
    - Fed agendas and the part of the meeting where the public is allowed to ask anything into an LLM that created "hot" topics that were loaded into NEO4J to help find relevant content in Milvus


## Realtime API behavior (high level)

- Query Neo4j first for graph “topic” hints; return relevant Milvus chunk IDs and related meeting context
- Perform both dense and sparse Milvus searches for additional context
- Deduplicate and rerank context with an LLM (Reranker)
- Pass top-N to an LLM that answers the user’s question (Answerer)
- Send the question, answer, and top context to a final LLM (Source Summarizer) that explains why each source supports the answer and returns URLs
- Wrap responses in logging and basic IP monitoring
- Any followup questions sends the context from the last 10 Q&A back to the LLM so there is relevant conversations in the same chat thread


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