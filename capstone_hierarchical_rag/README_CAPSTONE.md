# Capstone Project: Davie Government Transparency Q&A System

## 🎯 Project Overview

This project implements a **hybrid RAG (Retrieval-Augmented Generation) system** for government transparency, specifically designed for the Town of Davie. The system combines vector search with graph relationships to provide comprehensive answers about town meetings, council decisions, budgets, and policies.

### Business Problem
Citizens and journalists struggle to understand local government actions—what was discussed, which laws changed, how money was spent, and what the impact was—because information is scattered across meetings, laws, and financial documents.

### Solution
A unified Q&A system that enables anyone to ask plain-English questions about city government actions, with answers that synthesize meetings, decisions, law changes, and finances—all citing the original source documents.

## 🏗️ Architecture

### Hybrid RAG System
The system combines two powerful search approaches:

1. **Vector Search (Minerva/Zilliz)**: Semantic similarity search across meeting transcripts and agenda items
2. **Graph Search (Neo4j)**: Relationship-based search for entities, decisions, and temporal connections

### Technology Stack
- **LLM Provider**: OpenAI (embeddings, reranking, text generation)
- **Vector Database**: Minerva cloud hosted by Zilliz
- **Graph Database**: Neo4j for RAG graph functionality
- **Frontend**: Vercel deployment with modern UI
- **Backend**: FastAPI with comprehensive API endpoints

## 📁 Project Structure

```
capstone/
├── hybrid_rag_system.py      # Main hybrid RAG system
├── data_processor.py         # Data processing pipeline
├── test_system.py           # System testing and validation
├── requirements.txt         # Python dependencies
├── vercel.json             # Vercel deployment config
├── templates/
│   └── index.html          # Modern government transparency UI
├── downloads/              # Data sources
│   ├── town_meetings_youtube/  # YouTube transcripts
│   └── agendas/               # Meeting agendas
└── README_CAPSTONE.md      # This file
```

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file in the capstone directory:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Milvus/Minerva Configuration
MILVUS_URI=https://in03-d169f4e541cfbad.serverless.gcp-us-west1.cloud.zilliz.com
MILVUS_TOKEN=fb47fa99a372f71f41bcfd262a8ad4c00aa62530789dc6b1ecbea953c7a016b9d110bf84870abc350d9854e878f7db20fc06b585

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Process Data

```bash
python data_processor.py
```

This will:
- Load YouTube transcripts and agenda data
- Process and chunk the data
- Create embeddings using OpenAI
- Store data in both Milvus and Neo4j

### 4. Test the System

```bash
python test_system.py
```

This will run comprehensive tests to verify:
- System health and connections
- Vector search functionality
- Graph search functionality
- Hybrid search performance

### 5. Run the Application

```bash
python hybrid_rag_system.py
```

The application will be available at `http://localhost:8000`

### 6. Deploy to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod
```

## 🔍 System Features

### Hybrid Search Capabilities

1. **Vector Search**: Semantic similarity across meeting transcripts
2. **Graph Search**: Relationship-based search for entities and decisions
3. **Temporal Analysis**: Understanding of how decisions evolve over time
4. **Entity Recognition**: Identification of council members, topics, and decisions

### Query Examples

The system can handle complex queries like:

- "What was discussed about Pine Island Road beautification?"
- "How did the budget change in 2024?"
- "Which council members voted on the CRA extension?"
- "What contracts were awarded to construction companies?"
- "How much money was allocated for road improvements?"

### Data Sources

- **52 YouTube Transcripts**: Detailed meeting transcripts with timestamps
- **Multiple Agenda Documents**: Structured meeting agendas with item details
- **Meeting Metadata**: Titles, dates, descriptions, and URLs
- **Cross-References**: Links between meetings, agenda items, and decisions

## 📊 System Performance

### Data Quality Checks

1. **Transcript Processing**: 
   - Removes noise (music, silence)
   - Chunks by semantic boundaries
   - Preserves temporal information

2. **Agenda Processing**:
   - Extracts structured agenda items
   - Links items to meetings
   - Preserves hierarchical relationships

### RAG Implementation

- **1000+ Embeddings**: Stored in Milvus for semantic search
- **Graph Relationships**: Stored in Neo4j for entity connections
- **Hybrid Fusion**: Combines vector and graph results intelligently
- **Reranking**: Uses OpenAI to improve result relevance

## 🛡️ Security & Best Practices

### Security Features
- Environment variables for all sensitive data
- Input validation and sanitization
- Rate limiting for API endpoints
- Secure database connections

### Code Quality
- Comprehensive error handling
- Type hints throughout
- Comprehensive logging
- Unit and integration tests
- PEP 8 compliance

## 📈 Success Metrics

### Accuracy & Completeness
- **Answer Quality**: Comprehensive responses with source citations
- **Source Coverage**: All meetings and agenda items searchable
- **Temporal Accuracy**: Correct understanding of decision timelines

### User Experience
- **Response Time**: Fast hybrid search (< 5 seconds)
- **Interface**: Modern, accessible government transparency UI
- **Source Transparency**: Clear citations to original documents

### Technical Performance
- **Vector Search**: Semantic similarity with high recall
- **Graph Search**: Relationship queries with high precision
- **Hybrid Fusion**: Optimal combination of both approaches

## 🔧 Advanced Features

### Graph Database Schema

**Entities**:
- `Meeting`: Town council meetings with metadata
- `CouncilMember`: Elected officials and their roles
- `AgendaItem`: Individual agenda items and decisions
- `Resolution`: Formal resolutions and their details
- `Topic`: Discussion topics and their evolution

**Relationships**:
- `HAS_AGENDA_ITEM`: Meeting → AgendaItem
- `VOTED_ON`: CouncilMember → Resolution
- `DISCUSSED`: Meeting → Topic
- `AMENDED`: Resolution → Resolution
- `FUNDED`: Resolution → Budget

### Vector Search Features

- **Semantic Chunking**: Intelligent text segmentation
- **Multi-modal Embeddings**: Different strategies for different content types
- **Reranking**: OpenAI-powered result improvement
- **Context Preservation**: Maintains temporal and speaker context

## 🚀 Deployment

### Vercel Deployment

The system is configured for easy deployment to Vercel:

1. **Automatic Build**: Vercel detects FastAPI application
2. **Environment Variables**: Configured in Vercel dashboard
3. **Static Files**: Served efficiently by Vercel
4. **API Routes**: FastAPI endpoints automatically exposed

### Production Considerations

- **Database Scaling**: Minerva and Neo4j cloud instances
- **API Rate Limiting**: Protect against abuse
- **Monitoring**: Health checks and performance metrics
- **Backup**: Regular data backups and versioning

## 🔮 Future Enhancements

### Planned Features
- **Real-time Updates**: Automated data refresh from new meetings
- **Personalized Alerts**: Email notifications for topic updates
- **Advanced Analytics**: Decision impact analysis
- **Multi-language Support**: Spanish language interface
- **Mobile App**: Native mobile application

### Technical Improvements
- **Advanced Graph Queries**: More sophisticated relationship analysis
- **Multi-modal RAG**: Integration with video and image content
- **Federated Search**: Integration with external government databases
- **AI-powered Summaries**: Automatic meeting summaries and insights

## 📚 Documentation

### API Endpoints

- `GET /`: Main application interface
- `POST /api/search`: Hybrid search endpoint
- `GET /api/stats`: System health and statistics
- `GET /api/health`: Health check endpoint

### Data Processing

- `data_processor.py`: Complete data pipeline
- `hybrid_rag_system.py`: Core RAG functionality
- `test_system.py`: Comprehensive testing suite

## 🤝 Contributing

This project demonstrates advanced RAG techniques for government transparency. Key contributions include:

1. **Hybrid Search**: Combining vector and graph approaches
2. **Government Focus**: Specialized for civic engagement
3. **Production Ready**: Comprehensive error handling and deployment
4. **User Centric**: Modern UI for citizen accessibility

## 📄 License

This project is developed for educational purposes and government transparency initiatives.

---

**Built with ❤️ for Government Transparency**

*Powered by OpenAI + Milvus + Neo4j + FastAPI + Vercel* 

## Configuration Notes

- ENABLE_RERANKING: set to `true|false` to toggle GPT reranker
- RERANKING_MODEL: defaults to `gpt-4o-mini`
- CORS_ALLOW_ORIGINS: comma-separated list of allowed origins or `*`

## Security & Performance

- CORS enabled and basic rate limiting via slowapi
- Input validation on `query` and `top_k`
- Blocking Milvus/Neo4j calls offloaded to threads; dense/sparse and rerank/graph run concurrently 

## Graph relationships (modeling note)

Consider enriching the graph beyond `HAS_AGENDA_ITEM` to answer more questions:
- DISCUSSED (Meeting -> Topic)
- SPOKE_IN (Person -> Meeting) with time window
- RESULTED_IN (AgendaItem -> Resolution)
- LINKS_TO (AgendaItem -> Contract)
- SAME_TOPIC_AS (AgendaItem -> AgendaItem)

These can remain optional and be added incrementally as data allows. 