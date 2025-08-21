# Capstone Project: Civic Government Q&A System

## Project Overview
This capstone project implements a comprehensive RAG system for local government transparency. The system enables citizens and journalists to ask natural language questions about city government actions, providing unified access to scattered information across meetings, laws, and financial documents.

### Business Problem
Citizens and journalists struggle to understand local government actions—what was discussed, which laws changed, how money was spent, and what the impact was—because information is scattered across meetings, laws, and financial documents.

### Target Users
- Civic-minded citizens
- Local journalists

### Value Proposition
A unified Q&A system that enables anyone to ask plain-English questions about city government actions, with answers that synthesize meetings, decisions, law changes, and finances—all citing the original source documents.

### Key Features
- Natural language Q&A over meetings, laws, and finances
- Answers cite specific source documents with timestamps and page references
- Search and explore by keyword, topic, date, or entity
- Timeline view of law and financial changes
- Personalized topic alerts and summaries via email (nice to have)

## Requirements

### Grading Rubric Requirements (CRITICAL)
**Criteria 1: Project Spec**
- System/AI agent design diagram
- Screenshots (UI, example queries)
- Clear business problem definition

**Criteria 2: Write Up**
- Project purpose and expected outputs
- Dataset and technology choices with justifications
- Steps followed and challenges faced
- Possible future enhancements

**Criteria 3: Vectorizing Unstructured Data**
- At least 2 data quality checks for each data source
- Use at least 2 different data sources
- Document data cleaning steps

**Criteria 4: RAG Code**
- RAG model with at least 1000 embeddings in vector database
- Integration test with at least 5 queries
- Protect RAG from abuse with appropriate techniques
- Consider Graph features for standout capabilities
- Consider re-ranking capabilities

**Criteria 5: Live Deployment**
- Must provide live link to app
- Deploy to Vercel or other platform
- Submissions without live link will be automatically rejected

**Criteria 6: Project Scoping**
- Address real, non-trivial use case
- Implement end-to-end solution
- Define clear use cases and value proposition

### Stand Out Features (For Excellence)
- Add technical complexity
- Include real-time data integration
- Add analytics layer to "score" people
- Provide deeper business value
- Implement personalized experience with user authentication
- Apply advanced analytical patterns
- Use auto-prompt optimization and show improvement
- Build a really good-looking UI
- Consider posting on platforms like Kaggle or Medium

### Core RAG System Requirements
- **Dual Vector Search**: Implement both sparse (TF-IDF) and dense (OpenAI embeddings) vector search
- **Vector Storage**: Use Milvus for efficient vector storage and retrieval
- **Chunking Strategies**: Implement multiple chunking approaches (by file, by function, by semantic boundaries)
- **Reranking**: Use OpenAI for intelligent result reranking
- **API Endpoints**: FastAPI server with comprehensive endpoints
- **Testing**: Comprehensive unit and integration tests

### Technical Requirements
- **Language**: Python 3.8+
- **Framework**: FastAPI
- **LLM Provider**: OpenAI (embeddings, reranking, text generation)
- **Vector Database**: Minerva cloud hosted by Zilliz
- **Graph Database**: Neo4j for RAG graph functionality
- **Frontend Deployment**: Vercel
- **Testing**: pytest
- **Documentation**: Comprehensive README and API documentation
- **Data Refresh**: TBD - automated solution needed for periodic updates

### Quality Standards
- **Code Quality**: PEP 8 compliance, type hints, comprehensive docstrings
- **Error Handling**: Robust exception handling with meaningful messages
- **Security**: Environment variables for configuration, input validation
- **Performance**: Optimized vector search, caching, async operations
- **Testing**: High test coverage, edge case testing

## Project Structure
```
capstone/
├── .cursorrules                    # Cursor AI guidelines
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── town_meeting_downloader.py     # YouTube video downloader with incremental processing
├── agenda_downloader.py           # Meeting agenda PDF downloader (Selenium-based)
├── minutes_downloader.py          # Meeting minutes PDF downloader (year-based navigation)
├── ordinances_downloader.py       # Municode ordinances downloader (hierarchical structure)
├── archive_reports_downloader.py # Archive reports downloader (annual and monthly financial reports)
├── budget_downloader.py          # Current fiscal year budget downloader
├── src/                           # Source code (RAG system components)
├── tests/                         # Test files
├── docs/                          # Documentation
└── config/                        # Configuration files
```

## Data Collection Components

### YouTube Video Downloader (`town_meeting_downloader.py`)
Downloads and processes YouTube videos from the @townofdavietv channel with the following features:

**Filtering Criteria:**
- Videos must contain "Meeting" in the title (case-insensitive)
- Only videos uploaded from January 1, 2024 onwards
- Channel: @townofdavietv

**Incremental Processing:**
- Tracks last downloaded video in `last_downloaded.json`
- Only processes new videos on subsequent runs
- Compares videos by upload date and video ID
- Resumable processing - can be interrupted and resumed

**Output:**
- Audio files (temporarily, cleaned up after processing)
- JSON transcripts with timestamps and word-level timing
- Human-readable transcript files
- Comprehensive metadata files for RAG integration
- Processing summary reports

**Usage:**
```bash
# Process all meeting videos (default 50 videos)
python town_meeting_downloader.py

# Process with custom video limit
python town_meeting_downloader.py --max-videos 10

# Custom output directory
python town_meeting_downloader.py --output-dir "davie_meetings"
```

### Meeting Agenda Downloader (`agenda_downloader.py`)
Downloads PDF meeting agendas from the Town of Davie website with dynamic form handling:

**Features:**
- Selenium-based web scraping for dynamic forms
- Handles "Custom Date Range" dropdown selection
- Inputs "From" and "To" dates in "1/1/2024" format
- Extracts meeting date and meeting type metadata
- Only downloads agendas from January 1, 2024 onwards
- Incremental processing with tracking

**Target URL:** https://www.davie-fl.gov/165/Town-Council-Meeting-Agendas

**Dependencies:**
```bash
pip install selenium beautifulsoup4 requests
```

### Meeting Minutes Downloader (`minutes_downloader.py`)
Downloads PDF meeting minutes from the Town of Davie website with year-based navigation:

**Features:**
- Selenium-based year selection (2024, 2025, etc.)
- Automatic year detection based on current date
- Handles dynamic year navigation buttons
- Extracts meeting date and title metadata
- Only downloads minutes from January 1, 2024 onwards
- Incremental processing with tracking

**Target URL:** https://www.davie-fl.gov/1438/Regular-Town-Council-Meeting-Minutes

**Year Navigation Logic:**
- Automatically detects years to process (2024 to current year)
- Handles future years based on current date
- Clicks year buttons to navigate through different years
- Extracts all PDF links for each year

**Usage:**
```bash
# Download all new minutes from 2024 onwards (headless mode)
python minutes_downloader.py

# Download with browser visible for debugging
python minutes_downloader.py --no-headless

# Custom output directory
python minutes_downloader.py --output-dir "davie_minutes_2024"
```

### Municode Ordinances Downloader (`ordinances_downloader.py`)
Downloads ordinances and code updates from Municode with hierarchical structure preservation:

**Features:**
- Selenium-based version tab navigation
- Hierarchical data structure extraction (chapters, articles, sections)
- Version/addendum tracking (2024-2025 focus)
- Comprehensive metadata preservation
- Incremental processing with tracking

**Target URL:** https://library.municode.com/fl/davie/codes/code_of_ordinances

**Hierarchical Structure:**
- Preserves complete ordinance hierarchy (chapters → articles → sections → subsections)
- Extracts content at each level
- Maintains parent-child relationships
- Stores hierarchy in separate JSON files

**Data Collection Context:**
- **Purpose**: Ordinances and code updates for Town of Davie
- **Relationship**: Links to decisions made in town hall meetings, documented in agendas and minutes
- **Hierarchical Nature**: Complex structure with chapters, articles, sections, subsections
- **Version Tracking**: Multiple versions and addendums over time (2024-2025 focus)

**Usage:**
```bash
# Download all ordinances from 2024-2025 (headless mode)
python ordinances_downloader.py

# Download with browser visible for debugging
python ordinances_downloader.py --no-headless

# Custom output directory
python ordinances_downloader.py --output-dir "davie_ordinances_2024_2025"
```

### Archive Reports Downloader (`archive_reports_downloader.py`)
Downloads Archive Reports from the Town of Davie archive with support for multiple data sources:

**Supported Data Sources:**
- **Annual Reports** (AMID=37): Annual Comprehensive Financial Reports
- **Monthly Reports** (AMID=38): Monthly Financial Summary Reports

**Features:**
- Filters for reports from 2024 onwards
- Extracts fiscal year from report titles (FY 2024, etc.)
- Incremental processing with tracking (separate tracking per data source)
- Comprehensive metadata extraction
- PDF download with error handling
- Dynamic data source configuration

**Target URLs:**
- Annual: https://www.davie-fl.gov/Archive.aspx?AMID=37
- Monthly: https://www.davie-fl.gov/Archive.aspx?AMID=38

**Filtering Criteria:**
- **Annual Reports**: "annual comprehensive financial report", "comprehensive annual financial report", "financial report", "audit report"
- **Monthly Reports**: "financial summary", "monthly financial", "financial report"
- Fiscal year must be 2024 or later
- Only PDF format reports

**Fiscal Year Extraction:**
- **Annual**: Looks for "FY" followed by year (e.g., "FY 2024")
- **Monthly**: Extracts standalone years (e.g., "2024", "2025")
- Handles various date formats in report titles

**Data Collection Context:**
- **Annual Purpose**: Annual Comprehensive Financial Reports for Town of Davie
- **Monthly Purpose**: Monthly Financial Summary Reports for Town of Davie
- **Relationship**: Financial reports provide spending and budget context for town hall meetings and decisions
- **Fiscal Year Focus**: Reports from 2024 onwards
- **Source Archive**: Town of Davie official archive

**Usage:**
```bash
# Download annual financial reports (default)
python archive_reports_downloader.py

# Download monthly financial reports
python archive_reports_downloader.py --data-source monthly

# Force download all reports (ignore tracking)
python archive_reports_downloader.py --data-source annual --force

# Custom output directory
python archive_reports_downloader.py --data-source monthly --output-dir "davie_monthly_financial_2024"

# Only create summary report (skip downloading)
python archive_reports_downloader.py --data-source annual --summary-only
```

### Current Fiscal Year Budget Downloader (`budget_downloader.py`)
Downloads all budget files from the Town of Davie current fiscal year budget page with file replacement logic:

**Features:**
- Downloads all budget files from the current fiscal year page
- File replacement strategy (deletes existing files before downloading new ones)
- Comprehensive budget file categorization
- Extracts fiscal year, ordinance numbers, and metadata
- PDF download with error handling

**Target URL:** https://www.davie-fl.gov/989/Current-Fiscal-Year-Budget

**Budget File Categories:**
- **Amendments**: Budget amendments and updates
- **Adopted**: Adopted budget and millage rate ordinances
- **Tentative**: Tentative budget and millage rate ordinances
- **Budget Book**: Complete budget book documents
- **Millage**: Tax rate and millage documents
- **Ordinance**: General ordinance documents

**File Replacement Strategy:**
- Deletes existing files before downloading new ones
- Ensures fresh copies of all budget documents
- Handles file conflicts and overwrites

**Data Collection Context:**
- **Purpose**: Current Fiscal Year Budget files for Town of Davie
- **Relationship**: Budget files provide financial planning context for town hall meetings and decisions
- **File Replacement Strategy**: Delete existing files before downloading new ones
- **Budget Categories**: Comprehensive categorization of budget document types
- **Source URL**: Town of Davie official budget page

**Usage:**
```bash
# Download all budget files (with replacement)
python budget_downloader.py

# Custom output directory
python budget_downloader.py --output-dir "davie_budget_2025"

# Only create summary report (skip downloading)
python budget_downloader.py --summary-only
```

## Setup Instructions
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables
4. Configure Milvus connection
5. Run tests: `pytest tests/`

## Development Guidelines
- Follow the `.cursorrules` file for AI assistance guidelines
- Maintain comprehensive documentation
- Write tests for all new features
- Use type hints and docstrings
- Follow security best practices

## API Documentation
- FastAPI automatic documentation at `/docs`
- Comprehensive endpoint documentation
- Example requests and responses

## Testing Strategy
- Unit tests for all functions
- Integration tests for end-to-end functionality
- Performance testing for vector operations
- Security testing for input validation

## Deployment
- Docker configuration included
- Environment setup scripts
- Monitoring and health checks
- Scalability considerations

## Contributing
- Follow the established coding standards
- Write comprehensive tests
- Update documentation as needed
- Use meaningful commit messages

## 📋 Documentation Trail & Application Limitations

### Video Processing Limitations
**Current Filtering Criteria:**
- Only processes videos containing "Meeting" in the title
- Limited to @townofdavietv channel
- Date range: January 1, 2024 onwards only
- **Impact:** May miss relevant content that doesn't use "Meeting" in title

**Processing Constraints:**
- OpenAI Whisper transcription has rate limits and costs
- Long videos (>1.5 hours) require significant processing time
- Audio quality affects transcription accuracy
- Temporary storage of audio files (cleaned up after processing)

**Data Quality Considerations:**
- YouTube video quality varies
- Some videos may have poor audio quality
- Transcript accuracy depends on audio clarity and speaker clarity
- No manual verification of transcript accuracy

### Agenda Collection Limitations
**Current Scope:**
- Only Town of Davie agendas
- Limited to PDF format agendas
- Requires manual date range selection
- **Impact:** Limited to one municipality, may miss other relevant documents

**Technical Constraints:**
- Website structure changes may break scraping
- Dynamic form handling requires Selenium (browser automation)
- PDF parsing quality depends on document structure
- No OCR for image-based PDFs

### Minutes Collection Limitations
**Current Scope:**
- Only Town of Davie meeting minutes
- Limited to PDF format minutes
- Year-based navigation dependency
- **Impact:** Limited to one municipality, may miss other relevant documents

**Technical Constraints:**
- Website year navigation structure may change
- Dynamic year selection requires Selenium
- PDF parsing quality depends on document structure
- No OCR for image-based PDFs
- Year navigation buttons must be clickable

### Ordinances Collection Limitations
**Current Scope:**
- Only Town of Davie ordinances from Municode
- Limited to HTML format ordinances
- Version tab navigation dependency
- **Impact:** Limited to one municipality, may miss other legal documents

**Technical Constraints:**
- Municode website structure may change
- Version tab navigation requires Selenium
- HTML parsing quality depends on site structure
- Complex hierarchical structure extraction
- Version/addendum identification challenges

### Archive Reports Collection Limitations
**Current Scope:**
- Only Town of Davie Archive Reports (Annual and Monthly Financial Reports)
- Limited to PDF format reports
- Archive page structure dependency
- **Impact:** Limited to one municipality, may miss other archive documents

**Technical Constraints:**
- Archive page structure may change
- PDF parsing quality depends on document structure
- Fiscal year extraction from titles may be inaccurate
- No OCR for image-based PDFs
- Limited to specific financial report types
- Separate tracking files needed for each data source

### Budget Collection Limitations
**Current Scope:**
- Only Town of Davie Current Fiscal Year Budget files
- Limited to PDF format budget documents
- Budget page structure dependency
- **Impact:** Limited to one municipality, may miss other budget documents

**Technical Constraints:**
- Budget page structure may change
- PDF parsing quality depends on document structure
- File replacement strategy may cause data loss if interrupted
- No OCR for image-based PDFs
- Limited to current fiscal year budget documents
- No incremental processing (always replaces files)

### Processing Pipeline Limitations
**Video Processing:**
- Sequential processing (not parallel)
- No resume capability for individual video failures
- Limited error recovery for corrupted downloads
- No validation of transcript completeness

**Agenda Processing:**
- No automatic metadata extraction from PDF content
- Limited to agenda documents (no minutes, resolutions, etc.)
- No integration between video and agenda data
- Manual date range management

### Future Enhancements
**Data Collection:**
- Expand to multiple government channels
- Include additional document types (minutes, resolutions, budgets)
- Implement parallel processing for faster downloads
- Add OCR for image-based documents

**Quality Improvements:**
- Manual transcript verification system
- Audio quality assessment and filtering
- Duplicate content detection
- Cross-reference between video and agenda data

**Technical Improvements:**
- Automated periodic refresh scheduling
- Real-time processing of new content
- Advanced error recovery and retry mechanisms
- Integration with government APIs where available

### Compliance & Ethics
**Data Usage:**
- All content is publicly available government information
- Respects YouTube Terms of Service
- No personal information collection
- Transparent processing methodology

**Limitations:**
- No real-time processing
- Manual intervention may be required for website changes
- No guarantee of complete data coverage
- Processing delays may affect data freshness 