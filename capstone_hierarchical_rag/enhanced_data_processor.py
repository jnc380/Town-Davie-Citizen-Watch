#!/usr/bin/env python3
"""
Enhanced Data Processor for Hierarchical RAG System
Processes YouTube transcripts and agenda data with improved matching and hierarchical structure
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import asyncio
import argparse
import re

from data_models import (
    EnhancedMeetingData, AgendaData, AgendaItemData, 
    DocumentationData, TranscriptSegmentData, MeetingData
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedDataProcessor:
    """Enhanced data processor with hierarchical structure support"""
    
    def __init__(self, init_rag_system: bool = True):
        """Initialize the enhanced data processor"""
        self.rag_system = None
        self.init_rag_system = init_rag_system
        if init_rag_system:
            # Lazy import to avoid initializing OpenAI/servers when chunk-only
            from hybrid_rag_system import HybridRAGSystem  # type: ignore
            self.rag_system = HybridRAGSystem()
        
        # Base directories resolved relative to this file
        capstone_base = Path(__file__).resolve().parent
        default_downloads = capstone_base / "downloads"
        
        # YouTube transcripts directory
        self.youtube_dir = Path(os.getenv("YOUTUBE_DOWNLOAD_DIR", str(default_downloads / "town_meetings_youtube")))
        
        # Agendas directory
        self.agendas_dir = Path(os.getenv("AGENDAS_DIR", str(default_downloads / "agendas")))
        
        # Year filter
        self.year_filter = os.getenv("YEAR_FILTER", "2025")
        
        logger.info("✅ Enhanced data processor initialized")

    def _fetch_bytes(self, url: str, timeout: float = 15.0) -> Optional[bytes]:
        """Fetch bytes from URL using httpx if available, otherwise urllib"""
        try:
            try:
                import httpx  # type: ignore
                with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                    r = client.get(url)
                    if r.status_code == 200 and r.content:
                        return r.content
                    logger.warning(f"Fetch non-200 for {url}: {r.status_code}")
                    return None
            except Exception:
                import urllib.request
                with urllib.request.urlopen(url, timeout=timeout) as resp:  # type: ignore
                    return resp.read()
        except Exception as e:
            logger.warning(f"Failed to fetch URL {url}: {e}")
            return None

    def _html_to_text(self, html: str) -> str:
        """Lightweight HTML to text conversion"""
        try:
            try:
                from bs4 import BeautifulSoup  # type: ignore
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(" ")
            except Exception:
                text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception:
            return html

    def _extract_pdf_text(self, data: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            from pdfminer.high_level import extract_text_to_fp  # type: ignore
            from io import BytesIO, StringIO
            output = StringIO()
            extract_text_to_fp(BytesIO(data), output)
            return re.sub(r"\s+", " ", output.getvalue()).strip()
        except Exception:
            return ""

    def load_youtube_transcripts(self) -> List[Dict[str, Any]]:
        """Load all YouTube transcript files"""
        transcripts = []
        
        if not self.youtube_dir.exists():
            logger.warning(f"YouTube directory not found: {self.youtube_dir}")
            return transcripts
        
        # Find all transcript files
        transcript_files = list(self.youtube_dir.glob("*_transcript.json"))
        logger.info(f"Found {len(transcript_files)} transcript files")
        
        for transcript_file in transcript_files:
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                
                # Get corresponding info file
                info_file = transcript_file.parent / transcript_file.name.replace("_transcript.json", ".info.json")
                info_data = {}
                if info_file.exists():
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info_data = json.load(f)
                
                # Extract meeting ID from filename
                meeting_id = transcript_file.stem.replace("_transcript", "")

                # Filter by year
                title = info_data.get("title", meeting_id)
                date_raw = self._extract_meeting_date(title)
                date_norm = self._normalize_date(date_raw) if date_raw else None
                year_ok = False
                if date_norm and len(date_norm) >= 4:
                    year_ok = date_norm.startswith(self.year_filter)
                else:
                    upload_date = str(info_data.get("upload_date", ""))
                    if len(upload_date) >= 4:
                        year_ok = upload_date[:4] == self.year_filter
                if not year_ok:
                    logger.info(f"⏭️  Skipping transcript (year filter {self.year_filter}): {meeting_id}")
                    continue
                
                transcripts.append({
                    "meeting_id": meeting_id,
                    "transcript": transcript_data,
                    "info": info_data,
                    "file_path": str(transcript_file)
                })
                
                logger.info(f"✅ Loaded transcript: {meeting_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load transcript {transcript_file}: {e}")
        
        return transcripts

    def load_agenda_data(self) -> List[Dict[str, Any]]:
        """Load all agenda data with enhanced folder structure validation"""
        agendas = []
        
        if not self.agendas_dir.exists():
            logger.warning(f"Agendas directory not found: {self.agendas_dir}")
            return agendas
        
        # Find all agenda directories matching the year filter
        agenda_dirs = [d for d in self.agendas_dir.iterdir() 
                      if d.is_dir() and d.name.startswith(f"{self.year_filter}-")]
        logger.info(f"Found {len(agenda_dirs)} agenda directories")
        
        for agenda_dir in agenda_dirs:
            try:
                # Look for meeting metadata file
                metadata_file = agenda_dir / "meeting_metadata.json"
                if not metadata_file.exists():
                    logger.warning(f"No metadata file found in {agenda_dir}")
                    continue
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Extract meeting ID from directory name
                meeting_id = agenda_dir.name
                
                # Validate folder structure
                self._validate_agenda_folder_structure(agenda_dir)
                
                agendas.append({
                    "meeting_id": meeting_id,
                    "metadata": metadata,
                    "agenda_dir": str(agenda_dir)
                })
                
                logger.info(f"✅ Loaded agenda: {meeting_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load agenda {agenda_dir}: {e}")
        
        return agendas

    def _validate_agenda_folder_structure(self, agenda_dir: Path) -> None:
        """Validate that agenda folder follows the expected structure"""
        # Check for required files
        metadata_file = agenda_dir / "meeting_metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"Missing meeting_metadata.json in {agenda_dir}")
        
        # Check for agenda item files
        html_files = list(agenda_dir.glob("item_*_*.html"))
        pdf_files = list(agenda_dir.glob("item_*_*_nested_*.pdf"))
        
        if not html_files:
            logger.warning(f"No agenda item HTML files found in {agenda_dir}")
        
        logger.info(f"📁 Agenda folder {agenda_dir.name}: {len(html_files)} HTML files, {len(pdf_files)} PDF files")

    def enhanced_match_meeting_data(self, transcripts: List[Dict], agendas: List[Dict]) -> List[EnhancedMeetingData]:
        """Enhanced matching of transcript and agenda data with better validation"""
        meeting_data_list: List[EnhancedMeetingData] = []
        
        # Index agendas by normalized date and type
        agendas_by_date: Dict[str, List[Dict[str, Any]]] = {}
        agendas_by_id: Dict[str, Dict[str, Any]] = {}
        
        for a in agendas:
            md = a.get("metadata", {})
            info = md.get("meeting_info", {})
            a_date_raw = info.get("meeting_date", "")
            a_date = self._normalize_date(a_date_raw)
            meeting_id = a.get("meeting_id", "")
            
            if a_date:
                agendas_by_date.setdefault(a_date, []).append(a)
            if meeting_id:
                agendas_by_id[meeting_id] = a
        
        used_agenda_dirs: set[str] = set()

        # Process all transcripts
        for transcript in transcripts:
            meeting_id = transcript["meeting_id"]
            info_data = transcript["info"] or {}
            
            # Extract meeting information from transcript
            title = info_data.get("title", meeting_id)
            meeting_type = self._extract_meeting_type(title)
            meeting_date_raw = self._extract_meeting_date(title)
            meeting_date_norm = self._normalize_date(meeting_date_raw)
            
            # Try multiple matching strategies
            agenda_data: Optional[Dict[str, Any]] = None
            
            # Strategy 1: Direct meeting ID match
            if meeting_id in agendas_by_id:
                agenda_data = agendas_by_id[meeting_id]
                logger.info(f"🎯 Direct ID match for {meeting_id}")
            
            # Strategy 2: Date + type matching
            elif meeting_date_norm and meeting_date_norm in agendas_by_date:
                candidates = agendas_by_date[meeting_date_norm]
                if len(candidates) == 1:
                    agenda_data = candidates[0]
                    logger.info(f"📅 Single date match for {meeting_id}")
                else:
                    # Pick best match by meeting type similarity
                    best_match = self._find_best_type_match(candidates, meeting_type)
                    if best_match:
                        agenda_data = best_match
                        logger.info(f"🔍 Type-based match for {meeting_id}")
            
            # Create enhanced meeting data
            enhanced_meeting = self._create_enhanced_meeting_data(
                meeting_id, meeting_type, meeting_date_raw, title, 
                info_data, agenda_data, transcript
            )
            
            if agenda_data:
                used_agenda_dirs.add(agenda_data.get("agenda_dir", ""))
            
            meeting_data_list.append(enhanced_meeting)
            logger.info(f"✅ Enhanced meeting: {meeting_id} (agenda: {enhanced_meeting.agenda is not None})")
        
        # Add agenda-only meetings for unmatched agendas
        for a in agendas:
            a_dir = a.get("agenda_dir", "")
            if a_dir in used_agenda_dirs:
                continue
            
            md = a.get("metadata", {})
            info = md.get("meeting_info", {})
            meeting_date_raw = info.get("meeting_date", "")
            meeting_type = info.get("meeting_type", "Other Meeting")
            title = info.get("meeting_title", a.get("meeting_id", "Agenda Meeting"))
            meeting_id = a.get("meeting_id", a_dir)
            
            enhanced_meeting = self._create_enhanced_meeting_data(
                meeting_id, meeting_type, meeting_date_raw, title, 
                {}, a, None  # No transcript data
            )
            
            meeting_data_list.append(enhanced_meeting)
            logger.info(f"🗂️ Added agenda-only meeting: {meeting_id}")

        return meeting_data_list

    def _find_best_type_match(self, candidates: List[Dict[str, Any]], meeting_type: str) -> Optional[Dict[str, Any]]:
        """Find the best agenda match based on meeting type"""
        def type_rank(a_md: Dict[str, Any]) -> int:
            a_type = self._agenda_meeting_type(a_md.get("metadata", {})).lower()
            t = meeting_type.lower()
            
            # Exact type match
            if a_type == t:
                return 0
            
            # CRA matching
            if ("cra" in a_type) == ("cra" in t):
                return 1
            
            # Workshop matching
            if ("workshop" in a_type) == ("workshop" in t):
                return 2
            
            # Regular/Council matching
            if ("regular" in a_type or "council" in a_type) == ("regular" in t or "council" in t):
                return 3
            
            return 4
        
        sorted_candidates = sorted(candidates, key=type_rank)
        return sorted_candidates[0] if sorted_candidates else None

    def _create_enhanced_meeting_data(
        self, meeting_id: str, meeting_type: str, meeting_date: str, title: str,
        info_data: Dict[str, Any], agenda_data: Optional[Dict[str, Any]], 
        transcript: Optional[Dict[str, Any]]
    ) -> EnhancedMeetingData:
        """Create enhanced meeting data with hierarchical structure"""
        
        # Create agenda data if available
        agenda: Optional[AgendaData] = None
        if agenda_data:
            agenda = self._create_agenda_data(agenda_data)
        
        # Create metadata
        metadata = {
            "url": info_data.get("url", info_data.get("webpage_url", "")),
            "description": info_data.get("description", ""),
            "upload_date": info_data.get("upload_date", ""),
            "channel_name": info_data.get("channel_name", ""),
            "has_agenda": agenda is not None,
            "has_transcript": transcript is not None,
            "matched_agenda_dir": agenda_data.get("agenda_dir", "") if agenda_data else "",
        }
        
        return EnhancedMeetingData(
            meeting_id=meeting_id,
            meeting_type=meeting_type,
            meeting_date=meeting_date,
            title=title,
            agenda=agenda,
            metadata=metadata
        )

    def _create_agenda_data(self, agenda_data: Dict[str, Any]) -> AgendaData:
        """Create agenda data from raw agenda data"""
        md = agenda_data.get("metadata", {})
        info = md.get("meeting_info", {})
        
        agenda_id = agenda_data.get("meeting_id", "")
        meeting_id = agenda_data.get("meeting_id", "")
        meeting_date = info.get("meeting_date", "")
        agenda_type = self._agenda_meeting_type(info)
        
        # Extract agenda items
        agenda_items = self._extract_enhanced_agenda_items(
            md, agenda_data.get("agenda_dir", None)
        )
        
        return AgendaData(
            agenda_id=agenda_id,
            meeting_id=meeting_id,
            agenda_type=agenda_type,
            meeting_date=meeting_date,
            items=agenda_items,
            metadata=md
        )

    def _extract_enhanced_agenda_items(self, metadata: Dict[str, Any], agenda_dir: Optional[str] = None) -> List[AgendaItemData]:
        """Extract agenda items with enhanced documentation linking"""
        agenda_items: List[AgendaItemData] = []
        base_dir = Path(agenda_dir) if agenda_dir else None
        
        # Map nested links by parent_item_id
        nested_links = metadata.get("nested_links", [])
        nested_by_parent: Dict[str, List[Dict[str, Any]]] = {}
        for nl in nested_links:
            pid = nl.get("parent_item_id")
            if pid:
                nested_by_parent.setdefault(pid, []).append(nl)

        # Extract from pdf_links
        pdf_links = metadata.get("pdf_links", [])
        for i, link in enumerate(pdf_links):
            item_id = link.get("item_id", f"item_{i}")
            meeting_id = link.get("meeting_id", "")
            url = link.get("url", "")
            title_text = link.get("text", "").strip()

            # Extract documentation
            documentation = self._extract_item_documentation(
                item_id, meeting_id, base_dir, nested_by_parent.get(item_id, [])
            )

            # Create agenda item
            agenda_item = AgendaItemData(
                item_id=item_id,
                item_number=str(i + 1),
                title=title_text,
                description=self._extract_item_description(item_id, meeting_id, base_dir),
                order=i,
                url=url,
                documentation=documentation,
                metadata={
                    "source": "agenda",
                    "item_type": "agenda_item",
                    "meeting_id": meeting_id
                }
            )
            
            # Add structured agenda text if available
            if base_dir and base_dir.exists():
                try:
                    metadata_file = base_dir / "meeting_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            agenda_metadata = json.load(f)
                        
                        structured_text = agenda_metadata.get("structured_agenda_text", {})
                        if structured_text:
                            # Find matching item in structured text
                            for item in structured_text.get("agenda_items", []):
                                if item.get("item_id") == agenda_item.item_id:
                                    agenda_item.metadata["structured_content"] = item.get("content", "")
                                    agenda_item.metadata["structured_title"] = item.get("title", "")
                                    break
                            
                            # Add meeting structure context
                            agenda_item.metadata["meeting_structure"] = structured_text.get("meeting_structure", [])
                except Exception as e:
                    logger.warning(f"Failed to add structured text for item {agenda_item.item_id}: {e}")
            
            agenda_items.append(agenda_item)

        return agenda_items

    def _load_agenda_items_with_structure(self, meeting_id: str, base_dir: Path) -> List[AgendaItemData]:
        """Load agenda items with complete structure context"""
        agenda_items = []
        
        try:
            # Load the complete agenda structure
            metadata_file = base_dir / "meeting_metadata.json"
            if not metadata_file.exists():
                logger.warning(f"No meeting metadata found for {meeting_id}")
                return agenda_items
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                agenda_metadata = json.load(f)
            
            # Get the complete agenda structure
            complete_structure = agenda_metadata.get("complete_agenda_structure", {})
            sections = complete_structure.get("sections", [])
            
            # Process each section and its items
            for section in sections:
                section_id = section.get("section_id", "")
                section_title = section.get("title", "")
                
                for item in section.get("items", []):
                    item_id = item.get("item_id", "")
                    title = item.get("title", "")
                    description = item.get("description", "")
                    url = item.get("url", "")
                    has_pdf = item.get("has_pdf", False)
                    
                    # Create agenda item with enhanced context
                    agenda_item = AgendaItemData(
                        item_id=item_id,
                        item_number=item.get("item_number", ""),
                        title=title,
                        description=description,
                        order=len(agenda_items),  # Maintain order
                        url=url,
                        documentation=[],  # Will be populated separately
                        metadata={
                            "source": "agenda",
                            "item_type": "agenda_item",
                            "meeting_id": meeting_id,
                            "section_id": section_id,
                            "section_title": section_title,
                            "has_pdf": has_pdf,
                            "level": item.get("level", 2)
                        }
                    )
                    
                    # Add structured content if available
                    if item.get("structured_content"):
                        agenda_item.metadata["structured_content"] = item["structured_content"]
                    
                    if item.get("structured_title"):
                        agenda_item.metadata["structured_title"] = item["structured_title"]
                    
                    # Add meeting structure context
                    if complete_structure.get("meeting_structure"):
                        agenda_item.metadata["meeting_structure"] = complete_structure["meeting_structure"]
                    
                    # Add section hierarchy context
                    agenda_item.metadata["section_context"] = {
                        "section_id": section_id,
                        "section_title": section_title,
                        "section_level": section.get("level", 1),
                        "section_order": sections.index(section)
                    }
                    
                    agenda_items.append(agenda_item)
            
            logger.info(f"Loaded {len(agenda_items)} agenda items with structure for meeting {meeting_id}")
            
        except Exception as e:
            logger.error(f"Failed to load agenda items with structure for {meeting_id}: {e}")
        
        return agenda_items

    def process_agenda_directory(self, meeting_id: str, base_dir: Path) -> List[AgendaItemData]:
        """Process agenda directory and return agenda items with enhanced structure"""
        logger.info(f"Processing agenda directory for meeting {meeting_id}")
        
        # Use the new structure-aware loading method
        agenda_items = self._load_agenda_items_with_structure(meeting_id, base_dir)
        
        if not agenda_items:
            logger.warning(f"No agenda items found for meeting {meeting_id}")
            return []
        
        # Process documentation for each item
        for agenda_item in agenda_items:
            if agenda_item.metadata.get("has_pdf", False):
                # Load documentation from the original metadata
                metadata_file = base_dir / "meeting_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            agenda_metadata = json.load(f)
                        
                        # Find the item in the original metadata
                        pdf_links = agenda_metadata.get("pdf_links", [])
                        for link in pdf_links:
                            if link.get("item_id") == agenda_item.item_id:
                                nested_links = link.get("nested_links", [])
                                documentation = self._extract_item_documentation(
                                    agenda_item.item_id, 
                                    meeting_id, 
                                    base_dir, 
                                    nested_links
                                )
                                agenda_item.documentation = documentation
                                break
                    except Exception as e:
                        logger.warning(f"Failed to load documentation for item {agenda_item.item_id}: {e}")
        
        logger.info(f"Processed {len(agenda_items)} agenda items for meeting {meeting_id}")
        return agenda_items

    def _extract_item_documentation(self, item_id: str, meeting_id: str, base_dir: Optional[Path], 
                                  nested_links: List[Dict[str, Any]]) -> List[DocumentationData]:
        """Extract documentation for an agenda item"""
        documentation: List[DocumentationData] = []
        
        if not base_dir:
            return documentation
        
        # Look for HTML file
        html_file = base_dir / f"item_{item_id}_{meeting_id}.html"
        if html_file.exists():
            try:
                with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                    html_content = f.read()
                
                doc = DocumentationData(
                    doc_id=f"html_{item_id}_{meeting_id}",
                    filename=html_file.name,
                    file_path=str(html_file),
                    doc_type="html",
                    content=self._html_to_text(html_content),
                    metadata={"size": len(html_content)}
                )
                documentation.append(doc)
            except Exception as e:
                logger.warning(f"Failed to read HTML for item {item_id}: {e}")
        
        # Look for nested PDF files
        pdf_pattern = f"item_{item_id}_{meeting_id}_nested_*.pdf"
        pdf_files = list(base_dir.glob(pdf_pattern))
        
        for pdf_file in pdf_files:
            try:
                with open(pdf_file, "rb") as f:
                    pdf_content = f.read()
                
                doc = DocumentationData(
                    doc_id=f"pdf_{pdf_file.stem}",
                    filename=pdf_file.name,
                    file_path=str(pdf_file),
                    doc_type="nested_pdf",
                    content=self._extract_pdf_text(pdf_content),
                    metadata={"size": len(pdf_content)}
                )
                documentation.append(doc)
            except Exception as e:
                logger.warning(f"Failed to read PDF {pdf_file}: {e}")
        
        return documentation

    def _extract_item_description(self, item_id: str, meeting_id: str, base_dir: Optional[Path]) -> str:
        """Extract description for an agenda item"""
        if not base_dir:
            return ""
        
        # Try to get description from HTML file
        html_file = base_dir / f"item_{item_id}_{meeting_id}.html"
        if html_file.exists():
            try:
                with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                    html_content = f.read()
                return self._html_to_text(html_content)[:500]  # Limit to 500 chars
            except Exception:
                pass
        
        return ""

    def _extract_meeting_type(self, title: str) -> str:
        """Extract meeting type from title"""
        title_lower = title.lower()
        
        if "council meeting" in title_lower or "regular" in title_lower or "town council" in title_lower:
            return "Regular Council Meeting"
        elif "cra" in title_lower:
            return "CRA Meeting"
        elif "budget hearing" in title_lower or "budget" in title_lower:
            return "Budget Hearing"
        elif "special assessment" in title_lower:
            return "Special Assessment Hearing"
        elif "workshop" in title_lower:
            return "Workshop Meeting"
        else:
            return "Other Meeting"

    def _extract_meeting_date(self, title: str) -> str:
        """Extract meeting date from title"""
        date_patterns = [
            r'(\w+ \d{1,2},? \d{4})',  # January 17, 2024
            r'(\w+ \d{1,2}st,? \d{4})',  # January 17th, 2024
            r'(\w+ \d{1,2}nd,? \d{4})',  # January 17nd, 2024
            r'(\w+ \d{1,2}rd,? \d{4})',  # January 17rd, 2024
            r'(\w+ \d{1,2}th,? \d{4})',  # January 17th, 2024
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, title)
            if match:
                return match.group(1)
        
        return "Unknown Date"

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize various date formats to YYYY-MM-DD"""
        if not date_str or date_str == "Unknown Date":
            return None
        date_str = date_str.replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
        date_formats = [
            "%B %d, %Y",  # January 17, 2024
            "%b %d, %Y",  # Jan 17, 2024
            "%B %d %Y",   # January 17 2024
            "%b %d %Y",   # Jan 17 2024
        ]
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                continue
        return None

    def _agenda_meeting_type(self, metadata: Dict[str, Any]) -> str:
        """Extract meeting type from agenda metadata"""
        info = metadata.get("meeting_info", {}) if isinstance(metadata, dict) else {}
        mt = info.get("meeting_type", "")
        return mt or "Other Meeting"

    async def process_all_data(self, load_into_stores: bool = True):
        """Process and optionally load all data into the enhanced RAG system"""
        try:
            logger.info("🚀 Starting enhanced data processing...")
            
            # Load data
            transcripts = self.load_youtube_transcripts()
            agendas = self.load_agenda_data()
            
            logger.info(f"📊 Loaded {len(transcripts)} transcripts and {len(agendas)} agendas")
            
            # Match and process meeting data
            meeting_data_list = self.enhanced_match_meeting_data(transcripts, agendas)
            logger.info(f"🔗 Enhanced matching: {len(meeting_data_list)} meetings")
            
            # If not loading, just report stats and return
            if not load_into_stores:
                total_agenda_items = sum(len(m.agenda.items) if m.agenda else 0 for m in meeting_data_list)
                total_docs = sum(
                    sum(len(item.documentation) for item in m.agenda.items) 
                    for m in meeting_data_list if m.agenda
                )
                logger.info(f"📝 Enhanced summary: {total_agenda_items} agenda items across {len(meeting_data_list)} meetings")
                logger.info(f"📄 Documentation: {total_docs} documents extracted")
                return {
                    "meetings": len(meeting_data_list),
                    "agenda_items": total_agenda_items,
                    "documents": total_docs
                }
            
            # Load into RAG system (placeholder for now)
            if not self.rag_system:
                raise RuntimeError("RAG system not initialized but load_into_stores=True")
            
            logger.info("🎉 Enhanced data processing completed successfully!")
            return {"status": "success", "meetings_processed": len(meeting_data_list)}
            
        except Exception as e:
            logger.error(f"❌ Failed to process enhanced data: {e}")
            raise

    def close(self):
        """Clean up resources"""
        if self.rag_system:
            self.rag_system.close()


async def main():
    """Main function to process all enhanced data"""
    parser = argparse.ArgumentParser(description="Process enhanced hierarchical RAG data")
    parser.add_argument("--chunk-only", action="store_true", help="Only run processing and report stats; do not load into stores")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of meetings to process (0 = no limit)")
    args = parser.parse_args()

    processor = EnhancedDataProcessor(init_rag_system=not args.chunk_only)
    
    try:
        if args.limit and args.limit > 0:
            # Monkey-patch load_youtube_transcripts to limit results
            original_loader = processor.load_youtube_transcripts
            def limited_loader():
                items = original_loader()
                return items[: args.limit]
            processor.load_youtube_transcripts = limited_loader  # type: ignore
        
        await processor.process_all_data(load_into_stores=not args.chunk_only)
    finally:
        processor.close()


if __name__ == "__main__":
    asyncio.run(main()) 