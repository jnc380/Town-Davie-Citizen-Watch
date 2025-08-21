#!/usr/bin/env python3
"""
Data Processor for Capstone Project
Processes YouTube transcripts and agenda data for the hybrid RAG system
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import asyncio
import argparse
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MeetingData:
    meeting_id: str
    meeting_type: str
    meeting_date: str
    title: str
    transcript_chunks: List[Dict[str, Any]]
    agenda_items: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class DataProcessor:
    """Processes and loads data into the hybrid RAG system"""
    
    def __init__(self, init_rag_system: bool = True):
        """Initialize the data processor"""
        self.rag_system = None
        self.init_rag_system = init_rag_system
        if init_rag_system:
            # Lazy import to avoid initializing OpenAI/servers when chunk-only
            from hybrid_rag_system import HybridRAGSystem  # type: ignore
            self.rag_system = HybridRAGSystem()
        # Base directories resolved relative to this file, with env overrides for flexibility
        capstone_base = Path(__file__).resolve().parent
        default_downloads = capstone_base / "downloads"
        # YouTube transcripts directory can be overridden via env
        self.youtube_dir = Path(os.getenv("YOUTUBE_DOWNLOAD_DIR", str(default_downloads / "town_meetings_youtube")))
        # Agendas directory can be overridden via env
        self.agendas_dir = Path(os.getenv("AGENDAS_DIR", str(default_downloads / "agendas")))
        
        # Year filter: only process meetings whose meeting_date is in this year
        self.year_filter = os.getenv("YEAR_FILTER", "2025")
        
        logger.info("✅ Data processor initialized")

    def _fetch_bytes(self, url: str, timeout: float = 15.0) -> Optional[bytes]:
        """Fetch bytes from URL using httpx if available, otherwise urllib. Returns None on failure."""
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
        """Lightweight HTML to text: strip tags and collapse whitespace."""
        try:
            # Try BeautifulSoup if available for better parsing
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
        """Extract text from PDF bytes using pdfminer.six if available; else return empty string."""
        try:
            # Minimal dependency path; skip if not installed
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

                # Filter by year using title date or upload_date
                title = info_data.get("title", meeting_id)
                date_raw = self._extract_meeting_date(title)
                date_norm = self._normalize_date(date_raw) if date_raw else None
                year_ok = False
                if date_norm and len(date_norm) >= 4:
                    year_ok = date_norm.startswith(self.year_filter)
                else:
                    # Fallback to upload_date like YYYYMMDD
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
        """Load all agenda data"""
        agendas = []
        
        if not self.agendas_dir.exists():
            logger.warning(f"Agendas directory not found: {self.agendas_dir}")
            return agendas
        
        # Find all agenda directories
        agenda_dirs = [d for d in self.agendas_dir.iterdir() if d.is_dir() and d.name.startswith(f"{self.year_filter}-")]
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
                
                agendas.append({
                    "meeting_id": meeting_id,
                    "metadata": metadata,
                    "agenda_dir": str(agenda_dir)
                })
                
                logger.info(f"✅ Loaded agenda: {meeting_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load agenda {agenda_dir}: {e}")
        
        return agendas
    
    def process_transcript_chunks(self, transcript_data: List[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Process transcript data into chunks with overlap for context preservation"""
        if chunk_size <= overlap:
            overlap = max(0, chunk_size // 5)
        
        chunks: List[Dict[str, Any]] = []
        buffer_segments: List[Dict[str, Any]] = []
        buffer_text: str = ""
        
        def flush_chunk(seg_list: List[Dict[str, Any]], text: str):
            if not seg_list or not text.strip():
                return
            chunks.append({
                "text": text.strip(),
                "segments": seg_list.copy(),
                "start": seg_list[0].get("start", 0),
                "duration": sum(seg.get("duration", 0) for seg in seg_list)
            })
        
        for segment in transcript_data:
            text = segment.get("text", "").strip()
            if not text or text in {"[Music]", "e"}:
                continue
            
            next_text = (buffer_text + " " + text).strip() if buffer_text else text
            buffer_segments.append(segment)
            buffer_text = next_text
            
            if len(buffer_text) >= chunk_size:
                # Flush current chunk
                flush_chunk(buffer_segments, buffer_text)
                
                # Prepare overlap tail for next chunk
                if overlap > 0:
                    # Build overlap tail from the end of buffer_text
                    tail_text = buffer_text[-overlap:]
                    # Reduce segments to approximately cover tail_text window
                    tail_segments: List[Dict[str, Any]] = []
                    total_text = ""
                    for seg in reversed(buffer_segments):
                        seg_text = seg.get("text", "").strip()
                        if not seg_text:
                            continue
                        total_text = (seg_text + " " + total_text).strip()
                        tail_segments.insert(0, seg)
                        if len(total_text) >= overlap:
                            break
                    buffer_segments = tail_segments
                    buffer_text = total_text
                else:
                    buffer_segments = []
                    buffer_text = ""
        
        # Flush remaining
        flush_chunk(buffer_segments, buffer_text)
        return chunks
    
    def extract_agenda_items(self, metadata: Dict[str, Any], agenda_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract agenda items from meeting metadata, fetching CoverSheet HTML and optional PDF text when possible."""
        agenda_items: List[Dict[str, Any]] = []
        base_dir = Path(agenda_dir) if agenda_dir else None
        attachments_dir = (base_dir / "attachments") if base_dir else None
        if attachments_dir and not attachments_dir.exists():
            try:
                attachments_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        # Map nested links by parent_item_id for quick lookup
        nested_links = metadata.get("nested_links", [])
        nested_by_parent: Dict[str, List[Dict[str, Any]]] = {}
        for nl in nested_links:
            pid = nl.get("parent_item_id")
            if pid:
                nested_by_parent.setdefault(pid, []).append(nl)

        # Extract from pdf_links if available
        pdf_links = metadata.get("pdf_links", [])
        for i, link in enumerate(pdf_links):
            item_id = link.get("item_id", f"item_{i}")
            meeting_id = link.get("meeting_id", "")
            url = link.get("url", "")
            title_text = link.get("text", "").strip()

            coversheet_text = ""
            html_file: Optional[Path] = None
            if base_dir and item_id and meeting_id:
                html_file = base_dir / f"item_{item_id}_{meeting_id}.html"
            # Load saved HTML or fetch and save if missing
            if html_file and html_file.exists():
                try:
                    with open(html_file, "r", encoding="utf-8", errors="ignore") as hf:
                        html = hf.read()
                    coversheet_text = self._html_to_text(html)
                except Exception as e:
                    logger.warning(f"Could not read CoverSheet HTML for item {item_id}: {e}")
            elif url and "CoverSheet.aspx" in url:
                bytes_html = self._fetch_bytes(url)
                if bytes_html and html_file:
                    try:
                        html_str = bytes_html.decode("utf-8", errors="ignore")
                        with open(html_file, "w", encoding="utf-8") as out:
                            out.write(html_str)
                        coversheet_text = self._html_to_text(html_str)
                        logger.info(f"Saved CoverSheet HTML for item {item_id} to {html_file}")
                    except Exception as e:
                        logger.warning(f"Failed to save CoverSheet HTML for item {item_id}: {e}")
                elif bytes_html:
                    coversheet_text = self._html_to_text(bytes_html.decode("utf-8", errors="ignore"))

            # Append any nested link titles and attempt minimal PDF extraction
            nested_text_bits: List[str] = []
            for nl in nested_by_parent.get(item_id, []):
                nl_text = nl.get("text", "").strip()
                nl_url = nl.get("url", "")
                if nl_text:
                    nested_text_bits.append(f"Attachment: {nl_text}")
                if nl_url and nl_url.lower().endswith(".pdf"):
                    pdf_bytes = self._fetch_bytes(nl_url)
                    if pdf_bytes and attachments_dir is not None:
                        try:
                            pdf_path = attachments_dir / f"{item_id}_{meeting_id}_{i}.pdf"
                            with open(pdf_path, "wb") as pf:
                                pf.write(pdf_bytes)
                            extracted = self._extract_pdf_text(pdf_bytes)
                            if extracted:
                                nested_text_bits.append(extracted[:1000])
                        except Exception:
                            pass

            # Surface dollar amounts explicitly (boost retrieval)
            amounts: List[str] = []
            for source_text in (title_text, coversheet_text, " ".join(nested_text_bits)):
                if not source_text:
                    continue
                amounts.extend(re.findall(r"\$[0-9][0-9,]*\.?[0-9]{0,2}", source_text))
            amounts_str = f" Amounts: {'; '.join(sorted(set(amounts)))}" if amounts else ""

            full_text = " ".join([t for t in [title_text, coversheet_text, " ".join(nested_text_bits)] if t]).strip() + amounts_str

            agenda_items.append({
                "item_id": item_id,
                "text": full_text or title_text,
                "url": url,
                "order": i,
                "type": "agenda_item",
                "description": (coversheet_text or title_text)[:500],
            })

        return agenda_items
    
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
        import re
        
        # Look for date patterns
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
        """Normalize various date formats to YYYY-MM-DD."""
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
        info = metadata.get("meeting_info", {}) if isinstance(metadata, dict) else {}
        mt = info.get("meeting_type", "")
        return mt or "Other Meeting"

    def match_meeting_data(self, transcripts: List[Dict], agendas: List[Dict]) -> List[MeetingData]:
        """Match transcript and agenda data by meeting ID/date + type"""
        meeting_data_list: List[MeetingData] = []
        
        # Index agendas by normalized date
        agendas_by_date: Dict[str, List[Dict[str, Any]]] = {}
        for a in agendas:
            md = a.get("metadata", {})
            info = md.get("meeting_info", {})
            a_date_raw = info.get("meeting_date", "")
            a_date = self._normalize_date(a_date_raw)
            if a_date:
                agendas_by_date.setdefault(a_date, []).append(a)
        
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
            
            # Default no agenda
            agenda_data: Dict[str, Any] = {}
            
            # Try to match on date, then by meeting type
            if meeting_date_norm and meeting_date_norm in agendas_by_date:
                candidates = agendas_by_date[meeting_date_norm]
                if len(candidates) == 1:
                    agenda_data = candidates[0]
                else:
                    # Pick best match by meeting type similarity
                    def type_rank(a_md: Dict[str, Any]) -> int:
                        a_type = self._agenda_meeting_type(a_md.get("metadata", {})).lower()
                        t = meeting_type.lower()
                        if ("cra" in a_type) == ("cra" in t):
                            return 0
                        if ("workshop" in a_type) == ("workshop" in t):
                            return 1
                        if ("regular" in a_type) or ("council" in a_type):
                            return 2
                        return 3
                    agenda_data = sorted(candidates, key=type_rank)[0]
 
            # Process transcript chunks with overlap
            transcript_chunks = self.process_transcript_chunks(transcript["transcript"])
             
            # Extract agenda items
            agenda_items: List[Dict[str, Any]] = []
            if agenda_data:
                agenda_items = self.extract_agenda_items(agenda_data.get("metadata", {}), agenda_data.get("agenda_dir", None))
                used_agenda_dirs.add(agenda_data.get("agenda_dir", ""))
             
            # Create meeting data
            meeting_data = MeetingData(
                meeting_id=meeting_id,
                meeting_type=meeting_type,
                meeting_date=meeting_date_raw,
                title=title,
                transcript_chunks=transcript_chunks,
                agenda_items=agenda_items,
                metadata={
                    "url": info_data.get("url", info_data.get("webpage_url", "")),
                    "description": info_data.get("description", ""),
                    "upload_date": info_data.get("upload_date", ""),
                    "channel_name": info_data.get("channel_name", ""),
                    "has_agenda": bool(agenda_data),
                    "matched_agenda_dir": agenda_data.get("agenda_dir", "") if agenda_data else "",
                }
            )
             
            meeting_data_list.append(meeting_data)
            logger.info(f"✅ Processed meeting: {meeting_id} ({len(transcript_chunks)} chunks, {len(agenda_items)} agenda items)")
         
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
            agenda_items = self.extract_agenda_items(md, a_dir)
            if not agenda_items:
                continue
            meeting_id = a.get("meeting_id", a_dir)
            meeting_data = MeetingData(
                meeting_id=meeting_id,
                meeting_type=meeting_type,
                meeting_date=meeting_date_raw,
                title=title,
                transcript_chunks=[],
                agenda_items=agenda_items,
                metadata={
                    "url": "",
                    "description": info.get("description", ""),
                    "upload_date": "",
                    "channel_name": "",
                    "has_agenda": True,
                    "matched_agenda_dir": a_dir,
                }
            )
            meeting_data_list.append(meeting_data)
            logger.info(f"🗂️ Added agenda-only meeting: {meeting_id} ({len(agenda_items)} agenda items)")

        return meeting_data_list
    
    def _gather_corpus_texts(self, meetings: List[MeetingData]) -> List[str]:
        """Gather all chunk texts for TF-IDF vectorizer fitting."""
        texts: List[str] = []
        for md in meetings:
            for ch in md.transcript_chunks:
                texts.append(ch.get("text", ""))
            for ai in md.agenda_items:
                if ai.get("text"):
                    texts.append(ai.get("text"))
        return [t for t in texts if t]
    
    async def process_all_data(self, load_into_stores: bool = True):
        """Process and optionally load all data into the hybrid RAG system"""
        try:
            logger.info("🚀 Starting data processing...")
            
            # Load data
            transcripts = self.load_youtube_transcripts()
            agendas = self.load_agenda_data()
            
            logger.info(f"📊 Loaded {len(transcripts)} transcripts and {len(agendas)} agendas")
            
            # Match and process meeting data
            meeting_data_list = self.match_meeting_data(transcripts, agendas)
            logger.info(f"🔗 Matched {len(meeting_data_list)} meetings")
            
            # If not loading, just report stats and return
            if not load_into_stores:
                total_chunks = sum(len(m.transcript_chunks) for m in meeting_data_list)
                total_agenda_items = sum(len(m.agenda_items) for m in meeting_data_list)
                logger.info(f"🧩 Chunking summary: {total_chunks} transcript chunks across {len(meeting_data_list)} meetings")
                logger.info(f"📝 Agenda items: {total_agenda_items} extracted")
                return {
                    "meetings": len(meeting_data_list),
                    "transcript_chunks": total_chunks,
                    "agenda_items": total_agenda_items
                }
            
            # Fit sparse vectorizer on full corpus once
            corpus_texts = self._gather_corpus_texts(meeting_data_list)
            if corpus_texts and self.rag_system:
                self.rag_system.fit_sparse_vectorizer(corpus_texts)
                logger.info(f"🧠 Fitted TF-IDF vectorizer on {len(corpus_texts)} chunks")
            
            # Load into RAG system
            if not self.rag_system:
                raise RuntimeError("RAG system not initialized but load_into_stores=True")
            for i, meeting_data in enumerate(meeting_data_list):
                try:
                    await self.rag_system.load_meeting_data(meeting_data)
                    logger.info(f"✅ Loaded meeting {i+1}/{len(meeting_data_list)}: {meeting_data.meeting_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to load meeting {meeting_data.meeting_id}: {e}")
            
            logger.info("🎉 Data processing completed successfully!")
            
            # Print system stats
            stats = self.rag_system.get_system_stats()
            logger.info(f"📈 System stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to process data: {e}")
            raise
    
    def close(self):
        """Clean up resources"""
        if self.rag_system:
            self.rag_system.close()

async def main():
    """Main function to process all data"""
    parser = argparse.ArgumentParser(description="Process and optionally load capstone data")
    parser.add_argument("--chunk-only", action="store_true", help="Only run chunking and report stats; do not load into Milvus/Neo4j")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of meetings to process (0 = no limit)")
    args = parser.parse_args()

    processor = DataProcessor(init_rag_system=not args.chunk_only)
    
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