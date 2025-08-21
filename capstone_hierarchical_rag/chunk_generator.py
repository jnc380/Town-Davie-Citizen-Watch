#!/usr/bin/env python3
"""
Chunk Generator
Creates Milvus-ready chunks from alignment data with concatenated transcript_text
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for a chunk"""
    chunk_type: str
    item_id: Optional[str] = None
    item_number: Optional[str] = None
    sub_header: Optional[str] = None
    section: Optional[str] = None
    hierarchy: Optional[str] = None
    meeting_id: Optional[str] = None
    meeting_date: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    confidence: Optional[float] = None
    url: Optional[str] = None
    agenda_title: Optional[str] = None
    has_pdf: Optional[bool] = None
    pdf_url: Optional[str] = None
    neo4j_node_id: Optional[str] = None
    transcript_segment_count: Optional[int] = None
    total_transcript_duration: Optional[float] = None
    transcript_text: Optional[str] = None
    # Section-specific fields
    section_level: Optional[str] = None
    item_count: Optional[int] = None
    item_ids: Optional[List[str]] = None
    sub_headers: Optional[List[str]] = None

@dataclass
class Chunk:
    """A chunk for Milvus storage"""
    content: str
    metadata: ChunkMetadata

class ChunkGenerator:
    """Generates chunks from alignment data"""
    
    def __init__(self):
        self.alignment_report_path = Path("complete_15_meetings_with_simple_data.json")
        self.output_path = Path("generated_chunks.json")
        
    def concatenate_transcript_segments(self, evidence: List[Any]) -> str:
        """Concatenate all transcript segment text into one field"""
        if not evidence:
            return ""
        
        # Extract text from evidence (could be strings or dicts)
        transcript_parts = []
        for item in evidence:
            if isinstance(item, dict):
                # Extract text from dictionary
                text = item.get("text", "")
                if text:
                    transcript_parts.append(text)
            elif isinstance(item, str):
                # Direct string
                transcript_parts.append(item)
            else:
                # Try to convert to string
                transcript_parts.append(str(item))
        
        transcript_text = " ".join(transcript_parts)
        return transcript_text
    
    def create_agenda_item_chunk(self, alignment: Dict[str, Any], meeting_id: str) -> Chunk:
        """Create an agenda item-level chunk"""
        
        # Concatenate transcript text
        transcript_text = self.concatenate_transcript_segments(alignment.get("evidence", []))
        
        # Create content
        content_parts = []
        
        # Item header
        item_num = alignment.get("item_number", "N/A")
        sub_header = alignment.get("sub_header", "N/A")
        content_parts.append(f"Item {item_num} - {alignment.get('agenda_title', '')}")
        
        # Hierarchy
        section = alignment.get("section", "N/A")
        hierarchy = f"{section} -> {sub_header} -> Item {item_num}"
        content_parts.append(f"\nHierarchy: {hierarchy}")
        
        # Agenda details
        content_parts.append("\nAgenda Details:")
        content_parts.append(f"- Title: {alignment.get('agenda_title', '')}")
        content_parts.append(f"- Sub-header: {sub_header}")
        content_parts.append(f"- Item Number: {item_num}")
        content_parts.append(f"- Section: {section}")
        
        # Transcript discussion
        content_parts.append(f"\nTranscript Discussion:\n{transcript_text}")
        
        content = "\n".join(content_parts)
        
        # Create metadata
        metadata = ChunkMetadata(
            chunk_type="agenda_item",
            item_id=alignment.get("agenda_item_id"),
            item_number=item_num,
            sub_header=sub_header,
            section=section,
            hierarchy=hierarchy,
            meeting_id=meeting_id,
            meeting_date=self._extract_meeting_date(meeting_id),
            start_time=alignment.get("start_time"),
            end_time=alignment.get("end_time"),
            duration=alignment.get("duration"),
            confidence=alignment.get("confidence"),
            url=self._generate_youtube_url(alignment.get("start_time")),
            agenda_title=alignment.get("agenda_title"),
            has_pdf=True,  # Default assumption
            pdf_url=self._generate_pdf_url(alignment.get("agenda_item_id")),
            neo4j_node_id=f"agenda_item_{alignment.get('agenda_item_id')}",
            transcript_segment_count=len(alignment.get("evidence", [])),
            total_transcript_duration=alignment.get("duration"),
            transcript_text=transcript_text
        )
        
        return Chunk(content=content, metadata=metadata)
    
    def create_section_chunk(self, alignments: List[Dict[str, Any]], meeting_id: str, section: str) -> Optional[Chunk]:
        """Create a section-level chunk"""
        
        if not alignments:
            return None
        
        # Get section info
        section_level = self._extract_section_level(section)
        item_count = len(alignments)
        item_ids = [a.get("agenda_item_id") for a in alignments]
        sub_headers = list(set([a.get("sub_header") for a in alignments if a.get("sub_header")]))
        
        # Calculate section timing
        start_time = min(a.get("start_time", 0) for a in alignments)
        end_time = max(a.get("end_time", 0) for a in alignments)
        duration = end_time - start_time
        
        # Concatenate all transcript text for the section
        all_evidence = []
        for alignment in alignments:
            all_evidence.extend(alignment.get("evidence", []))
        transcript_text = self.concatenate_transcript_segments(all_evidence)
        
        # Create content
        content_parts = []
        content_parts.append(f"Section: {section}")
        content_parts.append(f"\nHierarchy: {section}")
        
        # Agenda items in this section
        content_parts.append(f"\nAgenda Items in this section:")
        for alignment in alignments:
            item_num = alignment.get("item_number", "N/A")
            sub_header = alignment.get("sub_header", "N/A")
            title = alignment.get("agenda_title", "")[:60] + "..." if len(alignment.get("agenda_title", "")) > 60 else alignment.get("agenda_title", "")
            content_parts.append(f"- Item {item_num} ({sub_header}): {title}")
        
        # Full discussion
        content_parts.append(f"\nFull Discussion:\n{transcript_text}")
        
        content = "\n".join(content_parts)
        
        # Create metadata
        metadata = ChunkMetadata(
            chunk_type="section",
            section=section,
            section_level=section_level,
            meeting_id=meeting_id,
            meeting_date=self._extract_meeting_date(meeting_id),
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            item_count=item_count,
            item_ids=item_ids,
            sub_headers=sub_headers,
            url=self._generate_youtube_url(start_time),
            neo4j_node_id=f"section_{section.lower().replace(' ', '_')}_{meeting_id.lower().replace(' ', '_')}",
            transcript_text=transcript_text
        )
        
        return Chunk(content=content, metadata=metadata)
    
    def _extract_meeting_date(self, meeting_id: str) -> str:
        """Extract meeting date from meeting ID"""
        # Extract date from meeting ID like "Town_of_Davie_Town_Council_Meeting_June_3rd_2025"
        date_match = re.search(r'(\w+)_(\d+)(?:st|nd|rd|th)_(\d{4})', meeting_id)
        if date_match:
            month, day, year = date_match.groups()
            # Convert month name to number
            month_map = {
                'January': '01', 'February': '02', 'March': '03', 'April': '04',
                'May': '05', 'June': '06', 'July': '07', 'August': '08',
                'September': '09', 'October': '10', 'November': '11', 'December': '12'
            }
            month_num = month_map.get(month, '01')
            return f"{year}-{month_num}-{day.zfill(2)}"
        return "2025-01-01"  # Default
    
    def _extract_section_level(self, section: str) -> str:
        """Extract section level (e.g., 'V' from 'V. APPROVAL OF CONSENT AGENDA')"""
        match = re.match(r'^([IVX]+)\.', section)
        return match.group(1) if match else "Unknown"
    
    def _generate_youtube_url(self, start_time: float) -> str:
        """Generate YouTube URL with timestamp"""
        return f"https://youtube.com/watch?v=example&t={int(start_time)}"
    
    def _generate_pdf_url(self, item_id: str) -> str:
        """Generate PDF URL for agenda item"""
        return f"https://davie.novusagenda.com/agendapublic/CoverSheet.aspx?ItemID={item_id}"
    
    def generate_chunks(self) -> List[Chunk]:
        """Generate all chunks from alignment data"""
        logger.info("🚀 Starting chunk generation...")
        
        if not self.alignment_report_path.exists():
            logger.error(f"❌ Alignment report not found: {self.alignment_report_path}")
            return []
        
        # Load alignment data
        with open(self.alignment_report_path, 'r', encoding='utf-8') as f:
            alignment_data = json.load(f)
        
        all_chunks = []
        
        # Process each meeting
        for meeting_data in alignment_data.get("detailed_alignments", []):
            meeting_id = meeting_data.get("meeting_id")
            alignments = meeting_data.get("alignments", [])
            
            logger.info(f"📋 Processing meeting: {meeting_id}")
            logger.info(f"   - Found {len(alignments)} alignments")
            
            # Group alignments by section
            sections = {}
            for alignment in alignments:
                section = alignment.get("section", "Unknown")
                if section not in sections:
                    sections[section] = []
                sections[section].append(alignment)
            
            # Create agenda item chunks
            for alignment in alignments:
                chunk = self.create_agenda_item_chunk(alignment, meeting_id)
                all_chunks.append(chunk)
            
            # Create section chunks
            for section, section_alignments in sections.items():
                chunk = self.create_section_chunk(section_alignments, meeting_id, section)
                if chunk:
                    all_chunks.append(chunk)
            
            logger.info(f"   ✅ Created {len(alignments)} item chunks + {len(sections)} section chunks")
        
        logger.info(f"🎯 Total chunks generated: {len(all_chunks)}")
        return all_chunks
    
    def save_chunks(self, chunks: List[Chunk]):
        """Save chunks to JSON file"""
        logger.info(f"💾 Saving {len(chunks)} chunks to {self.output_path}")
        
        # Convert chunks to serializable format
        serializable_chunks = []
        for chunk in chunks:
            chunk_dict = {
                "content": chunk.content,
                "metadata": asdict(chunk.metadata)
            }
            serializable_chunks.append(chunk_dict)
        
        # Save to file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Chunks saved successfully")
    
    def show_chunk_summary(self, chunks: List[Chunk]):
        """Show summary of generated chunks"""
        logger.info("\n📊 CHUNK GENERATION SUMMARY")
        logger.info("=" * 50)
        
        # Count by type
        item_chunks = [c for c in chunks if c.metadata.chunk_type == "agenda_item"]
        section_chunks = [c for c in chunks if c.metadata.chunk_type == "section"]
        
        logger.info(f"Agenda Item Chunks: {len(item_chunks)}")
        logger.info(f"Section Chunks: {len(section_chunks)}")
        logger.info(f"Total Chunks: {len(chunks)}")
        
        # Show sample chunks
        if item_chunks:
            logger.info(f"\n🎯 SAMPLE AGENDA ITEM CHUNK:")
            sample = item_chunks[0]
            logger.info(f"Item ID: {sample.metadata.item_id}")
            logger.info(f"Title: {sample.metadata.agenda_title[:60]}...")
            logger.info(f"Section: {sample.metadata.section}")
            logger.info(f"Duration: {sample.metadata.duration:.1f}s")
            logger.info(f"Transcript Text Length: {len(sample.metadata.transcript_text)} chars")
            logger.info(f"Transcript Text Preview: {sample.metadata.transcript_text[:100]}...")
        
        if section_chunks:
            logger.info(f"\n📋 SAMPLE SECTION CHUNK:")
            sample = section_chunks[0]
            logger.info(f"Section: {sample.metadata.section}")
            logger.info(f"Item Count: {sample.metadata.item_count}")
            logger.info(f"Duration: {sample.metadata.duration:.1f}s")
            logger.info(f"Transcript Text Length: {len(sample.metadata.transcript_text)} chars")
            logger.info(f"Transcript Text Preview: {sample.metadata.transcript_text[:100]}...")

def main():
    """Main function"""
    generator = ChunkGenerator()
    
    # Generate chunks
    chunks = generator.generate_chunks()
    
    if chunks:
        # Save chunks
        generator.save_chunks(chunks)
        
        # Show summary
        generator.show_chunk_summary(chunks)
    else:
        logger.error("❌ No chunks generated")

if __name__ == "__main__":
    main() 