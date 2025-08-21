#!/usr/bin/env python3
"""
Enhanced Data Models for Hierarchical RAG System
Defines the data structures for Meeting -> Agenda -> AgendaItem -> Documentation hierarchy
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class DocumentationData:
    """Represents documentation associated with an agenda item"""
    doc_id: str
    filename: str
    file_path: str
    doc_type: str  # 'html', 'pdf', 'nested_pdf'
    content: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class TranscriptSegmentData:
    """Represents a transcript segment aligned to an agenda item"""
    segment_id: str
    start_time: float
    end_time: float
    text: str
    speakers: List[str]
    confidence: float
    evidence: Dict[str, Any] = None


@dataclass
class AgendaItemData:
    """Represents an agenda item with enhanced metadata"""
    item_id: str
    item_number: str
    title: str
    description: str
    order: int
    url: Optional[str] = None
    documentation: List[DocumentationData] = None
    transcript_segments: List[TranscriptSegmentData] = None
    concepts: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.documentation is None:
            self.documentation = []
        if self.transcript_segments is None:
            self.transcript_segments = []
        if self.concepts is None:
            self.concepts = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgendaData:
    """Represents an agenda for a meeting"""
    agenda_id: str
    meeting_id: str
    agenda_type: str  # 'regular', 'cra', 'workshop', 'special'
    meeting_date: str
    items: List[AgendaItemData] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.items is None:
            self.items = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EnhancedMeetingData:
    """Enhanced meeting data with hierarchical structure"""
    meeting_id: str
    meeting_type: str
    meeting_date: str
    title: str
    agenda: Optional[AgendaData] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ConceptData:
    """Represents a concept extracted from agenda items"""
    concept_id: str
    name: str
    category: str  # 'location', 'project', 'event', 'entity'
    description: str
    related_concepts: List[str] = None
    agenda_items: List[str] = None  # item_ids
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.related_concepts is None:
            self.related_concepts = []
        if self.agenda_items is None:
            self.agenda_items = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AlignmentResult:
    """Result of transcript-agenda alignment"""
    item_id: str
    start_time: float
    end_time: float
    confidence: float
    evidence: Dict[str, Any]
    transcript_segments: List[TranscriptSegmentData] = None

    def __post_init__(self):
        if self.transcript_segments is None:
            self.transcript_segments = []
        if self.evidence is None:
            self.evidence = {}


# Legacy compatibility - keep existing MeetingData for transition
@dataclass
class MeetingData:
    """Legacy meeting data structure for backward compatibility"""
    meeting_id: str
    meeting_type: str
    meeting_date: str
    title: str
    transcript_chunks: List[Dict[str, Any]]
    agenda_items: List[Dict[str, Any]]
    metadata: Dict[str, Any] 