#!/usr/bin/env python3
"""
Integrate Nested Chunks - Combine existing agenda-transcript chunks with nested document chunks
Creates a comprehensive dataset for the enhanced RAG system
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NestedChunkIntegrator:
    """Integrates nested document chunks with existing agenda-transcript chunks"""
    
    def __init__(self):
        self.existing_chunks_file = "final_generated_chunks_all_meetings.json"
        self.nested_chunks_file = "nested_documents_results.json"
        self.output_file = "comprehensive_rag_dataset.json"
    
    def load_existing_chunks(self) -> List[Dict[str, Any]]:
        """Load existing agenda-transcript chunks"""
        try:
            with open(self.existing_chunks_file, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} existing chunks from {self.existing_chunks_file}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {self.existing_chunks_file}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return []
    
    def load_nested_chunks(self) -> List[Dict[str, Any]]:
        """Load nested document chunks"""
        try:
            with open(self.nested_chunks_file, 'r') as f:
                data = json.load(f)
            
            chunks = data.get('chunks', [])
            logger.info(f"Loaded {len(chunks)} nested chunks from {self.nested_chunks_file}")
            return chunks
        except FileNotFoundError:
            logger.error(f"File not found: {self.nested_chunks_file}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return []
    
    def enhance_nested_chunk(self, chunk: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """Enhance nested chunk with additional metadata for RAG integration"""
        
        # Create unique chunk ID
        chunk_id = f"nested_{chunk['parent_item_id']}_{chunk['chunk_index']}_{chunk_index}"
        
        # Extract meeting info from parent document path
        parent_doc = chunk.get('parent_document', '')
        meeting_info = self._extract_meeting_info(parent_doc)
        
        # Create enhanced chunk structure
        enhanced_chunk = {
            "chunk_id": chunk_id,
            "content": chunk.get('content', ''),
            "content_type": "nested_document",
            "document_type": chunk.get('content_type', 'general_document'),
            "chunk_type": chunk.get('chunk_type', 'smart'),
            "parent_item_id": chunk.get('parent_item_id', ''),
            "parent_document": chunk.get('parent_document', ''),
            "section_header": chunk.get('section_header'),
            "chunk_index": chunk.get('chunk_index', 0),
            "total_chunks": chunk.get('total_chunks', 1),
            "hierarchy": f"agenda_item_{chunk.get('parent_item_id', '')}_nested_{chunk.get('chunk_index', 0)}",
            "metadata": {
                "source": "nested_document",
                "document_type": chunk.get('content_type', 'general_document'),
                "chunk_type": chunk.get('chunk_type', 'smart'),
                "parent_item_id": chunk.get('parent_item_id', ''),
                "parent_document": chunk.get('parent_document', ''),
                "section_header": chunk.get('section_header'),
                "chunk_index": chunk.get('chunk_index', 0),
                "total_chunks": chunk.get('total_chunks', 1),
                "meeting_date": meeting_info.get('date'),
                "meeting_type": meeting_info.get('type'),
                "meeting_id": meeting_info.get('id'),
                "hierarchy_level": "nested_document",
                "content_quality": "high" if len(chunk.get('content', '')) > 100 else "low"
            }
        }
        
        return enhanced_chunk
    
    def _extract_meeting_info(self, parent_doc: str) -> Dict[str, str]:
        """Extract meeting information from parent document path"""
        # Example: item_8433_583_nested_1.pdf -> meeting info
        try:
            # Extract item ID and meeting number
            parts = parent_doc.replace('.pdf', '').split('_')
            if len(parts) >= 3:
                item_id = parts[1]
                meeting_num = parts[2]
                
                # Map meeting numbers to dates/types (this would need to be enhanced)
                meeting_mapping = {
                    "575": {"date": "2025-01-15", "type": "Town Council", "id": "Town_of_Davie_Town_Council_Meeting_January_15_2025"},
                    "576": {"date": "2025-01-15", "type": "CRA", "id": "Town_of_Davie_CRA_Meeting_January_15_2025"},
                    "577": {"date": "2025-01-15", "type": "Town Council", "id": "Town_of_Davie_Town_Council_Meeting_January_15_2025"},
                    "579": {"date": "2025-02-05", "type": "Town Council", "id": "Town_of_Davie_Town_Council_Meeting_February_5th_2025"},
                    "580": {"date": "2025-02-05", "type": "CRA", "id": "Town_of_Davie_CRA_Meeting_February_5th_2025"},
                    "582": {"date": "2025-02-19", "type": "Town Council", "id": "Town_of_Davie_Town_Council_Meeting_February_19th_2025"},
                    "583": {"date": "2025-02-19", "type": "CRA", "id": "Town_of_Davie_CRA_Meeting_February_19th_2025"},
                    "585": {"date": "2025-03-05", "type": "Town Council", "id": "Town_of_Davie_Regular_Council_Meeting_March_5th_2025"},
                    "586": {"date": "2025-03-05", "type": "CRA", "id": "Town_of_Davie_CRA_Meeting_March_5th_2025"},
                    "587": {"date": "2025-03-19", "type": "Town Council", "id": "Town_of_Davie_Town_Council_Meeting_March_19_2025"},
                    "589": {"date": "2025-04-16", "type": "Town Council", "id": "Town_of_Davie_Town_Council_Meeting_April_16th_2025"},
                    "596": {"date": "2025-05-07", "type": "Town Council", "id": "Town_of_Davie_Town_Council_Meeting_May_7th_2025"},
                    "597": {"date": "2025-05-07", "type": "CRA", "id": "Town_of_Davie_CRA_Meeting_May_7th_2025"},
                    "598": {"date": "2025-05-21", "type": "Town Council", "id": "Town_of_Davie_Town_Council_Meeting_May_21st_2025"},
                    "599": {"date": "2025-05-21", "type": "CRA", "id": "Town_of_Davie_CRA_Meeting_May_21st_2025"},
                    "600": {"date": "2025-06-03", "type": "Town Council", "id": "Town_of_Davie_Town_Council_Meeting_June_3rd_2025"},
                    "608": {"date": "2025-06-03", "type": "CRA", "id": "Town_of_Davie_CRA_Meeting_June_3rd_2025"},
                    "612": {"date": "2025-06-17", "type": "Town Council", "id": "Town_of_Davie_Town_Council_Meeting_June_17th_2025"},
                    "613": {"date": "2025-07-23", "type": "Town Council", "id": "Town_of_Davie_Town_Council_Meeting_July_23_2025"},
                    "614": {"date": "2025-07-23", "type": "CRA", "id": "Town_of_Davie_CRA_Meeting_July_23_2025"},
                    "630": {"date": "2025-08-06", "type": "Town Council", "id": "Town_of_Davie_Town_Council_Meeting_August_6th_2025"},
                    "631": {"date": "2025-08-06", "type": "CRA", "id": "Town_of_Davie_CRA_Meeting_August_6th_2025"}
                }
                
                return meeting_mapping.get(meeting_num, {
                    "date": "unknown",
                    "type": "unknown", 
                    "id": f"meeting_{meeting_num}"
                })
        except Exception as e:
            logger.warning(f"Could not extract meeting info from {parent_doc}: {e}")
        
        return {"date": "unknown", "type": "unknown", "id": "unknown"}
    
    def integrate_chunks(self) -> List[Dict[str, Any]]:
        """Integrate existing and nested chunks into comprehensive dataset"""
        
        # Load existing chunks
        existing_chunks = self.load_existing_chunks()
        if not existing_chunks:
            logger.error("No existing chunks found!")
            return []
        
        # Load nested chunks
        nested_chunks = self.load_nested_chunks()
        if not nested_chunks:
            logger.warning("No nested chunks found!")
            return existing_chunks
        
        # Enhance existing chunks with source metadata
        enhanced_existing = []
        for chunk in existing_chunks:
            enhanced_chunk = chunk.copy()
            enhanced_chunk['content_type'] = 'agenda_transcript'
            enhanced_chunk['metadata']['source'] = 'agenda_transcript'
            enhanced_chunk['metadata']['hierarchy_level'] = 'agenda_item'
            enhanced_existing.append(enhanced_chunk)
        
        # Enhance nested chunks
        enhanced_nested = []
        for i, chunk in enumerate(nested_chunks):
            enhanced_chunk = self.enhance_nested_chunk(chunk, i)
            enhanced_nested.append(enhanced_chunk)
        
        # Combine all chunks
        comprehensive_dataset = enhanced_existing + enhanced_nested
        
        logger.info(f"Integrated {len(enhanced_existing)} existing chunks + {len(enhanced_nested)} nested chunks = {len(comprehensive_dataset)} total chunks")
        
        return comprehensive_dataset
    
    def save_comprehensive_dataset(self, dataset: List[Dict[str, Any]]) -> None:
        """Save comprehensive dataset to file"""
        
        output_data = {
            "dataset_info": {
                "total_chunks": len(dataset),
                "source_chunks": len([c for c in dataset if c.get('content_type') == 'agenda_transcript']),
                "nested_chunks": len([c for c in dataset if c.get('content_type') == 'nested_document']),
                "document_types": {},
                "meeting_coverage": {}
            },
            "chunks": dataset
        }
        
        # Calculate statistics
        content_types = {}
        meeting_coverage = {}
        
        for chunk in dataset:
            # Content type stats
            content_type = chunk.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Meeting coverage stats
            meeting_id = chunk.get('metadata', {}).get('meeting_id', 'unknown')
            meeting_coverage[meeting_id] = meeting_coverage.get(meeting_id, 0) + 1
        
        output_data["dataset_info"]["document_types"] = content_types
        output_data["dataset_info"]["meeting_coverage"] = meeting_coverage
        
        # Save to file
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comprehensive dataset saved to {self.output_file}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 COMPREHENSIVE DATASET SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total chunks: {len(dataset)}")
        logger.info(f"Agenda-transcript chunks: {content_types.get('agenda_transcript', 0)}")
        logger.info(f"Nested document chunks: {content_types.get('nested_document', 0)}")
        logger.info(f"Meetings covered: {len(meeting_coverage)}")
        
        # Show top document types
        doc_types = {}
        for chunk in dataset:
            if chunk.get('content_type') == 'nested_document':
                doc_type = chunk.get('document_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        logger.info("Nested document types:")
        for doc_type, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {doc_type}: {count} chunks")

def main():
    """Main integration function"""
    
    integrator = NestedChunkIntegrator()
    
    # Integrate chunks
    logger.info("🚀 Starting chunk integration...")
    comprehensive_dataset = integrator.integrate_chunks()
    
    if comprehensive_dataset:
        # Save comprehensive dataset
        logger.info("💾 Saving comprehensive dataset...")
        integrator.save_comprehensive_dataset(comprehensive_dataset)
        
        logger.info("✅ Integration complete!")
        return comprehensive_dataset
    else:
        logger.error("❌ Integration failed!")
        return []

if __name__ == "__main__":
    main() 