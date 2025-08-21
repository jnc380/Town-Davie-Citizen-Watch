#!/usr/bin/env python3
"""
Test script for reranking functionality
"""

import asyncio
import os
from milvus_only_rag_system_simple import MilvusOnlyRAGSystem

async def test_reranking():
    """Test the reranking functionality"""
    
    print("🧪 Testing Reranking Functionality")
    print("=" * 50)
    
    # Initialize the RAG system
    try:
        rag_system = MilvusOnlyRAGSystem()
        print("✅ RAG system initialized successfully")
        print(f"📊 Enable reranking: {rag_system.enable_reranking}")
        print(f"📊 Reranking model: {rag_system.reranking_model}")
        print(f"📊 BM25 rerank: {rag_system.bm25_rerank}")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return
    
    # Test queries that should benefit from reranking
    test_queries = [
        "What was discussed about storage sheds?",
        "Tell me about Pine Island Road improvements",
        "What happened with the Orange Blossom Parade?",
        "What are the budget issues for 2025?",
        "Tell me about the March 5th meeting"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 40)
        
        try:
            # Test the full pipeline with reranking
            result = await rag_system.process_query(query)
            
            print(f"✅ Query processed successfully")
            print(f"📊 Search results: {result['search_results_count']}")
            print(f"📊 Reranking applied: {result.get('reranking_applied', False)}")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            print(f"📊 Sources: {len(result['sources'])}")
            
            # Show the answer
            print(f"💬 Answer: {result['answer'][:300]}...")
            
            # Show detailed source information
            if result['sources']:
                print(f"\n📋 Sources with explanations:")
                for j, source in enumerate(result['sources'][:3], 1):
                    print(f"  {j}. Summary: {source.get('summary', 'No summary')}")
                    print(f"     Why relevant: {source.get('why', 'No explanation')}")
                    print(f"     Score: {source.get('score', 'N/A')}")
                    print(f"     Meeting: {source.get('meeting_type', 'Unknown')} - {source.get('meeting_date', 'Unknown')}")
                    if source.get('evidence_quote'):
                        print(f"     Evidence: \"{source.get('evidence_quote', '')[:100]}...\"")
                    print()
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50)

async def test_reranking_only():
    """Test just the reranking functionality with mock data"""
    
    print("🧪 Testing Reranking with Mock Data")
    print("=" * 50)
    
    try:
        rag_system = MilvusOnlyRAGSystem()
        
        # Create mock search results
        from milvus_only_rag_system_simple import SearchResult
        
        mock_results = [
            SearchResult(
                chunk_id="chunk_1",
                content="The council discussed storage sheds in residential areas. Many residents complained about oversized sheds.",
                document_type="transcript",
                meeting_id="meeting_1",
                meeting_date="2025-03-05",
                meeting_type="Town Council",
                hierarchy="OPEN PUBLIC MEETING",
                section_header="Storage Shed Discussion",
                metadata={},
                score=0.85,
                search_type="dense"
            ),
            SearchResult(
                chunk_id="chunk_2", 
                content="Pine Island Road improvements were approved. The project will cost $2.5 million.",
                document_type="transcript",
                meeting_id="meeting_2",
                meeting_date="2025-02-19",
                meeting_type="Town Council",
                hierarchy="NEW BUSINESS",
                section_header="Road Improvements",
                metadata={},
                score=0.75,
                search_type="sparse"
            ),
            SearchResult(
                chunk_id="chunk_3",
                content="The Orange Blossom Parade route was changed due to construction on Main Street.",
                document_type="transcript", 
                meeting_id="meeting_3",
                meeting_date="2025-01-15",
                meeting_type="Town Council",
                hierarchy="OLD BUSINESS",
                section_header="Parade Planning",
                metadata={},
                score=0.65,
                search_type="dense"
            )
        ]
        
        query = "What was discussed about storage sheds?"
        
        print(f"🔍 Testing GPT reranking with query: '{query}'")
        print(f"📊 Input results: {len(mock_results)}")
        
        # Test GPT reranking
        reranked = await rag_system._rerank_with_gpt(query, mock_results, 3)
        
        print(f"✅ Reranking completed")
        print(f"📊 Output results: {len(reranked)}")
        
        for i, result in enumerate(reranked, 1):
            print(f"  {i}. {result.chunk_id} - Score: {result.score:.3f}")
            print(f"     Content: {result.content[:100]}...")
            print()
        
    except Exception as e:
        print(f"❌ Error in reranking test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Reranking Test Suite")
    print("=" * 50)
    
    # Check environment
    print("🔧 Environment Check:")
    print(f"   OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print(f"   ENABLE_RERANKING: {os.getenv('ENABLE_RERANKING', 'true')}")
    print(f"   RERANKING_MODEL: {os.getenv('RERANKING_MODEL', 'gpt-4o-mini')}")
    print()
    
    # Run tests
    asyncio.run(test_reranking_only())
    print("\n" + "="*50)
    asyncio.run(test_reranking()) 