#!/usr/bin/env python3
"""
Test script for PyMilvus RAG system
"""

import asyncio
import os
from milvus_pymilvus_rag import MilvusPyMilvusRAGSystem

async def test_pymilvus_rag():
    """Test the PyMilvus RAG system"""
    
    print("🧪 Testing PyMilvus RAG System")
    print("=" * 50)
    
    # Initialize the RAG system
    try:
        rag_system = MilvusPyMilvusRAGSystem()
        print("✅ RAG system initialized successfully")
        print(f"📊 Milvus URI: {rag_system.milvus_uri[:50]}...")
        print(f"📊 Collection: {rag_system.collection_name}")
        print(f"📊 OpenAI configured: {'Yes' if rag_system.openai_api_key else 'No'}")
        print(f"📊 Enable reranking: {rag_system.enable_reranking}")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test queries
    test_queries = [
        "What was discussed about storage sheds?",
        "Tell me about the Orange Blossom Parade",
        "What happened at the March 5th meeting?",
        "What are the current budget issues?",
        "Tell me about Pine Island Road"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 40)
        
        try:
            # Test hybrid search
            print("🔎 Performing hybrid search...")
            search_results = await rag_system.hybrid_search(query, limit=5)
            print(f"📊 Found {len(search_results)} search results")
            
            if search_results:
                print("📋 Top 3 results:")
                for j, result in enumerate(search_results[:3], 1):
                    print(f"  {j}. Score: {result.score:.3f} | Type: {result.search_type}")
                    print(f"     Meeting: {result.meeting_type} - {result.meeting_date}")
                    print(f"     Section: {result.section_header}")
                    print(f"     Content: {result.content[:100]}...")
                    print()
            
            # Test answer generation
            print("🤖 Generating answer...")
            result = await rag_system.process_query(query)
            
            print(f"✅ Answer generated successfully")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            print(f"📊 Sources: {len(result['sources'])}")
            print(f"📊 Search types: {result['search_types_used']}")
            print(f"📊 Reranking applied: {result.get('reranking_applied', False)}")
            print(f"💬 Answer: {result['answer'][:200]}...")
            
            # Show source details if available
            if result['sources'] and isinstance(result['sources'][0], dict):
                print("\n📋 Top sources:")
                for j, source in enumerate(result['sources'][:3], 1):
                    print(f"  {j}. {source.get('summary', 'No summary')}")
                    print(f"     Why: {source.get('why', 'No explanation')}")
                    print(f"     Score: {source.get('score', 'N/A')}")
                    print()
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50)

async def test_search_only():
    """Test search functionality only"""
    
    print("🔍 Testing Search Functionality Only")
    print("=" * 50)
    
    try:
        rag_system = MilvusPyMilvusRAGSystem()
        
        # Test dense search
        print("🔎 Testing dense search...")
        dense_results = await rag_system.search_milvus_dense("storage sheds", limit=3)
        print(f"📊 Dense search returned {len(dense_results)} results")
        
        # Test sparse search
        print("🔎 Testing sparse search...")
        sparse_results = await rag_system.search_milvus_sparse("storage sheds", limit=3)
        print(f"📊 Sparse search returned {len(sparse_results)} results")
        
        # Test hybrid search
        print("🔎 Testing hybrid search...")
        hybrid_results = await rag_system.hybrid_search("storage sheds", limit=5)
        print(f"📊 Hybrid search returned {len(hybrid_results)} results")
        
        if hybrid_results:
            print("📋 Hybrid search results:")
            for i, result in enumerate(hybrid_results, 1):
                print(f"  {i}. {result.search_type.upper()}: {result.score:.3f}")
                print(f"     {result.meeting_type} - {result.meeting_date}")
                print(f"     {result.content[:100]}...")
                print()
        
    except Exception as e:
        print(f"❌ Error in search test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 PyMilvus RAG System Test")
    print("=" * 50)
    
    # Check environment
    print("🔧 Environment Check:")
    print(f"   MILVUS_URI: {'Set' if os.getenv('MILVUS_URI') else 'Not set'}")
    print(f"   MILVUS_TOKEN: {'Set' if os.getenv('MILVUS_TOKEN') else 'Not set'}")
    print(f"   OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print()
    
    # Run tests
    asyncio.run(test_search_only())
    print("\n" + "="*50)
    asyncio.run(test_pymilvus_rag()) 