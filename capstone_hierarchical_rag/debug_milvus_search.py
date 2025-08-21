#!/usr/bin/env python3
"""
Debug script to test Milvus HTTP API directly
"""

import asyncio
import os
import json
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

async def test_milvus_http_api():
    """Test Milvus HTTP API directly"""
    
    print("🔍 Testing Milvus HTTP API Directly")
    print("=" * 50)
    
    # Get environment variables
    milvus_uri = os.getenv("MILVUS_URI")
    milvus_token = os.getenv("MILVUS_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not all([milvus_uri, milvus_token, openai_api_key]):
        print("❌ Missing environment variables")
        return
    
    print(f"📊 Milvus URI: {milvus_uri}")
    print(f"📊 Collection: TOWN_OF_DAVIE_RAG")
    
    # Initialize clients
    http_client = httpx.AsyncClient(timeout=30.0)
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    
    try:
        # Test 1: Simple query
        query = "storage sheds"
        
        # Generate embedding
        print(f"🔎 Testing query: '{query}'")
        embedding_response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding
        print(f"✅ Generated embedding: {len(query_embedding)} dimensions")
        
        # Test dense search
        print("\n🔎 Testing dense search...")
        search_data = {
            "collection_name": "TOWN_OF_DAVIE_RAG",
            "data": [query_embedding],
            "anns_field": "embedding",
            "param": {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            },
            "limit": 5,
            "output_fields": [
                "id", "content", "document_type", "meeting_id", 
                "meeting_date", "meeting_type", "hierarchy", 
                "section_header", "metadata_json"
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {milvus_token}",
            "Content-Type": "application/json"
        }
        
        print(f"📤 Sending request to: {milvus_uri}/v1/vector/search")
        print(f"📤 Search data: {json.dumps(search_data, indent=2)}")
        
        response = await http_client.post(
            f"{milvus_uri}/v1/vector/search",
            json=search_data,
            headers=headers
        )
        
        print(f"📥 Response status: {response.status_code}")
        print(f"📥 Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"📥 Response body: {json.dumps(results, indent=2)}")
            
            if "results" in results:
                print(f"✅ Found {len(results['results'])} results")
                for i, hit in enumerate(results["results"][:3]):
                    print(f"  {i+1}. ID: {hit.get('id')}")
                    print(f"     Score: {hit.get('score')}")
                    print(f"     Content: {hit.get('content', '')[:100]}...")
                    print()
            else:
                print("❌ No 'results' field in response")
        else:
            print(f"❌ Error response: {response.text}")
        
        # Test 2: Try a different approach - list collections
        print("\n🔎 Testing collection list...")
        list_response = await http_client.get(
            f"{milvus_uri}/v1/collections",
            headers=headers
        )
        
        print(f"📥 List response status: {list_response.status_code}")
        if list_response.status_code == 200:
            collections = list_response.json()
            print(f"📥 Collections: {json.dumps(collections, indent=2)}")
        
        # Test 3: Try describe collection
        print("\n🔎 Testing collection describe...")
        describe_response = await http_client.get(
            f"{milvus_uri}/v1/collections/TOWN_OF_DAVIE_RAG",
            headers=headers
        )
        
        print(f"📥 Describe response status: {describe_response.status_code}")
        if describe_response.status_code == 200:
            collection_info = describe_response.json()
            print(f"📥 Collection info: {json.dumps(collection_info, indent=2)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await http_client.aclose()

if __name__ == "__main__":
    asyncio.run(test_milvus_http_api()) 