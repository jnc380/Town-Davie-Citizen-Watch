#!/usr/bin/env python3
"""
Fix Milvus HTTP API integration
Research and implement correct Zilliz Cloud HTTP API format
"""

import asyncio
import os
import json
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

async def test_zilliz_api_formats():
    """Test different Zilliz Cloud HTTP API formats"""
    
    print("🔧 Testing Zilliz Cloud HTTP API Formats")
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
        # Test 1: Check if collection exists using different endpoints
        print("\n🔍 Test 1: Collection existence check")
        print("-" * 30)
        
        # Try different collection list endpoints
        endpoints_to_try = [
            f"{milvus_uri}/v1/collections",
            f"{milvus_uri}/v2/vectordb/collections",
            f"{milvus_uri}/collections",
            f"{milvus_uri}/api/v1/collections"
        ]
        
        headers = {
            "Authorization": f"Bearer {milvus_token}",
            "Content-Type": "application/json"
        }
        
        for endpoint in endpoints_to_try:
            try:
                print(f"🔎 Trying: {endpoint}")
                response = await http_client.get(endpoint, headers=headers)
                print(f"📥 Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"📥 Response: {json.dumps(data, indent=2)[:500]}...")
                    break
                else:
                    print(f"📥 Error: {response.text[:200]}...")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Test 2: Try to describe the collection
        print("\n🔍 Test 2: Collection description")
        print("-" * 30)
        
        describe_endpoints = [
            f"{milvus_uri}/v1/collections/TOWN_OF_DAVIE_RAG",
            f"{milvus_uri}/v2/vectordb/collections/describe",
            f"{milvus_uri}/collections/TOWN_OF_DAVIE_RAG"
        ]
        
        for endpoint in describe_endpoints:
            try:
                print(f"🔎 Trying: {endpoint}")
                if "describe" in endpoint:
                    payload = {"collectionName": "TOWN_OF_DAVIE_RAG"}
                    response = await http_client.post(endpoint, json=payload, headers=headers)
                else:
                    response = await http_client.get(endpoint, headers=headers)
                
                print(f"📥 Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"📥 Response: {json.dumps(data, indent=2)[:500]}...")
                    break
                else:
                    print(f"📥 Error: {response.text[:200]}...")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Test 3: Generate embedding and try search
        print("\n🔍 Test 3: Vector search with different formats")
        print("-" * 30)
        
        query = "storage sheds"
        
        # Generate embedding
        embedding_response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding
        print(f"✅ Generated embedding: {len(query_embedding)} dimensions")
        
        # Try different search formats
        search_formats = [
            # Format 1: Standard Milvus format
            {
                "collection_name": "TOWN_OF_DAVIE_RAG",
                "data": [query_embedding],
                "anns_field": "embedding",
                "param": {
                    "metric_type": "COSINE",
                    "params": {"nprobe": 10}
                },
                "limit": 5,
                "output_fields": ["id", "content", "document_type", "meeting_id", "meeting_date", "meeting_type", "hierarchy", "section_header", "metadata_json"]
            },
            # Format 2: Zilliz Cloud format
            {
                "collectionName": "TOWN_OF_DAVIE_RAG",
                "data": [query_embedding],
                "annsField": "embedding",
                "param": {
                    "metricType": "COSINE",
                    "params": {"nprobe": 10}
                },
                "limit": 5,
                "outputFields": ["id", "content", "document_type", "meeting_id", "meeting_date", "meeting_type", "hierarchy", "section_header", "metadata_json"]
            },
            # Format 3: Alternative field names
            {
                "collection_name": "TOWN_OF_DAVIE_RAG",
                "data": [query_embedding],
                "anns_field": "embedding",
                "param": {
                    "metric_type": "COSINE",
                    "params": {"nprobe": 10}
                },
                "limit": 5,
                "output_fields": ["id", "content"]
            }
        ]
        
        search_endpoints = [
            f"{milvus_uri}/v1/vector/search",
            f"{milvus_uri}/v2/vectordb/collections/search",
            f"{milvus_uri}/vector/search"
        ]
        
        for i, (endpoint, search_data) in enumerate(zip(search_endpoints, search_formats)):
            try:
                print(f"\n🔎 Search format {i+1}: {endpoint}")
                print(f"📤 Search data: {json.dumps(search_data, indent=2)}")
                
                response = await http_client.post(endpoint, json=search_data, headers=headers)
                print(f"📥 Status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"📥 Response: {json.dumps(data, indent=2)[:500]}...")
                    
                    if "results" in data:
                        print(f"✅ Found {len(data['results'])} results!")
                        for j, hit in enumerate(data["results"][:2]):
                            print(f"  {j+1}. ID: {hit.get('id')}")
                            print(f"     Score: {hit.get('score')}")
                            print(f"     Content: {hit.get('content', '')[:100]}...")
                        break
                    else:
                        print(f"❌ No 'results' field in response")
                else:
                    print(f"📥 Error: {response.text[:200]}...")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await http_client.aclose()

if __name__ == "__main__":
    asyncio.run(test_zilliz_api_formats()) 