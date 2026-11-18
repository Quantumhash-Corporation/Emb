"""
BGE API Test Script
Run: python tests/test_api.py
Make sure server is running: bash start.sh
"""

import httpx
import json
import time

BASE_URL = "http://localhost:8100"
API_KEY = "qh-key-change-this-in-production"  # .env se match karna chahiye

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}


def print_result(name: str, response: httpx.Response):
    status = "✅" if response.status_code == 200 else "❌"
    print(f"\n{status} [{name}] Status: {response.status_code}")
    data = response.json()
    if "embeddings" in data:
        emb = data["embeddings"][0]
        print(f"   Embedding dim: {len(emb)}")
        print(f"   First 5 values: {emb[:5]}")
        print(f"   Processing time: {data['processing_time_ms']}ms")
    elif "scores" in data:
        print(f"   Scores: {data['scores']}")
        print(f"   Processing time: {data['processing_time_ms']}ms")
    elif "results" in data:
        for r in data["results"][:3]:
            print(f"   Score {r['score']:.4f} → {r['document'][:60]}...")
    else:
        print(f"   Response: {json.dumps(data, indent=2)[:300]}")


def main():
    print("═══════════════════════════════════════")
    print("  BGE API Test Suite")
    print("═══════════════════════════════════════")

    client = httpx.Client(timeout=60.0)

    # 1. Health Check
    r = client.get(f"{BASE_URL}/health")
    print_result("Health Check", r)

    # 2. Single text embedding
    r = client.post(
        f"{BASE_URL}/v1/embeddings",
        headers=HEADERS,
        json={"texts": "Machine learning is amazing!", "is_query": False},
    )
    print_result("Single Text Embed", r)

    # 3. Batch embedding (documents)
    r = client.post(
        f"{BASE_URL}/v1/embeddings",
        headers=HEADERS,
        json={
            "texts": [
                "Python is a great programming language.",
                "FastAPI makes building APIs easy.",
                "BGE models are state-of-the-art embeddings.",
            ],
            "is_query": False,
            "normalize": True,
        },
    )
    print_result("Batch Documents Embed", r)

    # 4. Query embedding (instruction add hoti hai)
    r = client.post(
        f"{BASE_URL}/v1/embeddings",
        headers=HEADERS,
        json={
            "texts": ["What is RAG?", "How to use vector database?"],
            "is_query": True,
        },
    )
    print_result("Query Embeddings", r)

    # 5. Similarity
    r = client.post(
        f"{BASE_URL}/v1/similarity",
        headers=HEADERS,
        json={
            "queries": ["What is deep learning?"],
            "passages": [
                "Deep learning uses neural networks with many layers.",
                "Python is a programming language.",
                "Neural networks are inspired by the human brain.",
            ],
        },
    )
    print_result("Similarity Scores", r)

    # 6. Reranking
    r = client.post(
        f"{BASE_URL}/v1/rerank",
        headers=HEADERS,
        params={
            "query": "How does Qdrant vector database work?",
            "top_k": 3,
        },
        json=[
            "Qdrant is a vector similarity search engine.",
            "Pizza is delicious food.",
            "Qdrant supports dense and sparse vectors with payload filtering.",
            "Machine learning models generate embeddings.",
            "Qdrant has a REST API and Python client.",
        ],
    )
    print_result("Reranking", r)

    # 7. Auth test — wrong key
    r = client.post(
        f"{BASE_URL}/v1/embeddings",
        headers={"X-API-Key": "wrong-key", "Content-Type": "application/json"},
        json={"texts": "test"},
    )
    status = "✅" if r.status_code == 401 else "❌"
    print(f"\n{status} [Auth Test] Status: {r.status_code} (expected 401)")

    print("\n═══════════════════════════════════════")
    print("  Tests Complete!")
    print("═══════════════════════════════════════")
    client.close()


if __name__ == "__main__":
    main()
