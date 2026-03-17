#!/usr/bin/env python3
import sys
import argparse
import os
import json
import requests

# CONFIG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../config/rag.json"))
# Since current project has config in root, let's look there first
CONFIG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../config/rag.json"))

class SimpleRagService:
    def __init__(self, config):
        self.qdrant_config = config.get("qdrant", {})
        self.embedding_config = config.get("embedding", {})
        self.rerank_config = config.get("rerank", {})
        
        self.qdrant_url = f"http://{self.qdrant_config.get('host')}:{self.qdrant_config.get('port')}"
        self.embedding_url = f"{self.embedding_config.get('base_url')}/v1/embeddings"
        self.rerank_url = f"{self.rerank_config.get('base_url')}/v1/rerank"

    def get_embedding(self, text):
        payload = {
            "model": self.embedding_config.get("model"),
            "input": text
        }
        resp = requests.post(self.embedding_url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def search_qdrant(self, collection, vector, limit=5):
        payload = {
            "vector": vector,
            "limit": limit,
            "with_payload": True,
            "with_vector": False
        }
        url = f"{self.qdrant_url}/collections/{collection}/points/search"
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json().get("result", [])

    def rerank(self, query, documents):
        if not documents or not self.rerank_config:
            return documents
        
        # Limit to top 20 for reranking stability
        docs_to_rank = documents[:20]
        
        # Prepare content, ensuring it's a string and not too long for the reranker
        doc_texts = []
        for d in docs_to_rank:
            content = d.get("payload", {}).get("content", "") or str(d.get("payload", ""))
            # Truncate very long texts for reranker (approx 2000 tokens / 8000 chars)
            doc_texts.append(content[:8000])
            
        payload = {
            "model": self.rerank_config.get("model"),
            "query": query,
            "documents": doc_texts
        }
        
        try:
            resp = requests.post(self.rerank_url, json=payload, timeout=15)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            
            reranked_docs = []
            for r in results:
                idx = r["index"]
                doc = docs_to_rank[idx]
                doc["rerank_score"] = r["relevance_score"]
                reranked_docs.append(doc)
            
            # Add back items that weren't sent to reranker (at the end)
            if len(documents) > 20:
                reranked_docs.extend(documents[20:])
                
            return reranked_docs
        except Exception as e:
            print(f"Warning: Rerank failed ({e}). Returning original search order.", file=sys.stderr)
            return documents

def main():
    parser = argparse.ArgumentParser(description="Search the RAG Vector Database (Self-Contained)")
    parser.add_argument("query", nargs='+', help="The search query string")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--collection", "-c", type=str, action="append", help="Target collections (can specify multiple). If omitted, searches all.")
    
    args = parser.parse_args()
    query_str = " ".join(args.query)
    
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config not found at {CONFIG_PATH}")
        sys.exit(1)
        
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        
    service = SimpleRagService(config)
    collections = args.collection or config.get("qdrant", {}).get("collections", [])
    
    print(f"Searching for: '{query_str}' in {collections}")
    
    try:
        vector = service.get_embedding(query_str)
        all_results = []
        for col in collections:
            hits = service.search_qdrant(col, vector, limit=20) # Get more hits initially
            for h in hits:
                h["collection"] = col
            all_results.extend(hits)
            
        # Global sort by initial score
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Apply Rerank
        final_results = service.rerank(query_str, all_results)
        final_results = final_results[:args.limit]
        
        print(f"\nRetrieved {len(final_results)} results:")
        for i, res in enumerate(final_results):
            payload = res.get("payload", {})
            score = res.get("rerank_score", res.get("score", 0))
            source = payload.get("source", "Unknown")
            collection = res.get("collection", "Unknown")
            
            print(f"\n[{i+1}] Collection: {collection} | Source: {source} (Score: {score:.4f})")
            
            # Clean up metadata display
            meta_cleaned = {k: v for k, v in payload.items() if k not in ['content', 'source']}
            if meta_cleaned:
                print(f"    Metadata: {meta_cleaned}")
            
            content = payload.get("content", "No content")
            content_lines = str(content).split('\n')
            for line in content_lines:
                print(f"    {line}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Search failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
