#!/usr/bin/env python3
import sys
import argparse
import os
import json
import requests

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
        payload = {"model": self.embedding_config.get("model"), "input": text}
        resp = requests.post(self.embedding_url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def search_qdrant(self, collection, vector, limit=5):
        payload = {"vector": vector, "limit": limit, "with_payload": True}
        url = f"{self.qdrant_url}/collections/{collection}/points/search"
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json().get("result", [])

    def rerank(self, query, documents):
        if not documents or not self.rerank_config: return documents
        
        # Limit to top 20 for reranking stability
        docs_to_rank = documents[:20]
        
        doc_texts = []
        for d in docs_to_rank:
            content = d.get("payload", {}).get("content", "") or ""
            doc_texts.append(content[:8000]) # Truncate for safety
            
        payload = {"model": self.rerank_config.get("model"), "query": query, "documents": doc_texts}
        try:
            resp = requests.post(self.rerank_url, json=payload, timeout=15)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return [docs_to_rank[r["index"]] for r in results]
        except Exception:
            return documents

def main():
    parser = argparse.ArgumentParser(description="Retrieve context for LLM")
    parser.add_argument("query", nargs='+', help="The search query string")
    parser.add_argument("--max_tokens", type=int, default=2000, help="Approximate max tokens (chars/4)")
    parser.add_argument("--collection", "-c", type=str, action="append", help="Target collections")
    
    args = parser.parse_args()
    query_str = " ".join(args.query)
    
    if not os.path.exists(CONFIG_PATH):
        print("Error: Config not found")
        sys.exit(1)
        
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        
    service = SimpleRagService(config)
    collections = args.collection or config.get("qdrant", {}).get("collections", [])
    
    try:
        vector = service.get_embedding(query_str)
        all_hits = []
        for col in collections:
            all_hits.extend(service.search_qdrant(col, vector, limit=10))
            
        reranked = service.rerank(query_str, all_hits)
        
        context_parts = []
        total_chars = 0
        limit_chars = args.max_tokens * 4
        
        for hit in reranked:
            p = hit.get("payload", {})
            content = p.get("content", "")
            source = p.get("source", "Unknown")
            part = f"Source: {source}\nContent: {content}\n"
            if total_chars + len(part) > limit_chars:
                break
            context_parts.append(part)
            total_chars += len(part)
            
        print("\n".join(context_parts))
            
    except Exception as e:
        print(f"Context retrieval failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
