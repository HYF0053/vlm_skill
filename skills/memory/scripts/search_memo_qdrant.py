#!/usr/bin/env python3
import sys
import argparse
import os
import json
import requests

CONFIG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../config/memo.json"))

class MemoSearchService:
    def __init__(self, config):
        self.qdrant_config = config.get("qdrant", {})
        self.embedding_config = config.get("embedding", {})
        self.rerank_config = config.get("rerank", {})
        
        self.qdrant_url = f"http://{self.qdrant_config.get('host')}:{self.qdrant_config.get('port')}"
        self.embedding_url = f"{self.embedding_config.get('base_url')}/v1/embeddings"
        
        # Depending on if reranker is defined in memo.json
        if self.rerank_config:
            self.rerank_url = f"{self.rerank_config.get('base_url')}/v1/rerank"
        else:
            self.rerank_url = None

    def get_embedding(self, text):
        payload = {"model": self.embedding_config.get("model"), "input": text}
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
        
        if resp.status_code == 404:
            return [] # Collection does not exist yet

        resp.raise_for_status()
        return resp.json().get("result", [])

    def rerank(self, query, documents):
        if not documents or not self.rerank_url:
            return documents
        
        docs_to_rank = documents[:20]
        doc_texts = []
        for d in docs_to_rank:
            payload = d.get("payload", {})
            # For memo payloads, content might be 'content', 'workflow_summary', or fallback
            content = payload.get("content", payload.get("workflow_summary", str(payload)))
            doc_texts.append(content[:4000]) # Truncate for safety
            
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
            
            if len(documents) > 20:
                reranked_docs.extend(documents[20:])
                
            # Re-sort by rerank_score
            reranked_docs.sort(key=lambda x: x.get("rerank_score", 0) if "rerank_score" in x else x.get("score", 0), reverse=True)
            return reranked_docs
        except Exception as e:
            print(f"Warning: Rerank failed ({e}). Returning original order.", file=sys.stderr)
            return documents

def main():
    parser = argparse.ArgumentParser(description="Search the Long-term Memory Qdrant (memo.json)")
    parser.add_argument("query", nargs='+', help="The search query string")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--collection", "-c", type=str, action="append", help="Target collections. Defaults to all in memo.json.")
    
    args = parser.parse_args()
    query_str = " ".join(args.query)
    
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config not found at {CONFIG_PATH}")
        sys.exit(1)
        
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    service = MemoSearchService(config)
    collections = args.collection or [c["name"] for c in config.get("qdrant", {}).get("collections", [])]
    
    print(f"🔍 Searching long-term memory for: '{query_str}' in {collections}")
    
    try:
        vector = service.get_embedding(query_str)
        all_results = []
        for col in collections:
            hits = service.search_qdrant(col, vector, limit=15)
            for h in hits:
                h["collection"] = col
            all_results.extend(hits)
            
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        final_results = service.rerank(query_str, all_results)[:args.limit]
        
        print(f"\n📊 Retrieved {len(final_results)} results:")
        for i, res in enumerate(final_results):
            payload = res.get("payload", {})
            score = res.get("rerank_score", res.get("score", 0))
            collection = res.get("collection", "Unknown")
            
            print(f"\n[{i+1}] 📂 Collection: {collection} (Score: {score:.4f})")
            
            # Print specific fields based on payload structure we designed
            doc_type = payload.get("doc_type", payload.get("task_intent", "Unknown"))
            print(f"    🏷️ Type/Intent: {doc_type}")
            
            if "score" in payload:
                print(f"    ⭐ Payload Score: {payload['score']}")
                
            if "reflection" in payload and payload["reflection"]:
                print(f"    🧠 Reflection: {payload['reflection']}")
                
            content = payload.get("content", payload.get("workflow_summary", "No content found"))
            print(f"    📝 Content Snippet: {content[:1000]}")
            
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Search failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
