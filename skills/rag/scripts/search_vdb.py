#!/usr/bin/env python3
import sys
import argparse
import os

# Ensure the ai-agent-platform is in the Python path
sys.path.insert(0, "/home/ubuntu/ai-agent-platform")
os.chdir("/home/ubuntu/ai-agent-platform")

from infrastructure.di_container import get_container
from infrastructure.config.settings import settings
import json

CONFIG_PATH = "/home/ubuntu/vlm_skill/config/rag.json"

def apply_custom_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            
            if "qdrant" in config:
                settings.QDRANT_HOST = config["qdrant"].get("host", settings.QDRANT_HOST)
                settings.QDRANT_PORT = config["qdrant"].get("port", settings.QDRANT_PORT)
            if "embedding" in config:
                settings.EMBEDDING_BASE_URL = config["embedding"].get("base_url", settings.EMBEDDING_BASE_URL)
                settings.EMBEDDING_MODEL = config["embedding"].get("model", settings.EMBEDDING_MODEL)
                settings.EMBEDDING_BACKEND = config["embedding"].get("backend", settings.EMBEDDING_BACKEND)
            if "rerank" in config:
                settings.RERANK_BASE_URL = config["rerank"].get("base_url", settings.RERANK_BASE_URL)
                settings.RERANK_MODEL = config["rerank"].get("model", settings.RERANK_MODEL)
                settings.RERANK_BACKEND = config["rerank"].get("backend", settings.RERANK_BACKEND)
        except Exception as e:
            print(f"Failed to load custom RAG config: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Search the RAG Vector Database")
    parser.add_argument("query", nargs='+', help="The search query string")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--collection", "-c", type=str, action="append", help="Target collections (can specify multiple). If omitted, searches all.")
    
    args = parser.parse_args()
    
    query_str = " ".join(args.query)
    
    try:
        apply_custom_config()
        container = get_container()
        rag_service = container.get_rag_service()
    except Exception as e:
        print(f"Failed to initialize RAG Service: {e}")
        sys.exit(1)
        
    collections = args.collection if args.collection else []
    
    print(f"Searching for: '{query_str}'")
    if collections:
        print(f"Targeting collections: {collections}")
    
    try:
        results = rag_service._search_multiple(query_str, limit=args.limit, active_collections=collections)
    except Exception as e:
        print(f"Search failed: {e}")
        sys.exit(1)
        
    print(f"\nRetrieved {len(results)} results:")
    for i, res in enumerate(results):
        score_str = f"{res.score:.4f}" if hasattr(res, 'score') else "N/A"
        print(f"\n[{i+1}] Source: {res.source} (Score: {score_str})")
        if hasattr(res, 'metadata') and isinstance(res.metadata, dict):
            # Clean up metadata display
            meta_cleaned = {k: v for k, v in res.metadata.items() if k not in ['content', 'source']}
            if meta_cleaned:
                print(f"    Metadata: {meta_cleaned}")
        
        # Display content with indentation
        content_lines = str(res.content).split('\n')
        for line in content_lines:
            print(f"    {line}")
        print("-" * 60)

if __name__ == "__main__":
    main()
