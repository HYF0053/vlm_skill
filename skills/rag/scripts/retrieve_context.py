#!/usr/bin/env python3
import sys
import os
import argparse

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
    parser = argparse.ArgumentParser(description="Retrieve RAG Context formatted for LLM")
    parser.add_argument("query", nargs='+', help="The search query string")
    parser.add_argument("--limit", type=int, default=10, help="Number of documents to retrieve initially")
    parser.add_argument("--max_tokens", type=int, default=6000, help="Maximum tokens for the combined context")
    parser.add_argument("--collection", "-c", type=str, action="append", help="Target collections. If omitted, searches all.")
    
    args = parser.parse_args()
    
    query_str = " ".join(args.query)
    
    try:
        apply_custom_config()
        container = get_container()
        rag_service = container.get_rag_service()
    except Exception as e:
        print(f"Failed to initialize RAG Service: {e}", file=sys.stderr)
        sys.exit(1)
        
    collections = args.collection if args.collection else []
    
    try:
        context_str = rag_service.retrieve(
            query=query_str,
            limit=args.limit,
            max_total_tokens=args.max_tokens,
            active_collections=collections
        )
    except Exception as e:
        print(f"Retrieval failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(context_str)

if __name__ == "__main__":
    main()
