#!/usr/bin/env python3
import sys
import argparse
import os
import json
import requests
import time
import uuid
from datetime import datetime

CONFIG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../config/memo.json"))

class MemoQdrantService:
    def __init__(self, config):
        self.config = config
        self.qdrant_config = config.get("qdrant", {})
        self.embedding_config = config.get("embedding", {})
        self.qdrant_url = f"http://{self.qdrant_config.get('host')}:{self.qdrant_config.get('port')}"
        self.embedding_url = f"{self.embedding_config.get('base_url')}/v1/embeddings"
        self.collections_config = {c["name"]: c for c in self.qdrant_config.get("collections", [])}

    def ensure_collection_exists(self, collection_name):
        """Check if collection exists, if not, create it based on memo.json config."""
        check_url = f"{self.qdrant_url}/collections/{collection_name}"
        resp = requests.get(check_url, timeout=5)
        
        if resp.status_code == 200:
            return # Already exists
            
        print(f"Collection '{collection_name}' not found. Attempting to create it...")
        col_def = self.collections_config.get(collection_name)
        if not col_def:
            # Fallback default if not explicitly defined in memo.json
            vector_size = 1024
            distance = "Cosine"
        else:
            vector_size = col_def.get("vector_size", 1024)
            distance = col_def.get("distance", "Cosine")

        create_url = f"{self.qdrant_url}/collections/{collection_name}"
        payload = {
            "vectors": {
                "size": vector_size,
                "distance": distance
            }
        }
        create_resp = requests.put(create_url, json=payload, timeout=10)
        create_resp.raise_for_status()
        print(f"✅ Collection '{collection_name}' created successfully.")

    def get_embedding(self, text):
        payload = {"model": self.embedding_config.get("model"), "input": text}
        resp = requests.post(self.embedding_url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def upsert_point(self, collection, vector, payload_data):
        point_id = str(uuid.uuid4())
        data = {
            "points": [
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": payload_data
                }
            ]
        }
        url = f"{self.qdrant_url}/collections/{collection}/points?wait=true"
        resp = requests.put(url, json=data, timeout=10)
        resp.raise_for_status()
        return point_id


def main():
    parser = argparse.ArgumentParser(description="Upsert data to Long-term Memory Qdrant (memo.json)")
    parser.add_argument("--collection", "-c", required=True, choices=["document_memory"], help="Target collection")
    parser.add_argument("--score", type=float, default=5.0, help="Confidence/Importance score (1-10 or 1-100)")
    parser.add_argument("--intent_or_type", "-i", type=str, required=True, help="Task intent (for workflow) or Document type (for document)")
    parser.add_argument("--content", "-t", type=str, required=True, help="Main workflow summary or document content")
    parser.add_argument("--source", "-s", type=str, default="user_session", help="Source metadata")
    
    args = parser.parse_args()
    
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config not found at {CONFIG_PATH}")
        sys.exit(1)
        
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    service = MemoQdrantService(config)
    
    try:
        # 1. Ensure collection exists
        service.ensure_collection_exists(args.collection)
        
        # 2. Prepare Payload
        now_ts = int(time.time())
        iso_time = datetime.now().isoformat()
        
        payload = {
            "timestamp": now_ts,
            "iso_time": iso_time,
            "score": args.score,
            "source": args.source
        }
        
        text_to_embed = ""
        
        # document_memory
        payload["doc_type"] = args.intent_or_type
        payload["content"] = args.content
        # Embed content
        text_to_embed = f"[{args.intent_or_type}] {args.content}"
            
        print(f"Embedding text ({len(text_to_embed)} chars)...")
        vector = service.get_embedding(text_to_embed)
        
        print(f"Upserting point...")
        point_id = service.upsert_point(args.collection, vector, payload)
        
        print(f"✅ Successfully upserted point to {args.collection}. ID: {point_id}")
    except Exception as e:
        print(f"❌ Upsert failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
