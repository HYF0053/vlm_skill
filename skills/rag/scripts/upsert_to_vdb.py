#!/usr/bin/env python3
import sys
import argparse
import os
import json
import requests
import uuid
from datetime import datetime

CONFIG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../config/rag.json"))

class SimpleRagService:
    def __init__(self, config):
        self.qdrant_config = config.get("qdrant", {})
        self.embedding_config = config.get("embedding", {})
        self.qdrant_url = f"http://{self.qdrant_config.get('host')}:{self.qdrant_config.get('port')}"
        self.embedding_url = f"{self.embedding_config.get('base_url')}/v1/embeddings"

    def get_embedding(self, text):
        payload = {"model": self.embedding_config.get("model"), "input": text}
        resp = requests.post(self.embedding_url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def upsert_point(self, collection, vector, payload):
        point_id = str(uuid.uuid4())
        data = {
            "points": [
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": payload
                }
            ]
        }
        url = f"{self.qdrant_url}/collections/{collection}/points?wait=true"
        resp = requests.put(url, json=data, timeout=10)
        resp.raise_for_status()
        return point_id

def main():
    parser = argparse.ArgumentParser(description="Upsert data to RAG Vector Database")
    parser.add_argument("content", help="The content to store")
    parser.add_argument("--collection", "-c", default="agent_long_memory", help="Target collection")
    parser.add_argument("--source", "-s", default="conversation", help="Source metadata")
    parser.add_argument("--metadata", "-m", type=str, help="Additional metadata as JSON string")
    
    args = parser.parse_args()
    
    if not os.path.exists(CONFIG_PATH):
        print("Error: Config not found")
        sys.exit(1)
        
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        
    service = SimpleRagService(config)
    
    try:
        meta = json.loads(args.metadata) if args.metadata else {}
        meta["content"] = args.content
        meta["source"] = args.source
        meta["timestamp"] = datetime.now().isoformat()
        
        vector = service.get_embedding(args.content)
        point_id = service.upsert_point(args.collection, vector, meta)
        
        print(f"Successfully upserted point to {args.collection}. ID: {point_id}")
    except Exception as e:
        print(f"Upsert failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
