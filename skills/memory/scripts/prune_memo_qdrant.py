#!/usr/bin/env python3
import sys
import argparse
import os
import json
import requests
import time

CONFIG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../config/memo.json"))

def main():
    parser = argparse.ArgumentParser(description="Prune old/low-score points from Long-term Memory Qdrant (memo.json)")
    parser.add_argument("--days", type=int, default=30, help="Delete points older than this many days (default: 30)")
    parser.add_argument("--max-score", type=float, default=5.9, help="Only delete if score is strictly less than this (default: 5.9)")
    parser.add_argument("--collection", "-c", type=str, action="append", help="Target collections. Defaults to all in memo.json.")
    
    args = parser.parse_args()
    
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config not found at {CONFIG_PATH}")
        sys.exit(1)
        
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    qdrant_config = config.get("qdrant", {})
    host = qdrant_config.get("host", "10.1.1.7")
    port = qdrant_config.get("port", 26620)
    qdrant_url = f"http://{host}:{port}"
    
    collections = args.collection or [c["name"] for c in qdrant_config.get("collections", [])]
    
    now_ts = int(time.time())
    cutoff_ts = now_ts - (args.days * 86400)
    
    print(f"🧹 Pruning memory from collections: {collections}")
    print(f"📅 Cutoff Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cutoff_ts))}")
    print(f"⭐ Score Policy: Delete only if score < {args.max_score}")
    
    payload = {
        "filter": {
            "must": [
                {
                    "key": "timestamp",
                    "range": {
                        "lt": cutoff_ts
                    }
                },
                {
                    "key": "score",
                    "range": {
                        "lt": args.max_score
                    }
                }
            ]
        }
    }

    for col in collections:
        print(f"\nProcessing collection: {col} ...")
        url = f"{qdrant_url}/collections/{col}/points/delete"
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 404:
                print(f"⚠️ Collection '{col}' does not exist yet. Skipping.")
                continue
            
            resp.raise_for_status()
            res_data = resp.json()
            if res_data.get("status") == "ok":
                print(f"✅ Successfully submitted deletion request for '{col}'.")
            else:
                print(f"❌ Deletion warning for '{col}': {res_data}")
        except Exception as e:
            print(f"❌ Failed to reach Qdrant for '{col}': {e}")

if __name__ == "__main__":
    main()
