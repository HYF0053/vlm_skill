#!/usr/bin/env python3
"""
mem_prune_qdrant.py — Delete old, low-importance Qdrant points.

Handles both conv_YYYY_MM collections and document_memory.
Only cleans Qdrant — JSON files and original documents are NEVER touched.

Usage:
    python mem_prune_qdrant.py \\
        [--days 30] \\
        [--max-score 4.0] \\
        [--collection conv_2026_01|document_memory|all]
        # 'all' = every conv_* collection + document_memory
"""
import sys
import os
import json
import argparse
import time
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH  = os.path.join(PROJECT_ROOT, "config", "memo.json")

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def qdrant_base(cfg: dict) -> str:
    q = cfg["qdrant"]
    return f"http://{q['host']}:{q['port']}"

def list_all_collections(cfg: dict) -> list[str]:
    resp = requests.get(f"{qdrant_base(cfg)}/collections", timeout=5)
    resp.raise_for_status()
    return [c["name"] for c in resp.json().get("result", {}).get("collections", [])]

def list_conv_collections(cfg: dict) -> list[str]:
    prefix = cfg["qdrant"].get("conversation_collection_prefix", "conv_")
    return [n for n in list_all_collections(cfg) if n.startswith(prefix)]

def prune_collection(collection: str, cutoff_ts: int, max_score: float, cfg: dict) -> None:
    base = qdrant_base(cfg)
    url  = f"{base}/collections/{collection}/points/delete"

    filter_payload = {
        "filter": {
            "must": [
                {"key": "timestamp", "range": {"lt": cutoff_ts}},
                {"key": "importance_score", "range": {"lt": max_score}},
            ]
        }
    }

    resp = requests.post(url, json=filter_payload, timeout=15)
    if resp.status_code == 404:
        print(f"   ⚠️  Collection '{collection}' not found. Skipping.")
        return
    resp.raise_for_status()
    result = resp.json()
    if result.get("status") == "ok":
        print(f"   ✅ '{collection}': pruned (timestamp < {time.strftime('%Y-%m-%d', time.localtime(cutoff_ts))}, score < {max_score})")
    else:
        print(f"   ⚠️  '{collection}': unexpected response: {result}")

def main():
    parser = argparse.ArgumentParser(description="Prune old/low-score points from Qdrant")
    parser.add_argument("--days",      type=int,   default=30,  help="Delete points older than N days (default: 30)")
    parser.add_argument("--max-score", type=float, default=4.0, help="Only delete if importance_score < this (default: 4.0)")
    parser.add_argument("--collection", "-c", default="all",
                        help="Target: 'all' (default), 'document_memory', or a specific collection name like 'conv_2026_01'")
    args = parser.parse_args()

    cfg = load_config()

    now_ts     = int(time.time())
    cutoff_ts  = now_ts - args.days * 86400
    cutoff_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cutoff_ts))

    print(f"🧹 Pruning Qdrant memory")
    print(f"   Cutoff  : older than {args.days} days ({cutoff_str})")
    print(f"   Max score: < {args.max_score}")
    print(f"   Target  : {args.collection}\n")

    # Resolve target collections
    if args.collection == "all":
        conv_cols = list_conv_collections(cfg)
        collections = conv_cols + ["document_memory"]
        if not collections:
            print("ℹ️  No collections found.")
            return
        print(f"   Found collections: {collections}\n")
    else:
        collections = [args.collection]

    for col in collections:
        print(f"Processing '{col}' ...")
        try:
            prune_collection(col, cutoff_ts, args.max_score, cfg)
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    print("\n✅ Pruning complete. JSON files and original documents are untouched.")

if __name__ == "__main__":
    main()
