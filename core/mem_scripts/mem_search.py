#!/usr/bin/env python3
"""
mem_search.py — Unified semantic search across ALL memory:
    - conv_YYYY_MM collections  (conversation turns)
    - document_memory           (indexed files from results/ and tmp/)

Full content is always read from the original source (thread JSON or file).

Usage:
    python mem_search.py "<query>" [--limit 5] [--type all|conv|doc]
                                   [--months 2026-03,2026-02]
                                   [--min-importance 0.0]
"""
import sys
import os
import json
import argparse
import requests
from datetime import datetime

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH   = os.path.join(PROJECT_ROOT, "config", "memo.json")
THREADS_DIR   = os.path.join(PROJECT_ROOT, "data", "memory", "threads")
ARCHIVE_DIR   = os.path.join(PROJECT_ROOT, "data", "memory", "archived")

# ── helpers ──────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def qdrant_base(cfg: dict) -> str:
    q = cfg["qdrant"]
    return f"http://{q['host']}:{q['port']}"

def get_embedding(text: str, cfg: dict) -> list:
    emb = cfg["embedding"]
    url = f"{emb['base_url']}/v1/embeddings"
    resp = requests.post(url, json={"model": emb["model"], "input": text}, timeout=15)
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]

def list_all_collections(cfg: dict) -> list[str]:
    resp = requests.get(f"{qdrant_base(cfg)}/collections", timeout=5)
    resp.raise_for_status()
    return [c["name"] for c in resp.json().get("result", {}).get("collections", [])]

def list_conv_collections(cfg: dict, months: list[str] | None = None) -> list[str]:
    prefix = cfg["qdrant"].get("conversation_collection_prefix", "conv_")
    if months:
        cols = []
        for m in months:
            try:
                dt = datetime.strptime(m.strip(), "%Y-%m")
                cols.append(dt.strftime("conv_%Y_%m"))
            except ValueError:
                pass
        return cols
    all_cols = list_all_collections(cfg)
    return sorted([n for n in all_cols if n.startswith(prefix)], reverse=True)

def search_collection(collection: str, vector: list, limit: int,
                       extra_filter: list | None, cfg: dict) -> list[dict]:
    """Search one collection with optional extra filter conditions."""
    url  = f"{qdrant_base(cfg)}/collections/{collection}/points/search"
    must = []
    if extra_filter:
        must.extend(extra_filter)

    payload = {
        "vector":       vector,
        "limit":        limit,
        "with_payload": True,
        "with_vector":  False,
    }
    if must:
        payload["filter"] = {"must": must}

    resp = requests.post(url, json=payload, timeout=10)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    hits = resp.json().get("result", [])
    for h in hits:
        h["_collection"] = collection
    return hits

def rerank(query: str, hits: list[dict], cfg: dict) -> list[dict]:
    rerank_cfg = cfg.get("rerank", {})
    if not rerank_cfg or not hits:
        return hits
    url      = f"{rerank_cfg['base_url']}/v1/rerank"
    snippets = [h.get("payload", {}).get("summary", "")[:1000] for h in hits]
    try:
        resp = requests.post(
            url,
            json={"model": rerank_cfg["model"], "query": query, "documents": snippets},
            timeout=15,
        )
        resp.raise_for_status()
        for r in resp.json().get("results", []):
            hits[r["index"]]["rerank_score"] = r["relevance_score"]
        hits.sort(key=lambda x: x.get("rerank_score", x.get("score", 0)), reverse=True)
    except Exception as e:
        print(f"⚠️  Rerank failed ({e}), using vector scores.", file=sys.stderr)
    return hits

def read_full_conversation(payload: dict) -> dict | None:
    """
    Support both payload schemas:
      New: archived_file + turn_range  (from archive_and_summarize_session)
      Old: file_path + block_index     (from mem_upsert_conversation, legacy)
    """
    # ── New schema ────────────────────────────────────────────────────────
    archived_file = payload.get("archived_file")
    turn_range    = payload.get("turn_range")          # [start_1based, end_1based]
    if archived_file and turn_range:
        abs_path = os.path.join(PROJECT_ROOT, archived_file) if not os.path.isabs(archived_file) else archived_file
        if not os.path.exists(abs_path):
            return None
        try:
            data = json.loads(open(abs_path, encoding="utf-8").read())
            msgs = data.get("recent_messages", [])
            start = (turn_range[0] - 1) * 2          # convert 1-based turn → 0-based msg index
            end   = min(turn_range[1] * 2, len(msgs))
            chunk_msgs = msgs[start:end]
            turns = []
            for i in range(0, len(chunk_msgs) - 1, 2):
                turns.append({
                    "user":      chunk_msgs[i].get("content", ""),
                    "user_ts":   chunk_msgs[i].get("ts", ""),
                    "assistant": chunk_msgs[i + 1].get("content", "") if i + 1 < len(chunk_msgs) else "",
                    "asst_ts":   chunk_msgs[i + 1].get("ts", "") if i + 1 < len(chunk_msgs) else "",
                })
            return {"turns": turns, "schema": "new"}
        except Exception:
            return None

    # ── Legacy schema ─────────────────────────────────────────────────────
    file_path   = payload.get("file_path", "")
    block_index = payload.get("block_index", 0)
    abs_path = os.path.join(PROJECT_ROOT, file_path) if not os.path.isabs(file_path) else file_path
    if not os.path.exists(abs_path):
        return None
    try:
        data = json.loads(open(abs_path, encoding="utf-8").read())
        msgs = data.get("recent_messages", [])
        user_idx = block_index * 2
        asst_idx = user_idx + 1
        result: dict = {"schema": "legacy"}
        if user_idx < len(msgs):
            result["user"]    = msgs[user_idx].get("content", "")
            result["user_ts"] = msgs[user_idx].get("ts", "")
        if asst_idx < len(msgs):
            result["assistant"]    = msgs[asst_idx].get("content", "")
            result["assistant_ts"] = msgs[asst_idx].get("ts", "")
        return result
    except Exception:
        return None

def read_file_content(file_path: str, max_chars: int = 2000) -> str | None:
    abs_path = os.path.join(PROJECT_ROOT, file_path) if not os.path.isabs(file_path) else file_path
    if not os.path.exists(abs_path):
        return None
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars)
        if len(content) == max_chars:
            content += "\n…[truncated]"
        return content
    except Exception:
        return None

def print_hit(i: int, hit: dict, show_full: bool = False) -> None:
    p          = hit.get("payload", {})
    vec_score  = hit.get("score", 0)
    rer_score  = hit.get("rerank_score")
    collection = hit.get("_collection", "?")
    score_str  = f"{rer_score:.4f} (rerank)" if rer_score is not None else f"{vec_score:.4f} (vector)"

    is_conv = collection.startswith("conv_")
    kind    = "💬 Conv" if is_conv else "📄 Doc"

    display_time = (
        p.get("chunk_turn_start_ts") or p.get("indexed_at_iso") or p.get("iso_time") or ""
    )[:19]

    print(f"[{i}] {kind}  score={score_str}  time={display_time}")

    if is_conv:
        archived_file = p.get("archived_file")
        turn_range    = p.get("turn_range")
        chunk_idx     = p.get("chunk_index", p.get("block_index", "?"))
        print(f"    session={p.get('session_key')}  chunk={chunk_idx}  turns={turn_range}")
        print(f"    summary: {p.get('summary', '')}")
        if archived_file:
            print(f"    → read full: python core/mem_scripts/mem_search.py read_chunk '{archived_file}' {turn_range}")

        if show_full:
            full = read_full_conversation(p)
            if full:
                schema = full.get("schema", "legacy")
                if schema == "new":
                    for ti, turn in enumerate(full["turns"], 1):
                        u = turn.get("user", "")
                        a = turn.get("assistant", "")
                        print(f"\n    [Turn {ti}]")
                        print(f"    👤 {u[:600]}" + ("…" if len(u) > 600 else ""))
                        print(f"    🤖 {a[:600]}" + ("…" if len(a) > 600 else ""))
                else:
                    u = full.get("user", "")
                    a = full.get("assistant", "")
                    print(f"\n    👤 {u[:600]}" + ("…" if len(u) > 600 else ""))
                    print(f"    🤖 {a[:600]}" + ("…" if len(a) > 600 else ""))
            else:
                print(f"    ⚠️  Source file not found")
    else:
        print(f"    file={p.get('file_path')}  type={p.get('file_type')}")
        if p.get("is_dir"):
            print(f"    📁 Directory: {p.get('summary', '')}")
        else:
            print(f"    summary: {p.get('summary', '')}")
        if show_full:
            content = read_file_content(p.get("file_path", ""))
            if content:
                print(f"\n    📝 Content:\n    " + content.replace("\n", "\n    "))
            else:
                print(f"    ⚠️  File not found: {p.get('file_path')}")
    print("-" * 60)


def cmd_read_chunk(archived_file: str, turn_range_str: str) -> None:
    """
    Read and print the full conversation for a specific chunk.
    Usage: python mem_search.py read_chunk 'data/memory/archived/xxx.json' '[1,5]'
    """
    import re as _re
    # parse turn_range: e.g. "[1, 5]" or "1,5"
    nums = list(map(int, _re.findall(r"\d+", turn_range_str)))
    if len(nums) < 2:
        print(f"❌ Invalid turn_range: {turn_range_str}")
        return
    payload = {"archived_file": archived_file, "turn_range": nums[:2]}
    full = read_full_conversation(payload)
    if not full:
        print(f"❌ Could not read: {archived_file}")
        return
    print(f"📖 {archived_file}  turns={nums[:2]}\n")
    for ti, turn in enumerate(full.get("turns", []), 1):
        u = turn.get("user", "")
        a = turn.get("assistant", "")
        print(f"[Turn {ti}]")
        print(f"👤 {u}")
        print(f"🤖 {a}")
        print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Sub-command: read_chunk — fetch full turns from an archived file
    if len(sys.argv) >= 4 and sys.argv[1] == "read_chunk":
        cmd_read_chunk(sys.argv[2], sys.argv[3])
        return

    parser = argparse.ArgumentParser(
        description="Unified memory search: conversations (conv_*) + documents (document_memory)")
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("--limit",          type=int,   default=5,
                        help="Total results to return (default 5)")
    parser.add_argument("--type",           default="all",
                        choices=["all", "conv", "doc"],
                        help="Search scope: all / conv / doc")
    parser.add_argument("--months",         default=None,
                        help="YYYY-MM comma list to restrict conv search")
    parser.add_argument("--min-importance", type=float, default=0.0,
                        help="Minimum importance_score filter")
    parser.add_argument("--full",           action="store_true",
                        help="Also print full conversation/file content (default: summary only)")
    args = parser.parse_args()

    query_str = " ".join(args.query)
    cfg       = load_config()

    print(f"🔍 '{query_str}'  type={args.type}  limit={args.limit}")

    try:
        vector = get_embedding(query_str, cfg)
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        sys.exit(1)

    importance_filter = []
    if args.min_importance > 0:
        importance_filter.append({"key": "importance_score", "range": {"gte": args.min_importance}})

    all_hits: list[dict] = []
    fetch_per_col = max(args.limit * 2, 10)

    if args.type in ("all", "conv"):
        months_list = [m.strip() for m in args.months.split(",")] if args.months else None
        conv_cols = list_conv_collections(cfg, months_list)
        if conv_cols:
            conv_filter = [{"key": "is_deleted", "match": {"value": False}}] + importance_filter
            for col in conv_cols:
                hits = search_collection(col, vector, fetch_per_col, conv_filter, cfg)
                all_hits.extend(hits)
        elif args.type == "conv":
            print("ℹ️  No conv_* collections found.")

    if args.type in ("all", "doc"):
        hits = search_collection("document_memory", vector, fetch_per_col,
                                 importance_filter or None, cfg)
        all_hits.extend(hits)

    if not all_hits:
        print("ℹ️  No results found.")
        return

    all_hits.sort(key=lambda x: x.get("score", 0), reverse=True)
    all_hits = rerank(query_str, all_hits, cfg)[:args.limit]

    print(f"\n📊 {len(all_hits)} results:\n")
    for i, hit in enumerate(all_hits, 1):
        print_hit(i, hit, show_full=args.full)

if __name__ == "__main__":
    main()
