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
    # Use the most informative text available (summary or full_summary)
    snippets = [
        (h.get("payload", {}).get("summary")
         or h.get("payload", {}).get("full_summary", ""))[:1000]
        for h in hits
    ]
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


# ── Two-pass Top-K search with time-decay ─────────────────────────────────────────

def fetch_point_by_id(point_id: str, collection: str, cfg: dict) -> dict | None:
    """
    Fetch a single Qdrant point by its UUID. Returns a hit-like dict or None.
    Used to pull adjacent chunks (prev_chunk_id / next_chunk_id).
    """
    base = qdrant_base(cfg)
    try:
        resp = requests.post(
            f"{base}/collections/{collection}/points",
            json={"ids": [point_id], "with_payload": True, "with_vector": False},
            timeout=8,
        )
        if resp.status_code != 200:
            return None
        pts = resp.json().get("result", [])
        if not pts:
            return None
        pt = pts[0]
        pt["_collection"] = collection
        pt["score"] = 0.0          # no semantic score for context chunks
        pt["_is_context"] = True
        return pt
    except Exception:
        return None


def _apply_time_decay(hits: list[dict], time_weight: float) -> list[dict]:
    """
    Blend semantic score with recency score in-place.
    recency = (indexed_at_unix - t_min) / (t_max - t_min), normalised to [0, 1].
    newest chunk → recency=1.0, oldest → recency=0.0.
    """
    if time_weight <= 0 or not hits:
        for h in hits:
            h["blended_score"] = h.get("score", 0.0)
        return hits

    ts_vals = [h["payload"].get("indexed_at_unix", 0) for h in hits]
    t_min, t_max = min(ts_vals), max(ts_vals)

    for h in hits:
        sem = h.get("score", 0.0)
        ts  = h["payload"].get("indexed_at_unix", t_min)
        rec = (ts - t_min) / (t_max - t_min) if t_max > t_min else 1.0
        h["blended_score"] = (1.0 - time_weight) * sem + time_weight * rec
        h["_recency_score"] = rec
    return hits


def search_two_pass_topk(
    query_str: str,
    vector: list,
    cfg: dict,
    limit: int = 5,
    time_weight: float = 0.25,
    context_window: int = 1,
    months: list[str] | None = None,
    importance_filter: list | None = None,
) -> list[dict]:
    """
    Two-pass Top-K search with time-decay blending.

    Pass 1: Find top-3 sessions via session_summary points.
    Pass 2: Within those sessions, retrieve Top-K chunks semantically
            (not distance-limited — catches related chunks regardless of position).
    Pass 3: Apply time-decay blend score; sort descending.
    Pass 4: Expand context window (±context_window adjacent chunks).
    Final:  Sort by (session_key, chunk_index) for coherent reading order.
    """
    conv_cols = list_conv_collections(cfg, months)
    if not conv_cols:
        return []

    importance_filter = importance_filter or []

    # ── Pass 1: coarse session filter via session_summary points ───────────────
    session_hits: list[dict] = []
    for col in conv_cols:
        h = search_collection(
            col, vector, 5,
            [{"key": "type",       "match": {"value": "session_summary"}},
             {"key": "is_deleted", "match": {"value": False}}]
            + importance_filter,
            cfg,
        )
        session_hits.extend(h)

    session_hits.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_sessions = session_hits[:3]
    session_keys = [h["payload"]["session_key"] for h in top_sessions
                    if h.get("payload", {}).get("session_key")]

    if not session_keys:
        print("ℹ️  No session_summary points found. "
              "Archive sessions to enable two-pass search.", file=sys.stderr)
        return []

    # ── Pass 2: Top-K chunk semantic search within matched sessions ─────────
    fetch_n = max(limit * 3, 15)
    chunk_hits: list[dict] = []
    col_map: dict[str, str] = {}   # point_id -> collection

    for col in conv_cols:
        h = search_collection(
            col, vector, fetch_n,
            [{"key": "type",        "match": {"value": "chunk"}},
             {"key": "session_key", "match": {"any": session_keys}},
             {"key": "is_deleted",  "match": {"value": False}}]
            + importance_filter,
            cfg,
        )
        for pt in h:
            col_map[pt["id"]] = col
        chunk_hits.extend(h)

    if not chunk_hits:
        return []

    # ── Pass 3: time-decay blend + sort ───────────────────────────────────
    chunk_hits = _apply_time_decay(chunk_hits, time_weight)
    chunk_hits.sort(key=lambda x: x.get("blended_score", 0), reverse=True)
    top_chunks = chunk_hits[:limit]

    # ── Pass 4: expand ±context_window adjacent chunks ───────────────────
    if context_window > 0:
        seen_ids: set[str] = {h["id"] for h in top_chunks}
        extra: list[dict] = []
        for hit in list(top_chunks):   # iterate over a snapshot
            p   = hit.get("payload", {})
            col = col_map.get(hit["id"], hit.get("_collection", ""))
            for direction in ["prev_chunk_id", "next_chunk_id"]:
                cid = p.get(direction)
                if cid and cid not in seen_ids:
                    adj = fetch_point_by_id(cid, col, cfg)
                    if adj:
                        col_map[adj["id"]] = col
                        extra.append(adj)
                        seen_ids.add(cid)
        top_chunks.extend(extra)

    # Sort by (session_key, chunk_index) for coherent reading order
    top_chunks.sort(key=lambda x: (
        x.get("payload", {}).get("session_key", ""),
        x.get("payload", {}).get("chunk_index", 0),
    ))
    return top_chunks

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
    blended    = hit.get("blended_score")
    rec_score  = hit.get("_recency_score")
    is_context = hit.get("_is_context", False)
    collection = hit.get("_collection", "?")

    # Build score string
    if rer_score is not None:
        score_str = f"{rer_score:.4f} (rerank)"
    elif blended is not None and rec_score is not None:
        score_str = f"{blended:.4f} (blend: sem={vec_score:.3f} rec={rec_score:.3f})"
    elif blended is not None:
        score_str = f"{blended:.4f} (blended)"
    else:
        score_str = f"{vec_score:.4f} (vector)"

    is_conv  = collection.startswith("conv_")
    ptype    = p.get("type", "")
    if ptype == "session_summary":
        kind = "🗒️  Session"
    elif is_context:
        kind = "💬▸context"
    elif is_conv:
        kind = "💬 Conv"
    else:
        kind = "📄 Doc"

    display_time = (
        p.get("chunk_turn_start_ts") or p.get("indexed_at_iso") or p.get("iso_time") or ""
    )[:19]

    print(f"[{i}] {kind}  score={score_str}  time={display_time}")

    if is_conv or ptype in ("chunk", "session_summary"):
        if ptype == "session_summary":
            print(f"    session={p.get('session_key')}  turns={p.get('turn_count')}")
            print(f"    summary: {p.get('full_summary', '')}")
            print(f"    entities: {p.get('all_entities', [])}")
        else:
            archived_file = p.get("archived_file")
            turn_range    = p.get("turn_range")
            chunk_idx     = p.get("chunk_index", p.get("block_index", "?"))
            nav = []
            if p.get("prev_chunk_id"):
                nav.append(f"prev=chunk_{chunk_idx - 1 if isinstance(chunk_idx, int) else '?'}")
            if p.get("next_chunk_id"):
                nav.append(f"next=chunk_{chunk_idx + 1 if isinstance(chunk_idx, int) else '?'}")
            nav_str = "  (" + ", ".join(nav) + ")" if nav else ""
            print(f"    session={p.get('session_key')}  chunk={chunk_idx}  turns={turn_range}{nav_str}")
            print(f"    summary: {p.get('summary', '')}")
            if archived_file:
                print(f"    → read full: python core/mem_scripts/mem_search.py read_chunk '{archived_file}' {turn_range}")

        if show_full and ptype != "session_summary":
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
        print(f"\u274c Could not read: {archived_file}")
        return
    print(f"\ud83d\udcd6 {archived_file}  turns={nums[:2]}\n")
    for ti, turn in enumerate(full.get("turns", []), 1):
        u = turn.get("user", "")
        a = turn.get("assistant", "")
        print(f"[Turn {ti}]")
        print(f"\ud83d\udc64 {u}")
        print(f"\ud83e\udd16 {a}")
        print()


# \u2500\u2500 main \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

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
                        help="Also print full conversation/file content")
    # ── New PR-3 args ──────────────────────────────────────────────────
    parser.add_argument("--mode",           default="two-pass",
                        choices=["simple", "two-pass"],
                        help="Search mode: 'simple' (legacy flat search) | "
                             "'two-pass' session→chunk Top-K (default)")
    parser.add_argument("--time-weight",    type=float, default=0.25,
                        help="Recency weight blended with semantic score. "
                             "0=pure semantic, 1=pure recency (default 0.25)")
    parser.add_argument("--context-window", type=int,   default=1,
                        help="Expand ±N adjacent chunks as reading context (default 1)")
    args = parser.parse_args()

    query_str = " ".join(args.query)
    cfg       = load_config()
    months_list = [m.strip() for m in args.months.split(",")] if args.months else None

    print(f"🔍 '{query_str}'  type={args.type}  mode={args.mode}  "
          f"time-weight={args.time_weight}  limit={args.limit}")

    try:
        vector = get_embedding(query_str, cfg)
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        sys.exit(1)

    importance_filter = []
    if args.min_importance > 0:
        importance_filter.append(
            {"key": "importance_score", "range": {"gte": args.min_importance}}
        )

    all_hits: list[dict] = []

    # ── Conversation search ────────────────────────────────────────────
    if args.type in ("all", "conv"):
        if args.mode == "two-pass":
            tp_hits = search_two_pass_topk(
                query_str, vector, cfg,
                limit=args.limit,
                time_weight=args.time_weight,
                context_window=args.context_window,
                months=months_list,
                importance_filter=importance_filter,
            )
            if tp_hits:
                all_hits.extend(tp_hits)
            else:
                # Fallback to simple mode if no session_summary points exist yet
                print("⚠️  two-pass found nothing, falling back to simple mode",
                      file=sys.stderr)
                args.mode = "simple"

        if args.mode == "simple":
            fetch_per_col = max(args.limit * 2, 10)
            conv_cols = list_conv_collections(cfg, months_list)
            conv_filter = (
                [{"key": "is_deleted", "match": {"value": False}}] + importance_filter
            )
            for col in conv_cols:
                hits = search_collection(col, vector, fetch_per_col, conv_filter, cfg)
                all_hits.extend(hits)
            if not conv_cols:
                print("ℹ️  No conv_* collections found.")

    # ── Document search (always simple) ───────────────────────────────
    if args.type in ("all", "doc"):
        fetch_per_col = max(args.limit * 2, 10)
        hits = search_collection(
            "document_memory", vector, fetch_per_col,
            importance_filter or None, cfg,
        )
        all_hits.extend(hits)

    if not all_hits:
        print("ℹ️  No results found.")
        return

    # In simple mode, apply time-decay + sort here
    # (two-pass already did this internally)
    if args.mode == "simple" and args.type in ("all", "conv"):
        conv_only = [h for h in all_hits if h.get("_collection", "").startswith("conv_")]
        docs_only = [h for h in all_hits if not h.get("_collection", "").startswith("conv_")]
        _apply_time_decay(conv_only, args.time_weight)
        all_hits = conv_only + docs_only

    # Final sort + optional rerank (skip context chunks from rerank)
    main_hits    = [h for h in all_hits if not h.get("_is_context")]
    context_hits = [h for h in all_hits if h.get("_is_context")]
    main_hits.sort(key=lambda x: x.get("blended_score", x.get("score", 0)), reverse=True)
    main_hits = rerank(query_str, main_hits, cfg)[:args.limit]

    # Re-merge context chunks for display
    final_hits = main_hits + context_hits
    if args.mode == "two-pass":
        # Re-sort by (session_key, chunk_index) for reading coherence
        final_hits.sort(key=lambda x: (
            x.get("payload", {}).get("session_key", ""),
            x.get("payload", {}).get("chunk_index", 9999),
        ))

    print(f"\n📊 {len(final_hits)} results "
          f"({len(main_hits)} matched + {len(context_hits)} context):\n")
    for i, hit in enumerate(final_hits, 1):
        print_hit(i, hit, show_full=args.full)


if __name__ == "__main__":
    main()
