#!/usr/bin/env python3
"""General-purpose web search using the Tavily API.

Part of the `web_search` skill for vlm_skill.

Usage (can be called with or without quotes around multi-word queries):
    python search.py what is the latest news on NVIDIA --max_results 5 --format text
    python search.py "NVIDIA H100 price 2025" --max_results 3 --format json
    python search.py 台灣 AI 市場趨勢 2025 --max_results 5 --format text
    python search.py RTX 5090 release --format summary

Output formats:
    json    — JSON array: [{title, url, content, score}]
    text    — Human-readable numbered list (recommended for AI reasoning)
    summary — Condensed single-block text, ideal for feeding into LLM context
"""

import sys
import json
import argparse
import os


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    """Resolve Tavily API key from multiple sources (first valid wins).

    Order:
      1. config/tools.json  (canonical config)
      2. TAVILY_API_KEY environment variable
      3. .env file in project root (legacy fallback)
    """
    # Priority 1: config/tools.json relative to vlm_skill root
    config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../..", "config", "tools.json")
    )
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            key = cfg.get("tavily", {}).get("api_key", "")
            if key and key != "tvly-placeholder":
                return key
        except Exception:
            pass

    # Priority 2: environment variable
    key = os.environ.get("TAVILY_API_KEY", "")
    if key and key != "tvly-placeholder":
        return key

    env_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../../..", ".env")
    )
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("TAVILY_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    if key and key != "tvly-placeholder":
                        return key

    return ""


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------

def search(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
    topic: str = "general",
) -> list:
    """Execute a real-time web search via Tavily API.

    Args:
        query:        Search query string.
        max_results:  Maximum number of results (1-10).
        search_depth: "basic" (faster) or "advanced" (better content).
        topic:        "general" or "news" — affects result ranking.

    Returns:
        List of dicts: [{title, url, content, score}]
        On error, returns single-item list with an error entry.
    """
    api_key = _load_api_key()
    if not api_key:
        return [{
            "title": "Error: TAVILY_API_KEY not configured",
            "url": "",
            "content": (
                "Tavily API key is missing. Please set your 'api_key' in 'config/tools.json'."
            ),
            "score": 0,
        }]

    try:
        import httpx
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "httpx", "-q"], check=True)
        import httpx

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "topic": topic,
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": max(1, min(max_results, 10)),
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post("https://api.tavily.com/search", json=payload)
            resp.raise_for_status()
            raw = resp.json().get("results", [])
            return [
                {
                    "title":   r.get("title", ""),
                    "url":     r.get("url", ""),
                    "content": r.get("content", ""),
                    "score":   round(r.get("score", 0), 4),
                }
                for r in raw
            ]
    except Exception as e:
        return [{"title": "Search Error", "url": "", "content": str(e), "score": 0}]


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def fmt_json(results: list) -> str:
    return json.dumps(results, ensure_ascii=False, indent=2)


def fmt_text(results: list) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', '(no title)')}")
        lines.append(f"    URL: {r.get('url', '')}")
        snippet = r.get("content", "").replace("\n", " ")
        lines.append(f"    摘要: {snippet[:400]}")
        lines.append("")
    return "\n".join(lines).rstrip()


def fmt_summary(results: list) -> str:
    """Compact block for direct LLM context injection."""
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"[來源 {i}] {r.get('title', '')}\n"
            f"連結: {r.get('url', '')}\n"
            f"{r.get('content', '')[:600]}"
        )
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "General-purpose Tavily Web Search.\n"
            "Multi-word queries work WITHOUT quotes — words are auto-joined."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # nargs='+' collects all positional tokens and joins them — this is
    # intentional so execute_script (which splits on spaces) works correctly.
    parser.add_argument(
        "query",
        nargs="+",
        help="Search query (multi-word OK without quotes)",
    )
    parser.add_argument(
        "--max_results", "-n",
        type=int, default=5,
        help="Number of results to return (default: 5, max: 10)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "text", "summary"],
        default="text",
        help="Output format: json | text | summary (default: text)",
    )
    parser.add_argument(
        "--depth",
        choices=["basic", "advanced"],
        default="advanced",
        help="Search depth: basic (faster) | advanced (default, richer content)",
    )
    parser.add_argument(
        "--topic",
        choices=["general", "news"],
        default="general",
        help="Topic filter: general (default) | news (recent news-weighted)",
    )
    args = parser.parse_args()

    query_str = " ".join(args.query)
    results = search(query_str, args.max_results, args.depth, args.topic)

    if args.format == "json":
        print(fmt_json(results))
    elif args.format == "summary":
        print(fmt_summary(results))
    else:
        print(fmt_text(results))


if __name__ == "__main__":
    main()
