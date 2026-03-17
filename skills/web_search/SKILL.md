---
name: web-search
description: >
  General-purpose real-time web search skill using Tavily API. Use this skill whenever
  the user wants to search the internet for current information, recent news, prices,
  facts, events, or any topic requiring up-to-date data not available in training data.
  Triggers on: "搜尋", "search", "查一下", "最新", "latest", "找資料", "what is",
  "網路上", "google", "look up", "news about", "price of", "幫我查", "查詢",
  "即時資訊", "current", "today", "2024", "2025", "2026". Use proactively whenever
  the user's question needs real-world, recent, or factual information you might
  not have — don't wait to be asked explicitly to "search the web".
---

# Web Search Skill — 即時網路搜尋

This skill provides real-time internet search via the Tavily API. Use it whenever you need current information, news, prices, or facts that may be outside your training data.

## Quick Start

```bash
python skills/web_search/scripts/search.py <query words> [options]
```

Multi-word queries work **without quotes** — words are automatically joined:
```bash
python skills/web_search/scripts/search.py NVIDIA RTX 5090 price Taiwan --max_results 5
python skills/web_search/scripts/search.py 台灣 AI 市場 最新 趨勢 2025 --format summary
python skills/web_search/scripts/search.py latest news on DeepSeek R2 --topic news --format text
```

## Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--max_results` / `-n` | 1–10 | 5 | Number of search results |
| `--format` / `-f` | `text` \| `json` \| `summary` | `text` | Output format |
| `--depth` | `basic` \| `advanced` | `advanced` | Search depth (`advanced` gives richer content) |
| `--topic` | `general` \| `news` | `general` | Use `news` for time-sensitive news queries |

## Output Formats

### `text` (default) — Best for reading/reasoning:
```
[1] Title of result
    URL: https://...
    摘要: First 400 chars of content...

[2] ...
```

### `summary` — Best for injecting into LLM context:
```
[來源 1] Title
連結: https://...
Full content up to 600 chars

---

[來源 2] ...
```

### `json` — Best for programmatic use:
```json
[{"title": "...", "url": "...", "content": "...", "score": 0.95}]
```

## Workflow

1. **Identify what information is needed** from the user's query
2. **Choose search parameters** — topic keywords, `--topic news` for current events, `--depth basic` if speed matters
3. **Run the search script** via `execute_script` or `run_cli_command`
4. **Synthesize results** — read the output and answer the user's question, always citing sources with Markdown links

## Typical Patterns

### Pattern A: Simple fact query
```bash
# User: "TSMC 2025年最新消息"
python skills/web_search/scripts/search.py TSMC 2025 latest news --topic news --max_results 5 --format text
```

### Pattern B: Product/price research
```bash
# User: "RTX 5090 在台灣多少錢?"
python skills/web_search/scripts/search.py RTX 5090 Taiwan price 台灣 售價 2025 --max_results 5 --format text
```

### Pattern C: Multi-angle research (run 2-3 searches)
```bash
# Search 1: Overview
python skills/web_search/scripts/search.py DeepSeek R2 model release specs --max_results 3 --format summary

# Search 2: Comparison
python skills/web_search/scripts/search.py DeepSeek R2 vs GPT-4o performance benchmark --max_results 3 --format summary
```

### Pattern D: News monitoring
```bash
# User: "今天有什麼 AI 新聞?"
python skills/web_search/scripts/search.py AI artificial intelligence news today --topic news --max_results 8 --format text
```

## Source Citation Rules

After retrieving results, always:
- **Cite sources inline** using Markdown links: `[Source Title](https://url)`
- **Never fabricate URLs** — only use URLs from search results
- **Be honest about uncertainty** — if search results are inconclusive, say so
- **Use繁體中文** for final answers unless user requests otherwise

## Config

API key is managed via the platform's **Settings Tab** (Dynamic Configuration).

Resolution order:
1. **Platform Settings UI** (Injected via `TAVILY_API_KEY` env var) ← **Recommended**
2. `config/tools.json` → `{ "tavily": { "api_key": "tvly-..." } }`
3. `/home/ubuntu/ai-agent-platform/.env` (Legacy fallback)
