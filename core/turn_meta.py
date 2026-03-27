"""
core/turn_meta.py — Turn-level metadata extractor (zero LLM calls).

Extracts structured metadata from a single conversation turn,
to be stored inline with each AI message in recent_messages.

Usage:
    from core.turn_meta import extract_turn_meta
    meta = extract_turn_meta(user_query, ai_answer)
    # meta = {
    #   "summary":   "用戶問GPU選購 → RTX 4090 適合本地推論...",
    #   "entities":  ["GPU", "RTX", "推論", "VRAM"],
    #   "intent":    "explain",
    #   "time_refs": ["今天"],
    #   "ts":        "2026-03-27T...",
    # }
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Optional


# ── Intent keyword maps ────────────────────────────────────────────────────────

_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("explain",  ["如何", "怎麼", "怎樣", "什麼是", "how", "what is", "explain", "tell me"]),
    ("create",   ["幫我", "請幫", "幫助我", "請寫", "生成", "建立", "create", "write", "generate", "make", "build"]),
    ("reason",   ["為什麼", "為何", "原因", "why", "reason"]),
    ("compare",  ["比較", "差異", "vs", "versus", "difference", "哪個", "哪種", "which"]),
    ("search",   ["找", "搜尋", "查一下", "查詢", "search", "find", "look up"]),
    ("fix",      ["修復", "修正", "錯誤", "bug", "fix", "error", "問題", "失敗"]),
    ("summary",  ["總結", "摘要", "整理", "summarize", "recap"]),
]

_STOPWORDS = frozenset({
    "user", "assistant", "true", "false", "none", "null",
    "this", "that", "with", "from", "have", "will", "been",
    "were", "they", "there", "their", "which", "when", "also",
    "here", "some", "more", "than", "then", "into", "just",
    "only", "each", "such", "over", "very", "what", "your",
})

_TIME_PATTERN = re.compile(
    r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2}"
    r"|\d{1,2}月\d{1,2}日"
    r"|今天|昨天|明天|後天|上週|本週|上個月|這個月|last week|yesterday|today|tomorrow)"
    r"\b",
    re.IGNORECASE,
)

_CJK_PATTERN    = re.compile(r"[\u4e00-\u9fff]{2,4}")
_ENGLISH_PATTERN = re.compile(r"\b([A-Z]{2,}|[A-Z][a-z]{3,})\b")


# ── Core extraction ────────────────────────────────────────────────────────────

def _extract_intent(query: str) -> str:
    q_lower = query.strip().lower()
    for intent, keywords in _INTENT_PATTERNS:
        if any(q_lower.startswith(kw) or kw in q_lower for kw in keywords):
            return intent
    return "general"


def _extract_entities(user_query: str, ai_answer: str, max_entities: int = 12) -> list[str]:
    combined = user_query + " " + ai_answer
    entities: list[str] = []
    seen: set[str] = set()

    # CJK: 2~4 char clusters appearing ≥2 times, sorted by frequency
    cjk_words = _CJK_PATTERN.findall(combined)
    for word, count in Counter(cjk_words).most_common(20):
        if count >= 2 and word not in seen:
            entities.append(word)
            seen.add(word)

    # English: ALL-CAPS acronyms + TitleCase words ≥4 chars
    for m in _ENGLISH_PATTERN.finditer(combined):
        word = m.group(1)
        if word.lower() not in _STOPWORDS and word not in seen:
            entities.append(word)
            seen.add(word)
            if len(entities) >= max_entities:
                break

    return entities[:max_entities]


def _build_summary(user_query: str, ai_answer: str,
                   max_q: int = 80, max_a: int = 150) -> str:
    q = user_query.strip().replace("\n", " ")[:max_q]
    a = ai_answer.strip().replace("\n", " ")[:max_a]
    # Remove <think>...</think> blocks if present
    a = re.sub(r"<think>.*?</think>", "", a, flags=re.DOTALL).strip()[:max_a]
    return f"{q} → {a}"


def _extract_time_refs(text: str, max_refs: int = 5) -> list[str]:
    matches = _TIME_PATTERN.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
            if len(result) >= max_refs:
                break
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_turn_meta(
    user_query: str,
    ai_answer: str,
    *,
    max_entities: int = 12,
    summary_q_len: int = 80,
    summary_a_len: int = 150,
) -> dict:
    """
    Extract structured metadata from one conversation turn.
    Pure text processing — zero LLM / API calls.

    Returns:
        {
            "summary":   str,        # "<user intent> → <ai preview>"
            "entities":  list[str],  # up to 12 keywords
            "intent":    str,        # explain / create / reason / compare /
                                     # search / fix / summary / general
            "time_refs": list[str],  # date / time expressions found
            "ts":        str,        # ISO-8601 UTC timestamp
        }
    """
    if not isinstance(user_query, str):
        user_query = str(user_query or "")
    if not isinstance(ai_answer, str):
        ai_answer = str(ai_answer or "")

    return {
        "summary":   _build_summary(user_query, ai_answer, summary_q_len, summary_a_len),
        "entities":  _extract_entities(user_query, ai_answer, max_entities),
        "intent":    _extract_intent(user_query),
        "time_refs": _extract_time_refs(user_query + " " + ai_answer),
        "ts":        datetime.now(timezone.utc).isoformat(),
    }


# ── Aggregation helper (used during archiving) ─────────────────────────────────

def aggregate_session_meta(messages: list[dict]) -> dict:
    """
    Merge turn_meta from all AI messages in a session into session-level
    summary, entities, intents and time_refs. Used by archive_and_summarize_session().

    Args:
        messages: list of message dicts (recent_messages from ThreadMemory)

    Returns:
        {
            "full_summary":  str,       # first 5 turn summaries joined by " | "
            "all_entities":  list[str], # deduplicated, order-preserving
            "all_intents":   list[str], # unique intents seen
            "all_time_refs": list[str], # unique time references
        }
    """
    summaries:  list[str] = []
    entities:   list[str] = []
    intents:    list[str] = []
    time_refs:  list[str] = []
    seen_ent:   set[str]  = set()
    seen_int:   set[str]  = set()
    seen_tr:    set[str]  = set()

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        meta = msg.get("turn_meta")
        if not isinstance(meta, dict):
            continue

        # Summary — keep first 5
        s = meta.get("summary", "")
        if s and len(summaries) < 5:
            summaries.append(s)

        # Entities — deduplicated, order-preserving
        for e in meta.get("entities", []):
            if e not in seen_ent:
                seen_ent.add(e)
                entities.append(e)

        # Intents
        intent = meta.get("intent", "")
        if intent and intent not in seen_int:
            seen_int.add(intent)
            intents.append(intent)

        # Time refs
        for tr in meta.get("time_refs", []):
            if tr not in seen_tr:
                seen_tr.add(tr)
                time_refs.append(tr)

    return {
        "full_summary":  " | ".join(summaries),
        "all_entities":  entities[:30],
        "all_intents":   intents,
        "all_time_refs": time_refs[:10],
    }
