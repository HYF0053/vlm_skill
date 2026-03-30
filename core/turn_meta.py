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
    "about", "would", "could", "should", "truth", "blind", "flying",
    "please", "thank", "thanks", "hello", "hi", "sorry",
})

_CJK_STOPWORDS = frozenset({
    "完成", "尚未", "進行", "是否", "需要", "已經", "正在", "可以", 
    "提供", "說明", "根據", "關於", "顯示", "包括", "特別", "進行中",
    "回答", "問題", "請求", "處理", "執行", "日誌", "狀態", "結果",
    "摘要", "總結", "整理", "思考", "分析", "任務", "助手", "過程",
    "成功", "失敗", "原因", "錯誤", "內容", "部分", "階段", "步驟",
    "目前", "現在", "今天", "明天", "昨天", "請幫", "幫我", "幫助",
    "轉錄", "格式", "大小", "檔案", "片段", "分割", "增加", "減少",
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


def _clean_content_for_extraction(text: str) -> str:
    """
    Remove reasoning blocks, thought processes, and meta-agent talk
    to focus on the 'meat' of the conversation for entity extraction.
    """
    # 1. Strip <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # 2. Strip Thinking Process block (usually at start or after a separator)
    # This regex looks for 'Thinking Process:' and tries to find where it ends
    # (usually a double newline or when a known header starts)
    text = re.sub(r"(?i)Thinking Process:.*?(?=\n\n(?:[#*]|##)|\n\n[^\n]+:|$)", "", text, flags=re.DOTALL)
    
    # 3. Strip common meta-headers
    text = re.sub(r"(?im)^(Analysis|Thought|Plan):.*$", "", text)
    
    return text.strip()


def _extract_entities(user_query: str, ai_answer: str, max_entities: int = 12) -> list[str]:
    # Clean AI answer to remove thoughts
    cleaned_ai = _clean_content_for_extraction(ai_answer)
    
    # If cleaned_ai is too short/empty (meaning the message was ONLY thinking),
    # then fallback to the original but be very strict.
    if len(cleaned_ai) < 20:
        combined = user_query + " " + ai_answer
    else:
        combined = user_query + " " + cleaned_ai
        
    entities: list[str] = []
    seen: set[str] = set()

    # CJK: 2~4 char clusters appearing ≥2 times, sorted by frequency
    cjk_words = _CJK_PATTERN.findall(combined)
    for word, count in Counter(cjk_words).most_common(20):
        # Additional filtering for Chinese
        if count >= 2 and word not in seen:
            is_stop = any(sw in word for sw in _CJK_STOPWORDS)
            if not is_stop:
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


# ── Removed Legacy Aggregation Helper ──
