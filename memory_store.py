"""
memory_store.py — Persistent Memory Layer
==========================================

架構說明
--------
這個模組解決「清 session 就全忘」的問題，實作 session / memory 分層：

  ┌─────────────────────────────────────────────────────┐
  │  Working Memory (InMemorySaver / LangGraph state)   │  ← 本次對話
  │  session_key = "user:123:thread:project-abc"        │
  └────────────────────┬────────────────────────────────┘
                       │ 雙重觸發 → 壓縮摘要寫回持久層
                       ▼
  ┌─────────────────────────────────────────────────────┐
  │  Persistent Layer  (data/memory/*.json)             │  ← 跨 session 記憶
  │  - thread_{session_key}.json  : 單一 thread 摘要    │
  └─────────────────────────────────────────────────────┘

SessionKey 格式 (借鏡 Gateway 概念)
------------------------------------
  user:{user_id}:dm:main           → 單人私訊（共用）
  user:{user_id}:thread:{name}     → 具名專案/功能 thread
  group:{channel_id}               → 群組頻道
  default                          → 未指定時的 fallback

Summary Trigger（雙重觸發）
---------------------------
  觸發條件 A：距上次摘要已累積 >= SUMMARY_EVERY_N_TURNS 輪
  觸發條件 B：recent_messages 總字元數超過 SUMMARY_CHAR_BUDGET
  兩個條件任一成立 → 觸發壓縮（需要有 llm_summariser）
  
  好處：短對話（幾輪但每輪很長）也能及時壓縮，
        不會因為「才五輪但 token 已爆」而漏掉。
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Persistent storage directory (relative to this file)
DATA_DIR = Path(__file__).parent / "data" / "memory"

# ── 雙重壓縮觸發條件（任一成立就壓縮）────────────────────────────────────
# 觸發條件 A：距上次摘要至少累積了這麼多輪
SUMMARY_EVERY_N_TURNS: int = 10

# 觸發條件 B：recent_messages 的總字元數超過此值
# （每個英字母 ≈ 0.25 token，中文 ≈ 0.5~1 token，4000 chars ≈ ~1500 tokens）
# 可依模型 context window 調整，預設偏保守
SUMMARY_CHAR_BUDGET: int = 4_000
# ────────────────────────────────────────────────────────────────────────────

# 最多保留幾則「原文訊息」（每則 = 1 個 role 的發言，非一輪）
RECENT_MESSAGES_KEEP: int = 6


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ThreadMemory:
    """Persisted metadata for a single conversation thread."""
    session_key: str
    display_name: str = ""
    created_at: str = field(default_factory=lambda: _utcnow())
    last_updated_at: str = field(default_factory=lambda: _utcnow())
    turn_count: int = 0                  # total turns ever
    last_summarised_at_turn: int = 0     # which turn we last summarised
    summary: str = ""                    # compressed rolling summary
    # Last few recent messages (verbatim) for continuity
    recent_messages: list[dict] = field(default_factory=list)
    # Tags / topics discovered by the LLM during summarisation
    tags: list[str] = field(default_factory=list)
    # Free-form metadata the app can store (e.g. model used, language)
    meta: dict = field(default_factory=dict)


@dataclass
class UserProfile:
    """Cross-thread user profile."""
    user_id: str
    display_name: str = ""
    created_at: str = field(default_factory=lambda: _utcnow())
    last_seen_at: str = field(default_factory=lambda: _utcnow())
    preferred_language: str = "zh-TW"
    # Summary of who this user is, built up across sessions
    long_term_summary: str = ""
    # All session keys this user has used
    known_session_keys: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitise_key(key: str) -> str:
    """Turn a session_key into a safe filename fragment."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", key)


# ---------------------------------------------------------------------------
# Persistence Helpers
# ---------------------------------------------------------------------------

class _FileLock:
    """Lightweight per-file lock to avoid concurrent write corruption."""
    _locks: dict[str, threading.Lock] = {}
    _meta_lock = threading.Lock()

    @classmethod
    def get(cls, path: str) -> threading.Lock:
        with cls._meta_lock:
            if path not in cls._locks:
                cls._locks[path] = threading.Lock()
            return cls._locks[path]


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _FileLock.get(str(path))
    with lock:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)


# ---------------------------------------------------------------------------
# Core Store
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    Thread-safe persistent memory store.

    Typical usage
    -------------
    store = MemoryStore()
    key   = MemoryStore.build_key(user_id="alice", thread_name="invoice-project")

    # On session START — get context to inject into system prompt
    context = store.get_session_start_context(key)

    # After each agent turn
    store.record_turn(key, user_msg, ai_msg, llm_summariser=...)

    # On session END (optional explicit flush)
    store.flush_session(key, llm_summariser=...)
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # SessionKey helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_key(
        user_id: str = "anonymous",
        thread_name: str = "main",
        channel_id: Optional[str] = None,
    ) -> str:
        """
        Build a meaningful session key.

        Examples
        --------
        build_key("alice", "invoice-project")  →  "user:alice:thread:invoice-project"
        build_key(channel_id="ch-456")          →  "group:ch-456"
        build_key("alice")                      →  "user:alice:thread:main"
        """
        if channel_id:
            return f"group:{channel_id}"
        safe_user = re.sub(r"[^a-zA-Z0-9_\-]", "-", user_id)
        safe_thread = re.sub(r"[^a-zA-Z0-9_\-]", "-", thread_name)
        return f"user:{safe_user}:thread:{safe_thread}"

    @staticmethod
    def parse_key(session_key: str) -> dict[str, str]:
        """Parse a session_key back into its components."""
        if session_key.startswith("group:"):
            return {"type": "group", "channel_id": session_key[6:]}
        m = re.match(r"^user:([^:]+):thread:([^:]+)$", session_key)
        if m:
            return {"type": "user", "user_id": m.group(1), "thread_name": m.group(2)}
        m2 = re.match(r"^user:([^:]+):dm:([^:]+)$", session_key)
        if m2:
            return {"type": "dm", "user_id": m2.group(1), "channel": m2.group(2)}
        return {"type": "unknown", "raw": session_key}

    # ------------------------------------------------------------------
    # ThreadMemory CRUD
    # ------------------------------------------------------------------

    def _thread_path(self, session_key: str) -> Path:
        return self.data_dir / f"thread_{_sanitise_key(session_key)}.json"

    def load_thread(self, session_key: str) -> ThreadMemory:
        data = _load_json(self._thread_path(session_key))
        if not data:
            return ThreadMemory(session_key=session_key)
        # Reconstruct dataclass from dict
        return ThreadMemory(**{k: v for k, v in data.items() if k in ThreadMemory.__dataclass_fields__})

    def save_thread(self, mem: ThreadMemory) -> None:
        mem.last_updated_at = _utcnow()
        _save_json(self._thread_path(mem.session_key), asdict(mem))

    # ------------------------------------------------------------------
    # UserProfile CRUD
    # ------------------------------------------------------------------

    def _user_path(self, user_id: str) -> Path:
        return self.data_dir / f"user_{_sanitise_key(user_id)}.json"

    def load_user(self, user_id: str) -> UserProfile:
        data = _load_json(self._user_path(user_id))
        if not data:
            return UserProfile(user_id=user_id)
        return UserProfile(**{k: v for k, v in data.items() if k in UserProfile.__dataclass_fields__})

    def save_user(self, profile: UserProfile) -> None:
        profile.last_seen_at = _utcnow()
        _save_json(self._user_path(profile.user_id), asdict(profile))

    # ------------------------------------------------------------------
    # Session Start / Context Injection
    # ------------------------------------------------------------------

    def get_session_start_context(
        self,
        session_key: str,
        include_recent_n: int = RECENT_MESSAGES_KEEP,
    ) -> str:
        """
        Build a context block to prepend into the system prompt when a new
        session opens with this session_key.

        Returns "" if this is a brand-new thread with no history.
        """
        mem = self.load_thread(session_key)
        parts: list[str] = []

        if mem.summary:
            parts.append(
                "=== 持久記憶（過往對話摘要）===\n"
                f"{mem.summary}\n"
                f"[最後更新於 {mem.last_updated_at[:10]}，共 {mem.turn_count} 輪對話]"
            )

        if mem.recent_messages:
            recent_to_show = mem.recent_messages[-include_recent_n:]
            msgs_text = "\n".join(
                f"  [{m.get('role', '?')}] {m.get('content', '')[:300]}"
                for m in recent_to_show
            )
            parts.append(
                "=== 最近幾輪對話（接續上下文）===\n"
                f"{msgs_text}"
            )

        if not parts:
            return ""

        return (
            "\n\n"
            "================================================================\n"
            "📌 SESSION MEMORY (載入自持久層，請參考但不要直接重複輸出)\n"
            "================================================================\n"
            + "\n\n".join(parts)
            + "\n================================================================\n"
        )

    # ------------------------------------------------------------------
    # Turn Recording
    # ------------------------------------------------------------------

    def record_turn(
        self,
        session_key: str,
        user_message: str,
        ai_message: str,
        llm_summariser: Optional[Callable[[str, str], str]] = None,
        force_summarise: bool = False,
    ) -> ThreadMemory:
        """
        Record a completed turn (user + ai) to the thread memory.

        壓縮觸發邏輯（雙重觸發，任一成立即壓縮）
        -----------------------------------------
        條件 A：距上次摘要已累積 >= SUMMARY_EVERY_N_TURNS 輪
        條件 B：recent_messages 總字元數 >= SUMMARY_CHAR_BUDGET
                （適用於「對話輪數少但每輪很長」的場景）

        Parameters
        ----------
        session_key      : Thread identifier.
        user_message     : The user's message text (stripped of image data).
        ai_message       : The AI's response text.
        llm_summariser   : Optional callable(existing_summary, new_messages_text) -> new_summary.
                           If provided, triggers rolling compression when threshold is reached.
        force_summarise  : If True, always run summarisation regardless of conditions.

        Returns the updated ThreadMemory.
        """
        mem = self.load_thread(session_key)
        mem.turn_count += 1

        # Append to recent messages (verbatim, truncated per message)
        mem.recent_messages.append({
            "role": "user",
            "content": user_message[:800],  # cap individual message to avoid single-turn bloat
            "ts": _utcnow(),
        })
        mem.recent_messages.append({
            "role": "assistant",
            "content": ai_message[:800],
            "ts": _utcnow(),
        })

        # Keep only the last RECENT_MESSAGES_KEEP * 2 messages verbatim
        keep = RECENT_MESSAGES_KEEP * 2
        if len(mem.recent_messages) > keep:
            mem.recent_messages = mem.recent_messages[-keep:]

        # ── 雙重觸發判斷 ──────────────────────────────────────────────────
        turns_since_last = mem.turn_count - mem.last_summarised_at_turn
        total_chars = sum(len(m.get("content", "")) for m in mem.recent_messages)

        trigger_by_turns = turns_since_last >= SUMMARY_EVERY_N_TURNS
        trigger_by_size  = total_chars >= SUMMARY_CHAR_BUDGET

        should_summarise = (
            llm_summariser is not None
            and (force_summarise or trigger_by_turns or trigger_by_size)
        )

        if should_summarise:
            reason = "forced" if force_summarise else (
                f"turns={turns_since_last}" if trigger_by_turns else f"chars={total_chars}"
            )
            print(f"[MemoryStore] Summarising ({reason}) key={session_key}")
            new_summary = self._run_summarisation(mem, llm_summariser)  # type: ignore[arg-type]
            if new_summary:
                mem.summary = new_summary
                mem.last_summarised_at_turn = mem.turn_count
        # ──────────────────────────────────────────────────────────────────

        self.save_thread(mem)
        return mem

    def _run_summarisation(
        self,
        mem: ThreadMemory,
        llm_summariser: Callable[[str, str], str],
    ) -> str:
        """Call the LLM summariser to compress conversation history."""
        recent_text = "\n".join(
            f"[{m.get('role', '?')}]: {m.get('content', '')}"
            for m in mem.recent_messages
        )
        try:
            return llm_summariser(mem.summary, recent_text)
        except Exception as e:
            print(f"[MemoryStore] Summarisation failed: {e}")
            return mem.summary  # keep old summary on failure

    # ------------------------------------------------------------------
    # Explicit Session Flush
    # ------------------------------------------------------------------

    def flush_session(
        self,
        session_key: str,
        llm_summariser: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        """
        Explicitly flush / summarise at session end.
        Called e.g. when user clicks 'Clear Chat'.
        """
        mem = self.load_thread(session_key)
        if not mem.recent_messages:
            return  # nothing to flush

        if llm_summariser and mem.recent_messages:
            new_summary = self._run_summarisation(mem, llm_summariser)
            if new_summary:
                mem.summary = new_summary
                mem.last_summarised_at_turn = mem.turn_count

        self.save_thread(mem)

    # ------------------------------------------------------------------
    # List / Inspect
    # ------------------------------------------------------------------

    def list_threads(self) -> list[ThreadMemory]:
        """Return all persisted threads sorted by last_updated_at DESC."""
        threads: list[ThreadMemory] = []
        for p in self.data_dir.glob("thread_*.json"):
            data = _load_json(p)
            if data and "session_key" in data:
                threads.append(
                    ThreadMemory(**{k: v for k, v in data.items() if k in ThreadMemory.__dataclass_fields__})
                )
        threads.sort(key=lambda t: t.last_updated_at, reverse=True)
        return threads

    def delete_thread(self, session_key: str) -> bool:
        """Delete a thread's persistent memory. Returns True if deleted."""
        path = self._thread_path(session_key)
        if path.exists():
            path.unlink()
            return True
        return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# Shared store instance used by gradio_app.py
memory_store = MemoryStore()


# ---------------------------------------------------------------------------
# LLM Summariser Factory
# ---------------------------------------------------------------------------

def make_llm_summariser(llm) -> Callable[[str, str], str]:
    """
    Create a summariser function backed by the given LangChain LLM instance.

    The summariser is called as:
        new_summary = summariser(existing_summary, recent_messages_text)

    It compresses the new content into a concise rolling summary in Traditional
    Chinese, preserving key facts, decisions, and context.
    """

    def _summarise(existing_summary: str, recent_messages_text: str) -> str:
        system = (
            "你是一個記憶壓縮助手。你的工作是把對話記錄壓縮成一段精確、密集的摘要，"
            "保留所有重要資訊（人名、決策、數字、任務結果、待辦事項）。"
            "請用繁體中文回應。只輸出摘要，不要加任何前綴或說明文字。"
        )
        existing_block = f"【現有摘要】\n{existing_summary}\n\n" if existing_summary else ""
        user_content = (
            f"{existing_block}"
            f"【最新對話記錄】\n{recent_messages_text}\n\n"
            "請將上面的對話整合進現有摘要，輸出一份更新後的完整摘要（200~500字）。"
        )
        from langchain.schema import HumanMessage, SystemMessage as SM
        response = llm.invoke([SM(content=system), HumanMessage(content=user_content)])
        return response.content.strip()

    return _summarise


# ---------------------------------------------------------------------------
# Convenience: SessionKey UI helpers
# ---------------------------------------------------------------------------

SESSION_KEY_PRESETS = [
    "user:anonymous:thread:main",
    "user:anonymous:thread:ocr-project",
    "user:anonymous:dm:main",
]


def suggest_session_key(user_id: str = "anonymous", thread_name: str = "main") -> str:
    """Quick helper to build and return a session key string."""
    return MemoryStore.build_key(user_id=user_id, thread_name=thread_name)


# ---------------------------------------------------------------------------
# InMemorySaver Pruning
# ---------------------------------------------------------------------------
# 解決問題 2：InMemorySaver 歷史訊息無限累積導致 context window 爆炸。
#
# 策略：「摘要橋接」
#   當 JSON 摘要觸發時，同步修剪 InMemorySaver 裡的 messages：
#   ① 刪除所有「舊」的 HumanMessage / AIMessage
#   ② 插入一則 SystemMessage（bridge）把摘要帶進去
#   ③ 保留最近 KEEP_RECENT_TURNS 輪的真實訊息
#
# 修剪後的 checkpointer 狀態：
#   [system(原始)] → [system(摘要橋接)] → [最近 N 輪]
#   token 數從「全部歷史」降到「固定上限」
#
# 注意：這需要 LangGraph 的 RemoveMessage 機制。
# ---------------------------------------------------------------------------

# App 重啟或摘要後，InMemorySaver 保留最近幾輪的完整訊息
KEEP_RECENT_TURNS_IN_MEMORY: int = 4   # 每輪 = user + ai，所以實際 8 則訊息


def prune_checkpointer(
    checkpointer,       # LangGraph InMemorySaver instance
    session_key: str,
    summary: str,
    keep_recent_turns: int = KEEP_RECENT_TURNS_IN_MEMORY,
) -> bool:
    """
    修剪 LangGraph InMemorySaver，避免 context window 爆炸。

    在 LLM 摘要完成後呼叫此函式：
    1. 讀取 checkpointer 目前的 messages
    2. 用 RemoveMessage 刪除所有舊的 HumanMessage / AIMessage
    3. 在最前面插入一則 SystemMessage 作為摘要橋接
    4. 保留最近 keep_recent_turns 輪的真實訊息

    Parameters
    ----------
    checkpointer      : 全域的 InMemorySaver 實例
    session_key       : 對應的 thread_id（也是 session_key）
    summary           : 新產生的摘要文字
    keep_recent_turns : 保留幾輪完整訊息（預設 4 輪 = 8 則）

    Returns True if any messages were pruned, False otherwise.
    """
    try:
        from langchain_core.messages import RemoveMessage, SystemMessage, HumanMessage, AIMessage

        config = {"configurable": {"thread_id": session_key}}

        # 讀取目前 checkpoint 狀態
        checkpoint = checkpointer.get(config)
        if checkpoint is None:
            return False

        messages = checkpoint.get("channel_values", {}).get("messages", [])
        if not messages:
            return False

        # ── 找出要刪和要留的訊息 ─────────────────────────────────────────
        # 策略：保留 system messages + 最後 keep_recent_turns*2 則 human/ai
        system_msgs   = [m for m in messages if isinstance(m, SystemMessage)]
        non_sys_msgs  = [m for m in messages if not isinstance(m, SystemMessage)]

        keep_n = keep_recent_turns * 2   # user + ai per turn
        to_keep_ids  = set(id(m) for m in non_sys_msgs[-keep_n:])
        to_delete    = [m for m in non_sys_msgs if id(m) not in to_keep_ids]

        if not to_delete:
            return False   # nothing to prune

        # ── 用 RemoveMessage 刪除舊訊息 ───────────────────────────────────
        # RemoveMessage needs the message id (LangChain assigns one)
        delete_ops = []
        for m in to_delete:
            if hasattr(m, "id") and m.id:
                delete_ops.append(RemoveMessage(id=m.id))

        if not delete_ops:
            return False

        # ── 準備一則摘要橋接 SystemMessage ────────────────────────────────
        bridge = SystemMessage(
            content=(
                "【以下是過去對話的摘要，作為長期記憶參考】\n"
                f"{summary}\n"
                "【以上摘要結束，接下來是最近幾輪對話】"
            )
        )

        # ── 寫入新狀態 ────────────────────────────────────────────────────
        # We use checkpointer.put() with the modified messages.
        # The cleanest way: get checkpoint metadata and write back.
        from langgraph.checkpoint.base import CheckpointMetadata

        # Build updated message list: systems + bridge + recent non-sys
        updated_messages = system_msgs + [bridge] + non_sys_msgs[-keep_n:]

        checkpoint_copy = dict(checkpoint)
        channel_values = dict(checkpoint_copy.get("channel_values", {}))
        channel_values["messages"] = updated_messages
        checkpoint_copy["channel_values"] = channel_values

        # Get the checkpoint tuple to find metadata
        checkpoint_tuple = checkpointer.get_tuple(config)
        if checkpoint_tuple is None:
            return False

        metadata: CheckpointMetadata = checkpoint_tuple.metadata or {}
        checkpointer.put(config, checkpoint_copy, metadata, {})  # type: ignore[arg-type]

        pruned_count = len(to_delete)
        print(f"[MemoryStore] Pruned {pruned_count} old messages from InMemorySaver (key={session_key})")
        return True

    except Exception as e:
        # Pruning is best-effort; never crash the main flow
        print(f"[MemoryStore] prune_checkpointer skipped: {e}")
        return False
