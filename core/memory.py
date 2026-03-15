from __future__ import annotations
import json
import os
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Generator

# --- Constants & Config ---
DATA_DIR = Path(__file__).parent.parent / "data" / "memory"
SUMMARY_EVERY_N_TURNS = 10
SUMMARY_CHAR_BUDGET = 4000
RECENT_MESSAGES_KEEP = 6
KEEP_RECENT_TURNS_IN_MEMORY = 4

# SESSION_KEY_PRESETS = [
#     "user:anonymous:thread:main",
#     "user:anonymous:thread:ocr-project",
#     "user:anonymous:dm:main",
# ]

# --- Models ---
@dataclass
class ThreadMemory:
    session_key: str
    display_name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    turn_count: int = 0
    last_summarised_at_turn: int = 0
    summary: str = ""
    recent_messages: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    usage: dict = field(default_factory=lambda: {
        "input_prompt": 0,
        "input_context": 0,
        "input_history": 0,
        "output": 0,
        "tool_results": 0,
        "total": 0
    })

# --- Persistence Layer ---
class FileLock:
    _locks: dict[str, threading.Lock] = {}
    _meta_lock = threading.Lock()

    @classmethod
    def get(cls, path: str) -> threading.Lock:
        with cls._meta_lock:
            if path not in cls._locks:
                cls._locks[path] = threading.Lock()
            return cls._locks[path]

class MemoryStore:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _thread_path(self, session_key: str) -> Path:
        safe_key = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_key)
        return self.data_dir / f"thread_{safe_key}.json"

    def load_thread(self, session_key: str) -> ThreadMemory:
        path = self._thread_path(session_key)
        if not path.exists():
            return ThreadMemory(session_key=session_key)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return ThreadMemory(**{k: v for k, v in data.items() if k in ThreadMemory.__dataclass_fields__})
        except Exception:
            return ThreadMemory(session_key=session_key)

    def save_thread(self, mem: ThreadMemory) -> None:
        mem.last_updated_at = datetime.now(timezone.utc).isoformat()
        path = self._thread_path(mem.session_key)
        lock = FileLock.get(str(path))
        with lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(mem), f, ensure_ascii=False, indent=2)

    def get_session_start_context(self, session_key: str, include_recent_n: Optional[int] = None) -> str:
        if include_recent_n is None:
            include_recent_n = RECENT_MESSAGES_KEEP
        mem = self.load_thread(session_key)
        parts = []
        if mem.summary:
            parts.append(f"=== 持久記憶（過往對話摘要）===\n{mem.summary}\n[最後更新於 {mem.last_updated_at[:10]}, 共 {mem.turn_count} 輪]")
        
        if mem.recent_messages:
            recent_to_show = mem.recent_messages[-include_recent_n:]
            msgs_text = "\n".join(f"  [{m.get('role', '?')}] {m.get('content', '')[:300]}" for m in recent_to_show)
            parts.append(f"=== 最近幾輪對話（接續上下文）===\n{msgs_text}")
        
        if not parts: return ""
        return (
            "\n\n"
            "================================================================\n"
            "📌 SESSION MEMORY (載入自持久層，請參考但不要直接重複輸出)\n"
            "================================================================\n"
            + "\n\n".join(parts)
            + "\n================================================================\n"
        )

    def record_turn(self, session_key: str, user_message: str, ai_message: str, llm_summariser: Optional[Callable] = None, force_summarise: bool = False, memory_params: Optional[dict] = None) -> Generator[str, None, ThreadMemory]:
        mem = self.load_thread(session_key)
        mem.turn_count += 1
        mem.recent_messages.append({"role": "user", "content": user_message[:800], "ts": datetime.now(timezone.utc).isoformat()})
        mem.recent_messages.append({"role": "assistant", "content": ai_message[:800], "ts": datetime.now(timezone.utc).isoformat()})
        
        # Determine thresholds
        # keep = (memory_params.get("recent_messages_keep") if memory_params else RECENT_MESSAGES_KEEP) * 2
        char_budget = memory_params.get("summary_char_budget") if memory_params else SUMMARY_CHAR_BUDGET
        
        # if len(mem.recent_messages) > keep:
        #     mem.recent_messages = mem.recent_messages[-keep:]

        turns_since_last = mem.turn_count - mem.last_summarised_at_turn
        total_chars = sum(len(m.get("content", "")) for m in mem.recent_messages)
        should_summarise = llm_summariser and (force_summarise or turns_since_last >= SUMMARY_EVERY_N_TURNS or total_chars >= char_budget)

        if should_summarise:
            yield "正在進行對話摘要壓縮..."
            try:
                recent_text = "\n".join(f"[{m.get('role', '?')}]: {m.get('content', '')}" for m in mem.recent_messages)
                new_summary = llm_summariser(mem.summary, recent_text)
                if new_summary:
                    mem.summary = new_summary
                    mem.last_summarised_at_turn = mem.turn_count
            except Exception as e:
                print(f"Summarisation failed: {e}")

        self.save_thread(mem)
        yield "長期記憶已儲存。"
        return mem

    def delete_thread(self, session_key: str) -> bool:
        """Delete a thread's persistent memory. Returns True if deleted."""
        path = self._thread_path(session_key)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_threads(self) -> list[ThreadMemory]:
        """Return all persisted threads sorted by last_updated_at DESC."""
        threads: list[ThreadMemory] = []
        if not self.data_dir.exists():
            return []
        for p in self.data_dir.glob("thread_*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data and "session_key" in data:
                        threads.append(
                            ThreadMemory(**{k: v for k, v in data.items() if k in ThreadMemory.__dataclass_fields__})
                        )
            except Exception:
                continue
        threads.sort(key=lambda t: t.last_updated_at, reverse=True)
        return threads

    def list_session_keys(self) -> list[str]:
        """Return list of session keys."""
        threads = self.list_threads()
        return [t.session_key for t in threads]

    def get_next_session_name(self) -> str:
        """Generate next session name like session1, session2..."""
        keys = self.list_session_keys()
        i = 1
        while True:
            candidate = f"session{i}"
            if candidate not in keys:
                return candidate
            i += 1

# --- LLM Summariser Factory ---
def make_llm_summariser(llm) -> Callable[[str, str], str]:
    def _summarise(existing_summary: str, recent_messages_text: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        system = "你是一個記憶壓縮助手。請將對話記錄壓縮成精確的摘要（繁體中文），保留重要資訊。只輸出摘要。"
        user_content = f"【現有摘要】\n{existing_summary}\n\n【最新對話記錄】\n{recent_messages_text}\n\n請整合更新摘要。"
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user_content)])
        return response.content.strip()
    return _summarise

# --- InMemory pruning ---
def prune_checkpointer(checkpointer, session_key: str, summary: str, keep_recent_turns: Optional[int] = None) -> Generator[str, None, bool]:
    if keep_recent_turns is None:
        keep_recent_turns = KEEP_RECENT_TURNS_IN_MEMORY
    try:
        yield "正在進行 InMemorySaver 訊息修剪..."
        from langchain_core.messages import RemoveMessage, SystemMessage, HumanMessage, AIMessage
        config = {"configurable": {"thread_id": session_key}}
        checkpoint = checkpointer.get(config)
        if not checkpoint: return False
        
        messages = checkpoint.get("channel_values", {}).get("messages", [])
        if not messages: return False

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_sys_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        keep_n = keep_recent_turns * 2
        to_delete = [m for m in non_sys_msgs[:-keep_n]] if len(non_sys_msgs) > keep_n else []
        
        if not to_delete: return False

        bridge = SystemMessage(content=f"【過去對話摘要】\n{summary}\n【摘要結束】")
        updated_messages = system_msgs + [bridge] + non_sys_msgs[-keep_n:]
        
        checkpoint_copy = dict(checkpoint)
        channel_values = dict(checkpoint_copy.get("channel_values", {}))
        channel_values["messages"] = updated_messages
        checkpoint_copy["channel_values"] = channel_values
        
        from langgraph.checkpoint.base import CheckpointMetadata
        checkpoint_tuple = checkpointer.get_tuple(config)
        metadata = checkpoint_tuple.metadata if checkpoint_tuple else {}
        checkpointer.put(config, checkpoint_copy, metadata, {})
        
        yield f"已修剪 {len(to_delete)} 則舊訊息。"
        return True
    except Exception as e:
        print(f"Pruning failed: {e}")
        return False
