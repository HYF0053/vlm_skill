from __future__ import annotations
import json
import os
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable, Optional, Generator

# --- Constants & Config ---
DATA_DIR = Path(__file__).parent.parent / "data" / "memory"
RECENT_MESSAGES_KEEP = 10  # Increased slightly since we don't summarize automatically as often

# --- Models ---
@dataclass
class ThreadMemory:
    session_key: str
    display_name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    turn_count: int = 0
    # Structured Memory
    user_profile: dict = field(default_factory=dict)
    current_project_status: dict = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)
    preferences: dict = field(default_factory=dict)
    
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
    _locks: dict[str, threading.RLock] = {}
    _meta_lock = threading.Lock()

    @classmethod
    def get(cls, path: str) -> threading.RLock:
        with cls._meta_lock:
            if path not in cls._locks:
                cls._locks[path] = threading.RLock()
            return cls._locks[path]

class MemoryStore:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _thread_path(self, session_key: str) -> Path:
        safe_key = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_key)
        return self.data_dir / f"thread_{safe_key}.json"

    @contextmanager
    def _lock_session(self, session_key: str) -> Generator[ThreadMemory, None, None]:
        """Provides a thread-safe context for loading, modifying, and saving memory."""
        path = self._thread_path(session_key)
        lock = FileLock.get(str(path))
        with lock:
            mem = self.load_thread(session_key)
            yield mem
            self.save_thread(mem)

    def load_thread(self, session_key: str) -> ThreadMemory:
        path = self._thread_path(session_key)
        if not path.exists():
            return ThreadMemory(session_key=session_key)
        
        lock = FileLock.get(str(path))
        with lock:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if not content.strip():
                        return ThreadMemory(session_key=session_key)
                    data = json.loads(content)
                    valid_fields = {k: v for k, v in data.items() if k in ThreadMemory.__dataclass_fields__}
                    return ThreadMemory(**valid_fields)
            except Exception as e:
                print(f"Error loading thread {session_key}: {e}")
                return ThreadMemory(session_key=session_key)

    def save_thread(self, mem: ThreadMemory) -> None:
        mem.last_updated_at = datetime.now(timezone.utc).isoformat()
        path = self._thread_path(mem.session_key)
        lock = FileLock.get(str(path))
        with lock:
            temp_path = path.with_suffix(".tmp")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(mem), f, ensure_ascii=False, indent=2)
                temp_path.replace(path)
            except Exception as e:
                print(f"Error saving thread {mem.session_key}: {e}")
                if temp_path.exists(): temp_path.unlink()

    def get_session_system_context(self, session_key: str) -> str:
        """Constructs the system context based on structured memory."""
        mem = self.load_thread(session_key)
        parts = []
        
        # 1. User Profile & Preferences
        profile_parts = []
        if mem.user_profile:
            p_text = ", ".join(f"{k}: {v}" for k, v in mem.user_profile.items())
            profile_parts.append(f"User Profile: {p_text}")
        if mem.preferences:
            pref_text = ", ".join(f"{k}: {v}" for k, v in mem.preferences.items())
            profile_parts.append(f"Preferences: {pref_text}")
        if profile_parts:
            parts.append("\n".join(profile_parts))

        # 2. Project Status
        if mem.current_project_status:
            status_text = "\n".join(f"  - {k}: {v}" for k, v in mem.current_project_status.items())
            parts.append(f"Current Project Status:\n{status_text}")

        # 3. Known Facts
        if mem.facts:
            facts_text = "\n".join(f"  • {f}" for f in mem.facts)
            parts.append(f"Important Facts:\n{facts_text}")

        if not parts:
            return ""

        return (
            "\n\n"
            "================================================================\n"
            "📌 STRUCTURED MEMORY (Known Context)\n"
            "================================================================\n"
            + "\n\n".join(parts)
            + "\n================================================================\n"
        )

    def record_turn(self, session_key: str, user_message: str, ai_message: str) -> ThreadMemory:
        """Saves a turn to history. No automatic summarization here."""
        with self._lock_session(session_key) as mem:
            mem.turn_count += 1
            
            # Keep history manageable in the JSON, but the actual 'pruning' happens in LangGraph checkpointer
            mem.recent_messages.append({"role": "user", "content": user_message[:2000], "ts": datetime.now(timezone.utc).isoformat()})
            mem.recent_messages.append({"role": "assistant", "content": ai_message[:2000], "ts": datetime.now(timezone.utc).isoformat()})
            
            # Keep only last 20 messages in JSON for performance
            if len(mem.recent_messages) > 20:
                mem.recent_messages = mem.recent_messages[-20:]
            return mem

    def upsert_memory(self, session_key: str, key: str, value: Any, mem_type: str = "fact") -> str:
        """
        Tool-callable method to update structured memory.
        mem_type: 'fact' (append to list), 'preference' (overwrite key), 
                  'profile' (overwrite key), 'project' (overwrite key)
        """
        # Normalize key
        key = key.strip()
        msg = ""

        with self._lock_session(session_key) as mem:
            if mem_type == "fact":
                # Facts are append-only. We now include the key to make them searchable/clear.
                fact_entry = f"{key}: {value}" if key else str(value)
                mem.facts.append(fact_entry)
                msg = f"Fact recorded under category '{key}': {value}"
            elif mem_type == "preference":
                old_val = mem.preferences.get(key)
                mem.preferences[key] = value
                msg = f"Preference updated: {key} = {value}"
                if old_val:
                    msg += f" (previous value '{old_val}' was overwritten. If you wanted to keep it, you should have merged it in the 'value'.)"
            elif mem_type == "profile":
                old_val = mem.user_profile.get(key)
                mem.user_profile[key] = value
                msg = f"User profile updated: {key} = {value}"
                if old_val:
                    msg += f" (previous value '{old_val}' was overwritten)"
            elif mem_type == "project":
                old_val = mem.current_project_status.get(key)
                mem.current_project_status[key] = value
                msg = f"Project status updated: {key} = {value}"
                if old_val:
                    msg += f" (previous value '{old_val}' was overwritten)"
            else:
                return f"Unknown memory type: {mem_type}"

        return msg

    def delete_thread(self, session_key: str) -> bool:
        path = self._thread_path(session_key)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_threads(self) -> list[ThreadMemory]:
        threads: list[ThreadMemory] = []
        if not self.data_dir.exists(): return []
        for p in self.data_dir.glob("thread_*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data and "session_key" in data:
                        valid_fields = {k: v for k, v in data.items() if k in ThreadMemory.__dataclass_fields__}
                        threads.append(ThreadMemory(**valid_fields))
            except Exception: continue
        threads.sort(key=lambda t: t.last_updated_at, reverse=True)
        return threads

    def list_session_keys(self) -> list[str]:
        return [t.session_key for t in self.list_threads()]

    def get_next_session_name(self) -> str:
        keys = self.list_session_keys()
        i = 1
        while True:
            candidate = f"session{i}"
            if candidate not in keys: return candidate
            i += 1

