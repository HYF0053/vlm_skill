from __future__ import annotations
import json
import os
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable, Optional, Generator, List, Dict

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
    preferences: dict = field(default_factory=dict)
    agent_rules: dict = field(default_factory=dict)
    
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
        """Provides a cross-process and thread-safe context for loading, modifying, and saving memory."""
        import time
        path = self._thread_path(session_key)
        lock_path = path.with_suffix(".lock")
        
        # 1. Thread-level lock (within same process)
        thread_lock = FileLock.get(str(path))
        
        # 2. Cross-process file lock
        # We use atomic directory creation or O_EXCL file creation for cross-process locking
        # to avoid dependencies like portalocker/filelock.
        with thread_lock:
            start_time = time.time()
            acquired = False
            while time.time() - start_time < 10:  # 10s timeout
                try:
                    # O_EXCL is atomic on both Windows and Unix
                    fd = os.open(lock_path, os.environ.get("OS_OPEN_FLAGS", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                    with os.fdopen(fd, 'w') as f:
                        f.write(str(os.getpid()))
                    acquired = True
                    break
                except FileExistsError:
                    # Check for stale lock (older than 5 seconds)
                    try:
                        if time.time() - os.path.getmtime(lock_path) > 5:
                            os.remove(lock_path)
                    except Exception: pass
                    time.sleep(0.05)
            
            if not acquired:
                print(f"Warning: Could not acquire lock for {session_key} after 10s. Proceeding without lock (risk of race condition).")
            
            try:
                mem = self.load_thread(session_key)
                yield mem
                self.save_thread(mem)
            finally:
                if acquired and lock_path.exists():
                    try:
                        os.remove(lock_path)
                    except Exception: pass

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
        import uuid
        mem.last_updated_at = datetime.now(timezone.utc).isoformat()
        path = self._thread_path(mem.session_key)
        lock = FileLock.get(str(path))
        with lock:
            # 用 UUID 產生唯一的 temp 檔名，避免跨進程檔案衝突
            temp_path = path.with_name(f"{path.stem}_{uuid.uuid4().hex[:8]}.tmp")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(mem), f, ensure_ascii=False, indent=2)
                temp_path.replace(path)
            except Exception as e:
                print(f"Error saving thread {mem.session_key}: {e}")
                if temp_path.exists(): temp_path.unlink()

    # NOTE: Header/Prompt formatting logic moved to skills/memory/lib/logic.py
    # This keeps core/memory.py focused on pure IO and Session Persistence.

    def record_turn(self, session_key: str, user_message: str, ai_message: str, attachments: Optional[list[dict]] = None, usage: Optional[dict] = None) -> Generator[str, None, ThreadMemory]:
        """Saves a turn to history."""
        with self._lock_session(session_key) as mem:
            mem.turn_count += 1
            
            # Keep history manageable in the JSON
            user_entry = {"role": "user", "content": user_message[:2000], "ts": datetime.now(timezone.utc).isoformat()}
            if attachments:
                user_entry["attachments"] = attachments
                
            mem.recent_messages.append(user_entry)
            mem.recent_messages.append({"role": "assistant", "content": ai_message[:2000], "ts": datetime.now(timezone.utc).isoformat()})
            
            # Smart Trimming: First 2 messages (anchor) + latest rolling messages
            MAX_MSGS = 20
            if len(mem.recent_messages) > MAX_MSGS:
                anchor = mem.recent_messages[:2]
                rolling = mem.recent_messages[-(MAX_MSGS - 2):]
                mem.recent_messages = anchor + rolling

            yield "🧠 記憶已記錄。"
            return mem



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

