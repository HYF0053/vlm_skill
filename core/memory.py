from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Generator, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
DATA_DIR     = BASE_DIR / "data" / "memory" / "threads"
ARCHIVE_DIR  = BASE_DIR / "data" / "memory" / "archived"
PROFILE_PATH = BASE_DIR / "data" / "memory" / "global_profile.json"
CONFIG_PATH  = BASE_DIR / "config" / "memo.json"

RECENT_MESSAGES_KEEP = 20
CHUNK_SIZE           = 5    # turns per chunk (overlap = 1 turn with next chunk)
CHUNK_OVERLAP        = 1    # shared turns between adjacent chunks


# ── Models ─────────────────────────────────────────────────────────────────────
@dataclass
class ThreadMemory:
    session_key:       str
    display_name:      str = ""
    created_at:        str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated_at:   str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    turn_count:        int = 0
    is_deleted:        bool = False

    recent_messages:   list = field(default_factory=list)
    tags:              list = field(default_factory=list)
    meta:              dict = field(default_factory=dict)
    usage:             dict = field(default_factory=lambda: {
        "input_prompt": 0, "input_context": 0, "input_history": 0,
        "output": 0, "tool_results": 0, "total": 0,
    })


# ── Global Profile ─────────────────────────────────────────────────────────────
def load_global_profile() -> dict:
    if not PROFILE_PATH.exists():
        return {"schema_version": 1, "user_profile": {}, "preferences": {}, "agent_rules": {}}
    try:
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not load global_profile.json: %s", e)
        return {"schema_version": 1, "user_profile": {}, "preferences": {}, "agent_rules": {}}


def save_global_profile(profile: dict) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROFILE_PATH.with_name(f"global_profile_{uuid.uuid4().hex[:8]}.tmp")
    profile["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        tmp.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(PROFILE_PATH)
    except Exception as e:
        logger.error("Error saving global_profile.json: %s", e)
        if tmp.exists():
            tmp.unlink()


# ── Thread-safe file lock ──────────────────────────────────────────────────────
class FileLock:
    _locks: dict[str, threading.RLock] = {}
    _meta_lock = threading.Lock()

    @classmethod
    def get(cls, path: str) -> threading.RLock:
        with cls._meta_lock:
            if path not in cls._locks:
                cls._locks[path] = threading.RLock()
            return cls._locks[path]


# ── Chunked summarization helpers ──────────────────────────────────────────────

def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_embedding(text: str, cfg: dict) -> list:
    emb = cfg["embedding"]
    resp = requests.post(
        f"{emb['base_url']}/v1/embeddings",
        json={"model": emb["model"], "input": text},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _ensure_conv_collection(collection: str, cfg: dict) -> None:
    q = cfg["qdrant"]
    base = f"http://{q['host']}:{q['port']}"
    r = requests.get(f"{base}/collections/{collection}", timeout=5)
    if r.status_code == 200:
        return
    vector_size = 1024
    for c in q.get("collections", []):
        if c["name"] == collection:
            vector_size = c.get("vector_size", 1024)
    requests.put(
        f"{base}/collections/{collection}",
        json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        timeout=10,
    ).raise_for_status()
    
    # Create Full-text payload indexes for BM25 hybrid search
    for field in ["entities", "summary"]:
        requests.put(
            f"{base}/collections/{collection}/index",
            json={
                "field_name": field,
                "field_schema": {
                    "type": "text",
                    "tokenizer": "word",
                    "lowercase": True
                }
            },
            timeout=10,
        )
    logger.info("Created Qdrant collection '%s' with text indexes", collection)


def _upsert_conv_point(collection: str, point_id: str, vector: list,
                       payload: dict, cfg: dict) -> None:
    q = cfg["qdrant"]
    base = f"http://{q['host']}:{q['port']}"
    requests.put(
        f"{base}/collections/{collection}/points?wait=true",
        json={"points": [{"id": point_id, "vector": vector, "payload": payload}]},
        timeout=10,
    ).raise_for_status()


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _upsert_single_turn_point(session_key: str, turn_meta: dict, user_msg: str, ai_msg: str, cfg: dict) -> str:
    """
    Index a single turn into Qdrant. Returns the generated point ID.
    """
    collection = datetime.now(timezone.utc).strftime("conv_%Y_%m")
    _ensure_conv_collection(collection, cfg)
    
    point_id = str(uuid.uuid4())
    now_unix = int(time.time())
    iso_now  = datetime.now(timezone.utc).isoformat()
    
    # Clean the raw text for the context payload
    clean_user = _strip_think(user_msg)
    clean_ai   = _strip_think(ai_msg)
    summary    = turn_meta.get("summary", "")
    entities   = turn_meta.get("entities", [])
    intent     = turn_meta.get("intent", "general")
    
    payload = {
        "type": "turn",
        "session_key": session_key,
        "summary": summary,
        "entities": entities,
        "intent": intent,
        "ts_unix": now_unix,
        "dt": iso_now,
        "is_deleted": False,
        "context": {
            "query": clean_user,
            "answer": clean_ai[:3000] # Cap answer to avoid huge payloads
        }
    }
    
    embed_text = f"Query: {clean_user}\nSummary: {summary}"
    try:
        vector = _get_embedding(embed_text, cfg)
        _upsert_conv_point(collection, point_id, vector, payload, cfg)
        logger.info("Upserted turn point %s to %s", point_id, collection)
    except Exception as e:
        logger.error("Failed to upsert single turn for %s: %s", session_key, e)
        
    return point_id


def archive_and_summarize_session(session_key: str, llm, memory_store) -> None:
    # Legacy wrapper so old imports don't break immediately.
    # In V3, turn indexing happens live.
    pass

# ── Startup recovery ───────────────────────────────────────────────────────────

def recover_incomplete_archives(memory_store: "MemoryStore", llm) -> None:
    # No longer needed in V3, kept as stub for API compatibility
    pass


# ── MemoryStore ────────────────────────────────────────────────────────────────

class MemoryStore:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    def _thread_path(self, session_key: str) -> Path:
        safe_key = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_key)
        return self.data_dir / f"thread_{safe_key}.json"

    @contextmanager
    def _lock_session(self, session_key: str) -> Generator[ThreadMemory, None, None]:
        import time as _time
        path = self._thread_path(session_key)
        lock_path = path.with_suffix(".lock")
        thread_lock = FileLock.get(str(path))

        with thread_lock:
            start = _time.time()
            acquired = False
            while _time.time() - start < 10:
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    with os.fdopen(fd, "w") as f:
                        f.write(str(os.getpid()))
                    acquired = True
                    break
                except FileExistsError:
                    try:
                        if _time.time() - os.path.getmtime(lock_path) > 5:
                            os.remove(lock_path)
                    except Exception:
                        pass
                    _time.sleep(0.05)

            if not acquired:
                logger.warning("Could not acquire lock for %s after 10s, proceeding without lock.", session_key)

            try:
                mem = self.load_thread(session_key)
                yield mem
                self.save_thread(mem)
            finally:
                if acquired and lock_path.exists():
                    try:
                        os.remove(lock_path)
                    except Exception:
                        pass

    def load_thread(self, session_key: str) -> ThreadMemory:
        path = self._thread_path(session_key)
        if not path.exists():
            return ThreadMemory(session_key=session_key)
        lock = FileLock.get(str(path))
        with lock:
            try:
                content = path.read_text(encoding="utf-8")
                if not content.strip():
                    return ThreadMemory(session_key=session_key)
                data = json.loads(content)
                valid = {k: v for k, v in data.items() if k in ThreadMemory.__dataclass_fields__}
                return ThreadMemory(**valid)
            except Exception as e:
                logger.error("Error loading thread %s: %s", session_key, e)
                return ThreadMemory(session_key=session_key)

    def save_thread(self, mem: ThreadMemory) -> None:
        mem.last_updated_at = datetime.now(timezone.utc).isoformat()
        path = self._thread_path(mem.session_key)
        lock = FileLock.get(str(path))
        with lock:
            tmp = path.with_name(f"{path.stem}_{uuid.uuid4().hex[:8]}.tmp")
            try:
                tmp.write_text(
                    json.dumps(asdict(mem), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp.replace(path)
            except Exception as e:
                logger.error("Error saving thread %s: %s", mem.session_key, e)
                if tmp.exists():
                    tmp.unlink()

    def record_turn(self, session_key: str, user_message: str, ai_message: str,
                    attachments: Optional[list] = None,
                    usage: Optional[dict] = None,
                    turn_meta: Optional[dict] = None) -> Generator[str, None, ThreadMemory]:
        
        cfg = _load_config()
        
        # Async Qdrant upsert so we don't block the UI thread
        def perform_upsert(meta: dict):
            point_id = _upsert_single_turn_point(session_key, meta, user_message, ai_message, cfg)
            meta["qdrant_point_id"] = point_id

        if turn_meta and isinstance(turn_meta, dict):
            threading.Thread(target=perform_upsert, args=(turn_meta,), daemon=True).start()

        with self._lock_session(session_key) as mem:
            mem.turn_count += 1
            user_entry: dict = {
                "role": "user",
                "content": user_message[:2000],
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            if attachments:
                user_entry["attachments"] = attachments
            ai_entry: dict = {
                "role": "assistant",
                "content": ai_message[:2000],
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            # Attach pre-computed turn metadata
            if turn_meta and isinstance(turn_meta, dict):
                ai_entry["turn_meta"] = turn_meta

            mem.recent_messages.append(user_entry)
            mem.recent_messages.append(ai_entry)

            if len(mem.recent_messages) > RECENT_MESSAGES_KEEP:
                anchor  = mem.recent_messages[:2]
                rolling = mem.recent_messages[-(RECENT_MESSAGES_KEEP - 2):]
                mem.recent_messages = anchor + rolling

            if usage:
                for k, v in usage.items():
                    if k in mem.usage:
                        mem.usage[k] = v
                mem.usage["total"] = sum(
                    v for k, v in mem.usage.items() if k != "total"
                )

            yield "🧠 記憶已記錄。"
            return mem

    def delete_thread(self, session_key: str, llm=None) -> bool:
        """
        Delete a session file from the active threads, moving it to the archived directory
        so the raw JSON is preserved for historical debugging/records.
        """
        import shutil
        path = self._thread_path(session_key)
        if not path.exists():
            return False

        with self._lock_session(session_key) as mem:
            mem.is_deleted = True

        try:
            archive_path = ARCHIVE_DIR / f"{path.stem}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            shutil.move(str(path), str(archive_path))
            logger.info("Archived thread file to %s", archive_path.name)
        except Exception as e:
            logger.warning("Could not archive thread file %s: %s", path.name, e)
            
        return True

    def list_threads(self, include_deleted: bool = False) -> list[ThreadMemory]:
        threads: list[ThreadMemory] = []
        if not self.data_dir.exists():
            return []
        for p in self.data_dir.glob("thread_*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if not data or "session_key" not in data:
                    continue
                valid = {k: v for k, v in data.items() if k in ThreadMemory.__dataclass_fields__}
                t = ThreadMemory(**valid)
                if not include_deleted and t.is_deleted:
                    continue
                threads.append(t)
            except Exception:
                continue
        threads.sort(key=lambda t: t.last_updated_at, reverse=True)
        return threads

    def list_session_keys(self, include_deleted: bool = False) -> list[str]:
        return [t.session_key for t in self.list_threads(include_deleted=include_deleted)]

    def get_next_session_name(self) -> str:
        """Returns a unique UUID-based session key (e.g. sess_3f8a1b2c)."""
        return f"sess_{uuid.uuid4().hex[:8]}"


# ── Memory prompt formatter (used by SkillMiddleware & UIHandler) ──────────────

def format_memory_for_prompt(mem: "ThreadMemory") -> str:
    """
    Format global profile into a system prompt block.
    Memory indexing is fully automatic:
      - Sessions are indexed when deleted (archive_and_summarize_session)
      - Files in results/ and tmp/ are indexed by FileWatcher daemon
    AI only needs to: SEARCH memory and UPDATE profile.
    """
    parts: list[str] = []
    all_keys: list[str] = []

    profile      = load_global_profile()
    user_profile = profile.get("user_profile", {})
    preferences  = profile.get("preferences",  {})
    agent_rules  = profile.get("agent_rules",  {})

    if user_profile:
        parts.append("User Profile: " + ", ".join(f"{k}: {v}" for k, v in user_profile.items()))
        all_keys.extend(f"Profile: {k}" for k in user_profile)

    if preferences:
        parts.append("Preferences: " + ", ".join(f"{k}: {v}" for k, v in preferences.items()))
        all_keys.extend(f"Pref: {k}" for k in preferences)

    if agent_rules:
        rules_text = "\n".join(f"  - {k}: {v}" for k, v in agent_rules.items())
        parts.append(f"Agent Action Guidelines:\n{rules_text}")
        all_keys.extend(f"Rule: {k}" for k in agent_rules)

    keys_info = (
        "\n\n📋 REGISTERED MEMORY LABELS (use exact keys when upserting):\n- "
        + "\n- ".join(all_keys)
        if all_keys else "\n\n(No memory labels yet.)"
    )

    content = "\n\n".join(parts) if parts else "(No structured memory recorded yet.)"

    persona_block = (
        "\n\n"
        "================================================================\n"
        "📌 USER PERSONA & PREFERENCES (Known Context)\n"
        "================================================================\n"
        + content
        + keys_info
        + "\n================================================================\n"
    )

    action_rules = """
================================================================
🧠 MEMORY ACTIONS
================================================================
STEP 1 — SEARCH (returns summary + full turn context):
  run_cli_command("python core/mem_scripts/mem_search.py '<query>' --limit 5")
  run_cli_command("python core/mem_scripts/mem_search.py '<query>' --type conv --min-score 0.7")
  run_cli_command("python core/mem_scripts/mem_search.py '<query>' --type doc --min-score 0.7")

STEP 2 — READ DOCUMENTS (only when search summary is not enough):
  run_cli_command("cat results/some_file.txt")

UPDATE PROFILE (user preferences / rules / facts):
  run_cli_command("python core/mem_scripts/mem_upsert_profile.py <key> '<value>' --type <preference|profile|agent_rules>")

NOTE: Conversation turns are indexed automatically in real-time.
================================================================"""

    return persona_block + action_rules
