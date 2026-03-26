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

    # Archive state machine: null | "pending" | "summarizing" | "done"
    archive_state:     Optional[str] = None
    archive_started_at: Optional[str] = None
    archived_file:     Optional[str] = None        # relative path after rename
    archived_chunks:   list = field(default_factory=list)
    # Each entry: {"chunk_index": int, "qdrant_point_id": str, "turn_range": [start, end]}

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
    logger.info("Created Qdrant collection '%s'", collection)


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


def _extract_tags(text: str) -> list[str]:
    """
    Extract keyword tags from raw text without an LLM.
    Strategy:
      1. English capitalized words / acronyms (e.g. GPU, NVIDIA, RTX)
      2. English technical terms 4+ chars (e.g. social, media)
      3. CJK noun-ish phrases 2-4 chars (simple heuristic)
    Returns up to 5 unique tags, shortest first.
    """
    tags: list[str] = []
    seen: set[str] = set()

    # English: ALL-CAPS acronyms and Title-Case words 2+ chars
    for m in re.finditer(r"\b([A-Z]{2,}|[A-Z][a-z]{3,})\b", text):
        w = m.group(1)
        if w.lower() not in {"user", "assistant", "think", "true", "false"}:
            if w not in seen:
                seen.add(w)
                tags.append(w)

    # CJK: 2-4 char clusters that appear more than once OR follow keywords
    cjk_words = re.findall(r"[\u4e00-\u9fff]{2,4}", text)
    freq: dict[str, int] = {}
    for w in cjk_words:
        freq[w] = freq.get(w, 0) + 1
    for w, cnt in sorted(freq.items(), key=lambda x: -x[1]):
        if cnt >= 2 and w not in seen:
            seen.add(w)
            tags.append(w)

    return tags[:5]


def _extractive_summarize(turns: list[dict], chunk_idx: int,
                           prev_summary: str) -> dict:
    """
    Build a summary purely from the conversation text — no LLM required.

    Summary structure:
      [prev context if any]
      User intents (first 120 chars each)
      → Assistant response preview (first 120 chars of first assistant msg)

    Score: based on total content length (more content = more important).
    Tags:  extracted from all text with _extract_tags().
    """
    user_msgs = [
        _strip_think(m["content"])
        for m in turns if m.get("role") == "user"
    ]
    asst_msgs = [
        _strip_think(m["content"])
        for m in turns if m.get("role") == "assistant"
    ]

    # Build summary lines
    lines: list[str] = []
    if prev_summary:
        lines.append(f"【承接】{prev_summary[:80]}")

    # User intents
    user_text = " / ".join(u[:120] for u in user_msgs if u.strip())
    if user_text:
        lines.append(f"【提問】{user_text}")

    # First assistant response preview
    if asst_msgs:
        asst_preview = asst_msgs[0][:200].replace("\n", " ")
        lines.append(f"【回應】{asst_preview}")

    summary = "\n".join(lines) or f"Chunk {chunk_idx}"

    # Score: log-scaled content length, capped 1-9
    total_chars = sum(len(m.get("content", "")) for m in turns)
    import math
    score = min(9.0, max(1.0, round(1 + math.log10(max(total_chars, 10)) * 2, 1)))

    # Tags from all text
    all_text = " ".join(
        _strip_think(m.get("content", "")) for m in turns
    )
    tags = _extract_tags(all_text)

    logger.info(
        "Chunk %d extractive summary (%d chars) | tags=%s | score=%.1f",
        chunk_idx, total_chars, tags, score,
    )
    return {"summary": summary, "tags": tags, "score": score}


# Keep the old name as an alias so call sites don't change.
# `llm` param is accepted but ignored — kept for API compatibility.
def _summarize_chunk_with_llm(turns: list[dict], chunk_idx: int,
                               prev_summary: str, llm=None) -> dict:
    return _extractive_summarize(turns, chunk_idx, prev_summary)


def _build_chunks(messages: list[dict]) -> list[list[dict]]:
    """
    Split messages (pairs of user/assistant) into overlapping chunks.
    Each chunk contains CHUNK_SIZE turns, with CHUNK_OVERLAP turns shared
    with the next chunk to preserve semantic continuity.
    """
    # Work in turn pairs
    turns = []
    for i in range(0, len(messages) - 1, 2):
        turns.append(messages[i:i + 2])  # [user_msg, assistant_msg]

    if not turns:
        return []

    chunks = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    start = 0
    while start < len(turns):
        end = min(start + CHUNK_SIZE, len(turns))
        chunk_turns = [msg for pair in turns[start:end] for msg in pair]
        chunks.append(chunk_turns)
        if end >= len(turns):
            break
        start += step
    return chunks


# ── Archive & Summarize ────────────────────────────────────────────────────────



def archive_and_summarize_session(
    session_key: str,
    llm,
    memory_store: "MemoryStore",
) -> None:
    """
    Full archive pipeline triggered when a session is deleted:

    1. Mark archive_state = "pending" (crash-safe checkpoint)
    2. Rename thread JSON to archived/{date}_{slug}.json
    3. Split messages into overlapping chunks
    4. Summarize each chunk with LLM (resume from archived_chunks on crash)
    5. Upsert each chunk to Qdrant conv_YYYY_MM
    6. Mark archive_state = "done"

    Runs in a background thread — never blocks UI.
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()

    # ── Step 0: Load current thread ─────────────────────────────────────────
    mem = memory_store.load_thread(session_key)
    messages = mem.recent_messages

    if len(messages) < 2:
        logger.info("Session '%s' has < 1 turn, skipping archive summarization.", session_key)
        return

    # ── Step 1: Mark pending (first crash-safe checkpoint) ──────────────────
    if mem.archive_state not in ("pending", "summarizing", "done"):
        mem.archive_state = "pending"
        mem.archive_started_at = datetime.now(timezone.utc).isoformat()
        memory_store.save_thread(mem)

    if mem.archive_state == "done":
        logger.info("Session '%s' already archived, skipping.", session_key)
        return

    # ── Step 2: Determine archive filename ──────────────────────────────────
    if mem.archived_file is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Simple, predictable filename: date + session_key (no LLM, no content-based name)
        archive_filename = f"{date_str}_{session_key}.json"
        dest = ARCHIVE_DIR / archive_filename
        # Handle the extremely rare case of a collision on the same key+day
        counter = 1
        while dest.exists():
            archive_filename = f"{date_str}_{session_key}_{counter}.json"
            dest = ARCHIVE_DIR / archive_filename
            counter += 1

        # Write the JSON to archive location FIRST (keep original until done)
        thread_src = memory_store._thread_path(session_key)
        rel_archived = os.path.relpath(str(dest), str(BASE_DIR)).replace("\\", "/")
        mem.archived_file = rel_archived
        mem.archive_state = "summarizing"
        memory_store.save_thread(mem)

        # Atomic copy to archive dir
        import shutil
        shutil.copy2(str(thread_src), str(dest))
        logger.info("Copied thread JSON to archive: %s", dest)
    else:
        # Resume after crash: archived_file already set
        dest = BASE_DIR / mem.archived_file.replace("/", os.sep)

    # ── Step 3: Chunk the messages ───────────────────────────────────────────
    chunks = _build_chunks(messages)
    already_done = {c["chunk_index"] for c in mem.archived_chunks}
    collection = datetime.now(timezone.utc).strftime("conv_%Y_%m")
    session_created_at = mem.created_at
    rel_archived = mem.archived_file

    _ensure_conv_collection(collection, cfg)

    # Build prev_chunk lookup from already-done chunks
    prev_summaries: dict[int, str] = {}
    for done in mem.archived_chunks:
        prev_summaries[done["chunk_index"]] = done.get("summary", "")

    # ── Step 4 & 5: Summarize + Upsert each chunk ───────────────────────────
    for chunk_idx, chunk_msgs in enumerate(chunks):
        if chunk_idx in already_done:
            logger.info("Chunk %d already indexed, skipping.", chunk_idx)
            continue

        # Turn range (1-based)
        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        turn_start = chunk_idx * step + 1
        turn_end   = min(turn_start + CHUNK_SIZE - 1, len(messages) // 2)

        # Get timestamps from actual messages
        ts_start = next(
            (m.get("ts", "") for m in chunk_msgs if m.get("role") == "user"), ""
        )
        ts_end = next(
            (m.get("ts", "") for m in reversed(chunk_msgs) if m.get("role") == "assistant"), ""
        )

        prev_summary = prev_summaries.get(chunk_idx - 1, "")
        result = _summarize_chunk_with_llm(chunk_msgs, chunk_idx, prev_summary, llm)
        prev_summaries[chunk_idx] = result["summary"]

        point_id = str(uuid.uuid4())
        now_unix  = int(time.time())
        iso_now   = datetime.now(timezone.utc).isoformat()

        payload = {
            "session_key":        session_key,
            "archived_file":      rel_archived,
            "chunk_index":        chunk_idx,
            "turn_range":         [turn_start, turn_end],
            "summary":            result["summary"],
            "prev_chunk_summary": prev_summary,
            "tags":               result["tags"],
            "importance_score":   result["score"],
            "is_deleted":         False,
            # ── Time fields ─────────────────────────────────────────────────
            "session_created_at":  session_created_at,       # when session started
            "session_deleted_at":  datetime.now(timezone.utc).isoformat(),  # when archived
            "chunk_turn_start_ts": ts_start,                 # first msg ts in this chunk
            "chunk_turn_end_ts":   ts_end,                   # last msg ts in this chunk
            "indexed_at_unix":     now_unix,                 # ← integer, for range filter
            "indexed_at_iso":      iso_now,                  # ← display only
        }

        try:
            embed_text = f"[conversation] {result['summary']}"
            vector = _get_embedding(embed_text, cfg)
            _upsert_conv_point(collection, point_id, vector, payload, cfg)

            # Record completion IMMEDIATELY after each successful upsert
            mem.archived_chunks.append({
                "chunk_index":    chunk_idx,
                "qdrant_point_id": point_id,
                "turn_range":     [turn_start, turn_end],
                "summary":        result["summary"],
            })
            memory_store.save_thread(mem)  # atomic write checkpoint
            logger.info("Chunk %d → %s [%s]", chunk_idx, collection, point_id)

        except Exception as e:
            logger.error("Failed to upsert chunk %d for '%s': %s", chunk_idx, session_key, e)
            # Do NOT abort — try remaining chunks

    # ── Step 6: Mark done ───────────────────────────────────────────────────
    mem.archive_state = "done"
    memory_store.save_thread(mem)
    logger.info("Archive complete for session '%s' → %s", session_key, rel_archived)

    # Remove original thread file from threads/ — archived/ is the source of truth
    thread_file = DATA_DIR / f"thread_{session_key}.json"
    if thread_file.exists():
        try:
            thread_file.unlink()
            logger.info("Removed thread file after archiving: %s", thread_file.name)
        except Exception as e:
            logger.warning("Could not remove thread file %s: %s", thread_file.name, e)


# ── Startup recovery ───────────────────────────────────────────────────────────

def recover_incomplete_archives(memory_store: "MemoryStore", llm) -> None:
    """
    Called once at app startup.
    Resumes any archive that was interrupted by a crash.
    """
    for path in DATA_DIR.glob("thread_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        state = data.get("archive_state")
        if state in ("pending", "summarizing"):
            session_key = data.get("session_key", path.stem.removeprefix("thread_"))
            logger.info("Recovery: resuming archive for session '%s' (state=%s)", session_key, state)
            threading.Thread(
                target=archive_and_summarize_session,
                args=(session_key, llm, memory_store),
                daemon=True,
                name=f"recovery-{session_key}",
            ).start()


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
                    usage: Optional[dict] = None) -> Generator[str, None, ThreadMemory]:
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
        Delete a session:
        - Empty session (no messages) -> immediately remove the file, no archive needed.
        - Session with content        -> archive in background, file removed after archive.
        """
        path = self._thread_path(session_key)
        if not path.exists():
            return False

        with self._lock_session(session_key) as mem:
            has_content = len(mem.recent_messages) > 0 or mem.turn_count > 0
            mem.is_deleted = True

        if not has_content:
            # Empty session -- just delete the file immediately
            try:
                path.unlink()
                logger.info("Deleted empty thread file: %s", path.name)
            except Exception as e:
                logger.warning("Could not delete thread file %s: %s", path.name, e)
            return True

        # Session has content -- archive in background (file removed after archive completes)
        if llm is not None:
            t = threading.Thread(
                target=archive_and_summarize_session,
                args=(session_key, llm, self),
                daemon=True,
                name=f"archive-{session_key}",
            )
            t.start()
        else:
            # No LLM -- just delete the file (no archive possible)
            logger.warning(
                "delete_thread called without LLM -- session '%s' will not be archived.", session_key
            )
            try:
                path.unlink()
                logger.info("Deleted thread file (no LLM): %s", path.name)
            except Exception as e:
                logger.warning("Could not delete thread file %s: %s", path.name, e)
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
STEP 1 — SEARCH (returns summary + file pointer, lightweight):
  run_cli_command("python core/mem_scripts/mem_search.py '<query>' --limit 5")
  run_cli_command("python core/mem_scripts/mem_search.py '<query>' --type conv")
  run_cli_command("python core/mem_scripts/mem_search.py '<query>' --type doc")

STEP 2 — READ FULL CONTENT (only when summary is not enough):
  # Conversations: use the pointer from search result
  run_cli_command("python core/mem_scripts/mem_search.py read_chunk 'data/memory/archived/xxx.json' '[1,5]'")
  # Documents: read the file directly
  run_cli_command("cat results/some_file.txt")

UPDATE PROFILE (user preferences / rules / facts):
  run_cli_command("python core/mem_scripts/mem_upsert_profile.py <key> '<value>' --type <preference|profile|agent_rules>")

NOTE: Conversations and files are indexed automatically.
================================================================"""

    return persona_block + action_rules
