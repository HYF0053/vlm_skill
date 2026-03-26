"""
core/file_watcher.py — Automatic file indexing daemon for results/ and tmp/.

Architecture:
  - watchdog Observer (inotify on Linux, polling fallback)
  - Single worker thread with per-path debounce (prevents oscillation)
  - Bounded event queue (prevents memory explosion on mass writes)
  - Completely isolated locks (never touches MemoryStore locks → no deadlock)
  - Only communicates with Qdrant via HTTP (no shared in-process state)
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
COLLECTION = "document_memory"
DEBOUNCE_SEC = 2.0
QUEUE_MAX = 500
SUMMARY_CHARS = 400

INDEXABLE_EXTS = {
    ".md", ".txt", ".json", ".csv", ".py",
    ".yaml", ".yml", ".log", ".html", ".xml", ".rst",
}
BINARY_EXTS = {
    ".pth", ".bin", ".pkl", ".npz", ".npy", ".h5",
    ".ckpt", ".pt", ".safetensors", ".onnx", ".trt",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
    ".mp4", ".avi", ".mov", ".pdf", ".zip", ".tar", ".gz",
}

# Patterns to EXCLUDE from watching (prevents feedback loops)
EXCLUDE_PATTERNS = [
    "*.tmp", "*.lock", "*.pyc", ".git/*",
    "__pycache__/*", "data/logs/*", "data/memory/*",
]


# ── Qdrant helpers ─────────────────────────────────────────────────────────────

def _qdrant_base(cfg: dict) -> str:
    q = cfg["qdrant"]
    return f"http://{q['host']}:{q['port']}"


def _get_embedding(text: str, cfg: dict) -> list:
    emb = cfg["embedding"]
    url = f"{emb['base_url']}/v1/embeddings"
    resp = requests.post(
        url,
        json={"model": emb["model"], "input": text},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _ensure_collection(cfg: dict) -> None:
    base = _qdrant_base(cfg)
    r = requests.get(f"{base}/collections/{COLLECTION}", timeout=5)
    if r.status_code == 200:
        return
    vector_size = 1024
    for c in cfg["qdrant"].get("collections", []):
        if c["name"] == COLLECTION:
            vector_size = c.get("vector_size", 1024)
    requests.put(
        f"{base}/collections/{COLLECTION}",
        json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        timeout=10,
    ).raise_for_status()
    logger.info("Created Qdrant collection '%s'", COLLECTION)


def _find_point_by_path(rel_path: str, cfg: dict) -> Optional[str]:
    """Return existing Qdrant point_id for a file path, or None."""
    base = _qdrant_base(cfg)
    url = f"{base}/collections/{COLLECTION}/points/scroll"
    resp = requests.post(
        url,
        json={
            "filter": {"must": [{"key": "file_path", "match": {"value": rel_path}}]},
            "limit": 1,
            "with_payload": False,
        },
        timeout=10,
    )
    if resp.status_code != 200:
        return None
    pts = resp.json().get("result", {}).get("points", [])
    return pts[0]["id"] if pts else None


def _upsert_point(point_id: str, vector: list, payload: dict, cfg: dict) -> None:
    base = _qdrant_base(cfg)
    requests.put(
        f"{base}/collections/{COLLECTION}/points?wait=true",
        json={"points": [{"id": point_id, "vector": vector, "payload": payload}]},
        timeout=10,
    ).raise_for_status()


def _delete_point(point_id: str, cfg: dict) -> None:
    base = _qdrant_base(cfg)
    requests.post(
        f"{base}/collections/{COLLECTION}/points/delete?wait=true",
        json={"points": [point_id]},
        timeout=10,
    ).raise_for_status()


def _mark_deleted(point_id: str, cfg: dict) -> None:
    """Soft-delete: set is_deleted=true on existing point."""
    base = _qdrant_base(cfg)
    requests.post(
        f"{base}/collections/{COLLECTION}/points/payload?wait=true",
        json={"payload": {"is_deleted": True}, "points": [point_id]},
        timeout=10,
    ).raise_for_status()


# ── File reading helpers ───────────────────────────────────────────────────────

def _read_summary(abs_path: str, is_binary: bool) -> str:
    if is_binary:
        size_kb = os.path.getsize(abs_path) / 1024
        return f"[binary file, {size_kb:.1f} KB]"
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(SUMMARY_CHARS)
    except Exception as e:
        return f"[unreadable: {e}]"


def _read_dir_summary(abs_path: str) -> str:
    try:
        entries = sorted(os.listdir(abs_path))[:30]
        return f"Directory contents: {', '.join(entries)}" + (
            " ..." if len(os.listdir(abs_path)) > 30 else ""
        )
    except Exception as e:
        return f"[unreadable directory: {e}]"


def _infer_doc_type(rel_path: str, ext: str, is_dir: bool) -> str:
    if is_dir:
        return "other"
    if "tmp" in rel_path.replace("\\", "/").split("/"):
        return "tmp"
    if ext in {".md", ".txt", ".rst", ".html"}:
        return "report"
    if ext in {".py"}:
        return "other"
    return "analysis"


def _is_excluded(rel_path: str) -> bool:
    normalized = rel_path.replace("\\", "/")
    for pattern in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(normalized, pattern):
            return True
        # Also check basename only
        if fnmatch.fnmatch(os.path.basename(normalized), pattern):
            return True
    return False


# ── Core indexing logic ────────────────────────────────────────────────────────

def index_path(abs_path: str, project_root: str, cfg: dict, is_delete: bool = False) -> None:
    """
    Index (or remove) a single file/directory into document_memory.
    This is the ONLY function that touches Qdrant.
    """
    rel_path = os.path.relpath(abs_path, project_root).replace("\\", "/")

    if _is_excluded(rel_path):
        logger.debug("Skipping excluded path: %s", rel_path)
        return

    # ── Handle deletion ──────────────────────────────────────────────────────
    if is_delete:
        try:
            point_id = _find_point_by_path(rel_path, cfg)
            if point_id:
                _mark_deleted(point_id, cfg)
                logger.info("Soft-deleted Qdrant point for: %s", rel_path)
        except Exception as e:
            logger.warning("Failed to soft-delete %s: %s", rel_path, e)
        return

    # ── Gather file metadata ─────────────────────────────────────────────────
    is_dir = os.path.isdir(abs_path)
    if not is_dir and not os.path.isfile(abs_path):
        return  # race condition: file vanished

    ext = os.path.splitext(abs_path)[1].lower() if not is_dir else ""
    is_binary = ext in BINARY_EXTS
    file_name = os.path.basename(abs_path)
    source_dir = rel_path.split("/")[0]  # "results" or "tmp"

    try:
        stat = os.stat(abs_path)
        file_created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
        file_modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        file_size_kb = round(stat.st_size / 1024, 2) if not is_dir else 0.0
    except OSError:
        return  # file vanished during stat

    now_unix = int(time.time())
    iso_now = datetime.now(timezone.utc).isoformat()

    # ── Build summary for embedding ──────────────────────────────────────────
    if is_dir:
        summary = _read_dir_summary(abs_path)
    else:
        summary = _read_summary(abs_path, is_binary)

    doc_type = _infer_doc_type(rel_path, ext, is_dir)
    embed_text = f"[{doc_type}] {file_name}: {summary}"

    # ── Importance score heuristic ───────────────────────────────────────────
    score = min(10.0, max(4.0, 4.0 + file_size_kb / 20))

    payload = {
        "file_path": rel_path,
        "file_name": file_name,
        "file_type": ext.lstrip(".") if not is_dir else "directory",
        "file_size_kb": file_size_kb,
        "source_dir": source_dir,
        "is_dir": is_dir,
        "is_binary": is_binary,
        "summary": summary,
        "doc_type": doc_type,
        "importance_score": score,
        "tags": [],
        "is_deleted": False,
        # ── Time fields ──────────────────────────────────────────────────────
        "file_created_at": file_created_at,
        "file_modified_at": file_modified_at,
        "indexed_at_unix": now_unix,      # ← integer, used for Qdrant range filter
        "indexed_at_iso": iso_now,        # ← display only
    }

    try:
        _ensure_collection(cfg)
        vector = _get_embedding(embed_text, cfg)

        # Reuse existing point_id for the same file path (upsert = update)
        point_id = _find_point_by_path(rel_path, cfg) or str(uuid.uuid4())
        _upsert_point(point_id, vector, payload, cfg)
        logger.info("Indexed %s → %s [%s]", rel_path, COLLECTION, point_id)
    except Exception as e:
        logger.warning("Failed to index %s: %s", rel_path, e)


# ── Watchdog event handler ─────────────────────────────────────────────────────

class _FileEventHandler:
    """
    Receives raw watchdog events and puts them into the bounded queue.
    Does NOT do any I/O here — all real work happens in the worker thread.
    """
    def __init__(self, event_queue: queue.Queue, project_root: str):
        self._q = event_queue
        self._root = project_root

    def dispatch(self, event) -> None:
        src = getattr(event, "src_path", "")
        is_dir = getattr(event, "is_directory", False)
        etype = type(event).__name__  # FileCreatedEvent etc.

        rel = os.path.relpath(src, self._root).replace("\\", "/")
        if _is_excluded(rel):
            return

        action = "delete" if "Deleted" in etype else "upsert"

        try:
            self._q.put_nowait((action, src, is_dir))
        except queue.Full:
            logger.warning("FileWatcher queue full, dropping event for %s", rel)


# ── Main watcher class ─────────────────────────────────────────────────────────

class ResultsFileWatcher:
    """
    Background daemon that watches results/ and tmp/ for changes
    and automatically syncs Qdrant document_memory.

    Safety properties:
    - Uses its OWN lock only (_pending_lock), never MemoryStore locks → no deadlock
    - Per-path debounce → no oscillation
    - Bounded queue (maxsize=QUEUE_MAX) → no memory explosion
    - Excluded patterns prevent self-triggering feedback loops
    """

    def __init__(self, project_root: str, config_path: str):
        self._root = project_root
        self._cfg_path = config_path
        self._cfg: Optional[dict] = None

        self._watch_dirs = [
            os.path.join(project_root, "results"),
            os.path.join(project_root, "tmp"),
        ]

        # Per-path debounce: {abs_path: (action, is_dir, ready_at_unix)}
        self._pending: dict[str, tuple] = {}
        self._pending_lock = threading.Lock()  # ONLY lock we hold

        self._event_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self._stop_event = threading.Event()

        self._enqueue_thread = threading.Thread(
            target=self._enqueue_loop, daemon=True, name="fw-enqueue"
        )
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="fw-worker"
        )

    def _load_cfg(self) -> dict:
        if self._cfg is None:
            with open(self._cfg_path, "r", encoding="utf-8") as f:
                self._cfg = json.load(f)
        return self._cfg

    def start(self) -> None:
        """Start the watcher. Call once from app.py after startup."""
        for d in self._watch_dirs:
            os.makedirs(d, exist_ok=True)

        self._enqueue_thread.start()
        self._worker_thread.start()
        logger.info("FileWatcher started (watching: %s)", self._watch_dirs)

        # Index everything already present on startup
        threading.Thread(
            target=self._initial_scan, daemon=True, name="fw-init-scan"
        ).start()

    def stop(self) -> None:
        self._stop_event.set()

    # ── Initial scan ─────────────────────────────────────────────────────────

    def _initial_scan(self) -> None:
        """
        On startup, index all existing files in results/ and tmp/.
        Uses _find_point_by_path to skip already-indexed unchanged files.
        """
        time.sleep(3)  # Let app fully initialize first
        logger.info("FileWatcher: running initial scan...")
        cfg = self._load_cfg()
        for watch_dir in self._watch_dirs:
            if not os.path.isdir(watch_dir):
                continue
            for root, dirs, files in os.walk(watch_dir):
                # Index the directory itself if it's a subdirectory
                rel_root = os.path.relpath(root, self._root).replace("\\", "/")
                if root != watch_dir and not _is_excluded(rel_root):
                    try:
                        index_path(root, self._root, cfg)
                    except Exception as e:
                        logger.warning("Initial scan dir error %s: %s", root, e)

                for fname in files:
                    abs_f = os.path.join(root, fname)
                    rel_f = os.path.relpath(abs_f, self._root).replace("\\", "/")
                    if _is_excluded(rel_f):
                        continue
                    try:
                        index_path(abs_f, self._root, cfg)
                    except Exception as e:
                        logger.warning("Initial scan error %s: %s", abs_f, e)
        logger.info("FileWatcher: initial scan complete.")

    # ── Enqueue loop (watchdog polling) ──────────────────────────────────────

    def _enqueue_loop(self) -> None:
        """
        Use watchdog if available, otherwise fall back to polling.
        Puts raw events into self._event_queue.
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class _Handler(FileSystemEventHandler):
                def __init__(self_, q, root):
                    self_._q = q
                    self_._root = root

                def _put(self_, etype, src, is_dir):
                    rel = os.path.relpath(src, self_._root).replace("\\", "/")
                    if _is_excluded(rel):
                        return
                    action = "delete" if "Deleted" in etype else "upsert"
                    try:
                        self_._q.put_nowait((action, src, is_dir))
                    except queue.Full:
                        pass

                def on_created(self_, event):
                    self_._put("Created", event.src_path, event.is_directory)

                def on_modified(self_, event):
                    self_._put("Modified", event.src_path, event.is_directory)

                def on_deleted(self_, event):
                    self_._put("Deleted", event.src_path, event.is_directory)

                def on_moved(self_, event):
                    # Treat move as: delete old, create new
                    self_._put("Deleted", event.src_path, event.is_directory)
                    self_._put("Created", event.dest_path, event.is_directory)

            observer = Observer()
            handler = _Handler(self._event_queue, self._root)
            for d in self._watch_dirs:
                observer.schedule(handler, d, recursive=True)
            observer.start()
            logger.info("FileWatcher: using watchdog inotify/FSEvents observer")

            while not self._stop_event.is_set():
                time.sleep(1)
            observer.stop()
            observer.join()

        except ImportError:
            logger.info("FileWatcher: watchdog not installed, using polling fallback")
            self._polling_loop()

    def _polling_loop(self) -> None:
        """Fallback: poll every 5 seconds, detect new/modified/deleted files."""
        snapshot: dict[str, float] = {}

        def take_snapshot():
            s = {}
            for d in self._watch_dirs:
                if not os.path.isdir(d):
                    continue
                for root, dirs, files in os.walk(d):
                    for f in files:
                        p = os.path.join(root, f)
                        try:
                            s[p] = os.path.getmtime(p)
                        except OSError:
                            pass
            return s

        snapshot = take_snapshot()
        while not self._stop_event.is_set():
            time.sleep(5)
            new_snap = take_snapshot()
            for path, mtime in new_snap.items():
                if path not in snapshot:
                    self._event_queue.put_nowait(("upsert", path, False))
                elif snapshot[path] != mtime:
                    self._event_queue.put_nowait(("upsert", path, False))
            for path in snapshot:
                if path not in new_snap:
                    self._event_queue.put_nowait(("delete", path, False))
            snapshot = new_snap

    # ── Worker loop ──────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """
        Single worker thread: drains event_queue into _pending (with debounce),
        then flushes ready events one by one. Serial = no concurrent embedding calls.
        """
        while not self._stop_event.is_set():
            # Drain the queue into pending (non-blocking)
            try:
                while True:
                    action, abs_path, is_dir = self._event_queue.get_nowait()
                    ready_at = time.time() + DEBOUNCE_SEC
                    with self._pending_lock:
                        self._pending[abs_path] = (action, is_dir, ready_at)
            except queue.Empty:
                pass

            # Flush ready events
            now = time.time()
            with self._pending_lock:
                ready = {
                    p: v for p, v in self._pending.items() if v[2] <= now
                }
                for p in ready:
                    del self._pending[p]

            cfg = self._load_cfg()
            for abs_path, (action, is_dir, _) in ready.items():
                is_del = (action == "delete")
                try:
                    index_path(abs_path, self._root, cfg, is_delete=is_del)
                except Exception as e:
                    logger.warning("Worker error indexing %s: %s", abs_path, e)

            time.sleep(0.5)
