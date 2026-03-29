import logging
import os
import sys
from langgraph.checkpoint.memory import InMemorySaver
from core.memory import MemoryStore, recover_incomplete_archives
from core.skills import FileSystemSkillRepository, create_skill_tools
from core.file_watcher import ResultsFileWatcher
from ui.handlers import UIHandler
from ui.interface import create_ui

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR   = os.path.join(PROJECT_ROOT, "skills")
CONFIG_PATH  = os.path.join(PROJECT_ROOT, "config", "memo.json")

# Always anchor the process cwd to the project root.
# run_python_code / run_cli_command default to os.getcwd(), so without this
# any relative path like './results/' would break when the app is launched
# from a different directory.
os.chdir(PROJECT_ROOT)

# ── Core components ────────────────────────────────────────────────────────────
memory_store      = MemoryStore()
skill_repo        = FileSystemSkillRepository(SKILLS_DIR)
tool_registry     = create_skill_tools(skill_repo)
global_checkpointer = InMemorySaver()

# ── File Watcher daemon (results/ and tmp/ → Qdrant document_memory) ──────────
file_watcher = ResultsFileWatcher(project_root=PROJECT_ROOT, config_path=CONFIG_PATH)
file_watcher.start()
logger.info("FileWatcher daemon started.")

# ── UI Handler ─────────────────────────────────────────────────────────────────
handler = UIHandler(memory_store, skill_repo, tool_registry, global_checkpointer)

# ── Gradio UI ──────────────────────────────────────────────────────────────────
demo = create_ui(handler)

if __name__ == "__main__":
    # Ensure required directories exist
    for subdir in ["data/memory", "data/memory/threads", "data/memory/archived",
                   "data/logs", "results", "tmp"]:
        os.makedirs(os.path.join(PROJECT_ROOT, subdir), exist_ok=True)

    # Expose standard env vars for skill scripts
    os.environ["PROJECT_ROOT"] = PROJECT_ROOT
    results_dir = os.path.join(PROJECT_ROOT, "results")
    tmp_dir = os.path.join(PROJECT_ROOT, "tmp")
    os.environ["RESULTS_DIR"] = results_dir
    os.environ["TMPDIR"] = tmp_dir
    os.environ["TEMP"]   = tmp_dir
    os.environ["TMP"]    = tmp_dir

    # ── Startup recovery: resume any interrupted archive jobs ─────────────────
    # We need the LLM for summarization; create a minimal instance from env
    try:
        from langchain_openai import ChatOpenAI
        _api_url  = os.environ.get("LLM_API_URL", "http://10.1.1.7:11434")
        _model    = os.environ.get("LLM_MODEL", "qwen3:32b")
        _provider = os.environ.get("LLM_PROVIDER", "Ollama")
        _api_key  = "ollama" if _provider == "Ollama" else "EMPTY"
        _base_url = _api_url if _api_url.endswith("/v1") else f"{_api_url}/v1"
        _recovery_llm = ChatOpenAI(
            model=_model, api_key=_api_key, base_url=_base_url, temperature=0
        )
        recover_incomplete_archives(memory_store, _recovery_llm)
        logger.info("Startup recovery check complete.")
    except Exception as e:
        logger.warning("Recovery LLM init failed (non-fatal): %s", e)

    # ── Launch Gradio ──────────────────────────────────────────────────────────
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
