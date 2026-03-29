import os
from langchain_openai import ChatOpenAI
from core.memory import MemoryStore, recover_incomplete_archives

PROJECT_ROOT = "c:/Users/iaian/Desktop/vlm_skill"
os.chdir(PROJECT_ROOT)

memory_store = MemoryStore()

# Minimal LLM for recovery (using the same defaults as app.py)
_api_url  = os.environ.get("LLM_API_URL", "http://10.1.1.7:24131") # Use the port from logs
_model    = os.environ.get("LLM_MODEL", "qwen3:32b")
_base_url = _api_url if _api_url.endswith("/v1") else f"{_api_url}/v1"

_recovery_llm = ChatOpenAI(
    model=_model, api_key="EMPTY", base_url=_base_url, temperature=0
)

print("🚀 Starting manual recovery of incomplete archives...")
recover_incomplete_archives(memory_store, _recovery_llm)
print("✅ Recovery triggered. Wait a few seconds for background threads to work.")
import time
time.sleep(10)
print("🏁 Done.")
