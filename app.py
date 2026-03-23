import os
import sys
from langgraph.checkpoint.memory import InMemorySaver
from core.memory import MemoryStore
from core.skills import FileSystemSkillRepository, create_skill_tools
from ui.handlers import UIHandler
from ui.interface import create_ui

# Initialize Core Components
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(PROJECT_ROOT, "skills")

memory_store = MemoryStore()
skill_repo = FileSystemSkillRepository(SKILLS_DIR)
tool_registry = create_skill_tools(skill_repo)
global_checkpointer = InMemorySaver()

# Initialize UI Handler
handler = UIHandler(memory_store, skill_repo, tool_registry, global_checkpointer)

# Create and Launch UI
demo = create_ui(handler)

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "memory"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    tmp_dir = os.path.join(PROJECT_ROOT, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    # 讓所有 skills scripts 都能透過環境變數取得專案根目錄與標準路徑
    os.environ["PROJECT_ROOT"] = PROJECT_ROOT
    os.environ["TMPDIR"] = tmp_dir
    os.environ["TEMP"] = tmp_dir
    os.environ["TMP"] = tmp_dir
    
    # Launch Gradio
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
