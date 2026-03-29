import os
import sys
import subprocess
import shlex
import uuid
import yaml
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from langchain.tools import tool
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage

# --- Domain Layer ---
@dataclass
class Skill:
    name: str
    description: str
    path: str

class SkillRepository(ABC):
    @abstractmethod
    def get_all_skills(self) -> List[Skill]: pass
    @abstractmethod
    def get_skill_overview(self, skill_name: str) -> Optional[str]: pass
    @abstractmethod
    def get_skill_details(self, skill_name: str, file_path: str) -> Optional[str]: pass
    @abstractmethod
    def list_skill_files(self, skill_name: str) -> List[str]: pass

# --- Infrastructure Layer ---
class FileSystemSkillRepository(SkillRepository):
    def __init__(self, root_directory: str):
        self.root_directory = os.path.abspath(root_directory)
        self._cache_skills: Optional[List[Skill]] = None
        self._cache_file_contents: dict = {}

    def _read_file_cached(self, abs_path: str) -> Optional[str]:
        if abs_path not in self._cache_file_contents:
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    self._cache_file_contents[abs_path] = f.read()
            except Exception: return None
        return self._cache_file_contents[abs_path]

    def get_all_skills(self) -> List[Skill]:
        if self._cache_skills is not None: return self._cache_skills
        skills = []
        if not os.path.exists(self.root_directory): return []
        for item in os.listdir(self.root_directory):
            item_path = os.path.join(self.root_directory, item)
            if os.path.isdir(item_path):
                skill_md_path = os.path.join(item_path, "SKILL.md")
                if os.path.exists(skill_md_path):
                    name, desc = self._parse_frontmatter(skill_md_path)
                    name = name or item
                    skills.append(Skill(name=name, description=desc or "No description.", path=item_path))
        self._cache_skills = skills
        return skills

    def _parse_frontmatter(self, file_path: str) -> tuple[Optional[str], Optional[str]]:
        content = self._read_file_cached(os.path.abspath(file_path))
        if content and content.startswith('---'):
            try:
                end_idx = content.find('---', 3)
                if end_idx != -1:
                    data = yaml.safe_load(content[3:end_idx])
                    return data.get('name'), data.get('description')
            except Exception: pass
        return None, None

    def _find_skill_by_name(self, skill_name: str) -> Optional[Skill]:
        for s in self.get_all_skills():
            if s.name == skill_name or os.path.basename(s.path) == skill_name:
                return s
        return None

    def get_skill_overview(self, skill_name: str) -> Optional[str]:
        skill = self._find_skill_by_name(skill_name)
        return self._read_file_cached(os.path.join(skill.path, "SKILL.md")) if skill else None

    def get_skill_details(self, skill_name: str, file_path: str) -> Optional[str]:
        skill = self._find_skill_by_name(skill_name)
        if not skill: return None
        target_path = os.path.abspath(os.path.join(skill.path, file_path))
        if not target_path.startswith(os.path.abspath(skill.path)): return "Error: Access denied."
        return self._read_file_cached(target_path) or f"Error reading {file_path}"

    def list_skill_files(self, skill_name: str) -> List[str]:
        skill = self._find_skill_by_name(skill_name)
        if not skill: return []
        files_list = []
        for root, _, files in os.walk(skill.path):
            for f in files:
                files_list.append(os.path.relpath(os.path.join(root, f), skill.path))
        return sorted(files_list)

# --- Tool Definitions ---
# 這些工具在模組級別定義，以便 SkillMiddleware 可以訪問它們
# 我們使用工廠函數來創建綁定到特定 repo 的工具

def _create_tools_for_repo(repo: SkillRepository):
    """為給定的 repo 創建工具實例。"""
    
    @tool
    def load_skill_overview(skill_name: str) -> str:
        """Load the overview of a skill to understand its capabilities and usage."""
        content = repo.get_skill_overview(skill_name)
        if content:
            files = repo.list_skill_files(skill_name)
            extra = [f for f in files if f.upper() != "SKILL.MD"]
            return f"Loaded overview for {skill_name}\n\n{content}\n\nAdditional files: {', '.join(extra)}"
        return f"Skill '{skill_name}' not found."

    @tool
    def read_skill_file(skill_name: str, file_path: str) -> str:
        """Read a specific file from a skill's directory."""
        return repo.get_skill_details(skill_name, file_path)

    def _run_process_realtime(cmd, cwd, env):
        """Helper to run a subprocess with real-time terminal output and buffered capture."""
        full_output = []
        # Support for UI real-time display via a shared log file
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "logs"))
        os.makedirs(log_dir, exist_ok=True)
        live_log_path = os.path.join(log_dir, "live.log")
        
        try:
            msg = f"\n[⚡ Real-time Execution] { ' '.join(cmd) if isinstance(cmd, list) else cmd }"
            print(msg, flush=True)
            print("-" * 60, flush=True)
            
            # Start fresh log for this command
            with open(live_log_path, "w", encoding="utf-8") as f:
                f.write(f"{msg}\n" + "-"*60 + "\n")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=isinstance(cmd, str),
                cwd=cwd,
                env=env,
                encoding="utf-8",
                errors="replace",
                text=True,
                bufsize=1
            )

            # Read line by line from the pipe
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    # 1. Print to terminal
                    print(line, end="", flush=True)
                    # 2. Append to capturing buffer
                    full_output.append(line)
                    # 3. Append to live log for UI polling
                    try:
                        with open(live_log_path, "a", encoding="utf-8") as f:
                            f.write(line)
                    except Exception: pass
            
            print("-" * 60, flush=True)
            output_str = "".join(full_output)
            exit_code = process.poll()

            # --- Smart Truncation Logic (Model-Aware) ---
            max_model_len = int(os.environ.get("MAX_MODEL_LEN", 4096))
            chars_per_token = float(os.environ.get("CHARS_PER_TOKEN", 1.8))
            
            # Allow one tool output to occupy up to 15% of the total context (at least 3000 chars, at most 30000)
            max_allowed_chars = int(max_model_len * chars_per_token * 0.15)
            max_allowed_chars = max(3000, min(30000, max_allowed_chars))

            if len(output_str) > max_allowed_chars:
                # 1. Save full log to file for recovery/reference
                log_filename = f"tool_output_{uuid.uuid4().hex[:8]}.log"
                log_path = os.path.join(log_dir, log_filename)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(output_str)
                
                # 2. Perform Head-Tail Average Truncation
                keep_side = max_allowed_chars // 2
                head = output_str[:keep_side]
                tail = output_str[-keep_side:]
                trunc_msg = f"\n\n... [TRUNCATED DUE TO TOKEN LIMIT: {len(output_str)} chars total. Full log saved to: {log_path}] ...\n\n"
                
                output_str = head + trunc_msg + tail

            return output_str, exit_code
        except Exception as e:
            err_msg = f"Process failure: {e}"
            with open(live_log_path, "a", encoding="utf-8") as f: f.write(f"\n❌ {err_msg}")
            return err_msg, -1

    # Auto-indexing of results/ and tmp/ is now handled by core/file_watcher.py
    # (FileWatcher daemon started in app.py — no snapshot diffing needed here)

    @tool
    def execute_script(skill_name: str, script_path: str, script_args: str = "") -> str:
        """Execute a Python script inside a skill's directory with real-time feedback (Check terminal or Live Log)."""
        skill = repo._find_skill_by_name(skill_name)
        if not skill: return "Skill not found."
        abs_script = os.path.normpath(os.path.join(skill.path, script_path))
        if not abs_script.startswith(os.path.abspath(skill.path)) or not os.path.isfile(abs_script):
            return "Invalid script path."

        cmd = [sys.executable, "-u", abs_script] + shlex.split(script_args)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        output, code = _run_process_realtime(cmd, skill.path, env)
        return f"[stdout/stderr]\n{output}\n[exit code] {code}"

    @tool
    def run_cli_command(command: str, working_directory: str = "") -> str:
        """Execute a CLI command with real-time terminal feedback."""
        cwd = working_directory or os.getcwd()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        output, code = _run_process_realtime(command, cwd, env)
        return f"[stdout/stderr]\n{output}\n[exit code] {code}"

    @tool
    def run_python_code(code: str, working_directory: str = "") -> str:
        """Execute Python code with real-time feedback via temporary file."""
        import tempfile
        cwd = working_directory or os.getcwd()
        tmp_path = os.path.join(tempfile.gettempdir(), f"tmp_{uuid.uuid4().hex}.py")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f: f.write(code)
            cmd = [sys.executable, "-u", tmp_path]
            output, exit_code = _run_process_realtime(cmd, cwd, env)
            return f"[stdout/stderr]\n{output}\n[exit code] {exit_code}"
        except Exception as e: return f"Error: {e}"
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

    return [load_skill_overview, read_skill_file, execute_script, run_cli_command, run_python_code]


def create_skill_tools(repo: SkillRepository, memory_store: Any = None, session_key: str = "main"):
    """
    Returns a dictionary of tools, including the core memory tool if memory_store is provided.
    """
    tools = _create_tools_for_repo(repo)
    registry = {
        "load_skill_overview": tools[0],
        "read_skill_file": tools[1],
        "execute_script": tools[2],
        "run_cli_command": tools[3],
        "run_python_code": tools[4],
    }
    
    return registry

class SkillMiddleware(AgentMiddleware):
    # The tools available to the agent
    tools = []
    
    def __init__(self, repo: SkillRepository, memory_store: Any = None, session_key: str = "main"):
        self.repo = repo
        self.memory_store = memory_store
        self.session_key = session_key
        self.refresh_skills_prompt()
        
        # Create tools including the memory tool
        tool_dict = create_skill_tools(repo, memory_store, session_key)
        self.tools = list(tool_dict.values())

    def refresh_skills_prompt(self):
        import re
        skills = self.repo.get_all_skills()
        skills_list = []
        for s in skills:
            desc = s.description.replace('\n', ' ').strip()
            skill_entry = f'  Skill key "{s.name}"\n  Purpose: {desc}'
            
            # Dynamically include available tools if present in SKILL.md
            overview = self.repo.get_skill_overview(s.name)
            if overview:
                # Look for "## Currently Available MCP Tools" or similar sections
                tools_match = re.search(r"## Currently Available .*?Tools(.*?)(?=\n##|$)", overview, re.DOTALL)
                if tools_match:
                    tools_text = tools_match.group(1).strip()
                    # Remove markdown alerts like > [!IMPORTANT] for the system prompt
                    tools_text = re.sub(r"> \[!.*?\]\n", "", tools_text)
                    tools_text = re.sub(r">.*?\n", "", tools_text)
                    if tools_text:
                        # Indent it for better readability in system prompt
                        indented_tools = "\n".join([f"    {line}" for line in tools_text.split("\n") if line.strip()])
                        skill_entry += f"\n  Available Tools:\n{indented_tools}"
            
            skills_list.append(skill_entry)
        self.skills_prompt = "\n\n".join(skills_list) if skills_list else "No skills available."

    def get_skills_addendum(self) -> str:
        """Returns the dynamic skills addendum containing the protocol and library."""
        return (
            "\n\n"
            "================================================================\n"
            "⚠️ GENERAL SKILL-FIRST PROTOCOL (DECOUPLED)\n"
            "================================================================\n"
            "The system is modular. Capabilities are provided via 'Skills' listed below.\n\n"
            "MANDATORY SEARCH-FIRST RULE:\n"
            "  - BEFORE writing code or running CLI commands, scan the 'KNOWLEDGE SKILL LIBRARY'.\n"
            "  - If any skill's 'Purpose' matches your current intent (e.g. Training, ASR, PDF):\n"
            "    1. You MUST call load_skill_overview(skill_name) to inspect its scripts.\n"
            "    2. You MUST prioritize `execute_script(...)` over writing raw logic manually.\n"
            "    3. VIOLATION: Using raw CLI/Python for tasks covered by a Skill results in \n"
            "       misconfigured environments and log overflows.\n\n"
            "TOOL PREFERENCE ORDER:\n"
            "  Priority 1: Specialized Skill Scripts (`execute_script`)\n"
            "  Priority 2: Common CLI/Python (`run_cli_command`, `run_python_code`)\n"
            "              (Use ONLY for one-off system tasks or if NO skill matches.)\n"
            "\n"
            "FILE PATH ANCHOR RULE (CRITICAL):\n"
            "  1. NO SCATTERING: NEVER save files directly in the root or skill folders.\n"
            "  2. PERSISTENT: All experiment results/logs MUST go to the absolute path \n"
            "     stored in `os.environ['RESULTS_DIR']` (or `./results/` as fallback).\n"
            "  3. TEMPORARY: All temporary files MUST go to `os.environ['TMP_DIR']`.\n"
            "  4. SKILL HINT: When calling `execute_script`, the CWD is the skill root.\n"
            "     Check the code or load_skill_overview to see if it reads PROJECT_ROOT.\n"
            "================================================================\n"
            "KNOWLEDGE SKILL LIBRARY (DYNAMIC LOOKUP KEYS)\n"
            "================================================================\n"
            f"{self.skills_prompt}\n\n"
            "----------------------------------------------------------------\n"
            "HOW TO USE A SKILL (mandatory workflow):\n"
            "  Step 1: Compare task with 'Purpose' descriptions above.\n"
            "  Step 2: Call load_skill_overview(skill_name=\"<key>\") to see dedicated scripts.\n"
            "  Step 3: Call execute_script(skill_name, script_path, args) for high-level tasks.\n\n"
            "----------------------------------------------------------------\n"
            "CALLABLE TOOLS (all tools you may invoke directly):\n"
            "  - load_skill_overview(skill_name: str) -> str\n"
            "  - read_skill_file(skill_name: str, file_path: str) -> str\n"
            "  - execute_script(skill_name: str, script_path: str, script_args: str = \"\") -> str\n"
            "      Run a Python script from the skill's scripts/ folder.\n"
            "      Example: execute_script('pdf', 'scripts/convert_pdf_to_images.py', 'doc.pdf ../../results/out')\n"
            "  - run_cli_command(command: str, working_directory: str = \"\") -> str\n"
            "      Run any shell/CLI command (pdftotext, pip install, python -c ..., etc.)\n"
            "      Example: run_cli_command('pdftotext input.pdf output.txt', './results/')\n"
            "  - run_python_code(code: str, working_directory: str = \"\") -> str\n"
            "      Write Python code you compose yourself and execute it immediately.\n"
            "      Use this when you want to implement something based on skill examples\n"
            "      (e.g., from SKILL.md code blocks) without needing a pre-existing script.\n"
            "      Example: run_python_code(\"from reportlab.pdfgen import canvas\\nc = canvas.Canvas('out.pdf')\\n...\", 'C:/output')\n\n"
            "  Note: On Windows, use forward slashes (/) for paths to avoid escape issues.\n"
            "  Example: 'C:/Users/name/document.pdf' instead of 'C:\\Users\\name\\document.pdf'\n"
            "================================================================"
        )

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        # 1. Build Skill Library Documentation
        skills_addendum = self.get_skills_addendum()
        
        # 2. Add Structured Memory if available (Delegated to memory skill logic)
        memory_addendum = ""
        if self.memory_store:
            try:
                from core.memory import format_memory_for_prompt
                mem = self.memory_store.load_thread(self.session_key)
                memory_addendum = format_memory_for_prompt(mem)
            except Exception:
                memory_addendum = "\n\n(Memory system unavailable.)\n"
            
        import datetime
        current_time_info = (
            f"\n\n[System Info]\n"
            f"Current Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Day of the Week: {datetime.datetime.now().strftime('%A')}\n"
        )
            
        orig = request.system_message.content
        new_content = orig + current_time_info + memory_addendum + skills_addendum
        
        return handler(request.override(system_message=SystemMessage(content=new_content)))
