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

    @tool
    def execute_script(skill_name: str, script_path: str, script_args: str = "") -> str:
        """Execute a Python script located inside a skill's directory."""
        skill = repo._find_skill_by_name(skill_name)
        if not skill: return "Skill not found."
        abs_script = os.path.normpath(os.path.join(skill.path, script_path))
        if not abs_script.startswith(os.path.abspath(skill.path)) or not os.path.isfile(abs_script):
            return "Invalid script path."
        
        cmd = [sys.executable, abs_script] + shlex.split(script_args)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=600, cwd=skill.path, env=env)
            return f"[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}\n[exit code] {result.returncode}"
        except Exception as e: return f"Error: {e}"

    @tool
    def run_cli_command(command: str, working_directory: str = "") -> str:
        """Execute an arbitrary CLI / shell command on the host system."""
        cwd = working_directory or os.getcwd()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            result = subprocess.run(command, shell=True, capture_output=True, encoding="utf-8", errors="replace", timeout=600, cwd=cwd, env=env)
            return f"[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}\n[exit code] {result.returncode}"
        except Exception as e: return f"Error: {e}"

    @tool
    def run_python_code(code: str, working_directory: str = "") -> str:
        """Write a Python code string to a temporary file and execute it."""
        import tempfile
        cwd = working_directory or os.getcwd()
        tmp_path = os.path.join(tempfile.gettempdir(), f"tmp_{uuid.uuid4().hex}.py")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f: f.write(code)
            result = subprocess.run([sys.executable, tmp_path], capture_output=True, encoding="utf-8", errors="replace", timeout=600, cwd=cwd, env=env)
            return f"[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}\n[exit code] {result.returncode}"
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

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        # 1. Build Skill Library Documentation
        skills_addendum = (
            "\n\n"
            "================================================================\n"
            "KNOWLEDGE SKILL LIBRARY (READ-ONLY REFERENCE — NOT CALLABLE)\n"
            "================================================================\n"
            "The following are skill LOOKUP KEYS, NOT tools or functions.\n"
            "You MUST NOT call them directly. They do not exist as callable tools.\n\n"
            f"{self.skills_prompt}\n\n"
            "----------------------------------------------------------------\n"
            "HOW TO USE A SKILL (mandatory workflow):\n"
            "  Step 1: Call load_skill_overview(skill_name=\"<skill key above>\")\n"
            "  Step 2: Read the overview, then call read_skill_file(...) if needed\n"
            "  Step 3: Apply the skill instructions yourself, OR\n"
            "          run a helper script with execute_script(...), OR\n"
            "          run a CLI command with run_cli_command(...)\n\n"
            "CRITICAL BEHAVIORAL RULE:\n"
            "  - You are stateless between sessions. You CANNOT 'remember' anything just by saying 'I will remember this'.\n"
            "  - If the user states a rule, preference, habit, or project fact, you MUST IMMEDIATELY use `execute_script` to save it via the memory or rag skill.\n"
            "  - DO NOT say 'Okay, I will answer shortly' without ACTUALLY running the script to save the rule.\n\n"
            "CALLABLE TOOLS (all tools you may invoke directly):\n"
            "  - load_skill_overview(skill_name: str) -> str\n"
            "  - read_skill_file(skill_name: str, file_path: str) -> str\n"
            "  - execute_script(skill_name: str, script_path: str, script_args: str = \"\") -> str\n"
            "      Run a Python script from the skill's scripts/ folder.\n"
            "      Example: execute_script('pdf', 'scripts/convert_pdf_to_images.py', 'doc.pdf /tmp/out')\n"
            "  - run_cli_command(command: str, working_directory: str = \"\") -> str\n"
            "      Run any shell/CLI command (pdftotext, pip install, python -c ..., etc.)\n"
            "      Example: run_cli_command('pdftotext input.pdf output.txt', '/tmp')\n"
            "  - run_python_code(code: str, working_directory: str = \"\") -> str\n"
            "      Write Python code you compose yourself and execute it immediately.\n"
            "      Use this when you want to implement something based on skill examples\n"
            "      (e.g., from SKILL.md code blocks) without needing a pre-existing script.\n"
            "      Example: run_python_code(\"from reportlab.pdfgen import canvas\\nc = canvas.Canvas('out.pdf')\\n...\", 'C:/output')\n\n"
            "  Note: On Windows, use forward slashes (/) for paths to avoid escape issues.\n"
            "  Example: 'C:/Users/name/document.pdf' instead of 'C:\\Users\\name\\document.pdf'\n"
            "================================================================"
        )
        
        # 2. Add Structured Memory if available (Delegated to memory skill logic)
        memory_addendum = ""
        if self.memory_store:
            try:
                # Dynamically delegate formatting to the memory skill if it exists
                from skills.memory.lib.logic import format_memory_for_prompt
                mem = self.memory_store.load_thread(self.session_key)
                memory_addendum = format_memory_for_prompt(mem)
            except (ImportError, ModuleNotFoundError):
                # Fallback if the skill's logic is missing
                memory_addendum = "\n\n(Memory system logic unavailable.)\n"
            
        orig = request.system_message.content
        new_content = orig + memory_addendum + skills_addendum
        
        return handler(request.override(system_message=SystemMessage(content=new_content)))
