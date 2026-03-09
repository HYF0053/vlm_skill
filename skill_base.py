import uuid
import os
import sys
import subprocess
import shlex
from typing import TypedDict, NotRequired, Callable
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

# Import the new Skill Library
from skill_library import FileSystemSkillRepository

# Initialize Repository
# Dynamically resolve the skills directory relative to this file's location
# This works on both Windows and Linux without hardcoded paths
SKILL_REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
skill_repo = FileSystemSkillRepository(SKILL_REPO_PATH)


# Create skill loading tools
@tool
def load_skill_overview(skill_name: str) -> str:
    """Load the overview of a skill to understand its capabilities and usage.
    The SKILL.md content is already included in the return value \u2014 do NOT
    call read_skill_file for SKILL.md afterwards.

    Args:
        skill_name: The exact snake_case skill name (e.g., "form_ocr_skill").
    """
    content = skill_repo.get_skill_overview(skill_name)
    if content:
        files = skill_repo.list_skill_files(skill_name)
        # Exclude SKILL.md \u2014 its full content is already returned above.
        # Only show additional reference files the LLM may need to read.
        extra_files = [f for f in files if f.upper() != "SKILL.MD"]
        if extra_files:
            file_list_str = "\n".join([f"- {f}" for f in extra_files])
            files_section = (
                f"=== Additional reference files you MAY read ===\n"
                f"{file_list_str}\n"
                f"=== SKILL.md is already shown above \u2014 do NOT re-read it ===\n\n"
                f"Use read_skill_file(skill_name='{skill_name}', file_path=<path from list above>)"
            )
        else:
            files_section = (
                "No additional reference files. "
                "All instructions are contained in the overview above."
            )
        return (
            f"Loaded overview for skill: {skill_name}\n\n"
            f"{content}\n\n"
            f"{files_section}"
        )
    else:
        skills = skill_repo.get_all_skills()
        available = ", ".join(f'"{s.name}"' for s in skills)
        return (
            f"Skill '{skill_name}' not found.\n"
            f"Available skills (use exact name): {available}\n"
            f"Call load_skill_overview with one of the exact names above."
        )


@tool
def read_skill_file(skill_name: str, file_path: str) -> str:
    """Read a specific file from a skill's directory.
    
    Use this to read reference documents, examples, or specific instructions
    mentioned in the skill overview.

    Args:
        skill_name: The name of the skill.
        file_path: The relative path to the file within the skill directory (e.g., "references/accident_report.md").
    """
    content = skill_repo.get_skill_details(skill_name, file_path)
    return content


@tool
def execute_script(skill_name: str, script_path: str, args: str = "") -> str:
    """Execute a Python script located inside a skill's directory.

    Use this after reading a skill's SKILL.md and identifying a helper script
    you want to run (e.g., in the scripts/ sub-folder).  The script is run
    with the current Python interpreter so all installed packages are available.

    Args:
        skill_name: The exact skill name that owns the script.
        script_path: Relative path inside the skill directory to the .py file
                     (e.g., "scripts/convert_pdf_to_images.py").
        args: Optional space-separated command-line arguments passed to the
              script, exactly as you would write them on the command line
              (e.g., "input.pdf output_dir").  Leave blank if none.

    Returns:
        Combined stdout + stderr output of the script, or an error message.
    """
    skill = skill_repo._find_skill_by_name(skill_name)
    if skill is None:
        available = ", ".join(f'"{s.name}"' for s in skill_repo.get_all_skills())
        return f"Skill '{skill_name}' not found. Available skills: {available}"

    # Resolve and validate the script path
    abs_script = os.path.normpath(os.path.join(skill.path, script_path))
    if not os.path.abspath(abs_script).startswith(os.path.abspath(skill.path)):
        return "Error: script_path escapes the skill directory — access denied."
    if not os.path.isfile(abs_script):
        return f"Error: Script not found at '{abs_script}'."
    if not abs_script.endswith(".py"):
        return "Error: Only .py scripts may be executed via this tool."

    # Build the command
    cmd = [sys.executable, abs_script]
    if args.strip():
        cmd += shlex.split(args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2-minute safety timeout
            cwd=skill.path,   # run from the skill directory so relative paths in the script work
        )
        output_parts = []
        if result.stdout:
            output_parts.append(f"[stdout]\n{result.stdout.rstrip()}")
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr.rstrip()}")
        output_parts.append(f"[return code] {result.returncode}")
        return "\n\n".join(output_parts) or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Script execution timed out (>120 s)."
    except Exception as e:
        return f"Error running script: {e}"


@tool
def run_cli_command(command: str, working_directory: str = "") -> str:
    """Execute an arbitrary CLI / shell command on the host system.

    Use this when a skill's instructions ask you to run a command-line tool
    (such as pdftotext, qpdf, ffmpeg, pip install, etc.) or when you need
    to run a Python one-liner or any other shell command.

    Args:
        command: The full command string to execute (e.g., "pdftotext input.pdf output.txt",
                 or "pip install pypdf", or "python -c \"import sys; print(sys.version)\"").
        working_directory: Optional absolute path to use as the working directory.
                           Defaults to the vlm_skill project root when left blank.

    Returns:
        Combined stdout + stderr of the command, or an error message.

    IMPORTANT:
      - This tool has real side-effects (file creation, package installs, etc.).
      - Do NOT use it for destructive operations (rm -rf, format, etc.).
      - Maximum execution time is 120 seconds.
    """
    # Default working directory = project root (same as this file)
    cwd = working_directory.strip() if working_directory.strip() else os.path.dirname(os.path.abspath(__file__))

    if not os.path.isdir(cwd):
        return f"Error: working_directory '{cwd}' does not exist."

    try:
        result = subprocess.run(
            command,
            shell=True,          # allow full shell syntax including pipes, builtins, etc.
            capture_output=True,
            text=True,
            timeout=120,
            cwd=cwd,
        )
        output_parts = []
        if result.stdout:
            output_parts.append(f"[stdout]\n{result.stdout.rstrip()}")
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr.rstrip()}")
        output_parts.append(f"[return code] {result.returncode}")
        return "\n\n".join(output_parts) or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (>120 s)."
    except Exception as e:
        return f"Error running command: {e}"


@tool
def run_python_code(code: str, working_directory: str = "") -> str:
    """Write a Python code string to a temporary file and execute it.

    Use this when you want to write a new Python program based on examples or
    instructions from a skill's SKILL.md (or any reference file) and immediately
    run it.  The code is saved to a uniquely-named temp file, executed, then the
    temp file is automatically deleted.

    Args:
        code: The complete, valid Python source code to run.  Must be a
              self-contained script (all imports included).  Multi-line strings
              are fine — just pass the code as-is.
        working_directory: Optional absolute path to use as the working directory
                           while the script runs.  Defaults to the vlm_skill
                           project root when left blank.  Use this to control
                           where output files created by the script are saved.

    Returns:
        Combined stdout + stderr of the executed script, or an error message.

    Example workflow:
        1. Load a skill and read its examples.
        2. Write Python code based on those examples.
        3. Call run_python_code(code=<your_code>, working_directory=<target_dir>)
        4. Read the output / resulting files as needed.
    """
    import tempfile

    cwd = working_directory.strip() if working_directory.strip() else os.path.dirname(os.path.abspath(__file__))

    if not os.path.isdir(cwd):
        return f"Error: working_directory '{cwd}' does not exist."

    # Write code to a uniquely-named temp file so concurrent calls don't collide
    tmp_path = os.path.join(tempfile.gettempdir(), f"vlm_skill_tmp_{uuid.uuid4().hex}.py")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=cwd,
        )
        output_parts = []
        if result.stdout:
            output_parts.append(f"[stdout]\n{result.stdout.rstrip()}")
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr.rstrip()}")
        output_parts.append(f"[return code] {result.returncode}")
        return "\n\n".join(output_parts) or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Script execution timed out (>120 s)."
    except Exception as e:
        return f"Error running code: {e}"
    finally:
        # Always clean up the temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# Create skill middleware
class SkillMiddleware(AgentMiddleware):
    """Middleware that injects skill descriptions into the system prompt."""

    # Register the tools as class variables
    tools = [load_skill_overview, read_skill_file, execute_script, run_cli_command, run_python_code]

    def __init__(self):
        """Initialize and generate the skills prompt from the repository."""
        # Build skills prompt from the repository
        # We explicitly re-fetch here to ensure freshness if init happens at startup
        self.refresh_skills_prompt()

    def refresh_skills_prompt(self):
        skills = skill_repo.get_all_skills()
        skills_list = []
        for skill in skills:
            desc = skill.description.replace('\n', ' ').strip()
            # Use a format that looks like a reference table, NOT a function list
            skills_list.append(f'  Skill key "{skill.name}"\n  Purpose: {desc}')
        
        if not skills_list:
            self.skills_prompt = "No skills available."
        else:
            self.skills_prompt = "\n\n".join(skills_list)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync: Inject skill descriptions into system prompt."""
        # Build the skills addendum
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
            "CALLABLE TOOLS (all tools you may invoke directly):\n"
            "  - load_skill_overview(skill_name: str) -> str\n"
            "  - read_skill_file(skill_name: str, file_path: str) -> str\n"
            "  - execute_script(skill_name: str, script_path: str, args: str = \"\") -> str\n"
            "      Run a Python script from the skill's scripts/ folder.\n"
            "      Example: execute_script('pdf', 'scripts/convert_pdf_to_images.py', 'doc.pdf /tmp/out')\n"
            "  - run_cli_command(command: str, working_directory: str = \"\") -> str\n"
            "      Run any shell/CLI command (pdftotext, pip install, python -c ..., etc.)\n"
            "      Example: run_cli_command('pdftotext input.pdf output.txt', '/tmp')\n"
            "  - run_python_code(code: str, working_directory: str = \"\") -> str\n"
            "      Write Python code you compose yourself and execute it immediately.\n"
            "      Use this when you want to implement something based on skill examples\n"
            "      (e.g., from SKILL.md code blocks) without needing a pre-existing script.\n"
            "      Example: run_python_code(\"from reportlab.pdfgen import canvas\\nc = canvas.Canvas('out.pdf')\\n...\", 'C:/output')\n"
            "================================================================"
        )

        # Append to system message content blocks
        # Handle cases where content might be string or list of blocks
        original_content = request.system_message.content
        if isinstance(original_content, str):
            new_content = original_content + skills_addendum
        else:
             # It's likely a list of blocks
             new_content = list(original_content) + [
                {"type": "text", "text": skills_addendum}
             ]
             
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)

# ---------------------------------------------------------------------------
# Tool registry — maps tool name → callable (unwrapped from @tool decorator).
# Used by the Gradio streaming handler to intercept text-format <tool_call>
# blocks emitted by models that don't support structured function calling.
# ---------------------------------------------------------------------------
TOOL_REGISTRY: dict = {
    "load_skill_overview": load_skill_overview,
    "read_skill_file": read_skill_file,
    "execute_script": execute_script,
    "run_cli_command": run_cli_command,
    "run_python_code": run_python_code,
}

# Initialize Chat Model
# Disable LangSmith tracing to avoid errors if API key is missing/invalid
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

print("Initializing ChatOpenAI...")
llm = ChatOpenAI(
    model="Qwen/Qwen3-VL-32B-Instruct",
    api_key="",
    base_url="http://10.1.1.7:9000/v1",
    max_tokens=4096,
    temperature=0,
)

# Create the agent with skill support
agent = create_agent(
    llm,
    system_prompt=(
        "You are an intelligent assistant with access to a library of capabilities (skills). "
        "Use them to help the user."
    ),
    middleware=[SkillMiddleware()],
    checkpointer=InMemorySaver(),
)

# Example usage for verific# Example usage
if __name__ == "__main__":
    import argparse
    import base64
    from io import BytesIO

    def encode_image(image_path):
        """Encodes an image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error reading image: {e}")
            return None

    parser = argparse.ArgumentParser(description="Run the Skill Agent with optional image input.")
    parser.add_argument("--query", type=str, default="請幫我提取這張單據的資料，請自行判斷是哪一類單據，然後依照其種類去提取相對應的欄位資料", help="The user query.")
    parser.add_argument("--image", type=str, default="/home/ubuntu/Documents/2026-0120_新安東京 - 表單 AI OCR/A06現場圖/手寫/A06-H-N-260116-011.jpg")
 
    args = parser.parse_args()

    # Configuration for this conversation thread
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Running query: {args.query}")
    
    message_content = []
    
    # Add text content
    message_content.append({"type": "text", "text": args.query})
    
    # Add image content if provided
    if args.image:
        print(f"Loading image from: {args.image}")
        base64_image = encode_image(args.image)
        if base64_image:
            # LangChain/Ollama generic image format
            # Note: The exact format might depend on the specific ChatOllama implementation version,
            # but usually passing 'image_url' with a data URI or just base64 works for many.
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        else:
            print("Failed to load image. Proceeding with text only.")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message_content,
                }
            ]
        },
        config
    )

    # Print the conversation
    for message in result["messages"]:
        if hasattr(message, 'pretty_print'):
            message.pretty_print()
        else:
            print(f"\n{message.type.upper()}: {message.content}")