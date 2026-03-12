import gradio as gr
import os
import requests
import json
import uuid
import base64
from typing import List, Optional

# Persistent memory layer (session / memory separation)
from memory_store import (
    memory_store,
    make_llm_summariser,
    MemoryStore,
    SESSION_KEY_PRESETS,
    SUMMARY_EVERY_N_TURNS,
    prune_checkpointer,
)

# Import middleware and repo from existing codebase
# We wrap this in try-except in case the environment is not set up perfectly, 
# although we expect these files to exist based on analysis.
try:
    from skill_base import SkillMiddleware, skill_repo, TOOL_REGISTRY
    from skill_library import Skill
except ImportError:
    # Fallback for standalone testing if needed, though not expected to work fully without them
    print("Warning: Could not import skill_base or skill_library. Some features may not work.")
    SkillMiddleware = None
    skill_repo = None
    TOOL_REGISTRY = {}

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import SystemMessage

# --- Helper Functions ---

def get_ollama_models(api_url: str) -> List[str]:
    """Fetch available models from Ollama."""
    try:
        # standard ollama tag endpoint
        url = f"{api_url}/api/tags"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
    return []

def get_vllm_models(api_url: str) -> List[str]:
    """Fetch available models from a vLLM/OpenAI-compatible endpoint."""
    try:
        url = f"{api_url}/v1/models"
        if api_url.endswith("/v1"):
            url = f"{api_url}/models"
            
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model['id'] for model in data.get('data', [])]
    except Exception as e:
        print(f"Error fetching vLLM models: {e}")
    return []


def get_model_context_len(api_url: str, model_id: str) -> Optional[int]:
    """
    從 vLLM /v1/models 取得指定 model 的 max_model_len（最大 context token 數）。

    Returns None if the endpoint doesn't support it or request fails.
    """
    try:
        url = f"{api_url}/v1/models"
        if api_url.endswith("/v1"):
            url = f"{api_url}/models"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            for m in resp.json().get("data", []):
                if m.get("id") == model_id:
                    return m.get("max_model_len")  # vLLM-specific field
    except Exception as e:
        print(f"[context_len] Could not fetch max_model_len: {e}")
    return None


def compute_memory_params(
    max_model_len: int,
    max_output_tokens: int = 2048,
    system_prompt_chars: int = 2000,
    image_tokens_reserve: int = 0,    # 有圖片時建議傳入 2048~8192
    safety_margin: float = 0.15,      # 保留 15% 緩衝
    chars_per_token: float = 1.8,     # 中文偏低，英文偏高，混合約 1.8
) -> dict:
    """
    根據 max_model_len 動態計算各記憶體管理閾值。

    公式
    ----
    usable_tokens = max_model_len
                    - max_output_tokens
                    - system_prompt_tokens   (chars / chars_per_token)
                    - image_tokens_reserve
                    - safety_buffer          (usable * safety_margin)

    Returns dict with keys:
        usable_tokens       : 可用於對話歷史的 token 數
        summary_char_budget : 觸發 LLM 摘要的字元閾值（memory_store.py 用）
        map_reduce_chunk_size : map_reduce skill 的切片大小（字元）
        keep_recent_turns   : InMemorySaver 保留最近幾輪
        recent_messages_keep : JSON 持久層保留幾則近期訊息
    """
    system_tokens = system_prompt_chars / chars_per_token
    raw_usable = max_model_len - max_output_tokens - system_tokens - image_tokens_reserve
    usable_tokens = int(raw_usable * (1 - safety_margin))
    usable_tokens = max(1000, usable_tokens)  # 最低保底 1000 tokens

    # ── 字元換算 ──────────────────────────────────────────────────────────
    usable_chars = int(usable_tokens * chars_per_token)

    # 觸發摘要的閾值：佔可用空間的 40%（還有 60% 給 system prompt + 剩餘歷史）
    summary_char_budget = int(usable_chars * 0.40)

    # map_reduce chunk 大小：讓單次 Map call 佔 context 的 20%
    map_reduce_chunk_size = int(usable_chars * 0.20)
    map_reduce_chunk_size = max(800, min(map_reduce_chunk_size, 4000))  # 限 800~4000

    # InMemorySaver 保留幾輪：以每輪 ~600 chars 估算
    chars_per_turn = 600
    keep_recent_turns = max(2, min(int(usable_chars * 0.30 / chars_per_turn), 10))

    # JSON 持久層保留幾則（每則 = 1 role 的發言）
    recent_messages_keep = keep_recent_turns * 2  # 每輪 2 則

    return {
        "max_model_len":         max_model_len,
        "usable_tokens":         usable_tokens,
        "usable_chars":          usable_chars,
        "summary_char_budget":   summary_char_budget,
        "map_reduce_chunk_size": map_reduce_chunk_size,
        "keep_recent_turns":     keep_recent_turns,
        "recent_messages_keep":  recent_messages_keep,
    }

from PIL import Image
import io

def encode_image(image_path, max_size=4096):
    """Encodes an image to base64, resizing it if too large to save tokens."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (e.g. for PNGs with alpha)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Resize if either dimension is larger than max_size
            if max(img.width, img.height) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                print(f"Resized image to {img.width}x{img.height}")
            
            # Save to buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Create a single checkpointer to persist multi-turn sessions across identical thread_ids
# The InMemorySaver handles working memory (current session LangGraph state).
# For cross-session persistence, memory_store.py provides the persistent layer.
global_checkpointer = InMemorySaver()

def create_dynamic_agent(provider: str, api_url: str, model_name: str, system_prompt: str = None):
    """Creates a LangChain agent with the specified model configuration."""
    
    # Normalize base_url
    base_url = api_url
    if provider == "Ollama":
        if not base_url.endswith("/v1"):
             base_url = f"{base_url}/v1"
        api_key = "ollama" # dummy key
    else: # vLLM
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        api_key = "EMPTY" # Common for vLLM
    
    print(f"Initializing ChatOpenAI with base_url={base_url}, model={model_name}")
    
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        # max_tokens not set: let the API use the full remaining context window automatically
        temperature=0,
    )

    middleware = [SkillMiddleware()] if SkillMiddleware else []

    if not system_prompt:
        system_prompt = (
            "You are an intelligent assistant with access to a library of capabilities (skills). "
            "Use them to help the user. "
            "When analyzing images (OCR), trust the skills to guide you on extraction rules."
        )

    agent = create_agent(
        llm,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=global_checkpointer,
    )
    return agent

# Default LangGraph recursion steps (fallback)
MAX_AGENT_STEPS = 15


def _try_execute_text_tool_calls(content: str, log_lines: list) -> tuple[bool, str]:
    """Detect and execute <tool_call> JSON blocks found in plain-text AI content.

    Some models (e.g. Qwen) output tool calls as text tags instead of using
    the OpenAI structured function-calling API.  This function intercepts
    those blocks, runs the real tool, appends results to log_lines, and
    returns (True, non_tool_text) when at least one tool was executed,
    or (False, content) when no tool_call block was found.
    """
    import re

    pattern = re.compile(r"<tool_call>\s*({.*?})\s*</tool_call>", re.DOTALL)
    matches = pattern.findall(content)
    if not matches:
        return False, content

    executed_any = False
    for raw_json in matches:
        try:
            call = json.loads(raw_json)
        except json.JSONDecodeError as e:
            log_lines.append(f"⚠️ Failed to parse <tool_call> JSON: {e}\n{raw_json}")
            continue

        tool_name = call.get("name") or call.get("function", {}).get("name", "")
        # arguments may be under "arguments" or "parameters"
        args = call.get("arguments") or call.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        log_lines.append(f"🔧 [Text tool_call] Calling Tool: {tool_name} (args: {args})")

        tool_fn = TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            log_lines.append(f"❌ Tool '{tool_name}' not found in registry.")
            continue

        try:
            # LangChain @tool wraps the function; call it via .invoke() with a dict
            result = tool_fn.invoke(args)
        except Exception as e:
            result = f"Error invoking tool: {e}"

        log_lines.append(f"✅ Tool Output ({tool_name}):\n{result}")
        executed_any = True

    # Return the content with <tool_call> blocks stripped (the rest is reasoning text)
    remaining = pattern.sub("", content).strip()
    return executed_any, remaining

# --- Core Logic Handlers ---

def refresh_models(provider, api_url):
    """Refreshes the model dropdown based on provider and URL."""
    if not api_url:
        return gr.Dropdown(choices=[])
    
    models = []
    if provider == "Ollama":
        models = get_ollama_models(api_url)
    elif provider == "vLLM":
        models = get_vllm_models(api_url)
    
    if not models:
        gr.Info(f"No models found or error connecting to {provider} at {api_url}")
        return gr.Dropdown(choices=[], value=None)
    
    return gr.Dropdown(choices=models, value=models[0] if models else None)

# 支持直接個別讀取的文字檔類型
TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".html",
    ".log", ".yaml", ".yml", ".toml", ".ini", ".py",
    ".js", ".ts", ".sql",
}

# 支持經 base64 轉換送入模型的圖片類型
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def classify_uploaded_file(file_path: str) -> str:
    """
    判斷上傳檔案的類型。
    Returns: 'image' | 'text' | 'binary'
    """
    if not file_path:
        return 'none'
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return 'image'
    if ext in TEXT_EXTENSIONS:
        return 'text'
    return 'binary'   # pdf, ppt, xlsx 等—需要由 agent 透過 skill 處理


def run_agent_task(file_upload, provider, api_url, model_name, image_max_size, max_agent_steps, system_prompt, user_prompt, chatbot_history, session_key):
    """Execute the agent task with real-time streaming logs (generator) and multi-turn chat.
    
    file_upload: 上傳的檔案路徑（可能是圖片、PDF、CSV、文字撪 等任意檔案）
    """
    if chatbot_history is None:
        chatbot_history = []

    query = user_prompt.strip() if user_prompt and user_prompt.strip() else "請問有什麼我可以幫助您？"

    if not model_name:
        chatbot_history.append((query, "Please select a model."))
        yield chatbot_history, "", ""
        return

    # Prepare chat history to show the user's message
    file_type = classify_uploaded_file(file_upload) if file_upload else 'none'
    file_name = os.path.basename(file_upload) if file_upload else ""

    user_display = query
    if file_upload:
        icon = "🖼️" if file_type == 'image' else "📄"
        user_display = f"{icon} [{file_name}]\n\n{query}"

    chatbot_history.append([user_display, "⏳ Initializing Agent..."])
    yield chatbot_history, "⏳ Initializing Agent...", ""

    # ── Persistent Memory: load context BEFORE creating agent ────────────
    persistent_context = memory_store.get_session_start_context(session_key)
    effective_system_prompt = system_prompt
    if persistent_context:
        effective_system_prompt = system_prompt + persistent_context

    try:
        agent = create_dynamic_agent(provider, api_url, model_name, effective_system_prompt)
    except Exception as e:
        chatbot_history[-1][1] = f"Error creating agent: {e}"
        yield chatbot_history, "", ""
        return

    message_content = [{"type": "text", "text": query}]

    # ── 選擇檔案處理模式 ──────────────────────────────────────────────────
    if file_upload:
        if file_type == 'image':
            # 圖片：base64 內嵌送入 multimodal 模型
            chatbot_history[-1][1] = "⏳ Encoding Image..."
            yield chatbot_history, "⏳ Encoding Image...", ""
            base64_image = encode_image(file_upload, max_size=int(image_max_size))
            if base64_image:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            else:
                chatbot_history[-1][1] = "Failed to encode image."
                yield chatbot_history, "Failed to encode image.", ""
                return

        elif file_type == 'text':
            # 文字檔：直接讀內容嵌進 message
            chatbot_history[-1][1] = "⏳ Reading file..."
            yield chatbot_history, "⏳ Reading file...", ""
            try:
                with open(file_upload, "r", encoding="utf-8", errors="replace") as f:
                    file_text = f.read()
                file_preview = file_text[:8000]  # 超長則裁短，提醒用 map_reduce skill
                truncated = len(file_text) > 8000
                note = f"\n\n⚠️ 檔案過長（{len(file_text):,} 字元），內容已裁切至前 8000 字元。建議使用 map_reduce skill 處理完整內容。" if truncated else ""
                embedded = f"\n\n以下是上傳檔案 [{file_name}] 的內容：\n```\n{file_preview}\n```{note}"
                # 將檔案內容嵌入文字訊息
                message_content[0]["text"] = query + embedded
            except Exception as e:
                chatbot_history[-1][1] = f"Failed to read file: {e}"
                yield chatbot_history, f"Failed to read file: {e}", ""
                return

        else:  # binary: pdf, pptx, xlsx 等
            # 將檔案路徑告知 agent，讓它透過 skill 處理
            embedded = (
                f"\n\n使用者上傳了檔案：**{file_name}**（完整路徑：`{file_upload}`）\n"
                f"檔案類型：`{os.path.splitext(file_name)[1]}`\n"
                f"請使用適合的 skill（如 pdf、pptx、map_reduce 等）處理此檔案。"
            )
            message_content[0]["text"] = query + embedded

    # ── Persistent Memory: use session_key as LangGraph thread_id ──────────
    # Same key → same LangGraph checkpoint (working memory) +
    #            same persistent JSON (long-term summary)
    config = {
        "configurable": {"thread_id": session_key},
        "recursion_limit": int(max_agent_steps),  # prevent infinite loops
    }

    # ── Build final_system_prompt for UI display ───────────────────────────
    # This is what we show in the "Actual Injected System Prompt" text box.
    # The agent already has this via effective_system_prompt + middleware.
    final_system_prompt = effective_system_prompt
    if SkillMiddleware:
        mw = SkillMiddleware()
        skills_addendum = (
            "\n\n"
            "================================================================\n"
            "KNOWLEDGE SKILL LIBRARY (READ-ONLY REFERENCE — NOT CALLABLE)\n"
            "================================================================\n"
            "The following are skill LOOKUP KEYS, NOT tools or functions.\n"
            "You MUST NOT call them directly. They do not exist as callable tools.\n\n"
            f"{mw.skills_prompt if hasattr(mw, 'skills_prompt') else ''}\n\n"
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
            "      Example: run_python_code(\"from reportlab.pdfgen import canvas\\nc = canvas.Canvas('out.pdf')\\n...\", 'C:/output')\n"
            "================================================================"
        )
        final_system_prompt = effective_system_prompt + skills_addendum

    log_lines = ["⏳ Running Inference (this may take a while)..."]
    current_output = ""

    # --- Loop-detection state ---
    # Track (tool_name, args_repr) of last tool call; count consecutive duplicates
    _last_tool_sig: str = ""
    _dup_count: int = 0
    MAX_DUP_CALLS = 3  # abort if same tool+args called this many times in a row

    chatbot_history[-1][1] = current_output if current_output else "⏳ Running Inference..."
    yield chatbot_history, "\n\n".join(log_lines), final_system_prompt

    try:
        for update in agent.stream(
            {"messages": [{"role": "user", "content": message_content}]},
            config,
            stream_mode="updates",
        ):
            for node_name, node_output in update.items():
                msgs = node_output.get("messages", []) if isinstance(node_output, dict) else []

                for msg in msgs:
                    msg_type = getattr(msg, "type", None)

                    if msg_type == "ai":
                        content = getattr(msg, "content", "") or ""
                        tool_calls = getattr(msg, "tool_calls", None) or []

                        for tc in tool_calls:
                            name = tc.get("name", "")
                            args = tc.get("args", {})
                            if name:
                                # --- Duplicate-call detection ---
                                sig = f"{name}::{repr(args)}"
                                if sig == _last_tool_sig:
                                    _dup_count += 1
                                else:
                                    _last_tool_sig = sig
                                    _dup_count = 1

                                if _dup_count >= MAX_DUP_CALLS:
                                    warn = (
                                        f"\n⚠️  Detected repeated tool call "
                                        f"({name} x{_dup_count}). "
                                        f"Stopping to prevent infinite loop."
                                    )
                                    log_lines.append(warn)
                                    chatbot_history[-1][1] = current_output if current_output else warn
                                    yield chatbot_history, "\n\n".join(log_lines), final_system_prompt
                                    return

                                log_lines.append(f"🔧 Calling Tool: {name} (args: {args})")

                        if content:
                            if tool_calls:
                                log_lines.append(f"🤖 AI Thinking:\n{content}")
                            else:
                                # ---- Text-format tool_call fallback ----
                                executed, remaining = _try_execute_text_tool_calls(content, log_lines)
                                if executed:
                                    if remaining:
                                        log_lines.append(f"🤖 AI Thinking (after tool):\n{remaining}")
                                else:
                                    log_lines.append(f"🤖 Final Answer:\n{content}")
                                    current_output = content

                        if current_output:
                            chatbot_history[-1][1] = current_output
                        yield chatbot_history, "\n\n".join(log_lines), final_system_prompt

                    elif msg_type == "tool":
                        tool_name = getattr(msg, "name", "tool")
                        tool_content = getattr(msg, "content", "")
                        log_lines.append(f"✅ Tool Output ({tool_name}):\n{tool_content}")
                        if current_output:
                            chatbot_history[-1][1] = current_output
                        yield chatbot_history, "\n\n".join(log_lines), final_system_prompt

    except Exception as e:
        err_msg = str(e)
        if "recursion" in err_msg.lower() or "GraphRecursion" in err_msg:
            log_lines.append(
                f"\n⚠️  Agent exceeded max steps ({max_agent_steps}). "
                f"Task stopped to prevent infinite loop."
            )
        else:
            log_lines.append(f"\n❌ Error: {err_msg}")
            current_output += f"\n\ninference failed: {err_msg}"
        chatbot_history[-1][1] = current_output if current_output else err_msg
        yield chatbot_history, "\n\n".join(log_lines), final_system_prompt
        return

    final_answer = current_output if current_output else "Finished without generating response."
    chatbot_history[-1][1] = final_answer

    # ── Persistent Memory: record this turn → may trigger rolling summarise ─
    try:
        # Build a lightweight LLM summariser using the same model just used
        _llm_for_summary = None
        try:
            from langchain_openai import ChatOpenAI as _ChatOpenAI
            base_url = api_url if api_url.endswith("/v1") else f"{api_url}/v1"
            _api_key = "ollama" if provider == "Ollama" else "EMPTY"
            _llm_for_summary = _ChatOpenAI(
                model=model_name,
                api_key=_api_key,
                base_url=base_url,
                temperature=0,
            )
        except Exception:
            pass

        _summariser = make_llm_summariser(_llm_for_summary) if _llm_for_summary else None
        # Strip image data from user message before storing
        stored_user_msg = query  # text only
        updated_mem = memory_store.record_turn(
            session_key=session_key,
            user_message=stored_user_msg,
            ai_message=final_answer,
            llm_summariser=_summariser,
        )
        log_lines.append(f"💾 Memory saved → session_key: {session_key}")

        # ── InMemorySaver 修剪：如果刚才觸發了摘要，同步裁掉舊訊息 ─────────
        # 防止層 2（InMemorySaver）的 messages[] 無限堤積
        if updated_mem.summary and updated_mem.last_summarised_at_turn == updated_mem.turn_count:
            pruned = prune_checkpointer(
                checkpointer=global_checkpointer,
                session_key=session_key,
                summary=updated_mem.summary,
            )
            if pruned:
                log_lines.append("✂️ InMemorySaver 記憶已裁切，與摘要同步")
    except Exception as mem_err:
        log_lines.append(f"⚠️ Memory save failed: {mem_err}")

    yield chatbot_history, "\n\n".join(log_lines), final_system_prompt


# --- Skill Editor Handlers ---

def list_skills():
    """Returns list of skill names."""
    if not skill_repo:
        return []
    skills = skill_repo.get_all_skills()
    return [s.name for s in skills]

def on_skill_select(skill_name):
    """Updates file list and loads default file content when skill changes."""
    if not skill_name or not skill_repo:
        return gr.Dropdown(choices=[], value=None), ""
    
    files = skill_repo.list_skill_files(skill_name)
    # Default to SKILL.md if available
    default_file = "SKILL.md" if "SKILL.md" in files else (files[0] if files else None)
    
    content = ""
    if default_file:
         content = skill_repo.get_skill_details(skill_name, default_file) or ""
         
    return gr.Dropdown(choices=files, value=default_file), content

def load_skill_content(skill_name, file_path):
    """Loads content of a specific file for a given skill."""
    if not skill_name or not skill_repo:
        return ""
    
    # If file_path is not provided (e.g. initial load), try to find SKILL.md
    if not file_path:
        file_path = "SKILL.md"
        
    content = skill_repo.get_skill_details(skill_name, file_path)
    # If get_skill_details returns error string starting with "Error", we might want to handle it
    # But for now just return it.
    return content if content else ""

def save_skill_content(skill_name, file_path, new_content):
    """Saves new content to the specified file."""
    if not skill_repo:
        return "Repository not initialized."
    
    # We need to find the path
    skill = skill_repo._find_skill_by_name(skill_name)
    if not skill:
        return "Skill not found."
    
    if not file_path:
        return "No file selected."

    try:
        # Security check: ensure path is within skill dir
        target_path = os.path.join(skill.path, file_path)
        common_prefix = os.path.commonpath([os.path.abspath(skill.path), os.path.abspath(target_path)])
        if common_prefix != os.path.abspath(skill.path):
            return "Error: Access denied (outside skill dir)."

        # Ensure directory exists if it's a deep path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        return f"Successfully saved {file_path}"
    except Exception as e:
        return f"Error saving file: {e}"

def create_new_skill(new_skill_name):
    """Creates a new skill directory and template."""
    if not new_skill_name:
        return "Please enter a name.", gr.update()
    
    if not skill_repo:
        return "Repo not ready.", gr.update()

    # Sanitize name for folder
    folder_name = "".join(x for x in new_skill_name if x.isalnum() or x in ['_', '-'])
    new_path = os.path.join(skill_repo.root_directory, folder_name)
    
    if os.path.exists(new_path):
        return "Skill/Directory allready exists.", gr.update()
    
    try:
        os.makedirs(new_path)
        skill_md_path = os.path.join(new_path, "SKILL.md")
        initial_content = f"""---
name: {new_skill_name}
description: Description of the new skill.
---

# {new_skill_name}

Your instructions here.
"""
        with open(skill_md_path, "w", encoding="utf-8") as f:
            f.write(initial_content)
        
        skill_repo._cache_skills = None # Reset cache
        
        # Refresh dropdown
        new_list = list_skills()
        return (
            f"Created skill {new_skill_name}", 
            gr.Dropdown(choices=new_list, value=new_skill_name), 
            gr.Dropdown(choices=["SKILL.md"], value="SKILL.md"),
            initial_content
        )
    
    except Exception as e:
        return f"Error creating skill: {e}", gr.update(), gr.update(), gr.update()

def create_new_file(skill_name, new_file_name):
    """Creates a new file for the selected skill."""
    if not skill_name:
        return "Please select a skill first.", gr.update()
    if not new_file_name:
        return "Please enter a file name.", gr.update()
        
    if not skill_repo:
        return "Repo not initialized.", gr.update()
        
    skill = skill_repo._find_skill_by_name(skill_name)
    if not skill:
        return "Skill not found.", gr.update()
        
    try:
        # Sanitize path
        safe_name = new_file_name.strip().lstrip('/')
        target_path = os.path.join(skill.path, safe_name)
        
        # Check if outside skill dir
        if not os.path.abspath(target_path).startswith(os.path.abspath(skill.path)):
             return "Error: Invalid file path.", gr.update()
             
        if os.path.exists(target_path):
             return "Error: File already exists.", gr.update()
             
        # Create parent dirs if needed
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        with open(target_path, "w", encoding='utf-8') as f:
            f.write("") # Create empty file
            
        # Refresh file list
        files = skill_repo.list_skill_files(skill_name)
        return f"Created {safe_name}", gr.Dropdown(choices=files, value=safe_name), ""
        
    except Exception as e:
        return f"Error creating file: {e}", gr.update(), gr.update()

# --- GUI Construction ---

with gr.Blocks(title="Agentic Studio", css="footer {visibility: hidden}") as demo:
    gr.Markdown("# 🤖 Agentic Studio")
    
    with gr.Tabs():
        # --- TAB 1: Agent ---
        with gr.Tab("🤖 Agent"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 檔案輸入（任意類型）
                    file_input = gr.File(
                        label="📂 Upload File (image / PDF / PPT / CSV / TXT ...)",
                        file_count="single",
                        file_types=None,   # None = 任意檔案
                        height=160,
                    )
                    image_input = file_input   # alias 保留，方便現有綁定參考
                    
                    # Settings
                    with gr.Group():
                        gr.Markdown("### ⚙️ Model Settings")
                        provider_dropdown = gr.Dropdown(choices=["Ollama", "vLLM"], value="vLLM", label="Provider")
                        api_url_input = gr.Textbox(value="http://10.1.1.7:9000", label="API URL")
                        
                        with gr.Row():
                            model_dropdown = gr.Dropdown(label="Select Model", interactive=True, scale=3, allow_custom_value=True)
                            refresh_btn = gr.Button("🔄 Refresh", scale=1)
                        
                        gr.Markdown("### 🖼️ 圖片磁種設定（僅對圖片檔案有效）")
                        image_max_size_slider = gr.Slider(
                            minimum=256,
                            maximum=4096,
                            value=1024,
                            step=128,
                            label="圖片最大邊長 (px)",
                            info="對非圖片檔案無效。數值越小 token 用量越少"
                        )

                        gr.Markdown("### 🤖 Agent Settings")
                        max_agent_steps_slider = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=15,
                            step=1,
                            label="Max Agent Steps",
                            info="限制 Agent 思考與呼叫工具的最大次數，防止無限迴圈"
                        )

                        with gr.Accordion("📝 Prompt Settings", open=True):
                            system_prompt_input = gr.TextArea(
                                label="System Prompt",
                                value="You are an intelligent assistant with access to a library of capabilities (skills). Use them to help the user.",
                                lines=3
                            )

                    # User Query 常駐展示，預設空白
                    user_prompt_input = gr.TextArea(
                        label="💬 User Query",
                        value="",
                        placeholder="請輸入指令或問題（不填則由 agent 自定義默認回覆）",
                        lines=3
                    )
                
                with gr.Column(scale=1):
                    # Output
                    gr.Markdown("### 📝 Output")

                    with gr.Group():
                        gr.Markdown("### 🔑 Session Key (持久記憶識別碼)")
                        session_key_input = gr.Textbox(
                            label="Session Key",
                            value="user:anonymous:thread:main",
                            placeholder="user:alice:thread:invoice-project",
                            interactive=True,
                            info="相同 key = 接續舊記憶。格式: user:{id}:thread:{name} 或 group:{channel}"
                        )
                        with gr.Row():
                            session_key_preset = gr.Dropdown(
                                label="快速選擇預設 Key",
                                choices=SESSION_KEY_PRESETS,
                                interactive=True,
                                scale=3,
                            )
                            apply_preset_btn = gr.Button("套用", scale=1)

                    chatbot = gr.Chatbot(label="Chat History", height=400)
                    log_output = gr.Textbox(label="Thinking Process & Logs", lines=10, interactive=False)
                    final_system_prompt_output = gr.Textbox(label="Actual Injected System Prompt", lines=8, interactive=False)
                    with gr.Row():
                        run_btn = gr.Button("🚀 Send / Run Task", variant="primary", scale=3)
                        stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)
                        clear_btn = gr.Button("🗑️ Clear Chat (保留記憶)", scale=1)
            
            # Event Wiring
            refresh_btn.click(refresh_models, inputs=[provider_dropdown, api_url_input], outputs=model_dropdown)

            # Apply preset session key
            apply_preset_btn.click(
                fn=lambda v: v,
                inputs=[session_key_preset],
                outputs=[session_key_input],
            )
            
            # Clear Chat History but keep persistent memory (just reset working memory)
            def clear_session_history(current_key, current_provider, current_url, current_model):
                """Reset the chat display. Persistent memory in data/memory/ is NOT deleted.
                The next message on the same key will still load the old summary.
                """
                import threading
                # Flush current working memory to persistent layer in background
                try:
                    base_url = current_url if current_url.endswith("/v1") else f"{current_url}/v1"
                    _api_key = "ollama" if current_provider == "Ollama" else "EMPTY"
                    from langchain_openai import ChatOpenAI as _ChatOpenAI
                    _llm = _ChatOpenAI(model=current_model or "__none__", api_key=_api_key, base_url=base_url, temperature=0)
                    _summariser = make_llm_summariser(_llm)
                    memory_store.flush_session(current_key, _summariser)
                except Exception:
                    memory_store.flush_session(current_key)  # flush without summarising
                return [], None, "", ""
                
            clear_btn.click(
                clear_session_history,
                inputs=[session_key_input, provider_dropdown, api_url_input, model_dropdown],
                outputs=[chatbot, file_input, log_output, final_system_prompt_output]
            )

            run_event = run_btn.click(
                run_agent_task,
                inputs=[
                    file_input, provider_dropdown, api_url_input, model_dropdown,
                    image_max_size_slider, max_agent_steps_slider, system_prompt_input,
                    user_prompt_input, chatbot, session_key_input
                ],
                outputs=[chatbot, log_output, final_system_prompt_output],
            )
            # Stop button cancels the running generator immediately
            stop_btn.click(fn=None, cancels=[run_event])

        # --- TAB 2: SKILLS ---
        with gr.Tab("📚 Skill Manager"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📂 Select Skill")
                    skill_select = gr.Dropdown(choices=list_skills(), label="Skill", interactive=True)
                    
                    gr.Markdown("### 📄 Select File")
                    file_select = gr.Dropdown(label="File", interactive=True)
                    
                    gr.Markdown("### ➕ Create New File")
                    new_file_name_input = gr.Textbox(label="New File Name", placeholder="e.g. references/types.md")
                    create_file_btn = gr.Button("Create File")
                    create_file_status = gr.Markdown()

                    refresh_skills_btn = gr.Button("🔄 Refresh List")
                    
                    gr.Markdown("### ➕ Create New Skill")
                    new_skill_name = gr.Textbox(label="New Skill Name", placeholder="e.g. receipt_ocr")
                    create_skill_btn = gr.Button("Create Skill")
                    create_status = gr.Markdown()

                with gr.Column(scale=3):
                    gr.Markdown("### ✏️ Edit File Content")
                    skill_content = gr.Code(language="markdown", label="Content", lines=30, interactive=True)
                    save_skill_btn = gr.Button("💾 Save Changes", variant="primary")
                    save_status = gr.Markdown()

            # Event Wiring
            
            # When skill changes, populate file list AND load default content
            skill_select.change(on_skill_select, inputs=skill_select, outputs=[file_select, skill_content])
            
            # When file changes (or is set by the above), load content
            file_select.change(load_skill_content, inputs=[skill_select, file_select], outputs=skill_content)
            
            create_file_btn.click(
                 create_new_file,
                 inputs=[skill_select, new_file_name_input],
                 outputs=[create_file_status, file_select, skill_content]
            )

            refresh_skills_btn.click(lambda: gr.Dropdown(choices=list_skills()), outputs=skill_select)
            
            save_skill_btn.click(save_skill_content, inputs=[skill_select, file_select, skill_content], outputs=save_status)
            
            create_skill_btn.click(
                create_new_skill, 
                inputs=[new_skill_name], 
                outputs=[create_status, skill_select, file_select, skill_content]
            )

        # --- TAB 3: MEMORY MANAGER ---
        with gr.Tab("🧠 Memory Manager"):
            gr.Markdown(
                "## 🧠 持久記憶管理\n\n"
                "查看所有已儲存的 Session 記憶。同一個 Session Key = 同一份記憶，跨裝置、跨 session 都能接續。\n\n"
                f"📁 存放路徑：`data/memory/`　　🔄 每 **{SUMMARY_EVERY_N_TURNS}** 輪自動壓縮一次摘要"
            )
            with gr.Row():
                refresh_mem_btn = gr.Button("🔄 重新整理記憶列表", variant="primary")
                mem_key_to_delete = gr.Textbox(label="輸入要刪除的 Session Key", scale=3)
                delete_mem_btn = gr.Button("🗑️ 刪除此記憶", variant="stop")

            mem_status = gr.Markdown()
            mem_table = gr.Dataframe(
                headers=["Session Key", "輪數", "摘要長度", "最後更新", "摘要預覽"],
                label="所有 Thread 記憶",
                interactive=False,
                wrap=True,
            )
            mem_detail = gr.TextArea(label="📖 摘要完整內容（點選後貼上 Session Key 查詢）", lines=10, interactive=False)
            mem_key_inspect = gr.Textbox(label="查詢 Session Key（輸入後按 Enter）")

            def list_memories():
                threads = memory_store.list_threads()
                if not threads:
                    return [["(尚無記憶)", "-", "-", "-", "-"]]
                rows = []
                for t in threads:
                    summary_preview = (t.summary[:80] + "…") if len(t.summary) > 80 else (t.summary or "(無)")
                    rows.append([
                        t.session_key,
                        str(t.turn_count),
                        str(len(t.summary)),
                        t.last_updated_at[:16].replace("T", " "),
                        summary_preview,
                    ])
                return rows

            def inspect_memory(key):
                if not key:
                    return "(請輸入 Session Key)"
                mem = memory_store.load_thread(key.strip())
                if not mem.summary and not mem.recent_messages:
                    return f"找不到 key='{key.strip()}' 的記憶，或該 key 尚無內容。"
                lines = [
                    f"🔑 Session Key: {mem.session_key}",
                    f"📊 總輪數: {mem.turn_count}　最後摘要於第 {mem.last_summarised_at_turn} 輪",
                    f"🕐 建立: {mem.created_at[:16]}　更新: {mem.last_updated_at[:16]}",
                    "",
                    "─── 摘要 ───",
                    mem.summary or "(無)",
                    "",
                    "─── 最近幾輪對話 ───",
                ]
                for m in mem.recent_messages[-6:]:
                    role = "👤 使用者" if m.get("role") == "user" else "🤖 AI"
                    lines.append(f"{role}: {m.get('content', '')[:200]}")
                return "\n".join(lines)

            def delete_memory(key):
                if not key:
                    return "請輸入要刪除的 Session Key。"
                deleted = memory_store.delete_thread(key.strip())
                if deleted:
                    return f"✅ 已刪除 key='{key.strip()}' 的記憶。"
                return f"⚠️ 找不到 key='{key.strip()}'，無需刪除。"

            # Wire events
            refresh_mem_btn.click(list_memories, outputs=mem_table)
            mem_key_inspect.submit(inspect_memory, inputs=mem_key_inspect, outputs=mem_detail)
            delete_mem_btn.click(
                lambda key: (delete_memory(key), list_memories()),
                inputs=mem_key_to_delete,
                outputs=[mem_status, mem_table],
            )

            # Auto-load on tab render
            demo.load(list_memories, outputs=mem_table)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
