import gradio as gr
import os
import requests
import json
import uuid
import base64
from typing import List, Optional

# Import middleware and repo from existing codebase
# We wrap this in try-except in case the environment is not set up perfectly, 
# although we expect these files to exist based on analysis.
try:
    from skill_base import SkillMiddleware, skill_repo
    from skill_library import Skill
except ImportError:
    # Fallback for standalone testing if needed, though not expected to work fully without them
    print("Warning: Could not import skill_base or skill_library. Some features may not work.")
    SkillMiddleware = None
    skill_repo = None

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
        # OpenAI compatible models endpoint
        url = f"{api_url}/v1/models" # Adjust if user inputs full path
        if api_url.endswith("/v1"):
            url = f"{api_url}/models"
            
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model['id'] for model in data.get('data', [])]
    except Exception as e:
        print(f"Error fetching vLLM models: {e}")
    return []

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

def create_dynamic_agent(provider: str, api_url: str, model_name: str, max_tokens: int = 4096, system_prompt: str = None):
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
    
    print(f"Initializing ChatOpenAI with base_url={base_url}, model={model_name}, max_tokens={max_tokens}")
    
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens, 
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
        checkpointer=InMemorySaver(),
    )
    return agent

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

def run_ocr_task(image, provider, api_url, model_name, max_tokens, system_prompt, user_prompt, progress=gr.Progress()):
    """Execute the OCR task."""
    if not image:
        return "Please upload an image.", ""
    if not model_name:
        return "Please select a model.", ""

    progress(0, desc="Initializing Agent...")
    try:
        agent = create_dynamic_agent(provider, api_url, model_name, max_tokens, system_prompt)
    except Exception as e:
        return f"Error creating agent: {e}", ""

    progress(0.2, desc="Encoding Image...")
    base64_image = encode_image(image)
    if not base64_image:
        return "Failed to encode image.", ""

    query = user_prompt if user_prompt else "請幫我提取這張單據的資料，請自行判斷是哪一類單據，然後依照其種類去提取相對應的欄位資料。"

    message_content = [
        {"type": "text", "text": query},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    progress(0.4, desc="Running Inference (this may take a while)...")
    try:
        # Invoke agent
        result = agent.invoke(
            {"messages": [{"role": "user", "content": message_content}]},
            config
        )
        
        # Extract logs
        logs = []
        for msg in result["messages"]:
            if msg.type == "ai":
                if msg.content:
                    logs.append(f"🤖 AI Thinking:\n{msg.content}")
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        logs.append(f"🔧 Calling Tool: {tc['name']} (args: {tc['args']})")
            elif msg.type == "tool":
                logs.append(f"✅ Tool Output ({msg.name}):\n{msg.content}")
        
        log_str = "\n\n".join(logs)
        
        # Extract last message content
        last_message = result["messages"][-1].content
        return last_message, log_str

    except Exception as e:
        return f"inference failed: {e}", ""

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

with gr.Blocks(title="Agentic OCR Studio") as demo:
    gr.Markdown("# 🕵️‍♂️ Agentic OCR Studio")
    
    with gr.Tabs():
        # --- TAB 1: OCR ---
        with gr.Tab("🖼️ OCR & Inference"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Image Input
                    image_input = gr.Image(type="filepath", label="Upload Image", height=400)
                    
                    # Settings
                    with gr.Group():
                        gr.Markdown("### ⚙️ Model Settings")
                        provider_dropdown = gr.Dropdown(choices=["Ollama", "vLLM"], value="vLLM", label="Provider")
                        api_url_input = gr.Textbox(value="http://10.1.1.7:9000", label="API URL")
                        
                        with gr.Row():
                            model_dropdown = gr.Dropdown(label="Select Model", interactive=True, scale=3, allow_custom_value=True)
                            refresh_btn = gr.Button("🔄 Refresh", scale=1)
                        
                        max_tokens_slider = gr.Slider(minimum=256, maximum=8192, value=4096, step=256, label="Max Tokens")

                        with gr.Accordion("📝 Prompt Settings", open=False):
                            system_prompt_input = gr.TextArea(
                                label="System Prompt", 
                                value="You are an intelligent assistant with access to a library of capabilities (skills). Use them to help the user. When analyzing images (OCR), trust the skills to guide you on extraction rules.",
                                lines=3
                            )
                            user_prompt_input = gr.TextArea(
                                label="User Query", 
                                value="請幫我提取這張單據的資料，請自行判斷是哪一類單據，然後依照其種類去提取相對應的欄位資料。",
                                lines=3
                            )
                
                with gr.Column(scale=1):
                    # Output
                    gr.Markdown("### 📝 Output")
                    output_text = gr.Textbox(label="Agent Response", lines=10, interactive=False)
                    log_output = gr.Textbox(label="Thinking Process & Logs", lines=10, interactive=False)
                    run_btn = gr.Button("🚀 Run OCR Analysis", variant="primary", size="lg")
            
            # Event Wiring
            refresh_btn.click(refresh_models, inputs=[provider_dropdown, api_url_input], outputs=model_dropdown)
            
            # Auto-refresh on provider change (optional, maybe better explicit)
            # provider_dropdown.change(refresh_models, inputs=[provider_dropdown, api_url_input], outputs=model_dropdown)

            run_btn.click(
                run_ocr_task, 
                inputs=[image_input, provider_dropdown, api_url_input, model_dropdown, max_tokens_slider, system_prompt_input, user_prompt_input], 
                outputs=[output_text, log_output]
            )

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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css="footer {visibility: hidden}")
