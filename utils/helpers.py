import os
import requests
import base64
import io
from typing import List, Optional
from PIL import Image

# --- File Type Constants ---
TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".html",
    ".log", ".yaml", ".yml", ".toml", ".ini", ".py",
    ".js", ".ts", ".sql",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

def classify_uploaded_file(file_path: str) -> str:
    """判斷上傳檔案的類型: 'image' | 'text' | 'binary' | 'none'"""
    if not file_path:
        return 'none'
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return 'image'
    if ext in TEXT_EXTENSIONS:
        return 'text'
    return 'binary'

def encode_image(image_path: str, max_size: int = 4096) -> Optional[str]:
    """Encodes an image to base64, resizing it if too large to save tokens."""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            if max(img.width, img.height) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                print(f"Resized image to {img.width}x{img.height}")
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# --- LLM Provider Clients ---

def get_ollama_models(api_url: str) -> List[str]:
    """Fetch available models from Ollama."""
    try:
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

def get_model_context_len(provider: str, api_url: str, model_id: str) -> Optional[int]:
    """從 vLLM 或 Ollama 取得指定 model 的 max_model_len/context_length。"""
    try:
        if provider == "Ollama":
            url = f"{api_url}/api/show"
            resp = requests.post(url, json={"name": model_id}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                # Improved search for context length in model_info
                info = data.get("model_info", {})
                # Try specific common keys first
                for k, v in info.items():
                    if k.endswith(".context_length"):
                        return int(v)
                # Fallback for very specific formats if needed
                # Fallback: check if it's in templates or other fields if needed, 
                # but usually it's in model_info.
        else: # vLLM / OpenAI compatible
            url = f"{api_url}/v1/models"
            if api_url.endswith("/v1"):
                url = f"{api_url}/models"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                for m in resp.json().get("data", []):
                    if m.get("id") == model_id:
                        return m.get("max_model_len")
    except Exception as e:
        print(f"[context_len] Could not fetch max_model_len: {e}")
    return None

def compute_memory_params(
    max_model_len: int,
    max_output_tokens: int = 2048,
    system_prompt_chars: int = 2000,
    image_tokens_reserve: int = 0,
    safety_margin: float = 0.15,
    chars_per_token: float = 1.8,
) -> dict:
    """根據 max_model_len 動態計算各記憶體管理閾值。"""
    system_tokens = system_prompt_chars / chars_per_token
    raw_usable = max_model_len - max_output_tokens - system_tokens - image_tokens_reserve
    usable_tokens = int(raw_usable * (1 - safety_margin))
    usable_tokens = max(1000, usable_tokens)

    usable_chars = int(usable_tokens * chars_per_token)
    summary_char_budget = int(usable_chars * 0.40)
    map_reduce_chunk_size = int(usable_chars * 0.20)
    map_reduce_chunk_size = max(800, min(map_reduce_chunk_size, 4000))
    
    chars_per_turn = 600
    keep_recent_turns = max(2, min(int(usable_chars * 0.30 / chars_per_turn), 10))
    recent_messages_keep = keep_recent_turns * 2

    return {
        "max_model_len":         max_model_len,
        "usable_tokens":         usable_tokens,
        "usable_chars":          usable_chars,
        "summary_char_budget":   summary_char_budget,
        "map_reduce_chunk_size": map_reduce_chunk_size,
        "keep_recent_turns":     keep_recent_turns,
        "recent_messages_keep":  recent_messages_keep,
        "chars_per_token":       chars_per_token,
    }

def get_token_estimate(text_or_len: any, chars_per_token: float = 1.8) -> int:
    """粗略估算文字的 token 用量。支援傳入字串或已計算好的字數(int)。"""
    if not text_or_len: return 0
    length = text_or_len if isinstance(text_or_len, int) else len(text_or_len)
    return int(length / chars_per_token)

def generate_usage_html(usage: dict, max_tokens: int) -> str:
    """生成漂亮的 HTML 顯示 Token 使用量及其細目。"""
    current_tokens = usage.get("total", 0)
    percent = min(100, int((current_tokens / max_tokens) * 100)) if max_tokens > 0 else 0
    color = "#10b981" # Green
    if percent > 60: color = "#f59e0b" # Orange
    if percent > 85: color = "#ef4444" # Red
    
    return f"""
    <div style="margin-top: 15px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 15px; background: #f9fafb; border-radius: 12px; border: 1px solid #e5e7eb; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px;">
            <span style="font-weight: 700; color: #111827;">累計 Token 使用量</span>
            <span style="font-weight: 600; color: {color};">{current_tokens:,} / {max_tokens:,} (已用 {percent}%)</span>
        </div>
        
        <div style="width: 100%; background: #e5e7eb; border-radius: 9999px; height: 10px; overflow: hidden; margin-bottom: 15px;">
            <div style="width: {percent}%; background: {color}; height: 100%; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);"></div>
        </div>

        <div style="font-size: 13px; line-height: 1.6; color: #4b5563;">
            <div style="margin-bottom: 8px;">
                <b style="color: #111827;">📥 輸入 (Input Tokens):</b> <span style="color: #111827; font-weight: 600;">{usage.get('input_prompt', 0) + usage.get('input_context', 0) + usage.get('input_history', 0):,}</span>
                <div style="padding-left: 15px; font-size: 12px; color: #6b7280;">
                    • 你輸入的提示詞: <span style="color: #4b5563;">{usage.get('input_prompt', 0):,}</span><br>
                    • 專案開發環境/代碼上下文: <span style="color: #4b5563;">{usage.get('input_context', 0):,}</span><br>
                    • 歷史對話回顧: <span style="color: #4b5563;">{usage.get('input_history', 0):,}</span>
                </div>
            </div>
            <div style="margin-bottom: 4px;">
                <b style="color: #111827;">📤 輸出 (Output Tokens):</b> <span style="color: #111827; font-weight: 600;">{usage.get('output', 0):,}</span>
            </div>
            <div>
                <b style="color: #111827;">🔧 工具調用 (Tool Results):</b> <span style="color: #111827; font-weight: 600;">{usage.get('tool_results', 0):,}</span>
            </div>
        </div>
    </div>
    """
