"""
ASR Transcription Script
使用 vLLM 部署的 Qwen3-ASR 模型進行語音轉文字。
基於 OpenAI Audio Transcriptions API 相容介面。

Reference: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html#using-openai-sdk

注意：Qwen3-ASR 僅支援 response_format="text"，不支援 verbose_json。
"""

import argparse
import os
import sys
import json


# 取得專案根目錄：優先讀 app.py 設置的環境變數，fallback 用 __file__ 推算
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def resolve_path(path: str) -> str:
    """若 path 為相對路徑，以 PROJECT_ROOT 為基準轉為絕對路徑。"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def get_first_model(asr_url: str) -> str:
    """從 API 取得第一個可用的模型名稱。"""
    try:
        import requests
        url = f"{asr_url.rstrip('/')}/v1/models"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if models:
                return models[0]["id"]
    except Exception as e:
        print(f"[警告] 無法取得模型列表: {e}", file=sys.stderr)
    return "Qwen/Qwen3-ASR"


def transcribe(
    audio_path: str,
    asr_url: str,
    model: str = None,
    language: str = None,
    output_path: str = None,
) -> str:
    """
    使用 OpenAI 相容 API 將音頻檔轉換為文字。

    Args:
        audio_path: 音頻檔案路徑
        asr_url: ASR API URL (如 http://10.1.1.7:8000)
        model: 模型名稱，None 時自動偵測
        language: 語言代碼 (如 'zh', 'en')，None 時自動偵測
        output_path: 結果輸出路徑（可選），應使用 results/ 目錄

    Returns:
        轉錄的文字內容
    """
    from openai import OpenAI

    # 建立 OpenAI client，指向 vLLM endpoint
    base_url = asr_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    client = OpenAI(
        api_key="EMPTY",  # vLLM 不需要真實 API key
        base_url=base_url,
    )

    # 自動偵測模型
    if not model:
        model = get_first_model(asr_url)
        print(f"[INFO] 使用模型: {model}", file=sys.stderr)

    # 確認音頻檔案存在（相對路徑以 PROJECT_ROOT 為基準）
    audio_path = resolve_path(audio_path)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音頻檔案不存在: {audio_path}")

    # 呼叫 ASR API（Qwen3-ASR 僅支援 response_format="text"）
    with open(audio_path, "rb") as audio_file:
        kwargs = {
            "model": model,
            "file": audio_file,
            "response_format": "text",
        }
        if language:
            kwargs["language"] = language

        transcription = client.audio.transcriptions.create(**kwargs)

    # 取得文字結果
    result = transcription.text if hasattr(transcription, "text") else str(transcription)

    # 輸出到檔案（可選）
    if output_path:
        output_path = resolve_path(output_path)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"[INFO] 結果已儲存至: {output_path}", file=sys.stderr)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="語音轉文字 (ASR) - 使用 vLLM Qwen3-ASR"
    )
    parser.add_argument(
        "--audio_path",
        required=True,
        help="音頻檔案路徑 (支援 mp3, wav, flac, m4a, ogg, webm)",
    )
    parser.add_argument(
        "--asr_url",
        default=None,
        help="ASR API URL (例如 http://10.1.1.7:8000)。未指定時從環境變數 ASR_API_URL 讀取。",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="模型名稱 (例如 Qwen/Qwen3-ASR)。未指定時自動從 API 取得。",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="語言代碼 (例如 zh, en)。未指定時自動偵測。",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="輸出文字檔路徑 (可選)。建議使用 results/ 目錄，例如 results/transcription.txt。",
    )

    args = parser.parse_args()

    # 決定 ASR URL
    asr_url = args.asr_url or os.environ.get("ASR_API_URL", "http://localhost:8000")

    try:
        result = transcribe(
            audio_path=args.audio_path,
            asr_url=asr_url,
            model=args.model,
            language=args.language,
            output_path=args.output_path,
        )
        print(result)
    except FileNotFoundError as e:
        print(f"錯誤: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ASR 轉錄失敗: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
