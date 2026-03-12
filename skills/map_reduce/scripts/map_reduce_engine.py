"""
map_reduce_engine.py
====================
核心 Map-Reduce 引擎，可被其他腳本調用。

設計原則
--------
- 無外部依賴（只用 requests 和 python 標準庫）
- 支援任意長度文字
- 自動計算 chunk 數量
- 兩階段 Reduce（若 Map 結果仍太長則再壓一次）
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """LLM 連線設定，優先讀環境變數，fallback 到預設值。"""
    base_url: str = field(
        default_factory=lambda: os.environ.get("VLLM_BASE_URL", "http://10.1.1.7:9000/v1")
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("VLLM_API_KEY", "EMPTY")
    )
    model: str = field(
        default_factory=lambda: os.environ.get(
            "VLLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct"
        )
    )
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 120       # 秒


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_chunks(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """
    把長文切成有重疊的 chunks。

    Parameters
    ----------
    text       : 原始文字
    chunk_size : 每塊的目標字元數
    overlap    : 相鄰 chunk 重疊的字元數（防止邊界資訊遺失）

    Returns
    -------
    list of str — 切好的 chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # 嘗試在自然邊界（句號、換行）切割，避免句子被截斷
        if end < len(text):
            # 向後找最近的「。.！!？?\n」
            natural_end = _find_natural_boundary(text, end, search_back=min(200, overlap))
            if natural_end > start:
                end = natural_end
                chunk = text[start:end]

        chunks.append(chunk)
        start = end - overlap
        if start >= len(text) - overlap:
            break

    return chunks


def _find_natural_boundary(text: str, pos: int, search_back: int = 200) -> int:
    """從 pos 往前找最近的自然切割點。"""
    search_start = max(0, pos - search_back)
    segment = text[search_start:pos]
    # 從右往左找標點
    for i in range(len(segment) - 1, -1, -1):
        if segment[i] in "。.！!？?\n":
            return search_start + i + 1
    return pos  # fallback: 原始位置


# ---------------------------------------------------------------------------
# LLM Caller（純 requests，不依賴 LangChain）
# ---------------------------------------------------------------------------

def _call_llm(
    messages: list[dict],
    config: LLMConfig,
    log_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """呼叫 OpenAI-compatible API，回傳回覆文字。"""
    import requests

    url = config.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=config.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        if log_fn:
            log_fn(f"  ⚠️  LLM call failed: {e}")
        return f"[ERROR: {e}]"


# ---------------------------------------------------------------------------
# Map phase
# ---------------------------------------------------------------------------

def map_chunk(
    chunk: str,
    task: str,
    chunk_idx: int,
    total_chunks: int,
    config: LLMConfig,
    log_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """對單一 chunk 執行 Map 任務。"""
    system = (
        "你是一個精確的文件分析助手。"
        "你的工作是針對給定的文字片段執行任務，只輸出結果，不要加任何說明或前綴。"
    )
    user = (
        f"【任務】{task}\n\n"
        f"【文字片段 {chunk_idx}/{total_chunks}】\n"
        f"{chunk}\n\n"
        "請根據任務說明處理上面的片段，只輸出結果："
    )

    if log_fn:
        log_fn(f"[MAP] Chunk {chunk_idx}/{total_chunks} ({len(chunk)} chars) → processing...")

    t0 = time.time()
    result = _call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        config,
        log_fn,
    )
    elapsed = time.time() - t0

    if log_fn:
        log_fn(f"[MAP] Chunk {chunk_idx}/{total_chunks} done ({elapsed:.1f}s, {len(result)} chars output)")

    return result


# ---------------------------------------------------------------------------
# Reduce phase
# ---------------------------------------------------------------------------

def reduce_results(
    map_outputs: list[str],
    task: str,
    config: LLMConfig,
    log_fn: Optional[Callable[[str], None]] = None,
    max_reduce_chars: int = 6000,
) -> str:
    """
    把所有 Map 結果合併成最終輸出。

    如果合併後仍然太長，自動再做一次 Reduce（最多兩層）。
    """
    combined = "\n\n---\n\n".join(
        f"[片段 {i+1} 結果]\n{r}" for i, r in enumerate(map_outputs)
    )

    if log_fn:
        log_fn(f"[REDUCE] Combining {len(map_outputs)} results ({len(combined)} chars)...")

    # 如果合併結果太長，先做中間層 reduce
    if len(combined) > max_reduce_chars:
        if log_fn:
            log_fn(f"[REDUCE] Combined too long ({len(combined)} chars), doing intermediate reduce...")
        # 把 map_outputs 切一半再各自 reduce
        mid = len(map_outputs) // 2
        left  = reduce_results(map_outputs[:mid],  task, config, log_fn, max_reduce_chars)
        right = reduce_results(map_outputs[mid:], task, config, log_fn, max_reduce_chars)
        combined = f"[前半總結]\n{left}\n\n[後半總結]\n{right}"

    system = (
        "你是一個精確的文件分析助手。"
        "你的工作是把多個文字片段的處理結果整合成一份完整、連貫的最終輸出。"
        "消除重複內容，保持邏輯順序，只輸出整合後的結果。"
    )
    user = (
        f"【原始任務】{task}\n\n"
        f"【各片段的處理結果】\n{combined}\n\n"
        "請將上面所有片段的結果整合成一份完整的最終輸出："
    )

    result = _call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        config,
        log_fn,
    )

    if log_fn:
        log_fn(f"[REDUCE] Final output: {len(result)} chars")

    return result


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MapReduceResult:
    chunks: list[str]
    map_outputs: list[str]
    final_output: str
    total_input_chars: int
    total_chunks: int
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class MapReduceEngine:
    """
    Map-Reduce 長文處理引擎。

    Usage
    -----
    engine = MapReduceEngine(LLMConfig())
    result = engine.run(long_text, task="請摘要重點")
    print(result.final_output)
    """

    def __init__(self, config: LLMConfig = None, log_fn: Optional[Callable[[str], None]] = None):
        self.config = config or LLMConfig()
        self.log_fn = log_fn or print

    def run(
        self,
        text: str,
        task: str,
        chunk_size: int = 1500,
        overlap: int = 200,
    ) -> MapReduceResult:
        """
        執行完整的 Map-Reduce 流程。

        Parameters
        ----------
        text       : 輸入的長文字
        task       : 對每個 chunk 的任務描述（也用作 Reduce 的任務描述）
        chunk_size : 每塊字元數
        overlap    : 相鄰 chunk 重疊字元數

        Returns
        -------
        MapReduceResult
        """
        t0 = time.time()
        total_chars = len(text)

        # ── Split ──────────────────────────────────────────────────────────
        chunks = split_chunks(text, chunk_size=chunk_size, overlap=overlap)
        self.log_fn(f"[MAP-REDUCE] Input: {total_chars} chars → {len(chunks)} chunks")

        # ── Map ────────────────────────────────────────────────────────────
        map_outputs: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            result = map_chunk(
                chunk=chunk,
                task=task,
                chunk_idx=i,
                total_chunks=len(chunks),
                config=self.config,
                log_fn=self.log_fn,
            )
            map_outputs.append(result)

        # ── Reduce ─────────────────────────────────────────────────────────
        if len(map_outputs) == 1:
            # Only one chunk: no reduce needed
            final = map_outputs[0]
            self.log_fn("[REDUCE] Only 1 chunk, skipping reduce.")
        else:
            final = reduce_results(
                map_outputs=map_outputs,
                task=task,
                config=self.config,
                log_fn=self.log_fn,
            )

        elapsed = time.time() - t0
        self.log_fn(f"[MAP-REDUCE] Done in {elapsed:.1f}s")

        return MapReduceResult(
            chunks=chunks,
            map_outputs=map_outputs,
            final_output=final,
            total_input_chars=total_chars,
            total_chunks=len(chunks),
            elapsed_seconds=elapsed,
        )
