#!/usr/bin/env python3
"""
run_map_reduce.py
=================
CLI 入口腳本，由 AI agent 透過 execute_script() 呼叫。

用法
----
python run_map_reduce.py <輸入檔> <任務描述> [輸出檔] [chunk_size] [overlap]

範例
----
python run_map_reduce.py /tmp/report.txt "請摘要重點" /tmp/summary.txt
python run_map_reduce.py /tmp/ocr.txt "提取所有金額和日期，JSON格式" /tmp/data.json 800 100
"""

import sys
import os
import time

# 確保能 import 同目錄下的 engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_reduce_engine import MapReduceEngine, LLMConfig, split_chunks


def main():
    # ── 參數解析 ──────────────────────────────────────────────────────────
    if len(sys.argv) < 3:
        print("Usage: run_map_reduce.py <input_file> <task> [output_file] [chunk_size] [overlap]")
        print()
        print("Examples:")
        print('  python run_map_reduce.py /tmp/report.txt "請摘要重點" /tmp/summary.txt')
        print('  python run_map_reduce.py /tmp/ocr.txt "提取所有金額和日期，JSON格式" /tmp/out.json 800 100')
        sys.exit(1)

    input_file  = sys.argv[1]
    task        = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    chunk_size  = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
    overlap     = int(sys.argv[5]) if len(sys.argv) > 5 else 200

    # ── 讀取輸入 ──────────────────────────────────────────────────────────
    if not os.path.isfile(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        # 嘗試 latin-1 fallback
        with open(input_file, "r", encoding="latin-1") as f:
            text = f.read()

    if not text.strip():
        print("ERROR: Input file is empty.")
        sys.exit(1)

    print(f"[MAP-REDUCE] Input file  : {input_file}")
    print(f"[MAP-REDUCE] Input size  : {len(text)} chars")
    print(f"[MAP-REDUCE] Task        : {task}")
    print(f"[MAP-REDUCE] Chunk size  : {chunk_size} chars (overlap={overlap})")
    print(f"[MAP-REDUCE] Output      : {output_file or '(stdout)'}")

    # ── 預覽 chunk 計畫 ───────────────────────────────────────────────────
    preview_chunks = split_chunks(text, chunk_size=chunk_size, overlap=overlap)
    print(f"[MAP-REDUCE] Planned chunks: {len(preview_chunks)}")
    print()

    # ── 執行 ──────────────────────────────────────────────────────────────
    config = LLMConfig()   # 讀環境變數 VLLM_BASE_URL / VLLM_API_KEY / VLLM_MODEL
    engine = MapReduceEngine(config=config, log_fn=print)

    result = engine.run(
        text=text,
        task=task,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # ── 輸出 ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("=== MAP OUTPUTS (各 chunk 中間結果) ===")
    print("=" * 60)
    for i, out in enumerate(result.map_outputs, 1):
        print(f"\n--- Chunk {i}/{result.total_chunks} ---")
        print(out)

    print()
    print("=" * 60)
    print("=== FINAL OUTPUT ===")
    print("=" * 60)
    print(result.final_output)
    print("=" * 60)
    print(f"[MAP-REDUCE] Total time: {result.elapsed_seconds:.1f}s")
    print(f"[MAP-REDUCE] Input: {result.total_input_chars} chars → {result.total_chunks} chunks → {len(result.final_output)} chars output")

    # ── 寫入輸出檔 ────────────────────────────────────────────────────────
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.final_output)
        print(f"[MAP-REDUCE] Saved to: {output_file}")


if __name__ == "__main__":
    main()
