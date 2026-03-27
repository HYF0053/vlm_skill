# 長期記憶系統架構升級說明 (Memory V2)

本文件說明 `vlm_skill` 長期記憶系統在 2026 年 3 月進行的重大架構優化。

## 1. 核心改進點概要

針對舊版系統搜尋片段化、語意斷裂以及缺乏時間感知等問題，V2 版本引入了以下三大核心技術：
- **即時萃取 (Turn-level Metadata)**：對話中自動、零延遲地產出摘要與實體指標。
- **階層式索引 (Hierarchical Indexing)**：在 Qdrant 建立 Session-level 與 Chunk-level 的雙向鏈結結構。
- **兩步式 Top-K 搜尋 (Two-pass Search)**：先定位會話 (Session) 再搜尋片段 (Chunk)，並引入時間衰減評分。

---

## 2. 詳細技術實作

### 2.1 即時萃取模組 (`core/turn_meta.py`)
為了解決封存時才批次產生摘要導致的細節丟失問題，現在系統在每一輪 AI 回答後，會即時執行純文字萃取：
- **零 LLM 呼叫**：使用正則表達式 (Regex) 與高頻詞統計，確保 UI 無延遲。
- **萃取內容**：
    - `summary`: 「用戶問題 → AI 精簡回答」的結構化摘要。
    - `entities`: 自動識別 CJK 專有名詞與英文術語。
    - `intent`: 識別用戶意圖（explain/create/compare/fix 等）。
    - `time_refs`: 抓取文字中的時間詞（今天、昨天、2026-xx-xx）。
- **儲存位置**：直接附加在 `recent_messages` 裡 AI 訊息的 `turn_meta` 欄位中。

### 2.2 兩層向量索引結構
封存流程 (`archive_and_summarize_session`) 現在會在 Qdrant 建立兩種不同層級的 Point：

1.  **Session Summary Point** (`type="session_summary"`)
    - 儲存該 Session 的全局摘要、所有實體合集與轉帳輪數。
    - 作為搜尋的「入口點」，解決跨會話定位問題。
2.  **Chunk Point** (`type="chunk"`)
    - 帶有 `prev_chunk_id` 與 `next_chunk_id` 指標。
    - 帶有 `session_point_id` 反查 session 概覽。
    - 帶有 `indexed_at_unix` 用於時間權重排序。

### 2.3 搜尋策略升級 (`core/mem_scripts/mem_search.py`)
搜尋引擎現在支援 `--mode two-pass`（預設開啟）：
1.  **會話過濾**：先從 `session_summary` 中找到最相關的 3 個 Session。
2.  **精準搜尋**：在該 3 個 Session 內尋找 Top-K 個相關 Chunk（即使它們在時間上相距甚遠也能被一次抓齊）。
3.  **時間衰減 (Time-Decay)**：
    - 計算 `blended_score = (1-α) * semantic_score + α * recency_score`。
    - 確保「最新」的修正與決策能夠排在舊訊息之前。
4.  **上下文補全**：自動拉取 Top-K 核心結果前後的相鄰 Chunk，提供 AI 連貫的閱讀體驗。

---

## 3. 使用範例 (CLI)

### 3.1 進階搜尋
```bash
# 基本兩步搜尋（語意預設佔 75%，時間佔 25%）
python core/mem_scripts/mem_search.py "RTX 4090 的選購建議" --mode two-pass

# 增加時間權重（當你想要搜尋最新的偏好設定時，將 weight 調高）
python core/mem_scripts/mem_search.py "我的模型設定" --time-weight 0.40

# 展開更多前後文（展開前後 2 個 chunk）
python core/mem_scripts/mem_search.py "開發計畫" --context-window 2
```

### 3.2 讀取完整對話
```bash
# 根據搜尋結果中提示的指令讀取完整 JSON 內容
python core/mem_scripts/mem_search.py read_chunk 'data/memory/archived/2026-03-27_sess_xxx.json' '[5,9]'
```

---

## 4. 維護與相容性
- **向下相容**：系統偵測到舊版（無 metadata）的 Point 時，會自動降級到 `simple` 搜尋模式。
- **崩潰恢復**：封存流程中加入了 `session_point_id` 與 `archive_state` 的檢查點，若後台程式當機，下次重啟會自動續傳。

---
*Last Updated: 2026-03-27*
