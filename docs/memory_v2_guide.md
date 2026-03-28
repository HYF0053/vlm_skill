# 長期記憶系統架構升級說明 (Memory V2)

本文件說明 `vlm_skill` 長期記憶系統的核心架構，特別針對 2026 年 3 月優化後的「會話-片段」二層模型進行定義。

## 1. 核心流程概要

針對舊版系統搜尋片段化、語義斷裂等問題，V2 版本將記憶分為「索引點」與「原始內容」：
- **即時萃取 (Turn-level Metadata)**：每一輪對話後，自動由 `core/turn_meta.py` 產出摘要與實體指標。
- **階層式索引 (Hierarchical Indexing)**：在 Qdrant 建立 Session-level 與 Chunk-level 的對應結構。
- **兩步式檢索 (Two-pass Search)**：先依據語意與實體權重過濾會話 (Session)，再深入定位對話片段 (Chunk)。

---

## 2. 詳細技術實作與 Payload 結構

### 2.1 即時萃取模組 (`core/turn_meta.py`)
系統在每一輪 AI 回答後執行純文字萃取（零 LLM 負擔）：
- **summary**: 格式為 「用戶問題 → AI 預覽」。
- **entities**: 萃取 CJK 常用名詞與英文術語。
- **intent**: 識別意圖（explain/create/compare/fix 等）。
- **time_refs**: 抓取時間詞（今天、2026-xx-xx）。

### 2.2 Qdrant 儲存 Schema
在封存 (Archive) 時，Qdrant 內的 Payload 命名約定如下（請注意名稱差異）：

#### 1. Session Summary Point (`type="session_summary"`)
- **all_entities**: 該 Session 出現次數最多的前 30 個實體（回想時的主索引）。
- **full_summary**: 前 5 輪對話的彙整摘要。
- **session_key**: 作為與 Chunk 反向連結的 ID。

#### 2. Chunk Point (`type="chunk"`)
- **tags**: 對應單一 Turn 的實體 (Entities)。在此層級稱之為 `tags`。
- **prev_chunk_id / next_chunk_id**: 用於擴展前後文的雙向指標。
- **importance_score**: 根據內容長度計算的 1-9 分權重。
- **indexed_at_unix**: 檢索時計算「時間衰減」的時間戳記。

---

## 3. 搜尋策略細節 (`mem_search.py`)

搜尋引擎預設支援 `--mode two-pass`：
1.  **會話過濾 (Pass 1)**：先從 `session_summary` 中找到最相關的 3 個 Session。
2.  **精準搜尋 (Pass 2)**：在該 3 個 Session 內細挖相關 Chunk（即使時間相距甚遠也能被一次抓齊）。
3.  **分數混合與時間衰減 (Time-Decay)**：
    - `blended_score = (1-α) * semantic_score + α * recency_score`。
    - 確保「最新」的修正與偏好設定（如 ASUS vs MSI 的決策）能排在舊訊息之前。
4.  **上下文補全**：根據 Chunk 的導航指標，自動拉取 Top-K 結果前後的連貫對話。

---

## 4. 維護與故障排除
- **Payload 找不到 entities 欄位？**：請檢查 `all_entities` (Session 級別) 或 `tags` (Chunk 級別)。
- **搜尋不到最新對話？**：請確認對話是否已進入 `archive_state: done`。
- **如何讀取原始資料？**：透過 `read_chunk` 命令讀取 Payload 中 `archived_file` 指定的完整 JSON。

---
*Last Updated: 2026-03-28*

