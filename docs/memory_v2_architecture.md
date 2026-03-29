# 🧠 Memory V2 Architecture

## 概述 (Overview)
Memory V2 架構針對長時記憶的儲存與檢索進行了重整，提供了更高效率且精準的記憶調用能力，並解決了過去單純依賴 LLM 自動總結的耗時設計。

## 核心設計 (Core Design)

### 1. 結構化 Payload Schema
記憶的 Payload 設計區分為不同的層級：
- **`tags`**: 用於快速過濾與分類的輕量級標籤。
- **`all_entities`**: 包含詳細對話實體、偏好、與專案狀態的完整結構化內容。
這種設計允許在不載入巨量內容的狀況下進行快速篩選。

### 2. Two-Pass Search Logic (兩階段搜尋邏輯)
為了提升搜尋準確度與效能，系統採用兩階段檢索機制：
- **第一階段 (初篩) - Metadata/Tag Filtering**: 利用 `tags` 或特定的 Metadata 進行快速過濾，縮小候選記憶範圍。
- **第二階段 (精篩) - Semantic Search**: 針對初篩後的結果，進行向量語意相似度比對 (`mem_search.py`)，找出與當前對話最吻合的記憶實體。

### 3. 標準化 Memory Keys
確保所有 Agent 操作長期記憶時的名稱一致性，避免資料庫結構混亂：
- 例如：`brand_preference`, `tone`, `project_context`
- 強制 Agent 在寫入新記憶前，必須先透過檢索檢查現有 Keys，防止資料毀損與冗餘。

## 使用指引 (Usage Guidelines)
- 在自訂記憶存取腳本 (如 `upsert_memory.py`、`mem_search.py`) 時，必須嚴格遵守上述 Payload Schema。
- Session JSON (如 `thread_session1.json`) 的修改與載入應符合標準化命名慣例。
