# 🧠 Memory V3 Architecture

## 概述 (Overview)
Memory V3 架構徹底改變了長期記憶的處理方式，從傳統的「對話結束後批次總結 (Batch Summarization)」轉型為「即時單一回合索引 (Real-time Turn-level Indexing)」。
這一變更大幅降低了記憶延遲、移除了非同步寫入的複雜狀態機 (Archive State Machine)，並採用更先進的混合搜尋管道確保檢索精準度。

## 核心設計 (Core Design)

### 1. 即時回合索引 (Real-time Turn Indexing)
- 去除了複雜的 Chunk 與 Session 雙重檢索結構。
- 當 AI 生成回答且 `record_turn` 被觸發時，該回合的 Metadata (實體、摘要) 與真實對話脈絡會即時非同步寫入 Qdrant。
- `ThreadMemory` 內部的複雜 `archive_chunks` 與狀態紀錄皆被廢除。對話刪除時，系統只需進行直接移除操作。

### 2. 回合層級 Payload (Turn-level Payload Schema)
每個 Qdrant Point 就對應一個 AI 回答回合，確保向量檢索能夠精準到原汁原味的問答上下文。
Payload 設計格式如下：
```json
{
  "type": "turn",
  "session_key": "thread_xxx",
  "summary": "AI 回答之摘要",
  "entities": ["關鍵字1", "Keyword2"],
  "ts_unix": 1729837192,
  "context": {
    "query": "使用者的原始提問 (經過 Think 標籤清理)",
    "answer": "AI 原始作答內容 (上限 3000 字)"
  }
}
```

### 3. 混合搜尋管道 (Hybrid Search Pipeline)
V3 架構使用進階搜尋策略替代舊有的兩階段初篩/精篩機制：
- **第一階段定位 (Vector + BM25 Search)**: 利用 Embedding Vector 找尋語意相似的回合，並搭配實體提取機制 `Entities` 作為 BM25 的 Payload Match 確保精確關鍵字過濾。
- **時間權重衰減 (Time Decay Filtering)**: 透過 `ts_unix` 欄位將點位依照時間流逝打折，確保最新資訊擁有較高優先權。公式控制於純向量距離與 Recency 中做 Blend (例如 `0.25` Time weight 配置)。
- **LLM 重排序 (Reranking)**: 針對初步搜出結果，使用外部輕量型 API (如 Cohere 或 Qwen) 作二次排列，輸出精準 Top-K 給 Context 引擎。

## 使用指引 (Usage Guidelines)
- 在自訂記憶存取腳本 (如 `mem_search.py`) 時，預設模式已切換為 `turn` 的單一回合混合檢索。
- 不再需要手動執行 Archiving，`ThreadMemory` 的資料夾也不會產生臃腫的歸檔檔案。
- 為了確保提取的品質，推理模型輸出的 `<think>` 過程與無意義代名詞在索引階段已徹底過濾。
