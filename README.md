# 🤖 Agentic Studio (vlm_skill)

這是一個基於 LangChain 與 LangGraph 開發的多功能 AI Agent 系統，專為處理多模態（Multimodal）任務、OCR 提取、PDF/PPTX 處理、外部工具整合（MCP Server）以及長文本 RAG 而設計。

## 🌟 系統架構 (System Architecture)

本專案經過重新整理，採用模組化設計，提升了可維護性與可擴展性：

```text
vlm_skill/
├── app.py                     # 🚀 應用程式進入點 (Main Entry)
├── init_collections.py        # 🗄️ Qdrant 向量庫初始化腳本
├── core/                      # 🧠 核心邏輯層
│   ├── agent.py               # Agent 建立與執行策略、串流與錯誤控制
│   ├── skills.py              # Skill 系統、工具註冊與 Middleware
│   └── memory.py              # 長短期記憶管理 (ThreadMemory) 與截斷機制
├── ui/                        # 🎨 使用者介面層 (Gradio UI 與 Handlers)
├── utils/                     # 🛠️ 公用工具庫 (Helpers)
├── config/                    # ⚙️ 系統設定檔目錄 (如 memo.json)
├── skills/                    # 📚 技能庫 (含 memory, rag, mcp_client, web_search 等)
└── data/                      # 💾 持久化數據 (包含對話 session JSON 檔案)
```

## 🚀 核心功能 (Key Features)

### 1. 模組化技能系統 (Skill System)
Agent 可以動態載入並學習新的技能。包含：
- **SKILL.md**: 技能描述與操作指令。
- **scripts/**: 輔助執行的 Python 腳本。
現有技能包含 `map_reduce`、`mcp_client`、`memory`、`pdf`、`pptx`、`rag`、`table_ocr_skill` 與 `web_search` 等，具備高度靈活性。

### 2. 進階記憶管理 (Memory Management)
為解決長對話的 Token 消耗問題，系統全面重構了記憶架構：
- **Working Memory (短時記憶)**: 使用 `InMemorySaver` 與檔案鎖來存取，導入了 **Smart Trimming (智慧截斷)** 機制。僅保留對話最前期的錨點訊息和最新的數則回應，自動修剪冗長過期的歷史紀錄。
- **Persistent Memory (長時結構化記憶)**: 透過專屬的 `Memory Skill` (整合 Qdrant)，將使用者的偏好、事實、代理原則與專案狀態轉化為結構化數據，取代舊版單純依賴 LLM 自動總結的耗時設計，隨時精準調用。

### 3. 多模態與文件處理 (Multimodal & Document Processing)
- 支援圖片 OCR 與表格結構辨識 (`table_ocr_skill`)。
- 支援 PDF、PPTX 多種複雜檔案格式解析，強化了檔案編碼相容性（避免多國結字元遺失）。
- 自動判斷上傳檔案類型，引導 Agent 選擇對應的技能進行萃取與問答。

### 4. 錯誤機制與串流回饋 (Error Handling & Streaming)
- 支援即時的 AI Token Stream，以及工具執行的詳細日誌跟蹤，並即時運算 Token 成本。
- 具備防死結保護：如果 Agent 陷入無窮迴圈或超過最大執行步數，將會強制中斷，並主動呼叫 LLM 產生當前任務執行摘要，詢問用戶是否要在此總結或繼續任務。

### 5. 檢索優先權 (Retrieval Priority)
 Agent 在回答時，會遵循下列指引以確保精準與實效性：
1. **當前上下文 (Short-term)**：若資訊在當面對話內，直接回答。
2. **Agent Memory & User RAG (Long-term)**：涉及「過去開發歷史」、「專案規格」或「工作流程」時，嚴格觸發 `Memory Skill` 或 `RAG Skill` 去查詢 Qdrant。
3. **MCP 伺服器 (External/Real-time)**：讀寫如 GitHub, Slack 或本地環境中的即時資源時使用。
4. **Web Search (Online Knowledge)**：針對外部普遍新知或內部找不到數據時使用。

## 🛠️ 如何使用 (Getting Started)

1. **環境配置**:
   安裝基礎套件 (可能需根據技能個別補充):
   ```bash
   pip install gradio langchain langchain-openai langgraph pillow requests pyyaml
   ```

2. **初始化向量庫 (選擇性)**:
   如果您要啟用基於 Qdrant 的 Memory 或 RAG 技能：
   ```bash
   python init_collections.py
   ```

3. **啟動伺服器**:
   ```bash
   python app.py
   ```

4. **介面操作**:
   - 前往 **Settings** 標籤設定 LLM API 結點並指定對應的模型名稱。
   - 使用 **Session Key** 來切分或接續不同的工作對話。
   - 若對話超過一定長度或遇到瓶頸，可善用 UI 的清除機制，長時知識依然會保留在記憶向量內。
   - 觀看底端 **Execution Logs** 瞭解 AI 思考、調用的工具及使用資源（Tokens）。

## 📝 維護與擴展 (Maintenance)

- **日誌追蹤**: 透過詳細的 stdout 或對話視窗下方的 logs 追蹤 Agent 狀態。
- **內部儲存**:
  - 短期 Session：以 JSON 格式儲存於 `data/memory/` 目錄，可直接查閱原始對話。
  - 長期設定檔與向量配置：請查閱 `config/memo.json` 檔案。
- **技能開發**: 在 `skills/` 下建立新資料夾並寫好 `SKILL.md`，Agent 即可在下次請求時理解該新工具。

---
*Last updated: 2026-03-19*
