# 🤖 Agentic Studio (vlm_skill)

這是一個基於 LangChain 與 LangGraph 開發的多功能 AI Agent 系統，專為處理多模態（Multimodal）任務、OCR 提取、PDF/PPTX 處理、外部工具整合（MCP Server）、長文本 RAG 以及電腦視覺（Computer Vision）生產線架構而設計。

## 🌟 系統架構 (System Architecture)

本專案經過重新整理，採用模組化設計，提升了可維護性與可擴展性：

```text
vlm_skill/
├── app.py                     # 🚀 應用程式進入點 (Main Entry)
├── init_collections.py        # 🗄️ Qdrant 向量庫初始化腳本
├── core/                      # 🧠 核心邏輯層
│   ├── agent.py               # Agent 建立與執行策略、串流與錯誤控制、Token 截斷機制
│   ├── skills.py              # Skill 系統、工具註冊與 Middleware
│   └── memory.py              # 長短期記憶管理 (ThreadMemory)
├── ui/                        # 🎨 使用者介面層 (Gradio UI 與 Handlers)
├── utils/                     # 🛠️ 公用工具庫 (Helpers)
├── config/                    # ⚙️ 系統設定檔目錄 (如 memo.json)
├── skills/                    # 📚 技能庫 (含 computer-vision, memory, rag, mcp_client 等)
├── data/                      # 💾 持久化數據 (包含對話 session JSON 檔案)
├── results/                   # 📁 實驗產出與日誌輸出目錄 (單一資料夾實驗管理)
├── tmp/                       # 🗑️ 暫存檔案處理目錄
└── docs/                      # 📜 詳細技術規格與架構文件
```

> **詳細架構與設計指南**請參閱 [`docs/system_architecture.md`](docs/system_architecture.md)。

## 🚀 核心功能 (Key Features)

### 1. 模組化生產級技能系統 (Skill System)
Agent 可以動態載入並學習新的技能。包含：
- **SKILL.md**: 技能描述與操作指令。
- **scripts/**: 輔助執行的 Python 腳本。
現有技能包含 `computer-vision` (Ultralytics YOLO Expert 生產工作流)、`map_reduce`、`mcp_client`、`memory`、`pdf`、`pptx`、`rag`、`table_ocr_skill` 與 `web_search` 等，具備高度靈活性。
> 詳情見 [`docs/computer_vision_yolo.md`](docs/computer_vision_yolo.md) 與 [`docs/agent_optimizations.md`](docs/agent_optimizations.md)。

### 2. 進階記憶管理 (Memory V2 Architecture)
為解決長對話的 Token 消耗問題與記憶混亂，系統全面重構了記憶架構：
- **Working Memory (短時記憶)**: 導入了 **Model-Aware Truncation (智慧截斷)** 機制，嚴格監控 Tool Output 長度，防止無限迴圈與 Context Window 枯竭。
- **Persistent Memory (長時結構化記憶)**: 升級為 Memory V2 架構，透過 `tags` 與 `all_entities` 的 Payload Schema，搭配**兩階段搜尋邏輯 (Two-Pass Search)**，實現更精準且高效的 Qdrant 語意檢索與標準化 Memory Keys 管理。
> 詳情見 [`docs/memory_v2_architecture.md`](docs/memory_v2_architecture.md)。

### 3. 統一專案與實驗管理 (Standardized Directory & Experiment Management)
所有的檔案收納皆有嚴格規範以確保專案目錄乾淨：
- **`./results`**：模型訓練產出、評估日誌與推論優化紀錄（Inference Optimizer）。
- **`./tmp`**：錄音轉錄或其他中繼暫存檔。
- **`$PROJECT_ROOT` 解析**：全局確保系統能透過絕對路徑解決 Dataset 或配置載入的問題。

### 4. 多模態與文件處理 (Multimodal & Document Processing)
- 支援圖片 OCR 與表格結構辨識 (`table_ocr_skill`)。
- 支援 PDF、PPTX 多種複雜檔案格式解析，強化了檔案編碼相容性（避免多國結字元遺失）。
- 自動判斷上傳檔案類型，引導 Agent 選擇對應的技能進行萃取與問答。

### 5. 錯誤機制與防呆控制 (Error Handling & Loop Protection)
- Agent 被規範遵守 Strict **Skill-First Tool Usage**，不能單純口頭承諾而不呼叫對應 API。
- 具備防死結保護：如果 Agent 陷入無窮迴圈或超過最大執行步數，將會強制中斷，主動呼叫 LLM 產生當前任務執行摘要，詢問用戶後續動作。

### 6. 檢索優先權 (Retrieval Priority)
 Agent 在回答時，會遵循下列指引以確保精準與實效性：
1. **當前上下文 (Short-term)**：若資訊在當面對話內，直接回答。
2. **Agent Memory & User RAG (Long-term)**：涉及「過去開發歷史」、「專案規格」或「工作流程」時，嚴格觸發 `Memory Skill` 或 `RAG Skill` 去查詢 Qdrant。
3. **MCP 伺服器 (External/Real-time)**：讀寫如 GitHub, Slack 或本地環境中的即時資源時使用。
4. **Web Search (Online Knowledge)**：針對外部普遍新知或內部找不到數據時使用。

## 🛠️ 如何使用 (Getting Started)

您可以選擇推薦的 **Docker 容器化環境** 來徹底避免套件衝突，或使用原本的**本地端環境**。

### 🐳 方式一：Docker 容器執行 (推薦)

透過專案提供的 `Dockerfile`，只需兩個指令即可在隔離環境中啟動，並將本地腳本掛載同步：

1. **建置 Docker 映像檔**:
   ```bash
   docker build -t vlm-env .
   ```

2. **啟動容器並掛載資料夾 (Volume Mount)**:
   建議將當前專案目錄掛載到容器內的 `/app`，並映射 Gradio (`7860`) 與外部服務 (`63574`) 的連接埠：
   ```bash
   # Windows PowerShell 寫法 (${PWD}):
   docker run -it --rm -p 7860:7860 -p 63574:63574 -v ${PWD}:/app --gpus all vlm-env
   
   # Linux/macOS 寫法 ($(pwd)):
   # docker run -it --rm -p 7860:7860 -p 63574:63574 -v $(pwd):/app --gpus all vlm-env
   ```
   *注意：加上 `--gpus all` 可讓容器內的 Ultralytics 等套件順利調用宿主機的 NVIDIA GPU 進行自動訓練與預測。預設啟動會自動執行 `python app.py`。*

### 💻 方式二：本地端執行 (Local Execution)

1. **環境配置**:
   安裝基礎套件 (建議使用虛擬環境):
   ```bash
   pip install -r requirements.txt
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

- **技術文件**: 詳閱 `docs/` 目錄下的架構、記憶體機制、CV 實驗追蹤等核心設計。另外可於 `docs/architecture_diagram.md` 檢視系統邏輯架構圖。
- **內部儲存**:
  - 短期 Session：以 JSON 格式儲存於 `data/memory/` 目錄。
  - 長期設定檔與向量配置：請查閱 `config/memo.json` 檔案。
- **技能開發**: 在 `skills/` 下建立新資料夾並寫好 `SKILL.md`，Agent 即可在下次請求時理解該新工具。

## 📌 未來規劃與待辦清單 (TODO / Roadmap)

目前專案處於穩定階段，接下來從「原型實驗」邁向「生產級別」Agent 預計修善與優化的方向包含：

### 優先基礎設施 (Priority Infrastructure)
- [ ] **可觀測性與追蹤 (Observability & Tracing)**：整合 LangSmith、Langfuse 或 Phoenix 等機制，可視化記錄 `Input -> Prompt -> Tool Call -> Result -> Answer` 執行過程與 Token/Latency 消耗，解決單純看 log 追蹤的痛點。
- [ ] **自動化評估系統 (Evaluation Framework / Evals)**：建立標準測試集 (Dataset) 並導入 LLM-as-a-Judge 取代純人工審查。針對工具選擇正確率等指標建立自動化回歸測試 (Regression Testing)。
- [ ] **工具層級容錯與降級機制 (Error Handling & Fallback)**：實作 Tool 的 Try-Catch，當 API Timeout 或發生錯誤時，將錯誤資訊轉化為自然語言回傳給 LLM 讓其嘗試自我修正，並設計兜底降級方案。
- [ ] **安全性與護欄控制 (Safety & Guardrails)**：針對具潛在破壞性的敏感操作（如執行指令或改寫檔案）引入 Human-in-the-loop (HITL) 人工審批機制，並確保執行環境的安全隔離。

### 核心架構重構 (Core Architecture)
- [ ] **中央狀態流轉管理 (State Machine / Orchestration)**：參考 LangGraph 架構，將散落的對話邏輯、工具結果統一封裝為 State (狀態機)，讓 Agent 成為接收並改變狀態的節點，解決中央管理不足的問題。
- [ ] **長期記憶重構 (VectorDB Memory)**：將本地 JSON/TXT 機制升級為向量資料庫 (如 Chroma/Milvus)，並引入混合搜索 (Hybrid Search) 與記憶萃取衰減機制，改善 Token 消耗與擴充痛點。
- [ ] **大文本截斷與摘要 (Dynamic Context Management)**：標準化 Token-efficient truncation 策略，為回傳大量文本的工具加入「摘要層節點」，濃縮內容後再交給主 Agent。

### 智能進化 (Intelligence Evolution)
- [ ] **a2a 多智能體協作 (Multi-Agent System)**：引入 Supervisor 主控架構，建立負責理解與派發任務的 Router 與各領域專長的 Sub-Agent，處理單一 Agent Prompt 過載與幻覺機率。
- [ ] **反省與自我修正機制 (Reflection & Self-Correction)**：導入 Actor-Critic 模式，規定 Agent 在回覆人類或結束工具呼叫前，必須先進行內部驗證機制，若結果不合理需自我觸發 Retry。

### 技能模組優化 (Skill Development & Optimization)
- [ ] **電腦視覺訓練流程優化 (Computer Vision)**：加入重新訓練 (Retraining) 機制，並移除 `hyperparameters.yaml` 以統一配置方式，避免多個 YAML 設定檔造成配置混淆。
- [ ] **自動標註功能完善 (Auto-Labeling)**：接續完成 `skills/auto-labeling` 的核心功能開發與整合，實現自動化標註工作流。

### 其他優化 (Others)
- [ ] **多模態即時串流 (Real-time Multimodal Streaming)**：探索 WebRTC 或其他協議在語音與影像上的全雙工即時對話支持。
- [ ] **GraphRAG 引進**：升級目前的 Memory V2 架構，利用知識圖譜 (Knowledge Graph) 強化複雜邏輯關聯推導。
- [ ] **自動化測試與 CI/CD (E2E Testing)**：針對各獨立模組 (尤其 Computer Vision 訓練流程) 建立整合測試與自動化驗證部署流。
- [ ] **UI/UX 強化**：提供任務的樹狀/圖形化視覺呈現，讓使用者更清楚了解 Agent 正在執行哪些分支步驟與分析進度。

---
*Last updated: 2026-03-29*

