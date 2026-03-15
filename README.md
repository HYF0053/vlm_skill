# 🤖 Agentic Studio (vlm_skill)

這是一個基於 LangChain 與 LangGraph 開發的多功能 AI Agent 系統，專為處理多模態（Multimodal）任務、OCR 提取、PDF/PPTX 處理以及長文本 RAG 而設計。

## 🌟 系統架構 (System Architecture)

本專案經過重新整理，採用模組化設計，提升了可維護性與可擴展性：

```
vlm_skill/
├── app.py                     # 🚀 應用程式進入點 (Main Entry)
├── core/                      # 🧠 核心邏輯層
│   ├── agent.py               # Agent 建立與執行策略
│   ├── skills.py              # Skill 系統、工具註冊與 Middleware
│   └── memory.py              # 長短期記憶管理 (MemoryStore) 與摘要機制
├── ui/                        # 🎨 使用者介面層
│   ├── interface.py           # Gradio 介面定義
│   └── handlers.py            # UI 事件處理與核心橋接
├── utils/                     # 🛠️ 公用工具庫
│   └── helpers.py             # 圖片處理、LLM 檢測與數值計算
├── skills/                    # 📚 技能庫 (SKILL.md & 輔助腳本)
├── data/                      # 💾 持久化數據 (記憶儲存)
└── old_code/                  # 📦 舊版程式備份
```

## 🚀 核心功能 (Key Features)

### 1. 模組化技能系統 (Skill System)
Agent 可以動態載入並學習新的技能。每個技能包含：
- **SKILL.md**: 技能描述與操作指令。
- **scripts/**: 輔助執行的 Python 腳本。
- **tools**: 可選的 CLI 工具呼叫。

### 2. 進階記憶管理 (Memory Management)
為了解決長對話導致的 Token 爆炸問題，系統實作了雙層記憶架構：
- **Working Memory**: 使用 LangGraph 的 `InMemorySaver` 儲存目前對話。
- **Persistent Memory**: 當對話達到門檻時，會自動觸發 **LLM 摘要壓縮**，將歷史訊息轉化為長期記憶，並修剪運行時記憶以節省 Context Window。

### 3. 多模態支援 (Multimodal)
- 支援圖片 (OCR)、PDF、PPTX 等多種檔案格式。
- 自動判斷檔案類型（圖片、文字、二進位文件），並引導 Agent 使用適當的技能處理。

### 4. 靈活的 LLM 支援
- 支援 **Ollama** 與 **vLLM** (或任何 OpenAI 兼容的 API)。
- 自動偵測可用模型及其 Context 長度。

## 🛠️ 如何使用 (Getting Started)

1. **安裝依賴**:
   ```bash
   pip install gradio langchain langchain-openai langgraph pillow requests pyyaml
   ```

2. **執行程式**:
   ```bash
   python app.py
   ```

3. **介面操作**:
   - 在 **Settings** 標籤中設定您的 LLM API URL 並選取模型。
   - 在 **Agent** 標籤中輸入您的問題或上傳檔案。
   - 使用 **Session Key** 來管理不同的對話實施（相同 Key 可接續記憶）。
   - 在 **Skill Editor** 中可以直接修改或新增 Agent 的技能。

## 📝 維護與除錯 (Maintenance & Debugging)

- **日誌查看**: 執行時，介面下方的 "Execution Logs" 會顯示 Agent 的詳細思考路徑與工具執行結果。
- **記憶修改**: 所有的長期記憶都以 JSON 格式儲存在 `data/memory/` 目錄下。
- **技能開發**: 只要在 `skills/` 下建立新目錄並放入 `SKILL.md`，Agent 下次啟動（或重新整理模型）後即可學習新功能。

---
*Last updated: 2026-03-14*
