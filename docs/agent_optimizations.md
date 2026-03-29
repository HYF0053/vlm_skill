# 🤖 Agent Optimizations & Tooling

## 概述 (Overview)
為確保 Agent 在處理複雜多步驟任務與長文本時的穩定性，我們對系統進行了全方位的最佳化。

## 1. Skill-First Tool Usage (技能優先工具調用)
Agent 在解決問題時，被嚴格規範必須優先使用系統內建的標準化「技能 (Skill)」與腳本，禁止 Agent 單純以口頭承諾卻未實際呼叫對應 工具/API。
- **核心修正 (`core/agent.py`)**: 重構回應處理邏輯，確保原始回覆在經過任何文字過濾之前，工具呼叫就能被正確解析並執行。
- **嚴格 Prompting**: 限制 Agent 不得說出「已更新記憶」等字眼，除非同一回答中夾帶了 `upsert_memory` 工具呼叫。

## 2. Token-Efficient Truncation Strategy (智慧截斷機制)
過去長對話容易導致 Context Window 耗盡，甚至引發 Agent 進入無窮迴圈：
- **Model-Aware Truncation**: 針對目前調用的 LLM 模型，精確計算 Token 使用量。當逼近上限時，自動執行截斷作業。
- **Tool Output Truncation**: 特別針對工具輸出 (如大量的 Log、長的 JSON 或 Web 搜尋結果) 進行智慧縮略，僅保留前段關鍵資訊或摘要，防止垃圾資訊佔滿上下文空間。

## 3. 防死結與迴圈保護 (Deadlock & Loop Protection)
- Agent 如果無法在指定步數內完成運算，或偵測到重複相近的執行路徑，會強制中斷執行迴圈。
- 此時 Agent 將主動調用 LLM 產生「當前任務執行狀態總結」，並暫停行動，將決策權交還給用戶，避免無謂的 API 成本消耗。
