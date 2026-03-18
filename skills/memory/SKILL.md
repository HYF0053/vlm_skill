---
name: memory
description: Manage long-term memory (Qdrant) and global preferences (JSON). URGENT: You MUST explicitly call the memory tool when rules/facts are mentioned or bugs are solved. DO NOT just reply "I will remember."
---

## 🚨 致命問題與嚴格行動準則 (CRITICAL Execution Timing)

根據過去的行為紀錄，**你經常只在口頭上說「好的，我會記住」，卻沒有實際呼叫工具，導致記憶完全遺失！**
為了徹底解決這個問題，請嚴格遵守以下「強制執行時機」：

1. **收到 JSON 規則 / 偏好時：**
   - 當使用者說：「之後都...」、「這是我習慣的格式...」、「你要遵守...」。
   - ⚠️ **絕對不准先回答聊天！**你必須在看到提示的**當下這個回合 (turn)**，立刻主動呼叫 `upsert_memory.py`。
   
2. **獲得 Qdrant 專案重要資訊時：**
   - 當你知道了這個專案的技術棧、DB Schema、或特別的設定時。
   - ⚠️ **不要等使用者下令「幫我記起來」才存！**只要資訊有利於後續開發，你就必須**主動**使用 `upsert_memo_qdrant.py -c document_memory` 存下它。

3. **解決困難 Bug / 完成複雜重構時 (skill_workflow) ★最常忘記被記錄：**
   - 當你花費多步完成了某段程式碼的修正，並且**測試/驗證通過**的那一刻。
   - ⚠️ **絕對不要立刻對使用者說「任務完成」並結束對話，這樣工具就關閉了！**
   - 你的倒數第二步動作，**必須**是先呼叫 `upsert_memo_qdrant.py -c skill_workflow` 將除錯經驗存入記憶庫。只有在「呼叫工具並確認儲存成功」之後，你才能發出結尾回覆「我已完成修正」。

---

## 什麼時候該觸發記憶？ (Trigger Patterns)

看到以下【明確句型】或【情境】時，請立刻採取行動（呼叫對應的腳本）：

- **「記住...」 / 「幫我記下來...」** -> (存入 Qdrant `document_memory`)
- **「以後都...」 / 「之後這專案都要...」** -> (存入 JSON `agent_rules`)
- **「這是規則」 / 「規定是...」** -> (存入 JSON `preference` 或 `agent_rules`)
- **「這個專案的架構是...」 / 「我們決定用...」** -> (主動存入 Qdrant `document_memory`)

---

## 記憶分類指南 (Where to store)

不用過度糾結分類，請用**「影響範圍」**來判斷：

### 1. 全域偏好與行動規則 (JSON Memory)
跨專案、針對使用者的「全域設定」與「AI 行為準則」。
- **呼叫方式**：
  `execute_script("memory", "scripts/upsert_memory.py", "<key_name> '<value>' --type <agent_rules|preference|profile>")`
  *(注意：key_name 請用小寫英文及底線，如 `ui_preference`)*
- **範例**：
  - 「我習慣用 Python 且縮排用 4 空格」(`preference`)
  - 「之後都用這個格式寫」、「回答不要說廢話」(`agent_rules`)

### 2. 專案知識與工作流 (Qdrant Memory)
針對「當下專案」的事實、技術文件，或「剛剛解決的問題」。
- **簡化版呼叫方式**：
  `execute_script("memory", "scripts/upsert_memo_qdrant.py", "-c <collection> -i '<title_or_intent>' -t '<content>' --score 7")`
  - `-c document_memory`：專案事實、架構、業務邏輯。
  - `-c skill_workflow`：有效的除錯經驗、成功的任務步驟。
  - `--score`：不用想太多，**預設給 `7` 即可**（若是絕對核心的架構才給 `10`，失敗嘗試給 `4`）。
- **範例**：
  - 「我這個專案用 FastAPI」(`document_memory`)
  - 「剛剛解決了 CORS 問題，記下來」(`skill_workflow`)

---

## 最佳實踐與範例

**情境 A：使用者明確下達規定 (Keyword: 之後都)**
User: 之後寫程式都不要幫我加註解，我喜歡乾淨的 code。
Tool: `execute_script("memory", "scripts/upsert_memory.py", "no_comments 'Do not write any comments in code' --type agent_rules")`

**情境 B：剛完成一項複雜的任務 (主動觸發 - 回覆使用者前)**
# AI 測試通過了程式碼！此時絕對不能直接回覆結束對話！
Tool: `execute_script("memory", "scripts/upsert_memo_qdrant.py", "-c skill_workflow -i 'Fix FastAPI CORS' -t 'Added CORSMiddleware with allow_origins=[\"*\"] in main.py to fix connection issues.' --score 7")`
# 工具跑完後，AI 再回覆：我已經修好了，並且記錄到記憶庫中了。

**情境 C：需要回憶過去的專案知識**
Tool: `execute_script("memory", "scripts/search_memo_qdrant.py", "FastAPI CORS -c skill_workflow")`