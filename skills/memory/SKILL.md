---
name: memory
description: Save long-term memory and Manage Personal and Project Knowledge. Includes 'user preferences', 'project specs', and high-value structured info like 'tables', 'query results', 'conclusions', and 'plans' with importance scores. Use this to maintain long-term context and decision history.
---

## 🚨 致命問題與嚴格行動準則 (CRITICAL Execution Timing)

根據過去的行為紀錄，**你經常只在口頭上說「好的，我會記住」，卻沒有實際呼叫工具，導致記憶完全遺失！**
為了徹底解決這個問題，請嚴格遵守以下「強制執行時機」：

1. **收到 JSON 規則 / 偏好時：**
   - 當使用者說：「之後都...」、「這是我習慣的格式...」、「你要遵守...」。
   - ⚠️ **優先執行工具呼叫！**請於看到提示的**當下這個回合 (turn)**，立刻主動呼叫 `upsert_memory.py` 完成儲存。
   
2. **獲得 Qdrant 專案重要資訊時：**
   - 當你知道了這個專案的技術棧、DB Schema、或特別的設定時。
   - ⚠️ **自主判斷並主動存檔！**凡是有利於後續開發的資訊，請**主動**使用 `upsert_memo_qdrant.py -c document_memory` 存下它。

3. **解決困難 Bug / 完成複雜重構時 (skill_workflow) ★最常忘記被記錄：**
   - 當你花費多步完成了某段程式碼的修正，並且**測試/驗證通過**的那一刻。
   - ⚠️ **務必在完成記錄後才宣告結束！**
   - 你的倒數第二步動作，**應為**先呼叫 `upsert_memo_qdrant.py -c skill_workflow` 將除錯經驗存入記憶庫。請確認儲存成功後，再發出結尾回覆「我已完成修正」。

4. **生成重要結論、規劃或查詢結果時 (Conclusion/Plan/Query Result):**
   - 當你分析完數據、產出表格、制定了下一步開發計劃、或得出關鍵結論。
   - ⚠️ **這類「非對話」的高價值資訊應進行結構化儲存。** 務必確保資訊已透過工具存檔。
   - 請使用 `upsert_memo_qdrant.py -c document_memory` 並給予**重要性分數 (`--score`)**。通常關鍵計畫為 8-10，一般查詢結果為 5-7。
---

## 什麼時候該觸發記憶？ (Memory Trigger Patterns)

看到以下【明確句型】或【情境】時，請立刻採取行動（呼叫對應的腳本）：

- **「我們上次討論過...」 / 「回憶一下之前的對話」** -> (存入/查詢 Qdrant `document_memory` or `past_discussions`)
- **「我以後偏好...」 / 「我的習慣是...」** -> (存入 JSON `preference` 或 `agent_rules`)
- **「這個專案的規格 (Project Specs) 是...」 / 「架構決定用...」** -> (存入 Qdrant `document_memory`)
- **產出詳細表格、數據分析、複雜查詢結果或系統規劃** -> (結構化存入 `document_memory` 並加權評分)
- **得出關鍵技術結論或設計決策 (Design Decisions)** -> (存入 `document_memory` 並加權評分)

---

## 記憶分類指南 (Where to store)

### 1. 使用者個人偏好 (Preferences - JSON)
針對使用者的習慣、AI 行為風格。
- **範例**：
  - 「我習慣用 Python 且縮排用 4 空格」(`preference`)
  - 「請用簡潔的方式回答」(`agent_rules`)

### 2. 過去討論紀錄 (Past Discussions - Qdrant)
存儲對話中的重要結論、做過的決策，以便未來喚回。
- **Collection**：`document_memory` (或特定 `past_discussions`)

### 3. 專案技術規格 (Project Specifications - Qdrant)
當前開發專案的事實、技術架構、業務細節。
- **Collection**：`document_memory`
- **範例**：
  - 「這個專案的 DB Schema 長這樣...」
  - 「剛剛解決了 CORS 問題，記下來作為專案經驗」(`skill_workflow`)

---

## 🚫 行動準則：避免重複與混亂
在執行 `upsert_memory.py` 之前，你**必須**：
1. **查看當前已有的 Structured Memory**：如果已經有性質相近的 Key (例如已經有 `brand_preference`)，請**直接覆蓋**它，絕對不要自創 `memory_brand` 之類的新 Key。
2. **語義屬性優先**：不要使用過於空泛的 `preference` 作為 Key，應使用上述標準化表格中的具體 Key 名稱。

---

## 最佳實踐與範例

**情境 A：使用者明確下達規定 (Keyword: 之後都)**
User: 之後寫程式請保持 code 乾淨，不需要加註解。
Tool: `execute_script("memory", "scripts/upsert_memory.py", "no_comments 'Keep code clean without comments' --type agent_rules")`

**情境 B：剛完成一項複雜的任務 (主動觸發 - 回覆使用者前)**
# AI 測試通過了程式碼！此時應先完成記錄儲存，再回覆使用者結束對話。
Tool: `execute_script("memory", "scripts/upsert_memo_qdrant.py", "-c skill_workflow -i 'Fix FastAPI CORS' -t 'Added CORSMiddleware with allow_origins=[\"*\"] in main.py to fix connection issues.' --score 7")`
# 工具跑完後，AI 再回覆：我已經修好了，並且記錄到記憶庫中了。

**情境 C：需要回憶過去的專案知識**
Tool: `execute_script("memory", "scripts/search_memo_qdrant.py", "FastAPI CORS -c skill_workflow")`

**情境 D：產出關鍵規劃或分析結論 (Keyword: 評分與類型)**
User: 請分析一下當前系統的瓶頸，並給出優化計畫。
AI: (分析並產出詳細計畫)
Tool: `execute_script("memory", "scripts/upsert_memo_qdrant.py", "-c document_memory -i 'System Optimization Plan' -t '1. Add Redis caching 2. DB Indexing on UserID...' --score 9")`
# 工具跑完後，AI 再回覆：我已經完成了系統瓶頸分析與優化計畫，並將此高價值結論存入記憶庫 (Score: 9)。

---

## 🛠️ 腳本參數詳解 (Python Script API)
所有腳本皆位於 `skills/memory/scripts/` 目錄下。

### 1. `upsert_memory.py` (JSON 結構化記憶)
用於儲存使用者偏好、對話風格、個人檔案等輕量級資訊。
*   **參數格式**：`python upsert_memory.py <Key> "<Value>" [--type <Type>]`
*   **參數定義**：
    *   `Key` (必填)：記憶鍵值 (例如：`brand_preference`, `tone`)。建議遵循上述標準化規範。
    *   `Value` (必填)：具體的記憶內容。若包含空格請務必用引號包裹。
    *   `--type` (選填)：記憶類型，可選值為 `preference` (預設), `profile`, `agent_rules`。
*   **範例**：`execute_script("memory", "scripts/upsert_memory.py", "brand_preference '喜歡 NVIDIA 顯示卡' --type preference")`

### 2. `upsert_memo_qdrant.py` (Qdrant 長期記憶存入)
用於儲存大量文本、技術文件、Bug 經驗或重要專案決策。
*   **參數格式**：`python upsert_memo_qdrant.py --collection <Col> --intent_or_type <Type> --content "<Text>" [--score <Score>] [--source <Src>]`
*   **參數定義**：
    *   `--collection` / `-c` (必填)：目標集合，目前僅支援 `document_memory`。
    *   `--intent_or_type` / `-i` (必填)：分類標籤 (例如：`BugFix`, `Spec`, `Requirement`)。
    *   `--content` / `-t` (必填)：要存儲的詳細文本內容。
    *   `--score` (選填)：重要性評分 (1.0 - 10.0)，預設為 `5.0`。
    *   `--source` / `-s` (選填)：來源標籤，預設為 `user_session`。
*   **範例**：`execute_script("memory", "scripts/upsert_memo_qdrant.py", "-c document_memory -i 'RAG vs LongContext' -t '量化分析結論如下...' --score 9.5")`

### 3. `search_memo_qdrant.py` (Qdrant 語義搜尋)
用於從長期記憶庫中找回相關知識。
*   **參數格式**：`python search_memo_qdrant.py "<Query>" [--limit <N>] [--collection <Col>]`
*   **參數定義**：
    *   `query` (必填)：搜尋關鍵字或自然語言描述。
    *   `--limit` (選填)：回傳結果數量，預設為 `5`。
    *   `--collection` / `-c` (選填)：指定搜尋集合。未指定則搜尋所有配置中的集合。
*   **範例**：`execute_script("memory", "scripts/search_memo_qdrant.py", "FastAPI CORS 解決方案 --limit 3")`

### 4. `prune_memo_qdrant.py` (長期記憶清理)
用於刪除老舊且低重要性的記憶點。
*   **參數格式**：`python prune_memo_qdrant.py [--days <D>] [--max-score <S>] [--collection <Col>]`
*   **參數定義**：
    *   `--days` (選填)：刪除幾天前之前的資料，預設為 `30`。
    *   `--max-score` (選填)：僅刪除分數低於此數值的資料，預設為 `5.9`。
    *   `--collection` / `-c` (選填)：指定清理的集合。
*   **範例**：`execute_script("memory", "scripts/prune_memo_qdrant.py", "--days 60 --max-score 4.0")`
