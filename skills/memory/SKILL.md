---
name: memory
description: Mandatory! If user mentions RULES/HABITS/PERSONA, use `execute_script("memory", "scripts/upsert_memory.py", ...)`. If user mentions PROJECT FACTS/DOCUMENTS/PLANS, you MUST use `execute_script("rag", "scripts/upsert_to_vdb.py", ...)`. Do NOT mix them up!
---

Core Rules:
1. **Immediate Update**: Don't wait for "remember this".
2. **Key-Value (JSON)**: Use lowercase snake_case keys (e.g., `preferred_editor`).
3. **Overwrite (JSON)**: Types `profile`, `preference` will replace existing keys.

## Currently Available Memory Tools

- **`scripts/upsert_memory.py`**: ONLY FOR USER PERSONA (habits, rules, OS, likes/dislikes).
  - *Usage*: `execute_script("memory", "scripts/upsert_memory.py", "<key> '<value>' --type <type>")`
  - *Types*: `preference` (user tastes), `profile` (facts about user), `agent_rules` (explicit directives for the agent).
- **Trigger Conditions for JSON**:
  - When learning about the user personally (OS, timezone), their tastes ("I prefer Python"), or exactly how the agent must act ("Always reply in...", "Don't use emoji").

- **Trigger Conditions for QDRANT** (You MUST use Qdrant for these. Do NOT use JSON):
  - **Project/Business Plans**: The user reveals a roadmap, marketing strategy, business plan, or project sequence (e.g., "我們下個月要推廣新產品", "接下來要實作付款功能").
  - **Problem/Solution Archives**: You just solved a complex issue (tech bugs, strategy bottlenecks, data formatting) and the user wants to remember the solution ("記住這個解法").
  - **Domain Facts & Data**: Information about the user's business ecosystem, client profiles, code architecture, or market trends (e.g., "主要客戶是 B2B 科技廠", "後端架構是 FastAPI", "這份 PDF 是 2026 Q3 趨勢").
  - **Meeting / Document Notes**: Long-form context, meeting minutes, summaries of PDFs, or extensive research data.
  - *Usage*: `execute_script("rag", "scripts/upsert_to_vdb.py", "<content> --collection agent_long_memory")`

## 🧠 2026 記憶儲存準則 (Strict JSON vs. Qdrant Separation)

| 存儲位置 | 定位 | 內容範例 |
| :--- | :--- | :--- |
| **JSON (Memory Skill)** | **User Persona & Rules** | 偏好、作業系統、AI 行動準則、格式要求、禁忌 | 
| **Qdrant (RAG Skill)** | **External Facts** | 專案規劃、客戶資料、會議紀錄、產業趨勢、解決方案、技術文件 | 

#### JSON 記憶分類 (mem_type):
- `agent_rules`: Agent 行動最高指導原則 (e.g., "不要道歉", "回答要簡潔", "輸出前先思考")。
- `preference`: 使用者的習慣與風格偏好 (e.g., "喜歡用 Python", "程式碼用 4 空格")。
- `profile`: 用戶靜態事實 (e.g., "我是後端工程師", "時區是台北")。

> [!IMPORTANT]
> **絕對不要** 把專案文件、規劃、時間點、或長篇大論存入 JSON。任何非「用戶習慣」的資訊，必須存入 RAG (Qdrant)。

⚠️ Strategic Guidance for High Quality Memory:
──────────────────────────────────────────────
- **Key Selection**: Use short, lowercase, snake_case English keys (e.g., `primary_os`, `naming_convention`).
- **Value Quality**: Do NOT store vague values. 
    - ❌ `execute_script("memory", "scripts/upsert_memory.py", "os Yes --type profile")`
    - ✅ `execute_script("memory", "scripts/upsert_memory.py", "os 'Ubuntu 22.04 LTS' --type profile")`
- **Avoid Redundancy**: If the information is already in your "STRUCTURED MEMORY" block and hasn't changed, don't call it again.
- **Merge when needed**: If updating a `preference` for `code_style` and you already have "use tabs", update it to "use tabs, 4 spaces" if the user adds a new rule.

---

## 範例：

```python
# User: 我偏好用 Python，且我不喜歡用 emoji。
execute_script("memory", "scripts/upsert_memory.py", "preferred_language Python --type profile")
execute_script("memory", "scripts/upsert_memory.py", "emoji_usage none --type preference")
```