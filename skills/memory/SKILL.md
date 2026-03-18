---
name: memory
description: Mandatory long-term memory. Update IMMEDIATELY when user mentions preferences, OS/IDE, job, or project decisions. Use execute_script("memory", "scripts/upsert_memory.py", ...)
---

Core Rules:
1. **Immediate Update**: Don't wait for "remember this". If it's a fact/preference, save it now.
2. **Key-Value**: Use lowercase snake_case keys (e.g., `preferred_editor`).
3. **Overwrite**: Types `profile`, `preference`, `project` will replace existing keys.

## Currently Available Memory Tools

- **`scripts/upsert_memory.py`**: Principal tool for short JSON memory (< 300 tokens).
  - *Usage*: `execute_script("memory", "scripts/upsert_memory.py", "<key> '<value>' --type <type>")`
  - *Types*: `preference` (behavior), `profile` (facts about user), `project` (active project settings).
- **Trigger Conditions**: Update memory anytime the user mentions:
  - OS, editor, browser, role/job, timezone, country.
  - Likes/dislikes ("I prefer...", "Avoid...", "Don't use...").
  - Project decisions (folder paths, database choice, framework).

## 🧠 2026 記憶儲存準則 (JSON vs. Qdrant)

| 類型 | 存儲位置 | 內容範例 | 判斷基準 |
| :--- | :--- | :--- | :--- |
| **短期/精簡記憶** | **JSON (Memory Skill)** | 偏好、OS、暱稱、程式風格 | 長度 < 300 tokens、精準 Key-Value 檢索 |
| **長期/大型記憶** | **Qdrant (RAG Skill)** | 專案規劃、技術文件、過去對話摘要 | 長度 > 300 tokens、需語義搜尋、具時間/事件屬性 |

#### JSON 記憶分類 (mem_type):
- `preference`: 使用者明確要求的行為準則 (e.g., "不要用 emoji")。
- `profile`: 用戶靜態事實 (e.g., "我是後端工程師")。
- `project`: 目前 active 專案的小型設定 (e.g., "資料庫用 Postgres")。

> [!IMPORTANT]
> 若資訊超過一個段落或屬於知識片段，請務必主動調用 `rag` 技能下的 `upsert_to_vdb.py` 存入 **`agent_long_memory`** 集合。

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