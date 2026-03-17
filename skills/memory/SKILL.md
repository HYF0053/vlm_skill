---
name: memory
description: Structured long-term memory system for remembering important user information, preferences, background, project state, and key facts. You MUST actively use this skill to ensure consistency across separate sessions.
---

Core Rules – You MUST Follow These Strictly:
──────────────────────────────────────────────
1. Whenever you see information that is likely to be useful in future conversations, **immediately** update the memory using the `memory` skill's scripts.
2. Do NOT wait for the user to say “remember this”. Every preference, tool, or project detail mentioned should be saved.
3. **Overwriting**: Understand that `profile`, `preference`, and `project` types will **replace** the existing value for the same key.

### 🧠 2026 記憶儲存準則 (JSON vs. Qdrant)

| 類型 | 存儲位置 | 內容範例 | 判斷基準 |
| :--- | :--- | :--- | :--- |
| **短期/精簡記憶** | **JSON (Memory Skill)** | 偏好、OS、暱稱、程式風格 | 長度 < 300 tokens、精準 Key-Value 檢索 |
| **長期/大型記憶** | **Qdrant (RAG Skill)** | 專案規劃、技術文件、過去對話摘要 | 長度 > 300 tokens、需語義搜尋、具時間/事件屬性 |

#### JSON 記憶分類 (mem_type):
- `preference`: 使用者明確要求的行為準則 (e.g., "不要用 emoji")。
- `profile`: 用戶靜態事實 (e.g., "我是後端工程師")。
- `project`: 目前 active 專案的小型設定 (e.g., "資料庫用 Postgres")。
- `fact`: 已廢棄，請改用 **`rag` 技能** 存入 Qdrant。

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

Mandatory Trigger Conditions – You MUST update memory immediately when:
───────────────────────────────────────────────────────────────────────────────
- User mentions their OS, editor, IDE, browser, timezone, country/city, job title/role.
- User expresses any preference: “I like / I hate / please / better if / avoid / don’t use…”
- User gives project-related info: folder path, database, API key name, chosen framework/library, decision made.
- User expresses a correction: "Actually, I use Windows, not Mac."

### 使用腳本更新記憶 (Usage)

請透過 `execute_script` 調用以下腳本進行寫入：

**1. `upsert_memory.py` (JSON 偏好)**
適合 < 300 tokens 的「性格、設定、偏好」。
```bash
# 語法：execute_script("memory", "scripts/upsert_memory.py", "<key> '<value>' --type <type>")
execute_script("memory", "scripts/upsert_memory.py", "preferred_language Python --type profile")
```

**2. `rag` 技能下的 `upsert_to_vdb.py` (Qdrant 長期內容)**
**強烈建議：** 超過一個段落或具知識性的資料，應存入 RAG 的 `agent_long_memory`。

---

## 範例：

```python
# User: 我偏好用 Python，且我不喜歡用 emoji。
execute_script("memory", "scripts/upsert_memory.py", "preferred_language Python --type profile")
execute_script("memory", "scripts/upsert_memory.py", "emoji_usage none --type preference")
```