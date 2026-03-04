# VLM Skill 專案 — 問題修正記錄

> 修正日期：2026-03-04  
> 專案路徑：`c:\Users\iaian\Desktop\vlm_skill`

---

## Bug 1｜Skills 目錄路徑硬寫死（Linux 路徑，Windows 無法執行）

### 症狀
```
Warning: Skills directory not found at C:\home\ubuntu\ocr_test\skills
Skill 'form_ocr_skill' not found. Available skills:
```

### 根因
`skill_base.py` 中 `SKILL_REPO_PATH` 寫死了 Linux 絕對路徑，在 Windows 上找不到：
```python
# ❌ 修改前
SKILL_REPO_PATH = "/home/ubuntu/ocr_test/skills"
```

### 修正
改為相對於腳本所在位置動態解析，跨平台通用：
```python
# ✅ 修改後（skill_base.py）
SKILL_REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
```

---

## Bug 2｜Thinking Process & Logs 需等推論完成才顯示

### 症狀
按下 Run 後，整個推論過程完全空白，直到最後一次性顯示所有 logs。

### 根因
原本使用 `agent.invoke()` — 等待整個流程完成才回傳；Gradio 也是以 `return` 一次性輸出。

### 修正
將 `run_ocr_task` 改為 **generator 函式**，搭配 `agent.stream()` 實現即時串流：

```python
# ✅ 修改後（gradio_app.py）
def run_ocr_task(image, ...):       # generator，不是普通 function
    yield "", "⏳ Initializing Agent..."
    ...
    for update in agent.stream(..., stream_mode="updates"):
        ...
        yield current_output, "\n\n".join(log_lines)  # 有事件就立即推送
```

> Gradio 自動偵測到函式含 `yield` 會切換為 streaming 模式，`run_btn.click()` 無需改動。

---

## Feature｜新增 Stop 按鈕

### 需求
推論執行中可以隨時中止任務。

### 實作
利用 Gradio 內建的 `cancels` 機制，零額外程式碼：

```python
# gradio_app.py — UI 部分
with gr.Row():
    run_btn  = gr.Button("🚀 Run OCR Analysis", variant="primary", scale=3)
    stop_btn = gr.Button("⏹️ Stop",             variant="stop",    scale=1)

# Event Wiring
run_event = run_btn.click(run_ocr_task, inputs=[...], outputs=[output_text, log_output])
stop_btn.click(fn=None, cancels=[run_event])   # 點擊即取消正在執行的 generator
```

> 原理：Gradio 向 server 發送取消信號，Python generator 的下一次 `yield` 時拋出 `GeneratorExit` 並終止。

---

## Bug 3｜Skill 名稱含空格被 LLM 誤當工具呼叫

### 症狀
```
Error: Image to Markdown Reconstruction is not a valid tool,
       try one of [load_skill_overview, read_skill_file].
```

### 根因
`SKILL.md` frontmatter 中 `name` 使用了含空格的人類可讀名稱：
```yaml
# ❌ 修改前
name: Image to Markdown Reconstruction
name: Form OCR Extraction
```
LLM 看到這個名稱，誤認為可直接呼叫的 function（類似工具名稱）。

### 修正
改為 **snake_case**，與資料夾名稱一致：
```yaml
# ✅ 修改後
# table_ocr_skill/SKILL.md
name: image_to_markdown

# form_ocr_skill/SKILL.md
name: form_ocr_skill
```
description 末尾加上正確的呼叫範例，進一步引導 LLM：
```yaml
description: "... Access via load_skill_overview('form_ocr_skill')."
```

---

## Bug 4｜LLM 嘗試讀取不存在的 `references/SKILL.md`

### 症狀
```
Error: File references/SKILL.md not found in skill Image to Markdown Reconstruction.
```

### 根因
`form_ocr_skill` 確實有 `references/` 子目錄，LLM 記住此格式後，對其他 skill（如 `image_to_markdown`）也假設存在相同結構，導致猜測錯誤路徑。

### 修正
強化 `load_skill_overview` 工具的回傳訊息，加上明確邊界說明：

```python
# skill_base.py
return (
    f"Loaded overview for skill: {skill_name}\n\n"
    f"{content}\n\n"
    f"=== IMPORTANT: Only these files exist in this skill ===\n"
    f"{file_list_str}\n"
    f"=== Do NOT read any file NOT listed above ===\n\n"
    f"Use read_skill_file(skill_name='{skill_name}', "
    f"file_path=<exact path from list above>) to read a file."
)
```

找不到時也列出正確名稱供 LLM 對照：
```python
available = ", ".join(f'"{s.name}"' for s in skills)
return (
    f"Skill '{skill_name}' not found.\n"
    f"Available skills (use exact name): {available}\n"
    f"Call load_skill_overview with one of the exact names above."
)
```

---

## Bug 5｜OCR 任務沒有產生任何輸出（Agent Response 空白）

### 症狀
Logs 顯示工具呼叫與工具結果都正常，但 Agent Response 欄位完全空白。

### 根因
`stream_mode="messages"` 以 **token chunk** 為單位串流，chunk 上的 `type` 屬性因自訂 `create_agent` 實作而不符合預期，導致最終 AI 回應的 chunks 全部被跳過：

```
chunk(type="ai",  tool_calls=[...])  → ✅ 記錄工具呼叫
chunk(type="tool", content="...")   → ✅ 記錄工具結果
chunk(type=???,  content="JSON...") → ❌ type 不對 → 跳過 → output 永遠空白
```

### 修正
改用 `stream_mode="updates"` — 以**完整 node 更新**為單位，對各種 LangGraph agent 更可靠：

```python
# ✅ 修改後（gradio_app.py）
for update in agent.stream(..., stream_mode="updates"):
    for node_name, node_output in update.items():
        msgs = node_output.get("messages", []) if isinstance(node_output, dict) else []
        for msg in msgs:
            if msg.type == "ai":
                if msg.tool_calls:
                    # 有後續工具呼叫 → 推理過程，只進 logs
                    log_lines.append(f"🤖 AI Thinking:\n{content}")
                else:
                    # 無工具呼叫 → 最終答案，放進 output_text
                    log_lines.append(f"🤖 Final Answer:\n{content}")
                    current_output = content   # ← 這才出現在輸出框
```

| 模式 | 顆粒度 | 可靠性 |
|---|---|---|
| `stream_mode="messages"` | token chunk（部分訊息） | 依賴 chunk 的 type 屬性，自訂 agent 可能不符 |
| `stream_mode="updates"` | 完整 node 更新 | 訊息完整，type/content/tool_calls 保證存在 ✅ |


---

## Bug 6｜Skill 名稱仍被 LLM 視為可呼叫 function

### 症狀
```
The correct approach is to use the `image_to_markdown` function...
"function": "image_to_markdown",
"function_call": { "arguments": { "image": "..." } }
```
LLM 把 skill name 當 API function 直接呼叫，而不是透過 `load_skill_overview`。

### 根因
`SkillMiddleware` 把 skill 清單以 Markdown bullet 格式注入 system prompt：
```markdown
## Available Skills
- **image_to_markdown**: [OUTPUT: MARKDOWN ONLY]...
```
`- **name**: desc` 的格式在 LLM 眼中與工具清單完全相同，導致誤用。

### 修正
重新設計 system prompt 格式，加上明確警告並區分「lookup key」與「callable tool」：
```
================================================================
KNOWLEDGE SKILL LIBRARY (READ-ONLY REFERENCE — NOT CALLABLE)
================================================================
The following are skill LOOKUP KEYS, NOT tools or functions.
You MUST NOT call them directly.

  Skill key "form_ocr_skill"
  Purpose: [OUTPUT: JSON ONLY]...

  Skill key "image_to_markdown"
  Purpose: [OUTPUT: MARKDOWN ONLY]...

CALLABLE TOOLS (the ONLY two tools you may invoke directly):
  - load_skill_overview(skill_name: str) -> str
  - read_skill_file(skill_name: str, file_path: str) -> str
================================================================
```

**修改檔案**：`skill_base.py` → `SkillMiddleware.refresh_skills_prompt()` + `wrap_model_call()`

---

## Bug 7｜`load_skill_overview` 後 LLM 重複讀取 SKILL.md

### 症狀
```
🔧 Calling Tool: read_skill_file (args: {'skill_name': 'form_ocr_skill', 'file_path': 'SKILL.md'})
```
`load_skill_overview` 已回傳完整 SKILL.md 內容，但 LLM 仍再次呼叫 `read_skill_file('...', 'SKILL.md')`。

### 根因
舊版回傳的可讀檔案清單包含 `SKILL.md`：
```
=== Only these files exist ===
- SKILL.md           ← LLM 看到了，以為還沒讀
- references/accident_report.md
```

### 修正
從可讀清單中過濾掉 SKILL.md，並在回傳訊息中明確說明：
```python
# skill_base.py — load_skill_overview
extra_files = [f for f in files if f.upper() != "SKILL.MD"]
# 回傳訊息包含：
# "=== SKILL.md is already shown above — do NOT re-read it ==="
```

如果沒有額外 reference files（如 `image_to_markdown`）：
```
No additional reference files. All instructions are contained in the overview above.
```

**修改檔案**：`skill_base.py` → `load_skill_overview` tool function

---

## Opt 8｜Skill 檔案無快取 + `_parse_frontmatter` 重複讀磁碟

### 問題
```
App 啟動  → _parse_frontmatter()     → open(SKILL.md)  # 讀磁碟 #1
LLM 呼叫  → get_skill_overview()     → open(SKILL.md)  # 讀磁碟 #2（重複）
LLM 呼叫  → read_skill_file(ref.md)  → open(ref.md)    # 讀磁碟 #3
LLM 重複  → read_skill_file(ref.md)  → open(ref.md)    # 讀磁碟 #4（重複）
```

### 修正 A：新增 `_read_file_cached()` 統一 IO 入口
```python
# skill_library.py — FileSystemSkillRepository.__init__
self._cache_file_contents: dict = {}   # {abs_path -> content_str}

def _read_file_cached(self, abs_path: str) -> Optional[str]:
    if abs_path not in self._cache_file_contents:
        with open(abs_path, 'r', encoding='utf-8') as f:
            self._cache_file_contents[abs_path] = f.read()
    return self._cache_file_contents[abs_path]
```

### 修正 B：`_parse_frontmatter` 走 cache
```python
def _parse_frontmatter(self, file_path: str):
    content = self._read_file_cached(os.path.abspath(file_path))  # ← 改這裡
    ...
```

### 修正後讀取時序
```
App 啟動  → _parse_frontmatter()     → _read_file_cached → 讀磁碟並存 cache
LLM 呼叫  → get_skill_overview()     → _read_file_cached → ✅ 直接回傳 cache
LLM 呼叫  → read_skill_file(ref.md)  → _read_file_cached → 讀磁碟並存 cache
LLM 重複  → read_skill_file(ref.md)  → _read_file_cached → ✅ 直接回傳 cache
```

**修改檔案**：`skill_library.py` → `__init__`, `_read_file_cached`, `_parse_frontmatter`, `get_skill_overview`, `get_skill_details`


---

## Bug 9｜Agent 卡住—工具呼叫進入無限迴圈

### 症狀
UI 完全卡住不動，Logs 顯示相同工具不斷重複螺現，或僅顯示 `⏳ Running Inference...` 就停止更新。

### 根因
1. **無 recursion limit**：`agent.stream()` 的 config 沒有設 `recursion_limit`，LangGraph 預設許用 25 步，但模型如果反覆呼叫工具會走滿 25 步再也不停。
2. **無重複偼測機制**：同一工具 + 同一參數被呼叫多次時，沒有任何機制防止繼續。

### 修正 A：LangGraph 原生 `recursion_limit`
```python
# gradio_app.py（module level）
MAX_AGENT_STEPS = 15

# in run_ocr_task()
config = {
    "configurable": {"thread_id": str(uuid.uuid4())},
    "recursion_limit": MAX_AGENT_STEPS,   # 超過 15 步自動拋出 GraphRecursionError
}
```

### 修正 B：自訂重複呼叫偵測（完整實作）
```python
# in run_ocr_task() — generator 函式內部

_last_tool_sig: str = ""   # 上一次工具呼叫的 signature
_dup_count: int = 0
MAX_DUP_CALLS = 3           # 連續相同呼叫達 3 次就強制結束

for update in agent.stream(..., stream_mode="updates"):
    for node_name, node_output in update.items():
        msgs = node_output.get("messages", []) if isinstance(node_output, dict) else []

        for msg in msgs:
            if getattr(msg, "type", None) == "ai":
                for tc in (getattr(msg, "tool_calls", None) or []):
                    name = tc.get("name", "")
                    args = tc.get("args", {})
                    if name:
                        # 計算 signature：工具名稱 + 參數組合
                        sig = f"{name}::{repr(args)}"
                        if sig == _last_tool_sig:
                            _dup_count += 1
                        else:
                            _last_tool_sig = sig
                            _dup_count = 1

                        # 連續重複達上限 → 強制中止
                        if _dup_count >= MAX_DUP_CALLS:
                            warn = (
                                f"\n⚠️  Detected repeated tool call "
                                f"({name} x{_dup_count}). "
                                f"Stopping to prevent infinite loop."
                            )
                            log_lines.append(warn)
                            yield current_output, "\n\n".join(log_lines)
                            return  # generator 結束

                        log_lines.append(f"🔧 Calling Tool: {name} (args: {args})")
```

### exception 區分（recursion_limit 觸發時）
```python
except Exception as e:
    err_msg = str(e)
    if "recursion" in err_msg.lower() or "GraphRecursion" in err_msg:
        # LangGraph 內建的步數上限
        log_lines.append(
            f"\n⚠️  Agent exceeded max steps ({MAX_AGENT_STEPS}). "
            f"Task stopped to prevent infinite loop."
        )
    else:
        log_lines.append(f"\n❌ Error: {err_msg}")
        current_output = f"inference failed: {err_msg}"
    yield current_output, "\n\n".join(log_lines)
    return
```

### 錯誤訊息區分
```
⚠️  Agent exceeded max steps (15). Task stopped.          ← recursion_limit 觸發
⚠️  Detected repeated tool call (load_skill_overview x3). ← 自訂重複偵測觸發
❌ Error: <message>                                        ← 其他例外
🤖 Final Answer: ...                                       ← 正常完成
```

**修改檔案**：`gradio_app.py` → `MAX_AGENT_STEPS` 常數（module level）、`config`、streaming loop 內的重複偵測邏輯、`except` block


---

## 修改檔案總覽

| 檔案 | 修改內容 |
|---|---|
| `skill_base.py` | SKILL_REPO_PATH 動態路徑；load_skill_overview 過濾 SKILL.md；system prompt 格式重設計 |
| `gradio_app.py` | run_ocr_task 改 generator + stream(updates)；新增 Stop 按鈕；recursion_limit；重複呼叫偼測 |
| `skill_library.py` | 新增 _read_file_cached；_parse_frontmatter 走 cache；get_skill_overview / get_skill_details 走 cache |
| `skills/form_ocr_skill/SKILL.md` | name 改為 snake_case `form_ocr_skill` |
| `skills/table_ocr_skill/SKILL.md` | name 改為 snake_case `image_to_markdown` |
