---
name: rag
description: Access Product and Technical Knowledge. Use this to lookup 'product equipment information' (GPUs, servers) or 'technical industry regulations/specifications'.
---

## RAG 角色定位 (Role & Responsibility)

本工具專注於 **「外部技術知識」** 與 **「產品/產業規範」**。

1. **產品設備資訊 (Product Equipment Info)**
   - 包含：GPU 規格、工作站配置、零件相容性、Roadmap。
   - 來源：Leadtek 內部產品文件。

2. **產業/技術規範 (Technical Specifications/Regulations)**
   - 包含：技術標準、合規文件、硬體連線規範。

---

## 常用 Collection 與查詢情境

### 1. 產品規格查詢
- `hw_gpu_kb`: Leadtek 單張顯卡、規格、Roadmap。
- `hw_system_kb`: Leadtek 工作站與伺服器、相容性。

### 2. FAQ 與通用技術文件
- `test_docs`: AIDMS 產品介紹、常見問題。

---

## 最佳實踐

**情境 A：詢問特定硬體型號規格**
User: 「Leadtek RTX A6000 的耗電量是多少？」
Action: 呼叫 `search_vdb.py` 查詢 `hw_gpu_kb`。

**情境 B：尋找產業技術規範**
User: 「請確認這款顯卡是否符合伺服器插槽規範？」
Action: 使用 `retrieve_context.py` 從 RAG 提取規格。

**用法示範：**
```bash
python skills/rag/scripts/upsert_to_vdb.py "專案 A 的權限設計邏輯為..." -c agent_long_memory -s conversation --metadata '{"type": "knowledge", "importance": 4}'
```

### 1. `search_vdb.py`

This script searches the vector database using hybrid search (Keyword + Vector) and reranking. It provides a detailed, human-readable list of search results including scores and metadata. Use this when you want to explore the data structure or visually inspect search outcomes.

**Usage:**
```bash
python skills/rag/scripts/search_vdb.py "your search query" [--limit LIMIT] [-c COLLECTION_NAME]
```

**Options:**
- `query` (required): The search string (e.g., "PRO 6000 specifications").
- `--limit` (optional): Maximum number of results to return (default is 5).
- `-c` or `--collection` (optional): Restrict the search to a specific collection. Can specify multiple. If omitted, it searches all available collections natively.

**Available Collections:**
- `test_docs`: AIDMS product intro, FAQs.
- `hw_gpu_kb`: Leadtek standalone GPUs, specs, roadmaps, benchmarks.
- `hw_system_kb`: Leadtek Workstations and Servers, specs, compatibility.

**Example:**
```bash
# Search across all collections for GPU specifications
python skills/rag/scripts/search_vdb.py "Leadtek RTX A6000 specs" --limit 3

# Search strictly within the system KB collection
python skills/rag/scripts/search_vdb.py "WS650 supported GPU" -c hw_system_kb
```

### 2. `retrieve_context.py`

This script behaves similarly to `search_vdb.py`, but its output is formatted optimally for immediate use as context by Large Language Models. It concatenates the retrieved documents into a single block of text and intelligently truncates them to fit within a specified token limit. Use this when you are dynamically feeding retrieved documents into an LLM block.

**Usage:**
```bash
python skills/rag/scripts/retrieve_context.py "query" [--max_tokens MAX] [-c COLLECTION_NAME]
```

### 3. `upsert_to_vdb.py` (動態寫入 / 技術知識庫)

這是維護 **技術與產品知識庫** 的工具。請將產品手冊、技術規格、產業規範存入 Qdrant。

**推薦 Collection 名稱：** `agent_long_memory`

**Qdrant 元數據架構 (Standard Payload):**
| 欄位 | 說明 |
| :--- | :--- |
| `type` | `episodic` (事件), `plan` (規劃), `knowledge` (知識), `reflection` (反思) |
| `importance` | 重要程度 (1-5) |
| `session_id` | 對話 ID |
| `timestamp` | 格式: `ISO-8601` |

**用法示範：**
```bash
python skills/rag/scripts/upsert_to_vdb.py "專案 A 的權限設計邏輯為..." -c agent_long_memory -s conversation --metadata '{"type": "knowledge", "importance": 4}'
```

If you need deeper integration or customization for RAG, please refer to the project's core modules.
