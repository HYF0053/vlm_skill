---
name: map_reduce
description: "Use this skill when the user wants to process a long document, long text, or large file that would exceed the LLM context window (token limit). This includes summarising long PDFs, extracting structured data from long texts, analysing large log files, processing lengthy OCR output, translating long documents, or any task where the input is too large to fit in one prompt. The skill chunks the input, processes each chunk independently (Map), then combines the results (Reduce) into a single coherent output."
---

# Map-Reduce 長文處理技能

## 何時使用此技能

當遇到以下情況時，**必須**使用此技能：
- 輸入文字超過 **3,000 字**（約 1,500 tokens）
- PDF 轉換後的文字太長，直接問模型會報錯
- 需要從長文件中提取結構化資料
- 需要摘要或翻譯長篇文章

---

## 核心概念

```
長文本（10,000 字）
      ↓ split_chunks()
[chunk1] [chunk2] [chunk3] [chunk4]   ← Map 階段：每塊獨立處理
    ↓        ↓        ↓        ↓
 [結果1] [結果2] [結果3] [結果4]
      ↓ reduce()
   [最終彙整輸出]                      ← Reduce 階段：合併成完整結果
```

---

## 使用方式（固定工作流程）

### Step 1：呼叫腳本執行 Map-Reduce

```
execute_script(
    skill_name="map_reduce",
    script_path="scripts/run_map_reduce.py",
    script_args="<輸入檔路徑> <任務描述> [輸出檔路徑] [chunk_size] [overlap]"
)
```

**參數說明：**
| 參數 | 必填 | 說明 | 預設值 |
|------|------|------|--------|
| 輸入檔路徑 | ✅ | .txt / .md 文字檔的完整路徑 | - |
| 任務描述 | ✅ | 告訴每個 chunk 要做什麼（用引號包起來） | - |
| 輸出檔路徑 | ❌ | 結果儲存位置（不填則印到 stdout） | stdout |
| chunk_size | ❌ | 每塊字元數 | 1500 |
| overlap | ❌ | chunk 之間重疊字元數（防止邊界遺失） | 200 |

### Step 2：查看結果

腳本會輸出：
- 每個 chunk 的中間結果（Map 輸出）
- 最終 Reduce 合併結果
- 如果指定了輸出路徑，結果也會寫入檔案

---

## 使用範例

### 範例 1：摘要長 PDF 轉出的文字

```
# 先把 PDF 轉成文字（用 pdf skill）
run_cli_command("pdftotext /tmp/report.pdf /tmp/report.txt")

# 再用 map_reduce 摘要
execute_script(
    skill_name="map_reduce",
    script_path="scripts/run_map_reduce.py",
    script_args='/tmp/report.txt "請摘要這段文字的重點，保留關鍵數字和結論" /tmp/summary.txt'
)
```

### 範例 2：從長文件提取結構化資料

```
execute_script(
    skill_name="map_reduce",
    script_path="scripts/run_map_reduce.py",
    script_args='/tmp/ocr_output.txt "請從這段文字中提取所有金額、日期、和人名，以 JSON 格式輸出" /tmp/extracted.json'
)
```

### 範例 3：翻譯長文章

```
execute_script(
    skill_name="map_reduce",
    script_path="scripts/run_map_reduce.py",
    script_args='/tmp/article.txt "請將這段英文翻譯成繁體中文" /tmp/translated.txt 1000 150'
)
```

### 範例 4：分析長日誌文件

```
execute_script(
    skill_name="map_reduce",
    script_path="scripts/run_map_reduce.py",
    script_args='/tmp/app.log "分析這段日誌，列出所有 ERROR 和 WARNING，說明可能的原因" /tmp/analysis.txt'
)
```

---

## 直接從字串處理（不需要檔案）

如果文字已在對話中（不需要讀檔），也可以用 `run_python_code` 直接調用 Python API：

```python
# 示例：直接處理一段長文字
import sys
sys.path.insert(0, "/home/ubuntu/vlm_skill/skills/map_reduce/scripts")
from map_reduce_engine import MapReduceEngine, LLMConfig

config = LLMConfig(
    base_url="http://10.1.1.7:9000/v1",
    api_key="EMPTY",
    model="Qwen/Qwen3-VL-32B-Instruct",
)
engine = MapReduceEngine(config)

long_text = """（在這裡放入長文字）"""
task = "請摘要這段文字的重點"

result = engine.run(long_text, task)
print(result.final_output)
```

---

## 輸出格式

腳本輸出範例：
```
[MAP-REDUCE] Input: 8432 chars → 6 chunks
[MAP] Chunk 1/6 (1500 chars) → processing...
[MAP] Chunk 2/6 (1500 chars) → processing...
...
[REDUCE] Combining 6 results...
=== FINAL OUTPUT ===
（最終合併結果）
====================
[MAP-REDUCE] Done. Saved to /tmp/output.txt
```

---

## 進階設定

### 調整 chunk_size

| 場景 | 建議 chunk_size | 說明 |
|------|-----------------|------|
| 一般文字摘要 | 1500（預設） | 平衡品質與速度 |
| 表格/結構化資料提取 | 800~1000 | 避免表格被切斷 |
| 翻譯任務 | 1000 | 保持語義完整性 |
| 大量日誌分析 | 2000 | 加大以減少 API 呼叫次數 |

### 調整 overlap

overlap 是相鄰 chunk 重疊的字元數，防止重要資訊落在切割邊界：
- 預設 200 字元：適合大多數場景
- 調高至 300~400：文章有跨段落的邏輯關聯時

---

## 注意事項

1. **腳本需要能連接 LLM API**：腳本會從環境變數或預設值讀取 `VLLM_BASE_URL`
2. **大文件耗時**：10,000 字可能需要 1~3 分鐘（取決於模型速度）
3. **Reduce 也可能很長**：如果 Map 結果仍然很多，Reduce 會再做一層壓縮
4. **圖片無法處理**：此 skill 只處理文字；圖片請先 OCR 再使用此技能
