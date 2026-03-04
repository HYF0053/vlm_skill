# Accident Scene Diagram (事故現場圖) Reference

## Mission Objectives
Extract metadata and summaries from accident scene diagrams.

## Critical Output Rules
1.  **Verbatim Extraction**: Output **exactly** what appears in the document image/text. Do NOT correct spelling, grammar, or punctuation.
2.  **No Hallucination**: Do NOT guess or infer values. If a field is missing, output `null`.
3.  **Strict Format**: Follow the defined JSON keys.

## Field Definitions

| JSON Key | Chinese Labels (Search Keywords) | Description |
| :--- | :--- | :--- |
| `police_unit` | 警察局名稱, 處理單位 | Name of the police department handling the case. |
| `jurisdiction_branch` | 轄區分局名稱 | Name of the jurisdiction branch. |
| `case_number` | 處理編號 | Case number. |
| `accident_category` | 交通事故類別, 事故類別, 類別 | Category of the accident. |
| `occurrence_time` | 發生時間, 案發時間, 時間 | Date and time of the accident. |
| `location` | 地點, 發生地點, 案發地點, 地址 | Location where the accident occurred. |
| `weather` | 天候, 天氣 | Weather conditions. |
| `signal_status` | 號誌時相, 號誌 | Traffic signal status. |
| `speed_limit` | 第一當事人速限, 速限 | Speed limit relevant to the first party. |
| `involved_parties` | A車, B車, 當事人 | Identifier for vehicles/parties involved. |
| `summary` | 現場處理摘要, 肇事經過, 肇事經過摘要, 摘要 | Textual description of the accident scene or events. |
| `diagram_drawer` | 製圖人, 製圖, 繪圖員 | Name the person who drew the diagram. |
| `supervisor` | 主管, 所長, 隊長 | Name of the supervisor. |
| `drawing_date` | 製圖日期, 繪圖日期 | Date the diagram was created. |
| `remark` | 備註 | Notes or comments about the accident scene. |

## Extraction Rules
1.  **Checkboxes**: For fields like `weather` or `accident_category` which might be checkboxes, extract the text of the *selected* option if discernable, or the text label associated with the check.
2.  **Verbatim**: Extract descriptions exactly as written.

## Validation & Formatting Rules
1.  **Occurrence Time**: Ensure the time format follows `YYYY 年 MM 月 DD 日 HH 時 mm 分`. If OCR misses "年/月/日", infer based on position if obvious, but prefer verbatim if ambiguous.
2.  **Case Number**: Usually alphanumeric. Check for common OCR errors (e.g., 'O' vs '0').
