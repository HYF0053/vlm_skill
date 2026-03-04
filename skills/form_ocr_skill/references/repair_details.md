# Repair Details (維修明細) Reference

## Mission Objectives
Extract detailed parts and labor breakdown from repair detail sheets.

## Critical Output Rules
1.  **Verbatim Extraction**: Output **exactly** what appears in the document image/text. Do NOT correct spelling, grammar, or punctuation.
2.  **No Hallucination**: Do NOT guess or infer values. If a field is missing, output `null`.
3.  **Strict Format**: Follow the defined JSON keys.

## Field Definitions

| JSON Key | Chinese Labels (Search Keywords) | Description |
| :--- | :--- | :--- |
| `item_no` | NO, 項次, 編號 | Line item number. |
| `part_number` | 工時碼/件號, 料號, 件號, 代碼 | Part number or labor code. |
| `item_name` | 名稱, 品名, 項目說明, 零件名稱 | Name of the part or service. |
| `type` | 領/料別, 領料別, 類別 | Type category of the item. |
| `quantity` | 數量, 實際數量, 件數 | Quantity of items. |
| `unit_price` | 單價, 實際單價 | Price per unit. |
| `total_price` | 實收, 實際金額, 金額, 小計 | Total price for the line item. |

## Extraction Rules
1.  **Tabular Data**: This form is primarily a table. Extract row by row if possible, or map columns faithfully.
2.  **Verbatim**: Extract content exactly. Do not clear "0" or empty fields if they are printed.
