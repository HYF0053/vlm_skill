# Repair Estimate (維修估價單) Reference

## Mission Objectives
Extract repair items, costs, and vehicle information from repair estimates.

## Critical Output Rules
1.  **Verbatim Extraction**: Output **exactly** what appears in the document image/text. Do NOT correct spelling, grammar, or punctuation.
2.  **No Hallucination**: Do NOT guess or infer values. If a field is missing, output `null`.
3.  **Strict Format**: Follow the defined JSON keys.

## Field Definitions

| JSON Key | Chinese Labels (Search Keywords) | Description |
| :--- | :--- | :--- |
| `estimate_date` | 估價日期, 日期, 報價日期 | Date the estimate was created. |
| `plate_number` | 車號, 車牌, 車牌號碼 | License plate number of the vehicle. |
| `repair_items` | 修理項目, 交修項目, 維修名稱, 品名, 零件, 項目, 名稱 | List of items to be repaired or replaced. |
| `labor_fee` | 工資, 鈑金工資, 塗裝工資 | Cost of labor for specific items. |
| `amount` | 金額, 費用, 價格, 小計, 總計 | Cost amount for specific items or total. |

## Extraction Rules
1.  **Lists**: For fields like `repair_items`, `labor_fee`, and `amount`, these frequently appear in tables. Extract them conserving their association if possible, or as parallel lists.
2.  **Verbatim**: Extract content exactly. Do not calculate totals.
