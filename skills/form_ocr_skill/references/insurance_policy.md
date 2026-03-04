# Insurance Policy (保險單) Reference

## Mission Objectives
Extract policy details, parties involved, and contact information from insurance policy documents.

## Critical Output Rules
1.  **Verbatim Extraction**: Output **exactly** what appears in the document image/text. Do NOT correct spelling, grammar, or punctuation.
2.  **No Hallucination**: Do NOT guess or infer values. If a field is missing, output `null`.
3.  **Strict Format**: Follow the defined JSON keys.

## Field Definitions

| JSON Key | Chinese Labels (Search Keywords) | Description |
| :--- | :--- | :--- |
| `policy_number` | 保單號碼, 保險單號 | Unique identifier for the insurance policy. |
| `applicant` | 要保人, 要保人姓名 | The person or entity applying for the insurance. |
| `insured_person` | 被保險人, 被保險人姓名 | The person covered by the insurance. |
| `insured_id` | 被保險人身份證字號, 身分證號, 統編 | ID number of the insured person. |
| `address` | 地址, 聯絡地址, 通訊地址, 戶籍地址 | Address associated with the policy or parties. |
| `beneficiary_name` | 受益人姓名, 受益人, 身故受益人 | Name of the beneficiary. |
| `beneficiary_id` | 受益人身份證字號, 受益人身分證 | ID number of the beneficiary. |
| `beneficiary_relation` | 受益人與被保險人關係, 關係 | Relationship between beneficiary and insured. |
| `contact_phone` | 連絡電話, 聯絡電話, 電話, 手機, 電話號碼, 車主電話, 行動電話 | Contact phone number. |
| `effective_date` | 有效期限, 有效年月, 有限期間, 保險期間 | The date or period the policy is effective. |
| `card_number` | 卡號, 信用卡卡號 | Credit card number if present (for payment). |
| `cardholder_name` | 持卡人姓名, 中文姓名, 持卡人 | Name of the credit card holder. |
| `cardholder_id` | 持卡人身份證字號, 持卡人身分證字號, 持卡人身分證 | ID number of the credit card holder. |
| `auto_renewal` | 自動續保 | Indication of automatic renewal status. |

## Extraction Rules
1.  **Multiple Occurrences**: If a field like "地址" appears multiple times (e.g., for applicant and insured), prefer capturing the one most prominent or specifically labeled for the `applicant` if not otherwise specified.
2.  **Verbatim**: Extract content exactly. Do not reformat dates (e.g., maintain 112/01/01 vs 2023-01-01).

## Validation & Formatting Rules
1.  **ID Number Validation**:
    - Fields: `insured_id`, `beneficiary_id`, `cardholder_id`
    - Logic: Use the standard Taiwan ID format (1 English letter + 9 digits) to validate.
    - Correction: If OCR extracts "812345678" but the context is an ID, correct the first digit '8' to 'B' or 'R' etc. based on visual likelihood if ambiguous. Start with a Letter.
