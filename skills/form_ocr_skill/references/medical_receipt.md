# Medical Receipt (醫療單據) Reference

## Mission Objectives
Extract patient information, provider details, dates, and financial breakdown from medical receipts.

## Field Definitions

### Universal Fields
| JSON Key | Chinese Labels (Search Keywords) | Description |
| :--- | :--- | :--- |
| `patient_name` | 姓名, 病患名字, 病患姓名, 患者名稱, 患者姓名, 病人姓名 | The name of the person receiving treatment. |
| `patient_id` | 身份證, 身分證號, 身分證, 身份證字號, 統編 | The national identification number of the patient. |
| `birth_date` | 出生日期, 生日, 出生年月日 | Date of birth of the patient. |
| `visit_date` | 就診時間, 就診日, 治療時間, 就醫日期, 處方日期, 治療期間 | **Medical Service Period**. Extract the full date range (e.g., "114年01月04日至114年03月03日") or all distinct dates listed. NOT limited to a single day. |
| `provider_name` | 醫院名字, 診所, 醫療機構 | Name of the hospital or clinic. |
| `doctor_name` | 醫生名字, 醫師名字, 治療醫生, 治療醫師 | Name of the doctor who provided the service. |
| `department` | 科別, 就診科別 | The medical department (e.g., Internal Medicine). |

### Financial Fields
| JSON Key | Chinese Labels (Search Keywords) | Description |
| :--- | :--- | :--- |
| `total_amount` | 金額, 金錢, 應繳款項, 總金額, 合計, 總計, 自付費用, 部分負擔, 費用項目, 項目 | A single string listing ALL financial values (Fees, Self-Pay, Copayment, Total, item fee). Format: "Category:Value, Category:Value...". |

## Extraction Rules
1.  **Date Formats**: Extract dates exactly as they appear. Support date ranges (e.g., "112/01/01至112/01/05" or "112.01.01-112.01.05"). Do not truncate to a single date if a valid range is shown.
2.  **Amount Listing**: Extract ALL amount categories exactly as labeled (e.g., if there is a "Registration Fee" and a "Drug Fee", extract both into their respective keys). Do **NOT** sum them up. Do **NOT** perform any calculation.
3.  **Ambiguity**: If "姓名" appears near hospital logo, it might be the doctor's name, but usually it labels the Patient Name. Verify context based on position.
4.  **Cost Items**: Do NOT output the general list of cost items as text. INSTEAD, use the specific keys above (`registration_fee`, `drug_fee`, etc.) to capture the values from the cost table.

## Validation & Formatting Rules
1.  **ID Number Validation**:
    - Field: `patient_id`
    - Logic: Use the standard Taiwan ID format (1 English letter + 9 digits) to validate and correct OCR errors.
    - Correction: If the first character is ambiguous (e.g., '8' vs 'B'), prefer the English letter. If the numeric part has ambiguous chars (e.g., 'O' vs '0'), prefer digits.