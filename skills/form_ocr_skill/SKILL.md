---
name: Form OCR Extraction
description: "[OUTPUT: JSON ONLY] Extractor for structured form data. Use ONLY when JSON format is required. Supports: Accident Reports, Medical Receipts, and other forms. Do NOT use for Markdown table layout."
---

# Form OCR Extraction Skill

## Description
This skill extracts information from form documents (images or OCR text) into a unified JSON format. It uses modular reference files to define extraction rules for different form types (e.g., Accident Reports).

## Usage
When given a document (image or text) and a request to extract form data:
1.  **Check Output Format**: If the user asks for Markdown (MD), Table, or Layout reconstruction, **STOP** and use `Image to Markdown Reconstruction` instead. This skill produces **JSON ONLY**.
2.  **Identify Form Type**: Determine the type of form (e.g., "Accident Report", "Receipt", etc.).
3.  **Load Reference**: Find the corresponding extraction rules in the `references/` directory.
    - Example: For Accident Reports (事故聯單), use `references/accident_report.md`.
3.  **Extract Data**: specific fields defined in the reference file.
4.  **Output JSON**: Format the output as a valid JSON object.


## Critical Output Rules
1.  **Verbatim Extraction**: Output **exactly** what appears in the document image/text. Do NOT correct spelling, grammar, or punctuation.
2.  **No Hallucination**: Do NOT guess or infer values that are not explicitly present. If a field is missing, output `null` or an empty string as appropriate, do not invent data.
3.  **No Conversational Text**: Output **ONLY** the JSON object. Do not include markdown code blocks (```json ... ```), explanations, or any other text before or after the JSON.
4.  **Strict Format**: Follow the `references/` definitions precisely.

## Unified Output Format
All outputs must follow this JSON structure:

```json
{
  "form_type": "<identified_form_type_id>",
  "data": {
    "<key_from_reference>": "<extracted_value>",
    ...
  }
}
```

- **form_type**: The identifier of the form (e.g., `accident_report`).
- **data**: An object containing the extracted fields. The keys must match the "JSON Key" defined in the reference file.

## Extending with New Forms
To add support for a new form type:
1.  Create a new markdown file in the `references/` directory (e.g., `references/invoice.md`).
2.  Define the **Mission Objectives** and list the fields to extract.
3.  Assign a unique **JSON Key** for each field.
4.  Provide specific **Identification and positioning rules** for accurate extraction.
5.  (Optional) Provide examples or validation logic.

## Current Supported Forms
- **Accident Report (事故聯單)**: defined in [Accident Report Specs](references/accident_report.md)
  - Keys: `incident_time`, `incident_location`, `parties`
- **Medical Receipt (醫療單據)**: defined in [Medical Receipt Specs](references/medical_receipt.md)
  - Keys: `patient_name`, `total_amount`, `provider_name`, `visit_date`
- **Insurance Policy (保險單)**: defined in [Insurance Policy Specs](references/insurance_policy.md)
  - Keys: `policy_number`, `applicant`, `insured_person`, `insured_id`
- **Repair Estimate (維修估價單)**: defined in [Repair Estimate Specs](references/repair_estimate.md)
  - Keys: `estimate_date`, `repair_items`, `amount`
- **Repair Details (維修明細)**: defined in [Repair Details Specs](references/repair_details.md)
  - Keys: `item_name`, `part_number`, `total_price`
- **Accident Scene Diagram (事故現場圖)**: defined in [Accident Scene Diagram Specs](references/accident_scene_diagram.md)
  - Keys: `police_unit`, `occurrence_time`, `location`, `summary`