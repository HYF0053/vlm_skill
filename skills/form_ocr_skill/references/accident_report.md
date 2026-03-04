# Accident Report (事故聯單) Extraction Rules

## Mission Objectives
Only extract and output the contents of the following fields in the specific JSON keys:
1. 發生時間 (Incident Time) -> JSON Key: `incident_time`
2. 發生地點 (Incident Location) -> JSON Key: `incident_location`
3. 當事人資訊 (Parties Information) -> JSON Key: `parties`

## Identification and positioning rules
1. Scan all Chinese characters in the image (printed, handwritten, stamped). Do **not** output "all text". Only extract the above three fields.

2. Field source and content rules:
   - **發生時間** (`incident_time`):
     - Priority Rule:
       - If both "受理時間" and "發生時間" exist on the form, strictly extract the value from "受理時間".
       - If only "發生時間" exists, extract "發生時間".
     - Location Hint: "受理時間" is usually located in the top "報案人" section. Do not confuse it with "發生時間" in the middle section if both are present.
     - If the label field cannot be found, this field must be **null**.
     - Standardized Output: Always output in the format: YYYY 年 MM 月 DD 日 HH 時 mm 分.(e.g., 114 年 10 月 20 日 10 時 56 分).

   - **發生地點** (`incident_location`):
     - Target label: "地點" or "發生地".
     - Extraction Logic:
       - Multi-line Boundary: Capture all rows of text located to the right of the "地點" label within its logical cell boundary. Do not stop at the first line break or the "/" symbol.
       - Complete Content: Ensure that if an address spans two or more lines (e.g., Line 1: 29 巷口, Line 2: 107 巷 1 弄口), both segments are captured in full.
       - Normalization: Join the segments with a space or keep the original separator. Do not skip the street/lane numbers in the middle.
       - Verification: If the address appears as "A / B", output both "A" and "B" (e.g., 桃園市...29巷口 / 桃園市...107巷1弄口).
     - If the label field cannot be found, this field must be **blank**.

   - **當事人資訊** (`parties`):
     - Step 1 (Identify Primary Party): Extract Name and Phone from the "報案人" section.
       - Phone Formatting: Strip all hyphens, spaces, or special characters. Output only digits (e.g., 0912345678)value is only taken from the right side of the label "聯絡電話" or "電話" or "手機" or "電話號碼" or "車主電話".
       - Plate Formatting: Includes "- " (e.g., ABC-1234)
     - Step 2 (Contextual Vehicle Binding): In the "報案(受理)內容", identify the vehicle driven by the Reporter.
       - Look for keywords: "報案人駕駛", "本人騎乘", "我方車號".
       - Mandatory Link: The License Plate associated with these keywords MUST be paired with the Reporter's Name from Step 1. Do NOT simply pair the first name found with the first plate found.
     - Step 3 (Secondary Parties): Extract other involved parties mentioned as "對方", "B車", or "遭撞車輛".
     - Format: List each party. For JSON output, use a structured list of objects (e.g., `[{"name": "...", "plate": "...", "phone": "..."}]`).
