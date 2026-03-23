---
name: asr
description: "語音轉文字 (ASR) 技能。將音頻檔案轉換為文字。支援 mp3、wav、flac、m4a、ogg、youtube影片url 等格式。當用戶上傳音頻檔案並要求語音辨識、語音轉文字、transcription 時使用此技能。"
---

# ASR (Automatic Speech Recognition) Skill

## Description
使用 vLLM 部署的 Qwen3-ASR 模型，透過 OpenAI 相容 API 進行語音轉文字。

**When to Use**:
- 用戶上傳音頻檔案（.mp3, .wav, .flac, .m4a, .ogg, .webm）
- 用戶要求語音辨識或語音轉文字
- 用戶說 "transcribe", "語音轉文字", "辨識音頻"
- 用戶提供 YouTube 網址並要求轉錄內容

**When NOT to Use**:
- 用戶上傳的是圖片或文字檔案（使用其他技能）

---

## 路徑規範

> **重要**：skill 文檔目錄（`skills/asr/`）僅用於存放程式碼與說明文件，**不可**用來儲存任何輸出檔案。
>
> - **暫存檔案**（下載的音頻等）→ 存放至專案根目錄的 `tmp/` 資料夾
> - **最終結果**（轉錄文字、字幕檔等）→ 存放至專案根目錄的 `results/` 資料夾

---

## Script 1: YouTube 下載轉 MP3

```python
python skills/asr/scripts/youtube_to_mp3.py \
  --url <YouTube網址> \
  [--output_dir <輸出目錄>] \
  [--filename <自訂檔名>]
```

### Parameters
- `--url`: YouTube 影片網址（必填）
- `--output_dir`: 輸出目錄（預設 `tmp`）
- `--filename`: 自訂輸出檔名（不含副檔名），未指定時使用影片標題

### Output
輸出 mp3 檔案的絕對路徑到 stdout。

### Example
```bash
python skills/asr/scripts/youtube_to_mp3.py --url https://www.youtube.com/watch?v=xxxx --output_dir tmp
```

### 依賴
```bash
pip install yt-dlp
```
> 系統需要安裝 `ffmpeg`（用於音頻轉換）

---

## Script 2: 語音轉文字 (ASR)

```bash
python skills/asr/scripts/transcribe.py \
  --audio_path <音頻檔案路徑> \
  [--asr_url <ASR API URL>] \
  [--model <模型名稱>] \
  [--language <語言代碼>] \
  [--output_path <輸出檔案路徑>]
```

### Parameters
- `--audio_path`: 音頻檔案路徑（必填）。支援格式：mp3, wav, flac, m4a, ogg, webm。
- `--asr_url`: ASR API URL（預設從環境變數 `ASR_API_URL` 讀取，或使用 `http://localhost:8000`）。
- `--model`: 模型名稱（預設自動從 API 取得第一個可用模型）。
- `--language`: 語言代碼（可選，如 `zh`, `en`，不填則自動偵測）。
- `--output_path`: 輸出結果的檔案路徑（可選）。**應使用 `results/` 目錄**，例如 `results/transcription.txt`。

> **注意**：
> - Qwen3-ASR 僅支援純文字輸出，不支援 `verbose_json` 格式，因此無法產生含時間軸的 SRT/VTT 字幕。
> - 路徑若包含**空格或特殊字元**（如 `?`, `！`, `(` 等），**必須用引號包住**，例如 `--audio_path "tmp/my audio file.mp3"`。

### Output
輸出轉錄文字到 stdout，並可選擇同時儲存至檔案。

### Example 1: 純文字轉錄（結果印至 stdout）
```bash
python skills/asr/scripts/transcribe.py --audio_path tmp/audio.mp3
```

### Example 2: 轉錄並儲存至 results/
```bash
python skills/asr/scripts/transcribe.py --audio_path tmp/audio.mp3 --output_path results/transcription.txt
```

---

## 完整 YouTube → 文字流程

```bash
# 1. 下載 YouTube 影片為 mp3（暫存至 tmp/）
mp3_path=$(python skills/asr/scripts/youtube_to_mp3.py --url <YouTube URL> --output_dir tmp)

# 2. 轉錄並儲存至 results/
python skills/asr/scripts/transcribe.py --audio_path $mp3_path --output_path results/transcription.txt
```
