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

## Script 1: YouTube 下載轉 MP3

```python
python skills/asr/scripts/youtube_to_mp3.py \
  --url <YouTube網址> \
  [--output_dir <輸出目錄>] \
  [--filename <自訂檔名>]
```

### Parameters
- `--url`: YouTube 影片網址（必填）
- `--output_dir`: 輸出目錄（預設 `/tmp`）
- `--filename`: 自訂輸出檔名（不含副檔名），未指定時使用影片標題

### Output
輸出 mp3 檔案的絕對路徑到 stdout。

### Example
```bash
python skills/asr/scripts/youtube_to_mp3.py --url https://www.youtube.com/watch?v=xxxx --output_dir /tmp
```

### 依賴
```bash
pip install yt-dlp
```
> 系統需要安裝 `ffmpeg`（用於音頻轉換）

---

## Script 2: 語音轉文字 (ASR)

```python
python skills/asr/scripts/transcribe.py \
  --audio_path <音頻檔案路徑> \
  [--asr_url <ASR API URL>] \
  [--model <模型名稱>] \
  [--language <語言代碼>]
```

### Parameters
- `--audio_path`: 音頻檔案路徑（必填）
- `--asr_url`: ASR API URL（預設從環境變數 `ASR_API_URL` 讀取，或使用 `http://localhost:8000`）
- `--model`: 模型名稱（預設自動從 API 取得第一個可用模型，或從環境變數 `ASR_MODEL` 讀取）
- `--language`: 語言代碼（可選，如 `zh`, `en`，不填則自動偵測）
- `--output_path`: 輸出結果的文字檔路徑（可選）

### Output
輸出轉錄的文字內容到 stdout，格式為純文字。

### Example
```bash
python skills/asr/scripts/transcribe.py --audio_path /tmp/audio.mp3 --asr_url http://10.1.1.7:8000
```

---

## 完整 YouTube → 文字 流程

```python
# Step 1: 下載 YouTube 影片為 mp3
mp3_path = $(python skills/asr/scripts/youtube_to_mp3.py --url <YouTube URL> --output_dir /tmp)

# Step 2: 語音轉文字
python skills/asr/scripts/transcribe.py --audio_path $mp3_path --asr_url $ASR_API_URL
```
