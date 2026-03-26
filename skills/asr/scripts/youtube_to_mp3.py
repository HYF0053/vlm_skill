"""
YouTube to MP3 Downloader
輸入 YouTube 網址，下載影片並轉換成 mp3 音頻檔案，供後續 ASR 語音轉文字使用。

依賴套件: yt-dlp (pip install yt-dlp)
"""

import argparse
import os
import sys


# 取得專案根目錄：優先讀 app.py 設置的環境變數，fallback 用 __file__ 推算
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DEFAULT_TMP = os.path.join(PROJECT_ROOT, "tmp")


def resolve_path(path: str) -> str:
    """若 path 為相對路徑，以 PROJECT_ROOT 為基準轉為絕對路徑。"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def download_youtube_to_mp3(
    url: str,
    output_dir: str = DEFAULT_TMP,
    filename: str = None,
) -> str:
    """
    下載 YouTube 影片並轉換為 mp3。

    Args:
        url: YouTube 影片網址
        output_dir: 輸出目錄（預設為專案根目錄下的 tmp）
        filename: 輸出檔案名稱（不含副檔名），None 時使用影片標題

    Returns:
        下載完成的 mp3 檔案絕對路徑
    """
    try:
        import yt_dlp
    except ImportError:
        print("錯誤: 缺少 yt-dlp，請先安裝: pip install yt-dlp", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 設定輸出路徑模板
    if filename:
        outtmpl = os.path.join(output_dir, f"{filename}.%(ext)s")
    else:
        outtmpl = os.path.join(output_dir, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # 取得實際輸出的檔案名稱
        if filename:
            mp3_path = os.path.join(output_dir, f"{filename}.mp3")
        else:
            title = info.get("title", "audio")
            # yt-dlp 會自動清理特殊字元
            safe_title = ydl.prepare_filename(info).replace(f".{info.get('ext', 'webm')}", "")
            safe_title = os.path.basename(safe_title)
            mp3_path = os.path.join(output_dir, f"{safe_title}.mp3")

    # 確認檔案存在（yt-dlp 可能有不同路徑處理）
    if not os.path.exists(mp3_path):
        # fallback: 搜尋 output_dir 中最新的 .mp3 檔
        mp3_files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith(".mp3")
        ]
        if mp3_files:
            mp3_path = max(mp3_files, key=os.path.getmtime)
        else:
            raise FileNotFoundError(f"找不到輸出的 mp3 檔案，請確認 output_dir: {output_dir}")

    return mp3_path


def main():
    parser = argparse.ArgumentParser(
        description="下載 YouTube 影片並轉換為 MP3（供 ASR 語音轉文字使用）"
    )
    parser.add_argument(
        "--url",
        required=True,
        help="YouTube 影片網址",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_TMP,
        help=f"輸出目錄 (預設: {DEFAULT_TMP})",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="自訂輸出檔名（不含副檔名）。未指定時使用影片標題。",
    )

    args = parser.parse_args()

    try:
        mp3_path = download_youtube_to_mp3(
            url=args.url,
            output_dir=resolve_path(args.output_dir),
            filename=args.filename,
        )
        print(mp3_path)
        print(f"[INFO] MP3 已儲存至: {mp3_path}", file=sys.stderr)
    except Exception as e:
        print(f"下載失敗: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
