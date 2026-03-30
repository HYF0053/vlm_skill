# 🌟 System Architecture & Standards

## 概述 (Overview)
`vlm_skill` 系列系統已針對長期的擴展性與可維護性進行架構級別的更新。

## 📂 專案檔案存放規範 (File Storage Standards)
為解決過去開發過程中經常出現的「找不到檔案」、「路徑衝突」及「專案目錄髒亂」問題，系統全面導入標準化目錄收納方案：
1. ** `/results` (實驗與產出結果)**: 所有的輸出文件（模型、生成圖像、分析報告等）強制寫入至 `./results` 目錄。
2. ** `/tmp` (暫存檔案)**: 所有過程中的中繼資料、臨時下載或處理用快取（如 Youtube MP3 下載、Audio chunk 等）皆存於 `./tmp`，確保根目錄整潔，系統也可設計定期清理機制。
3. ** `$PROJECT_ROOT` 支持**: 設定檔可使用該環境變數標註根目錄，確保不同腳本啟動路徑不一時依然可以找到確切的資源位置。

## 🧠 模組化設計理念
- **高度解藕**: `core/`、`skills/` 與 `ui/` 職責徹底分離。
- **外部整合**: `mcp_client` 提供了一致的操作介面讓 Agent 與外部的 MCP Server 無縫互動。

## 📚 檔案指引目錄 (Documentation Index)
深入了解各項核心技術的設計細節：
- [Agent 最佳化與工具策略 (Agent Optimizations)](agent_optimizations.md)
- [電腦視覺與 YOLO 產線化 (Computer Vision & YOLO)](computer_vision_yolo.md)
- [第三代記憶架構 (Memory V3 Architecture)](memory_v3_architecture.md)
