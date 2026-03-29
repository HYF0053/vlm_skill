# 🚀 Computer Vision & YOLO Workflow

## 概述 (Overview)
`computer-vision` 技能已重構為生產環境等級的 `ultralytics-yolo-expert` 框架。本次更新旨在標準化 YOLO 模型訓練、驗證、預測及導出流程。

## 🏆 實驗管理系統 (Experiment Management)
為解決過去配置與輸出散落各處的問題，導入了全新的實驗管理機制：
1. **單一資料夾輸出**: 所有的實驗輸出（包含 Hyperparameters、YOLO Log、模型權重）都會自動集中至對應的 timestamp 專案資料夾內（統一存放於 `/results` 目錄下）。
2. **非破壞性架構 (Non-destructive)**: 每次訓練與測試都會建立獨立的紀錄，不會覆蓋歷史資料，確保可重現性 (Reproducibility)。
3. **路徑解析修復**: 全局導入 `$PROJECT_ROOT` 解析機制，修正過去 Dataset YAML 載入時經常遇到的 `FileNotFoundError`，使各配置檔皆能安全使用絕對路徑。

## 效能與最佳化 (Performance & Optimization)
- **Inference Optimizer (`inference_optimizer.py`)**:
  - 分析模型結構。
  - 對 PyTorch、ONNX 等多種格式進行推理效能標竿測試 (Benchmarking)。
  - 提供針對不同部屬目標設備的具體優化建議。

## 流程對齊 (Workflow Alignment)
所有的視覺工程操作皆對齊了 `app.py` 的標準輸出設定，使得 CV 相關技能在與 Agent 互動時，不會產生系統路徑衝突，並能由 Agent 自動產出完整的實驗追蹤報告。
