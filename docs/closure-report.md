# 115-18 計畫結案報告
# ML/DL 視覺化工具 -- 編纂教材

> 計畫編號：115-18
> 執行單位：馬偕醫護管理專科學校
> 計畫主持人：楊東偉
> 執行期間：2026 年
> 結案日期：2026-03-07

---

## 一、計畫目標與完成狀況

### 1.1 計畫目標

建置一套服務於大專生的「ML/DL 視覺化互動教學系統」，以 18 週完整課綱串接機器學習至深度學習概念，兼顧理論與實作。

### 1.2 產出物交付清單

| 編號 | 產出物 | 數量 | 狀態 | 備註 |
|------|--------|------|------|------|
| D-01 | 課程大綱 | 1 份 | 完成 | `curriculum/syllabus.md` (454 行) |
| D-02 | 週次講義 | 18 份 | 完成 | `curriculum/week-XX/lecture.md` |
| D-03 | 投影片 | 18 份 | 完成 | `curriculum/week-XX/slides.md` |
| D-04 | 實作 Notebook | 18 份 | 完成 | `curriculum/week-XX/notebook.ipynb` |
| D-05 | 作業說明 | 18 份 | 完成 | `curriculum/week-XX/assignment.md` |
| D-06 | 評量規準 | 18 份 | 完成 | `curriculum/week-XX/rubric.md` |
| D-07 | 教師手冊 | 18 份 | 完成 | `curriculum/week-XX/teacher-guide.md` |
| D-08 | 互動式視覺化元件 | 17 個 | 完成 | `platform/frontend/src/components/viz/` |
| D-09 | Web 教學平台 | 1 套 | 完成 | React + FastAPI fullstack |
| D-10 | AI 助教系統 | 1 套 | 完成 | 多 LLM 抽象層 + NLP pipeline |
| D-11 | 學習分析儀表板 | 1 套 | 完成 | 班級總覽 + 個人分析 |
| D-12 | 線上測驗系統 | 54 題 | 完成 | 3 題 x 18 週 + 自動批改 |
| D-13 | 範例資料集 | 3 組 | 完成 | iris, housing, mnist |
| D-14 | 期末專題評分表 | 1 份 | 完成 | `curriculum/assessment/project-rubric.md` |
| D-15 | 概念測驗模板 | 1 份 | 完成 | `curriculum/assessment/concept-quiz-template.md` |
| D-16 | 滿意度調查 | 1 份 | 完成 | `curriculum/assessment/satisfaction-survey.md` |
| D-17 | 平台使用手冊 | 1 份 | 完成 | `docs/user-manual.md` + `curriculum/platform-manual.md` |
| D-18 | 系統架構文件 | 1 份 | 完成 | `docs/architecture.md` |
| D-19 | 設計文件 | 1 份 | 完成 | `docs/plans/2026-03-07-ml-dl-visualization-design.md` |
| D-20 | 部署配置 | 1 套 | 完成 | Docker + docker-compose |
| D-21 | 安全性文件 | 1 份 | 完成 | `SECURITY.md` |
| D-22 | 貢獻指南 | 1 份 | 完成 | `CONTRIBUTING.md` |
| D-23 | 維護手冊 | 1 份 | 完成 | `MAINTENANCE.md` |
| D-24 | 授權清單 | 1 份 | 完成 | `LICENSES_INVENTORY.md` |
| D-25 | 版本紀錄 | 1 份 | 完成 | `CHANGELOG.md` |
| D-26 | CI/CD 配置 | 1 套 | 完成 | `.github/workflows/ci.yml` |
| D-27 | 驗收測試計畫 | 1 份 | 完成 | `docs/acceptance-test.md` |

**總計：18 週 x 6 檔 = 108 份教材 + 平台程式碼 + 文件 + 治理文件**

---

## 二、技術規格摘要

### 2.1 平台技術棧

| 層級 | 技術 | 版本 |
|------|------|------|
| 前端 | React + TypeScript + Vite + Tailwind CSS | 19 / 5.6 / 6.0 / 4.0 |
| 後端 | FastAPI + Python | 0.115+ / 3.11+ |
| ML 引擎 | scikit-learn | 1.6+ |
| LLM | Claude / GPT / Ollama / Local NLP | 多模型抽象層 |
| 資料庫 | SQLite + FTS5 | 內建 |
| 部署 | Docker + Nginx | docker-compose |

### 2.2 系統規模

| 指標 | 數值 |
|------|------|
| 後端 API 端點 | 30 個 |
| 前端視覺化元件 | 17 個 |
| NLP Pipeline 層數 | 7 層 |
| LLM Provider 支援 | 4 個（Claude, GPT, Ollama, Local） |
| 測驗題數 | 54 題 |
| 後端測試數 | 38 個 |
| 前端測試數 | 5 個 |
| 後端測試覆蓋率 | 48%（核心 API 層 >85%） |
| NLP 意圖分類準確率 | 98.9% |
| NLP 情緒偵測準確率 | 98.9% |

### 2.3 資安措施

- PBKDF2-HMAC-SHA256 密碼雜湊（100,000 次迭代）
- JWT Token 身份驗證（8 小時過期）
- 三級角色權限（admin / teacher / student）
- WebSocket 連線驗證
- Rate Limiting（60 req/min/IP）
- API 金鑰伺服端保管，不暴露於前端
- CORS 環境變數控制

---

## 三、18 週課程內容摘要

| 週次 | 主題 | 核心視覺化工具 | 難度 |
|------|------|---------------|------|
| 1 | Python 與資料科學環境 | 環境設置儀表板 | 核心 |
| 2 | 資料視覺化與 EDA | 互動 EDA 面板 | 核心 |
| 3 | 監督式學習、資料分割 | 資料分割視覺化 | 核心 |
| 4 | 線性回歸、梯度下降 | 損失地形 + 梯度動畫 | 核心 |
| 5 | 邏輯迴歸、ROC/PR 曲線 | 決策邊界互動 | 核心 |
| 6 | SVM 與核方法 | 核函數轉換視覺化 | 核心 |
| 7 | 樹模型與集成學習 | 決策樹生長動畫 | 核心 |
| 8 | 特徵重要度與 SHAP | SHAP 蜂群圖 | 核心 |
| 9 | 特徵工程與前處理管線 | Pipeline 流程圖 | 核心 |
| 10 | 超參數調校與學習曲線 | 超參數搜尋熱力圖 | 核心 |
| 11 | 神經網路基礎 | 激活函數互動 | 核心 |
| 12 | CNN 視覺化 | CNN 層級瀏覽器 | 進階 |
| 13 | RNN/Transformers | 序列注意力視覺化 | 進階 |
| 14 | 深度學習訓練技巧 | 訓練曲線比較器 | 核心 |
| 15 | 模型評估與公平性 | 公平性指標儀表板 | 進階 |
| 16 | MLOps 入門 | MLOps 流程圖 | 進階 |
| 17 | LLM 與嵌入應用 | 嵌入空間視覺化 | 進階 |
| 18 | 綜合專題展示 | 專題展示平台 | 核心 |

---

## 四、評量設計

| 類型 | 工具 | 說明 |
|------|------|------|
| 形成性 | 即時測驗 | 每週 3 題選擇題，系統自動批改 |
| 形成性 | 每週作業 | 含評量規準，AI 助教引導（不給答案） |
| 形成性 | AI 對話紀錄 | 學習分析追蹤提問主題與錯誤型態 |
| 總結性 | 期末專題 | 完整 ML pipeline + 報告 + 展示 |
| 回饋 | 滿意度調查 | 5 點量表，期中/期末各一次 |

---

## 五、專案治理與品質保證

| 文件 | 用途 | 路徑 |
|------|------|------|
| 安全性文件 | 資安措施、API 金鑰管理、部署注意事項 | `SECURITY.md` |
| 貢獻指南 | 開發環境、程式碼規範、提交流程 | `CONTRIBUTING.md` |
| 維護手冊 | 維護排程、備份還原、常見故障排除 | `MAINTENANCE.md` |
| 授權清單 | Python 16 + JS 13 相依套件之 SPDX 授權 | `LICENSES_INVENTORY.md` |
| 版本紀錄 | v0.1.0 → v0.3.0 變更紀錄 | `CHANGELOG.md` |
| CI/CD | 自動化測試（後端 pytest + 前端 vitest）與 Docker 建置 | `.github/workflows/ci.yml` |
| 驗收測試 | 10 大測試場景、40 項測試項目 | `docs/acceptance-test.md` |
| 授權聲明 | 教材 CC-BY-4.0 + 程式碼 MIT 雙授權 | `LICENSE` |

---

## 六、自評與展望

### 6.1 達成事項
- 完整 18 週課程教材（108 份文件）
- 全功能互動教學平台
- 多模型 AI 助教系統（含 NLP 管線）
- 學習分析儀表板
- Docker 一鍵部署
- 完整技術文件與使用手冊

### 6.2 未來改進方向
- 擴充至更多 LLM 模型支援
- 新增協作學習功能
- 整合更多 Kaggle 資料集
- 提供 PDF 版教材下載
- 建立教師社群分享機制
