# ML/DL 視覺化工具教學系統
# ML/DL Visualization Interactive Teaching System

> 115-18 計畫：ML/DL 視覺化工具 -- 編纂教材

18 週機器學習 (Machine Learning) 與深度學習 (Deep Learning) 視覺化互動教材，含互動平台、LLM 助教與學習分析。

## 架構 Architecture

```
ml-dl-visualization/
  curriculum/           18 週課程教材（講義、Notebook、作業、評量、教師手冊）
  platform/
    frontend/           React + TypeScript 互動平台
    backend/            FastAPI 後端 API
  datasets/             範例資料集
  docs/                 設計文件、使用手冊
  docker-compose.yml    一鍵啟動
```

## 技術棧 Tech Stack

| 層級 | 技術 |
|------|------|
| 前端 | React 19 + TypeScript + Vite + Tailwind CSS + Recharts/D3.js |
| 後端 | FastAPI + Python 3.11+ + scikit-learn |
| LLM | Anthropic Claude / OpenAI GPT / Ollama / 本地 NLP（多模型抽象層） |
| 資料庫 | SQLite + FTS5 |
| 部署 | Docker + Nginx |

## 快速開始 Quick Start

### Docker（推薦）

```bash
# 1. 設定環境變數
cp platform/backend/.env.example platform/backend/.env
# 編輯 .env 填入 API key

# 2. 啟動
docker compose up -d

# 3. 開啟瀏覽器
# 前端: http://localhost:3000
# API 文件: http://localhost:8000/docs
```

### 本地開發

```bash
# 後端
cd platform/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000

# 前端（另開終端）
cd platform/frontend
npm install
npm run dev
# 開啟 http://localhost:5173
```

## 預設帳號

| 帳號 | 密碼 | 角色 |
|------|------|------|
| admin | admin123 | 管理員 |

> 首次登入後請修改密碼。密碼可透過環境變數 `DEFAULT_ADMIN_PASSWORD` 設定。

## 測試 Testing

```bash
# 後端測試
cd platform/backend
pip install -e ".[dev]"
pytest -v

# 前端測試
cd platform/frontend
npm test
```

## 文件 Documentation

- [設計文件](docs/plans/2026-03-07-ml-dl-visualization-design.md)
- [系統架構圖](docs/architecture.md)
- [使用手冊](docs/user-manual.md)（教師 + 學生）
- [投影片渲染指南](docs/slides-guide.md)
- [資料集說明](datasets/README.md)
- [後端 API](platform/backend/README.md)
- [前端元件](platform/frontend/README.md)
- [資安與隱私政策](SECURITY.md)
- [版本紀錄](CHANGELOG.md)
- [貢獻指南](CONTRIBUTING.md)
- [維護移交計畫](MAINTENANCE.md)
- [第三方授權清單](LICENSES_INVENTORY.md)
- [結案報告](docs/closure-report.md)
- [驗收測試情境](docs/acceptance-test.md)（40 項 UAT 測試）
- [授權條款](LICENSE)（教材 CC-BY-4.0 / 程式碼 MIT）

## 課程內容 18-Week Curriculum

| 週次 | 主題 | 視覺化工具 |
|------|------|-----------|
| 1 | Python 與資料科學環境 | 環境設置儀表板 |
| 2 | 資料視覺化與 EDA | 互動 EDA 面板 |
| 3 | 監督式學習、資料分割 | 資料分割視覺化 |
| 4 | 線性回歸、梯度下降 | 損失地形 + 梯度動畫 |
| 5 | 分類：邏輯迴歸、ROC/PR | 決策邊界互動 |
| 6 | SVM 與核方法 | 核函數轉換視覺化 |
| 7 | 樹模型與集成學習 | 決策樹生長動畫 |
| 8 | 特徵重要度與 SHAP | SHAP 蜂群圖互動 |
| 9 | 特徵工程與前處理管線 | Pipeline 流程圖 |
| 10 | 超參數調校與學習曲線 | 超參數搜尋熱力圖 |
| 11 | 神經網路基礎 | 激活函數互動 |
| 12 | CNN 視覺化 | CNN 層級瀏覽器 |
| 13 | RNN/Transformers | 序列注意力視覺化 |
| 14 | 深度學習訓練技巧 | 訓練曲線比較器 |
| 15 | 模型評估與公平性 | 公平性指標儀表板 |
| 16 | MLOps 入門 | MLOps 流程圖 |
| 17 | LLM 與嵌入應用 | 嵌入空間視覺化 |
| 18 | 綜合專題展示 | 專題展示平台 |
