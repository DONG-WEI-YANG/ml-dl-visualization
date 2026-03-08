# 第三方套件授權清單 / Third-Party License Inventory

> 本專案教材以 CC-BY-4.0 授權，程式碼以 MIT 授權。
> 以下為所使用之第三方套件及其授權。

## 後端 (Python)

| 套件 | 版本 | 授權 | 用途 |
|------|------|------|------|
| FastAPI | >=0.115 | MIT | Web 框架 |
| Uvicorn | >=0.34 | BSD-3 | ASGI 伺服器 |
| Pydantic | >=2.10 | MIT | 資料驗證 |
| pydantic-settings | >=2.0 | MIT | 環境設定 |
| scikit-learn | >=1.6 | BSD-3 | ML 模型訓練 |
| NumPy | >=2.0 | BSD-3 | 數值運算 |
| Pandas | >=2.2 | BSD-3 | 資料處理 |
| Matplotlib | >=3.9 | PSF | 圖表生成 |
| Anthropic SDK | >=0.42 | MIT | Claude API 客戶端 |
| OpenAI SDK | >=1.60 | Apache-2.0 | GPT API 客戶端 |
| websockets | >=14.0 | BSD-3 | WebSocket 支援 |
| python-dotenv | >=1.0 | BSD-3 | .env 檔案載入 |
| httpx | >=0.28 | BSD-3 | HTTP 客戶端 |
| python-multipart | >=0.0.7 | Apache-2.0 | 表單解析 |
| pytest | >=8.0 | MIT | 測試框架 |
| pytest-cov | >=6.0 | MIT | 覆蓋率報告 |

## 前端 (JavaScript/TypeScript)

| 套件 | 版本 | 授權 | 用途 |
|------|------|------|------|
| React | ^19.0.0 | MIT | UI 框架 |
| React DOM | ^19.0.0 | MIT | DOM 渲染 |
| React Router DOM | ^7.1.0 | MIT | 路由管理 |
| Recharts | ^2.15.0 | MIT | 圖表元件 |
| D3.js | ^7.9.0 | ISC | 資料視覺化 |
| Lucide React | ^0.468.0 | ISC | 圖示 |
| Vite | ^6.0.5 | MIT | 建置工具 |
| TypeScript | ~5.6.2 | Apache-2.0 | 型別系統 |
| Tailwind CSS | ^4.0.0 | MIT | CSS 框架 |
| ESLint | ^9.17.0 | MIT | 程式碼檢查 |
| Vitest | ^3.0.0 | MIT | 測試框架 |
| Testing Library | ^16.1.0 | MIT | 元件測試 |
| jsdom | ^25.0.0 | MIT | DOM 模擬 |

## 資料集來源

| 資料集 | 來源 | 授權 |
|--------|------|------|
| Iris | scikit-learn 內建 | BSD-3 |
| Housing (自建) | 本專案生成 | CC-BY-4.0 |
| MNIST (metadata) | Yann LeCun et al. | CC-BY-SA-3.0 |

## 授權相容性

所有使用之套件授權（MIT, BSD-3, Apache-2.0, ISC, PSF）均與本專案之 MIT 授權相容，可自由用於教育用途。
