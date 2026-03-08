# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-03-07

### Added
- 18 週完整課程教材（講義、Notebook、作業、評量規準、教師手冊、投影片）
- 17 個前端互動視覺化元件（涵蓋所有 18 週）
- Quiz 系統：54 題（3 題 x 18 週）+ 自動批改
- ML-based NLP pipeline（意圖分類 98.9%、情緒偵測 98.9%）
- 多 LLM 抽象層（Claude / GPT / Ollama / 本地模型）
- RAG 教材檢索增強（SQLite FTS5）
- 學習分析儀表板（班級總覽 + 個人分析 + 錯誤型態）
- LLM 個人化（學習歷程注入系統提示）
- Docker 部署配置（Dockerfile + docker-compose）
- Rate limiting middleware（60 req/min/IP）
- Request logging middleware
- WebSocket 身份驗證
- Error Boundary + 404 頁面
- ARIA 無障礙標籤
- 38 個後端測試 + 5 個前端測試
- 使用手冊（教師 + 學生）
- 資安與隱私政策文件
- GitHub Actions CI/CD

### Security
- 密碼改為環境變數設定（`DEFAULT_ADMIN_PASSWORD`）
- JWT 密鑰警告機制
- CORS 來源可透過環境變數設定
- WebSocket 連線驗證 token

## [0.2.0] - 2026-03-07

### Added
- FastAPI 後端框架 + 40 個 API 端點
- React + TypeScript 前端平台
- 7 層 NLP 管線（意圖、情緒、難度、主題、上下文、重排序、回應）
- 使用者驗證系統（JWT + RBAC）
- 管理面板（使用者管理、LLM 設定）

## [0.1.0] - 2026-03-07

### Added
- 專案初始化
- 設計文件與實作計劃
- 基礎目錄結構
