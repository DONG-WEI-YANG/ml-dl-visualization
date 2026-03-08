# ML/DL 視覺化教學平台 - 使用手冊

## 系統啟動

### 方式一：Docker 一鍵啟動（推薦）

```bash
# 在專案根目錄
docker compose up -d
```

- 前端：http://localhost:3000
- 後端 API：http://localhost:8000
- API 文件：http://localhost:8000/docs

### 方式二：本地開發啟動

**後端：**
```bash
cd platform/backend
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # 編輯 .env 填入 API key
uvicorn app.main:app --reload --port 8000
```

**前端：**
```bash
cd platform/frontend
npm install
npm run dev
```

開啟 http://localhost:5173

---

## 教師使用指南

### 1. 登入系統

- 預設管理員帳號：`admin`
- 首次登入後請立即至管理面板修改密碼

### 2. 課程導覽

首頁顯示 18 週課程卡片，點擊進入各週頁面：
- **視覺化互動區** — 本週核心概念的互動式視覺化工具
- **即時測驗** — 3 題選擇題，即時批改
- **AI 助教** — 右側面板可與 AI 助教對話

### 3. 教學建議

每堂課 90 分鐘建議流程：
1. **理論講解**（30 分鐘）：使用 `curriculum/week-XX/lecture.md` 講義
2. **互動實作**（40 分鐘）：帶領學生操作平台視覺化工具
3. **討論與評量**（20 分鐘）：進行即時測驗、開放 AI 助教提問

### 4. 管理面板（/admin）

- **使用者管理**：新增/停用學生、教師帳號
- **LLM 設定**：切換 AI 模型（Claude / GPT / Ollama / 本地）
- **RAG 設定**：啟用/停用教材檢索增強
- **NLP 訓練**：重新訓練意圖/情緒分類模型

### 5. 學習分析儀表板（/dashboard）

- **班級總覽**：學生人數、事件總數、平均分數
- **熱門提問主題**：AI 助教最常被詢問的概念
- **學生個人分析**：輸入學生 ID 查看每週成績、學習時間、互動次數
- **錯誤型態分類**：學生常見錯誤類型統計

### 6. 作業與評量

- 每週作業說明：`curriculum/week-XX/assignment.md`
- 評量規準：`curriculum/week-XX/rubric.md`
- 期末專題評分表：`curriculum/assessment/project-rubric.md`
- 學生滿意度調查：`curriculum/assessment/satisfaction-survey.md`

---

## 學生使用指南

### 1. 進入平台

開啟瀏覽器前往平台網址，點選當週課程卡片。

### 2. 使用互動視覺化

每週頁面上方為互動式視覺化工具：
- 拖曳滑桿調整參數，觀察模型變化
- 點擊按鈕切換不同演算法或資料集
- 即時看到結果（如決策邊界、損失曲線、注意力熱力圖等）

### 3. 完成測驗

每週頁面包含 3 題隨堂測驗：
1. 閱讀題目並選擇答案
2. 點擊「提交」送出
3. 系統即時批改，綠色為正確、紅色為錯誤
4. 可看到正確答案與得分

### 4. 使用 AI 助教

右側面板可與 AI 助教對話：
- **一般模式**：自由提問 ML/DL 概念
- **作業模式**：AI 會引導你思考，不會直接給答案
- 支援中文提問
- AI 會根據你的學習歷程調整回答深度

### 5. 查看學習紀錄

前往「學習分析」頁面，輸入你的學生 ID：
- 查看每週成績趨勢
- 了解自己的學習時間分配
- 檢視常見錯誤型態

---

## 常見問題

**Q: 視覺化元件沒有顯示？**
A: 確認瀏覽器支援 JavaScript，建議使用 Chrome 或 Firefox 最新版。

**Q: AI 助教無回應？**
A: 確認後端服務正在運行，且 `.env` 中已設定有效的 API key。

**Q: 如何切換 AI 模型？**
A: 管理員在 /admin 頁面可切換 LLM 提供者（Claude/GPT/Ollama/本地模式）。

**Q: 忘記密碼？**
A: 請管理員在管理面板重設密碼。

**Q: 如何離線使用？**
A: 將 LLM 設為 `local` 模式（使用本地 NLP），視覺化元件不需網路即可運作。
