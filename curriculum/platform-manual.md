# 互動平台操作手冊
# Interactive Platform User Manual

> 版本 Version: 1.0
> 最後更新 Last Updated: 2026-03-07
> 適用對象：修課學生、助教、授課教師

---

## 目錄 Table of Contents

1. [平台總覽 Platform Overview](#1-平台總覽-platform-overview)
2. [帳號與登入 Account & Login](#2-帳號與登入-account--login)
3. [課程模組導覽 Course Module Navigation](#3-課程模組導覽-course-module-navigation)
4. [視覺化互動區操作 Visualization Interactive Area](#4-視覺化互動區操作-visualization-interactive-area)
5. [AI 助教使用指南 AI Teaching Assistant Guide](#5-ai-助教使用指南-ai-teaching-assistant-guide)
6. [任務與作業提交 Tasks & Assignment Submission](#6-任務與作業提交-tasks--assignment-submission)
7. [學習分析儀表板 Learning Analytics Dashboard](#7-學習分析儀表板-learning-analytics-dashboard)
8. [常見問題與除錯 FAQ & Troubleshooting](#8-常見問題與除錯-faq--troubleshooting)

---

## 1. 平台總覽 Platform Overview

### 1.1 系統架構 System Architecture

本平台採用前後端分離架構 (Decoupled Frontend/Backend Architecture)，各元件說明如下：

```
┌─────────────────────────────────────────────────────────────────┐
│                    使用者瀏覽器 User Browser                      │
│            (React 18 + TypeScript + Vite + Tailwind CSS)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS / WebSocket
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   後端 API 伺服器 Backend API Server              │
│                    (FastAPI + Python 3.11+)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ 課程 API     │  │ LLM 助教 API │  │ 學習分析 Analytics │    │
│  │ Course API   │  │ LLM Tutor    │  │ API                │    │
│  └──────────────┘  └──────┬───────┘  └────────────────────┘    │
└────────────────────────────┼────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ 資料庫       │  │ LLM 服務     │  │ ML 運算引擎      │
│ Database     │  │ (Claude /    │  │ (scikit-learn /   │
│              │  │  GPT / Ollama)│  │  PyTorch)        │
└──────────────┘  └──────────────┘  └──────────────────┘
```

**前端 Frontend：** 使用 React 18 搭配 TypeScript，透過 Vite 打包，使用 Tailwind CSS 進行樣式管理，以 Recharts 與 D3.js 呈現互動圖表。

**後端 Backend：** 使用 FastAPI 提供 RESTful API，整合 scikit-learn 與 PyTorch 進行即時模型訓練與推論，並透過多模型抽象層 (Multi-Model Abstraction Layer) 連接不同的 LLM 服務。

**LLM 助教模組：** 支援 Anthropic Claude、OpenAI GPT、Ollama 等多種 LLM 後端，依情境自動切換學習模式 (Learning Mode) 與作業模式 (Assignment Mode)。

**學習分析模組：** 收集並分析學生的學習行為資料，提供即時進度追蹤與弱點分析。

### 1.2 支援瀏覽器與系統需求 Supported Browsers & System Requirements

| 項目 Item | 最低需求 Minimum | 建議 Recommended |
|-----------|-----------------|------------------|
| 瀏覽器 Browser | Chrome 90+, Firefox 88+, Edge 90+, Safari 15+ | Chrome 最新版 (Latest Chrome) |
| 螢幕解析度 Resolution | 1280 x 720 | 1920 x 1080 或以上 |
| 網路 Network | 穩定網路連線 (Stable connection) | 10 Mbps 以上 |
| 作業系統 OS | Windows 10, macOS 11, Ubuntu 20.04 | Windows 11, macOS 14, Ubuntu 22.04 |
| JavaScript | 必須啟用 (Must be enabled) | — |
| 記憶體 RAM | 4 GB | 8 GB 以上 |

> **注意 Note：** 本平台大量使用 WebSocket 進行即時互動，請確保網路環境允許 WebSocket 連線。部分企業防火牆 (Firewall) 可能會阻擋 WebSocket 流量。

---

## 2. 帳號與登入 Account & Login

### 2.1 註冊流程 Registration

1. **開啟平台首頁**：在瀏覽器輸入平台網址（由授課教師提供）
2. **點選「註冊 Register」按鈕**：位於首頁右上角
3. **填寫基本資料**：
   - 電子郵件 (Email)：請使用學校信箱
   - 使用者名稱 (Username)：建議使用「學號_姓名」格式，例如 `s112001_王小明`
   - 密碼 (Password)：至少 8 個字元，需包含大小寫字母與數字
4. **輸入課程邀請碼 (Invitation Code)**：由授課教師於第一堂課提供
5. **驗證信箱**：點擊驗證信中的連結完成註冊
6. **完成 Complete**：自動導向課程首頁

### 2.2 登入流程 Login

1. 開啟平台首頁
2. 輸入 Email 與密碼
3. 點選「登入 Login」
4. 若啟用雙因素驗證 (2FA)，請輸入驗證碼

**忘記密碼 Forgot Password：** 點擊登入頁面的「忘記密碼」連結，系統會發送密碼重設信件至註冊信箱。

### 2.3 個人資料設定 Profile Settings

登入後，點擊右上角頭像進入「個人設定 Profile Settings」：

| 設定項目 Setting | 說明 Description |
|-----------------|------------------|
| 頭像 Avatar | 上傳個人照片或選擇預設頭像 |
| 顯示名稱 Display Name | 課堂討論與互動時顯示的名稱 |
| 偏好語言 Preferred Language | 介面語言（中文/英文） |
| 通知設定 Notifications | 電子郵件通知、平台內通知開關 |
| AI 助教偏好 AI Tutor Preferences | 回答語言、詳細程度（簡潔/標準/詳細）|
| 主題 Theme | 淺色模式 (Light) / 深色模式 (Dark) |
| 程式編輯器字體大小 Editor Font Size | 12px - 24px |

---

## 3. 課程模組導覽 Course Module Navigation

### 3.1 左側導覽列 Left Navigation Sidebar

登入後，畫面左側會顯示 18 週課程導覽列：

```
┌────────────────────┐
│  課程總覽 Overview  │
├────────────────────┤
│  W01 課程導論       │  ← 已完成 (Completed) 顯示綠色勾
│  W02 資料視覺化     │  ← 進行中 (In Progress) 顯示藍色點
│  W03 監督式學習     │  ← 鎖定 (Locked) 顯示灰色鎖
│  W04 線性回歸       │
│  W05 分類           │
│  W06 SVM            │
│  W07 樹模型         │
│  W08 特徵重要度     │
│  W09 特徵工程       │
│  W10 超參數調校     │
│  W11 神經網路基礎   │
│  W12 CNN            │
│  W13 RNN/Transformer│
│  W14 訓練技巧       │
│  W15 模型評估       │
│  W16 MLOps          │
│  W17 LLM 應用       │
│  W18 期末專題       │
├────────────────────┤
│  學習分析 Analytics │
│  設定 Settings      │
└────────────────────┘
```

**狀態標示 Status Indicators：**
- 綠色勾 (Green Check)：該週所有活動已完成
- 藍色點 (Blue Dot)：正在進行中
- 灰色鎖 (Gray Lock)：尚未解鎖（需完成前週活動）
- 橘色驚嘆號 (Orange Exclamation)：有待完成的作業即將到期

### 3.2 每週頁面結構 Weekly Page Structure

點擊任一週次後，右側主區域顯示該週內容，結構如下：

```
┌──────────────────────────────────────────────────┐
│  第 N 週：[主題名稱]                              │
│  Week N: [Topic Title]                            │
│                                                    │
│  難度等級 Level: ★★☆ (核心 Core / 進階 Advanced)  │
│  預計時間 Est. Time: 90 分鐘                       │
├──────────────────────────────────────────────────┤
│                                                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │ 講義    │ │ 投影片   │ │ Notebook │ │ 作業   │ │
│  │ Lecture │ │ Slides   │ │          │ │ HW     │ │
│  └─────────┘ └─────────┘ └─────────┘ └────────┘ │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │         視覺化互動區                          │ │
│  │         Visualization Interactive Area        │ │
│  │                                                │ │
│  │  [參數面板]          [圖表顯示區]              │ │
│  │  [Parameter Panel]   [Chart Display]           │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │         AI 助教對話區                          │ │
│  │         AI Tutor Chat                          │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │         學習檢核點                             │ │
│  │         Learning Checkpoints                   │ │
│  └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

**四大區塊 Four Main Sections：**

1. **教材標籤 Material Tabs**：講義 (Lecture)、投影片 (Slides)、Notebook、作業 (Assignment)
2. **視覺化互動區 Visualization Area**：該週核心互動元件
3. **AI 助教對話區 AI Tutor Chat**：即時問答與提示
4. **學習檢核點 Checkpoints**：確認學習進度的自我檢查清單

---

## 4. 視覺化互動區操作 Visualization Interactive Area

### 4.1 參數調整滑桿 Parameter Sliders

視覺化互動區的左側為參數面板 (Parameter Panel)，提供各種可調整的滑桿與輸入控制元件：

**滑桿操作 Slider Controls：**
- **拖曳 Drag**：按住滑桿圓鈕左右拖曳以調整數值
- **點擊軌道 Click Track**：直接點擊滑桿軌道可跳到該位置
- **精確輸入 Precise Input**：點擊滑桿旁的數值可直接鍵入精確數值
- **重設 Reset**：點擊參數名稱旁的「重設 Reset」按鈕可回復預設值

**常見參數範例 Common Parameters：**
| 參數 Parameter | 適用場景 Context | 範圍 Range |
|---------------|-----------------|------------|
| 學習率 Learning Rate | 梯度下降、神經網路 | 0.0001 ~ 1.0 |
| 正則化強度 Regularization (C) | SVM、邏輯迴歸 | 0.01 ~ 100 |
| 樹深度 Max Depth | 決策樹、隨機森林 | 1 ~ 20 |
| 隱藏層神經元數 Hidden Units | 神經網路 | 1 ~ 256 |
| Dropout Rate | 神經網路 | 0.0 ~ 0.9 |
| k 值 (k-NN) | k-近鄰 | 1 ~ 50 |
| Epochs 訓練輪數 | 模型訓練 | 1 ~ 1000 |

**快捷操作 Shortcuts：**
- `Ctrl + Z` / `Cmd + Z`：復原上一步參數變更 (Undo)
- `Ctrl + Shift + Z` / `Cmd + Shift + Z`：重做 (Redo)
- 雙擊 (Double-click) 滑桿：回復預設值

### 4.2 圖表互動 Chart Interaction

互動區右側為圖表顯示區 (Chart Display)，支援以下操作：

**縮放 Zoom：**
- **滾輪縮放 Scroll Zoom**：在圖表上滾動滑鼠滾輪放大或縮小
- **框選放大 Box Zoom**：按住 `Shift` 鍵並拖曳滑鼠，選取區域放大
- **雙擊重設 Double-click Reset**：雙擊圖表回到原始比例
- **觸控手勢 Touch Gesture**：在觸控裝置上使用兩指捏合縮放

**懸停提示 Hover Tooltips：**
- 將滑鼠移至圖表上的資料點 (Data Point)，會顯示該點的詳細資訊
- 顯示內容包含：座標值、類別標籤 (Label)、預測機率 (Probability) 等
- 懸停在圖例 (Legend) 項目上可單獨高亮該系列

**圖表匯出 Export：**
- 點擊圖表右上角的匯出按鈕 (Export Button)
- 支援格式 (Supported Formats)：
  - **PNG**：高解析度靜態圖片（適合報告插圖）
  - **SVG**：向量格式（適合論文與簡報）
  - **CSV**：匯出圖表底層資料
  - **JSON**：匯出完整圖表配置（可供後續重現）

**圖表類型 Chart Types：**
- 散佈圖 (Scatter Plot)：觀察資料分布與決策邊界
- 折線圖 (Line Chart)：追蹤訓練損失與指標變化
- 熱力圖 (Heatmap)：超參數搜尋結果、混淆矩陣
- 長條圖 (Bar Chart)：特徵重要度比較
- 3D 曲面圖 (3D Surface)：損失函數地形圖
- 動畫圖 (Animation)：梯度下降過程、決策樹生長

### 4.3 即時模型訓練操作 Real-time Model Training

部分週次提供即時模型訓練功能，操作方式如下：

1. **選擇資料集 Select Dataset**：從下拉選單選擇或上傳自訂資料
2. **設定模型參數 Set Parameters**：透過滑桿或輸入框調整超參數
3. **點擊「訓練 Train」按鈕**：啟動訓練流程
4. **觀察即時更新 Watch Real-time Updates**：
   - 訓練損失曲線 (Training Loss Curve) 即時繪製
   - 決策邊界 (Decision Boundary) 隨訓練迭代更新
   - 指標數值 (Metrics) 即時顯示
5. **暫停/繼續 Pause/Resume**：可隨時暫停觀察當前狀態
6. **重新訓練 Retrain**：調整參數後重新訓練以比較結果

**訓練進度指示 Training Progress Indicator：**
- 進度條 (Progress Bar) 顯示當前 Epoch / 總 Epochs
- 預估剩餘時間 (ETA) 顯示於進度條右側
- 即時指標面板顯示 Loss、Accuracy、F1 等數值

---

## 5. AI 助教使用指南 AI Teaching Assistant Guide

### 5.1 學習模式 vs 作業模式 Learning Mode vs Assignment Mode

AI 助教具備兩種運作模式，系統會根據使用情境自動切換：

**學習模式 Learning Mode：**
- **適用情境**：閱讀講義、觀看投影片、操作互動視覺化、自主學習時
- **行為特徵**：
  - 可提供詳細解釋與完整範例程式碼
  - 可直接回答概念性問題
  - 會主動推薦延伸閱讀資源
  - 可幫助除錯非作業相關的程式碼
- **標示**：對話區上方顯示 「學習模式 Learning Mode」 藍色標籤

**作業模式 Assignment Mode：**
- **適用情境**：進行每週作業、期中/期末專題時
- **行為特徵**：
  - **不會直接給出答案**，而是採用分層提示策略 (Scaffolded Hints)
  - 以引導式問題幫助學生思考
  - 可指出錯誤方向，但不會直接修正程式碼
  - 提供概念提示而非具體解法
- **標示**：對話區上方顯示「作業模式 Assignment Mode」 橘色標籤

> **重要 Important：** 系統會自動偵測使用情境並切換模式。若發現模式不符合當前需求，可透過對話輸入 `/mode learning` 或 `/mode assignment` 手動切換。

### 5.2 提問模板範例 Question Templates

為了獲得更好的回答品質，建議使用以下提問模板：

**概念理解型 Concept Understanding：**
```
我在學習 [主題名稱]，對於 [具體概念] 不太理解。
能否用簡單的比喻或圖解來說明 [概念] 的運作原理？
我目前的理解是 [你的理解]，這樣正確嗎？
```

**程式除錯型 Code Debugging：**
```
我在執行 [描述任務] 時遇到以下錯誤：
[貼上錯誤訊息]
我的程式碼如下：
[貼上相關程式碼片段]
我已經嘗試了 [你嘗試過的方法]，但問題仍然存在。
```

**視覺化解讀型 Visualization Interpretation：**
```
我正在觀察 [圖表類型]，其中 [描述你看到的現象]。
這代表模型 [你的推測] 嗎？
調整 [參數名稱] 會如何影響這個結果？
```

**比較分析型 Comparative Analysis：**
```
在 [任務/資料集] 上，[方法 A] 和 [方法 B] 的差異是什麼？
什麼情況下應該選擇 [方法 A]？又在什麼條件下 [方法 B] 更適合？
```

### 5.3 分層提示策略 Scaffolded Hint Strategy

在作業模式下，AI 助教採用三層漸進式提示策略：

**第一層 — 方向提示 Directional Hint：**
- 提示思考方向，不涉及具體做法
- 例如：「想想看，當資料不是線性可分 (Linearly Separable) 時，你可以如何轉換特徵空間？」

**第二層 — 概念提示 Conceptual Hint：**
- 提供相關概念與方法名稱
- 例如：「可以查閱核技巧 (Kernel Trick) 的概念，特別是 RBF 核函數的作用原理。」
- 需在第一層提示後 5 分鐘仍未解決才觸發，或學生主動請求

**第三層 — 步驟提示 Step Hint：**
- 提供具體的解題步驟框架（但不包含完整程式碼）
- 例如：「步驟一：使用 `SVC(kernel='rbf')` 建立模型；步驟二：調整 `gamma` 參數...」
- 需在第二層提示後 10 分鐘仍未解決才觸發，或學生主動請求

> **提示觸發指令 Hint Commands：**
> - `/hint` — 請求下一層提示
> - `/hint reset` — 重設提示層級
> - `/explain [concept]` — 在作業模式中請求概念解釋（不視為作弊）

### 5.4 錯誤回報格式 Bug Report Format

若發現 AI 助教回答有誤或平台功能異常，請使用以下格式回報：

```
/report
- 類型 Type: [概念錯誤 Concept Error / 程式錯誤 Code Error / 功能異常 Bug / 其他 Other]
- 週次 Week: [第 N 週]
- 描述 Description: [詳細描述問題]
- 重現步驟 Steps to Reproduce:
  1. [步驟一]
  2. [步驟二]
  3. ...
- 預期行為 Expected: [應該看到什麼]
- 實際行為 Actual: [實際看到什麼]
- 截圖 Screenshot: [如有，可附上]
```

### 5.5 學術誠信提醒 Academic Integrity Notice

使用 AI 助教時，請務必遵守以下學術誠信規範：

1. **合理使用 Fair Use：** AI 助教是學習輔助工具，而非代寫工具。你應該理解 AI 提供的每一行程式碼與每一個概念。

2. **禁止行為 Prohibited Actions：**
   - 直接複製 AI 助教在作業模式下的完整解答提交（系統會偵測並標記）
   - 使用外部 AI 工具繞過作業模式限制取得答案後提交
   - 將他人帳號的 AI 對話紀錄抄襲提交

3. **鼓勵行為 Encouraged Actions：**
   - 與 AI 助教討論概念，用自己的話重新整理
   - 在 AI 提示的基礎上自行實作與延伸
   - 在作業中註明「此段概念經 AI 助教討論後整理」

4. **系統監測 System Monitoring：**
   - 平台會記錄學生與 AI 助教的互動歷程
   - 作業提交時會與 AI 對話紀錄進行相似度比對 (Similarity Check)
   - 異常行為會自動標記並通知授課教師審查

> **聲明 Disclaimer：** AI 助教的回答可能存在錯誤。所有 AI 生成的內容都應經過驗證。如發現錯誤，請使用錯誤回報功能通知教學團隊。

---

## 6. 任務與作業提交 Tasks & Assignment Submission

### 6.1 作業提交流程 Submission Workflow

1. **進入作業頁面**：點擊該週的「作業 Assignment」標籤
2. **閱讀作業說明**：仔細閱讀題目要求、評分標準 (Rubric) 與截止日期 (Deadline)
3. **完成作業**：
   - **Notebook 作業**：直接在平台內建編輯器完成，或在本地完成後上傳 `.ipynb` 檔案
   - **報告型作業**：上傳 PDF 或 Markdown 檔案
   - **程式型作業**：上傳 `.py` 檔案或 GitHub 儲存庫連結
4. **自我檢核 Self-Check**：完成作業頁底部的自我檢核清單（見 6.2）
5. **提交 Submit**：點擊「提交作業 Submit Assignment」按鈕
6. **確認 Confirm**：系統會顯示提交確認視窗，確認後無法再修改（除非在截止日期前重新提交）
7. **查看狀態 Check Status**：提交後可在「我的作業 My Assignments」頁面查看批改狀態

**提交格式要求 Submission Format Requirements：**
| 作業類型 Type | 接受格式 Accepted Formats | 最大檔案大小 Max Size |
|--------------|--------------------------|---------------------|
| Notebook | `.ipynb` | 50 MB |
| 報告 Report | `.pdf`, `.md` | 20 MB |
| 程式碼 Code | `.py`, `.zip`, GitHub URL | 10 MB |
| 資料集 Dataset | `.csv`, `.json` | 100 MB |

### 6.2 檢核點確認 Checkpoint Verification

每週作業包含學習檢核點 (Learning Checkpoints)，學生需在提交前逐一確認：

**檢核點範例（以第 4 週線性回歸為例）：**
- [ ] 我能解釋損失函數 (Loss Function) 的意義
- [ ] 我能描述梯度下降 (Gradient Descent) 的運作步驟
- [ ] 我在互動平台上實驗了至少 3 種不同的學習率 (Learning Rate)
- [ ] 我能解釋過擬合 (Overfitting) 與欠擬合 (Underfitting) 的差異
- [ ] 我的 Notebook 包含完整的註解與說明
- [ ] 我已與 AI 助教討論了至少一個疑問

> **注意 Note：** 檢核點不計入成績，但未完成檢核點的作業會收到提醒。持續未完成檢核點可能影響課堂參與分數 (Participation Score)。

### 6.3 提示觸發條件 Hint Trigger Conditions

以下條件會觸發系統自動提供提示或提醒：

| 條件 Condition | 觸發動作 Action |
|---------------|----------------|
| 在某個問題停留超過 15 分鐘 | 自動顯示第一層方向提示 |
| 連續 3 次執行程式碼產生錯誤 | AI 助教主動提供除錯建議 |
| 距截止日期不足 24 小時且未開始 | 推送提醒通知 |
| 完成作業但未通過自我檢核 | 提示回顧未完成的檢核點 |
| 提交作業後得分低於 60 分 | 自動開放重新提交機會並提供學習建議 |

---

## 7. 學習分析儀表板 Learning Analytics Dashboard

點擊左側導覽列的「學習分析 Analytics」可進入個人學習儀表板。

### 7.1 進度追蹤 Progress Tracking

**整體進度 Overall Progress：**
- 18 週課程完成百分比 (Completion Rate) 的圓環圖
- 每週完成狀態的時間軸 (Timeline) 視圖
- 累計學習時數 (Total Study Hours) 統計

**作業進度 Assignment Progress：**
- 各週作業完成狀態（已提交 Submitted / 待提交 Pending / 遲交 Late / 未提交 Missing）
- 成績趨勢折線圖 (Grade Trend Chart)
- 與班級平均的比較（匿名化 Anonymized）

**互動參與度 Engagement Metrics：**
- AI 助教對話次數與品質評估
- 視覺化互動區使用時間
- 課堂討論參與紀錄

### 7.2 弱點分析 Weakness Analysis

系統會根據作業表現、AI 對話紀錄與互動行為，自動識別學習弱點：

**弱點類別 Weakness Categories：**

| 類別 Category | 分析指標 Indicators | 建議 Recommendation |
|--------------|--------------------|--------------------|
| 概念理解 Concept | 測驗分數、AI 提問類型 | 推薦補充教材與影片 |
| 程式實作 Coding | 程式錯誤類型、除錯次數 | 推薦練習題與範例程式碼 |
| 數學基礎 Math | 公式推導作業表現 | 推薦數學補充資源 |
| 視覺化解讀 Visualization | 圖表解讀題表現 | 推薦互動練習 |
| 統整應用 Integration | 專題表現、綜合題 | 推薦跨週複習路徑 |

**弱點雷達圖 Weakness Radar Chart：**
系統以五角雷達圖 (Radar Chart) 呈現上述五個維度的能力值（0-100 分），讓學生一目了然自己的強弱項。

### 7.3 目標設定 Goal Setting

學生可在儀表板中設定個人學習目標：

1. **每週學習時數目標 Weekly Study Hours Goal**：設定每週預計投入的學習時數
2. **成績目標 Grade Target**：設定期望的總成績等級（A+/A/B+/B/C+/C）
3. **技能目標 Skill Goals**：勾選希望強化的能力面向
4. **里程碑 Milestones**：設定期中、期末的階段性目標

系統會根據目標設定提供：
- 每週進度提醒（是否達標）
- 預測達標機率 (Predicted Probability of Achieving Goal)
- 個人化學習路徑建議 (Personalized Learning Path)

---

## 8. 常見問題與除錯 FAQ & Troubleshooting

### 8.1 登入問題 Login Issues

**Q: 無法登入，顯示「帳號或密碼錯誤」。**
A: 請確認以下事項：
1. 確認使用註冊時的 Email（非使用者名稱）
2. 檢查大小寫是否正確（密碼區分大小寫）
3. 嘗試使用「忘記密碼」功能重設密碼
4. 若仍無法登入，請聯繫助教或授課教師

**Q: 註冊時顯示「邀請碼無效」。**
A: 邀請碼有期限限制。請向授課教師確認是否使用正確的課程邀請碼，以及邀請碼是否已過期。

**Q: 驗證信沒有收到。**
A: 請檢查垃圾信件 (Spam) 資料夾。若仍未收到，可在登入頁面點擊「重新發送驗證信 Resend Verification」。如使用學校信箱，可能存在延遲，請等待 10 分鐘後再試。

### 8.2 操作問題 Operation Issues

**Q: 視覺化圖表無法載入，顯示空白。**
A: 請依序嘗試：
1. 重新整理頁面 (`Ctrl + F5` / `Cmd + Shift + R`)
2. 清除瀏覽器快取 (Clear Cache)
3. 確認 JavaScript 已啟用
4. 嘗試使用 Chrome 瀏覽器
5. 檢查網路連線是否穩定

**Q: 滑桿調整後圖表沒有更新。**
A: 部分複雜運算需要數秒處理時間。請觀察是否有載入指示 (Loading Indicator)。若超過 30 秒仍無反應，請重新整理頁面。

**Q: 即時訓練卡住，進度條不動。**
A: 可能原因：
1. 網路連線中斷 — 檢查網路狀態
2. 訓練參數導致不收斂 — 嘗試降低學習率 (Learning Rate) 或減少模型複雜度
3. 伺服器負載過高 — 等待數分鐘後重試
4. 點擊「停止 Stop」按鈕後重新設定參數再訓練

**Q: Notebook 無法上傳。**
A: 請確認：
1. 檔案格式為 `.ipynb`
2. 檔案大小不超過 50 MB
3. Notebook 中不包含過大的圖片嵌入（建議使用外部連結）
4. 檔名不包含特殊字元（建議使用英文與數字）

### 8.3 API 相關問題 API Issues

**Q: AI 助教回覆速度很慢或無回應。**
A: 可能原因：
1. LLM 服務暫時性負載過高 — 請等待 1-2 分鐘後重試
2. 網路延遲 — 檢查網路連線
3. 輸入內容過長 — 嘗試簡化提問
4. 若持續無回應，可嘗試在對話區輸入 `/reset` 重啟對話

**Q: AI 助教的回答看起來不正確。**
A: AI 助教的回答可能存在錯誤（即所謂的「幻覺 Hallucination」），建議：
1. 交叉驗證 — 對照講義、教科書或官方文件
2. 追問 — 請 AI 助教提供參考來源
3. 回報 — 使用 `/report` 指令回報錯誤（見 5.4 節）

**Q: 顯示「API 配額已用完」。**
A: 每位學生每日有 AI 助教使用次數上限。可在個人設定查看剩餘配額。配額每日凌晨重設。若需要額外配額，請聯繫授課教師。

### 8.4 作業問題 Assignment Issues

**Q: 作業提交後想要修改。**
A: 在截止日期前可以重新提交，新提交會覆蓋舊提交。截止日期後無法再次提交，如有特殊原因需延期，請聯繫授課教師。

**Q: 提交時顯示「檔案格式不支援」。**
A: 請確認上傳的檔案格式符合作業要求（見 6.1 提交格式要求表）。若使用 `.zip` 壓縮，請確保壓縮檔內的結構正確。

**Q: 作業得分偏低但不確定原因。**
A: 建議步驟：
1. 檢視作業回饋 (Feedback)：點擊已批改的作業查看詳細評語
2. 對照評分標準 (Rubric)：檢查各項得分細節
3. 諮詢 AI 助教：在學習模式下討論作業概念（注意：AI 不會直接提供作業答案）
4. 辦公室時間 (Office Hours)：預約授課教師或助教的面談時間

**Q: 遲交是否會扣分？**
A: 遲交政策依課程規定：
- 截止後 24 小時內提交：扣該次作業成績 10%
- 截止後 24-48 小時內提交：扣 20%
- 截止超過 48 小時：不接受提交（除非有事先核准的延期）

---

## 附錄 Appendix

### 快捷鍵一覽 Keyboard Shortcuts

| 快捷鍵 Shortcut | 功能 Function |
|-----------------|--------------|
| `Ctrl/Cmd + K` | 開啟快速搜尋 (Quick Search) |
| `Ctrl/Cmd + /` | 開啟/關閉 AI 助教面板 |
| `Ctrl/Cmd + S` | 儲存目前 Notebook |
| `Ctrl/Cmd + Enter` | 執行目前程式碼區塊 |
| `Shift + Enter` | 執行並跳到下一個區塊 |
| `Ctrl/Cmd + Z` | 復原 (Undo) |
| `Ctrl/Cmd + Shift + Z` | 重做 (Redo) |
| `Esc` | 關閉彈出視窗 |
| `←` / `→` | 切換上/下一週 |

### 聯繫方式 Contact

| 角色 Role | 聯繫方式 Contact |
|-----------|----------------|
| 授課教師 Instructor | 課程公告區或 Email |
| 助教 Teaching Assistant | 平台內訊息或 Email |
| 技術支援 Tech Support | 平台右下角「回報問題 Report Issue」按鈕 |
| 緊急問題 Urgent | 課程 Discord / LINE 群組 |

---

> 本手冊隨平台更新持續修訂。如有建議或發現錯誤，歡迎透過平台回報功能或聯繫教學團隊。
