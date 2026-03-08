# 期末專題評分表
# Final Project Grading Rubric

> 適用週次：第 18 週 (Week 18) 期末專題展示
> 滿分 Total Score: 100 分
> 評分者：授課教師 + 同儕互評 (Peer Review)

---

## 評分總覽 Grading Overview

| 面向 Dimension | 權重 Weight | 細項 Criteria |
|---------------|-----------|--------------|
| 技術實作 Technical Implementation | 30% | 程式品質、模型選擇、評估方法 |
| 資料處理 Data Processing | 20% | EDA、前處理、特徵工程 |
| 視覺化與展示 Visualization & Presentation | 20% | 圖表品質、簡報、Demo |
| 文件與可重現性 Documentation & Reproducibility | 15% | README、requirements、結構 |
| 倫理與反思 Ethics & Reflection | 15% | Ethics Checklist、反思報告 |

---

## 面向一：技術實作 Dimension 1: Technical Implementation（30 分）

### 評分細項 Criteria

#### 1.1 程式品質 Code Quality（10 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 9-10 | 程式碼結構清晰，模組化設計 (Modular Design)，遵循 PEP 8 風格，有充分的型別提示 (Type Hints) 與錯誤處理 (Error Handling)，函式命名具描述性。 |
| 良好 Good | 7-8 | 程式碼結構合理，大部分遵循 PEP 8，有基本的函式拆分與註解，偶有小瑕疵但不影響可讀性。 |
| 尚可 Satisfactory | 5-6 | 程式碼可執行但結構較鬆散，部分程式碼重複，命名或風格不一致，註解不足但核心邏輯可理解。 |
| 待改善 Needs Improvement | 3-4 | 程式碼可執行但難以閱讀，缺少模組化設計，大量重複程式碼，幾乎沒有註解或錯誤處理。 |
| 不足 Insufficient | 0-2 | 程式碼無法正常執行，或存在嚴重邏輯錯誤，結構混亂無法理解。 |

#### 1.2 模型選擇與實作 Model Selection & Implementation（10 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 9-10 | 根據問題特性合理選擇模型，嘗試至少 3 種以上模型進行比較，並清楚說明選擇理由。模型實作正確，超參數調校 (Hyperparameter Tuning) 有系統性（如 Grid Search / Random Search）。 |
| 良好 Good | 7-8 | 模型選擇合理，至少比較 2 種模型，有基本的超參數調校。實作正確，能說明模型選擇的考量。 |
| 尚可 Satisfactory | 5-6 | 選擇 1-2 種模型，有基本實作但超參數使用預設值或僅做少量調整，缺乏比較分析。 |
| 待改善 Needs Improvement | 3-4 | 模型選擇不太適合問題類型，實作有誤或不完整，沒有超參數調校的嘗試。 |
| 不足 Insufficient | 0-2 | 未實作模型，或模型選擇完全不適當，實作有嚴重錯誤。 |

#### 1.3 評估方法 Evaluation Methodology（10 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 9-10 | 使用多種適當的評估指標 (Evaluation Metrics)，正確實施交叉驗證 (Cross-Validation)，提供混淆矩陣 (Confusion Matrix)、ROC/PR 曲線等視覺化分析。結果解讀深入，包含統計顯著性考量。 |
| 良好 Good | 7-8 | 使用至少 2-3 種適當指標，實施交叉驗證，有基本的結果視覺化與解讀。 |
| 尚可 Satisfactory | 5-6 | 使用基本指標（如 Accuracy），有訓練集/測試集分割但可能缺少交叉驗證，結果解讀較表面。 |
| 待改善 Needs Improvement | 3-4 | 僅使用單一指標，評估流程有瑕疵（如資料洩漏 Data Leakage），結果缺乏解讀。 |
| 不足 Insufficient | 0-2 | 沒有正式的評估流程，或評估方法有嚴重錯誤。 |

---

## 面向二：資料處理 Dimension 2: Data Processing（20 分）

### 評分細項 Criteria

#### 2.1 探索式資料分析 EDA (Exploratory Data Analysis)（7 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 6-7 | 全面且有深度的 EDA，包含資料分布 (Distribution)、相關性 (Correlation)、缺失值 (Missing Values)、異常值 (Outliers) 分析。使用多種視覺化方式呈現發現。每個分析都附有洞察 (Insight) 說明。 |
| 良好 Good | 4-5 | 涵蓋主要的 EDA 面向，有適當的視覺化呈現，能從資料中提取有意義的發現。 |
| 尚可 Satisfactory | 3 | 有基本的 EDA（如 `df.describe()`、基本圖表），但分析深度不足或缺少洞察。 |
| 待改善 Needs Improvement | 1-2 | EDA 非常簡略，僅有少數統計量或圖表，缺少分析與說明。 |
| 不足 Insufficient | 0 | 沒有 EDA，或 EDA 內容與資料集不相關。 |

#### 2.2 資料前處理 Data Preprocessing（7 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 6-7 | 完善的前處理流程：缺失值處理策略合理、類別變數編碼正確 (Encoding)、數值特徵縮放適當 (Scaling)，使用 sklearn Pipeline 確保流程可重現。處理策略均有理由說明。 |
| 良好 Good | 4-5 | 前處理流程涵蓋主要步驟，大致正確，有基本的流程組織。 |
| 尚可 Satisfactory | 3 | 有基本前處理但可能遺漏重要步驟，或某些處理策略不太適當。 |
| 待改善 Needs Improvement | 1-2 | 前處理不完整或有明顯錯誤（如在分割前對全部資料做標準化造成資料洩漏）。 |
| 不足 Insufficient | 0 | 沒有資料前處理，或前處理有嚴重概念錯誤。 |

#### 2.3 特徵工程 Feature Engineering（6 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 5-6 | 創造性地進行特徵工程：衍生有意義的新特徵 (Derived Features)、使用特徵選擇 (Feature Selection) 方法、分析特徵重要度 (Feature Importance)，特徵工程的決策有領域知識 (Domain Knowledge) 支持。 |
| 良好 Good | 3-4 | 有基本的特徵工程嘗試（如多項式特徵 Polynomial Features、交互特徵 Interaction Features），並分析了特徵重要度。 |
| 尚可 Satisfactory | 2 | 僅做少量或基本的特徵處理，缺少創造性的特徵工程嘗試。 |
| 待改善 Needs Improvement | 1 | 幾乎沒有特徵工程，直接使用原始特徵。 |
| 不足 Insufficient | 0 | 完全沒有特徵工程的相關內容。 |

---

## 面向三：視覺化與展示 Dimension 3: Visualization & Presentation（20 分）

### 評分細項 Criteria

#### 3.1 圖表品質 Chart Quality（8 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 7-8 | 圖表設計專業：選用適當的圖表類型、標題與軸標籤完整、色彩搭配和諧且有意義 (Meaningful Color Encoding)、圖例清楚、解析度足夠。能使用互動式圖表 (Interactive Charts) 增強表達力。 |
| 良好 Good | 5-6 | 圖表整潔清楚，類型選擇適當，標籤大致完整，能有效傳達資訊。 |
| 尚可 Satisfactory | 3-4 | 有基本圖表但設計粗糙，可能缺少標題、標籤或圖例，或圖表類型選擇不夠理想。 |
| 待改善 Needs Improvement | 1-2 | 圖表品質差，難以閱讀或理解，或圖表數量過少。 |
| 不足 Insufficient | 0 | 幾乎沒有視覺化呈現。 |

#### 3.2 簡報表達 Presentation Delivery（6 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 5-6 | 簡報結構清晰（問題定義 -> 方法 -> 結果 -> 結論），投影片設計簡潔有力，口頭報告流暢且掌握時間。能清楚回答 Q&A。 |
| 良好 Good | 4 | 簡報結構合理，內容涵蓋重點，口頭表達尚可，能回答大部分提問。 |
| 尚可 Satisfactory | 3 | 簡報包含基本要素但結構可以更好，口頭報告可能過快/過慢或不太流暢。 |
| 待改善 Needs Improvement | 1-2 | 簡報結構不清楚或投影片內容過多/過少，口頭表達困難，難以回答提問。 |
| 不足 Insufficient | 0 | 沒有進行簡報，或簡報嚴重不完整。 |

#### 3.3 Demo 展示 Live Demo（6 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 5-6 | 現場展示流暢，清楚示範從輸入資料到模型預測的完整流程。能即時調整參數並解釋結果變化。有備用方案應對技術問題。 |
| 良好 Good | 4 | 展示大致流暢，能示範核心功能，遇到小問題能快速處理。 |
| 尚可 Satisfactory | 3 | 能進行基本展示但準備不夠充分，過程中有明顯停頓或需要查找資料。 |
| 待改善 Needs Improvement | 1-2 | 展示過程問題多，如程式無法執行、環境未設定好等。 |
| 不足 Insufficient | 0 | 無法進行 Demo 展示。 |

---

## 面向四：文件與可重現性 Dimension 4: Documentation & Reproducibility（15 分）

### 評分細項 Criteria

#### 4.1 README 文件 README Documentation（5 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 5 | README 完整且專業：含專案描述、安裝步驟、使用方式、資料來源、結果摘要、貢獻者資訊、授權條款 (License)。格式美觀，含目錄導覽。 |
| 良好 Good | 4 | README 涵蓋大部分必要資訊（專案描述、安裝、使用方式），格式清楚。 |
| 尚可 Satisfactory | 3 | README 存在但內容簡略，僅有基本描述，缺少安裝或使用說明。 |
| 待改善 Needs Improvement | 1-2 | README 非常簡略（如僅一行標題），或內容過時/不正確。 |
| 不足 Insufficient | 0 | 沒有 README 文件。 |

#### 4.2 環境與依賴管理 Environment & Dependency Management（5 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 5 | 提供 `requirements.txt` 或 `pyproject.toml` 並鎖定版本號，含 `.env.example` 範例。按照文件步驟可以一鍵重現環境。若使用 Docker，提供 `Dockerfile`。 |
| 良好 Good | 4 | 提供 `requirements.txt` 且版本號大致正確，按步驟可重現環境。 |
| 尚可 Satisfactory | 3 | 有 `requirements.txt` 但版本號不完整或遺漏套件，需要手動除錯才能重現。 |
| 待改善 Needs Improvement | 1-2 | 依賴管理檔案不完整或有錯誤，難以重現環境。 |
| 不足 Insufficient | 0 | 沒有任何依賴管理檔案。 |

#### 4.3 專案結構 Project Structure（5 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 5 | 專案結構清晰且組織良好：原始碼、資料、Notebook、文件各有獨立目錄。檔案命名一致且具描述性。含 `.gitignore`，不包含不必要的大型檔案。 |
| 良好 Good | 4 | 專案結構合理，有基本的目錄分類，檔案命名大致一致。 |
| 尚可 Satisfactory | 3 | 有基本結構但不夠整齊，部分檔案放置位置不太合理，或命名不一致。 |
| 待改善 Needs Improvement | 1-2 | 專案結構雜亂，所有檔案放在同一目錄，或包含大量不必要的檔案。 |
| 不足 Insufficient | 0 | 專案結構完全無組織，無法理解各檔案的用途。 |

**建議的專案結構範例 Recommended Project Structure：**
```
final-project/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── data/
│   ├── raw/              # 原始資料
│   └── processed/        # 前處理後資料
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluate.py
├── reports/
│   ├── figures/          # 圖表輸出
│   └── final_report.pdf
└── ethics/
    ├── ethics_checklist.md
    └── reflection.md
```

---

## 面向五：倫理與反思 Dimension 5: Ethics & Reflection（15 分）

### 評分細項 Criteria

#### 5.1 倫理檢核 Ethics Checklist（8 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 7-8 | 完整填寫倫理檢核清單，深入分析專案中的潛在偏誤 (Bias)、公平性 (Fairness)、隱私 (Privacy) 議題。提出具體的風險緩解措施 (Mitigation Strategies)。能從多方利害關係人 (Stakeholders) 角度思考。 |
| 良好 Good | 5-6 | 完成倫理檢核清單，對主要倫理議題有所討論，提出一些緩解措施。 |
| 尚可 Satisfactory | 3-4 | 有倫理檢核但內容較表面，僅列出議題但缺少深入分析或具體措施。 |
| 待改善 Needs Improvement | 1-2 | 倫理檢核非常簡略，僅形式性地填寫，缺乏實質思考。 |
| 不足 Insufficient | 0 | 沒有提供倫理檢核內容。 |

**倫理檢核清單範本 Ethics Checklist Template：**

> 請逐一回答以下問題：

- [ ] **資料來源 Data Source**：資料的取得是否合法合規？是否有適當的授權？
- [ ] **隱私保護 Privacy**：資料中是否包含個人可識別資訊 (PII)？是否已做去識別化 (Anonymization)？
- [ ] **偏誤分析 Bias Analysis**：訓練資料是否存在抽樣偏誤 (Sampling Bias)、標籤偏誤 (Label Bias)、歷史偏誤 (Historical Bias)？
- [ ] **公平性評估 Fairness**：模型對不同群體（性別 Gender、種族 Race、年齡 Age 等）的表現是否一致？
- [ ] **透明度 Transparency**：模型的決策過程是否可以被理解和解釋？
- [ ] **影響評估 Impact Assessment**：如果模型被部署，可能對哪些人群產生正面或負面影響？
- [ ] **誤用風險 Misuse Risk**：此模型/技術是否可能被惡意使用？有什麼防範措施？
- [ ] **環境影響 Environmental Impact**：模型訓練的計算資源消耗是否合理？

#### 5.2 反思報告 Reflection Report（7 分）

| 等級 Level | 分數 Score | 描述 Description |
|-----------|-----------|-----------------|
| 優秀 Excellent | 6-7 | 反思深入且具體：清楚描述專案過程中的學習收穫、遇到的挑戰與解決方式、技術決策的取捨考量 (Trade-offs)、對 ML/DL 實務的新認識。提出未來改進方向與自我學習計劃。 |
| 良好 Good | 4-5 | 反思內容具體，涵蓋學習收穫與挑戰，能指出技術上的優缺點與改進空間。 |
| 尚可 Satisfactory | 3 | 有反思但較為籠統，偏向表面描述而缺少深入思考。 |
| 待改善 Needs Improvement | 1-2 | 反思非常簡短或流於形式，沒有實質內容。 |
| 不足 Insufficient | 0 | 沒有反思報告。 |

**反思報告建議架構 Reflection Report Suggested Structure：**

1. **專案摘要 Project Summary**（100-200 字）
   - 問題定義、使用資料、核心方法

2. **學習收穫 What I Learned**（200-300 字）
   - 技術知識、軟技能、團隊合作

3. **挑戰與解決 Challenges & Solutions**（200-300 字）
   - 遇到的主要困難、嘗試的解決方案、最終結果

4. **技術決策反思 Technical Decision Reflection**（200-300 字）
   - 為什麼選擇某個模型/方法？回顧來看是否會做不同選擇？

5. **倫理思考 Ethical Considerations**（100-200 字）
   - 對 AI 倫理議題的個人反思

6. **未來改進方向 Future Improvements**（100-200 字）
   - 如果有更多時間/資源，會如何改善此專案？

---

## 評分彙總表 Scoring Summary Sheet

| 面向 Dimension | 細項 Criteria | 滿分 Max | 得分 Score |
|---------------|-------------|---------|-----------|
| **技術實作 (30%)** | | | |
| | 1.1 程式品質 Code Quality | 10 | ___ |
| | 1.2 模型選擇 Model Selection | 10 | ___ |
| | 1.3 評估方法 Evaluation | 10 | ___ |
| **資料處理 (20%)** | | | |
| | 2.1 EDA | 7 | ___ |
| | 2.2 前處理 Preprocessing | 7 | ___ |
| | 2.3 特徵工程 Feature Engineering | 6 | ___ |
| **視覺化與展示 (20%)** | | | |
| | 3.1 圖表品質 Chart Quality | 8 | ___ |
| | 3.2 簡報 Presentation | 6 | ___ |
| | 3.3 Demo 展示 | 6 | ___ |
| **文件與可重現性 (15%)** | | | |
| | 4.1 README | 5 | ___ |
| | 4.2 環境管理 Environment | 5 | ___ |
| | 4.3 專案結構 Structure | 5 | ___ |
| **倫理與反思 (15%)** | | | |
| | 5.1 倫理檢核 Ethics | 8 | ___ |
| | 5.2 反思報告 Reflection | 7 | ___ |
| **總分 Total** | | **100** | **___** |

---

## 同儕互評指引 Peer Review Guide

期末專題展示當天，每位學生需對其他組別進行同儕互評。

### 互評要求 Requirements

1. 每位學生需評審**至少 3 組**其他專題
2. 使用簡化版評分表（見下方），針對三個面向評分
3. 互評成績佔期末專題總分的 **20%**（即教師評分佔 80%，同儕互評佔 20%）
4. 需附上建設性的文字回饋（至少 50 字）

### 同儕互評簡化評分表 Simplified Peer Review Form

| 面向 Aspect | 評分 (1-5) | 說明 |
|------------|-----------|------|
| 技術深度 Technical Depth | ___ / 5 | 方法是否適當？結果是否可信？ |
| 展示品質 Presentation Quality | ___ / 5 | 簡報是否清楚？Demo 是否流暢？ |
| 整體印象 Overall Impression | ___ / 5 | 專題的完整性與創意 |

**文字回饋 Written Feedback：**
```
優點 Strengths:


改善建議 Suggestions for Improvement:


```

---

## 評分注意事項 Grading Notes

1. **抄襲處理 Plagiarism**：若發現程式碼或報告有抄襲嫌疑，將啟動學術誠信調查流程，嚴重者該次成績以零分計
2. **遲交扣分 Late Submission**：專題報告遲交每日扣總分 5%，展示當天缺席者需在一週內補報告，補報成績上限為 80%
3. **團隊貢獻 Team Contribution**：若為團隊專題，需附上貢獻度自評與互評表，授課教師保留依個人貢獻調整分數的權利
4. **加分項目 Bonus Points**：
   - 使用進階技術（如 Docker 部署、CI/CD、模型監測）：+5 分
   - 專題成果開源並獲得外部 Star/Fork：+3 分
   - 使用課程以外的資料集並有原創分析：+2 分
   - 加分上限：總分不超過 100 分
