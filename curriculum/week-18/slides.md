# 第 18 週投影片：綜合專題開發與展示、反思與課程回饋

---

## PART A: 專題展示模板 Final Project Presentation Template

---

## Slide A1: 封面頁 Title Slide
# [專題名稱 Project Title]
### [一句話描述你的專案 One-line Description]

- 成員 Members：[姓名1、姓名2]
- 日期 Date：[展示日期]
- 課程 Course：ML/DL 視覺化工具

---

## Slide A2: 大綱 Outline
### 今天的展示流程
1. 問題與動機 Problem & Motivation
2. 資料概覽 Data Overview
3. 方法與模型 Methods & Models
4. 實驗結果 Experimental Results
5. 模型解釋 Model Interpretation
6. Live Demo
7. 結論與反思 Conclusion & Reflection

---

## Slide A3: 問題定義 Problem Definition
### 我們要解決什麼問題？
- **背景 Background**：描述問題的背景與重要性
- **目標 Objective**：明確的量化目標
- **為什麼重要 Why It Matters**：對誰有幫助？有什麼影響？

> 範例：「預測信用卡交易是否為詐欺，幫助銀行減少損失，同時避免對正常使用者造成不便。」

---

## Slide A4: 資料概覽 Data Overview
### 資料集描述
| 項目 | 說明 |
|------|------|
| 資料來源 Source | [Kaggle / UCI / 自行蒐集] |
| 樣本數 Samples | [N 筆] |
| 特徵數 Features | [M 個] |
| 目標變數 Target | [類別/數值] |
| 類別分布 Class Distribution | [各類別比例] |

### 資料品質
- 缺失值 Missing Values：[X%]
- 異常值 Outliers：[處理方式]

---

## Slide A5: EDA 重點發現
### 探索式分析的關鍵發現
- [放 1-2 張關鍵圖表]
- 特徵分布的重要觀察
- 特徵間的相關性
- 對建模的啟示

> 提示：選擇最有洞察力的圖表，不要放所有圖

---

## Slide A6: 資料前處理 Data Preprocessing
### 前處理管線 Pipeline
```
Raw Data → 缺失值處理 → 異常值處理 → 特徵編碼 → 特徵縮放 → 特徵選擇
```

| 步驟 | 方法 | 理由 |
|------|------|------|
| 缺失值 | [中位數填補 / 刪除] | [說明理由] |
| 編碼 | [One-Hot / Label] | [說明理由] |
| 縮放 | [StandardScaler / MinMax] | [說明理由] |

---

## Slide A7: 模型選擇 Model Selection
### 為什麼選擇這些模型？
| 模型 | 選擇理由 |
|------|---------|
| Baseline: Logistic Regression | 簡單、可解釋、作為基準 |
| Random Forest | 處理非線性、特徵重要度 |
| XGBoost | 通常在表格資料上表現最好 |
| [Neural Network / CNN / ...] | [特定理由] |

---

## Slide A8: 實驗設計 Experimental Design
### 實驗設定
- **資料分割**：80% 訓練 / 10% 驗證 / 10% 測試
- **交叉驗證**：Stratified 5-Fold CV
- **評估指標**：[F1 / AUC / RMSE]
- **超參數調校**：[Random Search / Optuna]
- **隨機種子**：42

---

## Slide A9: 實驗結果 Results
### 模型效能比較
| 模型 | Accuracy | F1 Score | AUC | 訓練時間 |
|------|----------|----------|-----|---------|
| Baseline | ... | ... | ... | ... |
| Model A | ... | ... | ... | ... |
| Model B | **...** | **...** | **...** | ... |

- [放一張關鍵的效能比較圖表]
- 最佳模型：[Model B]，因為 [理由]

---

## Slide A10: 模型解釋 Model Interpretation
### SHAP 分析結果
- [放 SHAP Summary Plot]
- 最重要的前 5 個特徵：
  1. Feature A — 正向影響
  2. Feature B — 負向影響
  3. ...

### 關鍵洞察
- [從 SHAP 分析得到的業務洞察]

---

## Slide A11: Live Demo
### 互動展示
- 展示互動式視覺化
- 輸入範例資料，觀察預測結果
- 展示模型解釋

> 備案：若 Demo 失敗，切換到預錄截圖

---

## Slide A12: 倫理考量 Ethical Considerations
### 我們考慮了什麼？
- **資料隱私 Data Privacy**：[處理方式]
- **公平性 Fairness**：[是否檢查不同群體的表現差異]
- **透明度 Transparency**：[模型可解釋性措施]
- **限制 Limitations**：[模型已知的限制]
- **潛在風險 Potential Risks**：[誤用可能性與對策]

---

## Slide A13: 結論與反思 Conclusion & Reflection
### 主要成果
1. [成果 1]
2. [成果 2]
3. [成果 3]

### 限制與未來方向
- 限制：[列出 2-3 項限制]
- 未來：[若有更多時間/資源，會做什麼]

### 學到了什麼
- [個人/團隊的學習收穫]

---

## Slide A14: Q&A
# 感謝聆聽！
### Questions & Answers

GitHub: [repo-url]
Email: [contact]

---

---

## PART B: 課程總結投影片 Course Summary Slides

---

## Slide B1: 恭喜完成 18 週課程！
# ML/DL 視覺化工具 — 課程總結
### Congratulations on Completing the Course!

18 週的旅程，從 Python 環境到 LLM 應用。

---

## Slide B2: 我們走過的路
### 18 週課程地圖 Course Map

**基礎篇 Foundation (Week 1-3)**
Python 環境 → EDA 視覺化 → 監督學習框架

**經典 ML (Week 4-10)**
回歸 → 分類 → SVM → 樹模型 → SHAP → 特徵工程 → 超參數

**深度學習 DL (Week 11-14)**
神經網路 → CNN → RNN/Transformer → 訓練技巧

**進階應用 Advanced (Week 15-18)**
公平性 → MLOps → LLM → 專題展示

---

## Slide B3: 核心技能回顧
### 你已經掌握的能力

| 能力 | 細項 |
|------|------|
| 資料處理 | Pandas, EDA, 特徵工程, Pipeline |
| 視覺化 | Matplotlib, Seaborn, Plotly, SHAP |
| 傳統 ML | 回歸, 分類, SVM, 樹模型, 集成 |
| 深度學習 | NN, CNN, RNN, Transformer |
| 模型工程 | 調參, 評估, 公平性, MLOps |
| AI 應用 | LLM, RAG, Prompt Engineering |

---

## Slide B4: 傳統 ML 總覽
### 模型選擇指南 Model Selection Guide

```
問題類型？
├── 回歸 → Linear Reg → Ridge/Lasso → RF → XGBoost
├── 分類 → Logistic Reg → SVM → RF → XGBoost
└── 分群 → K-Means → DBSCAN → Hierarchical

資料類型？
├── 表格資料 → 集成方法 (RF, XGBoost)
├── 影像 → CNN
├── 序列/文字 → RNN / Transformer
└── 少量資料 → 簡單模型 + 正則化
```

---

## Slide B5: 深度學習總覽
### DL 架構一覽

| 架構 | 適用場景 | 核心機制 |
|------|---------|---------|
| MLP | 表格資料 | 全連接 + 激活 |
| CNN | 影像 | 卷積 + 池化 |
| RNN/LSTM | 序列 | 循環 + 門控 |
| Transformer | NLP/多模態 | Self-Attention |
| GAN | 生成 | 生成器 vs. 判別器 |

---

## Slide B6: 評估指標速查
### 你應該用哪個指標？

| 場景 | 推薦指標 |
|------|---------|
| 平衡分類 | Accuracy, F1 |
| 不平衡分類 | F1, PR-AUC, Recall |
| 二分類排序 | ROC-AUC |
| 回歸 | MSE, MAE, R-squared |
| 公平性 | Demographic Parity, Equalized Odds |

---

## Slide B7: 完整專案流程
### End-to-End Pipeline

```
1. 問題定義        → 明確目標與成功指標
2. 資料蒐集        → 確保品質與合法性
3. EDA             → 理解資料特性
4. 特徵工程        → 建立 Pipeline
5. 模型訓練        → 從簡單到複雜
6. 超參數調校      → 驗證集 + 搜尋策略
7. 評估與解釋      → 多指標 + SHAP
8. 部署            → MLflow + Docker
9. 監測            → Model Drift Detection
```

---

## Slide B8: 可重現性的重要性
### Reproducibility Matters

```
「如果你的實驗無法被重現，那它就不是科學。」
"If your experiment cannot be reproduced, it is not science."
```

**可重現性三要素：**
1. **環境** — requirements.txt / Docker
2. **資料** — 固定版本 + 下載腳本
3. **程式碼** — 固定 Seed + 版本控制

---

## Slide B9: AI 倫理的核心原則
### Ethics in AI

1. **公平性 Fairness** — 對所有群體平等對待
2. **透明度 Transparency** — 解釋模型如何做決策
3. **隱私 Privacy** — 保護個人資料
4. **安全 Safety** — 降低錯誤預測的危害
5. **問責 Accountability** — 誰為 AI 的決策負責

---

## Slide B10: 未來學習路徑
### Where to Go Next?

**學術深造**
- 碩/博士研究（ML/DL/NLP/CV）
- 發表論文、參加頂會（NeurIPS, ICML, CVPR, ACL）

**產業實戰**
- Data Scientist / ML Engineer
- Kaggle 競賽 + 開源貢獻
- 建立個人作品集

**持續學習**
- fast.ai / Stanford Online / DeepLearning.AI
- 論文閱讀 + 技術部落格
- 參與 AI 社群與 Meetup

---

## Slide B11: 職涯方向
### ML/DL Career Paths

```
        資料分析師
          ↑
    資料科學家 ←→ ML 工程師
          ↑          ↑
    DL 研究員    MLOps 工程師
          ↑
    AI 應用工程師
```

每個方向都需要：**扎實基礎 + 持續學習 + 實作經驗**

---

## Slide B12: 給同學的話
### 結語 Final Words

> 「學習 ML/DL 不是終點，而是起點。
> 你已經擁有了最重要的東西 — 基礎知識與學習的方法。
> 保持好奇心，持續實作，不要害怕犯錯。」

**三個帶走的習慣 Three Habits to Keep:**
1. **視覺化思考** — 把抽象概念變成圖表
2. **批判性思維** — 質疑資料、模型與結論
3. **終身學習** — AI 領域快速演進，學無止境

---

## Slide B13: 課程回饋
### 請填寫課程回饋問卷

- 你的意見對未來課程改進非常重要
- 問卷為匿名填寫
- 請誠實、具體地分享你的感受
- [問卷連結 / QR Code]

---

## Slide B14: 感謝
# 感謝每一位同學的參與！
### Thank You for a Great Semester!

祝福各位在 AI 的路上越走越遠。

授課教師：楊東偉
課程網站：[URL]
