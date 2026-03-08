# 第 15 週作業：模型評估與偏誤檢測、公平性與穩健性
# Week 15 Assignment: Model Evaluation, Bias Detection, Fairness & Robustness

**繳交期限 Due:** 下週上課前
**繳交方式 Submission:** 上傳 Notebook (.ipynb) 及報告 (.pdf) 至課程平台
**難度等級 Difficulty:** 進階 Advanced

---

## 作業一：進階模型評估（25%）

使用 scikit-learn 的 `digits` 資料集（手寫數字辨識，10 個類別）：

### 1a. 多類別混淆矩陣分析（10%）
1. 訓練一個分類器（如 Random Forest 或 SVM），在測試集上進行預測
2. 繪製 10x10 的混淆矩陣熱力圖 (Heatmap)
3. 分析哪些數字最容易被混淆，並提出可能原因（至少列出 3 對最常混淆的類別）
4. 計算每個數字的 Precision、Recall、F1 Score

### 1b. 平均指標比較（8%）
1. 計算 Macro Average、Micro Average、Weighted Average 三種 F1 Score
2. 解釋三者的差異，並說明在什麼情境下你會選擇哪一種
3. 人為製造類別不平衡（例如移除 80% 的數字 0 和 1 的樣本），重新計算三種平均，比較差異

### 1c. 校準曲線（7%）
1. 使用 One-vs-Rest 策略，為至少 3 個類別繪製校準曲線
2. 比較 Logistic Regression 和 Random Forest 的校準品質
3. 使用 `CalibratedClassifierCV` 進行校準，比較校準前後的差異
4. 計算各模型的 Brier Score

---

## 作業二：公平性分析（40%）

使用 [Adult Census Income](https://archive.ics.uci.edu/ml/datasets/adult) 資料集（也稱 Census Income），以 `sex` 和 `race` 作為敏感屬性進行公平性分析。

### 2a. 資料探索與偏誤識別（10%）
1. 載入 Adult Census 資料集，進行基本 EDA
2. 分析不同性別和種族群體的收入分布差異
3. 識別資料中可能存在的偏誤（歷史偏誤、代表性偏誤等）
4. 討論哪些特徵可能是敏感屬性的代理變數 (Proxy Variables)

### 2b. 公平性指標計算（15%）
1. 訓練一個收入預測模型（二元分類：>50K vs <=50K）
2. 以 `sex` 為敏感屬性，計算以下指標：
   - Demographic Parity Difference
   - Disparate Impact Ratio
   - Equalized Odds Difference
   - Average Odds Difference
3. 以 `race` 為敏感屬性，重複上述計算
4. 使用 `MetricFrame` 建立分群分析報告，展示各群體的 Accuracy、Precision、Recall
5. 繪製各群體的 ROC 曲線並比較 AUC

### 2c. 偏誤緩解實作（15%）
1. 選擇至少**兩種**偏誤緩解策略（從以下擇二）：
   - **預處理：** 使用重新加權 (Reweighting) 或重新採樣 (Resampling)
   - **訓練中：** 使用 Fairlearn 的 `ExponentiatedGradient` 或 `GridSearch` 搭配公平性約束
   - **後處理：** 使用 `ThresholdOptimizer` 進行閾值調整
2. 比較緩解前後的公平性指標變化
3. 分析緩解策略對整體準確率的影響（公平性-準確率權衡）
4. 繪製圖表展示緩解前後各群體表現的變化

---

## 作業三：穩健性測試與對抗樣本（20%）

使用 MNIST 或 CIFAR-10 資料集，搭配 PyTorch 進行穩健性測試。

### 3a. 對抗攻擊實作（10%）
1. 訓練或載入一個預訓練的影像分類模型
2. 實作 FGSM 攻擊，使用不同的 ε 值（0.01, 0.05, 0.1, 0.2, 0.3）
3. 視覺化展示：
   - 原圖、擾動、對抗樣本的並排比較
   - 準確率隨 ε 變化的折線圖
4. 分析哪些類別最容易受到攻擊

### 3b. 防禦策略（10%）
1. 實作對抗訓練 (Adversarial Training)：在訓練迴圈中加入 FGSM 生成的對抗樣本
2. 比較原始模型與對抗訓練模型在以下情境的準確率：
   - 乾淨測試資料 (Clean Test Data)
   - FGSM 對抗樣本 (ε=0.1)
   - FGSM 對抗樣本 (ε=0.3)
3. 繪製比較圖表，討論穩健性-準確率權衡 (Robustness-Accuracy Trade-off)

---

## 作業四：AI 倫理案例分析報告（15%）

撰寫一份 800-1200 字的案例分析報告，選擇以下主題之一：

### 可選主題：
- **(A) 招聘 AI 的公平性：** 分析 Amazon 履歷篩選案例，並提出你認為公平的招聘 AI 應該如何設計
- **(B) 刑事司法中的 AI：** 分析 COMPAS 系統的爭議，討論 AI 是否適合用於刑事風險評估
- **(C) 信用評分與金融公平：** 分析 Apple Card 案例，討論金融 AI 應遵循哪種公平性定義
- **(D) 自選主題：** 選擇一個你關心的 AI 倫理議題，進行深入分析

### 報告要求：
1. **背景介紹：** 說明案例的背景和問題（200 字）
2. **偏誤分析：** 識別偏誤的來源和類型（200 字）
3. **公平性討論：** 使用本週學到的公平性定義分析問題（200 字）
4. **改善建議：** 提出具體的技術和制度面改善建議（200 字）
5. **個人反思：** 分享你對此案例的看法和反思（200 字）

---

## 加分題 Bonus（10%）

### Bonus 1: 交叉公平性分析（5%）
分析 `sex` 和 `race` 的交叉效應 (Intersectional Fairness)：
- 計算「非裔女性」、「白人男性」等交叉群體的公平性指標
- 討論交叉分析為何比單一屬性分析更重要

### Bonus 2: 公平性與解釋性結合（5%）
- 使用 SHAP 或 LIME 解釋模型預測，分析不同群體的特徵重要性是否相同
- 視覺化展示：比較不同群體中前 10 個最重要特徵的 SHAP 值分布

---

## 繳交清單 Submission Checklist
- [ ] Notebook (.ipynb)：包含所有程式碼、輸出與內嵌說明
- [ ] AI 倫理案例分析報告 (.pdf)
- [ ] 程式碼可重現執行（含必要的 pip install 指令）
- [ ] 所有圖表有標題、軸標籤、圖例
- [ ] 分析有文字說明，不只有程式碼輸出

## 評分標準 Grading Criteria
- 程式碼正確性 Code Correctness: 30%
- 分析深度 Analysis Depth: 25%
- 視覺化品質 Visualization Quality: 20%
- 報告品質 Report Quality: 15%
- 反思與洞察 Reflection & Insights: 10%
