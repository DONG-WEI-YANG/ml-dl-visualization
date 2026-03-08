# 第 5 週作業：分類模型建構與 ROC 分析
# Week 5 Assignment: Classification Model Building & ROC Analysis

**繳交期限 Deadline：** 第 6 週上課前
**繳交格式 Format：** Jupyter Notebook (.ipynb) + PDF 匯出
**配分 Total Points：** 100 分

---

## 作業目標 Assignment Objectives

1. 獨立實作邏輯迴歸 (Logistic Regression) 分類模型
2. 理解並應用分類評估指標
3. 繪製與解讀 ROC/PR 曲線
4. 處理不平衡資料集 (Imbalanced Dataset) 並比較不同策略

---

## Part A: 概念題 Conceptual Questions (30 分)

### A1. Sigmoid 與決策邊界 (10 分)

1. **(3 分)** 請解釋為什麼邏輯迴歸使用 Sigmoid 函數而非直接用線性回歸做分類？至少列舉三個理由。

2. **(3 分)** 如果邏輯迴歸的權重為 $w = [2, -1]$，偏置為 $b = 0.5$，閾值 $\tau = 0.5$：
   - 寫出決策邊界的方程式
   - 判斷點 $(1, 3)$ 屬於正類還是負類
   - 計算點 $(1, 3)$ 屬於正類的機率

3. **(4 分)** 解釋正則化參數 C 對決策邊界的影響。畫出（或用文字描述）C 值很小和很大時，決策邊界的差異。何時應該選擇較大或較小的 C 值？

### A2. 損失函數 (10 分)

1. **(5 分)** 給定以下資料，手動計算 BCE Loss（寫出完整計算過程）：

   | 樣本 | 真實標籤 $y$ | 預測機率 $\hat{p}$ |
   |:----:|:---:|:---:|
   | 1 | 1 | 0.8 |
   | 2 | 0 | 0.3 |
   | 3 | 1 | 0.6 |
   | 4 | 0 | 0.9 |

2. **(5 分)** 觀察上面第 4 個樣本（$y=0, \hat{p}=0.9$），解釋為什麼這個樣本對總損失的貢獻特別大。如果使用 MSE 作為損失函數，這個樣本的損失是多少？比較兩者的差異，說明為什麼分類問題偏好使用 BCE 而非 MSE。

### A3. 評估指標分析 (10 分)

1. **(5 分)** 某醫院的癌症篩檢系統在 10,000 個樣本上的結果如下：

   |  | 預測為癌症 | 預測為正常 |
   |:---:|:---:|:---:|
   | **實際癌症** | 80 | 20 |
   | **實際正常** | 500 | 9,400 |

   計算以下指標：Accuracy, Precision, Recall, F1-Score, Specificity, FPR
   並回答：這個系統的 Accuracy 很高，但你認為它是一個好的篩檢系統嗎？為什麼？

2. **(5 分)** 在以下兩個應用場景中，你更關注 Precision 還是 Recall？請解釋理由：
   - 場景 A：銀行的信用卡詐欺偵測系統
   - 場景 B：社群媒體的仇恨言論自動過濾系統

---

## Part B: 實作題 Coding Tasks (50 分)

### B1. Breast Cancer 資料集分類 (25 分)

使用 `sklearn.datasets.load_breast_cancer` 資料集完成以下任務：

1. **(3 分)** 載入資料，進行基本的 EDA（資料形狀、類別分布、特徵統計）

2. **(3 分)** 將資料分割為訓練集 (70%) 和測試集 (30%)，使用分層抽樣 (Stratified Sampling)，並進行特徵標準化 (StandardScaler)

3. **(4 分)** 訓練邏輯迴歸模型，列印分類報告 (Classification Report)

4. **(5 分)** 繪製混淆矩陣熱力圖，並在圖上標註 TP、TN、FP、FN 的數值

5. **(5 分)** 繪製 ROC 曲線和 PR 曲線（在同一個 Figure 的兩個子圖中），標註 AUC 和 AP 值

6. **(5 分)** 測試 5 個不同的閾值（0.3, 0.4, 0.5, 0.6, 0.7），計算每個閾值下的 Precision、Recall、F1-Score，並用表格呈現。找出最佳閾值並解釋你的選擇標準。

> **提示：** 在癌症診斷的場景中，漏診 (False Negative) 的代價通常大於誤診 (False Positive)。這會如何影響你的閾值選擇？

### B2. 不平衡資料集實驗 (25 分)

使用以下程式碼建立不平衡資料集：

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=5,
    n_redundant=2, n_classes=2,
    weights=[0.95, 0.05],
    random_state=42
)
```

1. **(3 分)** 顯示類別分布，計算不平衡比 (Imbalance Ratio)

2. **(5 分)** 訓練一個不做任何處理的基線模型 (Baseline)，報告 Accuracy、Precision、Recall、F1-Score。解釋為什麼 Accuracy 很高但模型可能沒有實際用處。

3. **(7 分)** 分別使用以下三種策略處理不平衡資料，訓練模型並報告相同的指標：
   - 策略 A：`class_weight='balanced'`
   - 策略 B：隨機過採樣 (Random Oversampling) — 將少數類複製到與多數類相同數量
   - 策略 C：隨機欠採樣 (Random Undersampling) — 將多數類隨機抽樣到與少數類相同數量

4. **(5 分)** 繪製四種策略的 ROC 和 PR 曲線在同一張圖上，比較差異

5. **(5 分)** 撰寫分析報告（200-300 字），比較四種策略的優缺點，並說明：
   - 在什麼情境下你會選擇哪種策略？
   - ROC 和 PR 曲線的比較結果是否一致？如果不一致，以哪個為準？為什麼？

---

## Part C: 進階挑戰題 Advanced Challenge (20 分，選做 Optional)

### C1. 從零實作多類別 Softmax 回歸 (10 分)

不使用 scikit-learn，從零實作一個 Softmax 回歸分類器：

1. 實作 Softmax 函數
2. 實作交叉熵損失 (Cross-Entropy Loss)
3. 實作梯度下降更新
4. 在 `sklearn.datasets.load_iris` (3 類別) 上測試
5. 報告 Accuracy 並與 `sklearn.linear_model.LogisticRegression` 比較

### C2. 成本敏感分析 Cost-Sensitive Analysis (10 分)

假設你正在為銀行建立信用卡詐欺偵測系統，已知：
- False Negative（漏掉詐欺）的平均損失：$5,000
- False Positive（誤判為詐欺）的平均損失：$50（客服處理成本）

使用 Part B2 的不平衡資料集：

1. 建立一個成本函數：$\text{Total Cost} = 5000 \times FN + 50 \times FP$
2. 遍歷不同閾值（0.01 到 0.99，步長 0.01），計算每個閾值下的 Total Cost
3. 繪製 Total Cost vs. Threshold 的曲線
4. 找出使 Total Cost 最小化的最佳閾值
5. 比較此閾值與最大化 F1-Score 的閾值，說明差異及其商業意義

---

## 繳交清單 Submission Checklist

- [ ] Notebook 可以從頭到尾順利執行 (Kernel → Restart & Run All)
- [ ] 所有圖表都有標題、軸標籤和圖例 (Legend)
- [ ] 概念題的文字回答清楚完整
- [ ] 程式碼有適當的註解 (Comments)
- [ ] 分析報告有具體的數據支持
- [ ] 已匯出 PDF 版本

## 評分標準 Grading Criteria

| 項目 | 比重 | 說明 |
|------|:----:|------|
| 正確性 Correctness | 40% | 計算結果、程式碼執行無誤 |
| 完整性 Completeness | 25% | 涵蓋所有要求的任務 |
| 視覺化品質 Visualization | 15% | 圖表清晰、資訊完整、美觀 |
| 分析深度 Analysis | 15% | 對結果的解讀有深度與洞見 |
| 程式碼風格 Code Quality | 5% | 命名合理、結構清楚、有註解 |
