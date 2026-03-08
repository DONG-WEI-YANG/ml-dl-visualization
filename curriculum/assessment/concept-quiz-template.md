# 概念理解測驗模板
# Concept Understanding Quiz Template

> 適用對象：ML/DL 視覺化教學系統各週概念測驗
> 建議施測時間：30 分鐘
> 滿分 Total Score: 100 分

---

## 配分說明 Scoring Guide

| 題型 Question Type | 題數 Count | 每題配分 Points Each | 小計 Subtotal |
|-------------------|-----------|---------------------|--------------|
| 選擇題 Multiple Choice | 10 題 | 5 分 | 50 分 |
| 填空題 Fill-in-the-Blank | 5 題 | 6 分 | 30 分 |
| 簡答題 Short Answer | 3 題 | 依題配分 | 20 分 |
| **合計 Total** | **18 題** | — | **100 分** |

**評分原則 Grading Principles：**
- 選擇題為單選，答錯不倒扣
- 填空題需使用正確的專有名詞（接受中文或英文作答）
- 簡答題依概念完整性、邏輯清晰度評分，有部分給分

---

## 第一部分：選擇題 Part 1: Multiple Choice（共 10 題，每題 5 分）

### 第 1 題 — 監督式學習 Supervised Learning

在以下哪種情境中，最適合使用**監督式學習 (Supervised Learning)**？

- (A) 對客戶進行分群 (Clustering)，但不知道有幾種客戶類型
- (B) 根據歷史房價資料預測未來房價
- (C) 探索大量文本資料以發現潛在主題 (Topic Discovery)
- (D) 在沒有標籤 (Label) 的資料中尋找異常點 (Anomaly Detection)

**正確答案 Answer：** (B)

**解析 Explanation：** 監督式學習需要有標籤的訓練資料。房價預測屬於回歸 (Regression) 問題，使用歷史房價（標籤）來訓練模型。選項 (A)(C)(D) 皆屬於非監督式學習 (Unsupervised Learning) 情境。

---

### 第 2 題 — 過擬合 Overfitting

以下哪個現象最能說明模型發生了**過擬合 (Overfitting)**？

- (A) 訓練集準確率 (Training Accuracy) 90%，測試集準確率 (Test Accuracy) 88%
- (B) 訓練集準確率 99%，測試集準確率 60%
- (C) 訓練集準確率 55%，測試集準確率 53%
- (D) 訓練集準確率 75%，測試集準確率 76%

**正確答案 Answer：** (B)

**解析 Explanation：** 過擬合的典型特徵是訓練集表現極佳但測試集表現差，代表模型記住了訓練資料的雜訊 (Noise) 而非學到泛化 (Generalization) 的規律。選項 (C) 為欠擬合 (Underfitting)，選項 (A)(D) 為正常表現。

---

### 第 3 題 — 交叉驗證 Cross-Validation

使用 **5 折交叉驗證 (5-Fold Cross-Validation)** 時，每次迭代中有多少比例的資料用於驗證 (Validation)？

- (A) 50%
- (B) 80%
- (C) 20%
- (D) 10%

**正確答案 Answer：** (C)

**解析 Explanation：** 5 折交叉驗證將資料分成 5 等份，每次使用其中 1 份（即 1/5 = 20%）作為驗證集，其餘 4 份（80%）作為訓練集。

---

### 第 4 題 — 損失函數 Loss Function

線性回歸 (Linear Regression) 中最常使用的損失函數是：

- (A) 交叉熵 (Cross-Entropy)
- (B) 均方誤差 (Mean Squared Error, MSE)
- (C) 絞鏈損失 (Hinge Loss)
- (D) KL 散度 (KL Divergence)

**正確答案 Answer：** (B)

**解析 Explanation：** MSE 衡量預測值與實際值之間的平方差平均，是回歸問題的標準損失函數。交叉熵 (Cross-Entropy) 用於分類問題，絞鏈損失 (Hinge Loss) 用於 SVM，KL 散度主要用於機率分布比較。

---

### 第 5 題 — 梯度下降 Gradient Descent

學習率 (Learning Rate) 設定過大時，梯度下降 (Gradient Descent) 最可能出現什麼問題？

- (A) 收斂速度太慢 (Convergence too slow)
- (B) 在最小值附近震盪或發散 (Oscillation or divergence)
- (C) 陷入局部最小值 (Stuck in local minimum)
- (D) 模型參數不會更新 (Parameters not updated)

**正確答案 Answer：** (B)

**解析 Explanation：** 學習率過大會導致每步更新的幅度太大，使參數在最小值兩側來回跳動（震盪），甚至越跳越遠（發散）。學習率過小才會導致收斂太慢。

---

### 第 6 題 — 決策邊界 Decision Boundary

邏輯迴歸 (Logistic Regression) 的決策邊界 (Decision Boundary) 在二維特徵空間中是什麼形狀？

- (A) 曲線 (Curve)
- (B) 直線或超平面 (Straight line / Hyperplane)
- (C) 圓形 (Circle)
- (D) 不規則形狀 (Irregular shape)

**正確答案 Answer：** (B)

**解析 Explanation：** 標準邏輯迴歸是線性分類器 (Linear Classifier)，其決策邊界為直線（二維）或超平面（高維）。如果需要非線性決策邊界，需要使用核技巧 (Kernel Trick) 或多項式特徵 (Polynomial Features)。

---

### 第 7 題 — 特徵縮放 Feature Scaling

以下哪個演算法**最不受**特徵縮放 (Feature Scaling) 影響？

- (A) K-近鄰 (K-Nearest Neighbors, KNN)
- (B) 支持向量機 (SVM)
- (C) 隨機森林 (Random Forest)
- (D) 線性回歸搭配梯度下降 (Linear Regression with Gradient Descent)

**正確答案 Answer：** (C)

**解析 Explanation：** 決策樹 (Decision Tree) 及其集成方法（如隨機森林）基於特徵的分割閾值 (Split Threshold) 做決策，不涉及距離計算或梯度更新，因此不受特徵縮放影響。KNN 使用距離計算、SVM 依賴間隔最大化、梯度下降的收斂速度受特徵尺度影響。

---

### 第 8 題 — 混淆矩陣 Confusion Matrix

在一個二元分類問題中，若模型將一個實際為正例 (Positive) 的樣本誤判為負例 (Negative)，這稱為：

- (A) 真正例 (True Positive, TP)
- (B) 假正例 (False Positive, FP)
- (C) 真負例 (True Negative, TN)
- (D) 假負例 (False Negative, FN)

**正確答案 Answer：** (D)

**解析 Explanation：** 假負例 (FN) 指實際為正例但被模型錯誤預測為負例的樣本。在醫療診斷中，FN 代表有病但未檢出，通常是最需要避免的錯誤類型。

---

### 第 9 題 — 正則化 Regularization

在神經網路 (Neural Network) 中，**Dropout** 正則化的運作原理是：

- (A) 在損失函數中加入權重的 L2 範數 (L2 Norm)
- (B) 訓練時隨機將部分神經元的輸出設為零
- (C) 減少訓練資料的數量
- (D) 限制梯度的最大值 (Gradient Clipping)

**正確答案 Answer：** (B)

**解析 Explanation：** Dropout 在訓練過程中，以一定機率（如 0.5）隨機關閉部分神經元，迫使網路不依賴任何單一神經元，達到正則化效果以減輕過擬合。推論 (Inference) 時所有神經元都會啟用。

---

### 第 10 題 — 評估指標 Evaluation Metrics

當資料集嚴重不平衡 (Highly Imbalanced) 時（例如正例佔 1%，負例佔 99%），以下哪個指標**最不適合**作為主要評估指標？

- (A) 準確率 (Accuracy)
- (B) 精確率 (Precision)
- (C) 召回率 (Recall)
- (D) F1 Score

**正確答案 Answer：** (A)

**解析 Explanation：** 在極度不平衡的資料集中，一個永遠預測負例的「笨模型」也能達到 99% 的準確率，但它完全無法辨識正例。此時應使用精確率 (Precision)、召回率 (Recall)、F1 Score 或 AUC-PR 等指標來更準確地評估模型效能。

---

## 第二部分：填空題 Part 2: Fill-in-the-Blank（共 5 題，每題 6 分）

### 第 1 題

在偏差-變異權衡 (Bias-Variance Tradeoff) 中，模型過於簡單會導致高 ________（偏差/變異），模型過於複雜會導致高 ________（偏差/變異）。

**正確答案 Answer：** 偏差 (Bias)；變異 (Variance)

**評分標準 Scoring：** 每空 3 分，接受中文或英文作答。

---

### 第 2 題

支持向量機 (SVM) 的目標是找到一個超平面 (Hyperplane)，使得兩個類別之間的 ________ 最大化。位於間隔邊界上的資料點稱為 ________。

**正確答案 Answer：** 間隔 / 邊界 (Margin)；支持向量 (Support Vectors)

**評分標準 Scoring：** 每空 3 分。第一空接受「間隔」「邊界」「Margin」。第二空接受「支持向量」「Support Vectors」。

---

### 第 3 題

在梯度提升 (Gradient Boosting) 中，每一棵新的決策樹是針對前一輪模型的 ________ 進行擬合，而非針對原始目標值。

**正確答案 Answer：** 殘差 (Residuals) / 負梯度 (Negative Gradient)

**評分標準 Scoring：** 6 分。接受「殘差」「Residuals」「負梯度」「Negative Gradient」「梯度」。

---

### 第 4 題

卷積神經網路 (CNN) 中，________ 層負責提取局部特徵，而 ________ 層負責降低特徵圖 (Feature Map) 的空間維度。

**正確答案 Answer：** 卷積 (Convolution / Conv)；池化 (Pooling)

**評分標準 Scoring：** 每空 3 分。第一空接受「卷積」「Convolution」「Conv」。第二空接受「池化」「Pooling」。

---

### 第 5 題

SHAP (SHapley Additive exPlanations) 方法的理論基礎來自賽局理論 (Game Theory) 中的 ________ 值，用於衡量每個特徵對模型預測結果的 ________。

**正確答案 Answer：** Shapley；貢獻 / 邊際貢獻 (Contribution / Marginal Contribution)

**評分標準 Scoring：** 每空 3 分。第一空接受「Shapley」「夏普利」。第二空接受「貢獻」「邊際貢獻」「Contribution」。

---

## 第三部分：簡答題 Part 3: Short Answer（共 3 題，共 20 分）

### 第 1 題（6 分）

請解釋**過擬合 (Overfitting)** 的概念，並列舉至少**三種**可用來緩解過擬合的方法。

**參考答案 Reference Answer：**

過擬合是指模型在訓練資料上表現很好，但在未見過的新資料（測試集）上表現顯著下降。這表示模型學到了訓練資料中的雜訊和特定模式，而非真正的泛化規律。

緩解過擬合的方法（列舉三種即可）：
1. **增加訓練資料量 (More Training Data)**：讓模型接觸更多樣化的資料
2. **正則化 (Regularization)**：如 L1/L2 正則化、Dropout
3. **減少模型複雜度 (Reduce Model Complexity)**：如減少神經網路層數或節點數、限制決策樹深度
4. **交叉驗證 (Cross-Validation)**：使用 k 折交叉驗證來評估泛化能力
5. **早停 (Early Stopping)**：在驗證損失不再下降時停止訓練
6. **資料增強 (Data Augmentation)**：對現有資料進行變換以增加多樣性

**評分標準 Scoring：**
- 正確解釋過擬合概念：2 分
- 每列舉一種有效方法並簡要說明：每項 1-1.5 分（需列舉至少 3 種，最高 4 分）

---

### 第 2 題（7 分）

在二元分類問題中，**精確率 (Precision)** 和**召回率 (Recall)** 分別衡量什麼？請各給出一個實際應用場景，說明何時應優先考慮精確率，何時應優先考慮召回率。

**參考答案 Reference Answer：**

- **精確率 (Precision)**：在所有被模型預測為正例的樣本中，真正為正例的比例。公式為 `Precision = TP / (TP + FP)`。
- **召回率 (Recall)**：在所有實際為正例的樣本中，被模型正確預測為正例的比例。公式為 `Recall = TP / (TP + FN)`。

**優先考慮精確率的場景：** 垃圾郵件過濾 (Spam Filtering)。我們不希望把正常郵件誤判為垃圾郵件（降低 FP），否則使用者可能錯過重要信件。

**優先考慮召回率的場景：** 癌症篩檢 (Cancer Screening)。我們不希望漏掉任何一位患者（降低 FN），即使有些健康者被誤判為疑似患者（FP 較高），仍可透過進一步檢查排除。

**評分標準 Scoring：**
- 正確解釋精確率：2 分
- 正確解釋召回率：2 分
- 適當的精確率應用場景與說明：1.5 分
- 適當的召回率應用場景與說明：1.5 分

---

### 第 3 題（7 分）

請簡述**隨機森林 (Random Forest)** 如何透過 Bagging 策略來降低模型的變異 (Variance)，並解釋為什麼它通常比單棵決策樹 (Decision Tree) 有更好的泛化效能 (Generalization Performance)。

**參考答案 Reference Answer：**

隨機森林 (Random Forest) 透過 **Bagging (Bootstrap Aggregating)** 策略運作：

1. **自助取樣 (Bootstrap Sampling)**：從原始訓練資料中有放回地 (With Replacement) 抽取多個子樣本集
2. **獨立建樹**：在每個子樣本集上獨立訓練一棵決策樹。每次分裂節點時，從所有特徵中隨機選取一個子集 (Feature Subset) 來決定最佳分裂
3. **集成預測 (Ensemble Prediction)**：分類任務使用多數決 (Majority Voting)，回歸任務使用平均 (Averaging) 來綜合所有樹的預測結果

**降低變異的原理**：根據統計學，獨立隨機變數平均值的變異數等於單一變數變異數除以樣本數。雖然各棵樹並非完全獨立，但隨機特徵選取降低了樹之間的相關性 (Correlation)，使集成後的結果比單棵樹更穩定。

**優於單棵決策樹的原因**：單棵決策樹容易過擬合，對訓練資料的微小變動很敏感（高變異）。隨機森林透過多棵樹的平均/投票，平滑了個別樹的過擬合傾向，達到更好的泛化效能。

**評分標準 Scoring：**
- 正確描述 Bagging 流程：2 分
- 說明隨機特徵選取機制：2 分
- 解釋降低變異的統計原理：1.5 分
- 解釋為何優於單棵決策樹：1.5 分

---

## 使用說明 Usage Instructions

### 出題者指引 For Quiz Creators

1. **替換題目**：保留上方模板格式，將題目內容替換為對應週次的主題
2. **難度分配**：建議每份測驗包含 40% 基礎題、40% 中等題、20% 進階題
3. **對應週次**：每題應標註對應的週次與學習目標 (Learning Objective)
4. **題庫管理**：建議每個概念準備 3-5 題備用題，以利出不同版本的測驗
5. **時限建議**：
   - 選擇題：每題約 1-2 分鐘
   - 填空題：每題約 2-3 分鐘
   - 簡答題：每題約 5-7 分鐘

### 施測者指引 For Test Administrators

1. 測驗前確認學生已完成對應週次的學習活動
2. 在平台的作業模式 (Assignment Mode) 下進行，AI 助教不會提供直接答案
3. 測驗結束後，透過平台自動批改選擇題與填空題
4. 簡答題由助教或授課教師依評分標準人工批改
5. 建議測驗後安排概念回顧 (Concept Review) 環節

### 學生指引 For Students

1. 獨立作答，不可查閱筆記或網路（除非授課教師另行規定）
2. 填空題接受中文或英文作答
3. 簡答題請力求邏輯清晰、概念完整，可搭配圖示說明
4. 如遇到不確定的題目，先回答有把握的部分，善用部分給分機會
