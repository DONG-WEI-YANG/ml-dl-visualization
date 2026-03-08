# 第 6 週作業：SVM 與核方法視覺化實驗
# Week 6 Assignment: SVM & Kernel Methods Visualization Experiments

**繳交期限 Due:** 下週上課前
**繳交方式 Submission:** 上傳 Notebook (.ipynb) 至課程平台
**配分 Total:** 100 分

---

## 作業一：線性 SVM 基礎（15%）

### 題目
使用 `make_blobs(n_samples=200, centers=2, cluster_std=1.5, random_state=你的學號後4碼)` 產生資料集，完成以下任務：

1. 訓練一個線性 SVM (`kernel='linear', C=1`)
2. 繪製決策邊界、間隔邊界（$f(x) = \pm 1$）與支持向量
3. 在圖上標註以下資訊：
   - 間隔寬度 (Margin Width)
   - 支持向量數量
   - 法向量 $\mathbf{w}$ 和偏移量 $b$ 的值
4. 回答：間隔寬度與 $\|\mathbf{w}\|$ 之間的數學關係是什麼？

### 提示
```python
# 間隔寬度計算
w = clf.coef_[0]
margin = 2 / np.linalg.norm(w)

# 支持向量
support_vectors = clf.support_vectors_
```

---

## 作業二：C 值實驗報告（20%）

### 題目
使用作業一的資料集，系統性地實驗 C 參數的影響：

1. **視覺化比較**：使用 C = [0.001, 0.01, 0.1, 1, 10, 100] 六個值，繪製 2x3 的子圖陣列，每張圖顯示決策邊界和支持向量

2. **定量分析**：製作一個表格，記錄每個 C 值下的：
   - 間隔寬度 (Margin Width)
   - 支持向量數量 (Number of SVs)
   - 訓練準確率 (Training Accuracy)
   - 5-fold 交叉驗證準確率 (CV Accuracy)

3. **趨勢圖**：繪製 C 值（x 軸，對數尺度）vs 以上四項指標的折線圖

4. **書面分析**（至少 80 字）：解釋 C 值增大時，各項指標如何變化？為什麼？從偏差-方差權衡 (Bias-Variance Trade-off) 的角度來分析。

---

## 作業三：核函數比較實驗（25%）

### 題目
使用兩個非線性資料集進行實驗：
- 資料集 A：`make_moons(n_samples=300, noise=0.2, random_state=42)`
- 資料集 B：`make_circles(n_samples=300, noise=0.1, factor=0.4, random_state=42)`

對每個資料集：

1. **四核比較**：分別使用 Linear、Polynomial (degree=3)、RBF、Sigmoid 四種核函數訓練 SVM，繪製 2x4 的子圖（上排為資料集 A，下排為資料集 B）

2. **準確率比較**：使用 5-fold 交叉驗證計算每種核函數在兩個資料集上的準確率，以長條圖 (Bar Chart) 呈現

3. **書面分析**（至少 100 字）：
   - 哪種核函數在哪個資料集上表現最好？為什麼？
   - 線性核在這兩個資料集上的表現如何？分析原因。
   - 如果你只能選一種核函數處理未知資料，你會選哪個？為什麼？

### 注意事項
- 務必先進行特徵縮放 (Feature Scaling)
- `SVC(kernel='poly', degree=3, coef0=1)`
- `SVC(kernel='sigmoid', coef0=0)`

---

## 作業四：C-gamma 熱力圖與最佳模型（25%）

### 題目
使用 `make_moons(n_samples=400, noise=0.25, random_state=你的學號後4碼)` 產生資料集：

1. **資料分割**：訓練集 70%、測試集 30%（記得特徵縮放）

2. **網格搜索**：使用 `GridSearchCV` 對 RBF SVM 進行超參數搜索
   - C 範圍：$[10^{-2}, 10^{-1}, 10^0, 10^1, 10^2, 10^3]$
   - gamma 範圍：$[10^{-3}, 10^{-2}, 10^{-1}, 10^0, 10^1, 10^2]$
   - 使用 5-fold 交叉驗證

3. **熱力圖繪製**：
   - 繪製 C-gamma 交叉驗證準確率熱力圖（使用 `seaborn.heatmap`，附數值標注）
   - 標記最佳 (C, gamma) 的位置

4. **最佳模型分析**：
   - 輸出最佳參數、最佳 CV 準確率、測試集準確率
   - 繪製最佳模型在訓練集和測試集上的決策邊界（兩張子圖）
   - 繪製混淆矩陣 (Confusion Matrix)

5. **書面分析**（至少 80 字）：
   - 最佳的 C 和 gamma 分別是多少？為什麼這個組合最好？
   - 訓練集和測試集的準確率差異大嗎？這代表什麼？

---

## 作業五：SVM 與 Logistic Regression 比較分析（15%）

### 題目
使用 `make_moons(n_samples=500, noise=0.3, random_state=42)` 資料集：

1. **模型訓練與比較**：訓練以下三個模型（記得特徵縮放）：
   - Logistic Regression (`C=1`)
   - Linear SVM (`C=1`)
   - RBF SVM（使用 GridSearchCV 找最佳參數）

2. **效能比較表**：記錄每個模型的：
   - 5-fold CV 準確率（平均值 ± 標準差）
   - 測試集準確率
   - 訓練時間

3. **決策邊界並排比較**：繪製三張子圖，顯示三個模型的決策邊界

4. **書面討論**（至少 100 字）：
   - 在什麼情況下你會選擇 SVM 而非 Logistic Regression？
   - 在什麼情況下你會選擇 Logistic Regression 而非 SVM？
   - 考慮以下面向：準確度、訓練速度、可解釋性、機率輸出

---

## 繳交清單 Submission Checklist

- [ ] Notebook 可以從頭到尾順利執行（Restart & Run All）
- [ ] 所有圖表都有適當的標題、軸標籤
- [ ] 書面分析有足夠的深度與字數
- [ ] 使用自己的隨機種子（`random_state=你的學號後4碼`）
- [ ] 每個作業都有清楚的標題分隔

---

## 評分標準 Grading Criteria

| 項目 | 比重 | 說明 |
|------|------|------|
| 程式碼正確性 Code Correctness | 40% | 程式碼可執行、結果正確 |
| 視覺化品質 Visualization Quality | 25% | 圖表清晰、標注完整、美觀 |
| 書面分析深度 Analysis Depth | 25% | 理解概念、能解釋現象、有獨立思考 |
| 程式碼風格 Code Style | 10% | 變數命名、註解、結構清晰 |

---

## 加分題 Bonus (+10%)

選擇以下任一主題深入探索：

### 選項 A：SVR 視覺化
使用 `sklearn.svm.SVR` 對一維回歸問題進行視覺化：
- 生成非線性回歸資料（如 $y = \sin(x) + \text{noise}$）
- 比較不同核函數的擬合效果
- 視覺化 epsilon-tube

### 選項 B：手寫數字分類
使用 `sklearn.datasets.load_digits` 手寫數字資料集：
- 使用 SVM 進行多類分類
- 比較不同核函數的準確率
- 繪製混淆矩陣 (Confusion Matrix)
- 找出並視覺化被錯誤分類的樣本

### 選項 C：決策邊界動畫
使用 `matplotlib.animation` 製作動畫：
- 展示 C 值從 0.01 到 100 變化時，決策邊界的動態變化
- 或展示 gamma 從 0.01 到 100 變化時的邊界變化
