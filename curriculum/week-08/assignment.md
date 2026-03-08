# 第 8 週作業：特徵重要度與 SHAP 分析
# Week 8 Assignment: Feature Importance & SHAP Analysis

**繳交期限 Due:** 下週上課前
**繳交方式 Submission:** 上傳 Notebook (.ipynb) 至課程平台

---

## 作業一：特徵重要度比較（25%）

使用 scikit-learn 的**加州房價資料集 (California Housing Dataset)** 或本週 Notebook 使用的資料集：

1. 訓練一個隨機森林回歸模型 (RandomForestRegressor)
2. 取得模型內建的特徵重要度 (`feature_importances_`)
3. 計算排列重要度 (Permutation Importance)
4. 將兩種重要度並排繪製成水平長條圖 (Horizontal Bar Chart)
5. **分析：** 兩種方法的排名是否一致？若不一致，你認為原因是什麼？（至少寫 3 句）

**提示：**
```python
from sklearn.datasets import fetch_california_housing
from sklearn.inspection import permutation_importance
```

## 作業二：SHAP 蜂群圖解讀（25%）

1. 使用 TreeExplainer 計算上述模型的 SHAP 值
2. 產生蜂群圖 (Beeswarm Plot)
3. **回答以下問題（每題至少 2 句）：**
   - a. 哪個特徵的 SHAP 值範圍最廣？這代表什麼意義？
   - b. 選擇一個特徵，描述其紅色/藍色點的分布模式，判斷該特徵與預測值的關係（正相關/負相關/非線性）
   - c. 蜂群圖的特徵排名與作業一的排名相比如何？

## 作業三：力圖與局部解釋（25%）

1. 從測試集中選擇 **3 個樣本**：
   - 1 個預測值**最高**的樣本
   - 1 個預測值**最低**的樣本
   - 1 個預測值**接近平均**的樣本
2. 為每個樣本生成力圖 (Force Plot) 或瀑布圖 (Waterfall Plot)
3. **分析每個樣本（每個至少 3 句）：**
   - 哪些特徵推高了預測？哪些壓低了預測？
   - 最主要的驅動特徵是什麼？
   - 這個解釋是否符合你的直覺或領域知識？

## 作業四：模型比較與反思（25%）

1. 額外訓練一個**梯度提升模型 (GradientBoostingRegressor)**
2. 計算其 SHAP 值並生成蜂群圖
3. 與隨機森林的蜂群圖進行比較
4. 選擇一個特徵，為兩個模型各生成一張依賴圖 (Dependence Plot)
5. **撰寫比較分析（至少 150 字）：**
   - 兩個模型的特徵重要度排名是否相同？
   - 依賴圖顯示的特徵效果模式是否一致？
   - 哪個模型的 SHAP 解釋對你來說更直覺、更合理？為什麼？

---

## 繳交清單 Submission Checklist

- [ ] Notebook 包含所有程式碼並可完整執行
- [ ] 所有圖表都有清楚的標題與標籤
- [ ] 每道題目都有文字分析（寫在 Markdown Cell 中）
- [ ] 程式碼有適當的註解
- [ ] 比較分析部分至少 150 字

## 加分項目 Bonus（額外 10%）

- 使用 LIME 對同一個樣本進行解釋，並與 SHAP 的結果進行比較（5%）
- 使用 XGBoost 或 LightGBM 作為第三個比較模型（5%）

---

## 評分標準 Grading Criteria
- 程式碼正確執行 Code Execution: 40%
- 圖表品質與可讀性 Visualization Quality: 20%
- 文字分析深度 Analysis Depth: 30%
- 程式碼風格與註解 Code Quality: 10%
