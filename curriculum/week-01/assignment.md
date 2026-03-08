# 第 1 週作業：環境建置與基礎操作
# Week 1 Assignment: Environment Setup & Basic Operations

**繳交期限 Due:** 下週上課前
**繳交方式 Submission:** 上傳 Notebook (.ipynb) 至課程平台

---

## 作業一：環境驗證（20%）
執行以下程式碼並截圖，確認所有套件版本正確顯示：
```python
import numpy, pandas, matplotlib, sklearn, torch
print(f"NumPy: {numpy.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"PyTorch: {torch.__version__}")
```

## 作業二：NumPy 練習（20%）
1. 建立一個 10x10 的隨機矩陣 (Random Matrix)
2. 計算其行平均 (Row Mean) 與列平均 (Column Mean)
3. 找出矩陣中的最大值位置 (Index of Maximum)

## 作業三：Pandas 練習（20%）
使用 scikit-learn 的 Iris 資料集：
1. 載入資料並建立 DataFrame
2. 計算每個品種 (species) 的四個特徵平均值
3. 找出花瓣長度 (petal length) 最大的前 10 筆資料

## 作業四：視覺化練習（20%）
製作 Iris 資料集的以下圖表：
1. 花瓣長度 vs 花瓣寬度的散佈圖（依品種著色）
2. 各品種花萼長度的箱型圖 (Box Plot)

## 作業五：AI 助教互動（20%）
在課程平台的 AI 助教上提出 **至少 2 個問題**，並截圖記錄對話過程。
問題範例：
- 「監督式學習和非監督式學習的主要差異是什麼？」
- 「為什麼深度學習需要大量資料？」

---

## 評分標準 Grading Criteria
- 程式碼正確執行 Code Execution: 60%
- 圖表品質 Visualization Quality: 20%
- 說明與反思 Explanation & Reflection: 20%
