# 第 3 週作業：監督式學習概念、資料分割與交叉驗證
# Week 3 Assignment: Supervised Learning, Data Splitting & Cross-Validation

**繳交期限 Due Date：** 下週上課前
**繳交格式 Format：** Jupyter Notebook (.ipynb) + PDF 匯出
**評量佔比 Weight：** 每週作業 30% 的一部分

---

## 第一部分：概念理解 Conceptual Understanding（20 分）

### 題目 1（5 分）

請用自己的話解釋以下問題，每題回答 3-5 句：

**(a)** 為什麼不能用訓練資料來評估模型的效能？請用一個日常生活的類比來說明。

**(b)** 訓練集 (Training Set)、驗證集 (Validation Set) 和測試集 (Test Set) 各自的角色是什麼？它們之間最關鍵的差異是什麼？

**(c)** 什麼是「資料洩漏 (Data Leakage)」？請舉一個具體的例子說明它如何發生。

### 題目 2（5 分）

請判斷以下場景分別屬於**過擬合 (Overfitting)**、**欠擬合 (Underfitting)** 還是**良好擬合 (Good Fit)**，並說明判斷依據：

| 場景 | 訓練準確率 | 測試準確率 | 你的判斷 | 理由 |
|------|-----------|-----------|---------|------|
| (a) | 99.5% | 62.3% | | |
| (b) | 55.0% | 53.8% | | |
| (c) | 93.2% | 91.5% | | |
| (d) | 100.0% | 50.0% | | |
| (e) | 78.0% | 76.5% | | |

### 題目 3（5 分）

關於偏差-變異權衡 (Bias-Variance Tradeoff)：

**(a)** 用射擊靶心的類比解釋「高偏差 + 低變異」和「低偏差 + 高變異」分別代表什麼。

**(b)** 假設你有一個 3 次多項式模型和一個 15 次多項式模型來擬合同一組資料。哪個模型的偏差可能更高？哪個的變異可能更高？為什麼？

**(c)** 為什麼我們不能同時將偏差和變異都降到零？「不可約誤差 (Irreducible Noise)」在其中扮演什麼角色？

### 題目 4（5 分）

請回答以下關於交叉驗證的問題：

**(a)** 5-Fold 交叉驗證中，每個樣本被當作驗證資料幾次？被當作訓練資料幾次？

**(b)** 在一個有 95% 負樣本和 5% 正樣本的二元分類問題中，為什麼使用 `StratifiedKFold` 比使用普通 `KFold` 更好？

**(c)** 為什麼時間序列資料不能使用一般的 k-Fold 交叉驗證？`TimeSeriesSplit` 如何解決這個問題？

**(d)** LOOCV（Leave-One-Out Cross-Validation）的優點和缺點各是什麼？在什麼情況下適合使用？

---

## 第二部分：`train_test_split` 應用（20 分）

### 題目 5（10 分）

使用 scikit-learn 的 **Wine 資料集** (`load_wine`)，完成以下任務：

1. 載入資料集並簡要描述其特性（樣本數、特徵數、類別數及各類別樣本數）（2 分）
2. 使用 `train_test_split` 將資料分為 70% 訓練集和 30% 測試集，使用分層抽樣 (stratify)（2 分）
3. 驗證分割後各集合中三個類別的比例是否與原始資料一致（2 分）
4. 分別用**有 stratify** 和**沒有 stratify** 做 10 次不同 random_state 的分割，統計每次分割中各類別比例的標準差，比較兩種方法的穩定性（4 分）

**提示：**
```python
from sklearn.datasets import load_wine
wine = load_wine()
X_wine, y_wine = wine.data, wine.target
```

### 題目 6（10 分）

**資料洩漏偵錯練習：** 以下程式碼存在資料洩漏問題，請：

1. 找出所有存在洩漏的地方（4 分）
2. 解釋每個洩漏為什麼會導致效能被高估（3 分）
3. 寫出修正後的正確程式碼（3 分）

```python
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 載入資料
X, y = load_some_data()

# 步驟 1：標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 步驟 2：特徵選擇
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_scaled, y)

# 步驟 3：分割
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# 步驟 4：訓練
model = LogisticRegression()
model.fit(X_train, y_train)

# 步驟 5：評估
print(f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test))}")
```

---

## 第三部分：k-Fold 交叉驗證實作（20 分）

### 題目 7（10 分）

使用 Wine 資料集和 `LogisticRegression`，完成以下實驗：

1. 使用 `KFold` 進行 k = 3, 5, 7, 10 的交叉驗證（3 分）
2. 使用 `StratifiedKFold` 進行同樣的 k 值交叉驗證（3 分）
3. 繪製 boxplot 比較所有 8 組結果（KFold vs StratifiedKFold, 各 4 個 k 值）（2 分）
4. 分析並討論：k 值的選擇和是否使用分層抽樣對結果有什麼影響？（2 分）

### 題目 8（10 分）

**手動實作簡易 k-Fold CV**（不使用 `cross_val_score`）：

1. 寫一個函數 `manual_kfold_cv(model, X, y, k=5)`，手動實作 k-Fold 交叉驗證（6 分）
   - 將資料分為 k 份
   - 輪流用每份作為驗證集
   - 回傳每個 Fold 的準確率列表
2. 用 Wine 資料集測試你的函數，並與 `cross_val_score` 的結果比較，驗證一致性（2 分）
3. 繪製每個 Fold 的訓練準確率和驗證準確率的對比圖（2 分）

**函數骨架：**
```python
def manual_kfold_cv(model, X, y, k=5, random_state=42):
    """
    手動實作 k-Fold Cross-Validation

    Parameters:
    -----------
    model : sklearn estimator
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    k : int, number of folds
    random_state : int, random seed

    Returns:
    --------
    train_scores : list of float
    val_scores : list of float
    """
    # 你的程式碼
    pass
```

---

## 第四部分：偏差-變異權衡實驗（20 分）

### 題目 9（10 分）

使用以下真實函數生成資料：

$$y = 0.5 \cdot x^3 - x^2 + 0.5 \cdot x + 2 + \epsilon, \quad \epsilon \sim N(0, 0.5)$$

1. 生成 50 個訓練樣本和 200 個測試樣本（x 在 [-2, 2] 之間均勻分布）（2 分）
2. 用 degree = 1, 2, 3, 5, 8, 12 的多項式回歸分別擬合（2 分）
3. 繪製 2x3 的子圖，每個子圖顯示：資料點、真實函數、模型擬合曲線，並標註訓練 MSE 和測試 MSE（3 分）
4. 繪製訓練 MSE 和測試 MSE 隨 degree 變化的曲線，找出最佳 degree（3 分）

### 題目 10（10 分）

**Bootstrap 偏差-變異分解實驗：**

1. 使用題目 9 的真實函數，進行 50 次 bootstrap 取樣（每次取 50 個訓練樣本）（3 分）
2. 對 degree = 1, 3, 5, 10 各做 50 次擬合，在一系列固定測試點上記錄每次的預測值（3 分）
3. 計算每個測試點的：
   - 偏差平方 Bias^2 = (平均預測 - 真實值)^2
   - 變異 Variance = 預測值的方差
   - 總誤差 = Bias^2 + Variance（2 分）
4. 繪製 Bias^2、Variance、Total Error 隨 degree 的變化圖，驗證偏差-變異權衡的理論（2 分）

---

## 第五部分：綜合挑戰 -- 完整交叉驗證 Pipeline（20 分）

### 題目 11（20 分）

**建構一個完整且正確的模型評估流程：**

使用 scikit-learn 的 **Breast Cancer 資料集** (`load_breast_cancer`)，完成以下任務：

1. **資料探索**（3 分）
   - 載入資料集，描述基本特性
   - 檢查類別是否平衡
   - 簡單視覺化特徵分布

2. **正確的資料分割**（3 分）
   - 先分出 20% 作為最終測試集（使用分層抽樣）
   - 剩餘 80% 用於交叉驗證

3. **模型選擇**（5 分）
   - 使用 5-Fold Stratified CV 比較至少 3 種不同模型：
     - LogisticRegression
     - DecisionTreeClassifier（嘗試不同 max_depth）
     - 至少一種你自選的模型
   - 報告每個模型的平均 CV 分數和標準差
   - 繪製比較圖表

4. **超參數選擇**（4 分）
   - 對表現最好的模型，使用 CV 比較不同超參數設定
   - 選出最佳超參數組合

5. **最終評估**（5 分）
   - 用最佳模型和超參數在全部訓練集上重新訓練
   - 在測試集上評估一次，報告最終準確率
   - 確保整個過程中**沒有資料洩漏**
   - 寫一段 200 字以內的總結，說明你的方法、結果和發現

**重要提醒：**
- 資料前處理（如 StandardScaler）必須在 CV 內部進行
- 可以使用 `sklearn.pipeline.Pipeline` 來確保流程正確
- 測試集在整個過程中只使用一次

---

## 提交要求 Submission Requirements

1. **Jupyter Notebook** (.ipynb)：包含完整的程式碼、輸出、圖表和文字說明
2. **PDF 匯出**：從 Notebook 匯出 PDF 格式
3. 程式碼需有**適當的註解**（中文或英文皆可）
4. 每道題的**回答/討論**以 Markdown Cell 撰寫
5. 所有圖表需有**標題、軸標籤、圖例**
6. 設定 `random_state=42`（或其他固定值）確保**可重現性 (Reproducibility)**

## 評分標準 Grading Criteria

| 部分 | 分數 | 重點 |
|------|------|------|
| 第一部分：概念理解 | 20 | 理解正確性、解釋清晰度 |
| 第二部分：train_test_split | 20 | 實作正確性、洩漏偵錯能力 |
| 第三部分：交叉驗證 | 20 | CV 實作、手動實作能力 |
| 第四部分：偏差-變異權衡 | 20 | 實驗設計、視覺化品質 |
| 第五部分：綜合挑戰 | 20 | 完整性、正確性、無洩漏 |
| **總計** | **100** | |

**加分項目（額外 5 分）：**
- 使用 Plotly 製作互動式圖表
- 額外探索 GroupKFold 或 RepeatedKFold 的應用場景
- 自行找一個真實資料集（如 Kaggle）套用完整 CV Pipeline
