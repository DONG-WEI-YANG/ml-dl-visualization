# 第 6 週：SVM 與核方法視覺化
# Week 6: SVM & Kernel Methods Visualization

## 學習目標 Learning Objectives
1. 理解支持向量機 (Support Vector Machine, SVM) 的幾何直覺與數學原理
2. 了解間隔最大化 (Maximum Margin) 與支持向量 (Support Vectors) 的角色
3. 區分硬間隔 (Hard Margin) 與軟間隔 (Soft Margin) 分類器
4. 掌握正則化參數 C 對模型的影響
5. 透過視覺化理解核技巧 (Kernel Trick) 如何處理非線性問題
6. 比較不同核函數 (Linear, Polynomial, RBF, Sigmoid) 的決策邊界
7. 理解超參數 C 與 gamma 的交互影響
8. 比較 SVM 與邏輯迴歸 (Logistic Regression) 的異同

---

## 1. 支持向量機的直覺 Intuition Behind SVM

### 1.1 從分類問題出發 Starting from Classification

回顧第 5 週，我們學過邏輯迴歸 (Logistic Regression) 可以找到一條決策邊界 (Decision Boundary) 將兩類資料分開。但一個自然的問題是：**哪一條決策邊界最好？**

考慮一個二維平面上的二元分類問題，假設資料是線性可分的 (Linearly Separable)，也就是存在一條直線可以完美地將兩類資料分開。這樣的直線可能有無數條，邏輯迴歸會找到其中一條，但不一定是「最好的」。

SVM 的核心理念是：**找到那條離兩類資料都盡可能遠的決策邊界**。這就是所謂的「間隔最大化」原則。

### 1.2 幾何直覺 Geometric Intuition

想像你在兩群人之間畫一條分隔線：
- 邏輯迴歸像是隨便畫一條能分開兩群人的線
- SVM 則是要畫一條讓兩群人都離線最遠的線——就像在兩群人之間鋪一條盡可能寬的「馬路」

這條「馬路」的寬度就是**間隔 (Margin)**，SVM 的目標就是讓這條馬路越寬越好。

---

## 2. 間隔最大化 Maximum Margin

### 2.1 超平面 Hyperplane

在 $d$ 維空間中，SVM 的決策邊界是一個 **超平面 (Hyperplane)**，可以用以下方程式表示：

$$\mathbf{w} \cdot \mathbf{x} + b = 0$$

其中：
- $\mathbf{w}$ 是法向量 (Normal Vector)，決定超平面的方向
- $b$ 是偏移量 (Bias)，決定超平面與原點的距離
- $\mathbf{x}$ 是輸入特徵向量

對於二維空間 ($d=2$)，超平面就是一條直線；三維空間就是一個平面。

### 2.2 間隔的定義 Definition of Margin

SVM 定義兩條平行的邊界超平面：
- 正邊界：$\mathbf{w} \cdot \mathbf{x} + b = +1$
- 負邊界：$\mathbf{w} \cdot \mathbf{x} + b = -1$

兩條邊界之間的距離就是**間隔 (Margin)**：

$$\text{Margin} = \frac{2}{\|\mathbf{w}\|}$$

因此，最大化間隔等價於最小化 $\|\mathbf{w}\|$，也就是最小化 $\frac{1}{2}\|\mathbf{w}\|^2$（加入 $\frac{1}{2}$ 是為了數學上方便求導）。

### 2.3 最佳化問題 Optimization Problem

SVM 的最佳化目標可以表示為：

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$$

$$\text{subject to } y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1, \quad \forall i$$

其中 $y_i \in \{-1, +1\}$ 是第 $i$ 個樣本的標籤。約束條件要求所有樣本都被正確分類，且位於間隔之外。

> **為什麼用 $y_i \in \{-1, +1\}$ 而不是 $\{0, 1\}$？**
> 這樣可以將兩類的約束條件統一為同一個不等式，使數學推導更為簡潔。

---

## 3. 支持向量的角色 Role of Support Vectors

### 3.1 什麼是支持向量？ What are Support Vectors?

**支持向量 (Support Vectors)** 是指那些剛好落在間隔邊界上的資料點，滿足：

$$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) = 1$$

這些點「支撐」住了間隔的邊界——如果移動或移除任何一個支持向量，決策邊界就可能改變。

### 3.2 支持向量的重要性 Importance

- **決定性角色**：最終的決策邊界只由支持向量決定，其他資料點完全不影響結果
- **稀疏性 (Sparsity)**：通常只有少數資料點是支持向量，這使 SVM 具有計算效率優勢
- **魯棒性 (Robustness)**：遠離邊界的資料點不影響模型，因此 SVM 對局部噪音相對穩健

### 3.3 視覺化理解 Visual Understanding

在二維空間中：
```
        ●          |          ○
      ●            |            ○
    ●   [SV]●  ----+----  ○[SV]   ○
      ●            |            ○
        ●          |          ○
                   ↑
              決策邊界 (Decision Boundary)
        ←  Margin  →
```
- `[SV]` 標記的點是支持向量
- 只有支持向量會影響決策邊界的位置

---

## 4. 硬間隔 vs 軟間隔 Hard Margin vs Soft Margin

### 4.1 硬間隔 Hard Margin

**硬間隔 SVM** 要求所有資料點都被完美分類，不允許任何錯誤：

$$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1, \quad \forall i$$

**限制：**
- 資料必須是線性可分的 (Linearly Separable)
- 對異常值 (Outliers) 極度敏感——一個偏離的點可能讓整個超平面劇烈偏移
- 在實際應用中幾乎不可行

### 4.2 軟間隔 Soft Margin

**軟間隔 SVM** 引入鬆弛變數 (Slack Variables) $\xi_i$，允許部分資料點違反間隔約束：

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

$$\text{subject to } y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

鬆弛變數 $\xi_i$ 的含義：
- $\xi_i = 0$：資料點在間隔邊界外（正確分類且不在間隔內）
- $0 < \xi_i < 1$：資料點在間隔內但仍在正確的一側（正確分類但在間隔內）
- $\xi_i = 1$：資料點恰好在決策邊界上
- $\xi_i > 1$：資料點被錯誤分類

### 4.3 比較 Comparison

| 特性 | 硬間隔 Hard Margin | 軟間隔 Soft Margin |
|------|-------------------|-------------------|
| 容錯性 | 不允許錯誤 | 允許部分錯誤 |
| 對異常值敏感度 | 極高 | 可控制 |
| 資料要求 | 必須線性可分 | 不需要線性可分 |
| 實用性 | 低 | 高（實務中都用這個）|
| 調控參數 | 無 | 有參數 C |

---

## 5. 正則化參數 C 的影響 Effect of Regularization Parameter C

### 5.1 C 的含義 Meaning of C

參數 $C$ 控制了**間隔寬度**與**分類錯誤**之間的取捨 (Trade-off)：

- **大 C（例如 C=100）**：嚴格要求正確分類
  - 間隔較窄
  - 訓練集準確度高
  - 容易過擬合 (Overfitting)
  - 模型複雜度高

- **小 C（例如 C=0.01）**：容忍更多分類錯誤
  - 間隔較寬
  - 訓練集準確度可能較低
  - 泛化能力 (Generalization) 通常較好
  - 模型複雜度低

### 5.2 偏差-方差權衡 Bias-Variance Trade-off

C 值直接影響偏差-方差權衡 (Bias-Variance Trade-off)：

| C 值 | 偏差 (Bias) | 方差 (Variance) | 模型表現 |
|------|------------|----------------|---------|
| 很小 | 高 | 低 | 欠擬合 (Underfitting) |
| 適中 | 適中 | 適中 | 最佳泛化 |
| 很大 | 低 | 高 | 過擬合 (Overfitting) |

### 5.3 視覺化觀察

```
C = 0.01 (寬間隔)        C = 1 (中等)          C = 100 (窄間隔)
●    ●                   ●    ●                  ●    ●
  ●   |   ○               ● |  ○                  ●| ○
●  ---|---  ○ ○          ●  -|-  ○ ○             ● -|  ○ ○
  ●   |   ○               ● |  ○                  ●| ○
●    ●                   ●    ●                  ●    ●
← 寬 Margin →            ← 中 →                 ←窄→
允許一些誤分類             平衡                    幾乎不允許誤分類
```

---

## 6. 線性不可分與核技巧 Non-linear Separability & Kernel Trick

### 6.1 線性不可分問題 Non-linearly Separable Data

很多現實世界的資料無法用一條直線（超平面）分開。例如：
- **環形資料 (Circular Data)**：一類在圓心附近，另一類在外圈
- **月亮形資料 (Moon-shaped Data)**：兩類呈交錯的月牙形

面對這類資料，線性 SVM 無能為力。

### 6.2 核技巧的思路 The Kernel Trick Idea

核技巧的核心思想是：**將資料映射到更高維的空間，使其在高維中變得線性可分**。

#### 例子：二維到三維

考慮二維平面上的環形資料（圓內為一類、圓外為另一類）。

定義映射 $\phi$：
$$\phi(x_1, x_2) = (x_1^2, x_2^2, \sqrt{2} x_1 x_2)$$

在這個三維空間中，原本的環形資料可以被一個平面分開！

### 6.3 為什麼叫「技巧」？ Why is it a "Trick"?

如果我們真的把資料映射到高維空間，計算量會非常龐大。核技巧的精妙之處在於：**我們不需要真正計算高維空間中的座標，只需要計算兩個資料點在高維空間中的內積 (Inner Product)**。

核函數 (Kernel Function) $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$

這個計算可以直接在原始低維空間中完成，大幅降低計算複雜度。這就是所謂的 **Kernel Trick**。

### 6.4 數學補充：對偶問題 Dual Problem

SVM 的最佳化問題可以轉換為對偶形式 (Dual Form)，其中只涉及資料點之間的內積 $\mathbf{x}_i \cdot \mathbf{x}_j$。我們可以將所有內積替換為核函數 $K(\mathbf{x}_i, \mathbf{x}_j)$，就等同於在高維空間中運算，而不需要顯式計算映射。

---

## 7. 常見核函數 Common Kernel Functions

### 7.1 線性核 Linear Kernel

$$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j$$

- 等同於不做任何非線性映射
- 適用於線性可分或高維稀疏資料（如文本分類）
- 計算最快，參數最少

### 7.2 多項式核 Polynomial Kernel

$$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \, \mathbf{x}_i \cdot \mathbf{x}_j + r)^d$$

- $d$ 為多項式次數 (Degree)
- $\gamma$ 為係數，$r$ 為常數項（coef0）
- $d=2$ 時為二次核，可捕捉特徵間的交互作用 (Feature Interactions)
- 隨 $d$ 增大，決策邊界越複雜

### 7.3 RBF 核（高斯核）Radial Basis Function Kernel

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

- 最常用的核函數，也稱為高斯核 (Gaussian Kernel)
- 隱式映射到無限維空間
- $\gamma$ 控制影響範圍：
  - 大 $\gamma$：每個點影響範圍小，決策邊界複雜（容易過擬合）
  - 小 $\gamma$：每個點影響範圍大，決策邊界平滑（可能欠擬合）
- 當不確定用哪個核時，RBF 通常是個好的起點

### 7.4 Sigmoid 核 Sigmoid Kernel

$$K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \, \mathbf{x}_i \cdot \mathbf{x}_j + r)$$

- 類似神經網路的啟動函數
- 不滿足 Mercer 定理 (Mercer's Theorem)，並非所有參數組合都是有效的核函數
- 實務中較少使用

### 7.5 核函數比較 Kernel Comparison

| 核函數 | 映射維度 | 超參數 | 適用場景 | 決策邊界複雜度 |
|--------|---------|--------|---------|--------------|
| Linear | 原始維度 | 無 | 高維稀疏資料 | 低（直線/平面）|
| Polynomial | 有限維 | $d, \gamma, r$ | 特徵交互重要 | 中 |
| RBF | 無限維 | $\gamma$ | 通用（預設選擇）| 高（可任意彎曲）|
| Sigmoid | — | $\gamma, r$ | 少用 | 中 |

---

## 8. 核函數的幾何直覺 Geometric Intuition of Kernels

### 8.1 低維到高維映射 Low-to-High Dimensional Mapping

以最經典的 XOR 問題為例：

**原始二維空間（無法線性分離）：**
```
      x2
       |
  ○    |    ●
       |
-------+-------→ x1
       |
  ●    |    ○
       |
```

**映射到三維空間後（可以線性分離）：**

加入第三個維度 $x_3 = x_1 \cdot x_2$，四個點被「提起」到不同高度，此時一個平面就可以將兩類分開。

### 8.2 RBF 核的直覺 RBF Kernel Intuition

RBF 核可以理解為：**每個支持向量在特徵空間中放置一個「山丘」（高斯鐘形曲線），所有山丘的加總構成最終的決策函數**。

$$f(\mathbf{x}) = \sum_{i \in SV} \alpha_i y_i \exp(-\gamma \|\mathbf{x} - \mathbf{x}_i\|^2) + b$$

- 靠近正類支持向量的區域，函數值為正
- 靠近負類支持向量的區域，函數值為負
- $\gamma$ 控制每個「山丘」的陡峭程度

---

## 9. 超參數：C 和 gamma 的交互影響 Hyperparameters: C and gamma Interaction

### 9.1 雙參數搜索 Two-Parameter Search

C 和 gamma 共同決定了 RBF-SVM 的行為，兩者的交互作用非常重要：

| | 小 gamma | 大 gamma |
|--|---------|---------|
| **小 C** | 非常平滑的邊界（欠擬合）| 稍微彎曲（仍較簡單）|
| **大 C** | 平滑但精確的邊界（可能最佳）| 極度複雜的邊界（過擬合）|

### 9.2 網格搜索 Grid Search

實務中通常使用**網格搜索 (Grid Search)** 配合**交叉驗證 (Cross-Validation)** 來找最佳的 C 和 gamma：

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10]
}

grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"最佳參數: {grid_search.best_params_}")
```

### 9.3 搜索策略建議 Search Strategy Tips

1. **先用對數尺度 (Log Scale)**：C 和 gamma 都應以指數方式搜索（如 $10^{-3}$ 到 $10^3$）
2. **先粗搜再細搜 (Coarse-to-Fine)**：先用大範圍找到大致區域，再縮小範圍
3. **使用交叉驗證**：避免在驗證集上過擬合
4. **考慮 RandomizedSearchCV**：當參數空間很大時，隨機搜索比網格搜索更有效率

---

## 10. SVM vs Logistic Regression 比較

### 10.1 核心差異 Key Differences

| 比較項目 | SVM | Logistic Regression |
|---------|-----|-------------------|
| 決策邊界原則 | 間隔最大化 | 最大似然估計 (MLE) |
| 損失函數 | Hinge Loss | Log Loss (Cross-Entropy) |
| 輸出 | 類別標籤（需額外校準機率）| 機率值 |
| 核技巧 | 天然支援 | 需手動特徵工程 |
| 對異常值 | 相對穩健（只看支持向量）| 所有點都影響 |
| 正則化 | 參數 C | 參數 C 或 $\lambda$ |
| 高維表現 | 優秀 | 良好 |

### 10.2 損失函數比較 Loss Function Comparison

- **Hinge Loss**（SVM）：$\max(0, 1 - y_i \cdot f(\mathbf{x}_i))$
  - 對正確分類且離邊界夠遠的點，損失為 0（稀疏性來源）
- **Log Loss**（Logistic Regression）：$\log(1 + \exp(-y_i \cdot f(\mathbf{x}_i)))$
  - 對所有點都有非零損失，但正確分類的點損失趨近 0

### 10.3 選擇指南 When to Use Which

- **使用 SVM 當**：
  - 需要非線性決策邊界（核技巧）
  - 資料維度高、樣本少（如文本分類）
  - 不需要機率輸出
- **使用 Logistic Regression 當**：
  - 需要機率輸出
  - 需要可解釋性 (Interpretability)
  - 資料量大（LR 訓練更快）
  - 線性模型已足夠

---

## 11. SVM 用於回歸：SVR Support Vector Regression

### 11.1 基本思想 Basic Idea

SVM 不只能做分類，也能做回歸——稱為 **支持向量回歸 (Support Vector Regression, SVR)**。

SVR 的目標是找到一個函數 $f(\mathbf{x})$，使得所有資料點的預測值與真實值的差距不超過 $\epsilon$（容忍帶寬度）：

$$|y_i - f(\mathbf{x}_i)| \leq \epsilon$$

### 11.2 Epsilon 管道 Epsilon-tube

SVR 在預測函數周圍建立一個寬度為 $2\epsilon$ 的「管道」(Epsilon-tube)：
- 管道內的資料點不計算損失
- 管道外的資料點才會有懲罰
- 管道邊界上的點就是支持向量

```
      y
      |       ●
      |    ● /   ● (支持向量)
      |   __/_________ ← 上界 f(x) + ε
      |  / ●
      | /_____________ ← f(x)
      |/  ●
      /_______________ ← 下界 f(x) - ε
      |  ●  (支持向量)
      +------------------→ x
```

### 11.3 scikit-learn 中的 SVR

```python
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X_train, y_train)
y_pred = svr_rbf.predict(X_test)
```

---

## 12. 實務建議 Practical Tips

### 12.1 資料前處理 Data Preprocessing

**特徵縮放 (Feature Scaling) 非常重要！**

SVM 對特徵尺度非常敏感。使用 SVM 前，**必須**對特徵進行標準化 (Standardization) 或正規化 (Normalization)：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 注意：用 transform 而非 fit_transform
```

### 12.2 大規模資料的考量 Scalability Considerations

| 資料規模 | 建議方法 |
|---------|---------|
| $n < 10,000$ | 標準 SVM（`SVC`）|
| $10,000 < n < 100,000$ | LinearSVC 或 SGDClassifier |
| $n > 100,000$ | SGDClassifier（線性核）或考慮其他模型 |

SVM 的訓練時間複雜度約為 $O(n^2)$ 到 $O(n^3)$，大規模資料集上速度較慢。

### 12.3 多類分類 Multi-class Classification

SVM 本質上是二元分類器，但可以擴展到多類問題：
- **One-vs-One (OvO)**：每兩類訓練一個分類器，共 $\binom{K}{2}$ 個（scikit-learn `SVC` 預設策略）
- **One-vs-Rest (OvR)**：每類對其餘所有類訓練一個分類器，共 $K$ 個（scikit-learn `LinearSVC` 預設策略）

---

## 13. 關鍵詞彙表 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 支持向量機 | Support Vector Machine, SVM | 基於間隔最大化的分類/回歸演算法 |
| 超平面 | Hyperplane | 高維空間中的決策邊界 |
| 間隔 | Margin | 決策邊界到最近資料點的距離 |
| 支持向量 | Support Vectors | 落在間隔邊界上、決定超平面的資料點 |
| 硬間隔 | Hard Margin | 不允許任何分類錯誤的 SVM |
| 軟間隔 | Soft Margin | 允許部分錯誤以提高泛化能力的 SVM |
| 鬆弛變數 | Slack Variables ($\xi$) | 衡量資料點違反間隔約束的程度 |
| 正則化參數 C | Regularization Parameter C | 控制間隔寬度與分類錯誤的取捨 |
| 核技巧 | Kernel Trick | 在不顯式映射的情況下計算高維內積 |
| 核函數 | Kernel Function | 計算兩點在高維空間內積的函數 |
| 線性核 | Linear Kernel | 不做非線性映射的核函數 |
| 多項式核 | Polynomial Kernel | 映射到有限維特徵空間 |
| RBF 核 / 高斯核 | Radial Basis Function Kernel | 映射到無限維空間，最常用 |
| gamma ($\gamma$) | Gamma | RBF 核的參數，控制影響範圍 |
| 網格搜索 | Grid Search | 系統性搜索最佳超參數組合 |
| 交叉驗證 | Cross-Validation | 將資料切分為多個子集以評估泛化能力 |
| Hinge Loss | 鉸鏈損失 | SVM 的損失函數 |
| 對偶問題 | Dual Problem | SVM 最佳化的等價形式 |
| 支持向量回歸 | Support Vector Regression, SVR | SVM 的回歸版本 |
| 特徵縮放 | Feature Scaling | 將特徵調整到相同尺度 |
| One-vs-One | 一對一 | 多類分類策略，兩兩配對 |
| One-vs-Rest | 一對多 | 多類分類策略，一類對其餘 |

---

## 參考資源 References

1. Cortes, C. & Vapnik, V. (1995). "Support-vector networks." *Machine Learning*, 20(3), 273–297.
2. Burges, C. J. C. (1998). "A Tutorial on Support Vector Machines for Pattern Recognition." *Data Mining and Knowledge Discovery*, 2(2), 121–167.
3. scikit-learn SVM 文件：https://scikit-learn.org/stable/modules/svm.html
4. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 7.

---

## 下週預告 Next Week Preview

第 7 週我們將學習**樹模型與集成方法 (Tree Models & Ensemble Methods)**，包含決策樹 (Decision Tree)、隨機森林 (Random Forest) 與梯度提升 (Gradient Boosting, GBDT) 的視覺化理解。
