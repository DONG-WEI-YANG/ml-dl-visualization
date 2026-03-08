# 第 6 週投影片：SVM 與核方法視覺化

---

## Slide 1: 本週主題
# SVM 與核方法視覺化
### Support Vector Machine & Kernel Methods Visualization
- 上週回顧：邏輯迴歸、決策邊界、ROC/PR 曲線
- 本週核心：找到「最佳」的決策邊界
- 關鍵問題：**哪條線才是最好的分隔線？**

---

## Slide 2: SVM 的直覺 — 最寬的馬路
### Intuition: The Widest Road
```
邏輯迴歸：隨便一條能分開的線     SVM：讓兩邊最遠的那條線
     ●  ●                              ●  ●
   ●   / ○  ○                        ●   |   ○  ○
  ●   /   ○                         ●  ══|══  ○
   ● /  ○  ○                         ●   |   ○  ○
     ●                                    ●
                                     ← Margin →
```
- SVM 目標：**最大化間隔 (Maximum Margin)**
- 間隔 = 決策邊界到最近資料點的距離 x 2
- 間隔越大 → 泛化能力越強

---

## Slide 3: 數學表述 Mathematical Formulation
### 超平面 (Hyperplane) 與間隔 (Margin)
- 決策邊界：$\mathbf{w} \cdot \mathbf{x} + b = 0$
- 間隔寬度：$\text{Margin} = \frac{2}{\|\mathbf{w}\|}$
- 最佳化目標：

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1$$

| 符號 | 意義 |
|------|------|
| $\mathbf{w}$ | 法向量 (Normal Vector)，決定邊界方向 |
| $b$ | 偏移量 (Bias) |
| $y_i \in \{-1, +1\}$ | 類別標籤 |

---

## Slide 4: 支持向量 Support Vectors
### 決定一切的少數關鍵點
- **支持向量**：落在間隔邊界上的資料點
- 只有支持向量影響決策邊界 → **稀疏性 (Sparsity)**
- 移除非支持向量 → 模型完全不變
- 移除任一支持向量 → 邊界可能改變

> **類比：** 想像用幾根柱子撐起一面牆 —— 只有接觸牆面的柱子（支持向量）真正在「支撐」。

### Demo: 互動拖曳資料點，觀察哪些點影響邊界

---

## Slide 5: 硬間隔 vs 軟間隔
### Hard Margin vs Soft Margin

| | 硬間隔 Hard Margin | 軟間隔 Soft Margin |
|--|-------------------|-------------------|
| 容錯 | 零容忍 | 允許部分錯誤 |
| 異常值 | 極度敏感 | 可調控 |
| 資料要求 | 必須線性可分 | 不需要 |
| 實用性 | 理論概念 | **實務標準** |

**軟間隔引入鬆弛變數 $\xi_i$：**
$$\min \frac{1}{2}\|\mathbf{w}\|^2 + C\sum \xi_i$$

- $\xi_i = 0$：正確分類，在間隔外
- $0 < \xi_i < 1$：正確分類，但在間隔內
- $\xi_i > 1$：**錯誤分類**

---

## Slide 6: 參數 C 的影響
### The Effect of Regularization Parameter C

| C = 0.01 | C = 1 | C = 100 |
|----------|-------|---------|
| 寬間隔 Wide Margin | 中等間隔 | 窄間隔 Narrow Margin |
| 允許較多錯誤 | 平衡 | 幾乎不容錯 |
| 偏差高、方差低 | 最佳取捨 | 偏差低、方差高 |
| 可能欠擬合 | **通常最佳** | 可能過擬合 |

### Demo: 滑桿調整 C 值，即時觀察邊界變化

> **記憶口訣：** C 大 → 嚴格 (Cost of misclassification 高)；C 小 → 寬容

---

## Slide 7: 核技巧 — 升維的魔法
### Kernel Trick: The Magic of Higher Dimensions

**問題：** 非線性資料無法用直線分開
**解法：** 把資料「映射」到高維空間，在高維中用超平面分開

```
2D (無法線性分離)              3D (可以用平面分開！)
    ○ ○ ○                         ○  ○
  ○ ● ● ○        映射 φ          ○ /  ○
  ○ ● ● ○      ────────→       ● ● (被提高)
    ○ ○ ○                      ○     ○
                              (平面可以分開)
```

**核技巧的精妙：** 不需要真正計算高維座標！
$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$$
直接在低維空間計算「高維內積」

---

## Slide 8: 四大核函數比較
### Four Common Kernel Functions

| 核函數 | 公式 | 特性 |
|--------|------|------|
| **Linear** | $\mathbf{x}_i \cdot \mathbf{x}_j$ | 直線邊界，速度最快 |
| **Polynomial** | $(\gamma \mathbf{x}_i \cdot \mathbf{x}_j + r)^d$ | 曲線邊界，度數 $d$ 可控 |
| **RBF (高斯)** | $\exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$ | 任意曲線，最常用 |
| **Sigmoid** | $\tanh(\gamma \mathbf{x}_i \cdot \mathbf{x}_j + r)$ | 類似神經網路，少用 |

### Demo: 同一資料集，切換四種核函數觀察決策邊界
- Linear → 直線
- Polynomial (d=3) → 曲線
- RBF → 可任意彎曲
- 哪種效果最好？取決於資料！

---

## Slide 9: RBF 核的 gamma 參數
### The gamma Parameter in RBF Kernel
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

| gamma = 0.01 | gamma = 1 | gamma = 100 |
|--------------|-----------|-------------|
| 每個點影響範圍大 | 中等範圍 | 影響範圍極小 |
| 平滑邊界 | 適度彎曲 | 每個點一座「島」 |
| 可能欠擬合 | **通常最佳** | 嚴重過擬合 |

**直覺理解：**
- gamma 小 → 每個支持向量放一座「平緩的山丘」
- gamma 大 → 每個支持向量放一根「尖銳的針」

### Demo: 調整 gamma，觀察決策邊界從平滑到複雜

---

## Slide 10: C-gamma 熱力圖
### C-gamma Heatmap: Finding the Sweet Spot
- C 和 gamma **共同**決定模型行為
- 使用網格搜索 (Grid Search) + 交叉驗證 (Cross-Validation)

```
         小 gamma ←──────→ 大 gamma
大 C  │  好 / 最佳   │   過擬合    │
      │─────────────│────────────│
小 C  │  欠擬合      │   稍好      │
```

**實務策略：**
1. 以對數尺度搜索：$C, \gamma \in \{10^{-3}, 10^{-2}, ..., 10^3\}$
2. 先粗搜 (Coarse Search) → 再細搜 (Fine Search)
3. 繪製熱力圖 (Heatmap) 視覺化找到最佳區域

---

## Slide 11: SVM vs Logistic Regression
### 何時選擇哪一個？

| 情境 | 推薦模型 | 原因 |
|------|---------|------|
| 需要機率輸出 | Logistic Regression | SVM 不直接輸出機率 |
| 非線性決策邊界 | SVM (核方法) | 核技巧天然支援 |
| 高維稀疏資料 (文本) | SVM | 高維表現優秀 |
| 大量資料 ($n > 10^5$) | Logistic Regression | SVM 訓練較慢 $O(n^2)$ |
| 需要可解釋性 | Logistic Regression | 係數有明確意義 |
| 資料量少、維度高 | SVM | 間隔最大化有正則效果 |

**損失函數比較：**
- SVM: Hinge Loss → 正確且遠離邊界的點 Loss=0（稀疏）
- LR: Log Loss → 所有點都有非零 Loss（但快速遞減）

---

## Slide 12: 本週重點回顧 & 實作預告
### Week 6 Summary

**核心概念：**
1. SVM = 最大化間隔的分類器
2. 支持向量 = 決定邊界的關鍵少數點
3. 軟間隔 + 參數 C = 控制容錯程度
4. 核技巧 = 不用真正升維就能做非線性分類
5. RBF 核的 gamma = 控制決策邊界的複雜度
6. C 和 gamma 需要搭配調校

**今日實作：**
- 線性 SVM 決策邊界 & 支持向量視覺化
- C 值調整實驗（0.01 / 1 / 100）
- 四種核函數比較
- C-gamma 搜尋熱力圖
- Moon / Circle 非線性資料集挑戰

### 下週預告：樹模型與集成方法 (Decision Tree, Random Forest, GBDT)
