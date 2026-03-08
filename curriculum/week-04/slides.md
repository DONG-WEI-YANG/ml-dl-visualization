# 第 4 週投影片：線性回歸 — 損失函數、梯度下降視覺化
# Week 4 Slides: Linear Regression — Loss Function & Gradient Descent Visualization

---

## Slide 1: 本週主題
# 線性回歸：損失函數與梯度下降視覺化
### Linear Regression: Loss Function & Gradient Descent Visualization

- 本週是課程最重要的視覺化主題之一
- 掌握梯度下降 = 掌握所有 ML/DL 模型的訓練核心
- 學習目標：
  1. 理解線性回歸的數學原理
  2. 推導 MSE 損失函數的梯度
  3. 透過視覺化建立梯度下降的直覺
  4. 實驗學習率對收斂的影響

---

## Slide 2: 回歸問題的直覺
### 什麼是回歸？ What is Regression?

**目標：** 給定輸入 $x$，預測連續值 $y$

| 輸入 $x$ | 目標 $y$ | 例子 |
|:-:|:-:|:-:|
| 房屋坪數 | 房價 | 30坪 → 1200萬 |
| 學習時間 | 考試成績 | 5小時 → 85分 |
| 廣告預算 | 銷售額 | 10萬 → 50萬 |

**核心思想：** 找到一條「最佳擬合線」穿過資料點

$$\hat{y} = wx + b$$

> [視覺化] 散佈圖上顯示資料點與擬合線，拖動斜率與截距觀察變化

---

## Slide 3: 簡單 vs 多元線性回歸
### Simple vs Multiple Linear Regression

| | 簡單 Simple | 多元 Multiple |
|:-:|:-:|:-:|
| 特徵數 | 1 | $d \geq 2$ |
| 模型 | $\hat{y} = wx + b$ | $\hat{y} = w_1x_1 + w_2x_2 + \cdots + b$ |
| 幾何 | 擬合直線 | 擬合超平面 |
| 參數 | 2 個 | $d + 1$ 個 |

**矩陣形式 (向量化)：**

$$\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\theta}$$

> 多元的情況更常見，但今天先從簡單線性回歸建立直覺

---

## Slide 4: 損失函數 — 為什麼需要它？
### Loss Function — Why Do We Need It?

> 如果沒有損失函數，我們無法量化「這條線好不好」

**損失函數 = 衡量預測誤差的數學工具**

三大常用損失函數：

| 損失函數 | 公式 | 特性 |
|:-:|:-:|:-:|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 最常用、放大大誤差 |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | 對離群值穩健 |
| **Huber** | 混合 MSE+MAE | 兩者兼顧 |

> [視覺化] 三種損失函數的曲線形狀比較圖

---

## Slide 5: MSE 的幾何意義
### Geometric Meaning of Mean Squared Error

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**幾何直覺：**
- 每個殘差 $e_i = y_i - \hat{y}_i$ 是資料點到回歸線的垂直距離
- 以殘差為邊長畫出正方形
- MSE = **所有正方形面積的平均**

**為什麼用平方？**
1. 消除正負抵消
2. 處處可微分 → 可以用梯度下降
3. 放大大誤差 → 優先修正大偏差
4. **凸函數** → 保證唯一全域最小值

> [視覺化] 散佈圖上顯示殘差的正方形面積

---

## Slide 6: 梯度下降 — 核心直覺
### Gradient Descent — Core Intuition

> 想像你矇著眼睛站在山上，要走到谷底

**三步循環：**
1. **感受腳下傾斜方向** → 計算梯度 $\nabla \mathcal{L}$
2. **往最陡的下坡方向走一步** → 沿負梯度更新參數
3. **重複** → 直到到達谷底（收斂）

**更新公式：**

$$w \leftarrow w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w}$$

$$b \leftarrow b - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b}$$

- $\alpha$：學習率 (Learning Rate)，控制步幅大小
- $\nabla \mathcal{L}$：梯度，指向損失增加最快的方向
- $-\nabla \mathcal{L}$：負梯度，損失**下降**最快的方向

> [動畫] 一個球在碗狀曲面上滾動到谷底的過程

---

## Slide 7: MSE 梯度推導
### Deriving the MSE Gradient

$$\mathcal{L}(w, b) = \frac{1}{n}\sum_{i=1}^{n}(y_i - wx_i - b)^2$$

**對 $w$ 求偏導：**

$$\frac{\partial \mathcal{L}}{\partial w} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i) \cdot x_i$$

**對 $b$ 求偏導：**

$$\frac{\partial \mathcal{L}}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)$$

**推導使用連鎖律 (Chain Rule)：**
- 外層微分：$2(y_i - wx_i - b)$
- 內層微分（對 $w$）：$-x_i$
- 乘起來後取平均

---

## Slide 8: 學習率的影響（關鍵！）
### Impact of Learning Rate (Critical!)

> 學習率是梯度下降中**最重要的超參數**

| 學習率 | 行為 | 損失曲線 |
|:-:|:-:|:-:|
| **太大** ($\alpha = 1.0$) | 跳過谷底、發散 | 震盪、爆炸 |
| **太小** ($\alpha = 0.001$) | 步幅極小 | 緩慢下降 |
| **適當** ($\alpha = 0.01$) | 穩定收斂 | 平滑衰減 |

> [動畫] **重點動畫** — 同一等高線圖上，四種學習率 (0.001, 0.01, 0.1, 1.0) 的梯度下降軌跡同時顯示

**Tips：**
- 先用 $\{0.001, 0.01, 0.1\}$ 試驗
- 觀察損失曲線是否穩定下降
- 進階方法：自適應學習率 (Adam, RMSProp)

---

## Slide 9: 損失地形 3D 視覺化
### Loss Landscape 3D Visualization

**座標軸：**
- x 軸 → 權重 $w$
- y 軸 → 偏差 $b$
- z 軸 → 損失值 $\mathcal{L}(w, b)$

**MSE 的損失地形 = 碗狀曲面 (Bowl-shaped)**
- 唯一的全域最小值（凸函數保證）
- 梯度下降無論從哪出發，都會收斂到碗底

**等高線圖 = 損失地形的俯視圖**
- 等高線密集 → 梯度大（陡峭）
- 等高線稀疏 → 梯度小（平坦）
- 梯度方向 ⊥ 等高線

> [互動] 3D 旋轉碗狀曲面 + 等高線圖 + 梯度下降路徑動畫

---

## Slide 10: 梯度下降的三兄弟
### Three Variants of Gradient Descent

| | BGD (全批次) | SGD (隨機) | Mini-batch |
|:-:|:-:|:-:|:-:|
| 每步使用 | 全部 $n$ 個樣本 | 1 個樣本 | $m$ 個樣本 |
| 梯度品質 | 精準 | 噪聲大 | 中等 |
| 速度 | 慢 | 快 | 中等 |
| 收斂路徑 | 平滑 | 震盪 | 相對平滑 |
| GPU 利用 | 低效 | 低效 | 高效 |

> [視覺化] 三種方法在同一等高線圖上的軌跡對比

**實務上幾乎都用 Mini-batch GD**（batch size = 32, 64, 128, 256）

---

## Slide 11: 正規方程 vs 梯度下降
### Normal Equation vs Gradient Descent

**正規方程（封閉解）：**
$$\boldsymbol{\theta}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
→ 一步算出最佳解，不需迭代

| | 正規方程 | 梯度下降 |
|:-:|:-:|:-:|
| 迭代？ | 否 | 是 |
| 學習率？ | 不需要 | 需要調整 |
| $d > 10{,}000$ | 非常慢 ($O(d^3)$) | 仍可行 |
| 非線性模型 | 不適用 | 適用 |

> **結論：** 線性回歸可用正規方程；深度學習只能用梯度下降

---

## Slide 12: 殘差分析與評估指標
### Residual Analysis & Evaluation Metrics

**殘差圖解讀：**
| 模式 | 意涵 | 行動 |
|:-:|:-:|:-:|
| 隨機分散 | 模型假設成立 | 很好！ |
| 曲線形 | 非線性關係 | 加多項式項 |
| 喇叭形 | 異質變異 | 對 $y$ 取 log |

**回歸評估指標：**
| 指標 | 意義 |
|:-:|:-:|
| **RMSE** | 與目標同單位的平均誤差 |
| **$R^2$** | 模型解釋了多少 % 的變異 |
| **Adj. $R^2$** | 考慮特徵數量的修正版 |

> [視覺化] 殘差圖 + 指標儀表板

---

## Slide 13: 多項式回歸與過擬合
### Polynomial Regression & Overfitting

$$\hat{y} = w_0 + w_1 x + w_2 x^2 + \cdots + w_p x^p$$

> [視覺化] 同一組資料，三條曲線對比

| Degree = 1 | Degree = 3 | Degree = 10 |
|:-:|:-:|:-:|
| 欠擬合 Underfitting | 適當擬合 Good Fit | 過擬合 Overfitting |
| 太簡單，遺漏趨勢 | 平滑追蹤主要趨勢 | 穿過每個點，劇烈扭曲 |
| 訓練/測試損失都高 | 兩者都低 | 訓練低、測試高 |

**應對策略：**
- 交叉驗證 (Cross-Validation) 選擇最佳 degree
- 正則化 (Regularization)：Ridge / Lasso（下週預告）
- 增加訓練資料

---

## Slide 14: 本週實作與作業
### This Week's Lab & Assignment

**Notebook 實作（40 分鐘）：**
1. 從零實作梯度下降（不用 sklearn）
2. 損失地形 3D 視覺化
3. 四種學習率的比較實驗
4. 等高線圖上的梯度下降軌跡動畫
5. sklearn 對比驗證
6. 多項式回歸過擬合示範

**作業重點：**
1. 手動實作梯度下降（自己寫 Python）
2. 學習率影響實驗與分析
3. 損失地形視覺化
4. 殘差分析

**下週預告：** 分類 — 邏輯迴歸、決策邊界與 ROC/PR 曲線
