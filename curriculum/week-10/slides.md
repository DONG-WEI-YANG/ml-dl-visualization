# 第 10 週投影片：超參數調校與學習曲線

---

## Slide 1: 本週主題
# 超參數調校與學習曲線
### Hyperparameter Tuning & Learning Curves
- 如何科學地選擇超參數？
- 如何診斷模型的問題？
- 如何高效地搜尋最佳配置？

---

## Slide 2: 回顧上週
### Week 9: 特徵工程與資料前處理管線
- 特徵工程：編碼、縮放、衍生特徵
- sklearn Pipeline 建構
- 前處理對模型效能的影響

**本週重點：模型建好了，如何進一步調校？**

---

## Slide 3: 超參數 vs 模型參數
### 兩種「參數」的本質差異

| | 模型參數 | 超參數 |
|:---:|:---:|:---:|
| **誰決定？** | 訓練演算法 | 你（工程師） |
| **何時決定？** | 訓練過程中 | 訓練開始前 |
| **範例** | 權重 w, 偏差 b | 學習率, 樹深度 |

> 類比：模型參數 = 學生的答案；超參數 = 老師設定的考試規則

---

## Slide 4: 常見超參數一覽
### 每個模型都有需要調的「旋鈕」

```
決策樹：max_depth, min_samples_split
隨機森林：n_estimators, max_features
SVM：C, gamma, kernel
GBDT：learning_rate, n_estimators, max_depth
神經網路：learning_rate, batch_size, layers, dropout
```

**問題：這麼多旋鈕，該怎麼調？**

---

## Slide 5: 方法一 — Grid Search
### 窮舉搜尋：暴力但可靠

```
C     = [0.1, 1, 10]
gamma = [0.01, 0.1, 1]

3 × 3 = 9 種組合 → 全部試一遍
```

| 優點 | 缺點 |
|:---:|:---:|
| 簡單直觀 | 維度增加時爆炸 |
| 保證找到網格中最佳 | 只搜尋離散點 |
| 可重現 | 對不重要參數也花同樣資源 |

---

## Slide 6: Grid Search — 計算量
### 維度詛咒 Curse of Dimensionality

```
2 個超參數 × 5 值 × 5-fold = 50 次訓練
3 個超參數 × 5 值 × 5-fold = 625 次訓練
4 個超參數 × 5 值 × 5-fold = 3125 次訓練
5 個超參數 × 5 值 × 5-fold = 15625 次訓練！
```

**每次訓練 1 分鐘 → 5 個超參數需要 10+ 天**

---

## Slide 7: 方法二 — Random Search
### 隨機搜尋：聰明的捷徑

- 不嘗試所有組合
- 從分布中隨機取樣 N 組
- 可以使用連續分布

```python
param_dist = {
    'C': loguniform(0.01, 100),        # 連續
    'gamma': loguniform(0.0001, 1),    # 連續
}
RandomizedSearchCV(model, param_dist, n_iter=50)
```

---

## Slide 8: 為什麼 Random Search 更好？
### Bergstra & Bengio (2012) 的洞見

**關鍵觀察：通常只有少數超參數真正重要！**

```
Grid Search (9次):        Random Search (9次):
  B ↑                       B ↑
    | * * *                   |   *    *
    | * * *  ← 只看 3 個 A     | *    *   ← 看了 9 個 A
    | * * *                   |  *  *  *
    +------→ A(重要)           |  * *
                              +--------→ A(重要)
```

Random Search 在重要維度上探索更多可能性！

---

## Slide 9: 方法三 — Bayesian Optimization
### 從過去的經驗中學習

1. 試幾組隨機參數
2. 根據結果建立「代理模型」
3. 用模型預測哪裡可能更好
4. 在最有潛力的地方取樣
5. 重複 2-4

**優勢：更少的嘗試次數就能找到好結果**

---

## Slide 10: 三種方法比較
### Grid vs Random vs Bayesian

| | Grid | Random | Bayesian |
|---|:---:|:---:|:---:|
| 智慧程度 | 低 | 低 | 高 |
| 效率 | 低 | 中 | 高 |
| 實作難度 | 低 | 低 | 中 |
| 適合場景 | 少量超參數 | 一般場景 | 訓練耗時長 |
| 可平行化 | 高 | 高 | 低 |

---

## Slide 11: 學習曲線 Learning Curve
### 用資料量診斷模型問題

**X 軸：訓練樣本數**
**Y 軸：模型分數（訓練 & 驗證）**

```python
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5
)
```

---

## Slide 12: 學習曲線 — 三種模式

### 欠擬合 Underfitting
- 訓練 & 驗證分數都低
- 差距小
- 解法：增加模型複雜度

### 過擬合 Overfitting
- 訓練分數高、驗證分數低
- 差距大
- 解法：增加資料、正則化

### 良好擬合 Good Fit
- 兩者都高、差距小

---

## Slide 13: 驗證曲線 Validation Curve
### 用超參數值診斷模型問題

**X 軸：超參數值**
**Y 軸：模型分數（訓練 & 驗證）**

```
分數                    最佳值
  ↑        ┌──────┐      ↓
  │       / 訓練分數\
  │      /   ┌──┐   \
  │     / 驗證/ \分數 \
  │    /    /    \     \
  └───/────/──────\─────\──→ 超參數值
    欠擬合  最佳    過擬合
```

---

## Slide 14: 搜尋空間設計
### 三個設計原則

**1. 對數尺度 Log Scale**
- 學習率: 0.0001, 0.001, 0.01, 0.1（不是 0.1, 0.2, 0.3...）

**2. 先粗後細 Coarse-to-Fine**
- 第一輪：大範圍粗搜
- 第二輪：最佳區域細搜

**3. 考慮交互作用 Interactions**
- 學習率 & Batch Size 常需一起調
- max_depth & min_samples_split 互相制約

---

## Slide 15: 計算資源策略
### 在有限資源下最佳化

| 策略 | 節省幅度 |
|:---:|:---:|
| 減少 CV folds (5→3) | ~40% |
| 先用小資料集粗搜 | ~90% |
| Random Search | 可控 |
| Successive Halving | ~50-80% |
| 平行化 n_jobs=-1 | 依核心數 |
| 早停法 Early Stopping | 變動 |

---

## Slide 16: Successive Halving
### 逐步淘汰法

```
第 1 輪：81 組 × 100 筆資料 → 保留 27 名
第 2 輪：27 組 × 300 筆資料 → 保留 9 名
第 3 輪： 9 組 × 900 筆資料 → 保留 3 名
第 4 輪： 3 組 × 2700 筆資料 → 最佳 1 名
```

**核心：不值得在差的候選上浪費完整的計算量！**

---

## Slide 17: Optuna — 現代超參數調校
### 比 GridSearchCV 更聰明的選擇

- **Define-by-Run：** 動態定義搜尋空間
- **TPE 演算法：** 從歷史結果中學習
- **剪枝 Pruning：** 自動停止差的試驗
- **豐富視覺化：** 內建繪圖工具

```python
import optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

## Slide 18: 常見模型調校優先級
### 先調什麼最有效？

```
神經網路：Learning Rate >>> 架構 > 正則化 > 其他
GBDT：   Learning Rate > n_estimators > max_depth > subsample
RF：     n_estimators > max_depth > max_features
SVM：    C ≈ gamma > kernel
```

**經驗法則：從最有影響力的超參數開始！**

---

## Slide 19: 今日實作
### Notebook 動手做

1. GridSearchCV 實作與結果視覺化
2. RandomizedSearchCV 實作
3. 學習曲線繪製 — 診斷模型問題
4. 驗證曲線繪製 — 找最佳超參數
5. 超參數搜尋熱力圖
6. Grid vs Random 效率比較

---

## Slide 20: 本週作業
### 超參數調校實驗 + 學習曲線分析

1. 對指定資料集進行完整的超參數調校流程
2. 繪製並解讀學習曲線與驗證曲線
3. 比較不同搜尋策略的效率
4. 撰寫分析報告

---

## Slide 21: 下週預告
### Week 11: 神經網路基礎
- 感知器 (Perceptron) 與多層神經網路
- 激活函數 (Activation Functions) 視覺化
- 正則化技術：Dropout, BatchNorm
- 從傳統 ML 邁向深度學習！
