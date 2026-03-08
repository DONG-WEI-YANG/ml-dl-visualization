# 第 7 週投影片：樹模型與集成（RF、GBDT）

---

## Slide 1: 本週主題
# 樹模型與集成方法
### Tree Models & Ensemble Methods (RF, GBDT)

- 從決策樹到隨機森林與梯度提升
- 「三個臭皮匠，勝過一個諸葛亮」

**本週目標：** 理解樹模型的建構、剪枝，以及 Bagging vs. Boosting 的差異

---

## Slide 2: 決策樹直覺
### 決策樹 = 一系列 if-else 規則

```
        [花瓣長度 <= 2.45?]
         /              \
       Yes               No
       /                  \
   [Setosa]       [花瓣寬度 <= 1.75?]
                   /              \
                 Yes               No
                 /                  \
          [Versicolor]          [Virginica]
```

- **根節點 Root:** 起始決策
- **內部節點 Internal:** 依特徵分裂
- **葉節點 Leaf:** 最終預測

**適用於分類與回歸任務**

---

## Slide 3: 分裂準則 — 如何選擇最佳分裂？
### 目標：讓子節點越來越「純」

| 指標 | 公式 | 完美純 | 最不純（二元）|
|:---:|:---:|:---:|:---:|
| 熵 Entropy | $H = -\sum p_i \log_2 p_i$ | 0 | 1 |
| 基尼 Gini | $G = 1 - \sum p_i^2$ | 0 | 0.5 |

**資訊增益 (Information Gain):** 分裂後純度提升多少
$$IG = H_{parent} - \sum \frac{|S_v|}{|S|} H(S_v)$$

> sklearn 預設使用 Gini（CART 演算法），計算更快

---

## Slide 4: CART 演算法與剪枝
### CART: Classification and Regression Trees

**建構：** 對每個特徵、每個閾值，找基尼下降最大的分裂

**問題：** 不受限的樹 → 過擬合

**解法 — 剪枝 Pruning:**

| 預剪枝 Pre-pruning | 後剪枝 Post-pruning |
|:---:|:---:|
| 生長時設停止條件 | 先長完再修剪 |
| `max_depth`, `min_samples_split` | `ccp_alpha` (代價複雜度) |
| 快但可能錯過好分裂 | 慢但通常更好 |

**Demo:** 觀察不同 `max_depth` 的決策邊界變化

---

## Slide 5: 集成學習 — 群體的智慧
### Ensemble Learning: The Wisdom of Crowds

> 如果每位「評審」正確率 > 50% 且彼此獨立，多數決的正確率隨人數增加趨近 100%

**兩大策略：**

| Bagging | Boosting |
|:---:|:---:|
| 並行訓練，各自獨立 | 序列訓練，逐步修正 |
| 降低**變異** (Variance) | 降低**偏差** (Bias) |
| 多棵深樹取平均 | 多棵淺樹逐步累加 |
| 代表：Random Forest | 代表：GBDT, XGBoost |

---

## Slide 6: 隨機森林 Random Forest
### Bagging + 特徵隨機 = 雙重隨機性

**Step 1 — 資料隨機：** Bootstrap 取樣（有放回），每棵樹看到不同的資料

**Step 2 — 特徵隨機：** 每次分裂只考慮 $m = \sqrt{p}$ 個隨機特徵

**Step 3 — 聚合：** 投票（分類）或平均（回歸）

```
[Bootstrap 1] → Tree 1 ──┐
[Bootstrap 2] → Tree 2 ──┼── 投票/平均 → 最終預測
[Bootstrap 3] → Tree 3 ──┤
      ...         ...    ...
[Bootstrap B] → Tree B ──┘
```

**OOB Score:** 每棵樹約 36.8% 資料未使用，可作免費驗證

---

## Slide 7: RF 的 n_estimators 影響
### 樹越多，效能越穩定

| n_estimators | 效能 | 訓練時間 |
|:---:|:---:|:---:|
| 10 | 不穩定 | 快 |
| 100 | 趨穩 | 適中 |
| 500 | 幾乎飽和 | 較慢 |
| 1000+ | 邊際效益遞減 | 慢 |

**關鍵觀察：**
- 增加樹數**不會導致過擬合**（Bagging 的特性）
- 但計算成本線性增加
- 通常 100-500 棵已足夠

**Demo:** 觀察 n_estimators 從 1 到 500 的準確率曲線

---

## Slide 8: 梯度提升 Gradient Boosting (GBDT)
### 逐步修正殘差的序列學習

**核心思想：** 每棵新樹學習前面所有樹的「錯誤」（殘差）

```
F_0(x) = 初始預測（例如平均值）

第 1 棵樹：學習殘差 r_1 = y - F_0(x)
F_1(x) = F_0(x) + η · h_1(x)

第 2 棵樹：學習殘差 r_2 = y - F_1(x)
F_2(x) = F_1(x) + η · h_2(x)

... 持續修正 ...
```

| 真實值 | 第 1 棵預測 | 殘差 | 第 2 棵修正 | 累積預測 |
|:---:|:---:|:---:|:---:|:---:|
| 10 | 8 | +2 | +1.5 | 9.5 |
| 5 | 6 | -1 | -0.8 | 5.2 |

**學習率 η：** 控制每棵樹的貢獻，小 η + 多棵樹 = 更好泛化

---

## Slide 9: GBDT 超參數調校
### learning_rate 與 n_estimators 的交互作用

| learning_rate | n_estimators 需求 | 效能 | 過擬合風險 |
|:---:|:---:|:---:|:---:|
| 0.3 | 少 (~100) | 較差 | 高 |
| 0.1 | 中 (~300) | 良好 | 中 |
| 0.01 | 多 (~1000+) | 最好 | 低 |

**調參策略：**
1. 先用小 `learning_rate`（0.05-0.1）
2. 增加 `n_estimators` 直到驗證誤差不再下降
3. 調整 `max_depth`（通常 3-6）
4. 使用 Early Stopping 防止過擬合

---

## Slide 10: XGBoost 與 LightGBM
### 工業級 GBDT 實作

**XGBoost (2016, 陳天奇):**
- 正則化目標函數（L1/L2 懲罰）
- 二階泰勒展開（更精確的優化）
- 自動處理缺失值
- Kaggle 競賽的霸主

**LightGBM (2017, 微軟):**
- Leaf-wise 生長（更高效）
- Histogram-based 分裂（大幅加速）
- 訓練速度比 XGBoost 快 10-20 倍
- 記憶體使用更少

> 結構化資料 (Tabular Data) 上，這兩者通常是最佳選擇

---

## Slide 11: Bagging vs. Boosting 總結

| 面向 | Bagging (RF) | Boosting (GBDT) |
|:---:|:---:|:---:|
| 訓練 | 並行，獨立 | 序列，依賴 |
| 基學習器 | 深樹 | 淺樹 |
| 降低 | 變異 Variance | 偏差 Bias |
| 加樹過擬合？ | 不太會 | 可能會 |
| 雜訊敏感度 | 較強健 | 較敏感 |
| 調參難度 | 簡單 | 需仔細調 |
| Kaggle 排行榜 | 常見 | 霸主 |

**選擇建議：**
- 需要穩定基線 → RF
- 追求最佳效能 → GBDT/XGBoost/LightGBM
- 資料雜訊多 → RF

---

## Slide 12: 本週重點回顧與下週預告
### Key Takeaways

1. 決策樹透過遞迴分裂將特徵空間切割，使用 Gini/Entropy 作為準則
2. 剪枝（預剪枝 + 後剪枝）是防止過擬合的關鍵
3. **Random Forest** = Bagging + 特徵隨機 → 降低變異
4. **GBDT** = 序列訓練 + 殘差學習 → 降低偏差
5. XGBoost 和 LightGBM 是工業級的高效實作

### 本週作業
- 比較決策樹、RF、GBDT 在真實資料集上的效能
- 視覺化樹結構與決策邊界
- 分析超參數對模型的影響

### 下週預告：Week 8
**特徵重要度與 SHAP 值 — 讓模型可解釋**
