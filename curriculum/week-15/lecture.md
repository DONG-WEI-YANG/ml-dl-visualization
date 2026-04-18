# 第 15 週：模型評估與偏誤檢測、公平性與穩健性
# Week 15: Model Evaluation, Bias Detection, Fairness & Robustness

## 學習目標 Learning Objectives
1. 掌握進階評估指標：多類別混淆矩陣 (Multi-class Confusion Matrix)、Macro/Micro/Weighted Average
2. 理解並繪製校準曲線 (Calibration Curve)，評估模型預測機率的可靠性
3. 辨識模型偏誤 (Bias) 的來源：資料偏誤、標籤偏誤、選擇偏誤
4. 理解公平性 (Fairness) 的多種定義與指標計算
5. 認識穩健性 (Robustness) 測試與對抗樣本 (Adversarial Examples)
6. 分析 AI 倫理案例，建立負責任 AI (Responsible AI) 的思維

---

## 1. 進階評估指標 Advanced Evaluation Metrics

### 1.1 多類別混淆矩陣 Multi-class Confusion Matrix

在二元分類 (Binary Classification) 中，混淆矩陣是 2x2 的表格。當問題擴展到多類別分類 (Multi-class Classification) 時，混淆矩陣成為 KxK 的方陣，其中 K 為類別數。

**多類別混淆矩陣的結構：**

```svg
<figure class="md-figure">
<svg viewBox="0 0 520 380" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="多類別混淆矩陣熱圖">
  <rect x="0" y="0" width="520" height="380" fill="#ffffff"/>
  <defs>
    <linearGradient id="cmGrad" x1="0" y1="1" x2="0" y2="0">
      <stop offset="0%" stop-color="#f1f5f9"/>
      <stop offset="100%" stop-color="#1e3a8a"/>
    </linearGradient>
  </defs>
  <text x="260" y="24" text-anchor="middle" font-size="14" fill="#111827" font-weight="600">多類別混淆矩陣 (K=3, 範例：setosa / versicolor / virginica)</text>
  <!-- Column headers (預測) -->
  <text x="280" y="70" text-anchor="middle" font-size="12" fill="#111827" font-weight="600">預測類別 Predicted</text>
  <g font-size="12" fill="#111827" text-anchor="middle">
    <text x="180" y="92">setosa</text><text x="280" y="92">versicolor</text><text x="380" y="92">virginica</text>
  </g>
  <!-- Row headers (實際) -->
  <text x="70" y="200" text-anchor="middle" font-size="12" fill="#111827" font-weight="600" transform="rotate(-90 70 200)">實際類別 Actual</text>
  <g font-size="12" fill="#111827" text-anchor="end">
    <text x="125" y="145">setosa</text><text x="125" y="205">versicolor</text><text x="125" y="265">virginica</text>
  </g>
  <!-- Grid 3x3 (heatmap) -->
  <!-- Row setosa — perfect -->
  <rect x="135" y="108" width="90" height="60" fill="#1e3a8a" opacity="0.9" stroke="#475569"/>
  <rect x="225" y="108" width="90" height="60" fill="#e5e7eb" stroke="#d1d5db"/>
  <rect x="315" y="108" width="90" height="60" fill="#e5e7eb" stroke="#d1d5db"/>
  <!-- Row versicolor — mostly correct, confused with virginica -->
  <rect x="135" y="168" width="90" height="60" fill="#e5e7eb" stroke="#d1d5db"/>
  <rect x="225" y="168" width="90" height="60" fill="#1e3a8a" opacity="0.8" stroke="#475569"/>
  <rect x="315" y="168" width="90" height="60" fill="#60a5fa" opacity="0.5" stroke="#475569"/>
  <!-- Row virginica — confused more often with versicolor -->
  <rect x="135" y="228" width="90" height="60" fill="#e5e7eb" stroke="#d1d5db"/>
  <rect x="225" y="228" width="90" height="60" fill="#60a5fa" opacity="0.55" stroke="#475569"/>
  <rect x="315" y="228" width="90" height="60" fill="#1e3a8a" opacity="0.75" stroke="#475569"/>
  <!-- Counts overlay -->
  <g font-size="15" fill="#ffffff" text-anchor="middle" font-weight="700">
    <text x="180" y="145">50</text><text x="280" y="205">45</text><text x="360" y="265">42</text>
  </g>
  <g font-size="13" fill="#111827" text-anchor="middle">
    <text x="280" y="145">0</text><text x="360" y="145">0</text>
    <text x="180" y="205">0</text>
  </g>
  <g font-size="13" fill="#1e3a8a" text-anchor="middle" font-weight="600">
    <text x="360" y="205">5</text>
    <text x="280" y="265">8</text>
    <text x="180" y="265">0</text>
  </g>
  <!-- Annotation arrows -->
  <path d="M 420 145 Q 455 145 450 200" fill="none" stroke="#059669" stroke-width="1.2" marker-end="url(#cmArr)"/>
  <defs>
    <marker id="cmArr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#059669"/></marker>
  </defs>
  <text x="460" y="130" font-size="10" fill="#065f46" font-weight="600">對角線 = TP</text>
  <text x="460" y="144" font-size="10" fill="#065f46">（正確分類）</text>
  <text x="460" y="230" font-size="10" fill="#991b1b" font-weight="600">非對角 = 誤分類</text>
  <text x="460" y="244" font-size="10" fill="#991b1b">versi↔virgi 常混淆</text>
  <!-- Color scale bar -->
  <rect x="135" y="310" width="270" height="12" fill="url(#cmGrad)" stroke="#d1d5db"/>
  <text x="135" y="338" font-size="10" fill="#6b7280">0</text>
  <text x="405" y="338" text-anchor="end" font-size="10" fill="#6b7280">50</text>
  <text x="270" y="338" text-anchor="middle" font-size="10" fill="#6b7280">樣本數</text>
  <!-- Diagonal accuracy summary -->
  <text x="260" y="362" text-anchor="middle" font-size="11" fill="#111827">Accuracy = (50+45+42)/150 = 91.3% ・ versicolor 與 virginica 之間為主要混淆來源</text>
</svg>
<figcaption>示意圖：3 類別混淆矩陣熱圖。深色對角線代表各類別被正確分類的樣本數（TP）；非對角線的淺藍格反映模型在 versicolor 與 virginica 之間混淆，這類混淆資訊無法從單一 accuracy 數字看出，須透過矩陣診斷。</figcaption>
</figure>
```

- 對角線元素 (Diagonal Elements) 代表正確分類
- 非對角線元素代表誤分類，可以看出「哪個類別最容易被混淆成哪個類別」

**從混淆矩陣衍生指標（以類別 i 為例）：**

| 指標 | 計算方式 | 意義 |
|------|----------|------|
| 精確率 Precision_i | TP_i / (TP_i + 其他類被預測為 i 的總數) | 預測為 i 的樣本中，真正是 i 的比例 |
| 召回率 Recall_i | TP_i / (TP_i + i 被錯誤預測為其他類的總數) | 實際為 i 的樣本中，被正確識別的比例 |
| F1 Score_i | 2 * Precision_i * Recall_i / (Precision_i + Recall_i) | 精確率與召回率的調和平均數 |

### 1.2 Macro / Micro / Weighted Average

在多類別問題中，需要將各類別的指標彙總為整體指標。有三種常見的平均方式：

#### Macro Average（巨觀平均）

$$\text{Macro-Precision} = \frac{1}{K} \sum_{i=1}^{K} \text{Precision}_i$$

- **作法：** 先計算每個類別的指標，再取算術平均
- **特點：** 每個類別權重相同，不論該類別樣本數多寡
- **適用場景：** 當所有類別同等重要時（例如：罕見疾病診斷）

#### Micro Average（微觀平均）

$$\text{Micro-Precision} = \frac{\sum_{i=1}^{K} TP_i}{\sum_{i=1}^{K} TP_i + \sum_{i=1}^{K} FP_i}$$

- **作法：** 將所有類別的 TP、FP、FN 加總後再計算
- **特點：** 受大類別主導，等同於整體的正確率 (Accuracy) 在多類別情境下的計算
- **適用場景：** 當每個樣本同等重要時

#### Weighted Average（加權平均）

$$\text{Weighted-Precision} = \sum_{i=1}^{K} w_i \cdot \text{Precision}_i, \quad w_i = \frac{n_i}{N}$$

- **作法：** 以各類別的樣本數為權重進行加權平均
- **特點：** 介於 Macro 與 Micro 之間，考量類別不平衡 (Class Imbalance)
- **適用場景：** 一般預設選擇，scikit-learn 中 `classification_report` 預設使用

**三者比較：**

| 特性 | Macro | Micro | Weighted |
|------|:---:|:---:|:---:|
| 對小類別的敏感度 | 高 | 低 | 中 |
| 對大類別的敏感度 | 低 | 高 | 高 |
| 類別不平衡下差異 | 大 | 小 | 中 |
| scikit-learn 參數 | `average='macro'` | `average='micro'` | `average='weighted'` |

### 1.3 其他進階指標

- **Cohen's Kappa (κ)：** 考慮隨機一致性 (Chance Agreement) 的一致性指標，κ = (p_o - p_e) / (1 - p_e)。κ > 0.8 通常被認為優秀。
- **Matthews Correlation Coefficient (MCC)：** 適用於不平衡資料集的相關係數指標，範圍 [-1, 1]。MCC = 1 表示完美預測。
- **Log Loss (Cross-Entropy Loss)：** 評估模型預測機率分布的品質，對過度自信的錯誤預測懲罰更重。

---

## 2. 校準曲線 Calibration Curve

### 2.1 什麼是模型校準？ What is Model Calibration?

模型校準 (Model Calibration) 衡量的是：**當模型預測某事件的機率為 p 時，該事件實際發生的頻率是否也接近 p。**

> 例如：若模型對 100 個樣本預測「罹病機率為 0.7」，那麼這 100 個樣本中，理想上應該有 70 個確實罹病。

**為什麼校準很重要？**
- 在醫療決策中，醫生需要可靠的風險機率來做治療決定
- 在金融風控中，信用評分模型的輸出必須準確反映違約機率
- 在自動駕駛中，對物件偵測的信心分數必須準確，才能做出安全決策

### 2.2 校準曲線的繪製方法

**步驟：**
1. 將模型的預測機率排序後分成若干區間 (Bins)，例如 [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
2. 對每個區間，計算該區間內樣本的平均預測機率 (Mean Predicted Probability) 和實際正類比例 (Fraction of Positives)
3. 以平均預測機率為 x 軸，實際正類比例為 y 軸繪製曲線
4. 完美校準的模型會落在 y = x 的對角線上

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# 計算校準曲線
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true, y_prob, n_bins=10, strategy='uniform'
)

# 繪製
plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
plt.plot([0, 1], [0, 1], '--', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Reliability Diagram)')
```

### 2.3 校準度量指標

- **Expected Calibration Error (ECE)：** 所有區間的校準誤差加權平均
  $$ECE = \sum_{b=1}^{B} \frac{n_b}{N} |\text{accuracy}(b) - \text{confidence}(b)|$$
- **Brier Score：** 預測機率與實際結果的均方誤差，$BS = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2$，越小越好

### 2.4 常見模型的校準特性

| 模型 | 校準特性 |
|------|----------|
| 邏輯回歸 Logistic Regression | 通常校準良好 (Well-calibrated) |
| 隨機森林 Random Forest | 傾向保守，機率集中在 0.5 附近 |
| SVM | 輸出非機率，需要 Platt Scaling |
| 神經網路 Neural Networks | 常過度自信 (Overconfident)，需要 Temperature Scaling |
| XGBoost / LightGBM | 中等校準品質，可受益於後處理校準 |

### 2.5 校準方法 Calibration Methods

1. **Platt Scaling：** 在模型輸出後接一個邏輯回歸，學習從原始分數到校準機率的映射
2. **Isotonic Regression（保序回歸）：** 非參數方法，更靈活但需要更多資料
3. **Temperature Scaling：** 專為神經網路設計，使用單一溫度參數 T 來縮放 logits
   $$\hat{p}_i = \text{softmax}(z_i / T)$$

---

## 3. 模型偏誤的來源 Sources of Model Bias

### 3.1 什麼是模型偏誤？ What is Model Bias?

模型偏誤 (Model Bias) 指的是模型在預測時，對特定群體 (Group) 產生系統性的不公平結果。這裡的「偏誤」不同於統計學中的「偏差」(Bias-Variance Tradeoff 中的 Bias)，而是指社會性的歧視與不公平。

### 3.2 資料偏誤 Data Bias

資料偏誤是最常見的偏誤來源，模型只能學到訓練資料中存在的模式。

| 偏誤類型 | 英文 | 說明 | 範例 |
|----------|------|------|------|
| 歷史偏誤 | Historical Bias | 資料反映了過去的不公平現象 | 歷史招聘資料中女性錄取率較低 |
| 代表性偏誤 | Representation Bias | 某些群體在資料中比例不足 | 醫學影像資料集主要來自白人患者 |
| 測量偏誤 | Measurement Bias | 資料收集方式對不同群體有系統性差異 | 犯罪資料受到巡邏密度影響 |
| 聚合偏誤 | Aggregation Bias | 忽略群體間的異質性而使用統一模型 | 不同族裔對相同藥物反應不同 |

### 3.3 標籤偏誤 Label Bias

標籤偏誤發生在資料標註 (Data Annotation) 過程中：

- **標註者偏見 (Annotator Bias)：** 標註者的主觀判斷受到其文化背景、成見影響
- **代理標籤 (Proxy Labels)：** 使用代理變數代替真實目標，例如用「逮捕紀錄」代替「犯罪事實」
- **標籤噪音不均 (Uneven Label Noise)：** 某些群體的標籤錯誤率系統性地高於其他群體

**案例：** 某情緒分析 (Sentiment Analysis) 資料集中，標註者對非裔美國人慣用語 (African American Vernacular English, AAVE) 的語句更容易標註為「有害」或「負面」。

### 3.4 選擇偏誤 Selection Bias

選擇偏誤 (Selection Bias) 發生在資料收集階段，某些樣本比其他樣本更有可能被納入資料集：

- **存活者偏差 (Survivorship Bias)：** 只看到「成功」的案例，忽略了失敗者
- **自選擇偏差 (Self-selection Bias)：** 參與者自願加入導致樣本不具代表性
- **排除偏差 (Exclusion Bias)：** 資料前處理中系統性地排除某些群體

**案例：** 線上調查資料排除了不使用網路的人群（通常是年長者或低收入群體），導致模型對這些群體預測不準確。

### 3.5 演算法偏誤 Algorithmic Bias

即使資料無偏，演算法本身也可能引入偏誤：

- **最佳化偏誤 (Optimization Bias)：** 模型最佳化目標函數時可能犧牲少數群體的表現
- **特徵選擇偏誤 (Feature Selection Bias)：** 看似中立的特徵實際上是敏感屬性的代理變數 (Proxy)，例如郵遞區號間接編碼了種族資訊
- **回饋迴圈 (Feedback Loop)：** 模型的偏誤預測影響了未來的資料收集，進而強化偏誤

---

## 4. 公平性定義 Fairness Definitions

### 4.1 敏感屬性 Sensitive/Protected Attributes

公平性分析的核心是識別**敏感屬性 (Sensitive Attributes)** 或**受保護屬性 (Protected Attributes)**，例如：
- 性別 (Gender)
- 種族 (Race/Ethnicity)
- 年齡 (Age)
- 宗教 (Religion)
- 身心障礙狀態 (Disability Status)

我們通常將人群分為**優勢群體 (Privileged Group)** 與**弱勢群體 (Unprivileged Group)**，並比較模型對兩者的表現差異。

### 4.2 人口統計均等 Demographic Parity (Statistical Parity)

**定義：** 模型的正向預測率 (Positive Prediction Rate) 在所有群體間應相等。

```svg
<figure class="md-figure">
<svg viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="群組公平性差距示意圖">
  <rect x="0" y="0" width="680" height="340" fill="#ffffff"/>
  <text x="340" y="24" text-anchor="middle" font-size="14" fill="#111827" font-weight="600">公平性指標：三群組的正向預測率比較</text>
  <!-- Axes -->
  <line x1="110" y1="260" x2="620" y2="260" stroke="#374151" stroke-width="1.2"/>
  <line x1="110" y1="60" x2="110" y2="260" stroke="#374151" stroke-width="1.2"/>
  <!-- Y ticks -->
  <g font-size="10" fill="#6b7280" text-anchor="end">
    <text x="102" y="264">0%</text><text x="102" y="220">20%</text><text x="102" y="180">40%</text>
    <text x="102" y="140">60%</text><text x="102" y="100">80%</text><text x="102" y="64">100%</text>
  </g>
  <g stroke="#e5e7eb" stroke-width="0.8" stroke-dasharray="3 3">
    <line x1="110" y1="220" x2="620" y2="220"/><line x1="110" y1="180" x2="620" y2="180"/>
    <line x1="110" y1="140" x2="620" y2="140"/><line x1="110" y1="100" x2="620" y2="100"/>
  </g>
  <text x="50" y="160" text-anchor="middle" font-size="12" fill="#111827" font-weight="600" transform="rotate(-90 50 160)">正向預測率 P(Ŷ=1 | A)</text>
  <!-- Bars: Group A 72%, Group B 45%, Group C 58% -->
  <!-- Group A -->
  <rect x="170" y="116" width="80" height="144" fill="#2563eb" stroke="#1e3a8a" stroke-width="1.5"/>
  <text x="210" y="110" text-anchor="middle" font-size="13" fill="#1e3a8a" font-weight="700">72%</text>
  <text x="210" y="280" text-anchor="middle" font-size="12" fill="#111827">群組 A (優勢)</text>
  <text x="210" y="295" text-anchor="middle" font-size="10" fill="#6b7280">Base rate: 65%</text>
  <!-- Group B — the disadvantaged group -->
  <rect x="310" y="170" width="80" height="90" fill="#f87171" stroke="#991b1b" stroke-width="1.5"/>
  <text x="350" y="164" text-anchor="middle" font-size="13" fill="#7f1d1d" font-weight="700">45%</text>
  <text x="350" y="280" text-anchor="middle" font-size="12" fill="#111827">群組 B (弱勢)</text>
  <text x="350" y="295" text-anchor="middle" font-size="10" fill="#6b7280">Base rate: 63%</text>
  <!-- Group C -->
  <rect x="450" y="144" width="80" height="116" fill="#a78bfa" stroke="#5b21b6" stroke-width="1.5"/>
  <text x="490" y="138" text-anchor="middle" font-size="13" fill="#5b21b6" font-weight="700">58%</text>
  <text x="490" y="280" text-anchor="middle" font-size="12" fill="#111827">群組 C</text>
  <text x="490" y="295" text-anchor="middle" font-size="10" fill="#6b7280">Base rate: 64%</text>
  <!-- Gap indicator between A and B -->
  <line x1="250" y1="116" x2="310" y2="116" stroke="#d97706" stroke-width="1.5"/>
  <line x1="250" y1="116" x2="250" y2="170" stroke="#d97706" stroke-width="1.5" stroke-dasharray="3 2"/>
  <line x1="310" y1="170" x2="310" y2="170" stroke="#d97706" stroke-width="1.5"/>
  <line x1="250" y1="170" x2="310" y2="170" stroke="#d97706" stroke-width="1.5"/>
  <text x="280" y="108" text-anchor="middle" font-size="12" fill="#b45309" font-weight="700">DPD = 27pp</text>
  <!-- 4/5 rule threshold: Group B should be ≥ 80% of Group A = 72%*0.8 = 57.6% -->
  <line x1="140" y1="146" x2="600" y2="146" stroke="#059669" stroke-width="1.5" stroke-dasharray="5 3"/>
  <text x="595" y="142" text-anchor="end" font-size="10" fill="#065f46" font-weight="600">4/5 規則門檻 = 57.6% (= 72%·0.8)</text>
  <!-- Verdict banner -->
  <rect x="130" y="310" width="490" height="24" fill="#fee2e2" stroke="#991b1b" stroke-width="1"/>
  <text x="375" y="326" text-anchor="middle" font-size="12" fill="#7f1d1d" font-weight="600">⚠ 群組 B 的 45% 低於 4/5 門檻 57.6%，疑似違反 EEOC Disparate Impact 準則</text>
</svg>
<figcaption>示意圖：群組公平性差距。儘管三群組 base rate 幾乎相同（63–65%），模型對群組 B 的正向預測率（45%）顯著低於群組 A（72%）。Demographic Parity Difference（DPD）= |72% - 45%| = 27 個百分點；依美國 EEOC 4/5 規則，弱勢群體正向率應 ≥ 優勢群體 ×0.8 = 57.6%，此模型未達標。</figcaption>
</figure>
```

$$P(\hat{Y}=1 | A=a) = P(\hat{Y}=1 | A=b), \quad \forall a, b$$

其中 A 是敏感屬性，$\hat{Y}$ 是模型預測。

**衡量指標：**
- Demographic Parity Difference (DPD) = |P(Ŷ=1|A=a) - P(Ŷ=1|A=b)|
- DPD 越接近 0，表示越公平
- 美國 EEOC 的「四分之五規則 (4/5 Rule)」：弱勢群體的正向率不應低於優勢群體的 80%

**優點：** 直觀、易於計算
**缺點：** 不考慮各群體的真實基礎比率 (Base Rate) 差異，可能導致合格者被拒絕或不合格者被接受

### 4.3 等化機會 Equalized Odds

**定義：** 模型的真正率 (True Positive Rate, TPR) 和假正率 (False Positive Rate, FPR) 在所有群體間應相等。

$$P(\hat{Y}=1 | Y=y, A=a) = P(\hat{Y}=1 | Y=y, A=b), \quad \forall y \in \{0, 1\}, \forall a, b$$

**特例 — 機會均等 (Equal Opportunity)：** 只要求 TPR 相等（Y=1 的條件下）。

$$P(\hat{Y}=1 | Y=1, A=a) = P(\hat{Y}=1 | Y=1, A=b)$$

**優點：** 在控制真實標籤的情況下比較預測結果，允許不同群體有不同的基礎比率
**缺點：** 需要知道真實標籤，在歷史標籤有偏誤時可能不適用

### 4.4 校準公平 Calibration Fairness (Predictive Parity)

**定義：** 在給定模型預測機率的情況下，不同群體的實際正類比例應相等。

$$P(Y=1 | \hat{Y}=p, A=a) = P(Y=1 | \hat{Y}=p, A=b), \quad \forall p, \forall a, b$$

**白話解釋：** 當模型說「你有 70% 的機率違約」時，無論你是哪個群體，你真正違約的機率都應該是 70%。

**優點：** 確保模型預測的機率對所有群體同樣可靠
**缺點：** 不保證各群體得到相同的正向預測率

### 4.5 公平性不可能定理 Impossibility Theorem

> **重要結論：** Chouldechova (2017) 和 Kleinberg et al. (2016) 證明，除了在非常特殊的情況下（基礎比率完全相等或完美分類器），**人口統計均等、等化機會和校準公平三者無法同時滿足。**

這意味著在實務中，我們必須根據應用場景選擇最適當的公平性定義：

| 場景 | 建議的公平性定義 | 原因 |
|------|------------------|------|
| 招聘決策 | Demographic Parity | 確保不同群體有平等的機會 |
| 疾病篩檢 | Equal Opportunity | 確保不遺漏任何群體的患者 |
| 信用評分 | Calibration Fairness | 確保預測風險對所有人同樣準確 |
| 刑事風險評估 | Equalized Odds | 同時控制 TPR 和 FPR |

---

## 5. 公平性指標計算 Fairness Metrics Computation

### 5.1 常用公平性指標一覽

| 指標 | 英文 | 公式 | 理想值 |
|------|------|------|:---:|
| 人口統計均等差異 | Demographic Parity Difference | P(Ŷ=1\|A=0) - P(Ŷ=1\|A=1) | 0 |
| 異質影響比率 | Disparate Impact Ratio | P(Ŷ=1\|A=0) / P(Ŷ=1\|A=1) | 1 |
| 等化機會差異 | Equalized Odds Difference | max(\|TPR_0 - TPR_1\|, \|FPR_0 - FPR_1\|) | 0 |
| 平均機會差異 | Average Odds Difference | 0.5 * (\|FPR_0 - FPR_1\| + \|TPR_0 - TPR_1\|) | 0 |
| 預測值均等差異 | Predictive Parity Difference | PPV_0 - PPV_1 | 0 |
| Theil Index | Theil Index | 衡量個體公平性的不平等指標 | 0 |

### 5.2 使用 Fairlearn 計算

```python
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 計算公平性指標
dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)

# 使用 MetricFrame 進行分組分析
metric_frame = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive
)
print(metric_frame.by_group)
print(metric_frame.difference())  # 群體間的最大差異
```

### 5.3 公平性視覺化最佳實踐

1. **分群 ROC 曲線：** 為每個敏感群體分別繪製 ROC 曲線，比較 AUC 差異
2. **公平性雷達圖 (Radar Chart)：** 在多個公平性維度上比較不同模型
3. **閾值分析圖 (Threshold Analysis)：** 展示不同決策閾值對各群體的影響
4. **混淆矩陣分群視覺化：** 分別展示各群體的混淆矩陣

---

## 6. 穩健性測試 Robustness Testing

### 6.1 什麼是對抗樣本？ What are Adversarial Examples?

對抗樣本 (Adversarial Examples) 是經過精心設計的微小擾動 (Perturbation)，使模型產生錯誤預測，但人類幾乎無法察覺變化。

> Szegedy et al. (2013) 首次發現，對影像加入人眼不可見的微小噪音，就能使深度神經網路的分類結果完全改變。

### 6.2 對抗攻擊方法 Adversarial Attack Methods

#### FGSM (Fast Gradient Sign Method)

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(\theta, x, y))$$

- 沿著損失函數梯度方向添加固定大小的擾動
- 優點：計算高效，只需一步
- 缺點：攻擊強度有限

#### PGD (Projected Gradient Descent)

$$x_{t+1} = \Pi_{x + S} \left( x_t + \alpha \cdot \text{sign}(\nabla_x L(\theta, x_t, y)) \right)$$

- FGSM 的迭代版本，多次小步移動
- 被認為是「最強的一階攻擊」(strongest first-order attack)
- Π 表示投影回允許的擾動範圍

#### 其他攻擊方法

| 方法 | 類型 | 特點 |
|------|------|------|
| C&W Attack | 最佳化型 | 找到最小擾動使分類改變 |
| DeepFool | 幾何型 | 計算到最近決策邊界的距離 |
| Patch Attack | 實體型 | 在圖片上貼一個對抗性貼紙 |
| Universal Perturbation | 通用型 | 一個擾動適用於所有圖片 |

### 6.3 對抗防禦方法 Adversarial Defense Methods

1. **對抗訓練 (Adversarial Training)：** 在訓練過程中加入對抗樣本
   ```python
   for x, y in dataloader:
       x_adv = fgsm_attack(model, x, y, epsilon=0.03)
       loss = criterion(model(x), y) + criterion(model(x_adv), y)
       loss.backward()
   ```

2. **輸入前處理 (Input Preprocessing)：**
   - JPEG 壓縮
   - 空間平滑 (Spatial Smoothing)
   - 特徵壓縮 (Feature Squeezing)

3. **模型正則化 (Model Regularization)：**
   - Lipschitz 約束
   - Spectral Normalization
   - 梯度正則化 (Gradient Regularization)

4. **認證防禦 (Certified Defense)：**
   - Randomized Smoothing：提供可證明的穩健性保證
   - 以機率方式保證在某個擾動範圍內，模型的預測不會改變

### 6.4 穩健性與準確率的權衡

> **核心洞察：** 穩健性 (Robustness) 與標準準確率 (Standard Accuracy) 之間存在本質性的權衡 (Trade-off)。更穩健的模型在乾淨資料上的準確率通常會略有下降。

Tsipras et al. (2019) 的研究表明，穩健模型學到的特徵在語義上更有意義 (Semantically Meaningful)，而標準模型則可能依賴非穩健特徵 (Non-robust Features) 來提高準確率。

---

## 7. AI 倫理案例分析 AI Ethics Case Studies

### 7.1 招聘 AI — Amazon 的案例

**背景：** 2018 年，路透社報導 Amazon 內部開發的履歷篩選 AI 對女性求職者存在系統性歧視。

**問題分析：**
- **資料偏誤：** 訓練資料取自過去 10 年的錄取紀錄，而科技業長期以男性為主
- **特徵代理：** 模型學會對「女性」相關詞彙（如「女子棋社社長」、「女子學院」）降低評分
- **回饋迴圈：** 即使移除性別欄位，模型仍透過其他關聯特徵間接推斷性別

**教訓：**
1. 移除敏感屬性不足以消除偏誤（「公平性靠遮蔽」的迷思 — Fairness Through Unawareness 的失敗）
2. 歷史資料反映的是過去的不公平，而非理想的決策
3. AI 系統必須經過公平性審查 (Fairness Audit) 才能部署

### 7.2 刑事司法 — COMPAS 的案例

**背景：** COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) 是美國法院廣泛使用的再犯風險評估工具 (Recidivism Risk Assessment)。

**爭議：** 2016 年 ProPublica 的調查發現：
- 非裔被告被錯誤標記為「高風險」的比例是白人被告的兩倍（FPR 差異）
- 白人被告被錯誤標記為「低風險」而再犯的比例更高（FNR 差異）

**公平性定理的體現：**
- Northpointe（COMPAS 的開發公司）辯稱模型具有「校準公平性」：在同樣被預測為高風險的人中，不分種族，再犯率相近
- ProPublica 則以「等化機會」的角度指出模型對非裔被告不公平
- 這正是公平性不可能定理的實際體現：**兩種公平性無法同時滿足**

### 7.3 信用評分 — Apple Card 的案例

**背景：** 2019 年，多位使用者發現 Apple Card 的信用額度對女性系統性地低於男性，即使在收入和信用評分相同的情況下。

**問題分析：**
- Goldman Sachs（Apple Card 的發卡銀行）聲稱未使用性別作為模型輸入
- 但其他特徵（如消費模式、職業類型）可能間接編碼了性別資訊
- 紐約金融監管機構介入調查

**啟示：**
- 即使模型不直接使用敏感屬性，代理變數 (Proxy Variables) 仍可能導致歧視
- 金融領域需要更嚴格的公平性測試和監管

### 7.4 醫療健康 — Optum 的案例

**背景：** Obermeyer et al. (2019) 在 Science 期刊上發表研究，揭示美國醫療系統中一個廣泛使用的演算法對黑人患者存在嚴重偏誤。

**問題分析：**
- 該演算法使用**醫療費用**作為「健康需求」的代理標籤
- 由於結構性不平等，黑人患者在同等疾病嚴重度下獲得較少的醫療資源，因此醫療費用較低
- 演算法因此系統性地低估了黑人患者的健康風險

**修正策略：**
- 將代理標籤從「醫療費用」改為更直接的健康指標（如慢性疾病數量）
- 修正後，被識別為需要額外照護的黑人患者比例從 17.7% 提升到 46.5%

---

## 8. 負責任 AI 框架 Responsible AI Framework

### 8.1 負責任 AI 的核心原則

| 原則 | 英文 | 說明 |
|------|------|------|
| 公平性 | Fairness | 模型對所有群體公正對待 |
| 可解釋性 | Explainability / Interpretability | 模型的決策過程可以被理解 |
| 透明度 | Transparency | 公開模型的設計、訓練過程與限制 |
| 隱私保護 | Privacy | 保護個人資料不被不當使用 |
| 安全性 | Safety & Security | 模型穩健可靠，不易被攻擊 |
| 問責性 | Accountability | 有明確的責任歸屬與申訴機制 |
| 包容性 | Inclusiveness | 設計過程納入多元觀點 |

### 8.2 業界框架

| 組織 | 框架名稱 | 重點特色 |
|------|----------|----------|
| Google | Responsible AI Practices | 強調公平性與隱私 |
| Microsoft | Responsible AI Standard | 六大原則 + 實施指南 |
| Meta | Responsible AI Pillars | 重視透明度與治理 |
| OECD | AI Principles | 國際政策標準 |
| EU | AI Act (2024) | 風險分級監管 |
| 台灣 | AI 基本法 (2024) | 聚焦人權保障與創新平衡 |

### 8.3 AI 影響評估 AI Impact Assessment

在部署 AI 系統前，應進行完整的影響評估 (Impact Assessment)：

1. **問題定義檢核：**
   - 這個問題是否適合用 AI 解決？
   - 使用 AI 可能帶來哪些風險？

2. **資料審查 (Data Audit)：**
   - 資料來源是否多元且具代表性？
   - 標註過程是否有品質控管？

3. **模型公平性測試：**
   - 使用多種公平性定義進行測試
   - 記錄各群體的表現差異

4. **穩健性測試：**
   - 對模型進行對抗性測試
   - 測試分布外 (Out-of-Distribution) 資料的表現

5. **部署後監控 (Post-deployment Monitoring)：**
   - 持續追蹤模型表現與公平性指標
   - 建立使用者回饋與申訴機制
   - 定期重新評估模型

### 8.4 偏誤緩解策略 Bias Mitigation Strategies

偏誤緩解可以在機器學習管線的不同階段進行：

#### 預處理 (Pre-processing)
- **重新採樣 (Resampling)：** 平衡各群體的樣本數
- **重新加權 (Reweighting)：** 為不同樣本賦予不同權重
- **資料轉換 (Disparate Impact Remover)：** 修改特徵使其獨立於敏感屬性

#### 訓練中 (In-processing)
- **約束最佳化 (Constrained Optimization)：** 在損失函數中加入公平性約束
- **對抗去偏 (Adversarial Debiasing)：** 訓練模型使其無法從表示中預測敏感屬性
- **公平表示學習 (Fair Representation Learning)：** 學習不包含敏感資訊的表示

#### 後處理 (Post-processing)
- **閾值調整 (Threshold Adjustment)：** 為不同群體使用不同的決策閾值
- **校準等化 (Calibrated Equalized Odds)：** 後處理調整使模型滿足等化機會
- **拒絕選項分類 (Reject Option Classification)：** 對不確定的預測重新分配

---

## 關鍵詞彙表 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 混淆矩陣 | Confusion Matrix | 分類模型預測結果與實際標籤的交叉表 |
| 巨觀平均 | Macro Average | 各類別指標的算術平均 |
| 微觀平均 | Micro Average | 將所有類別的 TP/FP/FN 匯總後計算 |
| 加權平均 | Weighted Average | 以樣本數加權的指標平均 |
| 校準曲線 | Calibration Curve | 評估預測機率可靠性的圖形 |
| 期望校準誤差 | Expected Calibration Error (ECE) | 量化校準偏差的指標 |
| 布賴爾分數 | Brier Score | 預測機率的均方誤差 |
| 模型偏誤 | Model Bias | 模型對特定群體的系統性不公平 |
| 資料偏誤 | Data Bias | 資料中存在的系統性偏差 |
| 標籤偏誤 | Label Bias | 標註過程中引入的偏差 |
| 選擇偏誤 | Selection Bias | 資料收集過程中的樣本選取偏差 |
| 敏感屬性 | Sensitive/Protected Attribute | 受法律或倫理保護的個人屬性 |
| 人口統計均等 | Demographic Parity | 各群體的正向預測率應相等 |
| 等化機會 | Equalized Odds | 各群體的 TPR 和 FPR 應相等 |
| 機會均等 | Equal Opportunity | 各群體的 TPR 應相等 |
| 校準公平 | Calibration Fairness | 各群體在相同預測值下的實際比例應相等 |
| 異質影響 | Disparate Impact | 模型對不同群體產生不成比例的負面影響 |
| 對抗樣本 | Adversarial Examples | 經精心設計使模型誤判的輸入 |
| 對抗訓練 | Adversarial Training | 使用對抗樣本增強模型穩健性 |
| FGSM | Fast Gradient Sign Method | 基於梯度的快速對抗攻擊方法 |
| PGD | Projected Gradient Descent | 迭代式梯度對抗攻擊方法 |
| 穩健性 | Robustness | 模型對輸入擾動的抵抗能力 |
| 負責任 AI | Responsible AI | 強調公平、透明、安全的 AI 開發框架 |
| 偏誤緩解 | Bias Mitigation | 減少模型偏誤的技術策略 |
| 公平表示學習 | Fair Representation Learning | 學習不包含敏感資訊的資料表示 |
| 溫度縮放 | Temperature Scaling | 使用溫度參數調整預測機率校準 |
| 公平性不可能定理 | Impossibility Theorem | 證明多種公平性定義無法同時滿足 |

---

## 延伸閱讀 Further Reading

- Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities*. [https://fairmlbook.org](https://fairmlbook.org)
- Mehrabi, N., et al. (2021). "A Survey on Bias and Fairness in Machine Learning." *ACM Computing Surveys*.
- Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR*.
- Google Responsible AI Toolkit: [https://www.tensorflow.org/responsible_ai](https://www.tensorflow.org/responsible_ai)
- Fairlearn 官方文件: [https://fairlearn.org](https://fairlearn.org)
- Microsoft Responsible AI Standard: [https://www.microsoft.com/en-us/ai/responsible-ai](https://www.microsoft.com/en-us/ai/responsible-ai)
- Obermeyer, Z., et al. (2019). "Dissecting racial bias in an algorithm used to manage the health of populations." *Science*.
