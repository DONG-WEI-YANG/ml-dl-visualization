# 第 15 週投影片：模型評估與偏誤檢測、公平性與穩健性

---

## Slide 1: 本週主題
# 模型評估與偏誤檢測、公平性與穩健性
### Model Evaluation, Bias Detection, Fairness & Robustness
- 你的模型「準確」就夠了嗎？
- 準確率背後可能隱藏著不公平
- 本週將學習如何全面評估模型

---

## Slide 2: 本週大綱
### 八大主題
1. 進階評估指標 — Macro / Micro / Weighted
2. 校準曲線 — 模型「說的機率」可信嗎？
3. 偏誤來源 — 資料、標籤、選擇
4. 公平性定義 — 三大定義與不可能定理
5. 公平性指標計算
6. 穩健性與對抗樣本
7. AI 倫理案例
8. 負責任 AI 框架

---

## Slide 3: 多類別混淆矩陣
### Multi-class Confusion Matrix
```
              預測 Cat  預測 Dog  預測 Bird
實際 Cat        45        3         2
實際 Dog         5       40         5
實際 Bird        2        7        41
```
- 對角線 = 正確分類
- 非對角線 = 誤分類模式
- 「Dog 最容易被誤認為 Bird」→ 可針對性改善

---

## Slide 4: Macro vs Micro vs Weighted
### 三種平均方式比較

| | Macro | Micro | Weighted |
|--|:---:|:---:|:---:|
| 概念 | 先算各類再平均 | 匯總所有再算 | 按樣本數加權 |
| 小類別權重 | 相同 | 被大類主導 | 按比例 |
| 適用場景 | 類別同等重要 | 樣本同等重要 | 一般用途 |

**重要：** 當類別不平衡時，Macro 和 Micro 差異顯著！

---

## Slide 5: 校準曲線
### Calibration Curve (Reliability Diagram)
- X 軸：模型預測機率
- Y 軸：實際正類比例
- 理想：落在對角線 y = x 上

```
          ┌────────────────────┐
  實際    │          /·····    │ ← 過度自信
  正類    │        /·         │
  比例    │      /·           │
          │    /              │ ← 完美校準
          │  /·               │
          │/···               │ ← 過度保守
          └────────────────────┘
              預測機率
```

---

## Slide 6: 各模型校準特性
### 誰的機率最可信？

| 模型 | 校準品質 | 常見問題 |
|------|:---:|------|
| Logistic Regression | 好 | 本身就是機率模型 |
| Random Forest | 中 | 機率集中在 0.5 附近 |
| SVM | 差 | 輸出非機率 |
| Neural Network | 差 | 過度自信 |
| XGBoost | 中 | 可透過校準改善 |

**校準方法：** Platt Scaling, Isotonic Regression, Temperature Scaling

---

## Slide 7: 偏誤從何而來？
### Sources of Bias

```
資料收集 → 資料標註 → 特徵工程 → 模型訓練 → 模型部署
   ↓          ↓          ↓          ↓          ↓
選擇偏誤   標籤偏誤   特徵代理   最佳化偏誤  回饋迴圈
```

每個階段都可能引入偏誤！

---

## Slide 8: 資料偏誤類型
### Data Bias Types

| 類型 | 範例 |
|------|------|
| 歷史偏誤 Historical | 過去招聘資料反映性別歧視 |
| 代表性偏誤 Representation | 醫學影像缺乏深色皮膚樣本 |
| 測量偏誤 Measurement | 巡邏密度影響犯罪統計 |
| 聚合偏誤 Aggregation | 忽略族裔間的藥物反應差異 |

---

## Slide 9: 公平性三大定義
### Three Fairness Definitions

**1. 人口統計均等 Demographic Parity**
- P(Ŷ=1 | A=男) = P(Ŷ=1 | A=女)
- 各群體的「錄取率」要相同

**2. 等化機會 Equalized Odds**
- TPR 和 FPR 在各群體間相等
- 控制真實標籤後，預測結果不應受群體影響

**3. 校準公平 Calibration Fairness**
- 同樣預測「70% 風險」→ 各群體的實際風險都是 70%

---

## Slide 10: 不可能定理
### Impossibility Theorem

> 除非基礎比率相等或分類器完美，否則三種公平性定義**無法同時滿足**。

```
   Demographic         Equalized
    Parity ◄───×────► Odds
        \              /
         \            /
          ×          ×
           \        /
            ▼      ▼
          Calibration
           Fairness
```

**實務意義：** 必須根據場景選擇最適合的公平性定義。

---

## Slide 11: 場景對應表
### 哪種公平性適合哪種場景？

| 場景 | 建議定義 | 理由 |
|------|----------|------|
| 招聘 Hiring | Demographic Parity | 確保平等機會 |
| 疾病篩檢 Screening | Equal Opportunity | 不漏掉任何群體 |
| 信用評分 Credit | Calibration | 預測風險要準確 |
| 司法風險 Criminal | Equalized Odds | 控制誤判率 |

---

## Slide 12: 公平性指標計算
### Fairness Metrics

```python
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)

# 分群分析
mf = MetricFrame(
    metrics={'accuracy': accuracy_score},
    y_true=y_true, y_pred=y_pred,
    sensitive_features=gender
)
print(mf.by_group)        # 各群體表現
print(mf.difference())    # 群體間差異
```

---

## Slide 13: 對抗樣本
### Adversarial Examples

```
     原圖          +    微小擾動      =   對抗樣本
   🐼 Panda         ε · sign(∇L)      🦊 "Gibbon"
   信心 99.3%                          信心 99.7%
```

- 人類看不出差異，但模型判斷完全改變
- 對 AI 安全構成嚴重威脅

---

## Slide 14: 攻擊方法
### Attack Methods

**FGSM (Fast Gradient Sign Method)**
- x_adv = x + ε · sign(∇_x L)
- 一步攻擊，快速但強度有限

**PGD (Projected Gradient Descent)**
- 迭代版 FGSM，多步小擾動
- 更強但計算量更大

**其他：** C&W Attack, DeepFool, Patch Attack

---

## Slide 15: 防禦方法
### Defense Methods

| 方法 | 階段 | 特點 |
|------|------|------|
| 對抗訓練 Adversarial Training | 訓練中 | 最有效但計算成本高 |
| 輸入前處理 Input Preprocessing | 推論前 | JPEG壓縮、平滑 |
| 認證防禦 Certified Defense | 理論保證 | Randomized Smoothing |
| Temperature Scaling | 推論後 | 改善信心校準 |

**核心權衡：** 穩健性↑ 通常意味著標準準確率↓

---

## Slide 16: 案例 — Amazon 招聘 AI
### Amazon Resume Screening AI (2018)

- 訓練資料：過去 10 年的錄取紀錄（男性主導）
- 結果：對包含「女性」相關詞彙的履歷降分
- **教訓：** 移除性別欄位不夠！代理特徵仍會洩露

---

## Slide 17: 案例 — COMPAS 刑事司法
### COMPAS Recidivism Risk Assessment

**ProPublica 發現：**
- 非裔被告被誤判為「高風險」的比例 = 白人的 2 倍
- Northpointe 辯稱具有校準公平

→ 這正是**不可能定理**的現實體現：
  等化機會 ≠ 校準公平，兩者不可兼得。

---

## Slide 18: 案例 — Apple Card 信用歧視
### Apple Card Credit Line Discrimination (2019)

- 女性獲得的信用額度系統性低於男性
- Goldman Sachs 聲稱未使用性別變數
- 但消費模式、職業等**代理變數**間接編碼了性別

→ **代理歧視 (Proxy Discrimination)** 是最難偵測的偏誤

---

## Slide 19: 偏誤緩解策略
### Bias Mitigation Strategies

```
┌──────────────┬──────────────┬──────────────┐
│  預處理       │  訓練中       │  後處理       │
│  Pre-process │  In-process  │  Post-process│
├──────────────┼──────────────┼──────────────┤
│ • 重新採樣    │ • 約束最佳化  │ • 閾值調整    │
│ • 重新加權    │ • 對抗去偏    │ • 校準等化    │
│ • 資料轉換    │ • 公平表示    │ • 拒絕選項    │
└──────────────┴──────────────┴──────────────┘
```

---

## Slide 20: 負責任 AI 七大原則
### 7 Principles of Responsible AI

| 原則 | 核心問題 |
|------|----------|
| 公平性 Fairness | 對誰有利？對誰不利？ |
| 可解釋性 Explainability | 為什麼做出這個決定？ |
| 透明度 Transparency | 模型的設計和限制公開嗎？ |
| 隱私 Privacy | 個人資料受到保護嗎？ |
| 安全性 Safety | 模型穩健且不易被攻擊嗎？ |
| 問責性 Accountability | 出問題時誰負責？ |
| 包容性 Inclusiveness | 設計過程中納入多元觀點了嗎？ |

---

## Slide 21: AI 影響評估流程
### AI Impact Assessment Process

```
問題定義 → 資料審查 → 模型開發 → 公平性測試
    ↓          ↓          ↓           ↓
適合AI嗎？   來源多元？   有約束嗎？   各群體表現？
    ↓          ↓          ↓           ↓
            穩健性測試 → 部署 → 持續監控
                             ↓
                         使用者回饋
```

---

## Slide 22: 全球 AI 監管趨勢
### Global AI Regulation Trends

| 地區 | 法規 | 特點 |
|------|------|------|
| 歐盟 | AI Act (2024) | 風險分級，高風險 AI 需嚴格審查 |
| 美國 | Blueprint for AI Bill of Rights | 原則性指引，非硬性法規 |
| 台灣 | AI 基本法 (2024) | 人權保障與創新平衡 |
| 中國 | 生成式 AI 管理辦法 | 強調內容審核與安全 |

---

## Slide 23: 本週實作重點
### Hands-on Lab
1. 多類別混淆矩陣熱力圖
2. 校準曲線繪製與比較
3. 公平性指標計算（Fairlearn）
4. 分群 ROC 曲線
5. FGSM 對抗擾動示範
6. 偏誤緩解策略實作

---

## Slide 24: 本週作業
### Assignment
1. 分析真實資料集的公平性（40%）
2. 穩健性測試與對抗樣本（30%）
3. AI 倫理案例分析報告（30%）

**繳交期限：** 下週上課前

---

## Slide 25: 本週反思問題
### Reflection Questions
1. 準確率高的模型一定公平嗎？
2. 如果公平性定義之間互相矛盾，你會如何選擇？
3. 誰應該為 AI 的偏誤負責？開發者？使用者？還是整個社會？
4. AI 是否應該被用於高風險決策（如刑事判決）？

---

## Slide 26: 下週預告
### Week 16 Preview
- MLOps 與模型部署 (Model Deployment)
- 模型版本控制與持續監控
- 從實驗到生產環境的完整流程
