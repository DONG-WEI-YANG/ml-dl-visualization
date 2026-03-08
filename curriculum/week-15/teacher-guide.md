# 第 15 週教師手冊
# Week 15 Teacher Guide: Model Evaluation, Bias Detection, Fairness & Robustness

## 時間分配 Time Allocation（180 分鐘，含休息）

| 時段 | 分鐘 | 活動 | 說明 |
|------|:---:|------|------|
| 開場暖身 | 10 | 課前討論：「你遇過不公平的演算法嗎？」 | 引發思考，從生活經驗切入 |
| 理論一 | 25 | 進階評估指標 + 校準曲線 | Slide 3-6，搭配互動 Demo |
| 實作一 | 20 | 多類別混淆矩陣 + 校準曲線繪製 | Notebook 前半段 |
| 休息 | 10 | — | — |
| 理論二 | 30 | 偏誤來源 + 公平性定義 | Slide 7-12，重點講解不可能定理 |
| 實作二 | 25 | 公平性指標計算 + 分群 ROC | Notebook 中段 |
| 休息 | 10 | — | — |
| 理論三 | 15 | 穩健性與對抗樣本 | Slide 13-15，搭配圖片 Demo |
| 實作三 | 15 | FGSM 對抗擾動 + 偏誤緩解 | Notebook 後半段 |
| 案例討論 | 15 | AI 倫理案例分組討論 | Slide 16-18，小組活動 |
| 總結 | 5 | 回顧 + 作業說明 | Slide 24-26 |

---

## 教學重點 Key Teaching Points

### 核心訊息 Core Messages
1. **準確率不是萬能的：** 本週的核心目標是讓學生理解，模型評估需要超越單一指標。高準確率的模型可能對某些群體不公平，或在對抗攻擊下完全失效。
2. **公平性是一個選擇題：** 不同的公平性定義之間存在本質性的衝突（不可能定理），沒有「唯一正確」的公平性標準，必須根據應用場景做出取捨。
3. **技術與倫理不可分割：** AI 工程師不僅要會寫程式，更要理解技術決策的社會影響。

### 教學策略 Teaching Strategies

#### 策略一：從直覺到形式化
- 先用生活化的例子建立直覺（如：「考試及格線對不同群體公平嗎？」）
- 再引入數學定義和公式
- 最後用程式碼實際計算

#### 策略二：案例驅動 (Case-driven)
- 每介紹一個概念，立刻搭配真實案例
- 讓學生看到理論如何對應到現實問題
- 例如：講「Equalized Odds」時，立刻介紹 COMPAS 案例

#### 策略三：角色扮演
- 在 COMPAS 案例討論中，讓學生分組扮演不同角色：
  - 演算法開發者（辯護校準公平）
  - 公民權利律師（主張等化機會）
  - 法官（需要做出判決）
  - 被告（受到模型影響）
- 透過角色衝突，加深對不可能定理的理解

---

## 分段教學指引 Section-by-Section Guide

### 第一段：進階評估指標（35 分鐘）

**開場提問：**
> 「如果一個模型在 10 類分類問題中整體準確率 90%，但對某一類的 Recall 只有 20%，你覺得這個模型好嗎？」

**教學要點：**
- 多類別混淆矩陣：強調「看模式」比「看單一數字」重要
- 三種平均方式：用不平衡資料的例子（如垃圾郵件偵測）說明差異
- 校準曲線：先解釋「機率」的意義，再講如何檢驗

**常見迷思 Common Misconceptions：**
- 「Accuracy 高就是好模型」→ 反例：99% 的交易是正常的，全部預測正常也有 99% 準確率
- 「所有模型的 predict_proba 都可以當作真實機率」→ 需要校準
- 「Macro 比 Micro 好（或反之）」→ 各有適用場景

**互動 Demo 建議：**
- 在 Notebook 中調整類別不平衡程度，即時觀察三種平均的變化
- 繪製不同模型的校準曲線，讓學生猜哪個模型最「誠實」

### 第二段：偏誤與公平性（55 分鐘）

**情境引入：**
> 「假設你開發了一個信用貸款 AI，上線後發現對女性的核貸率只有男性的 60%。你的老闆問你：『這公平嗎？』你怎麼回答？」

**教學要點：**
- 偏誤來源：強調偏誤可能出現在 ML 管線的每個階段，不只是資料
- 公平性定義：這是本週最核心的內容，花足夠時間讓學生理解三種定義的直覺
- 不可能定理：這個概念的震撼力很大，要讓學生感受到「公平性不是技術問題，而是價值選擇」

**講解公平性定義的建議順序：**
1. 先從 Demographic Parity 開始（最直覺：各群體的結果比例要一樣）
2. 然後問：「但如果兩群人的能力確實不同呢？」→ 引出 Equalized Odds
3. 再問：「但如果真實標籤本身就有偏誤呢？」→ 引出 Calibration Fairness
4. 最後揭示不可能定理：三者不可能同時成立

**小組討論題目：**
- 「大學入學考試應該使用哪種公平性定義？為什麼？」
- 「如果 AI 的公平性比人類決策者更好（但不完美），我們應該使用 AI 嗎？」

### 第三段：穩健性與對抗樣本（30 分鐘）

**震撼開場：**
- 展示經典的 Panda → Gibbon 對抗樣本圖片
- 問學生：「你覺得這對自動駕駛意味著什麼？」

**教學要點：**
- FGSM 的直覺：沿著讓模型最「困惑」的方向移動一小步
- 對抗訓練的思路：把對抗樣本也加進訓練集
- 穩健性-準確率權衡：更穩健的模型通常犧牲一些標準準確率

**實作建議：**
- 讓學生調整 ε 值，觀察對抗效果的變化
- 可以讓學生嘗試對自己手寫的數字圖片做對抗攻擊（增加趣味性）

### 第四段：AI 倫理案例討論（15 分鐘）

**分組活動設計：**
1. 將學生分成 4-5 組
2. 每組分配一個案例（Amazon 招聘、COMPAS、Apple Card、醫療演算法）
3. 每組回答三個問題：
   - 偏誤的來源是什麼？
   - 應該使用哪種公平性定義？
   - 如何修正？
4. 各組用 3 分鐘報告
5. 全班討論共通的教訓

---

## 檢核點 Checkpoints

- [ ] 學生能解釋 Macro/Micro/Weighted Average 的差異
- [ ] 學生能繪製並解讀校準曲線
- [ ] 學生能用自己的話說明三種公平性定義
- [ ] 學生理解不可能定理的意義
- [ ] 學生能使用 Fairlearn 計算基本公平性指標
- [ ] 學生能解釋 FGSM 的原理
- [ ] 學生能在 AI 倫理案例中應用所學的概念
- [ ] 學生對「負責任 AI」有初步認識

---

## AI 助教設定 AI Tutor Configuration

本週助教設定為「批判思考模式」：
- 當學生直接問「這公平嗎？」時，引導學生先定義「公平」的標準
- 不直接給出「哪種公平性定義最好」的答案，而是幫助學生分析場景
- 對倫理問題，提供多元觀點而非單一結論
- 在公平性指標計算部分，可以提供程式碼提示
- 鼓勵學生提出自己的立場並為之辯護

**提示範例 Hint Examples：**
- Level 1: 「你覺得『公平』應該從結果的角度定義，還是從過程的角度定義？」
- Level 2: 「Demographic Parity 關注的是結果的比例，而 Equalized Odds 關注的是在控制真實情況後的表現差異。」
- Level 3: 「試著用 Fairlearn 的 MetricFrame，分別計算男性和女性的 TPR 和 FPR。」
- Level 4: 「可以參考以下程式碼結構：`mf = MetricFrame(metrics={...}, y_true=..., y_pred=..., sensitive_features=...)`」

---

## 常見問題與排除 Troubleshooting

### Q1: Fairlearn 安裝問題
```bash
pip install fairlearn
```
- 如遇版本衝突，嘗試：`pip install fairlearn==0.10.0`
- Google Colab 使用者：`!pip install fairlearn`

### Q2: Adult Census 資料集載入
```python
# 方法一：使用 Fairlearn 內建
from fairlearn.datasets import fetch_adult
data = fetch_adult(as_frame=True)

# 方法二：使用 UCI 原始資料
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, header=None, names=[...])
```

### Q3: PyTorch 對抗攻擊太慢
- 使用 MNIST 而非 CIFAR-10（更小的模型和圖片）
- 只使用部分測試集（前 1000 張）
- 確保使用 GPU（如有）

### Q4: 學生對「公平性不可能定理」感到沮喪
- 強調這不是「放棄公平」的理由，而是「必須做出明確選擇」的理由
- 引導學生思考：在特定場景中，哪種公平性最重要？
- 指出實務中常用的妥協方法（如多指標監控、利益相關者對話）

### Q5: 倫理討論偏離主題或變得情緒化
- 以尊重的方式將討論拉回技術層面
- 強調：本課程不是要下道德判斷，而是要建立分析框架
- 如果時間不夠，案例討論可簡化為教師講述 + 全班簡短 Q&A

---

## 差異化教學 Differentiated Instruction

### 對進度較快的學生 For Advanced Students
- 挑戰加分題：交叉公平性分析、SHAP 結合公平性
- 閱讀 Barocas et al. 的教科書章節
- 嘗試實作 PGD 攻擊（比 FGSM 更進階）
- 探索 AIF360 (AI Fairness 360) 工具包

### 對需要額外支援的學生 For Students Needing Support
- 提供已完成部分程式碼的 Notebook 模板
- 作業三（對抗樣本）可降低要求：只需實作 FGSM 一種 ε 值
- 作業四（報告）可縮短至 500-800 字
- 安排課後輔導時間或 AI 助教加強引導

---

## 備課提醒 Preparation Notes

### 課前準備
- [ ] 確認 Fairlearn 在示範環境中正確安裝
- [ ] 預先下載 Adult Census 資料集（避免課堂網路問題）
- [ ] 準備 COMPAS 案例的 ProPublica 原始報導連結
- [ ] 測試 FGSM 對抗攻擊的 Demo（確認圖片顯示正常）
- [ ] 準備經典對抗樣本圖片（Panda → Gibbon）作為開場素材
- [ ] 準備分組討論用的角色卡或題目卡

### 課堂氛圍
- 本週涉及社會議題，需營造開放、尊重的討論氛圍
- 避免對特定族群或性別做出刻板印象的陳述
- 強調技術人員的社會責任，但不要說教
- 讓學生感受到這些問題的真實性和重要性

### 延伸資源
- ProPublica COMPAS 報導：https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
- Fairlearn 官方教學：https://fairlearn.org/main/user_guide/
- Google What-If Tool：互動式公平性分析工具
- AI Fairness 360 (IBM)：https://aif360.mybluemix.net/

---

## 與前後週次的銜接 Connections to Other Weeks

### 回顧連結 Looking Back
- **Week 8 (SHAP/LIME)：** 可解釋性是公平性分析的基礎，本週進一步探討「為什麼模型對不同群體做出不同預測」
- **Week 14 (訓練技巧)：** 正則化、Dropout 等技術可以提升模型穩健性，呼應本週的對抗防禦

### 前瞻連結 Looking Forward
- **Week 16 (MLOps)：** 模型部署後的持續監控是實現負責任 AI 的關鍵環節
- **Week 18 (專題展示)：** 學生的期末專題應包含公平性和穩健性分析
