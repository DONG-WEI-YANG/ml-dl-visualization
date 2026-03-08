# 第 5 週教師手冊：分類 — 邏輯迴歸、決策邊界與 ROC/PR 曲線
# Week 5 Teacher Guide: Classification — Logistic Regression, Decision Boundary & ROC/PR Curves

---

## 課程概覽 Session Overview

| 項目 | 內容 |
|------|------|
| **主題** | 分類 (Classification)：邏輯迴歸、決策邊界與 ROC/PR 曲線 |
| **時長** | 90 分鐘（理論 30 分 + 實作 40 分 + 討論 20 分） |
| **難度** | 核心 Core |
| **先備知識** | Week 1-4（Python 基礎、EDA、監督學習、線性回歸） |
| **核心能力指標** | 理解分類模型原理、掌握評估指標、能繪製與解讀 ROC/PR 曲線 |

---

## 教學目標與對應活動 Objectives & Activities Mapping

| 學習目標 | 教學活動 | 評估方式 | 時間 |
|----------|----------|----------|:----:|
| 理解從回歸到分類的轉變 | Slide 2 + 互動提問 | 概念題 A1 | 5 分 |
| 掌握 Sigmoid 函數與邏輯迴歸 | Slide 3-4 + Notebook Part 1-2 | 練習 1-2 | 10 分 |
| 理解決策邊界的形成與 C 值影響 | Slide 5 + Notebook Part 3-4 | 作業 B1 | 10 分 |
| 掌握二元交叉熵損失 | Slide 4 + 數學推導 | 概念題 A2 | 5 分 |
| 理解分類評估指標 | Slide 7 + Notebook Part 5 | 概念題 A3 | 10 分 |
| 繪製與解讀 ROC/PR 曲線 | Slide 8-9 + Notebook Part 6-7 | 作業 B1 | 15 分 |
| 了解閾值選擇的影響 | Slide 6 + Notebook Part 8 | 作業 B1-6 | 10 分 |
| 處理不平衡資料集 | Slide 10 + Notebook Part 9 | 作業 B2 | 10 分 |
| 認識多類別分類方法 | Slide 11 | 進階題 C1 | 5 分 |

---

## 詳細教學流程 Detailed Teaching Flow

### 第一階段：理論講解 (30 分鐘)

#### 0:00 - 0:05 開場與回顧 Opening & Review

**教學策略：** 從第 4 週的線性回歸銜接到分類問題。

- 回顧上週：「線性回歸預測的是連續值，如房價。但如果我們要判斷一封郵件是否為垃圾郵件呢？」
- 拋出問題：「如果直接用線性回歸做分類會怎樣？」

> **教學提示：** 用簡單的圖示說明線性回歸做分類時的問題（輸出超出 [0,1]、受離群值影響）。可以在黑板上畫一個散佈圖，標示正/負類，然後畫一條回歸線，讓學生看到預測值可能是負數或大於 1 的問題。

#### 0:05 - 0:15 Sigmoid 函數與邏輯迴歸 Sigmoid & Logistic Regression

**教學策略：** 視覺化驅動，讓學生「看到」Sigmoid 如何解決問題。

**展示順序：**
1. 展示 Slide 3 (Sigmoid 函數)
2. 開啟 Notebook Part 1，執行 Sigmoid 視覺化
3. 帶學生觀察：值域 (0,1)、對稱性、中心點 (0, 0.5)

**重要概念強調：**
- Sigmoid 的輸出可以解釋為「機率」— 這不只是巧合，而是有嚴格的數學推導（MLE）
- 邏輯迴歸本質上仍然是線性模型（在 log-odds 空間中），Sigmoid 只是最後的「包裝」

> **常見誤解 Common Misconception：** 學生常以為邏輯迴歸是「非線性模型」。要強調它的決策邊界是線性的，只是輸出通過了非線性的 Sigmoid 函數。

#### 0:15 - 0:20 二元交叉熵與決策邊界 BCE Loss & Decision Boundary

**教學策略：** 用表格對比 BCE 和 MSE，讓學生理解為什麼分類用 BCE。

1. 展示 Slide 4 的損失表格
2. 強調：BCE 對「高信心的錯誤預測」施加特別大的懲罰
3. 展示 Slide 5 (決策邊界)
4. 講解閾值 = 0.5 時決策邊界就是 $w^Tx + b = 0$

> **教學技巧：** 讓學生手算 $-\ln(0.01) \approx 4.6$ vs $-\ln(0.99) \approx 0.01$，體會 BCE 的「懲罰機制」。

#### 0:20 - 0:30 分類指標與 ROC/PR 曲線 Metrics & ROC/PR Curves

**教學策略：** 用實際場景帶入指標的意義。

**建議的教學順序：**
1. 先介紹混淆矩陣 (Slide 7)——這是所有指標的基礎
2. 用「疾病篩檢」場景說明 Precision 和 Recall 的直覺意義
3. 介紹 ROC 曲線 (Slide 8)——強調 AUC 的統計解釋
4. 介紹 PR 曲線 (Slide 9)——強調不平衡資料時的優勢
5. 快速帶過不平衡資料處理 (Slide 10) 和多類別分類 (Slide 11)

> **教學提示：** Precision/Recall 的差別是學生最容易搞混的概念。建議用以下口訣：
> - **Precision**：「模型說是正的，有多準？」（看的是預測結果的品質）
> - **Recall**：「真正的正樣本，抓到多少？」（看的是覆蓋率）

**互動問題：**
- 「如果你是醫生，你更擔心 FP（把健康人誤診為病人）還是 FN（把病人漏掉）？」
- 「垃圾郵件過濾呢？FP 和 FN 哪個更嚴重？」

---

### 第二階段：實作環節 (40 分鐘)

#### 0:30 - 0:40 Notebook Part 1-2: Sigmoid 與從零實作

**引導方式：**
1. 先讓學生執行 Sigmoid 視覺化，觀察不同 w、b 的影響
2. 帶學生逐步理解 `LogisticRegressionScratch` 類別的每個方法
3. 重點解釋梯度公式的直覺意義：$\nabla_w = \frac{1}{N}X^T(\hat{p} - y)$
   - 「如果預測太高 ($\hat{p} > y$)，梯度為正，權重會被減小」
   - 「如果預測太低 ($\hat{p} < y$)，梯度為負，權重會被增大」

> **教學提示：** 與 sklearn 的比較結果很重要——讓學生看到自己從零實作的模型能達到接近的性能，會很有成就感。若結果略有差異，可以解釋 sklearn 使用的是 L-BFGS 等更高級的優化器。

#### 0:40 - 0:55 Notebook Part 3-5: 決策邊界與混淆矩陣

**引導方式：**
1. 決策邊界視覺化是本週的「重頭戲」——讓學生花時間理解顏色、邊界線的含義
2. C 值實驗：讓學生預測 C 改變後邊界會怎麼變，再執行看結果
3. 混淆矩陣：讓學生指出每個格子對應的 TP/TN/FP/FN

**期望的學生反應：**
- 看到 C 值很大時邊界變得「扭曲」會覺得新奇
- 理解混淆矩陣的四個格子並不困難，但理解指標的業務意義需要引導

#### 0:55 - 1:10 Notebook Part 6-9: ROC/PR 曲線與不平衡資料

**引導方式：**
1. ROC 曲線：指出曲線越靠近左上角越好
2. PR 曲線：指出曲線越靠近右上角越好
3. 閾值調整實驗：這是最直觀的部分——讓學生看到移動閾值如何影響混淆矩陣
4. 不平衡資料：先讓學生看 Baseline 的結果，問「這個模型好嗎？」等學生意識到 Recall 極低後，再介紹處理策略

> **教學重點：** 不平衡資料的部分是最具現實意義的。強調在真實世界中，大多數分類問題的資料都是不平衡的。

---

### 第三階段：討論與總結 (20 分鐘)

#### 1:10 - 1:25 小組討論 Group Discussion

**討論題目（選 2-3 題）：**

1. **閾值選擇辯論：**「你被聘為某醫院的 AI 顧問，需要為癌症篩檢系統設定閾值。院方的兩位醫生有不同意見：一位主張閾值設低一點（0.3），另一位主張設高一點（0.7）。你會如何分析？最終建議是什麼？」

2. **Accuracy 的陷阱：**「你的同事宣稱他訓練的模型 Accuracy 達到 99.5%。在什麼情況下，這個數字可能毫無意義？如何防止被 Accuracy 誤導？」

3. **ROC vs PR 辯論：**「你有兩個模型：模型 A 的 AUC = 0.95，模型 B 的 AUC = 0.92，但模型 B 的 PR-AUC = 0.80，模型 A 的 PR-AUC = 0.65。你的資料集是不平衡的（正類佔 3%）。你會選哪個模型？為什麼？」

4. **現實場景應用：**「SMOTE 在什麼情況下效果不好？試想一個場景，說明為什麼生成合成樣本可能引入錯誤。」

#### 1:25 - 1:30 總結與作業說明 Wrap-up & Assignment

**本週三大收穫（讓學生回答）：**
1. 邏輯迴歸的本質是什麼？
2. 為什麼不同場景需要不同的評估指標？
3. 不平衡資料為什麼需要特殊處理？

**作業提醒：**
- Part A 概念題需要手動計算，不能只寫答案
- Part B 實作題需要圖表有完整標籤
- 進階題 C 是選做，但值得挑戰

**下週預告：** 第 6 週 — SVM 與核方法視覺化。邏輯迴歸的決策邊界是線性的，下週我們將看到如何用核技巧 (Kernel Trick) 建立非線性決策邊界。

---

## 教學資源 Teaching Resources

### 預備教材 Materials to Prepare
- 投影片 `slides.md` (12 張)
- 實作 Notebook `notebook.ipynb`
- 作業說明 `assignment.md`
- 評量規準 `rubric.md`（公開給學生看）

### 所需軟體與套件 Required Software
```
numpy, pandas, matplotlib, seaborn, scikit-learn
```
所有套件在 Week 1 已安裝，本週不需要額外安裝。

### 備用資料集 Backup Datasets
如果 `make_classification` 生成的資料太簡單或太難，可使用：
- `sklearn.datasets.load_breast_cancer`（真實醫療資料，569 samples, 30 features）
- `sklearn.datasets.load_iris`（多類別，用於 Softmax 示範）
- `sklearn.datasets.make_moons`（非線性可分，展示邏輯迴歸的局限）

---

## 常見學生問題與回應 Anticipated Questions & Responses

### Q1:「邏輯迴歸為什麼叫『回歸』？它不是做分類的嗎？」

**回應：** 好問題！名稱確實容易混淆。邏輯迴歸實際上是在對「對數勝算 (Log-Odds)」做線性回歸。它的輸出是一個連續的機率值，只是最後我們用閾值將其轉為類別。歷史上它在統計學中被發展出來時，被視為一種「廣義線性模型 (Generalized Linear Model, GLM)」，所以保留了「回歸」的名稱。

### Q2:「AUC = 0.5 是完全隨機，那 AUC < 0.5 代表什麼？」

**回應：** 如果 AUC < 0.5，表示你的模型比隨機還差。但好消息是，只要把預測標籤翻轉（0 變 1，1 變 0），就會得到 AUC > 0.5 的模型。實務上 AUC < 0.5 通常代表標籤定義反了，或是模型訓練過程有錯誤。

### Q3:「SMOTE 和直接複製少數類有什麼不同？」

**回應：** 直接複製（隨機過採樣）只是重複已有的樣本，容易導致過擬合——模型可能會「死記」這些重複的樣本。SMOTE 則是在少數類樣本之間的空間中生成**新的合成樣本**，增加了多樣性，減少過擬合的風險。但 SMOTE 也有局限：如果少數類的分布本身很複雜或不連續，插值可能產生不合理的樣本。

### Q4:「Precision 和 Recall 不能同時都高嗎？」

**回應：** 理論上可以——如果模型非常好。但在實務中，通常存在 Precision-Recall Trade-off。降低閾值會提高 Recall 但降低 Precision（因為更多負樣本被預測為正）。這就是為什麼我們需要根據業務需求來選擇平衡點。F1-Score 是一個常用的折衷指標。

### Q5:「多類別分類時，OvR 和 Softmax 哪個比較好？」

**回應：** 在邏輯迴歸的場景下，兩者通常效果相似。OvR 的優點是概念簡單，可以平行訓練。Softmax 的優點是輸出天然就是機率分布（和為 1），而且在深度學習中是標準做法。如果你後續要學習神經網路，建議先理解 Softmax。

### Q6:「class_weight='balanced' 的原理是什麼？」

**回應：** 它會自動計算每個類別的權重，公式是 $w_c = N / (K \times N_c)$。例如有 950 個負類和 50 個正類，正類的權重 = 1000 / (2 * 50) = 10，負類的權重 = 1000 / (2 * 950) ≈ 0.53。效果等同於讓每個正類樣本在損失函數中「算 10 次」，這樣模型就不會忽視少數類了。

---

## 差異化教學策略 Differentiated Instruction

### 對於學習較快的學生 For Advanced Students
- 鼓勵完成 Part C 進階挑戰題
- 引導思考：邏輯迴歸的決策邊界是線性的，如何不用核方法也能處理非線性問題？（答案：添加多項式特徵或交互作用特徵）
- 介紹 L1 正則化 (Lasso) 在邏輯迴歸中的特徵選擇效果
- 讓他們嘗試在 Notebook Part 2 中加入 L2 正則化項

### 對於需要更多支持的學生 For Students Needing More Support
- Sigmoid 函數：提供數值表格，讓他們手動填入不同 z 值對應的 sigma(z)
- 混淆矩陣：使用具體的數字例子（如 10 個樣本），逐一判斷 TP/TN/FP/FN
- 指標計算：提供公式卡，讓他們對照計算
- 程式碼：先讓他們修改現有程式碼的參數，再嘗試撰寫新程式碼

### 銜接第 6 週 Bridging to Week 6
- 本週建立的「決策邊界」概念將直接延伸到 SVM
- 邏輯迴歸的線性邊界 → SVM 的最大間隔邊界 → 核方法的非線性邊界
- 建議在總結時預留 2 分鐘，用一張圖預覽 SVM 的概念

---

## 教學反思與改進建議 Reflection & Improvement Notes

### 可能的教學瓶頸 Potential Bottlenecks
1. **MLE 推導太數學化**：如果學生數學基礎較弱，可以跳過 MLE 的詳細推導，直接給出 BCE Loss 公式並用直覺解釋
2. **ROC 和 PR 曲線容易混淆**：建議花足夠時間比較兩者的橫縱軸差異，用表格對照
3. **不平衡資料的處理策略太多**：建議先聚焦 `class_weight='balanced'`（最簡單最常用），再選擇性介紹 SMOTE

### 課後評估 Post-Class Assessment
- [ ] 學生是否能正確解釋 Sigmoid 函數的機率意義？
- [ ] 學生是否能區分 Precision 和 Recall？
- [ ] 學生是否理解為什麼不平衡資料中 Accuracy 會誤導？
- [ ] 學生是否能獨立繪製 ROC 曲線並計算 AUC？
- [ ] 學生是否理解閾值選擇與業務需求的關係？

### 時間管理備註 Time Management Notes
- 如果理論部分超時：可以將多類別分類 (Slide 11) 留到下週課前 5 分鐘簡短帶過
- 如果實作部分超時：Part 8 (閾值調整) 和 Part 9 (不平衡資料) 可以選擇其一深入，另一個留作課後自學
- 討論環節至少保留 10 分鐘——學生對閾值選擇的場景討論通常很有收穫

---

## 補充教學素材 Supplementary Materials

### 推薦影片 Recommended Videos
- StatQuest: Logistic Regression (YouTube, ~15 min)
- StatQuest: ROC and AUC (YouTube, ~15 min)
- 3Blue1Brown: Gradient Descent (延續上週主題)

### 進階閱讀 Advanced Reading (給教師)
- Bishop, "Pattern Recognition and Machine Learning," Chapter 4: Linear Models for Classification
- James et al., "An Introduction to Statistical Learning," Chapter 4: Classification
- He & Garcia, "Learning from Imbalanced Data," IEEE TKDE, 2009

### 業界案例 Industry Case Studies
- Google: How YouTube Uses ML for Content Moderation (不平衡資料 + 閾值選擇)
- Netflix: Recommendation vs Classification — When and Why
- 台灣健保署: AI 輔助醫療影像判讀的 Precision/Recall 要求
