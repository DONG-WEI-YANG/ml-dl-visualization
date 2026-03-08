# 第 2 週教師手冊：資料視覺化與 EDA（互動圖表）
# Week 2 Teacher Guide: Data Visualization & EDA (Interactive Charts)

---

## 課程資訊 Course Info

- **主題：** 資料視覺化與 EDA（互動圖表）
- **難度等級：** 核心 Core
- **先備知識：** Week 1 Python 環境已建置完成，學生能操作 Jupyter Notebook
- **所需環境：** Python 3.11+、Jupyter Notebook/Lab、matplotlib、seaborn、plotly、pandas、numpy、scikit-learn

---

## 時間分配（90 分鐘）

| 時段 | 分鐘 | 活動 | 教學方式 | 說明 |
|------|------|------|---------|------|
| 開場 | 0-5 | 課程回顧與本週預告 | 講述 | 回顧 Week 1 環境建置，確認全班環境就緒。預告本週重點：三大視覺化工具 + EDA 流程 |
| 前段 | 5-20 | EDA 方法論與視覺化原則 | 投影片 + 講述 | 使用 Slides 2-4。以 Anscombe's Quartet 引入「為什麼要視覺化」，介紹三大工具定位與選擇策略 |
| 中段-1 | 20-35 | Matplotlib 進階 Demo | 即時演示 + 跟做 | 開啟 notebook.ipynb，帶領學生操作子圖佈局 (subplots, GridSpec)、樣式切換、注解。重點：物件導向 API |
| 中段-2 | 35-50 | Seaborn 統計圖表 Demo | 即時演示 + 跟做 | pairplot, heatmap, violinplot 三大圖表。使用 Iris 或 Titanic 資料集。強調「一行程式碼 = 一張專業圖」的優勢 |
| 中段-3 | 50-65 | Plotly 互動圖表 Demo | 即時演示 + 互動體驗 | 散佈圖互動操作（懸停、縮放、篩選）、3D 散佈圖旋轉、動畫。讓學生親手操作互動功能 |
| 後段-1 | 65-75 | 完整 EDA 流程示範 | 即時演示 | 以 Titanic 資料集走過四步驟 EDA，同步說明缺失值偵測與處理策略 |
| 後段-2 | 75-85 | 練習時間 + 討論 | 學生實作 + 巡場 | 學生自行完成 Notebook 練習題，教師與助教巡場協助。鼓勵學生使用 AI 助教提問 |
| 收尾 | 85-90 | 作業說明 + Q&A | 講述 + 對話 | 說明本週作業要求（五部分），回答學生問題，預告下週主題 |

---

## 教學重點 Teaching Focus

### 核心概念（必須涵蓋）

1. **Matplotlib 物件模型：** 學生需理解 Figure → Axes → Axis 的層次結構，這是後續所有視覺化的基礎。強調使用 `fig, ax = plt.subplots()` 而非 `plt.plot()` 全域 API。

2. **Seaborn 與 DataFrame 的整合：** Seaborn 的威力在於直接接受 DataFrame 的欄位名稱作為參數。示範 `data=df, x='col1', y='col2', hue='category'` 的統一語法。

3. **Plotly 的互動價值：** 不僅是「好看」，互動功能能讓你在 EDA 階段更快發現異常值和資料模式。讓學生實際體驗懸停查看個別資料點的細節。

4. **EDA 是一個流程，不是隨意畫圖：** 強調四步驟的邏輯：先看全貌 → 再看個體 → 再看關聯 → 最後總結。很多學生的問題是「不知道該從哪裡開始分析」。

5. **缺失值不是簡單刪除或填 0：** 需要理解缺失機制 (MCAR/MAR/MNAR)，不同機制對應不同處理策略。這個概念初學者容易忽略。

### 重要提醒（常被忽略的點）

- **相關 ≠ 因果：** 在展示 heatmap 時務必提醒學生，高相關不代表因果關係。舉「冰淇淋銷量 vs 溺水人數」的經典例子。
- **色覺無障礙：** 提醒學生 8% 男性有色覺異常，避免紅綠配色。推薦 `viridis` 色圖。
- **圖表標題應描述發現：** 好的標題是「女性存活率 (74%) 遠高於男性 (19%)」，而非「性別與存活率的關係」。

---

## 檢核點 Checkpoints

### 環境檢核（課堂開始時，5 分鐘內完成）

- [ ] 全班學生能成功執行 `import matplotlib.pyplot as plt; import seaborn as sns; import plotly.express as px`
- [ ] Plotly 圖表能在 Jupyter Notebook 中正常顯示（若使用 JupyterLab 可能需要安裝 plotly extension）
- [ ] 學生能載入範例資料集（`sns.load_dataset('iris')` 或 `sns.load_dataset('titanic')`）

### 概念檢核（講述段後）

- [ ] 學生能說出 Matplotlib / Seaborn / Plotly 三者的定位差異
- [ ] 學生能解釋「為什麼相同的描述統計量可能對應完全不同的資料分布」（Anscombe's Quartet）
- [ ] 學生理解 EDA 四步驟的邏輯順序

### 實作檢核（實作段中，巡場時確認）

- [ ] 學生能用 `plt.subplots()` 建立至少 2x2 的子圖佈局
- [ ] 學生能用 `sns.pairplot()` 繪製成對關係圖，並正確使用 `hue` 參數
- [ ] 學生能用 `sns.heatmap()` 繪製相關係數矩陣
- [ ] 學生能用 `px.scatter()` 建立互動散佈圖，並體驗懸停、縮放功能
- [ ] 學生能對資料集呼叫 `df.describe()` 並解釋輸出中至少 3 個指標的意義

### 綜合檢核（課堂結束前）

- [ ] 學生理解缺失值的三種機制（MCAR / MAR / MNAR）的基本定義
- [ ] 學生能說出至少 2 種缺失值處理方法及其適用情境
- [ ] 學生清楚本週作業的五部分要求

---

## AI 助教引導策略 AI Tutor Guidance Strategy

### 本週助教設定

本週 AI 助教的角色是**視覺化顧問 + EDA 引導者**。學生在學習初期常遇到的困難是「不知道該用什麼圖表」和「不知道從 EDA 的哪裡開始」。

### 提示層級策略 (Hint Ladder)

#### 問題類型 1：「不知道該用什麼圖表」

| 層級 | 助教回應策略 |
|------|-------------|
| Level 1 (釐清) | 「你想呈現的資訊是什麼？是分布、比較、趨勢、還是關聯？你的變數是數值型還是類別型？」 |
| Level 2 (概念) | 「對於兩個數值變數的關聯，散佈圖 (scatter plot) 是最直覺的選擇。如果你想看分布，可以考慮直方圖或箱型圖。」 |
| Level 3 (步驟) | 「你可以用 Seaborn 的 `pairplot` 一次看所有數值變數的關係，再針對有趣的變數對深入分析。」 |
| Level 4 (範例) | 提供一行範例程式碼，但只給骨架，讓學生自己填入欄位名稱。 |

#### 問題類型 2：「Plotly 圖表不顯示」

| 層級 | 助教回應策略 |
|------|-------------|
| Level 1 (釐清) | 「你使用的是 Jupyter Notebook 還是 JupyterLab？錯誤訊息是什麼？」 |
| Level 2 (概念) | 「Plotly 在不同的 Jupyter 環境中可能需要不同的渲染器 (renderer)。」 |
| Level 3 (步驟) | 「試試在 Notebook 開頭加入 `import plotly.io as pio; pio.renderers.default = 'notebook'`。如果是 JupyterLab，可能需要安裝 jupyterlab-plotly 擴充。」 |

#### 問題類型 3：「EDA 不知道該分析什麼」

| 層級 | 助教回應策略 |
|------|-------------|
| Level 1 (釐清) | 「你已經做了哪些步驟？目前看到什麼？有沒有什麼讓你意外的發現？」 |
| Level 2 (概念) | 「EDA 的核心是回答幾個基本問題：資料長什麼樣？有沒有缺失？哪些變數可能相關？有沒有異常值？」 |
| Level 3 (步驟) | 「建議你先跑 `df.describe()` 和 `df.isnull().sum()`，然後對每個數值欄位畫直方圖，看看分布是否有偏態或異常。」 |

#### 問題類型 4：「缺失值該怎麼處理」

| 層級 | 助教回應策略 |
|------|-------------|
| Level 1 (釐清) | 「哪些欄位有缺失值？缺失的比例大約是多少？」 |
| Level 2 (概念) | 「缺失值處理的第一步是判斷缺失機制。如果缺失是完全隨機的 (MCAR)，簡單刪除通常沒問題。但如果缺失與其他變數相關 (MAR)，就需要更精細的填補方法。」 |
| Level 3 (步驟) | 「試試看用 `df.groupby('class')['age'].transform(lambda x: x.fillna(x.median()))` 進行分組填補，比全體中位數填補更合理。」 |

### 作業模式特別注意

- 如果學生直接問「幫我做 EDA」或「幫我寫報告」，助教應回覆：「請先告訴我你目前做到哪裡了，以及你觀察到什麼。我可以幫你確認方向是否正確。」
- 鼓勵學生用自己的話描述圖表中看到的現象，再由助教確認或補充。

---

## 常見問題排除 Troubleshooting

### 1. Plotly 圖表在 Jupyter Notebook 中不顯示

**症狀：** 執行 `fig.show()` 後無任何輸出，或顯示空白區域。

**解法：**
```python
# 方法 1：設定預設渲染器
import plotly.io as pio
pio.renderers.default = 'notebook'

# 方法 2：使用 iframe 渲染器（Notebook 相容性較好）
pio.renderers.default = 'iframe'

# 方法 3：如果使用 JupyterLab
# 需安裝擴充：pip install jupyterlab-plotly

# 方法 4：Google Colab 通常無需額外設定
```

### 2. Seaborn 版本不相容

**症狀：** `sns.histplot` 或 `sns.kdeplot` 報 AttributeError。

**解法：**
```bash
# 確認 Seaborn 版本 >= 0.11
pip install seaborn --upgrade
```
如果學生使用舊版 Seaborn（< 0.11），`histplot` 不存在，需改用 `distplot`（已棄用）。建議統一升級。

### 3. 中文字體顯示為方塊

**症狀：** Matplotlib 圖表中的中文標題或標籤顯示為空白方塊。

**解法：**
```python
# Windows
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# macOS
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# Linux
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC']

# 通用：安裝完後可能需要清除 Matplotlib 快取
# 刪除 ~/.matplotlib/fontlist-v330.json（路徑因版本而異）
```

### 4. pairplot 執行太久

**症狀：** `sns.pairplot` 在大型資料集上執行超過 30 秒。

**解法：**
```python
# 方法 1：抽樣
sns.pairplot(df.sample(500), hue='target')

# 方法 2：只選部分欄位
sns.pairplot(df[['col1', 'col2', 'col3', 'target']], hue='target')

# 方法 3：對角線不用 KDE（KDE 較慢）
sns.pairplot(df, diag_kind='hist')
```

### 5. IterativeImputer 無法 import

**症狀：** `from sklearn.impute import IterativeImputer` 失敗。

**解法：**
```python
# IterativeImputer 仍在實驗階段，需先啟用
from sklearn.experimental import enable_iterative_imputer  # 必須先 import 這行
from sklearn.impute import IterativeImputer
```

### 6. 學生搞混 Pandas plot 與 Matplotlib/Seaborn

**症狀：** 學生混用 `df.plot()` 和 `plt.xxx()`，圖表行為不如預期。

**解法：** 說明 `df.plot()` 底層就是 Matplotlib，但參數命名方式不同。建議在本課程中統一使用 Seaborn 或 Matplotlib 的 OO API，避免混淆。

### 7. Plotly 圖表檔案過大

**症狀：** 包含 Plotly 圖表的 Notebook 檔案達到數十 MB。

**解法：**
```python
# 儲存前清除 Plotly 輸出
# 在 Jupyter 中：Cell > All Output > Clear

# 或在程式中控制點數
fig = px.scatter(df.sample(1000), ...)  # 抽樣減少資料量
```

---

## 教學建議 Teaching Tips

### 開場引導

建議用 Anscombe's Quartet 或 Datasaurus Dozen 作為開場引入。這個例子能立即讓學生理解「光看數字不夠，必須看圖」的重要性。可以先給學生四組描述統計量，問他們「這四組資料一樣嗎？」，然後才揭曉視覺化結果。

### 即時演示策略

- **不要用預寫好的程式碼直接執行。** 建議從空白 Cell 開始，邊講邊打程式碼。學生更能跟上思路。
- **刻意犯錯。** 例如忘記設定 `hue`，讓學生看到沒有分組的圖表，然後加上 `hue` 後對比效果。
- **讓學生選擇。** 例如問：「接下來我們想看 age 的分布，你覺得該用什麼圖？」讓學生參與決策。

### 分組活動建議

如果時間允許（可在後段的練習時間中執行），可以：
1. 每組分配不同的資料集（Titanic, Tips, Diamonds, Penguins）
2. 各組用 10 分鐘做 Step 1-2 的 EDA
3. 各組分享 1 個最有趣的發現
4. 全班討論：不同資料集的 EDA 策略有何異同？

### 彈性調整

- **如果學生基礎較弱：** 減少 Plotly 3D/動畫的演示，把時間留給 Matplotlib 基礎和 Seaborn。確保大家至少能畫 histplot + heatmap。
- **如果學生基礎較強：** 加入 Plotly Dash 的簡介（只需 5 分鐘），或展示 missingno 套件的缺失值視覺化。
- **如果環境問題太多：** 切換到 Google Colab，確保所有人在同一環境下操作。

---

## 與課程平台的整合 Platform Integration

### 視覺化互動區

本週的課程平台可提供：
- **EDA 互動面板：** 學生上傳 CSV 後自動產出 describe()、缺失值摘要、correlation heatmap
- **圖表選擇器：** 輸入變數類型，推薦適合的圖表類型
- **Plotly 即時預覽：** 在平台上調整參數即時更新圖表

### AI 助教整合

- 本週助教的系統提示 (System Prompt) 應聚焦於視覺化建議與 EDA 方法論
- 建議在作業模式下，要求學生先描述「我看到了什麼」再由助教確認
- 追蹤學生向助教詢問最多的問題類型，作為下次上課的補充教學素材

---

## 下週銜接 Bridge to Next Week

在課堂結尾提醒學生：
- 本週學到的 EDA 技巧在整學期都會用到
- 下週（Week 3）將進入監督式學習，會用到本週的資料視覺化能力來觀察訓練/測試集的分布差異
- 作業中的 EDA 流程就是未來每次建模前的「標準前置作業」
