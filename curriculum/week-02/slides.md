# 第 2 週投影片：資料視覺化與 EDA（互動圖表）
# Week 2 Slides: Data Visualization & EDA (Interactive Charts)

---

## Slide 1: 本週主題 This Week's Topic

### 資料視覺化與 EDA（互動圖表）
### Data Visualization & EDA (Interactive Charts)

**學習目標 Learning Objectives:**
1. 掌握 Matplotlib / Seaborn / Plotly 三大視覺化工具
2. 了解 EDA（探索式資料分析）的完整流程
3. 能製作互動式圖表並解讀資料分布特性
4. 學會處理缺失值與異常值

> "Without data, you're just another person with an opinion." — W. Edwards Deming

---

## Slide 2: 為什麼視覺化？ Why Visualize?

### Anscombe's Quartet（安斯庫姆四重奏）

四組資料擁有 **完全相同** 的描述統計量：
- 均值 (Mean) X = 9, Y = 7.50
- 標準差 (Std) X = 3.32, Y = 2.03
- 相關係數 (Correlation) = 0.816

**但視覺化後完全不同！**

```
Dataset I:  線性趨勢          Dataset II:  曲線關係
Dataset III: 有一個異常值      Dataset IV:  槓桿點效應
```

**結論：** 永遠不要只看數字，一定要看圖！

---

## Slide 3: Python 三大視覺化工具 Three Visualization Libraries

| | Matplotlib | Seaborn | Plotly |
|---|-----------|---------|--------|
| **角色** | 基礎引擎 | 統計專家 | 互動大師 |
| **語法** | 低階、靈活 | 高階、簡潔 | 高階 + 底層 |
| **互動** | 靜態為主 | 靜態 | 原生互動 |
| **最佳用途** | 細緻客製化 | 快速統計圖 | 儀表板/簡報 |

### 使用建議
- **學術論文 / 報告** → Matplotlib + Seaborn
- **探索分析 / 開發階段** → Seaborn（快速）
- **互動展示 / 儀表板** → Plotly
- **複雜客製化** → Matplotlib（底層控制）

---

## Slide 4: Matplotlib 進階 — 子圖佈局 Subplot Layouts

### Figure-Axes 物件模型

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# axes[row, col] 各自獨立
```

### GridSpec：不等大小子圖
```python
gs = GridSpec(2, 3, figure=fig)
ax_main  = fig.add_subplot(gs[0, :2])   # 寬幅主圖
ax_side  = fig.add_subplot(gs[0, 2])    # 側欄
ax_bottom = fig.add_subplot(gs[1, :])   # 底部全寬
```

### 樣式切換
```python
plt.style.use('seaborn-v0_8-whitegrid')  # 學術風
plt.style.use('ggplot')                   # R 語言風
plt.style.use('dark_background')          # 深色主題
```

**重點：** 使用物件導向 (OO) API 而非 pyplot 全域 API，可獲得更好的控制力。

---

## Slide 5: Seaborn 統計視覺化 Statistical Visualization

### 分布圖家族

| 圖表 | 函式 | 用途 |
|------|------|------|
| 直方圖 | `histplot()` | 頻率分布 |
| KDE 圖 | `kdeplot()` | 平滑密度估計 |
| ECDF 圖 | `ecdfplot()` | 累積分布 |
| 箱型圖 | `boxplot()` | 五數摘要 + 異常值 |
| 小提琴圖 | `violinplot()` | 箱型圖 + KDE 形狀 |
| 蜂群圖 | `swarmplot()` | 每個資料點的位置 |

### 關鍵圖表

```python
# 成對關係一覽
sns.pairplot(df, hue='species', diag_kind='kde')

# 相關係數熱力圖
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# 分組小提琴圖
sns.violinplot(data=df, x='class', y='age', hue='survived', split=True)
```

---

## Slide 6: Plotly 互動圖表 Interactive Charts

### Plotly Express — 一行搞定互動
```python
fig = px.scatter(df, x='age', y='fare', color='survived',
                 size='pclass', hover_data=['name'],
                 title='Titanic 乘客分布')
fig.show()
```

### 互動功能
- **懸停 Hover：** 滑鼠移過顯示詳細資訊
- **縮放 Zoom：** 框選放大局部區域
- **篩選 Filter：** 點擊圖例開關類別
- **下載 Export：** 一鍵儲存為 PNG

### 3D 視覺化
```python
fig = px.scatter_3d(df, x='x1', y='x2', z='x3',
                    color='label')
```

### 動畫 Animation
```python
fig = px.scatter(df, x='gdp', y='life_exp',
                 animation_frame='year', size='pop')
```

---

## Slide 7: EDA 流程四步驟 Four Steps of EDA

```
  +-------------------+
  | Step 1: 資料概覽  |  df.shape, df.info(), df.describe()
  | Data Overview      |  df.isnull().sum()
  +--------+----------+
           |
  +--------v----------+
  | Step 2: 單變數分析 |  每個變數的分布、中心、離散
  | Univariate         |  histplot, countplot
  +--------+----------+
           |
  +--------v----------+
  | Step 3: 多變數分析 |  變數間的關聯、交互
  | Multivariate       |  pairplot, heatmap, boxplot by group
  +--------+----------+
           |
  +--------v----------+
  | Step 4: 洞察摘要  |  發現、假設、下一步行動
  | Insights           |  記錄在筆記中
  +-------------------+
```

**核心精神：** 先看資料，再建模型。讓資料告訴你故事。

---

## Slide 8: 描述統計與分布判讀 Descriptive Statistics

### 描述統計指標

| 面向 | 指標 | Pandas 方法 |
|------|------|------------|
| 集中趨勢 | 均值 Mean、中位數 Median、眾數 Mode | `.mean()`, `.median()`, `.mode()` |
| 離散程度 | 標準差 Std、IQR、全距 Range | `.std()`, `.quantile()` |
| 分布形狀 | 偏態 Skewness、峰態 Kurtosis | `.skew()`, `.kurtosis()` |

### 偏態判讀

| 偏態值 | 意義 | 常見例子 |
|--------|------|---------|
| Skew ≈ 0 | 對稱分布 | 身高 |
| Skew > 0 | 右偏（正偏）| 收入、房價 |
| Skew < 0 | 左偏（負偏）| 考試成績（高分群集中）|

**實務技巧：** 當 Mean > Median 時，通常為右偏分布。

---

## Slide 9: 異常值與缺失值處理 Outliers & Missing Values

### 異常值偵測方法

| 方法 | 原理 | 適用情境 |
|------|------|---------|
| IQR 法 | 超出 Q1-1.5*IQR ~ Q3+1.5*IQR | 通用，不假設分布 |
| Z-score 法 | \|z\| > 3 | 近似常態分布 |
| 箱型圖目視 | 觀察鬍鬚外的點 | 快速初步檢查 |

### 缺失值處理決策樹

```
缺失比例 > 50%？
  → Yes: 考慮刪除該欄位
  → No: 缺失機制是 MCAR？
         → Yes: 可安全刪除或簡單填補
         → No: 使用分組填補或模型填補 (KNN / MICE)
```

### 填補方法

| 方法 | 適用 | 優缺點 |
|------|------|--------|
| 均值/中位數 | 數值 | 簡單但不保留分布形狀 |
| 眾數 | 類別 | 適合低基數類別 |
| 分組填補 | 數值/類別 | 更精準，考慮子群差異 |
| KNN Imputer | 數值 | 利用相似樣本，較準確 |
| MICE | 數值 | 迭代填補，最精準但慢 |

---

## Slide 10: 相關性分析深入 Correlation Analysis

### Pearson vs Spearman

| | Pearson | Spearman |
|---|---------|----------|
| 衡量 | 線性相關 | 單調相關（含非線性） |
| 假設 | 常態分布、連續 | 無分布假設、序數即可 |
| 對異常值 | 敏感 | 穩健 |
| 範圍 | [-1, 1] | [-1, 1] |

### 相關 ≠ 因果 Correlation ≠ Causation

- 冰淇淋銷量 與 溺水人數 高度正相關
- **混淆因子 (Confounding Variable)：** 氣溫！
- EDA 發現相關後，需設計實驗或利用領域知識判斷因果

### 熱力圖解讀技巧
```python
# 只看上三角，避免重複
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, center=0, cmap='RdBu_r')
```

---

## Slide 11: 視覺化最佳實踐 Visualization Best Practices

### 圖表選擇指南

| 你想呈現... | 推薦圖表 |
|-------------|---------|
| 分布的形狀 | 直方圖 + KDE |
| 分布的比較 | 箱型圖 / 小提琴圖 |
| 兩變數關聯 | 散佈圖 |
| 多變數相關 | 熱力圖 |
| 類別計數 | 長條圖 |
| 時間趨勢 | 折線圖 |
| 比例構成 | 堆疊長條圖（避免圓餅圖）|

### 色彩原則

1. **連續數值** → Sequential 色圖（`viridis`, `Blues`）
2. **正負數值** → Diverging 色圖（`RdBu`, `coolwarm`）
3. **無序類別** → Qualitative 色圖（`Set2`, `Paired`）
4. **無障礙** → 避免紅綠配色，使用 `viridis` / `cividis`

### Edward Tufte 原則
- 最大化 **資料墨水比 (Data-Ink Ratio)**
- 移除不必要的裝飾 (Chartjunk)
- 標題描述發現，而非描述資料

---

## Slide 12: 本週實作與作業 Practice & Assignment

### 課堂實作 In-Class Practice

1. **Matplotlib 子圖組合：** 用 GridSpec 建立 EDA 儀表板
2. **Seaborn 統計圖：** pairplot, heatmap, violinplot 實戰
3. **Plotly 互動圖：** 散佈圖 + 3D + 動畫
4. **完整 EDA 流程：** Titanic / Iris 資料集實作

### 本週作業（五部分各 20%）

| 部分 | 內容 | 工具 |
|------|------|------|
| 1 | Seaborn 統計圖表（3 種以上） | Seaborn |
| 2 | Plotly 互動圖表（含 3D） | Plotly |
| 3 | 完整 EDA（四步驟） | 自選 |
| 4 | 缺失值處理實作 | Pandas + sklearn |
| 5 | EDA 發現報告（500字） | Markdown |

### 下週預告 Next Week Preview
**第 3 週：監督式學習概念、資料分割與交叉驗證**
- 訓練集 / 測試集分割 (Train/Test Split)
- k 折交叉驗證 (k-Fold Cross-Validation)
- 過擬合 (Overfitting) 與欠擬合 (Underfitting)
