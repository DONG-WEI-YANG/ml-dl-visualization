# 第 2 週：資料視覺化與 EDA（互動圖表）
# Week 2: Data Visualization & EDA (Interactive Charts)

## 學習目標 Learning Objectives
1. 掌握 Matplotlib 進階繪圖技巧（子圖佈局、樣式自訂、專業出版品質圖表）
2. 使用 Seaborn 進行統計視覺化（分布圖、熱力圖、箱型圖、小提琴圖）
3. 利用 Plotly 製作互動式圖表（散佈圖、3D 圖、動態更新）
4. 理解並實踐探索式資料分析 (Exploratory Data Analysis, EDA) 的完整流程
5. 學會處理缺失值 (Missing Values) 與偵測異常值 (Outliers)
6. 掌握資料視覺化最佳實踐（圖表選擇、色彩學、無障礙設計）

---

## 1. 資料視覺化概論 Introduction to Data Visualization

### 1.1 為什麼視覺化？ Why Visualize?

人類大腦處理視覺資訊的速度比文字快 60,000 倍。在資料科學流程中，視覺化扮演三個關鍵角色：

1. **探索 Exploration：** 在分析初期快速掌握資料的結構、分布與關聯
2. **解釋 Explanation：** 將分析結果以直覺的方式傳達給非技術受眾
3. **驗證 Verification：** 檢視模型假設是否合理、結果是否可信

> "The greatest value of a picture is when it forces us to notice what we never expected to see." — John Tukey

### 1.2 Python 三大視覺化工具比較 Comparison of Python Visualization Libraries

| 特性 | Matplotlib | Seaborn | Plotly |
|------|-----------|---------|--------|
| 定位 | 低階基礎繪圖引擎 | 統計視覺化高階介面 | 互動式圖表 |
| 學習曲線 | 中等 | 較平緩（基於 Matplotlib） | 中等 |
| 互動性 | 基本（需搭配 widget） | 無（靜態圖） | 原生互動（縮放、懸停、篩選） |
| 適用場景 | 細緻客製化、出版品質圖表 | 快速統計圖、學術論文 | 儀表板、報告、簡報 |
| 3D 支援 | 有（mpl_toolkits） | 無 | 原生且高品質 |
| 輸出格式 | PNG, SVG, PDF | PNG, SVG, PDF | HTML, PNG, SVG |

### 1.3 視覺化在資料科學流程中的角色 Role of Visualization in Data Science

```
資料蒐集 → 資料清理 → 【探索式視覺化 EDA】 → 特徵工程 → 建模 → 【結果視覺化】 → 部署
              ↑                                                          |
              └──────────── 反覆迭代 Iterative Process ─────────────────┘
```

---

## 2. Matplotlib 進階 Advanced Matplotlib

### 2.1 Figure 與 Axes 架構 Figure-Axes Architecture

Matplotlib 採用物件導向 (Object-Oriented, OO) 架構。理解 `Figure`（畫布）與 `Axes`（子圖）的關係是進階使用的基礎。

```python
import matplotlib.pyplot as plt
import numpy as np

# 建立畫布與子圖
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 每個 axes 都是獨立的繪圖區域
axes[0, 0].plot(x, y1)
axes[0, 1].scatter(x, y2)
axes[1, 0].bar(categories, values)
axes[1, 1].hist(data, bins=30)

plt.tight_layout()
plt.show()
```

**關鍵概念：**
- `Figure`：整張圖的容器，控制大小、解析度 (DPI)、背景色
- `Axes`：實際的繪圖區域，包含座標軸 (Axis)、標題、圖例 (Legend)
- `Axis`：座標軸物件，控制刻度 (Ticks)、標籤 (Labels)、範圍 (Limits)

### 2.2 子圖佈局技巧 Subplot Layout Techniques

#### GridSpec：不等大小子圖

```python
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 3, figure=fig)

ax_main = fig.add_subplot(gs[0, :2])     # 上方左側，佔兩欄
ax_side = fig.add_subplot(gs[0, 2])       # 上方右側
ax_bottom = fig.add_subplot(gs[1, :])     # 下方，佔整列
```

#### subplot2grid：精細控制

```python
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
```

### 2.3 樣式與主題 Styles and Themes

Matplotlib 內建多種樣式表 (Style Sheets)，可快速切換圖表風格：

```python
# 查看所有可用樣式
print(plt.style.available)

# 常用樣式
plt.style.use('seaborn-v0_8-whitegrid')  # 學術風格
plt.style.use('ggplot')                   # R 語言風格
plt.style.use('dark_background')          # 深色主題
```

#### 自訂 rcParams

```python
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
})
```

### 2.4 注解與標記 Annotations and Markers

```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)

# 標記特定點
ax.annotate('最大值 Max', xy=(x_max, y_max), xytext=(x_max+1, y_max+5),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, color='red', fontweight='bold')

# 添加垂直/水平參考線
ax.axhline(y=mean_val, color='gray', linestyle='--', label=f'平均值 = {mean_val:.2f}')
ax.axvline(x=threshold, color='orange', linestyle=':', label=f'閾值 = {threshold}')

# 陰影區域
ax.fill_between(x, y_lower, y_upper, alpha=0.2, color='blue', label='信賴區間')
```

---

## 3. Seaborn 統計視覺化 Seaborn Statistical Visualization

Seaborn 建構在 Matplotlib 之上，專為統計資料視覺化而設計。它與 Pandas DataFrame 無縫整合，能用簡潔的語法產出專業的統計圖表。

### 3.1 分布圖 Distribution Plots

#### histplot — 直方圖

```python
import seaborn as sns

# 基礎直方圖
sns.histplot(data=df, x='age', bins=30, kde=True)

# 分組直方圖
sns.histplot(data=df, x='age', hue='survived', multiple='stack', palette='Set2')
```

#### kdeplot — 核密度估計圖 (Kernel Density Estimation)

```python
# 單變數 KDE
sns.kdeplot(data=df, x='fare', fill=True, alpha=0.5)

# 雙變數 KDE（等高線圖）
sns.kdeplot(data=df, x='age', y='fare', fill=True, levels=10, cmap='Blues')
```

**核密度估計的原理：** KDE 是一種非參數方法 (Non-parametric Method)，透過在每個資料點放置一個核函數 (Kernel Function)，再加總所有核函數來估計機率密度函數 (Probability Density Function, PDF)。常用的核函數為高斯核 (Gaussian Kernel)。

#### ecdfplot — 經驗累積分布函數圖

```python
sns.ecdfplot(data=df, x='age', hue='survived')
```

ECDF (Empirical Cumulative Distribution Function) 顯示小於或等於特定值的資料點比例，是檢驗分布特性的有力工具。

### 3.2 關聯圖 Relational Plots

#### pairplot — 成對關係圖

```python
# 成對散佈圖矩陣
sns.pairplot(data=df, hue='species', diag_kind='kde',
             plot_kws={'alpha': 0.6}, palette='husl')
```

`pairplot` 一次顯示所有數值變數的兩兩關係，是 EDA 中快速發現變數間相關性的利器。對角線 (Diagonal) 顯示單變數分布，非對角線 (Off-diagonal) 顯示雙變數關聯。

#### heatmap — 熱力圖（相關係數矩陣）

```python
# 計算相關係數矩陣
corr_matrix = df.select_dtypes(include='number').corr()

# 繪製熱力圖
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5)
plt.title('特徵相關係數矩陣 Correlation Matrix')
```

**解讀要點：**
- 相關係數 (Correlation Coefficient) 範圍為 [-1, 1]
- 接近 +1：強正相關 (Strong Positive Correlation)
- 接近 -1：強負相關 (Strong Negative Correlation)
- 接近 0：無線性相關（但可能有非線性關係！）

### 3.3 分類圖 Categorical Plots

#### boxplot — 箱型圖

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 基礎箱型圖
sns.boxplot(data=df, x='class', y='fare', ax=axes[0])
axes[0].set_title('票價分布（依艙等）Fare by Class')

# 分組箱型圖
sns.boxplot(data=df, x='class', y='fare', hue='survived', ax=axes[1])
axes[1].set_title('票價分布（依艙等與存活）')
```

**箱型圖五數摘要 (Five-Number Summary)：**
- 最小值 (Minimum)：Q1 - 1.5 * IQR 以內的最小值
- 第一四分位數 (Q1, 25th Percentile)
- 中位數 (Median, Q2, 50th Percentile)
- 第三四分位數 (Q3, 75th Percentile)
- 最大值 (Maximum)：Q3 + 1.5 * IQR 以內的最大值
- 超出鬍鬚 (Whiskers) 範圍的點標記為異常值 (Outliers)

其中 IQR (Interquartile Range) = Q3 - Q1

#### violinplot — 小提琴圖

```python
sns.violinplot(data=df, x='class', y='age', hue='survived',
               split=True, inner='quart', palette='Set2')
```

小提琴圖結合了箱型圖與核密度估計，能同時展現分布的中心趨勢 (Central Tendency) 與整體形狀 (Shape)。`split=True` 參數可在同一小提琴中對比兩個分組。

#### swarmplot / stripplot — 蜂群圖 / 帶狀圖

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.stripplot(data=df, x='class', y='fare', jitter=True, alpha=0.4, ax=axes[0])
sns.swarmplot(data=df, x='class', y='fare', size=3, ax=axes[1])
```

這些圖表顯示每個資料點的實際位置，適合小型資料集的視覺化。

### 3.4 聯合分布圖 Joint Distribution

```python
sns.jointplot(data=df, x='age', y='fare', kind='hex',
              marginal_kws={'bins': 30})
```

`jointplot` 同時顯示雙變數的聯合分布與各自的邊際分布 (Marginal Distribution)，`kind` 可選 `scatter`、`kde`、`hex`、`reg`、`hist`。

---

## 4. Plotly 互動圖表 Plotly Interactive Charts

Plotly 是一個強大的互動式視覺化函式庫，產出的圖表支援縮放 (Zoom)、平移 (Pan)、懸停顯示 (Hover Tooltip)、框選 (Box Select) 等互動操作。

### 4.1 Plotly Express 快速繪圖

Plotly Express 是 Plotly 的高階介面，語法類似 Seaborn，能用一行程式碼產出互動圖表。

```python
import plotly.express as px

# 互動散佈圖
fig = px.scatter(df, x='sepal_length', y='sepal_width',
                 color='species', size='petal_length',
                 hover_data=['petal_width'],
                 title='鳶尾花特徵散佈圖 Iris Feature Scatter Plot')
fig.show()
```

#### 常用圖表類型

```python
# 長條圖 Bar Chart
fig = px.bar(df_grouped, x='category', y='count', color='group',
             barmode='group', title='分類統計')

# 折線圖 Line Chart
fig = px.line(df_time, x='date', y='value', color='series',
              title='時間序列趨勢 Time Series Trend')

# 直方圖 Histogram
fig = px.histogram(df, x='age', color='survived', nbins=30,
                   marginal='box', title='年齡分布 Age Distribution')

# 箱型圖 Box Plot
fig = px.box(df, x='class', y='fare', color='survived',
             points='outliers', title='票價箱型圖')
```

### 4.2 3D 視覺化 3D Visualization

Plotly 的 3D 圖表支援旋轉、縮放，非常適合多維資料的探索。

```python
# 3D 散佈圖
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_length',
                    color='species', symbol='species',
                    title='鳶尾花 3D 特徵空間 Iris 3D Feature Space')
fig.update_layout(scene=dict(
    xaxis_title='花萼長度',
    yaxis_title='花萼寬度',
    zaxis_title='花瓣長度'
))
fig.show()

# 3D 曲面圖 Surface Plot
fig = px.scatter_3d(...)  # 也可用 go.Surface
```

#### Graph Objects 底層介面

```python
import plotly.graph_objects as go

# 使用 Graph Objects 建立更複雜的圖表
fig = go.Figure()
fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines+markers', name='系列 A'))
fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name='系列 B',
                          line=dict(dash='dash')))
fig.update_layout(
    title='多系列折線圖',
    xaxis_title='X 軸',
    yaxis_title='Y 軸',
    hovermode='x unified'
)
fig.show()
```

### 4.3 動態更新與動畫 Animation

```python
# 動畫散佈圖（依年份變化）
fig = px.scatter(df, x='gdpPercap', y='lifeExp',
                 size='pop', color='continent',
                 animation_frame='year', animation_group='country',
                 size_max=60, range_x=[100, 100000], range_y=[25, 90],
                 log_x=True, title='Gapminder 國家發展動態圖')
fig.show()
```

### 4.4 子圖與多面板 Subplots and Facets

```python
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=2,
                    subplot_titles=('散佈圖', '直方圖', '箱型圖', '小提琴圖'))

fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers'), row=1, col=1)
fig.add_trace(go.Histogram(x=df['x']), row=1, col=2)
fig.add_trace(go.Box(y=df['y']), row=2, col=1)
fig.add_trace(go.Violin(y=df['y']), row=2, col=2)

fig.update_layout(height=700, title_text='多面板儀表板 Multi-Panel Dashboard')
fig.show()
```

#### Facet 分面圖

```python
fig = px.scatter(df, x='sepal_length', y='sepal_width',
                 color='species', facet_col='species',
                 title='分面散佈圖 Faceted Scatter Plot')
fig.show()
```

---

## 5. EDA 流程方法論 EDA Methodology

探索式資料分析 (Exploratory Data Analysis, EDA) 是由統計學家 John Tukey 在 1977 年提出的資料分析方法論。EDA 強調用圖形與統計摘要來理解資料，而非過早進入假設檢定 (Hypothesis Testing)。

### 5.1 EDA 四步驟 Four Steps of EDA

```
Step 1: 資料概覽 Data Overview
    ↓
Step 2: 單變數分析 Univariate Analysis
    ↓
Step 3: 雙變數/多變數分析 Bivariate/Multivariate Analysis
    ↓
Step 4: 洞察摘要與假設生成 Insights & Hypothesis Generation
```

### 5.2 Step 1：資料概覽 Data Overview

```python
import pandas as pd

# 載入資料
df = pd.read_csv('titanic.csv')

# 基本資訊
print(df.shape)              # 列數與欄數 (rows, columns)
print(df.info())             # 欄位型態、非空計數
print(df.describe())         # 數值欄位的描述統計
print(df.describe(include='object'))  # 類別欄位的統計

# 前幾筆資料
print(df.head())

# 缺失值概覽
print(df.isnull().sum())
print(df.isnull().mean() * 100)  # 缺失比例 (%)
```

**描述統計 (Descriptive Statistics) 關鍵指標：**

| 指標 | 英文 | 意義 |
|------|------|------|
| count | Count | 非空值數量 |
| mean | Mean | 算術平均值 |
| std | Standard Deviation | 標準差，衡量離散程度 |
| min | Minimum | 最小值 |
| 25% | Q1 (1st Quartile) | 第一四分位數 |
| 50% | Median (Q2) | 中位數 |
| 75% | Q3 (3rd Quartile) | 第三四分位數 |
| max | Maximum | 最大值 |

### 5.3 Step 2：單變數分析 Univariate Analysis

目的：了解每個變數的分布形狀、中心趨勢、離散程度與異常值。

#### 數值變數 Numerical Variables

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
numerical_cols = df.select_dtypes(include='number').columns

for i, col in enumerate(numerical_cols[:6]):
    row, col_idx = divmod(i, 3)
    sns.histplot(df[col], kde=True, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'{col} 分布')
    # 標記均值與中位數
    axes[row, col_idx].axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
    axes[row, col_idx].axvline(df[col].median(), color='green', linestyle=':', label='Median')
    axes[row, col_idx].legend()

plt.tight_layout()
```

**分布形狀判讀：**
- **偏態 (Skewness)：** 衡量分布的對稱性。正偏 (Right-skewed) 表示右尾較長（如收入分布）；負偏 (Left-skewed) 表示左尾較長
- **峰態 (Kurtosis)：** 衡量分布的尖峭程度。高峰態表示有較多極端值

```python
from scipy import stats

for col in numerical_cols:
    skew = df[col].skew()
    kurt = df[col].kurtosis()
    print(f'{col}: Skewness={skew:.3f}, Kurtosis={kurt:.3f}')
```

#### 類別變數 Categorical Variables

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
categorical_cols = df.select_dtypes(include='object').columns

for i, col in enumerate(categorical_cols[:3]):
    value_counts = df[col].value_counts()
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
    axes[i].set_title(f'{col} 分布')
    # 加上百分比標記
    total = len(df)
    for j, v in enumerate(value_counts.values):
        axes[i].text(j, v + 0.5, f'{v/total*100:.1f}%', ha='center')
```

### 5.4 Step 3：雙變數/多變數分析 Bivariate/Multivariate Analysis

#### 數值 vs 數值：相關性分析 Correlation Analysis

```python
# 相關係數矩陣
corr = df.select_dtypes(include='number').corr()

# 只顯示上三角 (Upper Triangle)
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
```

#### 數值 vs 類別：分組比較 Group Comparison

```python
# 分組箱型圖 + 小提琴圖
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(data=df, x='survived', y='age', ax=axes[0])
sns.violinplot(data=df, x='survived', y='age', ax=axes[1])
```

#### 類別 vs 類別：交叉表 Cross-tabulation

```python
# 交叉表
ct = pd.crosstab(df['sex'], df['survived'], normalize='index')
ct.plot(kind='bar', stacked=True, figsize=(8, 5))
plt.title('性別與存活率 Gender vs Survival Rate')
plt.ylabel('比例 Proportion')
```

### 5.5 Step 4：洞察摘要 Insights Summary

EDA 的最終產出是一份洞察清單，作為後續建模的依據：

1. **資料品質問題 Data Quality Issues：** 缺失欄位、異常值、資料型態錯誤
2. **變數特性 Variable Characteristics：** 分布形狀、離散程度、類別基數 (Cardinality)
3. **變數間關係 Relationships：** 強相關的變數對、潛在的交互效果 (Interaction Effects)
4. **初步假設 Hypotheses：** 可能影響目標變數的重要特徵

---

## 6. 異常值偵測 Outlier Detection

### 6.1 IQR 方法

```python
def detect_outliers_iqr(df, column):
    """使用 IQR 方法偵測異常值"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound
```

### 6.2 Z-score 方法

```python
from scipy import stats

def detect_outliers_zscore(df, column, threshold=3):
    """使用 Z-score 方法偵測異常值（超過 threshold 個標準差）"""
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outlier_mask = z_scores > threshold
    return df[column].dropna()[outlier_mask]
```

### 6.3 視覺化異常值

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 箱型圖
sns.boxplot(data=df, y='fare', ax=axes[0])
axes[0].set_title('箱型圖 Box Plot')

# 直方圖 + KDE
sns.histplot(df['fare'], kde=True, ax=axes[1])
axes[1].set_title('直方圖 Histogram')

# 散佈圖（按索引）
axes[2].scatter(range(len(df)), df['fare'], alpha=0.5, s=10)
axes[2].set_title('資料點分布 Data Point Distribution')
axes[2].set_xlabel('Index')
axes[2].set_ylabel('Fare')
```

### 6.4 異常值處理策略 Outlier Treatment Strategies

| 策略 | 英文 | 適用情境 |
|------|------|---------|
| 移除 | Removal | 確認為資料錯誤 |
| 截尾 | Winsorization / Capping | 保留極端值但限制影響 |
| 對數轉換 | Log Transformation | 右偏分布 |
| 分箱 | Binning | 將連續變數離散化 |
| 保留 | Keep as is | 極端值有業務意義 |

```python
# 截尾處理範例
lower = df['fare'].quantile(0.01)
upper = df['fare'].quantile(0.99)
df['fare_capped'] = df['fare'].clip(lower, upper)
```

---

## 7. 缺失值處理 Missing Values Handling

### 7.1 缺失值類型 Types of Missing Data

了解缺失機制 (Missing Data Mechanism) 對選擇正確的處理策略至關重要：

| 類型 | 英文 | 說明 | 範例 |
|------|------|------|------|
| 完全隨機缺失 | Missing Completely At Random (MCAR) | 缺失與任何變數無關 | 問卷因紙張印刷缺漏 |
| 隨機缺失 | Missing At Random (MAR) | 缺失與其他觀察到的變數有關 | 收入缺失與年齡有關 |
| 非隨機缺失 | Missing Not At Random (MNAR) | 缺失與缺失值本身有關 | 高收入者傾向不填收入 |

### 7.2 缺失值視覺化

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 缺失值矩陣
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('缺失值矩陣 Missing Value Matrix')

# 缺失值比例長條圖
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
plt.figure(figsize=(10, 5))
missing_pct.plot(kind='bar')
plt.title('各欄位缺失比例 Missing Value Percentage by Column')
plt.ylabel('缺失比例 (%)')
```

### 7.3 處理策略 Handling Strategies

#### 刪除法 Deletion

```python
# 刪除含缺失值的列 (Listwise Deletion)
df_dropped = df.dropna()

# 刪除缺失比例超過 50% 的欄位
threshold = 0.5
cols_to_drop = df.columns[df.isnull().mean() > threshold]
df_cleaned = df.drop(columns=cols_to_drop)
```

#### 填補法 Imputation

```python
# 數值欄位：均值/中位數填補
df['age'].fillna(df['age'].median(), inplace=True)

# 類別欄位：眾數填補
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# 前向/後向填補（時間序列適用）
df['value'].fillna(method='ffill', inplace=True)  # Forward Fill
df['value'].fillna(method='bfill', inplace=True)  # Backward Fill

# 分組填補（更精確）
df['age'] = df.groupby('class')['age'].transform(lambda x: x.fillna(x.median()))
```

#### 進階填補 Advanced Imputation

```python
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# KNN 填補
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(knn_imputer.fit_transform(df_numerical),
                       columns=df_numerical.columns)

# 迭代填補 (MICE: Multiple Imputation by Chained Equations)
mice_imputer = IterativeImputer(max_iter=10, random_state=42)
df_mice = pd.DataFrame(mice_imputer.fit_transform(df_numerical),
                        columns=df_numerical.columns)
```

### 7.4 填補前後比較

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 填補前
sns.histplot(df_original['age'].dropna(), kde=True, ax=axes[0], color='blue', label='原始')
axes[0].set_title('填補前 Before Imputation')

# 填補後
sns.histplot(df_imputed['age'], kde=True, ax=axes[1], color='green', label='填補後')
sns.histplot(df_original['age'].dropna(), kde=True, ax=axes[1], color='blue', alpha=0.3, label='原始')
axes[1].set_title('填補後 After Imputation')
axes[1].legend()
```

---

## 8. 資料視覺化最佳實踐 Data Visualization Best Practices

### 8.1 選擇合適的圖表 Choosing the Right Chart

| 目的 | 推薦圖表 | 說明 |
|------|---------|------|
| 分布 Distribution | 直方圖、KDE、箱型圖、小提琴圖 | 了解資料的形狀與離散程度 |
| 比較 Comparison | 長條圖、群組箱型圖 | 跨類別的數值比較 |
| 關聯 Relationship | 散佈圖、熱力圖、氣泡圖 | 變數之間的相關性 |
| 趨勢 Trend | 折線圖、面積圖 | 隨時間變化的模式 |
| 比例 Proportion | 圓餅圖、堆疊長條圖 | 各部分佔整體的比例 |
| 組成 Composition | 堆疊面積圖、樹狀圖 | 整體的組成結構 |

### 8.2 色彩學基礎 Color Theory

#### 色彩尺度類型 Color Scale Types

| 類型 | 英文 | 適用情境 | 範例 Colormap |
|------|------|---------|--------------|
| 連續型 | Sequential | 從低到高的數值 | `viridis`, `plasma`, `Blues` |
| 發散型 | Diverging | 有中心點的數值（如正負值） | `RdBu`, `coolwarm`, `BrBG` |
| 類別型 | Qualitative | 無序類別 | `Set2`, `Paired`, `husl` |

#### 色彩選擇原則

1. **一致性 Consistency：** 同一份報告中，相同類別使用相同顏色
2. **對比度 Contrast：** 確保相鄰顏色有足夠的視覺區分度
3. **直覺性 Intuitiveness：** 紅色暗示警告/負面，綠色暗示正常/正面
4. **有限使用 Restraint：** 單張圖中不超過 7 種顏色

### 8.3 無障礙設計 Accessibility

全球約 8% 的男性與 0.5% 的女性有色覺異常 (Color Vision Deficiency)。設計圖表時應考慮：

1. **避免紅綠配色：** 改用藍橘配色 (Blue-Orange) 或藍紅配色
2. **使用形狀區分：** 散佈圖用不同標記符號 (Markers) 區分類別
3. **加入文字標籤：** 不要只靠顏色傳遞資訊
4. **選擇色覺友善色圖：** `viridis`, `cividis`, `inferno` 對色覺異常者友善

```python
# 色覺友善配色範例
colorblind_palette = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']
sns.set_palette(colorblind_palette)
```

### 8.4 圖表設計原則 Design Principles

1. **資料墨水比 (Data-Ink Ratio)：** 由 Edward Tufte 提出，應最大化用於呈現資料的墨水比例，移除不必要的裝飾 (Chartjunk)
2. **標題明確：** 標題應描述圖表的主要發現，而非僅描述資料
3. **適當的座標軸：** Y 軸通常應從 0 開始（長條圖），避免截斷誤導
4. **圖例位置：** 放在不遮擋資料的位置，或直接在資料旁標註
5. **字體大小：** 確保縮小後仍可閱讀，簡報用圖字體應加大

---

## 9. 完整 EDA 實戰範例 Complete EDA Example

以 Titanic 資料集為例，展示完整的 EDA 流程：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# === Step 1: 資料載入與概覽 ===
df = pd.read_csv('titanic.csv')
print(f"資料形狀: {df.shape}")
print(f"\n欄位資訊:")
print(df.info())
print(f"\n描述統計:")
print(df.describe())
print(f"\n缺失值:")
print(df.isnull().sum())

# === Step 2: 單變數分析 ===
# 數值變數
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['age'].dropna(), kde=True, ax=axes[0])
sns.histplot(df['fare'], kde=True, ax=axes[1])
sns.countplot(data=df, x='survived', ax=axes[2])

# === Step 3: 雙變數分析 ===
# 存活率 vs 各因素
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.barplot(data=df, x='sex', y='survived', ax=axes[0, 0])
sns.barplot(data=df, x='pclass', y='survived', ax=axes[0, 1])
sns.boxplot(data=df, x='survived', y='age', ax=axes[1, 0])
sns.boxplot(data=df, x='survived', y='fare', ax=axes[1, 1])

# 相關性分析
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)

# === Step 4: 洞察摘要 ===
# 1. 女性存活率遠高於男性
# 2. 頭等艙乘客存活率最高
# 3. 票價與存活率正相關
# 4. Age 欄位約 20% 缺失，需處理
```

---

## 關鍵詞彙 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 探索式資料分析 | Exploratory Data Analysis (EDA) | 用圖形與統計摘要系統性地理解資料 |
| 描述統計 | Descriptive Statistics | 以數值摘要描述資料特性（均值、標準差、四分位數等） |
| 直方圖 | Histogram | 以長條顯示數值變數的頻率分布 |
| 核密度估計 | Kernel Density Estimation (KDE) | 以平滑曲線估計機率密度函數 |
| 箱型圖 | Box Plot | 以五數摘要與異常值呈現分布特性 |
| 小提琴圖 | Violin Plot | 結合箱型圖與 KDE 的分布圖 |
| 熱力圖 | Heatmap | 以顏色深淺呈現矩陣數值的大小 |
| 散佈圖 | Scatter Plot | 以點的位置呈現兩個數值變數的關係 |
| 相關係數 | Correlation Coefficient | 衡量兩變數線性相關程度的統計量，範圍 [-1, 1] |
| 偏態 | Skewness | 衡量分布不對稱程度的統計量 |
| 峰態 | Kurtosis | 衡量分布尾部厚度的統計量 |
| 異常值 | Outlier | 偏離大多數資料點的極端值 |
| 四分位距 | Interquartile Range (IQR) | Q3 - Q1，衡量中間 50% 資料的離散程度 |
| 缺失值 | Missing Value | 資料中的空值或未記錄值 |
| 填補 | Imputation | 以估計值填入缺失值的方法 |
| 子圖 | Subplot | 在同一畫布中排列多張圖的佈局方式 |
| 圖例 | Legend | 說明圖表中各資料系列代表意義的標註 |
| 色圖 | Colormap | 將數值對應到顏色的映射函數 |
| 無障礙設計 | Accessibility | 確保所有使用者（含色覺異常者）都能理解的設計 |
| 資料墨水比 | Data-Ink Ratio | 用於呈現資料的墨水占總墨水的比例（Tufte 概念） |

---

## 延伸閱讀 Further Reading

- Edward Tufte, *The Visual Display of Quantitative Information*（資訊視覺化經典著作）
- Claus Wilke, *Fundamentals of Data Visualization*（免費線上版：clauswilke.com/dataviz）
- Matplotlib 官方教學：https://matplotlib.org/stable/tutorials/
- Seaborn 官方教學：https://seaborn.pydata.org/tutorial.html
- Plotly Python 文件：https://plotly.com/python/
- Python Data Science Handbook — Chapter 4: Visualization with Matplotlib
- Kaggle Titanic EDA 案例：https://www.kaggle.com/competitions/titanic
