# 第 9 週教師手冊
# Week 9 Teacher Guide

## 時間分配 Time Allocation（90 分鐘）

| 時段 | 分鐘 | 活動 | 說明 |
|------|:---:|------|------|
| 回顧 | 5 | Week 8 回顧 + 本週預告 | 從特徵重要度銜接特徵工程 |
| 理論一 | 15 | 數值縮放 + 類別編碼 | Slide 4-10，重點講解三種 Scaler 差異 |
| Demo 一 | 10 | Scaler 效果互動視覺化 | 現場展示含離群值資料的 Scaler 差異 |
| 理論二 | 10 | 缺失值處理 + 特徵選擇 | Slide 11-15，強調選擇策略 |
| 實作 | 30 | Pipeline 建構 + Notebook | 學生動手建構 ColumnTransformer + Pipeline |
| 理論三 | 5 | PCA + 多項式特徵 | Slide 21-23，概念介紹 |
| Demo 二 | 5 | PCA 2D 視覺化 | 展示 Iris 資料集的 PCA 投影 |
| 討論 | 5 | 期中專題方向 | 引導學生討論可用的資料集與方向 |
| 總結 | 5 | 回顧 + 作業說明 | Slide 26-28 |

## 教學重點 Key Points

### 核心概念
1. **資料洩漏 (Data Leakage) 是本週最重要的概念：** 反覆強調「先 split 再 fit」，Pipeline 自動防止洩漏
2. **Scaler 的選擇取決於資料分布和模型類型：** 不是所有模型都需要縮放（樹模型不需要）
3. **編碼方法的選擇取決於類別的性質和基數：** 名義 vs. 有序、低基數 vs. 高基數
4. **Pipeline 是工業級 ML 的標準做法：** 強調可重現性與部署便利性

### 教學銜接
- **承接 Week 8：** 特徵重要度 → 特徵選擇（自然延伸）
- **銜接 Week 10：** Pipeline + GridSearchCV → 超參數調校（本週建立的 Pipeline 是下週調參的基礎）
- **期中專題：** 本週是期中專題提案的時間點，引導學生思考如何在自己的專題中運用 Pipeline

### 常見迷思 Common Misconceptions
1. **迷思：** 「所有特徵都需要標準化」
   → **澄清：** 樹模型不受特徵尺度影響
2. **迷思：** 「One-Hot Encoding 總是最好的」
   → **澄清：** 高基數時會導致維度爆炸，Target Encoding 可能更適合
3. **迷思：** 「缺失值直接刪除就好」
   → **澄清：** 刪除可能引入樣本偏差，且損失大量資料
4. **迷思：** 「PCA 能找到最重要的特徵」
   → **澄清：** PCA 找的是最大變異方向，不等於最重要的特徵；主成分是原始特徵的線性組合，不直接對應原始特徵

## 檢核點 Checkpoints
- [ ] 學生能解釋三種 Scaler 的差異及適用情境
- [ ] 學生能正確選擇編碼方法（名義 vs. 有序）
- [ ] 學生能識別資料洩漏的錯誤用法
- [ ] 學生成功建構 ColumnTransformer + Pipeline
- [ ] 學生能在 Pipeline 中使用 GridSearchCV
- [ ] 學生能繪製 PCA Scree Plot 並解讀
- [ ] 學生提出期中專題的初步方向

## AI 助教設定 AI Tutor Configuration

本週助教設定為「引導模式」：
- 當學生問 Scaler 選擇時，先反問資料的分布特性和使用的模型
- 當學生遇到 Pipeline 語法錯誤時，引導檢查參數命名規則（步驟名__參數名）
- 對於資料洩漏相關問題，先請學生解釋 fit 和 transform 的差異
- 期中專題諮詢：幫助學生評估資料集可行性，但不直接建議主題

### 分層提示策略示例

**學生問：** 「Pipeline + GridSearchCV 參數名怎麼寫？」

| 層級 | 提示內容 |
|------|---------|
| Level 1 | Pipeline 的參數命名有一個特殊的格式，和步驟名稱有關。你能先告訴我你的 Pipeline 長什麼樣嗎？ |
| Level 2 | 參數名稱的格式是「步驟名__參數名」，使用雙下底線連接。如果有巢狀 Pipeline，需要逐層展開。 |
| Level 3 | 例如：如果前處理步驟叫 'preprocessor'，裡面的數值管線叫 'num'，填補器叫 'imputer'，那策略參數就是 'preprocessor__num__imputer__strategy'。 |
| Level 4 | 完整範例：`{'preprocessor__num__imputer__strategy': ['mean', 'median']}` |

## 實作引導 Lab Guidance

### 引導順序（建議學生按此順序完成 Notebook）

1. **先載入資料並做 EDA（5 分鐘）**
   - 使用 `df.info()`, `df.describe()`, `df.isnull().sum()` 了解資料
   - 區分數值與類別欄位

2. **分別實驗 Scaler（10 分鐘）**
   - 先用簡單資料觀察三種 Scaler 的效果
   - 加入離群值再觀察

3. **建構 ColumnTransformer（10 分鐘）**
   - 先分別定義數值管線和類別管線
   - 再用 ColumnTransformer 組合

4. **組裝完整 Pipeline（5 分鐘）**
   - 將 ColumnTransformer 與模型串接
   - 執行 fit + predict + score

5. **加上 GridSearchCV（5 分鐘，進階學生）**
   - 定義參數空間
   - 查看最佳參數

### 預期困難與解法

| 困難 | 解法 |
|------|------|
| ColumnTransformer 欄位名稱寫錯 | 先用 `df.select_dtypes()` 確認 |
| GridSearchCV 參數名報錯 | 用 `pipe.get_params().keys()` 查看所有可用參數 |
| OneHotEncoder 遇到未知類別 | 設定 `handle_unknown='ignore'` |
| PCA 前忘記標準化 | 強調 PCA 對尺度敏感，必須先標準化 |
| Pipeline 中 sparse 矩陣與 dense 矩陣混用 | OneHotEncoder 加 `sparse_output=False` |

## 常見問題與排除 Troubleshooting

### Q1: ColumnTransformer 輸出形狀不對
```python
# 檢查每個轉換器的輸出
preprocessor.fit(X_train)
print(preprocessor.get_feature_names_out())
```

### Q2: GridSearchCV 跑太久
- 減少參數空間
- 使用 `n_jobs=-1` 平行運算
- 先用 `RandomizedSearchCV` 縮小範圍

### Q3: 類別欄位的 dtype 不對
```python
# 確保類別欄位為 string 或 category
df['embarked'] = df['embarked'].astype(str)
```

### Q4: Pipeline 中存取中間步驟的結果
```python
# 存取 Pipeline 中特定步驟
preprocessor = full_pipeline.named_steps['preprocessor']
X_transformed = preprocessor.transform(X_test)
```

### Q5: KNNImputer 與 OneHotEncoder 衝突
- KNNImputer 需要數值輸入，不能直接處理類別特徵
- 解法：在 ColumnTransformer 中分開處理，數值管線用 KNNImputer，類別管線用 SimpleImputer

## 期中專題引導 Midterm Project Guidance

### 時間點提醒
本週是第 9 週，依課綱期中專題應於本週繳交提案。可在討論時段引導學生：

1. **選擇資料集：** 建議使用 Kaggle 上的結構化資料集（如 Titanic、Housing Prices、Credit Card Fraud）
2. **明確定義問題：** 分類或回歸？評估指標是什麼？
3. **規劃 Pipeline：** 本週學到的 Pipeline 建構方法直接應用於專題
4. **時程規劃：** 下週 Week 10 學完超參數調校後就有完整的 ML 工作流

### 專題提案格式建議
- 資料集來源與描述（100 字）
- 預測目標與評估指標
- 計劃使用的前處理方法
- 計劃嘗試的模型
- 預期交付物

## 備課提醒 Preparation Notes

- 提前確認 Titanic 資料集的下載方式（`seaborn.load_dataset('titanic')` 最簡單）
- 準備一份含離群值的合成資料，用於 Scaler Demo
- 測試 Pipeline + GridSearchCV 的執行時間（在學生機器上可能較慢，準備較小的參數空間）
- 本週內容量較大，可視學生程度調整理論/實作比例
- 如時間不足，PCA 部分可簡化為「觀看 Demo + 課後練習」
- 準備 Google Colab 版本的 Notebook，以防本地環境問題

## 補充資源 Supplementary Resources
- scikit-learn Pipeline 官方範例：https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
- Kaggle Titanic 入門教程：https://www.kaggle.com/c/titanic
- Feature Engineering 最佳實務（YouTube 推薦影片可提前準備）
