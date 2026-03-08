# 範例資料集 Sample Datasets

本目錄存放課程所使用的範例資料集 (Sample Datasets)，供各週教材的 Notebook 與作業使用。

## 用途 Purpose
- 提供機器學習 (Machine Learning) 與深度學習 (Deep Learning) 課程練習所需的資料
- 涵蓋分類 (Classification)、迴歸 (Regression)、分群 (Clustering)、影像 (Image) 等任務類型
- 資料集規模適中，適合教學與互動視覺化展示

## 使用方式 Usage
各週課程的 Notebook 會透過相對路徑引用本目錄中的資料集，也可在互動平台上直接載入。

### 自動下載腳本 Download Script

使用 `download.py` 可自動下載各週所需資料集：

```bash
# 安裝相依套件 Install dependencies
pip install kaggle scikit-learn pandas numpy

# 列出所有資料集資訊（不下載）
python download.py --list

# 下載所有週次的資料集
python download.py --all

# 下載特定週次的資料集
python download.py --week 1
python download.py --week 4 7 12

# 下載並顯示資料集基本資訊
python download.py --week 1 --info

# 指定下載目錄
python download.py --week 1 --output ./my_data
```

> **注意 Note：** 使用 Kaggle 資料集下載功能前，需先設定 Kaggle API Token。前往 https://www.kaggle.com/settings 建立 API Token，並將 `kaggle.json` 放到 `~/.kaggle/` 目錄下。

---

## 每週資料集清單 Weekly Dataset Catalog

### 第 1 週 Week 1：課程導論、Python 與資料科學環境
Introduction, Python & Data Science Environment

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| Iris 鳶尾花資料集 | sklearn (`load_iris`) | 分類 Classification | ~7 KB |
| Tips 小費資料集 | Kaggle (`ranjeetjain3/seaborn-tips-dataset`) | 探索分析 EDA | ~8 KB |

- **Iris**：經典的多類別分類資料集，含 150 筆樣本、4 個特徵、3 個類別。適合初學者練習資料載入與基本操作。
- **Tips**：餐廳小費資料，含帳單金額、小費、性別、吸菸與否等欄位。適合 Pandas 入門練習。

---

### 第 2 週 Week 2：資料視覺化與 EDA
Data Visualization & EDA

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| Titanic 鐵達尼號資料集 | Kaggle (`yasserh/titanic-dataset`) | 分類 / EDA | ~60 KB |
| Penguins 企鵝資料集 | Kaggle (`parulpandey/palmer-archipelago-antarctica-penguin-data`) | 分類 / EDA | ~12 KB |

- **Titanic**：經典的 Kaggle 入門資料集，包含乘客資訊與存活狀態。適合 EDA 與視覺化練習，含缺失值處理情境。
- **Penguins**：Palmer 群島企鵝體態量測資料。Iris 的現代替代品，適合視覺化。

---

### 第 3 週 Week 3：監督式學習、資料分割與交叉驗證
Supervised Learning, Data Splitting & Cross-Validation

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| Wine 葡萄酒資料集 | sklearn (`load_wine`) | 分類 Classification | ~11 KB |
| Breast Cancer 乳癌資料集 | sklearn (`load_breast_cancer`) | 分類 Classification | ~100 KB |

- **Wine**：義大利三種葡萄酒的化學分析結果，含 13 個特徵與 3 個類別。適合練習分割與交叉驗證。
- **Breast Cancer**：乳癌腫瘤特徵資料集，含 30 個數值特徵。適合練習二元分類與交叉驗證。

---

### 第 4 週 Week 4：線性回歸 — 損失函數、梯度下降視覺化
Linear Regression — Loss Function & Gradient Descent Visualization

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| California Housing 加州房價 | sklearn (`fetch_california_housing`) | 迴歸 Regression | ~450 KB |
| Student Performance 學生成績 | Kaggle (`spscientist/students-performance-in-exams`) | 迴歸 Regression | ~18 KB |

- **California Housing**：加州房價資料集，含 8 個特徵（收入、房齡、房間數等）與房價中位數。取代已棄用的 Boston Housing。
- **Student Performance**：學生考試成績資料，含性別、種族、父母學歷、午餐類型等社經因素。

---

### 第 5 週 Week 5：分類 — 邏輯迴歸、決策邊界與 ROC/PR 曲線
Classification — Logistic Regression, Decision Boundary & ROC/PR Curves

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| Heart Disease 心臟病資料集 | Kaggle (`johnsmith88/heart-disease-dataset`) | 分類 Classification | ~12 KB |
| Make Moons / Make Circles | sklearn (`make_moons`) | 分類 Classification | 動態生成 |

- **Heart Disease**：心臟病診斷資料集，含年齡、性別、胸痛類型、血壓等 13 個特徵。適合二元分類與 ROC 分析。
- **Make Moons**：scikit-learn 合成的二維非線性可分資料，適合視覺化決策邊界。

---

### 第 6 週 Week 6：SVM 與核方法視覺化
SVM & Kernel Methods Visualization

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| Make Blobs (合成資料) | sklearn (`make_blobs`) | 分類 Classification | 動態生成 |
| Digits 手寫數字資料集 | sklearn (`load_digits`) | 分類 Classification | ~350 KB |

- **Make Blobs**：scikit-learn 合成的群集資料，可調整類別數與特徵數。適合 SVM 核方法比較視覺化。
- **Digits**：8x8 像素的手寫數字影像（0-9），共 1797 筆。輕量版 MNIST，適合 SVM 多類別分類。

---

### 第 7 週 Week 7：樹模型與集成（RF、GBDT）
Tree Models & Ensemble Methods (RF, GBDT)

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| Adult Income 成人收入 | Kaggle (`wenruliu/adult-income-dataset`) | 分類 Classification | ~4 MB |
| Diabetes 糖尿病資料集 | sklearn (`load_diabetes`) | 迴歸 Regression | ~50 KB |

- **Adult Income**：美國人口普查資料，預測收入是否超過 50K 美元。含類別型與數值型特徵，適合樹模型。
- **Diabetes**：糖尿病病程進展資料集，含 10 個基線特徵。適合迴歸樹與集成方法比較。

---

### 第 8 週 Week 8：特徵重要度與 SHAP 值
Feature Importance & SHAP Values

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| Ames Housing 房價資料集 | Kaggle (`prevek18/ames-housing-dataset`) | 迴歸 Regression | ~460 KB |

- **Ames Housing**：Iowa 州 Ames 市房屋銷售資料，含 79 個解釋變數。特徵豐富，適合特徵重要度與 SHAP 分析。

---

### 第 9 週 Week 9：特徵工程與資料前處理管線
Feature Engineering & Data Preprocessing Pipeline

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| Spaceship Titanic | Kaggle (`competitions/spaceship-titanic`) | 分類 Classification | ~1 MB |
| Credit Card Fraud 信用卡詐欺 | Kaggle (`mlg-ulb/creditcardfraud`) | 分類（不平衡）| ~144 MB |

- **Spaceship Titanic**：Kaggle 競賽資料集，包含多種特徵類型。含缺失值，適合建構完整前處理 Pipeline。
- **Credit Card Fraud**：信用卡交易詐欺偵測資料集，極度不平衡（正例僅 0.17%）。適合重取樣練習。

---

### 第 10 週 Week 10：超參數調校與學習曲線
Hyperparameter Tuning & Learning Curves

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| Wine Quality 葡萄酒品質 | Kaggle (`uciml/red-wine-quality-cortez-et-al-2009`) | 分類 / 迴歸 | ~84 KB |

- **Wine Quality**：紅酒品質評分資料，含 11 個理化特徵。適合超參數搜尋與學習曲線分析。

---

### 第 11 週 Week 11：神經網路基礎
Neural Network Basics

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| MNIST 手寫數字 | PyTorch (`torchvision.datasets.MNIST`) | 分類 Classification | ~12 MB |
| Fashion-MNIST | PyTorch (`torchvision.datasets.FashionMNIST`) | 分類 Classification | ~30 MB |

- **MNIST**：經典的手寫數字辨識資料集，28x28 灰階影像，60,000 筆訓練 + 10,000 筆測試。
- **Fashion-MNIST**：時尚物件 28x28 灰階影像。MNIST 的進階替代品。

---

### 第 12 週 Week 12：CNN 視覺化
CNN Visualization (Filters, Feature Maps, CAM/Grad-CAM)

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| CIFAR-10 | PyTorch (`torchvision.datasets.CIFAR10`) | 分類 Classification | ~170 MB |
| Cats vs Dogs（子集）| Kaggle (`biaiscience/dogs-vs-cats`) | 分類 Classification | ~800 MB |

- **CIFAR-10**：10 類彩色影像資料集，32x32 RGB 影像。適合 CNN 卷積核與特徵圖視覺化。
- **Cats vs Dogs**：貓狗影像分類。適合 CNN 分類與 Grad-CAM 視覺化練習。建議使用子集。

---

### 第 13 週 Week 13：RNN / 序列建模
RNN/Sequence Modeling (LSTM/GRU; Transformers Concepts)

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| IMDB 電影評論 | PyTorch (`torchtext.datasets.IMDB`) | 分類 (NLP) | ~84 MB |
| Airline Passengers 航空旅客 | URL (直接下載) | 時間序列預測 | ~2 KB |

- **IMDB**：IMDB 電影評論正負面情感分類，50,000 筆評論。適合 RNN/LSTM 文本分類。
- **Airline Passengers**：1949-1960 年國際航空旅客月度數據。經典的時間序列預測資料集。

---

### 第 14 週 Week 14：深度學習訓練技巧
Deep Learning Training Techniques

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| CIFAR-100 | PyTorch (`torchvision.datasets.CIFAR100`) | 分類 Classification | ~170 MB |

- **CIFAR-100**：100 類彩色影像資料集。適合比較不同訓練策略（LR 排程、資料增強等）的效果。

---

### 第 15 週 Week 15：模型評估與偏誤檢測、公平性與穩健性
Model Evaluation, Bias Detection, Fairness & Robustness

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| German Credit 德國信用風險 | Kaggle (`uciml/german-credit`) | 分類 / 公平性 | ~56 KB |
| COMPAS Recidivism 再犯率 | Kaggle (`danofer/compass`) | 分類 / 公平性 | ~5 MB |

- **German Credit**：信用風險評估資料集，含性別、年齡等敏感屬性。適合公平性分析。
- **COMPAS**：美國刑事司法系統再犯率預測資料。經典的 AI 公平性案例研究資料集。

---

### 第 16 週 Week 16：MLOps 入門
MLOps Introduction (Model Versioning, Serving, Monitoring)

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| California Housing（同第 4 週）| sklearn (`fetch_california_housing`) | 迴歸 Regression | ~450 KB |

- **California Housing**：複用第 4 週資料集。本週重點在模型部署與版本管理流程而非資料本身。

---

### 第 17 週 Week 17：LLM 與嵌入應用
LLM & Embedding Applications (RAG, Prompt Engineering)

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| 20 Newsgroups 新聞群組 | sklearn (`fetch_20newsgroups`) | 分類 / NLP | ~14 MB |
| Wikipedia Simple English（子集）| URL (直接下載) | NLP / RAG | ~250 MB |

- **20 Newsgroups**：20 個新聞群組的文本資料，約 20,000 篇文章。適合文字嵌入與 RAG 練習。
- **Wikipedia Simple English**：簡易英文維基百科文本。適合建構 RAG 系統知識庫。建議使用預處理後的子集。

---

### 第 18 週 Week 18：綜合專題開發與展示
Final Project Presentation, Reflection & Course Feedback

| 資料集 Dataset | 來源 Source | 任務 Task | 大小 Size |
|---------------|-----------|----------|----------|
| 自選資料集 Student's Choice | Kaggle 或其他公開平台 | 依專題而定 | 依資料集而定 |

- 期末專題由學生自行選擇適合的資料集。建議從 Kaggle Datasets、UCI ML Repository 或政府開放資料平台挑選。

---

## 資料集統計 Dataset Statistics

| 類別 Category | 數量 Count |
|--------------|-----------|
| scikit-learn 內建 / 合成 | 10 |
| Kaggle 資料集 | 12 |
| PyTorch / torchvision | 5 |
| URL 直接下載 | 2 |
| 學生自選 | 1 |
| **總計 Total** | **30** |

## 注意事項 Notes

1. **Kaggle API 認證**：下載 Kaggle 資料集前需設定 API Token（見 `download.py` 使用說明）
2. **PyTorch 資料集**：透過 `torchvision` 首次執行時自動下載，不需手動處理
3. **資料集大小**：部分資料集（如 CIFAR、Cats vs Dogs）檔案較大，建議在穩定網路環境下下載
4. **合成資料**：sklearn 的 `make_*` 系列為程式動態生成，每次執行可指定不同參數
5. **倫理注意**：部分資料集（如 COMPAS、German Credit）涉及敏感屬性，使用時請注意 AI 倫理議題
