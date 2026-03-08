# ML/DL 視覺化工具 — 18 週課程大綱
# ML/DL Visualization Tools — 18-Week Syllabus

> 授課教師：楊東偉
> 適用對象：大專院校學生
> 每週時數：90 分鐘（理論 30 分 + 實作 40 分 + 討論 20 分）

---

## 課程總覽 Course Overview

本課程以視覺化 (Visualization) 與互動式學習 (Interactive Learning) 為核心，帶領學生從機器學習 (Machine Learning) 基礎邁向深度學習 (Deep Learning) 應用，搭配 LLM 即時助教輔助學習。

### 核心能力指標 Core Competencies
1. 理解 ML/DL 核心概念並能以視覺化方式解釋
2. 使用 Python 實作完整的資料科學流程
3. 運用互動工具進行模型訓練、評估與調校
4. 具備基礎 MLOps 與 LLM 應用能力
5. 培養 AI 倫理意識與負責任使用的態度

### 評量方式 Assessment
| 項目 | 比重 | 說明 |
|------|------|------|
| 每週作業 Weekly Assignments | 30% | 含實作與概念測驗 |
| 期中專題 Midterm Project | 20% | 第 9 週繳交 |
| 期末專題 Final Project | 30% | 第 18 週展示 |
| 課堂參與 Participation | 10% | 討論、AI 助教互動 |
| 學習歷程 Learning Portfolio | 10% | Notebook 紀錄與反思 |

---

## 第 1 週：課程導論、Python 與資料科學環境
### Week 1: Introduction, Python & Data Science Environment

**難度等級 Level:** 核心 Core

**學習目標 Learning Objectives:**
1. 了解機器學習 (ML) 與深度學習 (DL) 的基本概念與應用場景
2. 建立 Python 資料科學開發環境（Anaconda/Miniconda）
3. 熟悉 Jupyter Notebook / Google Colab 操作
4. 認識課程平台與 AI 助教的使用方式

**先備知識 Prerequisites:** 基礎程式設計概念

**教學活動 Activities:**
- 理論：ML/DL 發展脈絡與應用案例介紹
- 實作：環境安裝、基礎 Python 操作、NumPy/Pandas 入門
- 討論：課程平台導覽、AI 助教使用規範

**評量 Assessment:** 環境建置確認 + 基礎操作練習

---

## 第 2 週：資料視覺化與 EDA（互動圖表）
### Week 2: Data Visualization & EDA (Interactive Charts)

**難度等級 Level:** 核心 Core

**學習目標 Learning Objectives:**
1. 掌握 Matplotlib、Seaborn、Plotly 三大視覺化工具
2. 了解探索式資料分析 (Exploratory Data Analysis, EDA) 流程
3. 能製作互動式圖表並解讀資料分布特性
4. 學會處理缺失值 (Missing Values) 與異常值 (Outliers)

**先備知識 Prerequisites:** Week 1 Python 環境

**教學活動 Activities:**
- 理論：EDA 方法論與視覺化原則
- 實作：使用 Plotly 建立互動圖表、資料清洗練習
- 討論：不同圖表類型的適用情境

**評量 Assessment:** EDA 報告 + 互動圖表作業

---

## 第 3 週：監督式學習概念、資料分割與交叉驗證
### Week 3: Supervised Learning, Data Splitting & Cross-Validation

**難度等級 Level:** 核心 Core

**學習目標 Learning Objectives:**
1. 理解監督式學習 (Supervised Learning) 的基本框架
2. 掌握訓練集/測試集分割 (Train/Test Split) 的原則
3. 了解 k 折交叉驗證 (k-Fold Cross-Validation) 的原理與應用
4. 認識過擬合 (Overfitting) 與欠擬合 (Underfitting) 的概念

**先備知識 Prerequisites:** Week 1-2

**教學活動 Activities:**
- 理論：監督式學習框架、偏差-變異權衡 (Bias-Variance Tradeoff)
- 實作：視覺化不同分割策略的影響、交叉驗證實驗
- 討論：何時該用分層抽樣 (Stratified Sampling)？

**評量 Assessment:** 交叉驗證實作 + 概念測驗

---

## 第 4 週：線性回歸 — 損失函數、梯度下降視覺化
### Week 4: Linear Regression — Loss Function & Gradient Descent Visualization

**難度等級 Level:** 核心 Core

**學習目標 Learning Objectives:**
1. 理解線性回歸 (Linear Regression) 的數學原理
2. 了解損失函數 (Loss Function) 與均方誤差 (MSE) 的意義
3. 透過視覺化理解梯度下降 (Gradient Descent) 的運作過程
4. 實驗學習率 (Learning Rate) 對收斂的影響

**先備知識 Prerequisites:** Week 1-3, 基礎微積分

**教學活動 Activities:**
- 理論：損失函數推導、梯度計算、過/欠擬合
- 實作：互動平台觀察梯度下降動畫、學習率實驗、殘差圖分析
- 討論：模型假設何時不成立、如何診斷資料問題

**評量 Assessment:** 梯度下降視覺化實驗報告 + 小測

---

## 第 5 週：分類 — 邏輯迴歸、決策邊界與 ROC/PR 曲線
### Week 5: Classification — Logistic Regression, Decision Boundary & ROC/PR Curves

**難度等級 Level:** 核心 Core

**學習目標 Learning Objectives:**
1. 理解邏輯迴歸 (Logistic Regression) 與 Sigmoid 函數
2. 透過視覺化觀察決策邊界 (Decision Boundary) 的形成
3. 掌握 ROC 曲線與 PR 曲線的解讀方法
4. 了解分類評估指標：準確率、精確率、召回率、F1

**先備知識 Prerequisites:** Week 1-4

**教學活動 Activities:**
- 理論：從回歸到分類、機率解釋、閾值選擇
- 實作：互動調整參數觀察決策邊界變化、繪製 ROC/PR 曲線
- 討論：不平衡資料集 (Imbalanced Dataset) 的處理策略

**評量 Assessment:** 分類模型建構 + ROC 分析作業

---

## 第 6 週：SVM 與核方法視覺化
### Week 6: SVM & Kernel Methods Visualization

**難度等級 Level:** 核心 Core

**學習目標 Learning Objectives:**
1. 理解支持向量機 (Support Vector Machine, SVM) 的原理
2. 了解間隔最大化 (Margin Maximization) 與支持向量 (Support Vectors)
3. 透過視覺化理解核技巧 (Kernel Trick) 的非線性轉換
4. 比較不同核函數 (RBF, Polynomial, Linear) 的效果

**先備知識 Prerequisites:** Week 1-5

**教學活動 Activities:**
- 理論：SVM 原理、核函數、正則化參數 C
- 實作：互動調整 C 值與核函數觀察決策邊界變化
- 討論：SVM vs. Logistic Regression 的選擇時機

**評量 Assessment:** SVM 核方法比較實驗

---

## 第 7 週：樹模型與集成（RF、GBDT）
### Week 7: Tree Models & Ensemble Methods (RF, GBDT)

**難度等級 Level:** 核心 Core

**學習目標 Learning Objectives:**
1. 理解決策樹 (Decision Tree) 的建構與剪枝 (Pruning)
2. 了解隨機森林 (Random Forest) 的 Bagging 原理
3. 掌握梯度提升 (Gradient Boosting, GBDT) 的概念
4. 透過視覺化觀察樹的生長與集成效果

**先備知識 Prerequisites:** Week 1-6

**教學活動 Activities:**
- 理論：CART 演算法、Bagging vs. Boosting
- 實作：決策樹生長動畫、比較 RF/GBDT 效能
- 討論：偏差-變異在集成方法中的角色

**評量 Assessment:** 集成模型比較報告

---

## 第 8 週：特徵重要度與 Shapley 示意
### Week 8: Feature Importance & SHAP Values

**難度等級 Level:** 核心 Core

**學習目標 Learning Objectives:**
1. 理解特徵重要度 (Feature Importance) 的計算方式
2. 了解排列重要度 (Permutation Importance) 方法
3. 掌握 SHAP (SHapley Additive exPlanations) 的基本原理
4. 能解讀 SHAP 蜂群圖 (Beeswarm Plot) 與力圖 (Force Plot)

**先備知識 Prerequisites:** Week 1-7

**教學活動 Activities:**
- 理論：可解釋性 AI (Explainable AI, XAI) 概念、SHAP 值推導
- 實作：計算特徵重要度、生成 SHAP 互動圖表
- 討論：模型透明度 vs. 預測能力的取捨

**評量 Assessment:** SHAP 分析報告

---

## 第 9 週：特徵工程與資料前處理管線
### Week 9: Feature Engineering & Data Preprocessing Pipeline

**難度等級 Level:** 核心 Core

**學習目標 Learning Objectives:**
1. 掌握常見特徵工程技術（編碼、縮放、衍生特徵）
2. 了解 sklearn Pipeline 的建構與使用
3. 能建立可重現的資料前處理流程
4. 視覺化前處理對模型效能的影響

**先備知識 Prerequisites:** Week 1-8

**教學活動 Activities:**
- 理論：One-Hot Encoding、StandardScaler、特徵選擇
- 實作：建構完整 Pipeline、視覺化前後對比
- 討論：期中專題方向討論

**評量 Assessment:** 期中專題提案 + Pipeline 實作

---

## 第 10 週：超參數調校與學習曲線
### Week 10: Hyperparameter Tuning & Learning Curves

**難度等級 Level:** 核心 Core

**學習目標 Learning Objectives:**
1. 理解超參數 (Hyperparameters) 與模型參數的區別
2. 掌握 Grid Search 與 Random Search 方法
3. 透過學習曲線 (Learning Curve) 診斷模型問題
4. 了解驗證曲線 (Validation Curve) 的應用

**先備知識 Prerequisites:** Week 1-9

**教學活動 Activities:**
- 理論：超參數空間、搜尋策略、早停法則
- 實作：互動式超參數搜尋熱力圖、學習曲線分析
- 討論：計算資源限制下的調校策略

**評量 Assessment:** 超參數調校實驗 + 學習曲線分析

---

## 第 11 週：神經網路基礎（激活/正則化/BatchNorm）
### Week 11: Neural Network Basics (Activation/Regularization/BatchNorm)

**難度等級 Level:** 核心 Core

**學習目標 Learning Objectives:**
1. 理解感知器 (Perceptron) 與多層神經網路結構
2. 了解不同激活函數 (Activation Functions) 的特性
3. 掌握正則化技術：Dropout、L1/L2、BatchNorm
4. 透過視覺化觀察激活函數與正則化的效果

**先備知識 Prerequisites:** Week 1-10, 基礎線性代數

**教學活動 Activities:**
- 理論：前饋網路、反向傳播、梯度消失/爆炸
- 實作：互動式激活函數比較、正則化效果視覺化
- 討論：何時該用哪種正則化技術

**評量 Assessment:** 神經網路實作 + 激活函數實驗

---

## 第 12 週：CNN 視覺化（卷積核、特徵圖、CAM/Grad-CAM）
### Week 12: CNN Visualization (Filters, Feature Maps, CAM/Grad-CAM)

**難度等級 Level:** 進階 Advanced

**學習目標 Learning Objectives:**
1. 理解卷積神經網路 (CNN) 的基本結構
2. 透過視覺化觀察卷積核 (Filters) 與特徵圖 (Feature Maps)
3. 掌握 CAM/Grad-CAM 技術解釋 CNN 決策
4. 了解經典 CNN 架構（LeNet、VGG、ResNet 概念）

**先備知識 Prerequisites:** Week 1-11

**教學活動 Activities:**
- 理論：卷積運算、池化、CNN 架構演進
- 實作：CNN 層級瀏覽器、Grad-CAM 熱度圖生成
- 討論：CNN 在不同領域的應用

**評量 Assessment:** CNN 視覺化分析報告

**進階延伸 Extended:**
- 遷移學習 (Transfer Learning) 實驗
- 資料增強策略視覺化

---

## 第 13 週：RNN/序列建模（LSTM/GRU；Transformers 概念）
### Week 13: RNN/Sequence Modeling (LSTM/GRU; Transformers Concepts)

**難度等級 Level:** 進階 Advanced

**學習目標 Learning Objectives:**
1. 理解遞迴神經網路 (RNN) 處理序列資料的原理
2. 了解 LSTM 與 GRU 解決長程依賴的機制
3. 初步認識 Transformer 架構與自注意力 (Self-Attention) 機制
4. 透過視覺化觀察序列模型的注意力分布

**先備知識 Prerequisites:** Week 1-12

**教學活動 Activities:**
- 理論：RNN 結構、LSTM 門控機制、Attention 概念
- 實作：序列預測任務、注意力視覺化
- 討論：為何 Transformer 取代了 RNN

**評量 Assessment:** 序列模型比較實驗

**進階延伸 Extended:**
- Transformer 完整實作（選修）
- 位置編碼 (Positional Encoding) 視覺化

---

## 第 14 週：深度學習訓練技巧
### Week 14: Deep Learning Training Techniques

**難度等級 Level:** 核心 Core

**學習目標 Learning Objectives:**
1. 掌握學習率排程策略 (LR Scheduling)：Step、Cosine、Warmup
2. 理解早停 (Early Stopping) 的原理與實作
3. 了解資料增強 (Data Augmentation) 的常用技術
4. 透過視覺化比較不同訓練策略的效果

**先備知識 Prerequisites:** Week 1-13

**教學活動 Activities:**
- 理論：學習率策略、正則化、增強技術
- 實作：訓練曲線比較器、不同策略的效果視覺化
- 討論：實務中的訓練技巧與經驗法則

**評量 Assessment:** 訓練策略比較實驗

---

## 第 15 週：模型評估與偏誤檢測、公平性與穩健性
### Week 15: Model Evaluation, Bias Detection, Fairness & Robustness

**難度等級 Level:** 進階 Advanced

**學習目標 Learning Objectives:**
1. 深入理解混淆矩陣 (Confusion Matrix) 與多類別評估
2. 了解模型偏誤 (Bias) 的來源與檢測方法
3. 認識公平性指標 (Fairness Metrics)
4. 掌握穩健性 (Robustness) 測試的基本方法

**先備知識 Prerequisites:** Week 1-14

**教學活動 Activities:**
- 理論：公平性定義、偏誤來源、穩健性概念
- 實作：公平性指標儀表板、對抗樣本視覺化
- 討論：AI 倫理案例分析

**評量 Assessment:** 公平性分析報告

---

## 第 16 週：MLOps 入門（模型版本、推論服務、監測）
### Week 16: MLOps Introduction (Model Versioning, Serving, Monitoring)

**難度等級 Level:** 進階 Advanced

**學習目標 Learning Objectives:**
1. 了解 MLOps 的基本概念與流程
2. 學會使用 MLflow 進行模型版本管理
3. 認識模型部署 (Model Deployment) 的基本方式
4. 了解模型監測 (Model Monitoring) 的重要性

**先備知識 Prerequisites:** Week 1-15

**教學活動 Activities:**
- 理論：MLOps 流程、CI/CD for ML、模型漂移
- 實作：MLflow 實驗追蹤、簡易 API 部署
- 討論：從研究到生產的挑戰

**評量 Assessment:** MLOps 流程設計

**進階延伸 Extended:**
- Docker 容器化部署（選修）
- 模型漂移 (Model Drift) 監測

---

## 第 17 週：LLM 與嵌入應用（檢索增強、提示工程基礎）
### Week 17: LLM & Embedding Applications (RAG, Prompt Engineering Basics)

**難度等級 Level:** 進階 Advanced

**學習目標 Learning Objectives:**
1. 了解大型語言模型 (LLM) 的基本原理
2. 理解文字嵌入 (Text Embeddings) 與向量搜尋
3. 認識檢索增強生成 (Retrieval-Augmented Generation, RAG) 架構
4. 掌握提示工程 (Prompt Engineering) 的基礎技巧

**先備知識 Prerequisites:** Week 1-16

**教學活動 Activities:**
- 理論：Transformer 回顧、嵌入空間、RAG 架構
- 實作：嵌入空間視覺化、簡易 RAG 系統
- 討論：LLM 的能力與限制

**評量 Assessment:** RAG 應用設計 + 提示工程練習

---

## 第 18 週：綜合專題開發與展示、反思與課程回饋
### Week 18: Final Project Presentation, Reflection & Course Feedback

**難度等級 Level:** 核心 Core

**學習目標 Learning Objectives:**
1. 完成期末專題的整合、測試與展示準備
2. 能完整報告從資料到部署的全流程
3. 進行課程學習反思與自我評估
4. 提供課程回饋以改善未來教學

**先備知識 Prerequisites:** Week 1-17 全部

**教學活動 Activities:**
- 專題展示（每組 15 分鐘，含 Q&A）
- 同儕互評 (Peer Review)
- 課程回饋問卷
- 學習歷程總結與未來學習路徑建議

**評量 Assessment:** 期末專題報告 + 展示 + 反思文件

**期末專題要求 Final Project Requirements:**
- 完整資料科學流程（資料→前處理→模型→評估→視覺化）
- 包含可重現性 (Reproducibility) 文件
- 含風險與倫理檢核 (Ethics Checklist)
- GitHub 儲存庫 + README

---

## 參考資源 References
- Kaggle (kaggle.com) — 資料集與競賽
- scikit-learn 官方文件 (scikit-learn.org)
- PyTorch 官方教學 (pytorch.org/tutorials)
- GitHub — 開源專案與案例
- SHAP 文件 (shap.readthedocs.io)
- MLflow 文件 (mlflow.org)
