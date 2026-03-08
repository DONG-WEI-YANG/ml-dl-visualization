# 第 18 週：綜合專題開發與展示、反思與課程回饋
# Week 18: Final Project Presentation, Reflection & Course Feedback

## 學習目標 Learning Objectives
1. 理解期末專題的完整要求與評分標準
2. 回顧完整資料科學專案流程 (End-to-End Data Science Pipeline)
3. 掌握專題展示技巧：簡報結構、Demo 準備、Q&A 應對
4. 學會撰寫技術文件：README、requirements.txt、Docker 部署文件
5. 建立可重現性 (Reproducibility) 與倫理檢核 (Ethics Checklist) 意識
6. 總結 18 週課程核心知識
7. 探索未來學習路徑與 ML/DL 職涯方向

---

## 1. 期末專題要求與評分標準 Final Project Requirements & Grading

### 1.1 專題目標 Project Objectives

期末專題要求同學展示 **完整的資料科學或機器學習專案流程**，從問題定義 (Problem Definition) 到模型部署 (Deployment)，體現本課程 18 週所學的知識與技能。

### 1.2 基本要求 Basic Requirements

| 項目 | 說明 |
|------|------|
| 問題定義 Problem Definition | 明確描述要解決的問題與動機 |
| 資料來源 Data Source | 公開資料集或自行蒐集，需說明資料取得方式 |
| 探索式分析 EDA | 包含資料分布、缺失值處理、特徵分析 |
| 模型建構 Modeling | 至少嘗試兩種以上模型並比較 |
| 模型評估 Evaluation | 使用適當的評估指標，含交叉驗證 |
| 可解釋性 Explainability | 至少包含一種模型解釋方法（如 SHAP） |
| 視覺化 Visualization | 使用互動式圖表呈現關鍵結果 |
| 技術文件 Documentation | README、requirements.txt、程式碼註解 |
| 倫理檢核 Ethics Review | 完成倫理檢核清單 |
| 可重現性 Reproducibility | 他人可依據文件重現你的結果 |

### 1.3 評分標準總覽 Grading Overview

| 面向 | 比重 | 主要評估項目 |
|------|:---:|------|
| 技術深度 Technical Depth | 30% | 模型選擇、調參、程式品質 |
| 展示表達 Presentation | 20% | 簡報品質、Demo 流暢度、Q&A |
| 文件品質 Documentation | 20% | README、程式註解、可重現性 |
| 創意與價值 Creativity & Value | 15% | 問題選題、解決方案的創新性 |
| 倫理與反思 Ethics & Reflection | 15% | 倫理檢核、限制分析、反思品質 |

---

## 2. 完整資料科學專案流程回顧 End-to-End Data Science Pipeline Review

### 2.1 流程總覽 Pipeline Overview

```
1. 問題定義 Problem Definition
   ↓
2. 資料蒐集與清洗 Data Collection & Cleaning
   ↓
3. 探索式資料分析 Exploratory Data Analysis (EDA)
   ↓
4. 特徵工程 Feature Engineering
   ↓
5. 模型選擇與訓練 Model Selection & Training
   ↓
6. 超參數調校 Hyperparameter Tuning
   ↓
7. 模型評估與解釋 Model Evaluation & Interpretation
   ↓
8. 部署與監測 Deployment & Monitoring
   ↓
9. 文件與溝通 Documentation & Communication
```

### 2.2 各步驟對應的課程週次 Steps Mapped to Course Weeks

| 步驟 | 對應週次 | 核心工具/技術 |
|------|---------|-------------|
| 問題定義 | Week 1 | 問題類型判斷（分類/回歸/分群） |
| 資料清洗 | Week 2 | Pandas, 缺失值/異常值處理 |
| EDA | Week 2 | Matplotlib, Seaborn, Plotly |
| 資料分割 | Week 3 | Train/Test Split, k-Fold CV |
| 回歸模型 | Week 4 | Linear Regression, 梯度下降 |
| 分類模型 | Week 5 | Logistic Regression, ROC/PR |
| SVM | Week 6 | 核方法, 決策邊界 |
| 樹模型/集成 | Week 7 | RF, GBDT, XGBoost |
| 模型解釋 | Week 8 | SHAP, Feature Importance |
| 特徵工程 | Week 9 | Pipeline, Encoding, Scaling |
| 超參數調校 | Week 10 | Grid/Random Search, Learning Curves |
| 神經網路 | Week 11 | Activation, Regularization, BatchNorm |
| CNN | Week 12 | 卷積核, Feature Maps, Grad-CAM |
| RNN/Transformer | Week 13 | LSTM, GRU, Self-Attention |
| 訓練技巧 | Week 14 | LR Scheduling, Early Stopping, Data Augmentation |
| 評估與公平性 | Week 15 | Bias Detection, Fairness Metrics |
| MLOps | Week 16 | MLflow, Model Serving, Monitoring |
| LLM 應用 | Week 17 | Embeddings, RAG, Prompt Engineering |
| 專題展示 | Week 18 | 展示、文件、反思 |

### 2.3 專案流程常見錯誤 Common Mistakes in Projects

| 錯誤類型 | 說明 | 如何避免 |
|----------|------|----------|
| 資料洩漏 Data Leakage | 在分割前做標準化或特徵選擇 | 所有前處理放在 Pipeline 中 |
| 評估指標不當 Wrong Metric | 不平衡資料集用 Accuracy | 根據問題特性選擇 F1/AUC/PR-AUC |
| 過度調參 Over-tuning | 在測試集上反覆調參 | 使用驗證集調參，測試集只用一次 |
| 忽略基準模型 No Baseline | 直接用複雜模型 | 先建立簡單基準，再逐步改進 |
| 缺乏可重現性 Not Reproducible | 未固定隨機種子或紀錄版本 | 設定 seed、使用 requirements.txt |

---

## 3. 專題展示技巧 Presentation Skills

### 3.1 簡報結構 Presentation Structure

一個好的專題展示通常包含以下結構，建議 15 分鐘展示：

| 段落 | 時間 | 內容 |
|------|:---:|------|
| 開場 Opening | 1 分鐘 | 自我介紹、專題名稱、一句話總結 |
| 問題與動機 Motivation | 2 分鐘 | 為什麼做這個題目？解決什麼問題？ |
| 資料與方法 Data & Method | 3 分鐘 | 資料來源、EDA 重點發現、方法選擇理由 |
| 實驗結果 Results | 4 分鐘 | 模型表現、關鍵圖表、SHAP 解釋 |
| Demo 展示 Live Demo | 3 分鐘 | 實際操作或互動視覺化 |
| 結論與反思 Conclusion | 2 分鐘 | 主要發現、限制、未來方向、倫理考量 |

### 3.2 簡報設計原則 Slide Design Principles

1. **一張投影片一個重點 (One Slide, One Point)**：避免資訊過載
2. **圖表優先 (Visualize First)**：用圖表取代文字表格
3. **字體大小 (Font Size)**：標題至少 28pt，內文至少 18pt
4. **配色一致 (Consistent Color Scheme)**：全簡報使用統一色系
5. **減少動畫 (Minimal Animation)**：除非動畫幫助理解概念
6. **程式碼精簡 (Concise Code)**：只放關鍵片段，不要整段程式碼

### 3.3 Demo 準備技巧 Demo Preparation Tips

```
Demo 前的檢核清單 Pre-Demo Checklist:
□ 環境已安裝且測試通過
□ 網路連線正常（若需要線上資源）
□ 準備離線備案（截圖/錄影）
□ Jupyter Notebook 已重新跑過 (Restart & Run All)
□ 關閉不相關的應用程式與通知
□ 確認螢幕解析度在投影時可讀
□ 準備 2-3 個不同輸入案例
```

**常見 Demo 風險與對策：**

| 風險 | 對策 |
|------|------|
| 程式碼執行出錯 | 預先準備已執行好的 Notebook 備份 |
| 網路斷線 | 預載資料、準備離線版本 |
| 模型載入太久 | 預先載入模型，Demo 時只做推論 |
| 投影螢幕字太小 | 調大字體、使用 zoom 功能 |

### 3.4 Q&A 應對技巧 Handling Q&A

1. **仔細聆聽 Listen Carefully**：讓對方把問題問完再回答
2. **確認問題 Clarify**：不確定時可以反問「您的意思是...？」
3. **誠實以對 Be Honest**：不知道的問題就說「這是很好的問題，我需要進一步研究」
4. **結構化回答 Structured Answer**：先回答核心，再補充細節
5. **連結課程內容 Connect to Course**：將回答連結到課程學到的概念

**常見 Q&A 問題範例：**
- 「你為什麼選擇這個模型而不是 XXX？」
- 「你的模型在什麼情況下可能失效？」
- 「如果資料量更大，你的方法還適用嗎？」
- 「你如何處理模型的公平性問題？」
- 「這個專案的實際應用場景是什麼？」

---

## 4. 技術文件撰寫 Technical Documentation

### 4.1 README.md 撰寫指南

一份好的 README 是專案的門面，應包含以下內容：

```markdown
# 專案名稱 Project Title

一句話描述專案目的。

## 目錄 Table of Contents
- [問題描述](#問題描述)
- [安裝與使用](#安裝與使用)
- [專案結構](#專案結構)
- [實驗結果](#實驗結果)
- [貢獻者](#貢獻者)

## 問題描述 Problem Description
描述專案要解決的問題、動機與目標。

## 安裝與使用 Installation & Usage

### 環境要求
- Python >= 3.9
- CUDA >= 11.8 (若使用 GPU)

### 安裝步驟
git clone https://github.com/username/project.git
cd project
pip install -r requirements.txt

### 執行
python main.py --config config.yaml

## 專案結構 Project Structure
project/
├── data/           # 資料目錄
├── notebooks/      # Jupyter Notebook
├── src/            # 原始碼
│   ├── data/       # 資料處理
│   ├── models/     # 模型定義
│   ├── utils/      # 工具函式
│   └── train.py    # 訓練腳本
├── results/        # 實驗結果
├── README.md
├── requirements.txt
├── Dockerfile
└── .gitignore

## 資料集 Dataset
- 來源、大小、特徵說明
- 資料取得方式

## 實驗結果 Results
| 模型 | Accuracy | F1 Score | AUC |
|------|----------|----------|-----|
| Baseline | ... | ... | ... |
| Model A | ... | ... | ... |

## 倫理聲明 Ethics Statement
簡要說明倫理考量。

## 授權 License
MIT License

## 貢獻者 Contributors
- 姓名（學號）
```

### 4.2 requirements.txt 撰寫

```bash
# 生成方式一：從當前環境匯出（推薦使用 pip freeze 的精簡版）
pip freeze > requirements.txt

# 生成方式二：使用 pipreqs 自動偵測（推薦）
pip install pipreqs
pipreqs /path/to/project --force

# 生成方式三：手動撰寫（精確控制版本）
```

**requirements.txt 範例：**
```
numpy==1.26.4
pandas==2.2.1
matplotlib==3.8.3
seaborn==0.13.2
scikit-learn==1.4.1
torch==2.2.1
plotly==5.19.0
shap==0.44.1
mlflow==2.11.1
jupyter==1.0.0
```

**注意事項：**
- 固定主要版本號 (Pin Major Versions) 以確保可重現性
- 區分開發依賴 (dev dependencies) 與執行依賴 (runtime dependencies)
- 考慮使用 `pyproject.toml` 或 `setup.cfg` 進行更結構化的依賴管理

### 4.3 Docker 部署文件

Docker 可以確保環境的完全一致性，是可重現性的最佳實踐。

**Dockerfile 範例：**
```dockerfile
# 基礎映像 Base Image
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製依賴文件
COPY requirements.txt .

# 安裝依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案檔案
COPY . .

# 暴露連接埠（若為 Web 服務）
EXPOSE 8000

# 執行指令
CMD ["python", "main.py"]
```

**docker-compose.yml 範例：**
```yaml
version: '3.8'
services:
  ml-app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    environment:
      - MODEL_PATH=/app/models/best_model.pth
```

**常用 Docker 指令：**
```bash
# 建構映像 Build Image
docker build -t my-ml-project .

# 啟動容器 Run Container
docker run -p 8000:8000 my-ml-project

# 使用 docker-compose
docker-compose up -d
```

### 4.4 .gitignore 範例

```gitignore
# 資料檔案（避免上傳大型資料集）
data/raw/
*.csv
*.parquet
!data/sample.csv

# 模型檔案
*.pth
*.pkl
*.h5
*.onnx

# 環境
.env
venv/
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# 系統檔案
.DS_Store
Thumbs.db

# 實驗暫存
mlruns/
wandb/
```

---

## 5. 可重現性檢核清單 Reproducibility Checklist

可重現性 (Reproducibility) 是科學研究與工程實務的基本要求。以下檢核清單幫助你確保專案可被他人重現：

### 5.1 環境可重現性 Environment Reproducibility

- [ ] 提供 `requirements.txt` 或 `environment.yml`，包含所有套件的確切版本
- [ ] 指定 Python 版本
- [ ] 若使用 GPU，註明 CUDA 版本與 GPU 型號
- [ ] 提供 Dockerfile（選修但推薦）
- [ ] 在不同機器上測試安裝流程

### 5.2 資料可重現性 Data Reproducibility

- [ ] 說明資料來源 (Data Source) 與取得方式
- [ ] 若為公開資料集，提供下載連結與版本
- [ ] 若資料無法公開，提供資料格式描述與合成範例
- [ ] 記錄資料前處理步驟（含參數）
- [ ] 資料分割的隨機種子 (Random Seed) 已固定

### 5.3 模型可重現性 Model Reproducibility

- [ ] 固定所有隨機種子 (Random Seeds)
  ```python
  import random, numpy as np, torch
  SEED = 42
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(SEED)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  ```
- [ ] 記錄所有超參數設定
- [ ] 提供模型訓練腳本，可從頭開始訓練
- [ ] 提供預訓練模型權重（選修）
- [ ] 記錄訓練環境（硬體規格、訓練時長）

### 5.4 結果可重現性 Result Reproducibility

- [ ] Notebook 可 "Restart & Run All" 成功
- [ ] 提供完整的實驗結果表格
- [ ] 包含隨機性的實驗需報告多次執行的平均與標準差
- [ ] 圖表的生成程式碼完整可執行
- [ ] 關鍵數值與報告中的數值一致

### 5.5 文件可重現性 Documentation Reproducibility

- [ ] README 包含完整的安裝與執行指南
- [ ] 步驟說明清晰，新手可按步操作
- [ ] 提供預期輸出範例
- [ ] 已請他人實際測試可重現性

---

## 6. 倫理檢核清單 Ethics Checklist

AI 倫理 (AI Ethics) 是負責任的機器學習實踐者必備的意識。以下檢核清單協助你反思專案的倫理面向：

### 6.1 資料倫理 Data Ethics

- [ ] **資料來源合法性 Data Legality**：資料的蒐集與使用符合法規（如個資法、GDPR）
- [ ] **知情同意 Informed Consent**：若涉及個人資料，是否已取得同意
- [ ] **資料去識別化 De-identification**：是否已移除或遮蔽個人可辨識資訊 (PII)
- [ ] **資料偏誤 Data Bias**：是否分析了資料中可能存在的偏誤（性別、種族、年齡等）
- [ ] **資料代表性 Representativeness**：訓練資料是否合理代表目標群體

### 6.2 模型倫理 Model Ethics

- [ ] **公平性 Fairness**：模型對不同群體的表現是否有顯著差異
- [ ] **可解釋性 Explainability**：能否解釋模型為何做出特定預測
- [ ] **透明度 Transparency**：是否清楚揭露模型的能力範圍與限制
- [ ] **穩健性 Robustness**：模型在邊界情況 (Edge Cases) 下的表現如何
- [ ] **安全性 Safety**：模型的錯誤預測可能造成什麼影響

### 6.3 應用倫理 Application Ethics

- [ ] **使用場景 Use Cases**：是否明確定義了模型的預期使用場景
- [ ] **誤用風險 Misuse Risk**：是否分析了模型可能被誤用的方式
- [ ] **人類監督 Human Oversight**：是否設計了人類介入的機制
- [ ] **影響評估 Impact Assessment**：是否評估了模型對利害關係人的影響
- [ ] **持續監測 Ongoing Monitoring**：部署後是否有監測公平性與效能的計畫

### 6.4 溝通倫理 Communication Ethics

- [ ] **結果呈現 Result Presentation**：是否誠實呈現結果，不誇大模型效能
- [ ] **限制說明 Limitation Disclosure**：是否清楚說明模型的限制
- [ ] **不確定性量化 Uncertainty Quantification**：是否提供預測的信心程度

---

## 7. 課程知識總複習 Course Knowledge Review

### 7.1 第一階段：ML 基礎 (Week 1-10)

#### Week 1-2：基礎工具 Foundation Tools
- **核心概念**：ML/DL 的層次關係、EDA 流程
- **關鍵技能**：Python 環境建置、Pandas 資料處理、Matplotlib/Seaborn/Plotly 視覺化
- **精華公式**：描述性統計（平均值、標準差、相關係數）

#### Week 3：監督式學習框架 Supervised Learning Framework
- **核心概念**：Train/Test Split、k-Fold Cross-Validation、Bias-Variance Tradeoff
- **關鍵技能**：正確分割資料、避免資料洩漏 (Data Leakage)
- **精華口訣**：「訓練集學規律、驗證集調參數、測試集評效能」

#### Week 4：線性回歸 Linear Regression
- **核心概念**：損失函數 (Loss Function)、梯度下降 (Gradient Descent)、MSE
- **關鍵公式**：
  - 預測：$\hat{y} = w^T x + b$
  - 損失：$L = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$
  - 更新：$w \leftarrow w - \alpha \nabla L$
- **精華觀念**：學習率太大會震盪，太小會收斂慢

#### Week 5：分類 Classification
- **核心概念**：Logistic Regression、Sigmoid、決策邊界、ROC/PR 曲線
- **關鍵指標**：Accuracy, Precision, Recall, F1 Score, AUC
- **精華觀念**：不平衡資料集不要只看 Accuracy

#### Week 6：SVM
- **核心概念**：最大間隔 (Maximum Margin)、支持向量、核技巧 (Kernel Trick)
- **關鍵參數**：C（正則化強度）、gamma（核函數寬度）
- **精華觀念**：核技巧將低維不可分的資料映射到高維空間

#### Week 7：樹模型與集成 Tree Models & Ensemble
- **核心概念**：Decision Tree、Random Forest (Bagging)、GBDT (Boosting)
- **關鍵差異**：Bagging 降變異、Boosting 降偏差
- **精華觀念**：集成方法幾乎總是優於單一模型

#### Week 8：可解釋性 Explainability
- **核心概念**：Feature Importance、Permutation Importance、SHAP Values
- **關鍵圖表**：SHAP Summary Plot、SHAP Force Plot
- **精華觀念**：全域解釋 vs. 局部解釋，SHAP 兼具兩者

#### Week 9：特徵工程 Feature Engineering
- **核心概念**：One-Hot Encoding、StandardScaler、sklearn Pipeline
- **關鍵技能**：建立端到端的前處理管線
- **精華觀念**：好的特徵 > 好的模型

#### Week 10：超參數調校 Hyperparameter Tuning
- **核心概念**：Grid Search、Random Search、Learning Curve、Validation Curve
- **關鍵技能**：透過學習曲線診斷 overfitting/underfitting
- **精華觀念**：Random Search 在高維空間通常比 Grid Search 更有效率

### 7.2 第二階段：DL 進階 (Week 11-17)

#### Week 11：神經網路基礎 Neural Network Basics
- **核心概念**：Perceptron、Activation Functions (ReLU, Sigmoid, Tanh)、Backpropagation
- **正則化技術**：Dropout、L1/L2 Regularization、Batch Normalization
- **精華觀念**：ReLU 解決了梯度消失 (Vanishing Gradient) 問題

#### Week 12：CNN
- **核心概念**：卷積運算 (Convolution)、池化 (Pooling)、Feature Maps
- **視覺化技術**：Filter Visualization、Grad-CAM
- **精華觀念**：CNN 利用空間局部性 (Spatial Locality) 與權重共享 (Weight Sharing)

#### Week 13：RNN/Transformer
- **核心概念**：RNN、LSTM、GRU、Self-Attention、Transformer
- **關鍵架構**：Encoder-Decoder、Multi-Head Attention
- **精華觀念**：Transformer 的 Self-Attention 解決了 RNN 的長程依賴問題

#### Week 14：訓練技巧 Training Techniques
- **核心概念**：LR Scheduling (Step, Cosine, Warmup)、Early Stopping、Data Augmentation
- **精華觀念**：好的學習率排程可以大幅提升模型效能

#### Week 15：評估與公平性 Evaluation & Fairness
- **核心概念**：Confusion Matrix、Bias Detection、Fairness Metrics、Robustness
- **精華觀念**：模型效能好不代表模型公平

#### Week 16：MLOps
- **核心概念**：MLflow、Model Versioning、Model Serving、Model Monitoring
- **關鍵流程**：CI/CD for ML、Model Drift Detection
- **精華觀念**：部署只是開始，持續監測才是關鍵

#### Week 17：LLM 應用
- **核心概念**：Text Embeddings、RAG (Retrieval-Augmented Generation)、Prompt Engineering
- **關鍵技能**：建構 RAG 系統、設計有效的 Prompt
- **精華觀念**：LLM 不是萬能的，RAG 可以彌補知識不足

### 7.3 核心觀念對照表 Key Concepts Quick Reference

| 概念 | 傳統 ML | 深度學習 DL |
|------|--------|-----------|
| 特徵擷取 | 人工設計 (Hand-crafted) | 自動學習 (Learned) |
| 資料需求 | 較少 | 較多 |
| 可解釋性 | 較高 | 較低（需 XAI 工具） |
| 計算資源 | CPU 可處理 | 通常需 GPU |
| 典型問題 | 結構化資料 | 影像/文字/音訊 |
| 超參數調校 | Grid/Random Search | LR Scheduling + 多種技巧 |
| 部署複雜度 | 較低 | 較高（模型大、推論慢） |

---

## 8. 未來學習路徑建議 Future Learning Path

### 8.1 深化路線 Deepening Paths

```
路線 A：機器學習工程 ML Engineering
  ├── 進階統計學習（Bayesian Methods, Causal Inference）
  ├── 進階 MLOps（Kubeflow, Feature Store, A/B Testing）
  └── 系統設計（分散式訓練、高效推論）

路線 B：深度學習研究 DL Research
  ├── 進階架構（Vision Transformer, Diffusion Models, State Space Models）
  ├── 自監督學習（Self-Supervised Learning）
  └── 多模態學習（Multimodal Learning）

路線 C：應用專精 Domain Specialization
  ├── 自然語言處理 NLP（Fine-tuning LLM, RLHF）
  ├── 電腦視覺 CV（3D Vision, Video Understanding）
  ├── 推薦系統 RecSys（Deep Learning for RecSys）
  └── 生物資訊 Bioinformatics（AlphaFold, Drug Discovery）

路線 D：資料科學與分析 Data Science & Analytics
  ├── 進階統計與因果推論
  ├── 商業分析與策略
  └── 資料視覺化與敘事
```

### 8.2 推薦學習資源 Recommended Resources

| 類別 | 資源 | 說明 |
|------|------|------|
| 線上課程 | Stanford CS229 / CS231n / CS224n | ML/CV/NLP 經典課程 |
| 線上課程 | fast.ai Practical Deep Learning | 實作導向的 DL 課程 |
| 線上課程 | DeepLearning.AI Specialization | Andrew Ng 的系列課程 |
| 書籍 | "Hands-On Machine Learning" (Geron) | ML 實戰入門經典 |
| 書籍 | "Deep Learning" (Goodfellow et al.) | DL 理論經典 |
| 書籍 | "Designing Machine Learning Systems" (Huyen) | MLOps 與系統設計 |
| 實戰 | Kaggle Competitions | 競賽練功 |
| 實戰 | Papers with Code | 追蹤最新研究與實作 |
| 社群 | Hugging Face | 開源模型生態系 |
| 社群 | arXiv / Twitter(X) / Reddit r/MachineLearning | 追蹤最新動態 |

### 8.3 自學策略建議 Self-Learning Strategies

1. **建立作品集 Build a Portfolio**：在 GitHub 持續發布專案
2. **參加競賽 Join Competitions**：Kaggle、AI CUP 等
3. **閱讀論文 Read Papers**：每週至少讀一篇相關論文
4. **寫技術部落格 Write Blog Posts**：教是最好的學（費曼技巧 Feynman Technique）
5. **開源貢獻 Contribute to Open Source**：從小 PR 開始
6. **加入社群 Join Communities**：參加研討會、Meetup、線上論壇

---

## 9. ML/DL 職涯方向介紹 Career Paths in ML/DL

### 9.1 常見職位 Common Roles

| 職位 | 英文 | 核心技能 | 工作內容 |
|------|------|---------|---------|
| 資料科學家 | Data Scientist | 統計、ML、溝通 | 分析資料、建構模型、產出洞察 |
| 機器學習工程師 | ML Engineer | ML、軟體工程、MLOps | 將模型部署到生產環境 |
| 資料工程師 | Data Engineer | ETL、分散式系統、SQL | 建構資料管線與基礎設施 |
| 深度學習研究員 | DL Researcher | 數學、DL、論文撰寫 | 開發新演算法與架構 |
| AI 應用工程師 | AI Application Engineer | LLM、RAG、Prompt Engineering | 建構 AI 驅動的應用產品 |
| MLOps 工程師 | MLOps Engineer | DevOps、Kubernetes、Monitoring | 管理 ML 系統的生命週期 |
| 資料分析師 | Data Analyst | SQL、BI 工具、統計 | 產出報表、商業分析 |

### 9.2 技能矩陣 Skill Matrix

```
                    數學/統計  程式設計  領域知識  溝通能力  系統設計
資料科學家            ★★★★    ★★★     ★★★★    ★★★★    ★★
ML 工程師            ★★★     ★★★★★   ★★★     ★★      ★★★★
資料工程師            ★★      ★★★★★   ★★      ★★      ★★★★★
DL 研究員            ★★★★★   ★★★     ★★★★    ★★★     ★★
AI 應用工程師         ★★      ★★★★    ★★★     ★★★     ★★★
```

### 9.3 求職準備建議 Job Preparation Tips

1. **技術面試準備 Technical Interview Prep**
   - ML 基礎概念（偏差-變異、過擬合、評估指標）
   - 程式設計能力（LeetCode、SQL 練習）
   - 系統設計（ML System Design）
   - 實作題（現場建模、程式碼審查）

2. **作品集 Portfolio**
   - 3-5 個完整的 ML 專案，放在 GitHub
   - 含清楚的 README、可重現的程式碼
   - 展現從問題定義到部署的全流程能力

3. **履歷重點 Resume Highlights**
   - 量化成果（「將模型 AUC 從 0.82 提升至 0.91」）
   - 展示工具與技術棧
   - 強調端到端的專案經驗

### 9.4 產業趨勢 Industry Trends (2025-2026)

| 趨勢 | 說明 |
|------|------|
| LLM 應用爆發 | 企業大量採用 LLM 進行各類任務自動化 |
| AI Agent | 自主型 AI 代理成為新的應用典範 |
| 多模態 AI | 整合文字、影像、語音的模型越來越普遍 |
| Edge AI | 在邊緣裝置上部署 AI 的需求增長 |
| AI 治理 | AI 法規與倫理框架日益重要（EU AI Act 等） |
| MLOps 成熟 | ML 工程化、自動化流程成為標準 |
| 合成資料 | 使用 AI 生成訓練資料的實踐越來越多 |

---

## 關鍵詞彙 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 可重現性 | Reproducibility | 他人依據文件能重現相同結果 |
| 倫理檢核 | Ethics Checklist | 系統性地審查 AI 專案的倫理面向 |
| 資料洩漏 | Data Leakage | 訓練時意外使用了測試資料的資訊 |
| 公平性 | Fairness | 模型對不同群體的表現均衡 |
| 可解釋性 | Explainability | 能理解模型做出預測的原因 |
| 模型漂移 | Model Drift | 部署後模型效能隨時間下降 |
| 持續整合/部署 | CI/CD | 軟體開發的自動化整合與部署流程 |
| 知情同意 | Informed Consent | 資料主體知悉並同意其資料被使用 |
| 去識別化 | De-identification | 移除或遮蔽個人可辨識資訊 |
| 作品集 | Portfolio | 展示能力與經驗的專案合集 |

---

## 延伸閱讀 Further Reading
- "The Checklist Manifesto" — Atul Gawande（檢核清單的力量）
- "Responsible AI Practices" — Google AI（https://ai.google/responsibility/responsible-ai-practices/）
- "Model Cards for Model Reporting" — Mitchell et al., 2019
- "Datasheets for Datasets" — Gebru et al., 2021
- "Designing Machine Learning Systems" — Chip Huyen
- Papers with Code: https://paperswithcode.com/
