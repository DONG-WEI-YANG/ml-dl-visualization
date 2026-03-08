# 第 1 週：課程導論、Python 與資料科學環境
# Week 1: Introduction, Python & Data Science Environment

## 學習目標 Learning Objectives
1. 了解機器學習 (Machine Learning, ML) 與深度學習 (Deep Learning, DL) 的基本概念
2. 認識 ML/DL 在各領域的應用場景
3. 建立 Python 資料科學開發環境
4. 熟悉 Jupyter Notebook 與 Google Colab 操作
5. 認識本課程的互動平台與 AI 助教

---

## 1. 什麼是機器學習？ What is Machine Learning?

### 1.1 定義 Definition
機器學習是人工智慧 (Artificial Intelligence, AI) 的一個分支，讓電腦能夠從資料中學習規律，而不需要被明確地程式化 (explicitly programmed)。

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." — Tom Mitchell, 1997

### 1.2 ML 的三大類型 Three Types of ML

| 類型 | 英文 | 說明 | 範例 |
|------|------|------|------|
| 監督式學習 | Supervised Learning | 有標註資料 (Labeled Data) | 圖片分類、房價預測 |
| 非監督式學習 | Unsupervised Learning | 無標註資料 | 客戶分群、異常偵測 |
| 強化學習 | Reinforcement Learning | 透過獎勵信號學習 | 遊戲 AI、機器人控制 |

### 1.3 ML 與 DL 的關係 Relationship Between ML and DL

```
人工智慧 (AI)
  └── 機器學習 (ML)
        ├── 傳統方法：線性回歸、SVM、決策樹...
        └── 深度學習 (DL)
              ├── CNN（影像）
              ├── RNN/Transformer（序列/文字）
              └── GAN（生成）
```

深度學習是機器學習的子集，使用多層神經網路 (Neural Networks) 來自動學習資料的層次化表徵 (Hierarchical Representations)。

### 1.4 應用場景 Applications

| 領域 | 應用 | 技術 |
|------|------|------|
| 醫療 Healthcare | 醫學影像診斷 | CNN |
| 金融 Finance | 信用評分、詐欺偵測 | XGBoost, Random Forest |
| 自然語言處理 NLP | 翻譯、聊天機器人 | Transformer, LLM |
| 推薦系統 Recommendation | 商品/內容推薦 | Collaborative Filtering |
| 自動駕駛 Autonomous Driving | 物件偵測、路徑規劃 | CNN, RL |

---

## 2. Python 資料科學環境 Python Data Science Environment

### 2.1 為什麼選 Python？ Why Python?

- 豐富的科學計算生態系 (Rich Ecosystem)：NumPy, Pandas, Matplotlib, scikit-learn, PyTorch
- 社群活躍 (Active Community)：Stack Overflow, GitHub
- 語法簡潔 (Concise Syntax)：適合快速原型開發 (Rapid Prototyping)

### 2.2 環境安裝 Environment Setup

#### 方法一：Anaconda（推薦 Recommended）

1. 下載 Anaconda：https://www.anaconda.com/download
2. 安裝後開啟 Anaconda Prompt
3. 建立虛擬環境 (Virtual Environment)：

```bash
conda create -n ml-viz python=3.11
conda activate ml-viz
```

4. 安裝核心套件：

```bash
conda install numpy pandas matplotlib seaborn scikit-learn jupyter
pip install plotly torch torchvision
```

#### 方法二：pip + venv

```bash
python -m venv ml-viz
source ml-viz/bin/activate  # Linux/Mac
ml-viz\Scripts\activate     # Windows
pip install numpy pandas matplotlib seaborn scikit-learn jupyter plotly torch
```

### 2.3 驗證安裝 Verify Installation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"PyTorch: {torch.__version__}")
print("環境建置成功！ Environment setup complete!")
```

---

## 3. Jupyter Notebook 入門 Getting Started with Jupyter

### 3.1 啟動 Launch
```bash
jupyter notebook
```
或使用 JupyterLab：
```bash
jupyter lab
```

### 3.2 基本操作 Basic Operations
| 快捷鍵 | 功能 |
|--------|------|
| Shift + Enter | 執行目前 Cell 並移至下一個 |
| Ctrl + Enter | 執行目前 Cell |
| A | 在上方插入新 Cell |
| B | 在下方插入新 Cell |
| M | 切換為 Markdown |
| Y | 切換為 Code |
| DD | 刪除 Cell |

### 3.3 Google Colab 替代方案
- 網址：https://colab.research.google.com
- 優點：免安裝、提供免費 GPU
- 缺點：連線時間限制、資料需上傳

---

## 4. 課程平台導覽 Platform Overview

### 4.1 平台功能
- **視覺化互動區 Visualization Panel:** 調整參數即時觀察模型行為
- **AI 助教 AI Tutor:** 基於 LLM 的即時問答，採分層提示策略
- **作業提交 Assignment Submission:** 線上繳交與自動檢核
- **學習紀錄 Learning Analytics:** 追蹤學習進度與弱點

### 4.2 AI 助教使用規範 AI Tutor Guidelines
1. AI 助教會**引導**你思考，不會直接給答案
2. 提問時請提供：錯誤訊息、已嘗試的方法、你的理解
3. 作業模式下，需先提交思路才能獲得提示
4. 請勿將 AI 助教的回覆直接作為作業答案

---

## 關鍵詞彙 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 人工智慧 | Artificial Intelligence (AI) | 模擬人類智慧的技術總稱 |
| 機器學習 | Machine Learning (ML) | 從資料中學習規律的方法 |
| 深度學習 | Deep Learning (DL) | 使用多層神經網路的 ML 方法 |
| 神經網路 | Neural Network | 受生物神經啟發的計算模型 |
| 資料集 | Dataset | 用於訓練與評估的資料集合 |
| 特徵 | Feature | 描述資料的屬性 |
| 標籤 | Label | 監督式學習中的目標值 |
| 過擬合 | Overfitting | 模型過度適應訓練資料 |
| 欠擬合 | Underfitting | 模型無法捕捉資料規律 |

---

## 延伸閱讀 Further Reading
- Andrew Ng, "Machine Learning Yearning"
- scikit-learn 官方教學：https://scikit-learn.org/stable/tutorial/
- Google ML Crash Course：https://developers.google.com/machine-learning/crash-course
