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

```svg
<figure class="md-figure">
<svg viewBox="0 0 720 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="三種 ML 類型的輸入輸出對比">
  <rect x="0" y="0" width="720" height="220" fill="#ffffff"/>
  <!-- Supervised -->
  <g transform="translate(20,20)">
    <rect x="0" y="0" width="220" height="180" rx="12" fill="#dbeafe" stroke="#1e40af" stroke-width="1.5"/>
    <text x="110" y="24" text-anchor="middle" font-size="14" fill="#1e3a8a" font-weight="700">監督式 Supervised</text>
    <!-- Input -->
    <rect x="20" y="40" width="80" height="40" fill="#ffffff" stroke="#1e40af"/>
    <text x="60" y="64" text-anchor="middle" font-size="11" fill="#1e3a8a">x (輸入)</text>
    <!-- Label -->
    <rect x="120" y="40" width="80" height="40" fill="#fef3c7" stroke="#b45309"/>
    <text x="160" y="64" text-anchor="middle" font-size="11" fill="#92400e">y (標註)</text>
    <!-- Arrow down -->
    <path d="M 110 82 L 110 100" stroke="#374151" stroke-width="1.5" marker-end="url(#typeArr)"/>
    <defs>
      <marker id="typeArr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#374151"/></marker>
    </defs>
    <rect x="20" y="105" width="180" height="32" fill="#1e40af"/>
    <text x="110" y="126" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="600">學習 f(x) → y</text>
    <text x="110" y="160" text-anchor="middle" font-size="11" fill="#1e3a8a">例：圖片分類、房價預測</text>
    <text x="110" y="174" text-anchor="middle" font-size="10" fill="#6b7280">linear / tree / NN</text>
  </g>
  <!-- Unsupervised -->
  <g transform="translate(250,20)">
    <rect x="0" y="0" width="220" height="180" rx="12" fill="#d1fae5" stroke="#059669" stroke-width="1.5"/>
    <text x="110" y="24" text-anchor="middle" font-size="14" fill="#065f46" font-weight="700">非監督 Unsupervised</text>
    <!-- Input only -->
    <rect x="65" y="40" width="90" height="40" fill="#ffffff" stroke="#059669"/>
    <text x="110" y="64" text-anchor="middle" font-size="11" fill="#065f46">x (僅輸入)</text>
    <path d="M 110 82 L 110 100" stroke="#374151" stroke-width="1.5" marker-end="url(#typeArr)"/>
    <rect x="20" y="105" width="180" height="32" fill="#059669"/>
    <text x="110" y="126" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="600">發現結構 / 群組</text>
    <text x="110" y="160" text-anchor="middle" font-size="11" fill="#065f46">例：客戶分群、異常偵測</text>
    <text x="110" y="174" text-anchor="middle" font-size="10" fill="#6b7280">k-means / PCA / DBSCAN</text>
  </g>
  <!-- Reinforcement -->
  <g transform="translate(480,20)">
    <rect x="0" y="0" width="220" height="180" rx="12" fill="#fee2e2" stroke="#dc2626" stroke-width="1.5"/>
    <text x="110" y="24" text-anchor="middle" font-size="14" fill="#7f1d1d" font-weight="700">強化 Reinforcement</text>
    <!-- Agent & Env cycle -->
    <rect x="15" y="40" width="80" height="36" fill="#ffffff" stroke="#dc2626"/>
    <text x="55" y="63" text-anchor="middle" font-size="11" fill="#7f1d1d">Agent</text>
    <rect x="125" y="40" width="80" height="36" fill="#ffffff" stroke="#dc2626"/>
    <text x="165" y="63" text-anchor="middle" font-size="11" fill="#7f1d1d">Env</text>
    <!-- Action arrow -->
    <path d="M 95 52 L 125 52" stroke="#374151" stroke-width="1.5" marker-end="url(#typeArr)"/>
    <text x="110" y="48" text-anchor="middle" font-size="9" fill="#374151">action</text>
    <!-- Reward arrow -->
    <path d="M 125 68 L 95 68" stroke="#d97706" stroke-width="1.5" marker-end="url(#typeArr)"/>
    <text x="110" y="84" text-anchor="middle" font-size="9" fill="#b45309">reward / state</text>
    <rect x="20" y="105" width="180" height="32" fill="#dc2626"/>
    <text x="110" y="126" text-anchor="middle" font-size="12" fill="#ffffff" font-weight="600">最大化長期累積獎勵</text>
    <text x="110" y="160" text-anchor="middle" font-size="11" fill="#7f1d1d">例：遊戲 AI、機器人控制</text>
    <text x="110" y="174" text-anchor="middle" font-size="10" fill="#6b7280">Q-learning / PPO / DQN</text>
  </g>
</svg>
<figcaption>示意圖：三種 ML 類型的輸入輸出對比。監督式有 (x, y) 配對，目標學 f(x)→y；非監督式僅有 x，目標發現潛在結構；強化學習由 agent 與環境互動，以獎勵信號引導長期決策策略。</figcaption>
</figure>
```

### 1.3 ML 與 DL 的關係 Relationship Between ML and DL

```svg
<figure class="md-figure">
<svg viewBox="0 0 600 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="AI / ML / DL 的巢狀關係">
  <rect x="0" y="0" width="600" height="320" fill="#ffffff"/>
  <!-- Outer ring: AI -->
  <ellipse cx="300" cy="160" rx="280" ry="135" fill="#eff6ff" stroke="#1e40af" stroke-width="2"/>
  <text x="300" y="40" text-anchor="middle" font-size="15" fill="#1e3a8a" font-weight="700">人工智慧 Artificial Intelligence (AI)</text>
  <text x="300" y="56" text-anchor="middle" font-size="10" fill="#6b7280">規則系統・專家系統・搜尋演算法・ML 以外的 AI 技術</text>
  <!-- Middle ring: ML -->
  <ellipse cx="300" cy="180" rx="220" ry="105" fill="#dbeafe" stroke="#1e40af" stroke-width="2"/>
  <text x="300" y="100" text-anchor="middle" font-size="14" fill="#1e3a8a" font-weight="700">機器學習 Machine Learning (ML)</text>
  <text x="155" y="185" font-size="11" fill="#1e3a8a">線性回歸</text>
  <text x="155" y="205" font-size="11" fill="#1e3a8a">SVM</text>
  <text x="155" y="225" font-size="11" fill="#1e3a8a">決策樹</text>
  <text x="400" y="185" font-size="11" fill="#1e3a8a">Random Forest</text>
  <text x="400" y="205" font-size="11" fill="#1e3a8a">XGBoost</text>
  <text x="400" y="225" font-size="11" fill="#1e3a8a">k-NN</text>
  <!-- Inner ring: DL -->
  <ellipse cx="300" cy="200" rx="130" ry="65" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="300" y="170" text-anchor="middle" font-size="13" fill="#7f1d1d" font-weight="700">深度學習 Deep Learning (DL)</text>
  <text x="235" y="200" font-size="11" fill="#7f1d1d">CNN</text>
  <text x="285" y="200" font-size="11" fill="#7f1d1d">Transformer</text>
  <text x="235" y="220" font-size="11" fill="#7f1d1d">RNN</text>
  <text x="285" y="220" font-size="11" fill="#7f1d1d">GAN / Diffusion</text>
  <text x="300" y="240" text-anchor="middle" font-size="10" fill="#991b1b">使用多層神經網路學習層次化表徵</text>
  <!-- Footer note -->
  <text x="300" y="308" text-anchor="middle" font-size="11" fill="#6b7280">巢狀關係：DL ⊂ ML ⊂ AI。DL 是 ML 的子集，專指基於深層神經網路的方法</text>
</svg>
<figcaption>示意圖：AI / ML / DL 的巢狀關係。人工智慧涵蓋所有「讓機器展現智慧行為」的技術；機器學習是其中從資料學習規律的子集；深度學習則是 ML 中運用多層神經網路自動學習特徵層次的方法。</figcaption>
</figure>
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
