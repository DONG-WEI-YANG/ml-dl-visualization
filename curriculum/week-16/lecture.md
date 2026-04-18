# 第 16 週：MLOps 入門 — 模型版本、推論服務、監測
# Week 16: Introduction to MLOps — Model Versioning, Inference Serving & Monitoring

## 學習目標 Learning Objectives
1. 理解 MLOps 的定義、核心原則與生命週期 (Lifecycle)
2. 認識從研究到生產的差距 (Research-to-Production Gap) 與應對策略
3. 使用 MLflow 進行實驗追蹤 (Experiment Tracking) 與模型版本管理 (Model Registry)
4. 掌握模型部署 (Model Deployment) 的主要方式：REST API、Batch、Edge
5. 使用 FastAPI 建立推論服務 (Inference Service)
6. 了解模型監測 (Model Monitoring) 中的資料漂移 (Data Drift) 與概念漂移 (Concept Drift)
7. 認識 CI/CD for ML、Docker 容器化與 ML Pipeline 自動化

---

## 1. MLOps 的定義與生命週期 MLOps Definition & Lifecycle

### 1.1 什麼是 MLOps？ What is MLOps?

MLOps（Machine Learning Operations）是一套結合機器學習 (ML)、軟體工程 (Software Engineering) 與 DevOps 實踐的方法論，目的是讓 ML 模型能夠可靠、高效、可重現地從開發走向生產環境，並持續維護。

> "MLOps is the discipline of deploying and maintaining machine learning models in production reliably and efficiently." — Chip Huyen

傳統軟體開發中，我們有 DevOps 來銜接開發 (Dev) 與運維 (Ops)。在 ML 專案中，情況更加複雜，因為除了程式碼之外，還需要管理**資料 (Data)**、**模型 (Model)** 與**實驗 (Experiment)**。

### 1.2 MLOps 的核心原則 Core Principles

| 原則 | 英文 | 說明 |
|------|------|------|
| 可重現性 | Reproducibility | 任何人在任何時間都能重現同樣的實驗結果 |
| 自動化 | Automation | 從資料處理到模型部署的流程應盡可能自動化 |
| 持續監測 | Continuous Monitoring | 模型上線後需持續追蹤效能與資料品質 |
| 版本控制 | Versioning | 程式碼、資料、模型、參數都需要版本控制 |
| 協作 | Collaboration | 資料科學家、ML 工程師、DevOps 團隊之間的高效協作 |
| 治理 | Governance | 模型的可審計性 (Auditability)、合規性 (Compliance) |

### 1.3 ML 生命週期 ML Lifecycle

```svg
<figure class="md-figure">
<svg viewBox="0 0 720 360" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MLOps 生命週期循環圖">
  <rect x="0" y="0" width="720" height="360" fill="#ffffff"/>
  <defs>
    <marker id="mlopsArr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#374151"/></marker>
  </defs>
  <text x="360" y="26" text-anchor="middle" font-size="14" fill="#111827" font-weight="600">MLOps 生命週期：六階段閉環</text>
  <!-- 6 stages arranged in a cycle -->
  <!-- Top row -->
  <!-- Stage 1: Problem Definition -->
  <rect x="40" y="60" width="180" height="70" rx="10" fill="#e0e7ff" stroke="#4338ca" stroke-width="1.5"/>
  <text x="130" y="85" text-anchor="middle" font-size="13" fill="#312e81" font-weight="700">1. 問題定義</text>
  <text x="130" y="103" text-anchor="middle" font-size="11" fill="#4338ca">Problem Definition</text>
  <text x="130" y="120" text-anchor="middle" font-size="9" fill="#6b7280">業務目標・成功指標・上線條件</text>
  <!-- Stage 2: Data Collection -->
  <rect x="270" y="60" width="180" height="70" rx="10" fill="#dbeafe" stroke="#1e40af" stroke-width="1.5"/>
  <text x="360" y="85" text-anchor="middle" font-size="13" fill="#1e3a8a" font-weight="700">2. 資料收集</text>
  <text x="360" y="103" text-anchor="middle" font-size="11" fill="#1e40af">Data Collection</text>
  <text x="360" y="120" text-anchor="middle" font-size="9" fill="#6b7280">資料來源・版本化・品質檢查</text>
  <!-- Stage 3: Data Preparation -->
  <rect x="500" y="60" width="180" height="70" rx="10" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>
  <text x="590" y="85" text-anchor="middle" font-size="13" fill="#92400e" font-weight="700">3. 資料準備</text>
  <text x="590" y="103" text-anchor="middle" font-size="11" fill="#b45309">Data Preparation</text>
  <text x="590" y="120" text-anchor="middle" font-size="9" fill="#6b7280">清洗・特徵工程・Pipeline</text>
  <!-- Bottom row (reverse order) -->
  <!-- Stage 6: Monitoring -->
  <rect x="40" y="210" width="180" height="70" rx="10" fill="#fecaca" stroke="#991b1b" stroke-width="1.5"/>
  <text x="130" y="235" text-anchor="middle" font-size="13" fill="#7f1d1d" font-weight="700">6. 模型監測</text>
  <text x="130" y="253" text-anchor="middle" font-size="11" fill="#991b1b">Monitoring</text>
  <text x="130" y="270" text-anchor="middle" font-size="9" fill="#6b7280">效能追蹤・drift 偵測・告警</text>
  <!-- Stage 5: Deployment -->
  <rect x="270" y="210" width="180" height="70" rx="10" fill="#fed7aa" stroke="#c2410c" stroke-width="1.5"/>
  <text x="360" y="235" text-anchor="middle" font-size="13" fill="#7c2d12" font-weight="700">5. 模型部署</text>
  <text x="360" y="253" text-anchor="middle" font-size="11" fill="#c2410c">Deployment</text>
  <text x="360" y="270" text-anchor="middle" font-size="9" fill="#6b7280">API / 批次・CI/CD・A/B 測試</text>
  <!-- Stage 4: Training/Evaluation -->
  <rect x="500" y="210" width="180" height="70" rx="10" fill="#d1fae5" stroke="#059669" stroke-width="1.5"/>
  <text x="590" y="235" text-anchor="middle" font-size="13" fill="#065f46" font-weight="700">4. 訓練 / 評估</text>
  <text x="590" y="253" text-anchor="middle" font-size="11" fill="#059669">Training / Eval</text>
  <text x="590" y="270" text-anchor="middle" font-size="9" fill="#6b7280">實驗追蹤・Model Registry</text>
  <!-- Arrows between stages -->
  <g stroke="#374151" stroke-width="1.8" fill="none">
    <!-- 1 → 2 -->
    <line x1="220" y1="95" x2="265" y2="95" marker-end="url(#mlopsArr)"/>
    <!-- 2 → 3 -->
    <line x1="450" y1="95" x2="495" y2="95" marker-end="url(#mlopsArr)"/>
    <!-- 3 → 4 (down right) -->
    <path d="M 590 130 L 590 205" marker-end="url(#mlopsArr)"/>
    <!-- 4 → 5 -->
    <line x1="500" y1="245" x2="455" y2="245" marker-end="url(#mlopsArr)"/>
    <!-- 5 → 6 -->
    <line x1="270" y1="245" x2="225" y2="245" marker-end="url(#mlopsArr)"/>
    <!-- 6 → back to 1 (big return arrow) -->
    <path d="M 130 210 L 130 160 Q 130 140 150 140 L 130 140" stroke-dasharray="0"/>
    <path d="M 130 210 L 130 145" marker-end="url(#mlopsArr)"/>
  </g>
  <!-- Central "Continuous" label -->
  <rect x="240" y="150" width="240" height="50" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text x="360" y="170" text-anchor="middle" font-size="12" fill="#7c2d12" font-weight="700">持續改善 Continuous Loop</text>
  <text x="360" y="188" text-anchor="middle" font-size="10" fill="#9a3412">CI / CD / CT / CM (持續整合・交付・訓練・監測)</text>
  <!-- Maturity level indicators -->
  <text x="360" y="316" text-anchor="middle" font-size="11" fill="#6b7280">Level 0 手動 → Level 1 Pipeline 自動化 → Level 2 CI/CD 全自動</text>
  <text x="360" y="338" text-anchor="middle" font-size="10" fill="#6b7280">成熟的 MLOps 系統讓「訓練新模型」從數週縮短至小時/分鐘，且每次變更皆可追溯</text>
</svg>
<figcaption>示意圖：MLOps 生命週期閉環。六個階段依序連結，監測階段（6）發現效能衰退時回饋到問題定義（1）觸發新一輪迭代。中央橘色虛框代表 CI/CD/CT/CM 四個持續性流程，Level 2 成熟度下所有階段都自動化。</figcaption>
</figure>
```

### 1.4 MLOps 成熟度等級 Maturity Levels

Google 提出了 MLOps 的三個成熟度等級：

| 等級 | 名稱 | 描述 |
|:---:|------|------|
| Level 0 | 手動流程 Manual Process | 模型訓練與部署皆為手動，沒有 CI/CD，無監測 |
| Level 1 | ML Pipeline 自動化 | 訓練流程自動化，持續訓練 (Continuous Training, CT)，但部署仍需人工 |
| Level 2 | CI/CD Pipeline 自動化 | 完整的自動化，包含持續整合 (CI)、持續交付 (CD)、持續訓練 (CT) 與持續監測 (CM) |

---

## 2. 從研究到生產的挑戰 Research-to-Production Gap

### 2.1 為什麼 87% 的 ML 專案無法上線？

根據 VentureBeat 的調查，大約 87% 的 ML 模型從未進入生產環境。主要挑戰包括：

| 類別 | 挑戰 | 說明 |
|------|------|------|
| 資料 | 資料品質 Data Quality | 生產資料往往比研究資料更嘈雜、有缺漏 |
| 資料 | 資料管線 Data Pipeline | 需要穩定、可擴展的資料抽取-轉換-載入 (ETL) 流程 |
| 模型 | 可重現性 Reproducibility | Notebook 中的實驗難以重現 |
| 模型 | 模型衰退 Model Decay | 模型效能隨時間下降 |
| 工程 | 技術債 Technical Debt | ML 系統比傳統軟體有更多隱性技術債 |
| 工程 | 服務化 Serving | 從 `.pkl` 檔到高可用 API 有巨大落差 |
| 組織 | 團隊協作 | 資料科學家與工程師之間的溝通鴻溝 |

### 2.2 ML 系統的隱性技術債 Hidden Technical Debt

Google 在 2015 年發表了經典論文 *"Hidden Technical Debt in Machine Learning Systems"*，指出 ML 系統中，模型程式碼只佔整體的一小部分：

```
┌─────────────────────────────────────────────────────────┐
│                    ML 系統全貌                            │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  資料收集  │  資料驗證  │  特徵抽取  │  資料分析  │    │
│  ├───────────────────┬─────────────────────────────┤    │
│  │  Process Mgmt     │  ┌─────────┐  │  Serving    │    │
│  │  資源管理          │  │ ML Code │  │  推論服務   │    │
│  │  Configuration    │  │ (很小！) │  │  Monitoring │    │
│  ├───────────────────┴──┴─────────┴──┴─────────────┤    │
│  │  機器資源管理  │  Infrastructure  │  自動化工具    │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 2.3 Notebook 到生產的常見問題

| 問題 | 說明 | 解決方案 |
|------|------|----------|
| 全域狀態 Global State | Cell 執行順序影響結果 | 模組化程式碼 |
| 硬編碼路徑 Hardcoded Paths | 換環境就壞掉 | 使用設定檔 (Config Files) |
| 缺乏測試 No Tests | 沒有單元測試 | 加入 pytest |
| 依賴不明 Unclear Dependencies | `pip install` 散落各處 | requirements.txt / pyproject.toml |
| 無法版控 Not Version-Controlled | `.ipynb` JSON 格式不利於 diff | 搭配 `.py` 模組使用 |

---

## 3. 實驗追蹤 Experiment Tracking：MLflow 介紹

### 3.1 為什麼需要實驗追蹤？

在模型開發過程中，資料科學家通常會執行數十甚至數百次實驗：調整超參數 (Hyperparameters)、更換特徵 (Features)、嘗試不同演算法。如果只靠手動記錄（例如 Excel 或筆記），很快就會失控。

**實驗追蹤的好處：**
- 自動記錄每次實驗的參數 (Parameters)、指標 (Metrics)、成品 (Artifacts)
- 方便比較不同實驗的結果
- 支援團隊協作與知識分享
- 確保實驗的可重現性 (Reproducibility)

### 3.2 MLflow 概覽

[MLflow](https://mlflow.org/) 是由 Databricks 開發的開源 MLOps 平台，提供四大核心模組：

| 模組 | 英文 | 功能 |
|------|------|------|
| 追蹤 | MLflow Tracking | 記錄實驗參數、指標、模型 |
| 專案 | MLflow Projects | 可重現的程式碼打包格式 |
| 模型 | MLflow Models | 統一的模型格式與部署介面 |
| 登錄 | MLflow Model Registry | 模型版本管理與生命週期管理 |

### 3.3 MLflow Tracking 核心概念

```python
import mlflow

# 開始一個實驗 Start an experiment
mlflow.set_experiment("my-classification-experiment")

with mlflow.start_run(run_name="random-forest-v1"):
    # 記錄參數 Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 42)

    # ... 訓練模型 ...

    # 記錄指標 Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("f1_score", 0.93)
    mlflow.log_metric("auc_roc", 0.97)

    # 記錄模型 Log model
    mlflow.sklearn.log_model(model, "random-forest-model")

    # 記錄額外檔案 Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
```

**核心術語：**

| 術語 | 英文 | 說明 |
|------|------|------|
| 實驗 | Experiment | 一組相關的 Run，例如「分類任務 v2」 |
| 運行 | Run | 單次訓練過程的完整記錄 |
| 參數 | Parameter | 輸入設定，如 learning_rate=0.01 |
| 指標 | Metric | 效能衡量，如 accuracy=0.95 |
| 成品 | Artifact | 產出物，如模型檔案、圖片、資料 |

### 3.4 MLflow UI

MLflow 提供 Web UI 來檢視與比較實驗結果：

```bash
# 啟動 MLflow UI
mlflow ui --port 5000
# 瀏覽器開啟 http://localhost:5000
```

在 UI 上可以：
- 檢視所有實驗與 Run
- 比較不同 Run 的參數與指標
- 以圖表視覺化指標變化
- 下載模型與成品

---

## 4. 模型版本管理 Model Registry

### 4.1 為什麼需要 Model Registry？

當模型數量增加，你需要一個中央化的模型倉庫來管理：
- **哪些模型是目前正在線上服務的？** (Production)
- **哪些模型正在測試中？** (Staging)
- **舊版模型是否需要保留？** (Archived)

### 4.2 MLflow Model Registry

MLflow Model Registry 提供完整的模型生命週期管理：

```python
import mlflow

# 註冊模型 Register a model
result = mlflow.register_model(
    model_uri="runs:/<run_id>/model",
    name="fraud-detection-model"
)

# 模型階段轉換 Transition model stage
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="fraud-detection-model",
    version=3,
    stage="Production"   # None → Staging → Production → Archived
)

# 載入特定版本的模型 Load a specific version
model = mlflow.pyfunc.load_model(
    model_uri="models:/fraud-detection-model/Production"
)
```

### 4.3 模型版本管理流程

```
開發 Development
    │
    ▼
┌─────────┐     ┌─────────┐     ┌──────────┐     ┌──────────┐
│  None    │ ──→ │ Staging │ ──→ │Production│ ──→ │ Archived │
│ (新註冊) │     │ (測試中) │     │ (上線中)  │     │ (已退役)  │
└─────────┘     └─────────┘     └──────────┘     └──────────┘
                     │
                     ▼
               自動化測試 & 驗證
               (A/B Test, Shadow Mode)
```

### 4.4 其他 Model Registry 工具

| 工具 | 特色 | 適用場景 |
|------|------|----------|
| MLflow Model Registry | 開源、與 MLflow 整合 | 通用場景 |
| Weights & Biases (W&B) | 強大的視覺化、團隊協作 | 研究導向團隊 |
| DVC (Data Version Control) | 資料 + 模型版本控制 | 資料密集場景 |
| Neptune.ai | 靈活的 metadata 管理 | 企業環境 |
| Vertex AI Model Registry | GCP 原生 | Google Cloud 用戶 |

---

## 5. 模型部署方式 Model Deployment Strategies

### 5.1 三種主要部署方式

| 部署方式 | 英文 | 特點 | 適用場景 |
|----------|------|------|----------|
| 即時推論 | REST API (Online) | 低延遲、即時回應 | 詐欺偵測、推薦系統 |
| 批次推論 | Batch Inference | 定期處理大量資料 | 月報、信用評分更新 |
| 邊緣推論 | Edge Inference | 模型部署在終端設備 | IoT、手機 App、自駕車 |

### 5.2 REST API 部署

最常見的線上部署方式。客戶端 (Client) 發送 HTTP 請求，伺服器回傳預測結果。

```
客戶端 Client                     伺服器 Server
    │                                  │
    │  POST /predict                   │
    │  {"features": [5.1, 3.5, ...]}   │
    │  ─────────────────────────────→  │
    │                                  │  載入模型
    │                                  │  執行推論
    │  {"prediction": "setosa",        │
    │   "probability": 0.97}           │
    │  ←─────────────────────────────  │
    │                                  │
```

### 5.3 Batch Inference

適合不需要即時結果的場景：

```python
import pandas as pd
import joblib

# 載入模型
model = joblib.load("model.pkl")

# 批次讀取資料
df = pd.read_csv("new_data.csv")

# 批次推論
predictions = model.predict(df[feature_columns])

# 儲存結果
df["prediction"] = predictions
df.to_csv("predictions_output.csv", index=False)
```

### 5.4 Edge Inference

將模型部署到邊緣設備（手機、嵌入式系統、瀏覽器）：

| 框架 | 目標平台 | 模型格式 |
|------|----------|----------|
| TensorFlow Lite | Android/iOS | .tflite |
| Core ML | Apple 裝置 | .mlmodel |
| ONNX Runtime | 跨平台 | .onnx |
| TensorRT | NVIDIA GPU | .engine |

---

## 6. 使用 FastAPI 建立推論服務 Building an Inference Service with FastAPI

### 6.1 為什麼選 FastAPI？

| 特性 | 說明 |
|------|------|
| 高效能 High Performance | 基於 ASGI，效能接近 Node.js / Go |
| 自動文件 Auto Docs | 自動產生 Swagger UI / ReDoc |
| 型別提示 Type Hints | 利用 Pydantic 做資料驗證 |
| 非同步 Async Support | 原生支援 async/await |
| 易於學習 Easy to Learn | Flask 風格的路由 (Routing) 設計 |

### 6.2 完整推論 API 範例

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# 定義請求格式 Request Schema
class PredictionRequest(BaseModel):
    features: list[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

# 定義回應格式 Response Schema
class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model_version: str

# 初始化 App
app = FastAPI(
    title="Iris Classification API",
    description="使用訓練好的模型進行鳶尾花分類",
    version="1.0.0"
)

# 載入模型（啟動時載入一次）
model = joblib.load("iris_model.pkl")
class_names = ["setosa", "versicolor", "virginica"]

@app.get("/health")
def health_check():
    """健康檢查端點 Health Check Endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """執行推論 Run Inference"""
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features).max()

        return PredictionResponse(
            prediction=class_names[prediction],
            probability=round(float(probability), 4),
            model_version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
def predict_batch(requests: list[PredictionRequest]):
    """批次推論 Batch Inference"""
    features = np.array([r.features for r in requests])
    predictions = model.predict(features)
    probabilities = model.predict_proba(features).max(axis=1)

    return [
        {
            "prediction": class_names[pred],
            "probability": round(float(prob), 4)
        }
        for pred, prob in zip(predictions, probabilities)
    ]
```

### 6.3 啟動與測試

```bash
# 啟動服務
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 測試推論（使用 curl）
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# 或使用 Python requests
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
print(response.json())
# {"prediction": "setosa", "probability": 0.9733, "model_version": "1.0.0"}
```

瀏覽 `http://localhost:8000/docs` 即可看到自動產生的 Swagger UI 文件。

---

## 7. 模型監測 Model Monitoring

### 7.1 為什麼需要模型監測？

模型上線後，效能會隨著時間下降，這稱為**模型衰退 (Model Decay / Model Degradation)**。原因主要有二：

### 7.2 資料漂移 Data Drift

**定義：** 輸入資料的分布 (Distribution) 隨時間改變，但輸入與輸出之間的關係不變。

**數學表示：** P_train(X) ≠ P_production(X)，但 P(Y|X) 不變。

**範例：** 電商推薦系統在疫情期間，使用者購物行為大幅改變（例如居家用品需求暴增）。

**偵測方法：**

| 方法 | 英文 | 說明 |
|------|------|------|
| KS 檢定 | Kolmogorov-Smirnov Test | 比較兩個分布的最大差異 |
| PSI | Population Stability Index | 衡量分布變化的穩定性指標 |
| JS 散度 | Jensen-Shannon Divergence | 衡量兩個分布的相似度 |
| 卡方檢定 | Chi-squared Test | 適用於類別特徵 |

```python
from scipy.stats import ks_2samp

# 比較訓練資料與生產資料的特徵分布
statistic, p_value = ks_2samp(train_feature, production_feature)

if p_value < 0.05:
    print("警告：偵測到資料漂移！Data drift detected!")
```

### 7.3 概念漂移 Concept Drift

**定義：** 輸入與輸出之間的關係 (Relationship) 本身發生改變。

**數學表示：** P_train(Y|X) ≠ P_production(Y|X)

**範例：** 信用評分模型中，經濟衰退導致相同收入水準的違約率上升，原本「低風險」的客戶變成「高風險」。

**概念漂移的類型：**

```
指標值
  ↑
  │    ┌──┐
  │    │  │ 突變型 Sudden
  │────┘  └────────
  │
  │    ╱‾‾‾‾‾‾‾
  │   ╱  漸變型 Gradual
  │──╱
  │
  │  ┌─┐   ┌─┐
  │  │ │   │ │ 反覆型 Recurring
  │──┘ └───┘ └──
  │
  └──────────────→ 時間
```

### 7.4 監測指標 Monitoring Metrics

| 類別 | 指標 | 說明 |
|------|------|------|
| 模型效能 | Accuracy, F1, AUC | 需要 Ground Truth Label |
| 資料品質 | 缺失率、異常值比例 | 無需標籤即可監測 |
| 資料分布 | PSI, KS Test | 偵測特徵分布變化 |
| 預測分布 | 預測類別比例、信心度 | 偵測預測行為異常 |
| 系統效能 | 延遲 (Latency)、吞吐量 (Throughput) | 服務品質監測 |

### 7.5 監測工具

| 工具 | 類型 | 特色 |
|------|------|------|
| Evidently AI | 開源 | 資料漂移報告、視覺化 Dashboard |
| Whylogs | 開源 | 輕量級資料 Profiling |
| Prometheus + Grafana | 開源 | 系統指標監測與告警 |
| Arize AI | 商用 | 企業級 ML 觀測平台 |
| Fiddler AI | 商用 | 可解釋性 + 監測整合 |

---

## 8. CI/CD for ML 概念

### 8.1 傳統 CI/CD vs. ML CI/CD

| 面向 | 傳統軟體 CI/CD | ML CI/CD |
|------|----------------|----------|
| 測試對象 | 程式碼 | 程式碼 + 資料 + 模型 |
| 建構產物 | 可執行檔 / 容器 | 可執行檔 + 訓練好的模型 |
| 觸發條件 | 程式碼變更 | 程式碼變更 OR 資料變更 OR 排程 |
| 驗證標準 | 通過單元測試 | 通過測試 + 模型效能達標 |

### 8.2 ML CI/CD Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  程式碼   │    │ 自動訓練  │    │ 模型驗證  │    │ 自動部署  │
│  變更     │ →  │ Auto     │ →  │ Model    │ →  │ Auto     │
│  Code    │    │ Training │    │ Validate │    │ Deploy   │
│  Change  │    │          │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     CI               CT              CI/CD            CD
```

### 8.3 ML 專案的測試類型

| 測試類型 | 說明 | 範例 |
|----------|------|------|
| 單元測試 Unit Tests | 測試個別函式 | 特徵工程函式的輸出格式 |
| 整合測試 Integration Tests | 測試元件間的互動 | 資料管線端到端 |
| 資料驗證 Data Validation | 確認資料品質 | Schema 檢查、分布檢查 |
| 模型驗證 Model Validation | 確認模型品質 | 效能指標是否達到閾值 |
| A/B 測試 | 線上比較新舊模型 | 新模型 vs. 舊模型的轉換率 |

---

## 9. Docker 容器化基礎 Docker Containerization Basics

### 9.1 為什麼要用 Docker？

ML 模型部署面臨「在我機器上可以跑」(Works on My Machine) 的問題。Docker 透過容器化 (Containerization) 解決環境一致性問題。

| 概念 | 說明 |
|------|------|
| 映像檔 Image | 打包好的環境 + 應用程式（唯讀模板） |
| 容器 Container | 映像檔的運行實例 |
| Dockerfile | 定義如何建構映像檔的指令檔 |
| Registry | 儲存與分享映像檔的倉庫（如 Docker Hub） |

### 9.2 ML 推論服務的 Dockerfile 範例

```dockerfile
# 使用 Python 官方映像檔
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製依賴檔案
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式與模型
COPY app.py .
COPY iris_model.pkl .

# 暴露端口
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

# 啟動命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 9.3 建構與運行

```bash
# 建構映像檔 Build image
docker build -t iris-inference:v1.0 .

# 運行容器 Run container
docker run -d -p 8000:8000 --name iris-api iris-inference:v1.0

# 檢視運行中的容器
docker ps

# 查看容器日誌
docker logs iris-api
```

### 9.4 Docker Compose 多服務編排

在生產環境中，推論服務通常搭配其他元件（如 Redis 快取、Prometheus 監測）：

```yaml
# docker-compose.yml
version: "3.8"
services:
  inference:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/iris_model.pkl
    volumes:
      - ./models:/models

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

---

## 10. ML Pipeline 自動化

### 10.1 什麼是 ML Pipeline？

ML Pipeline 是將 ML 工作流程中的各個步驟串連成自動化流程，每個步驟的輸出作為下一步的輸入：

```
┌──────┐   ┌──────────┐   ┌──────────┐   ┌──────┐   ┌──────┐   ┌──────┐
│ 資料  │ → │ 資料前處理 │ → │ 特徵工程  │ → │ 訓練  │ → │ 評估  │ → │ 部署  │
│ 擷取  │   │ Data     │   │ Feature  │   │Train │   │Eval  │   │Deploy│
│Ingest │   │ Process  │   │ Engineer │   │      │   │      │   │      │
└──────┘   └──────────┘   └──────────┘   └──────┘   └──────┘   └──────┘
```

### 10.2 Pipeline 工具比較

| 工具 | 開發者 | 特點 | 適用場景 |
|------|--------|------|----------|
| Apache Airflow | Apache | 通用 DAG 排程 | 資料工程 + ML |
| Kubeflow Pipelines | Google | Kubernetes 原生 | 雲端大規模 |
| Prefect | Prefect | 現代化 Python API | 中型團隊 |
| ZenML | ZenML | MLOps 框架整合 | 標準化 ML 流程 |
| Metaflow | Netflix | 簡潔 Python API | 資料科學家友善 |

### 10.3 簡單的 Pipeline 範例（使用 Python）

```python
# pipeline.py — 簡化版 ML Pipeline

from dataclasses import dataclass
from typing import Any

@dataclass
class PipelineStep:
    name: str
    function: callable

class MLPipeline:
    def __init__(self, name: str):
        self.name = name
        self.steps: list[PipelineStep] = []
        self.artifacts: dict[str, Any] = {}

    def add_step(self, name: str, func: callable):
        self.steps.append(PipelineStep(name=name, function=func))
        return self

    def run(self):
        print(f"開始執行 Pipeline: {self.name}")
        for step in self.steps:
            print(f"  執行步驟: {step.name}")
            result = step.function(self.artifacts)
            self.artifacts.update(result or {})
        print(f"Pipeline 完成！")
        return self.artifacts

# 使用範例
def load_data(artifacts):
    from sklearn.datasets import load_iris
    data = load_iris()
    return {"X": data.data, "y": data.target}

def train_model(artifacts):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(artifacts["X"], artifacts["y"])
    return {"model": model}

def evaluate_model(artifacts):
    score = artifacts["model"].score(artifacts["X"], artifacts["y"])
    print(f"    Accuracy: {score:.4f}")
    return {"accuracy": score}

pipeline = MLPipeline("iris-training")
pipeline.add_step("載入資料", load_data)
pipeline.add_step("訓練模型", train_model)
pipeline.add_step("評估模型", evaluate_model)
pipeline.run()
```

---

## 關鍵詞彙表 Glossary

| 中文 | 英文 | 說明 |
|------|------|------|
| 機器學習運維 | MLOps | 結合 ML + DevOps 的實踐方法論 |
| 實驗追蹤 | Experiment Tracking | 自動記錄實驗參數、指標與產物 |
| 模型登錄 | Model Registry | 中央化的模型版本管理系統 |
| 模型部署 | Model Deployment | 將模型發布為可使用的服務 |
| 推論 | Inference | 使用訓練好的模型對新資料做預測 |
| 推論服務 | Inference Service / Model Serving | 提供推論功能的 API 服務 |
| 資料漂移 | Data Drift | 輸入資料分布隨時間改變 |
| 概念漂移 | Concept Drift | 輸入與輸出的關係隨時間改變 |
| 模型衰退 | Model Decay / Model Degradation | 模型效能隨時間下降 |
| 持續整合 | Continuous Integration (CI) | 自動化程式碼測試與建構 |
| 持續交付 | Continuous Delivery (CD) | 自動化程式碼部署 |
| 持續訓練 | Continuous Training (CT) | 自動化模型重新訓練 |
| 容器化 | Containerization | 使用 Docker 打包應用與環境 |
| 映像檔 | Image | Docker 容器的唯讀模板 |
| 管線 | Pipeline | 自動化的工作流程 |
| 序列化 | Serialization | 將物件轉換為可儲存的格式 |
| 資料驗證 | Data Validation | 確認資料符合預期的格式與品質 |
| A/B 測試 | A/B Testing | 線上比較新舊版本效能的方法 |
| 影子模式 | Shadow Mode | 新模型平行運行但不實際服務 |
| 技術債 | Technical Debt | 為求快速開發而累積的維護成本 |

---

## 延伸閱讀 Further Reading

- Chip Huyen, *"Designing Machine Learning Systems"* (O'Reilly, 2022)
- Google Cloud, ["MLOps: Continuous delivery and automation pipelines in machine learning"](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- D. Sculley et al., *"Hidden Technical Debt in Machine Learning Systems"* (NeurIPS 2015)
- MLflow 官方文件：https://mlflow.org/docs/latest/index.html
- FastAPI 官方教學：https://fastapi.tiangolo.com/tutorial/
- Evidently AI 文件：https://docs.evidentlyai.com/
- Docker 官方入門：https://docs.docker.com/get-started/
