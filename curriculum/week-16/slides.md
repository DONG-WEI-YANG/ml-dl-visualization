# 第 16 週投影片：MLOps 入門 — 模型版本、推論服務、監測

---

## Slide 1: 本週主題
# MLOps 入門
### Model Versioning, Inference Serving & Monitoring
- 從 Notebook 到生產環境
- 實驗追蹤、模型管理、推論服務、監測

---

## Slide 2: 為什麼需要 MLOps？
### 87% 的 ML 模型從未進入生產環境
| 問題 | 說明 |
|:---:|:---:|
| 可重現性 Reproducibility | 實驗結果無法復現 |
| 模型衰退 Model Decay | 上線後效能逐漸下降 |
| 環境不一致 | "在我機器上可以跑" |
| 手動流程 | 部署靠人工，容易出錯 |

**MLOps 就是解決這些問題的方法論**

---

## Slide 3: MLOps 定義
### MLOps = ML + DevOps + Data Engineering

```
     DevOps (自動化部署)
         ╲
          ╲
ML (建模)──── MLOps ────Data Eng (資料管線)
```

核心目標：讓模型**可靠地**從開發到生產，並**持續維護**。

---

## Slide 4: ML 生命週期
### ML Lifecycle — 持續循環

```
問題定義 → 資料收集 → 資料準備
    ↑                      ↓
    │                 訓練 / 評估
    │                      ↓
  持續改善 ← 模型監測 ← 模型部署
```

- 不是線性流程，而是**迭代循環 (Iterative Cycle)**
- 每個環節都需要自動化與版本控制

---

## Slide 5: MLOps 成熟度等級
### Google MLOps Maturity Model

| Level 0 | Level 1 | Level 2 |
|:---:|:---:|:---:|
| 手動 Manual | Pipeline 自動化 | CI/CD 自動化 |
| Script 驅動 | 持續訓練 CT | CI + CD + CT + CM |
| 無監測 | 有基本監測 | 全面自動化 |

**今天的目標：理解 Level 0 → Level 1 的關鍵步驟**

---

## Slide 6: Research-to-Production Gap
### 從研究到生產的鴻溝

| 研究環境 | 生產環境 |
|:---:|:---:|
| Jupyter Notebook | 模組化程式碼 |
| 小量資料 | 大規模資料流 |
| 手動實驗 | 自動化 Pipeline |
| 單次執行 | 24/7 服務 |
| 個人開發 | 團隊協作 |

---

## Slide 7: ML 系統的真相
### ML Code 只是冰山一角

```
╔═══════════════════════════════════════╗
║  資料收集 │ 資料驗證 │ 特徵工程 │ 監測  ║
║──────────────────────────────────────║
║  設定管理 │ ┌────────┐ │ 推論服務 ║
║           │ │ML Code │ │          ║
║           │ │ (很小!) │ │          ║
║──────────────────────────────────────║
║  基礎設施 │ 自動化工具 │ 資源管理  ║
╚═══════════════════════════════════════╝
```
— Google, "Hidden Technical Debt in ML Systems"

---

## Slide 8: 實驗追蹤 Experiment Tracking
### 為什麼？ — 實驗爆炸問題

手動記錄的災難：
- "lr=0.01 那次的 accuracy 是多少？"
- "上週二跑的模型用了哪些特徵？"
- "哪個版本的資料配哪個版本的程式碼？"

**解法：使用實驗追蹤工具自動記錄一切**

---

## Slide 9: MLflow 四大模組
### MLflow — 最受歡迎的開源 MLOps 平台

| 模組 | 功能 |
|:---:|:---:|
| Tracking | 記錄參數、指標、模型 |
| Projects | 可重現的程式碼打包 |
| Models | 統一模型格式 |
| Model Registry | 模型版本管理 |

---

## Slide 10: MLflow Tracking 實作
### 三行程式碼開始追蹤

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

- `log_param()` — 記錄超參數
- `log_metric()` — 記錄效能指標
- `log_model()` — 儲存模型
- `log_artifact()` — 儲存任意檔案

---

## Slide 11: MLflow UI Demo
### 視覺化比較實驗

```
┌─────────────────────────────────────────┐
│  Experiment: iris-classification        │
│─────────────────────────────────────────│
│  Run     │ n_est │ depth │ accuracy    │
│  run-001 │  50   │   5   │ 0.9200     │
│  run-002 │ 100   │  10   │ 0.9533  ◄  │
│  run-003 │ 200   │  15   │ 0.9467     │
│─────────────────────────────────────────│
│  [比較] [刪除] [下載模型]               │
└─────────────────────────────────────────┘
```

啟動：`mlflow ui --port 5000`

---

## Slide 12: Model Registry
### 模型版本的生命週期

```
None (新註冊)
  ↓
Staging (測試中)
  ↓ ← 通過驗證
Production (上線中)
  ↓ ← 被新版取代
Archived (已退役)
```

- 每個模型可以有多個版本 (Versions)
- 每個版本有明確的階段 (Stage)
- 支援模型描述、標籤、註解

---

## Slide 13: 模型部署三種方式
### Deployment Strategies

| 方式 | 延遲 | 適用場景 |
|:---:|:---:|:---:|
| REST API (Online) | 毫秒級 | 即時推薦、詐欺偵測 |
| Batch Inference | 分鐘~小時 | 報表、批量評分 |
| Edge Inference | 毫秒級 | 手機 App、IoT |

---

## Slide 14: FastAPI — 推論服務首選
### 為什麼選 FastAPI？

- 效能接近 Go / Node.js
- 自動產生 API 文件 (Swagger)
- Pydantic 資料驗證
- 原生 async 支援
- Python 生態完整整合

---

## Slide 15: FastAPI 推論 API 架構
### 完整流程

```
Client                    FastAPI Server
  │                           │
  │ POST /predict             │
  │ {"features": [...]}       │
  │ ─────────────────────→    │
  │                           │ 1. 驗證輸入 (Pydantic)
  │                           │ 2. 載入模型 (joblib)
  │                           │ 3. 執行推論
  │    {"prediction": "...",  │
  │     "probability": 0.97}  │
  │ ←─────────────────────    │
```

---

## Slide 16: 模型序列化
### 儲存模型的三種方式

| 格式 | 工具 | 優點 | 缺點 |
|:---:|:---:|:---:|:---:|
| Pickle | pickle | Python 原生 | 安全性低、版本相依 |
| Joblib | joblib | 大型陣列最佳化 | 僅限 Python |
| ONNX | onnx | 跨語言、跨平台 | 轉換有時需調整 |

---

## Slide 17: 模型監測 — 為什麼？
### 模型上線不是終點，而是起點

```
準確率
  ↑  ────┐
  │      │
  │      └──────┐
  │              └──────┐
  │                      └────── 模型衰退
  └────────────────────────────→ 時間
```

兩大元凶：**資料漂移** 與 **概念漂移**

---

## Slide 18: 資料漂移 Data Drift
### 輸入資料的分布改變

P_train(X) ≠ P_production(X)

```
    訓練資料           生產資料
    ╱╲                    ╱╲
   ╱  ╲               ╱╱    ╲
  ╱    ╲             ╱        ╲
 ╱      ╲           ╱          ╲
╱────────╲         ╱────────────╲
    μ₁                  μ₂ (偏移!)
```

偵測方法：KS Test, PSI, JS Divergence

---

## Slide 19: 概念漂移 Concept Drift
### 輸入與輸出的關係改變

P_train(Y|X) ≠ P_production(Y|X)

**三種型態：**
- 突變型 Sudden — 規則突然改變
- 漸變型 Gradual — 慢慢偏移
- 反覆型 Recurring — 季節性變化

**範例：** 疫情改變了消費行為

---

## Slide 20: 監測工具
### Model Monitoring Stack

```
┌────────────────────────────────────┐
│         Grafana Dashboard          │
│  ┌──────┐ ┌──────┐ ┌───────────┐  │
│  │準確率 │ │延遲  │ │資料漂移   │  │
│  │趨勢圖 │ │分布  │ │PSI 指標   │  │
│  └──────┘ └──────┘ └───────────┘  │
├────────────────────────────────────┤
│  Prometheus (指標收集)              │
├────────────────────────────────────┤
│  Evidently AI (漂移偵測)           │
├────────────────────────────────────┤
│  FastAPI (推論服務)                │
└────────────────────────────────────┘
```

---

## Slide 21: CI/CD for ML
### 比傳統軟體多了什麼？

```
傳統: Code → Build → Test → Deploy
ML:   Code → Build → Test → Deploy
      Data → Validate → ─────┘
      Model → Train → Evaluate ┘
```

觸發條件不只是程式碼變更，還有**資料變更**與**效能下降**

---

## Slide 22: Docker 容器化
### 解決環境一致性

```
開發環境          Docker 容器          生產環境
┌──────┐       ┌──────────┐       ┌──────┐
│Python │       │ Python   │       │ 完全 │
│3.11   │  →   │ 3.11     │  →    │ 相同 │
│各種   │  打包 │ 所有依賴  │ 部署  │ 的   │
│套件   │       │ + 模型   │       │ 環境 │
└──────┘       └──────────┘       └──────┘
```

一次建構，到處運行 — "Build Once, Run Anywhere"

---

## Slide 23: Dockerfile 範例
### 推論服務容器化

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py iris_model.pkl .
EXPOSE 8000
CMD ["uvicorn", "app:app",
     "--host", "0.0.0.0",
     "--port", "8000"]
```

```bash
docker build -t iris-api:v1 .
docker run -p 8000:8000 iris-api:v1
```

---

## Slide 24: ML Pipeline 自動化
### 把步驟串起來

```
資料擷取 → 資料前處理 → 特徵工程
                              ↓
         部署 ← 評估 ← 模型訓練
```

工具選擇：
- **Airflow** — 通用 DAG 排程
- **Kubeflow** — Kubernetes 原生
- **Prefect** — 現代 Python API
- **ZenML** — MLOps 整合框架

---

## Slide 25: MLOps 工具全景圖
### MLOps Landscape

| 環節 | 工具 |
|:---:|:---:|
| 實驗追蹤 | MLflow, W&B, Neptune |
| 版本控制 | Git, DVC, LakeFS |
| Pipeline | Airflow, Kubeflow, Prefect |
| 模型服務 | FastAPI, TFServing, Triton |
| 監測 | Evidently, Whylogs, Arize |
| 容器化 | Docker, Kubernetes |
| CI/CD | GitHub Actions, GitLab CI |

---

## Slide 26: 今日實作
### Hands-on Lab

1. 用 MLflow 追蹤 Iris 分類實驗
2. 將最佳模型註冊到 Model Registry
3. 用 FastAPI 建立推論 API
4. 測試不同序列化格式 (joblib / ONNX)

開啟 Week 16 Notebook！

---

## Slide 27: 本週作業預覽
### Assignment

- 建立完整的 MLOps 小型專案
- 包含：實驗追蹤 + 推論服務 + 資料漂移偵測
- 額外挑戰：Docker 容器化

---

## Slide 28: 重點回顧
### Key Takeaways

1. **MLOps** 讓模型從實驗室走向生產環境
2. **MLflow** 解決實驗追蹤與模型版本管理
3. **FastAPI** 是建立推論服務的高效工具
4. **資料漂移 & 概念漂移**是模型效能下降的主因
5. **Docker** 解決環境一致性問題
6. **ML Pipeline** 實現端到端自動化

---

## Slide 29: 下週預告
### Week 17: LLM 與大型語言模型
- Transformer 架構回顧
- Pre-training & Fine-tuning
- Prompt Engineering
- 使用 Hugging Face 生態系
- RAG（檢索增強生成）概念
