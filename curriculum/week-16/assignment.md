# 第 16 週作業：MLOps 實踐 — 實驗追蹤、推論服務與模型監測
# Week 16 Assignment: MLOps in Practice — Experiment Tracking, Inference Serving & Monitoring

**繳交期限 Due:** 下週上課前
**繳交方式 Submission:** 上傳專案資料夾（含程式碼、截圖、報告）至課程平台
**難度等級 Difficulty:** 進階 Advanced

---

## 作業一：MLflow 實驗追蹤（25%）

使用 scikit-learn 的任一分類資料集（Iris、Wine、Breast Cancer 等），完成以下任務：

### 1.1 至少執行 5 次實驗 Run

每次實驗需：
- 使用不同的演算法或超參數組合
- 記錄所有參數 (`log_param`)
- 記錄至少 3 個評估指標 (`log_metric`)：accuracy, f1_score, 以及一個你自選的指標
- 記錄模型 (`log_model`)

### 1.2 比較實驗

- 使用 MLflow UI 比較所有實驗結果
- 截圖 MLflow UI 的比較頁面（含參數與指標表格）
- 寫一段分析（100-200 字）：哪個模型表現最好？為什麼？

### 1.3 程式碼要求

```python
# 提示：你的程式碼結構應類似這樣
import mlflow
from sklearn.model_selection import train_test_split

mlflow.set_experiment("week16-assignment")

experiments = [
    {"model": "RandomForest", "params": {"n_estimators": 100, "max_depth": 5}},
    {"model": "RandomForest", "params": {"n_estimators": 200, "max_depth": 10}},
    # ... 至少 5 組
]

for exp in experiments:
    with mlflow.start_run(run_name=f"{exp['model']}-{exp['params']}"):
        # 訓練、評估、記錄
        pass
```

---

## 作業二：Model Registry 操作（15%）

### 2.1 註冊最佳模型

- 從作業一的實驗中，選出最佳模型
- 將其註冊到 MLflow Model Registry
- 模型名稱格式：`week16-<你的學號>-<資料集名稱>`

### 2.2 版本管理

- 將模型版本從 `None` 轉換為 `Staging`
- 添加模型描述 (Description) 與標籤 (Tags)
- 截圖 Model Registry 頁面，顯示模型版本與狀態

### 2.3 載入與使用

```python
# 從 Registry 載入模型進行推論
import mlflow

model = mlflow.pyfunc.load_model("models:/<model_name>/Staging")
prediction = model.predict(X_test[:5])
print(prediction)
```

---

## 作業三：FastAPI 推論服務（30%）

### 3.1 建立推論 API

建立一個 FastAPI 應用程式，包含以下端點 (Endpoints)：

| 端點 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康檢查，回傳 `{"status": "healthy"}` |
| `/model/info` | GET | 回傳模型資訊（名稱、版本、特徵名稱） |
| `/predict` | POST | 單筆推論，接收特徵值，回傳預測結果與機率 |
| `/predict/batch` | POST | 批次推論，接收多筆資料 |

### 3.2 資料驗證

使用 Pydantic BaseModel 定義請求 (Request) 與回應 (Response) 的資料格式：

```python
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=4, max_length=4)

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model_version: str
```

### 3.3 測試

- 使用 Swagger UI (`/docs`) 測試所有端點
- 截圖成功的推論結果（含請求與回應）
- 使用 Python `requests` 或 `curl` 撰寫至少 3 個測試案例

### 3.4 錯誤處理

- 處理輸入格式錯誤（如特徵數量不對）
- 處理無效的特徵值（如 NaN、極端值）
- 回傳適當的 HTTP 狀態碼與錯誤訊息

---

## 作業四：模型序列化比較（10%）

### 4.1 比較不同格式

將作業一的最佳模型分別以三種格式儲存：

1. **pickle** — `model.pkl`
2. **joblib** — `model.joblib`
3. **ONNX** — `model.onnx`（選做 Optional）

### 4.2 比較表

製作比較表，包含以下資訊：

| 格式 | 檔案大小 | 儲存時間 | 載入時間 | 推論時間 (1000筆) |
|------|----------|----------|----------|-------------------|
| pickle | ? KB | ? ms | ? ms | ? ms |
| joblib | ? KB | ? ms | ? ms | ? ms |
| ONNX | ? KB | ? ms | ? ms | ? ms |

---

## 作業五：資料漂移偵測（20%）

### 5.1 模擬資料漂移

使用以下方法模擬資料漂移：

```python
import numpy as np

# 原始資料分布
X_train = np.random.normal(loc=0, scale=1, size=(1000, 4))

# 模擬漂移後的資料（平均值偏移）
X_drifted = np.random.normal(loc=0.5, scale=1.2, size=(1000, 4))
```

### 5.2 偵測漂移

使用至少**兩種方法**偵測資料漂移：

1. **KS 檢定 (Kolmogorov-Smirnov Test)**
2. **PSI (Population Stability Index)** 或 **JS Divergence**

### 5.3 視覺化

- 繪製訓練資料與漂移資料的特徵分布對比圖
- 使用直方圖或 KDE 圖顯示分布差異
- 標註漂移指標的數值

### 5.4 分析報告

撰寫 200-300 字的分析：
- 哪些特徵發生了顯著漂移？
- 漂移的程度如何？
- 如果這是真實的生產環境，你會建議什麼對策？

---

## 加分挑戰 Bonus Challenge（+10%）

### Docker 容器化

將作業三的 FastAPI 推論服務容器化：

1. 撰寫 `Dockerfile`
2. 撰寫 `requirements.txt`
3. 成功建構映像檔並運行容器
4. 從容器外部成功呼叫 API
5. 截圖 `docker ps` 與成功的 API 回應

---

## 繳交清單 Submission Checklist

- [ ] 作業一：MLflow 實驗程式碼 + MLflow UI 截圖 + 分析文字
- [ ] 作業二：Model Registry 操作程式碼 + 截圖
- [ ] 作業三：FastAPI 程式碼 (`app.py`) + Swagger UI 截圖 + 測試程式碼
- [ ] 作業四：序列化比較程式碼 + 比較表
- [ ] 作業五：漂移偵測程式碼 + 視覺化圖表 + 分析報告
- [ ] （加分）Dockerfile + 建構/運行截圖

---

## 評分標準 Grading Criteria

- 程式碼正確執行且結構清晰 Code Quality: 40%
- 分析與報告深度 Analysis Depth: 25%
- 視覺化品質 Visualization Quality: 15%
- 完整性與細節 Completeness: 20%
