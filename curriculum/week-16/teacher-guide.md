# 第 16 週教師手冊
# Week 16 Teacher Guide: MLOps 入門

## 時間分配 Time Allocation（90 分鐘）

| 時段 | 分鐘 | 活動 | 說明 |
|------|:---:|------|------|
| 開場 | 5 | 回顧 Week 15 + 本週目標 | 連結模型評估與生產部署 |
| 理論一 | 20 | MLOps 概覽 + 實驗追蹤 | Slide 1-11，含 MLflow Demo |
| 理論二 | 15 | 模型部署 + FastAPI | Slide 12-16，含程式碼走讀 |
| 實作 | 30 | Notebook 動手做 | MLflow + FastAPI 實作 |
| 理論三 | 10 | 模型監測 + Docker | Slide 17-24，概念說明 |
| 總結 | 10 | 回顧 + 作業說明 | Slide 25-29，Q&A |

---

## 教學重點 Key Teaching Points

### 1. 建立正確的心態

本週的核心訊息是：**訓練出模型只完成了 ML 專案的 20%，剩下的 80% 是部署、維護與監測。**

建議開場時先問學生：
- "你訓練完一個模型之後，怎麼讓別人用？"
- "你的模型上線一年後，效能會變好還是變差？"

這些問題能引發思考，讓學生理解 MLOps 的必要性。

### 2. 實驗追蹤（MLflow）

**教學策略：先 Demo 痛點，再介紹解法。**

1. 先展示一個 Excel 追蹤實驗的「混亂」範例
2. 提問："如果有 100 次實驗，你怎麼找到最好的那次？"
3. 引入 MLflow，Demo 即時的 UI 操作

**現場 Demo 步驟：**
```bash
# 課前先準備好環境
pip install mlflow scikit-learn
# 跑一個簡單的實驗
python -c "
import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
mlflow.set_experiment('demo')
with mlflow.start_run():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    mlflow.log_param('n_estimators', 100)
    mlflow.log_metric('accuracy', model.score(X, y))
    mlflow.sklearn.log_model(model, 'model')
"
# 啟動 UI
mlflow ui --port 5000
```

### 3. FastAPI 推論服務

**教學策略：從最簡單的例子開始，逐步加入功能。**

先展示只有 5 行的最簡 API：
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello MLOps!"}
```

然後逐步加入：
1. POST 端點
2. Pydantic 驗證
3. 模型載入
4. 錯誤處理

**重點提醒：**
- 確保學生理解 HTTP 方法（GET vs. POST）
- 讓學生實際在 Swagger UI (`/docs`) 上測試
- 強調 Pydantic 自動驗證的便利性

### 4. 模型監測

**教學策略：用具體案例說明為什麼需要監測。**

準備 2-3 個真實世界案例：
- 疫情期間電商推薦系統失效（資料漂移）
- 經濟衰退導致信用評分模型不準（概念漂移）
- 新地區上線後翻譯品質下降（協變量偏移 Covariate Shift）

**視覺化 Demo：**
- 用 matplotlib 畫出訓練資料 vs. 漂移資料的分布圖
- 現場計算 KS 統計量，讓學生看到 p-value 的變化

### 5. Docker（概念介紹為主）

Docker 在 90 分鐘內不太可能讓學生全部動手做。建議：
- **課堂上：** 概念說明 + 展示 Dockerfile + Demo 建構/運行流程
- **作業中：** 設為加分題，讓有興趣的學生自行嘗試
- 確保學生理解「為什麼需要容器化」比「怎麼寫 Dockerfile」更重要

---

## 檢核點 Checkpoints

- [ ] 學生成功安裝 MLflow 並能啟動 UI
- [ ] 學生能使用 `mlflow.log_param()` 和 `mlflow.log_metric()` 記錄實驗
- [ ] 學生能啟動 FastAPI 服務並透過 Swagger UI 測試
- [ ] 學生理解資料漂移 (Data Drift) 與概念漂移 (Concept Drift) 的差異
- [ ] 學生能用 KS Test 判斷是否有資料漂移
- [ ] 學生理解 Docker 容器化的目的與基本流程

---

## AI 助教設定 AI Tutor Configuration

本週助教設定為「工程導向模式」：
- 偏重實作引導，而非理論解釋
- 當學生遇到安裝問題時，優先提供排除步驟
- 對於 MLflow、FastAPI 的使用問題，引導學生查閱官方文件
- 資料漂移的問題可提供公式輔助，但需引導學生自行計算

**助教可回答的範圍：**
- MLflow API 使用方式
- FastAPI 路由 (Routing) 與資料驗證
- Docker 基本概念與命令
- 序列化格式的優缺點比較
- KS Test 與 PSI 的計算方式

**助教應引導而非直接回答的範圍：**
- 作業中的分析報告內容
- 模型選擇與參數調整策略
- 漂移偵測結果的解讀

---

## 常見問題與排除 Troubleshooting

### Q1: MLflow 安裝失敗或啟動 UI 失敗
- 確認 `pip install mlflow` 成功
- Windows 上若 `mlflow ui` 無法使用，嘗試 `python -m mlflow ui`
- 確認 port 5000 未被佔用：`netstat -an | findstr 5000`
- 如果 SQLite 相關錯誤，刪除 `mlruns/` 重新開始

### Q2: FastAPI 服務無法啟動
- 確認安裝 `pip install fastapi uvicorn`
- 確認模型檔案路徑正確
- `uvicorn app:app --reload` 中的 `app:app` 表示「檔案 app.py 中的 app 變數」
- Windows 若 port 8000 被佔用，換用其他 port：`--port 8001`

### Q3: 模型序列化問題
- joblib 通常比 pickle 更適合大型 NumPy 陣列
- ONNX 轉換需安裝 `skl2onnx`：`pip install skl2onnx onnxruntime`
- ONNX 轉換需要指定輸入的型別與形狀

### Q4: KS Test 結果解讀
- p-value < 0.05 表示兩個分布有顯著差異（漂移）
- p-value > 0.05 表示無法拒絕「兩個分布相同」的假設
- 提醒學生：統計顯著不等於實務顯著

### Q5: Docker 無法在學生電腦上安裝
- Windows Home 需要 WSL2 + Docker Desktop
- 如果電腦不支援虛擬化，建議使用 Google Cloud Shell（免費）
- 或者使用 GitHub Codespaces（免費額度）
- 加分題設為選做，不強制要求

### Q6: 學生對 REST API 概念不熟悉
- 先用瀏覽器展示 GET 請求（直接在網址列輸入 URL）
- 再用 Swagger UI 展示 POST 請求
- 類比：API 就像餐廳的點菜流程 — 你（Client）看菜單（API 文件）點菜（Request），廚房（Server）做菜後送上來（Response）

---

## 課前準備 Pre-class Preparation

### 環境準備
```bash
# 確認以下套件已安裝在教學環境中
pip install mlflow fastapi uvicorn joblib scikit-learn pandas numpy matplotlib scipy
# 可選
pip install skl2onnx onnxruntime evidently
```

### Demo 準備
1. 事先跑幾次 MLflow 實驗，確保 UI 有資料可展示
2. 準備好一個能運行的 FastAPI app.py
3. 如果要 Demo Docker，事先在教師機上建構好映像檔
4. 準備好漂移資料的視覺化圖表

### 教材準備
- 投影片檔案已備妥 (slides.md)
- Notebook 檔案已測試可執行 (notebook.ipynb)
- 確認 MLflow UI 在教室網路環境下可正常使用

---

## 教學差異化 Differentiation

### 對於基礎較弱的學生
- 聚焦在 MLflow Tracking 與 FastAPI 兩個核心主題
- 提供更多程式碼模板，減少從零開始的壓力
- Docker 與 ONNX 可暫時跳過
- 作業五的分析報告可降低字數要求

### 對於進度超前的學生
- 鼓勵嘗試 Docker 容器化（加分題）
- 引導探索 ONNX 轉換與跨平台推論
- 可嘗試使用 Evidently AI 生成完整的漂移報告
- 挑戰：建立一個含 CI/CD 的 GitHub Actions workflow

---

## 與前後週次的銜接 Curriculum Connection

### 與前一週（Week 15: 模型評估與選擇）的銜接
- Week 15 教了模型評估指標 → 本週用 MLflow 自動記錄這些指標
- Week 15 的模型比較 → 本週用 MLflow UI 視覺化比較
- 強調：評估不只發生在訓練階段，上線後也要持續評估（監測）

### 與下一週（Week 17: LLM 與大型語言模型）的銜接
- 本週的推論服務概念 → Week 17 的 LLM API 服務
- 本週的模型版本管理 → LLM 的版本管理更複雜（模型更大）
- 預告：LLM 的部署有其特殊挑戰（模型大小、推論成本）

---

## 反思與改進 Reflection

每次授課後請記錄：
- 學生在哪個環節花最多時間？
- MLflow 安裝是否順利？是否需要課前預裝？
- FastAPI Demo 是否足夠清楚？學生能否獨立完成？
- 資料漂移的概念學生是否理解？是否需要更多案例？
- 90 分鐘是否足夠？是否需要調整時間分配？
