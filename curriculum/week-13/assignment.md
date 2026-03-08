# 第 13 週作業：RNN/序列建模實驗
# Week 13 Assignment: RNN/Sequence Modeling Experiments

**繳交期限 Due:** 下週上課前
**繳交方式 Submission:** 上傳 Notebook (.ipynb) 與實驗報告 (.pdf) 至課程平台
**難度等級 Level:** 進階 Advanced

---

## 作業一：序列預測任務（25%）

使用 PyTorch 建構一個 LSTM 模型來預測正弦波 (Sine Wave) 的下一個時間步。

### 要求：
1. 生成一段帶有噪音的正弦波資料（至少 1000 個時間步）
2. 使用滑動視窗 (Sliding Window) 方法建立訓練集（視窗大小自選，建議 20-50）
3. 建構一個至少包含 **2 層 LSTM** 的模型
4. 訓練模型並繪製：
   - 訓練損失曲線 (Training Loss Curve)
   - 預測值 vs 真實值的比較圖
5. 嘗試不同的**隱藏維度** (Hidden Size: 16, 32, 64) 並比較結果

### 提示 Hints:
```python
# 資料生成範例
t = np.linspace(0, 100, 1000)
signal = np.sin(t) + 0.1 * np.random.randn(len(t))
```

---

## 作業二：RNN vs LSTM vs GRU 比較實驗（25%）

在相同的序列任務上，比較三種模型的表現差異。

### 要求：
1. 使用**相同的**資料集與超參數設定（隱藏維度、學習率、epoch 數）
2. 分別建構 `nn.RNN`、`nn.LSTM`、`nn.GRU` 模型
3. 記錄並比較以下指標：
   - 最終訓練損失 (Final Training Loss)
   - 最終測試損失 (Final Test Loss)
   - 訓練時間 (Training Time)
   - 預測品質（視覺化比較）
4. 製作一張**比較表格**和一組**比較圖表**
5. 撰寫簡要分析（100-200 字）：三種模型的差異原因

### 進階加分 Bonus (+5%):
嘗試一個**更長序列依賴**的任務（如 sequence length > 100），觀察 Vanilla RNN 的效能下降。

---

## 作業三：LSTM 門控值視覺化（20%）

深入理解 LSTM 的內部機制。

### 要求：
1. 訓練一個 LSTM 模型（可使用作業一的模型）
2. 取出一段輸入序列，提取並視覺化以下內部狀態：
   - 遺忘門 (Forget Gate) 的激活值
   - 輸入門 (Input Gate) 的激活值
   - 輸出門 (Output Gate) 的激活值
   - 細胞狀態 (Cell State) 隨時間的變化
3. 使用**熱力圖 (Heatmap)** 展示門控值（x 軸 = 時間步，y 軸 = 隱藏維度）
4. 根據視覺化結果，解釋 LSTM 在不同時間步的記憶行為（100-150 字）

### 提示 Hints:
```python
# 提取 LSTM 門控值需要使用 LSTMCell 手動實作前向傳播
# 或者使用 register_forward_hook 來擷取中間值
```

---

## 作業四：注意力權重視覺化與分析（20%）

實作並視覺化簡化版的注意力機制。

### 要求：
1. 實作一個帶有 **Additive Attention** (Bahdanau) 的 Seq2Seq 模型，或使用 Notebook 中的 Self-Attention 實作
2. 在一個簡單的序列任務（如字元反轉、排序或正弦波預測）上訓練
3. 視覺化注意力權重矩陣 (Attention Weight Matrix)：
   - 使用熱力圖展示 Attention Weights
   - x 軸 = 輸入位置 (Source Position)
   - y 軸 = 輸出位置 (Target Position)
4. 分析注意力分布是否合理（100-150 字），例如：
   - 對角線模式 (Diagonal Pattern) 表示單調對齊
   - 注意力是否集中在正確的輸入位置？

---

## 作業五：思考題（10%）

請以 200-300 字回答以下問題（可用中文或英文）：

1. **為什麼 Transformer 能取代 RNN 成為主流架構？** 從以下三個角度分析：
   - 計算效率 (Computational Efficiency)
   - 長程依賴 (Long-Range Dependencies)
   - 可擴展性 (Scalability)

2. **RNN 是否仍有存在價值？** 舉出至少兩個 RNN/LSTM 仍然優於 Transformer 的應用場景，並說明原因。

3. **（選答 Optional）** 最近的狀態空間模型 (State Space Models, 如 Mamba) 嘗試結合 RNN 的線性複雜度與 Transformer 的效能。你認為這個方向有前景嗎？為什麼？

---

## 評分標準 Grading Criteria

| 項目 | 比重 | 說明 |
|------|:---:|------|
| 程式碼正確性 Code Correctness | 35% | 模型可正確訓練與推論 |
| 視覺化品質 Visualization Quality | 25% | 圖表清晰、標籤完整、配色適當 |
| 實驗分析 Experiment Analysis | 25% | 觀察深入、分析合理、結論有據 |
| 程式碼風格 Code Style | 10% | 註解清楚、組織有序 |
| 進階加分 Bonus | +5% | 額外的深入探索 |

---

## 提交清單 Submission Checklist

- [ ] Notebook (.ipynb) 已重新執行所有 Cell（Kernel → Restart & Run All）
- [ ] 所有圖表均有標題、軸標籤、圖例
- [ ] 文字分析段落完整
- [ ] 比較表格數據正確
- [ ] 程式碼有適當的註解
