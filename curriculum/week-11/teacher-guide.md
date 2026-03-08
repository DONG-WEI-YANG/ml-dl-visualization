# 第 11 週教師手冊
# Week 11 Teacher Guide

## 時間分配 Time Allocation（90 分鐘）

| 時段 | 分鐘 | 活動 | 對應投影片 | 說明 |
|------|:---:|------|:---:|------|
| 回顧 | 5 | 複習 Week 10 超參數調校重點 | — | 銜接上週內容，引入「進入深度學習」的轉折 |
| 理論 I | 15 | 感知器 → MLP → 激活函數 | Slide 2-12 | 重點在建立直覺，數學適度 |
| 理論 II | 10 | 反向傳播 + 梯度消失/爆炸 | Slide 13-15 | 用工廠追責的類比，避免過多推導 |
| Demo | 10 | 激活函數視覺化 + 梯度消失動畫 | Slide 24 | 開啟 Notebook 現場跑前 3 個 Cell |
| 理論 III | 10 | 正則化 + BatchNorm + 初始化 | Slide 16-21 | 重點在 Dropout 和 BatchNorm 的直覺 |
| 實作 | 30 | 學生動手：MLP on MNIST | Slide 22-23 | PyTorch 程式碼帶做，學生跟著改參數 |
| 總結 | 5 | 回顧 + 作業說明 | Slide 25-26 | 預告 Week 12 CNN |
| 緩衝 | 5 | Q&A | — | 彈性時間 |

---

## 教學重點 Key Teaching Points

### 1. 本週定位：從傳統 ML 跨入深度學習的關鍵轉折

- 前 10 週以 scikit-learn 為主，本週正式進入 PyTorch
- 強調**連續性**：梯度下降（Week 4）→ 反向傳播（本週）是同一概念的延伸
- 強調**差異性**：手工特徵工程 → 自動學習特徵表徵 (Learned Representations)

### 2. 激活函數：先建直覺，再看數學

- **不要一次介紹所有 6 個激活函數**，會造成資訊過載
- 建議順序：
  1. 先講 Step Function（感知器）→ 為什麼不好？（不可微）
  2. Sigmoid → 解決可微問題，但引入梯度消失
  3. ReLU → 解決梯度消失，但引入死亡 ReLU
  4. Leaky ReLU → 修補 ReLU
  5. GELU / Swish → 現代的選擇（可略講）
- **關鍵 Demo**：在互動平台上即時切換激活函數，觀察決策邊界變化

### 3. 反向傳播：重直覺、輕推導

- 用「工廠品管追責」類比：品管（損失函數）往回追溯每個工序（層）的責任
- **不要**在課堂上推導完整的矩陣微分，留給想深入的學生自學
- 強調 PyTorch autograd 會**自動完成**反向傳播 — 理解概念即可

### 4. 正則化：並排視覺化最有效

- 同時展示有/無 Dropout 的訓練曲線，讓學生**看到差異**
- L1 vs. L2 用幾何解釋（菱形 vs. 圓）最直觀
- BatchNorm 的解釋可以簡化為：「每層都做類似 StandardScaler 的事」

### 5. PyTorch 入門：帶做而非講解

- **不要**花太多時間講 Tensor 的 API，學生跟著打程式碼就會了
- 重點放在**訓練循環的四步驟**：前向 → 算損失 → 反向 → 更新
- 讓學生修改隱藏層大小、激活函數、學習率，觀察效果

---

## 檢核點 Checkpoints

- [ ] 學生能說出激活函數的用途（引入非線性）
- [ ] 學生能解釋為什麼 Sigmoid 在深層網路會有梯度消失
- [ ] 學生能用 PyTorch 定義一個簡單的 `nn.Module`
- [ ] 學生能完成一個完整的訓練循環（forward → loss → backward → step）
- [ ] 學生能說出 Dropout 防止過擬合的原理
- [ ] 學生知道 `model.train()` 和 `model.eval()` 的差異

---

## AI 助教設定 AI Tutor Configuration

本週助教設定為「引導模式」：

### 分層提示策略 Hint Ladder
1. **Level 1 — 概念釐清**：學生問「什麼是 BatchNorm？」→ 反問「你覺得為什麼每層的輸入分佈會改變？」
2. **Level 2 — 方向引導**：提示相關概念（如：「想想 StandardScaler 在 ML 中的作用」）
3. **Level 3 — 局部提示**：給出 BatchNorm 的正規化公式，但不給完整程式碼
4. **Level 4 — 程式碼框架**：給出 `nn.BatchNorm1d` 的用法，但讓學生自己整合進模型

### 常見問題預設回答
| 學生問題 | 助教引導方向 |
|---------|------------|
| ReLU 為什麼比 Sigmoid 好？ | 引導比較兩者的導數在 z=5 時的值 |
| Dropout 為什麼只在訓練時用？ | 反問：如果測試時也隨機關閉，預測結果會怎樣？ |
| BatchNorm 放在激活前還是後？ | 引導學生兩種都試試，比較結果 |
| backward() 到底在做什麼？ | 用計算圖的例子，讓學生手動算一次簡單的鏈式法則 |

---

## 常見問題與排除 Troubleshooting

### Q1: PyTorch 安裝問題
- **CUDA 版本不匹配**：建議先使用 CPU 版本，GPU 可在之後設定
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- **Mac M1/M2**：使用 `device = 'mps'` 而非 `'cuda'`

### Q2: MNIST/Fashion-MNIST 下載失敗
- 提供離線資料集檔案（放在課程平台上）
- 或使用 Google Colab（已預裝 torchvision）

### Q3: 訓練時 Loss 不下降
- 常見原因：學習率過大/過小
- 排查步驟：
  1. 檢查資料是否正確載入（印出幾張圖看看）
  2. 檢查模型輸出維度是否正確
  3. 將學習率設為 0.001 先試
  4. 確認 `optimizer.zero_grad()` 有被呼叫

### Q4: 記憶體不足 (OOM)
- 減小 batch_size（從 64 改為 32）
- 使用 `with torch.no_grad():` 在驗證/測試時
- Colab 中選擇 GPU runtime

### Q5: 學生對 nn.Module 的 OOP 概念不熟
- 提供完整的模板 (Template) 讓學生填空
- 強調只需要改 `__init__` 和 `forward` 兩個方法
- 類比：`__init__` = 準備工具，`forward` = 使用工具加工

### Q6: model.train() vs. model.eval() 搞混
- 口訣：「訓練開火車 (train)，考試要評估 (eval)」
- 實際影響：Dropout 和 BatchNorm 在兩個模式下行為不同
- 常見 bug：忘記切換模式，導致測試準確率異常

---

## 教學素材清單 Teaching Materials

| 素材 | 說明 | 檔案 |
|------|------|------|
| 投影片 | 26 張 Marp 風格 | `slides.md` |
| 實作 Notebook | 激活函數視覺化 + MLP + 正則化 | `notebook.ipynb` |
| 作業說明 | 四大題 + 加分題 | `assignment.md` |
| 評量規準 | 各題評分細則 | `rubric.md` |

---

## 差異化教學 Differentiated Instruction

### 進度較快的學生 Advanced Students
- 挑戰加分題：梯度流視覺化、學習率排程
- 鼓勵閱讀原始論文（BatchNorm: Ioffe & Szegedy, 2015）
- 嘗試在 CIFAR-10 上訓練（需要更複雜的架構）

### 進度較慢的學生 Students Needing Support
- 提供 Notebook 的「填空版」：架構已寫好，只需填入激活函數和超參數
- 降低作業二門檻：85% 準確率即可得滿分
- 加強 AI 助教的提示層級（直接提供程式碼框架）
- 課後提供額外的 PyTorch 基礎教學影片連結

### 跨領域學生 Cross-disciplinary Students
- 提供領域相關的案例（如：醫學影像分類、金融詐欺偵測）
- 強調 MLP 是「通用工具」，不限於特定領域

---

## 備課提醒 Preparation Notes

1. **提前測試 Notebook**：確保所有 Cell 可在 10 分鐘內跑完
2. **準備備用方案**：若教室電腦無 GPU，確認 CPU 訓練時間可接受（MNIST MLP 約 2-3 分鐘）
3. **下載 Fashion-MNIST**：在課程平台上放離線版本
4. **本週是轉折點**：學生從 sklearn 切換到 PyTorch，可能會有適應期，要有耐心
5. **強調 Week 12 的連續性**：本週的 MLP 是下週 CNN 的基礎
6. **準備一個「WOW moment」**：在課堂結尾 Demo 一個訓練好的 CNN 做圖片分類，預告下週內容

---

## 課後反思 Post-class Reflection

課後請記錄以下幾點，作為教學改善參考：
- [ ] 學生對 PyTorch 語法的接受程度如何？
- [ ] 反向傳播的類比是否有效？學生的表情/反饋？
- [ ] 實作時間是否足夠？哪個部分最花時間？
- [ ] 有沒有學生成功完成 MNIST 分類器？
- [ ] AI 助教的引導是否恰當？有無需要調整的地方？
