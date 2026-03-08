# 第 12 週教師手冊
# Week 12 Teacher Guide

## 時間分配 Time Allocation（90 分鐘）

| 時段 | 分鐘 | 活動 | 說明 |
|------|:---:|------|------|
| 回顧與導入 | 5 | 第 11 週回顧 + CNN 動機引入 | 從全連接層的問題出發 |
| 理論（上半） | 20 | 卷積運算、卷積核、Padding/Stride | Slide 2-6 |
| 快速實作 | 10 | 手動卷積 + 觀察卷積核效果 | Notebook 前半段 |
| 理論（下半） | 15 | 池化、經典架構、CAM/Grad-CAM | Slide 7-17 |
| 核心實作 | 30 | CNN 訓練 + 卷積核/特徵圖/Grad-CAM 視覺化 | Notebook 後半段 |
| 總結與作業 | 10 | 回顧重點 + 作業說明 + 遷移學習預告 | Slide 21-22 |

---

## 教學重點 Key Teaching Points

### 1. 概念建構策略 Conceptual Scaffolding

**從直覺到數學**：
- 先用「滑動窗口 + 模板匹配」的比喻建立直覺
- 再引入數學公式，確認學生理解每個符號的意義
- 最後用程式碼驗證數學計算

**從手工到學習**：
- 先展示手工設計的卷積核（邊緣偵測、模糊等）效果
- 問學生：「如果我們不知道什麼特徵是重要的呢？」
- 引出 CNN 自動學習卷積核的核心概念

**從局部到全局**：
- 用「感受野」串連整個架構的邏輯
- 淺層看局部 → 深層看全局
- 這是理解特徵圖層次化學習的關鍵

### 2. 視覺化教學要點 Visualization Teaching Notes

**卷積核視覺化**：
- 重點放在第一層，因為可以直接解讀（與 RGB 通道對應）
- 讓學生對比：初始化的隨機卷積核 vs 訓練後的卷積核
- 引導學生發現：訓練後的卷積核呈現有意義的結構

**特徵圖視覺化**：
- 準備同一張圖在不同層的特徵圖對比圖
- 讓學生注意：淺層保留空間細節，深層更抽象
- 問學生：「第 1 層的特徵圖像什麼？最後一層呢？」

**Grad-CAM**：
- 這是本週最重要的「高潮」環節
- 先展示幾個引人注目的例子（如：模型關注背景而非物件）
- 再解釋原理
- 強調 Grad-CAM 的實用價值：除錯、信任、安全

### 3. 常見迷思澄清 Common Misconceptions

| 迷思 | 正確理解 |
|------|---------|
| CNN 只能處理影像 | CNN 也可用於 1D（時間序列/文字）和 3D（體積）資料 |
| 更深的網路一定更好 | 超過一定深度會退化（ResNet 的動機），還有計算成本問題 |
| 卷積 = 數學上的卷積 | DL 中的「卷積」其實是互相關 (Cross-correlation)，省略了核翻轉 |
| 池化是必要的 | 現代架構（如 All-Convolutional Net）可以用 stride>1 的卷積取代池化 |
| Grad-CAM 顯示「CNN 看到的東西」 | 更精確地說，它顯示「對預測最重要的區域」，不等於人類的視覺注意力 |
| 特徵圖的亮區 = 偵測到的特徵 | 特徵圖的值需要經過 ReLU 等激活函數，亮區代表「強激活」 |

---

## 課堂互動設計 In-Class Activities

### 活動一：手動卷積體驗（5 分鐘）
**目的**：讓學生用手算理解卷積運算

給學生一個 5x5 的小影像矩陣和一個 3x3 卷積核，讓他們：
1. 手動計算左上角 3x3 區域的卷積輸出
2. 預測整個輸出的大小
3. 思考如果加了 padding=1 輸出大小會怎樣

### 活動二：猜猜卷積核（5 分鐘）
**目的**：建立卷積核→效果的直覺

展示幾張經過不同卷積核處理的影像，讓學生猜：
- 「這個效果是哪個卷積核造成的？」
- 邊緣偵測、模糊、銳化的結果讓學生配對

### 活動三：Grad-CAM 診斷（10 分鐘）
**目的**：體驗 CNN 可視化的實用價值

展示一些「有問題的」Grad-CAM 結果：
- 模型預測「馬」，但 Grad-CAM 關注的是草地背景（Clever Hans 問題）
- 模型預測「醫療影像正常」，但 Grad-CAM 關注的是影像角落的文字標記
- 討論：這些發現對模型部署有什麼影響？

### 活動四：架構設計挑戰（選用，5 分鐘）
**目的**：理解架構設計的權衡

給學生一個約束條件（如：輸入 32x32x3，輸出 10 類，參數量 < 100K），讓他們設計 CNN 架構並計算參數量。

---

## 技術準備 Technical Preparation

### 環境需求
```
Python 3.9+
PyTorch >= 2.0
torchvision
matplotlib
numpy
opencv-python (cv2)  # 用於 Grad-CAM 的熱度圖疊加
```

### 預先下載資料集
```python
# 在課前執行，確保資料集已下載
import torchvision
torchvision.datasets.CIFAR10(root='./data', download=True)
torchvision.datasets.FashionMNIST(root='./data', download=True)
```

### GPU 注意事項
- 如果教室電腦沒有 GPU，CNN 訓練會較慢
- 建議方案：
  1. 準備預訓練好的模型檔案 (.pth)，讓學生可以直接載入做視覺化
  2. 準備 Google Colab 版本的 Notebook（可使用免費 GPU）
  3. 減少訓練 epoch 數（課堂示範用 3-5 epoch 即可看到趨勢）

### 預訓練模型準備
```python
# 課前準備：訓練好一個模型並保存
torch.save(model.state_dict(), 'week12_cnn_cifar10.pth')

# 課堂上學生可以載入
model.load_state_dict(torch.load('week12_cnn_cifar10.pth'))
```

---

## 檢核點 Checkpoints

- [ ] 學生能解釋卷積運算的過程（不只是呼叫 API）
- [ ] 學生能正確計算給定參數下的輸出尺寸
- [ ] 學生能使用 PyTorch Hook 提取中間層特徵圖
- [ ] 學生能生成並解讀 Grad-CAM 熱度圖
- [ ] 學生能區分淺層與深層特徵圖的差異
- [ ] 學生能解釋 Grad-CAM 的基本原理（梯度加權）
- [ ] 學生了解 CNN 可視化的實用價值（不只是學術練習）

---

## AI 助教設定 AI Tutor Configuration

本週助教設定為「進階探索模式」：
- 可以回答 CNN 架構相關的概念問題
- 對 Grad-CAM 的原理提供分層提示（先直覺、再數學）
- 鼓勵學生嘗試不同的影像和層來觀察視覺化結果
- 不直接提供完整的 Grad-CAM 實作程式碼（鼓勵學生參考 Notebook 自行修改）
- 可以討論 CNN 在不同領域的應用案例

### AI 助教提示策略
1. **Level 1**：引導學生回憶卷積的基本概念
2. **Level 2**：提示相關的 PyTorch API（如 register_forward_hook）
3. **Level 3**：提供部分程式碼框架，讓學生填空
4. **Level 4**：給出完整範例但要求學生修改適應自己的模型

---

## 常見問題與排除 Troubleshooting

### Q1: Grad-CAM 全黑或全白
- 檢查是否對正確的層做了 hook（應該是最後一個卷積層）
- 確認是否在 `model.eval()` 模式下運行
- 檢查梯度是否正確計算（`retain_grad()` 或 `backward()` 是否被呼叫）
- 嘗試正規化熱度圖到 [0, 1] 範圍

### Q2: 特徵圖提取失敗
- 確認 hook 註冊在正確的層上
- 打印模型結構 (`print(model)`) 確認層名
- 確認 forward hook 的 callback 格式正確

### Q3: CIFAR-10 訓練準確率很低
- CIFAR-10 的圖片很小 (32x32)，簡單 CNN 約 70-80% 是合理的
- 檢查資料正規化是否正確（mean 和 std）
- 確認標籤沒有錯誤
- 學習率可能過高或過低

### Q4: 記憶體不足 (Out of Memory)
- 減小 batch size
- 減少模型參數（更少的 filters）
- 使用 `torch.no_grad()` 進行推論
- 如果用 GPU，確認 tensor 在正確的 device 上

### Q5: 卷積核視覺化看起來都一樣
- 確認是訓練後的模型（不是初始化的模型）
- 對每個卷積核**個別**做正規化（min-max 到 [0, 1]）
- 可能需要增加訓練 epoch 數讓卷積核分化

### Q6: 遷移學習的 ResNet 報錯
- 確認輸入尺寸正確（ResNet 預設 224x224，CIFAR-10 是 32x32 需要 resize）
- 確認最後一層 FC 的輸出維度已修改為目標類別數
- `model.fc = nn.Linear(model.fc.in_features, num_classes)`

---

## 差異化教學 Differentiated Instruction

### 對進度較快的學生
- 引導他們嘗試進階挑戰（遷移學習 + 多方法比較）
- 鼓勵探索 Grad-CAM++ 或 Score-CAM
- 討論 CNN 在醫療影像、衛星圖像等領域的應用
- 嘗試在自己的圖片上做 Grad-CAM 分析

### 對進度較慢的學生
- 提供預訓練好的模型，讓他們專注在**視覺化與分析**
- 簡化 Grad-CAM 的數學推導，先用「加權平均」的直覺理解
- 提供更多的程式碼模板，減少從零開始的壓力
- 鼓勵使用 AI 助教來理解程式碼中不懂的部分

### 對有數學背景的學生
- 深入討論 Grad-CAM 的數學推導
- 比較 CAM 和 Grad-CAM 在數學上的等價性（在 GAP 架構下）
- 討論高階梯度方法（Grad-CAM++）的改進

---

## 備課提醒 Preparation Notes

- [ ] 課前測試 Notebook 中所有程式碼能正常執行
- [ ] 準備預訓練好的 CNN 模型檔案（.pth），以防課堂訓練時間不夠
- [ ] 準備幾張有趣的 Grad-CAM 結果圖（如：Clever Hans 效應的案例）
- [ ] 確認 CIFAR-10 資料集已下載到教室共用目錄
- [ ] 準備 Google Colab 版本作為備用（含 GPU 支援）
- [ ] 預先測試 pretrained ResNet 的載入與推論速度
- [ ] 準備一些生活化的 CNN 應用案例（如：手機人臉辨識、Google Photos 搜尋）

---

## 本週與前後週的銜接 Curriculum Connections

### 與前一週（Week 11：神經網路基礎）的銜接
- 回顧全連接層的概念，說明 CNN 如何改進
- 回顧激活函數（特別是 ReLU）在 CNN 中的角色
- 回顧反向傳播，為 Grad-CAM 的梯度計算做鋪墊

### 與下一週（Week 13：RNN/序列建模）的預告
- CNN 處理空間結構 → RNN 處理時間/序列結構
- CNN 的 weight sharing（同一卷積核掃描全圖）→ RNN 的 weight sharing（同一權重在每個時間步共用）
- 預告：Transformer 如何統一處理序列和空間

### 與期末專題的關聯
- CNN 視覺化是許多專題的核心組件（如：醫學影像分析、物件偵測可視化）
- Grad-CAM 是展示模型可解釋性的重要工具
- 遷移學習是專題中快速建構強效模型的實用技巧
