# 第 13 週投影片：RNN/序列建模（LSTM/GRU；Transformers 概念）

---

## Slide 1: 本週主題
# RNN / 序列建模
### LSTM、GRU 與 Transformer 概念
- 從序列資料談起，到 Attention 革命
- 難度等級：進階 Advanced

---

## Slide 2: 學習地圖
### 本週在課程中的位置
```
Week 11: 神經網路基礎 ← 前饋網路
Week 12: CNN ← 空間特徵
Week 13: RNN / Transformer ← 序列特徵  ★ 本週
Week 14: 訓練技巧
```
**核心問題：** 如何讓神經網路處理「有順序的資料」？

---

## Slide 3: 序列資料無所不在
### Sequential Data Is Everywhere
| 類型 | 範例 |
|:---:|:---:|
| 自然語言 NLP | "今天天氣真好" |
| 時間序列 Time Series | 股票、溫度、心電圖 |
| 語音 Speech | 語音辨識、語音合成 |
| 生物序列 Bio | DNA: ATCGGATC... |
| 行為序列 Behavior | 使用者點擊流 |

**共通點：** 順序很重要！"狗咬人" ≠ "人咬狗"

---

## Slide 4: 為什麼 FC 網路不行？
### Why Not Fully-Connected?
1. **固定長度** — 句子長短不一怎麼辦？
2. **忽略順序** — "not good" ≠ "good not"
3. **參數爆炸** — 無法共享不同位置學到的知識

**解法：** 設計能「記住過去」的網路 → RNN

---

## Slide 5: RNN 基本結構
### Recurrent Neural Network
```
摺疊視圖：           展開視圖：
    y                  y₁    y₂    y₃
    ↑                  ↑     ↑     ↑
  ┌───┐             ┌───┐ ┌───┐ ┌───┐
  │ h │←╮     h₀ → │h₁ │→│h₂ │→│h₃ │→...
  └───┘  │          └───┘ └───┘ └───┘
    ↑    │           ↑     ↑     ↑
    x  ──╯          x₁    x₂    x₃
```
**公式：** $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)$

**關鍵：** 所有時間步共享相同的 $W_{xh}$, $W_{hh}$ → 參數共享

---

## Slide 6: BPTT 與梯度消失
### Backpropagation Through Time
```
梯度大小
  │ ████
  │ ███
  │ ██
  │ █
  │ ░   ← 越早的時間步，梯度越小
  │  ·
  └──────────────────→ 時間步 (越早→)
```
- 梯度需穿越連乘的 $W_{hh}$ 矩陣
- 特徵值 < 1 → **梯度消失** → 無法學習長程依賴
- 特徵值 > 1 → **梯度爆炸** → 用梯度裁剪解決

---

## Slide 7: 長程依賴問題
### Long-Range Dependency
"The **cat**, which already ate a lot of food earlier that day, **was** full."

- RNN 需要記住 "cat" 是單數，才能在很久之後正確生成 "was"
- 但梯度消失讓 RNN 「忘記」了遠處的資訊
- **我們需要更好的記憶機制！**

---

## Slide 8: LSTM — 核心思想
### Long Short-Term Memory (1997)
> **類比：** RNN 像一條河流，資訊流經水壩被消耗。
> LSTM 加了一條「高速公路」讓重要資訊暢通無阻。

**兩條資訊通道：**
- $h_t$ — 短期記憶（隱藏狀態）
- $C_t$ — 長期記憶（細胞狀態） ← **新增！**

**三個門控制資訊流動：**
遺忘門 | 輸入門 | 輸出門

---

## Slide 9: LSTM — 遺忘門
### Forget Gate
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- 輸出 0~1 之間的值
- **0 = 完全遺忘** | **1 = 完全保留**
- 例：讀到新句子 → 遺忘前一句的主語

---

## Slide 10: LSTM — 輸入門
### Input Gate
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

- $i_t$ 決定哪些維度要更新
- $\tilde{C}_t$ 產生候選新記憶
- 例：讀到新主語 → 存入記憶

---

## Slide 11: LSTM — 細胞狀態更新
### Cell State Update
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

```
C(t-1) ──→ [× f_t] ──→ [+] ──→ C(t)
                         ↑
              [× i_t] ← C̃(t)
```

**關鍵：加法更新**讓梯度可以不衰減地流過 → 解決梯度消失！

---

## Slide 12: LSTM — 輸出門
### Output Gate
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

- 根據當前上下文，決定展現記憶的哪個面向
- 例：需要判斷動詞形式時，輸出主語的數量資訊

---

## Slide 13: GRU — 簡化版 LSTM
### Gated Recurrent Unit (2014)

| 比較 | LSTM | GRU |
|:---:|:---:|:---:|
| 門數 | 3 | 2 |
| 狀態 | $h_t + C_t$ | $h_t$ |
| 參數 | 多 | 少 ~25% |

**兩個門：**
- **重置門** $r_t$：決定忽略多少過去
- **更新門** $z_t$：決定新舊狀態的混合比例

$$h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

---

## Slide 14: 雙向 RNN
### Bidirectional RNN
```
正向: h→₁ ──→ h→₂ ──→ h→₃ ──→ h→₄
       ↑       ↑       ↑       ↑
      x₁      x₂      x₃      x₄
       ↓       ↓       ↓       ↓
反向: h←₁ ←── h←₂ ←── h←₃ ←── h←₄
```
- 正向 RNN 捕捉「左側上下文」
- 反向 RNN 捕捉「右側上下文」
- 輸出 = 拼接兩方向的隱藏狀態
- **限制：** 需要完整序列，不能用於即時串流

---

## Slide 15: Seq2Seq 的瓶頸
### The Bottleneck Problem
```
Encoder: x₁ → x₂ → x₃ → [context c] → Decoder: y₁ → y₂
```
- 整個輸入壓縮成**一個固定向量** c
- 序列越長，資訊損失越嚴重
- 超過 20-30 tokens 品質急降

**怎麼辦？** → 注意力機制！

---

## Slide 16: 注意力機制
### Attention Mechanism (2014)
> 不要壓縮成一個向量，讓 Decoder **回頭看** Encoder 的所有狀態！

```
Encoder:    h₁    h₂    h₃    h₄
             │     │     │     │
注意力:    0.1   0.7   0.15  0.05
             │     │     │     │
context = 0.1·h₁ + 0.7·h₂ + 0.15·h₃ + 0.05·h₄
```

- 每步計算**不同的**注意力權重
- 動態聚焦於最相關的輸入位置

---

## Slide 17: Transformer 登場
### "Attention Is All You Need" (2017)
**革命性主張：** 完全不需要遞迴！只用注意力！

| RNN 的問題 | Transformer 方案 |
|:---------:|:---------------:|
| 無法平行 | Self-Attention 完全平行 |
| 長程依賴靠梯度長傳 | 任意位置間路徑 = O(1) |
| 固定隱藏狀態瓶頸 | 動態關注全序列 |

---

## Slide 18: Self-Attention
### Query-Key-Value
```
每個 token 扮演三個角色：
  Q (查詢): "我在找什麼？"
  K (鍵):   "我有什麼？"
  V (值):   "我的內容是什麼？"
```

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- 每個位置直接關注所有其他位置
- 注意力權重 = softmax(Q 與 K 的相似度)

---

## Slide 19: Multi-Head Attention
### 多頭注意力 — 多角度觀察
```
Head 1: 語法依賴 (主語-動詞)
Head 2: 語義相似 (同義詞)
Head 3: 位置接近 (相鄰詞)
Head 4: 指代關係 (代名詞)
...
→ Concat → 線性投影 → 輸出
```
- 通常 $h=8$ 個頭
- 每個頭維度 = $d_{model}/h$
- 計算量不變，表達力更強

---

## Slide 20: 位置編碼
### Positional Encoding
**問題：** Self-Attention 不知道順序！
"狗咬人" 和 "人咬狗" 的注意力結果一樣

**解法：** 加入位置信號

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

Input = Token Embedding **+** Positional Encoding

---

## Slide 21: Transformer 完整架構
### Encoder-Decoder Structure
```
      Encoder (×6)              Decoder (×6)
   ┌─────────────┐          ┌─────────────┐
   │ Feed-Forward │          │ Feed-Forward │
   │ + Add & Norm │          │ + Add & Norm │
   │              │          │              │
   │ Self-Attn    │     ┌──→ │ Cross-Attn   │
   │ + Add & Norm │     │    │ + Add & Norm │
   └──────┬───────┘     │    │              │
          │             │    │Masked Self-Attn│
          └─────────────┘    │ + Add & Norm │
                             └──────────────┘
```
- Encoder: Self-Attention → FFN
- Decoder: Masked Self-Attn → Cross-Attn → FFN
- 每一層都有殘差連接 + LayerNorm

---

## Slide 22: 技術演進全景
### From RNN to Transformer
```
1997  LSTM ─── 解決梯度消失
  ↓
2014  GRU ──── 簡化 LSTM
  ↓
2014  Attention ─ 打破瓶頸
  ↓
2017  Transformer ─ 拋棄遞迴
  ↓
2018+ BERT / GPT ── 預訓練大模型
  ↓
2023+ 多模態 Transformer
```

---

## Slide 23: RNN vs LSTM vs GRU vs Transformer
### 總結比較
| | RNN | LSTM | GRU | Transformer |
|:---|:---:|:---:|:---:|:---:|
| 長程依賴 | 差 | 好 | 好 | 最好 |
| 平行化 | 不行 | 不行 | 不行 | 完全平行 |
| 參數量 | 少 | 多 | 中 | 最多 |
| 訓練速度 | 快* | 慢 | 中 | 快(GPU) |
| 主流程度 | 低 | 中 | 中 | 最高 |

*RNN 雖然單步快，但無法平行化

---

## Slide 24: 本週實作預告
### Notebook 實作內容
1. PyTorch 建構 RNN / LSTM
2. 正弦波序列預測任務
3. LSTM 門控值視覺化
4. 注意力權重視覺化
5. RNN vs LSTM vs GRU 比較實驗
6. 簡化版 Self-Attention 實作

---

## Slide 25: 本週作業
### Assignment
1. 完成 Notebook 所有練習題
2. 序列模型比較實驗報告
3. 思考題：為什麼 Transformer 能取代 RNN？

---

## Slide 26: 下週預告
### Week 14: 深度學習訓練技巧
- 學習率排程 (LR Scheduling)
- 早停 (Early Stopping)
- 資料增強 (Data Augmentation)
- 不同訓練策略的視覺化比較
