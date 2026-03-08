# 第 1 週教師手冊
# Week 1 Teacher Guide

## 時間分配 Time Allocation（90 分鐘）

| 時段 | 分鐘 | 活動 | 說明 |
|------|:---:|------|------|
| 開場 | 5 | 課程介紹與自我介紹 | 建立師生關係 |
| 理論 | 25 | ML/DL 概覽 + 投影片 | Slide 1-5 |
| 實作 | 40 | 環境建置 + Notebook 操作 | 學生動手，教師巡場 |
| 平台導覽 | 15 | Demo 互動平台 + AI 助教 | 現場示範 |
| 總結 | 5 | 回顧 + 作業說明 | Slide 10-11 |

## 教學重點 Key Points
1. 第一週重點在**建立環境**與**激發學習動機**，不需深入理論
2. 強調視覺化的重要性 — 可先 Demo 後面幾週的互動視覺化引起興趣
3. 確保每位學生都成功安裝環境（這是後續週次的基礎）

## 檢核點 Checkpoints
- [ ] 學生成功啟動 Jupyter Notebook
- [ ] 學生能執行 `import numpy, pandas, matplotlib, sklearn`
- [ ] 學生成功登入課程平台
- [ ] 學生在 AI 助教上完成至少一次對話

## AI 助教設定 AI Tutor Configuration
本週助教設定為「歡迎模式」：
- 語氣友善鼓勵
- 回答較為直接（降低初次使用的門檻）
- 主動介紹自己的功能與限制
- 引導學生了解分層提示策略

## 常見問題與排除 Troubleshooting

### Q1: Anaconda 安裝失敗
- Windows: 檢查路徑是否有中文或空格
- 改用 Miniconda（輕量版本）
- 確認管理員權限

### Q2: pip install 報錯
- 更新 pip: `python -m pip install --upgrade pip`
- 使用國內鏡像源（如清華 TUNA）

### Q3: Jupyter 無法啟動
- 檢查是否已啟動虛擬環境
- 嘗試 `python -m jupyter notebook`

### Q4: 中文亂碼
- Matplotlib 需設定中文字型：
```python
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
```

## 備課提醒 Preparation Notes
- 提前測試環境安裝流程
- 準備離線安裝包以防網路問題
- 準備 Google Colab 作為備用方案
- 第一週氣氛輕鬆為主，讓學生感受到課程的趣味性
