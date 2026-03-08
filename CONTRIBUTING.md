# 貢獻指南 / Contributing Guide

## 開發環境設定

```bash
# 後端
cd platform/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"

# 前端
cd platform/frontend
npm install
```

## 程式碼規範

### Python (後端)
- Python 3.11+
- Type hints 必要
- 使用 Pydantic v2 做資料驗證
- 測試使用 pytest

### TypeScript (前端)
- 嚴格模式 (`strict: true`)
- 函數元件 + Hooks（不用 class component）
- Tailwind CSS 做樣式
- ESLint 檢查

## 提交流程

1. 建立 feature branch：`git checkout -b feature/描述`
2. 確認測試通過：
   ```bash
   # 後端
   cd platform/backend && pytest tests/ -v
   # 前端
   cd platform/frontend && npm test && npx tsc --noEmit
   ```
3. 提交並推送
4. 建立 Pull Request

## 目錄結構

- `curriculum/` — 教材內容（Markdown + Jupyter Notebook）
- `platform/frontend/` — React 前端
- `platform/backend/` — FastAPI 後端
- `datasets/` — 範例資料集
- `docs/` — 設計文件與手冊

## 新增一週教材

每週需要 6 個檔案：
```
curriculum/week-XX/
  lecture.md        # 講義
  slides.md         # 投影片（Markdown）
  notebook.ipynb    # 實作 Notebook
  assignment.md     # 作業
  rubric.md         # 評量規準
  teacher-guide.md  # 教師手冊
```

## 新增視覺化元件

1. 建立 `src/components/viz/YourViz.tsx`
2. 在 `src/pages/WeekPage.tsx` 的 `weekComponents` 中註冊
3. 加入基礎 ARIA 標籤（`role`, `aria-label`）
