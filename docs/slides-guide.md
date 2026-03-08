# 投影片使用指南 / Slides Rendering Guide

每週的 `slides.md` 使用 Markdown 格式，以 `---` 分隔每張投影片。

## 渲染方式

### 方式一：Slidev（推薦）

```bash
# 安裝
npm install -g @slidev/cli

# 渲染單週投影片
slidev curriculum/week-01/slides.md

# 匯出 PDF
slidev export curriculum/week-01/slides.md
```

### 方式二：Reveal.js

```bash
# 安裝 reveal-md
npm install -g reveal-md

# 渲染
reveal-md curriculum/week-01/slides.md

# 匯出 PDF
reveal-md curriculum/week-01/slides.md --print slides.pdf
```

### 方式三：VS Code 預覽

安裝 [Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode) 擴充，直接在編輯器中預覽投影片。

### 方式四：直接使用 Markdown

投影片內容本身就是結構化的 Markdown 文件，教師可直接閱讀或複製到任何簡報工具（PowerPoint、Google Slides）中使用。

## 格式說明

```markdown
# 投影片標題          ← H1 作為整份投影片主題
---                   ← 分頁符號
## Slide N: 小標題    ← H2 作為每頁標題
### 副標題            ← H3 作為內容區塊
- 項目符號內容
---
```

## 全部 18 週投影片列表

| 週次 | 檔案路徑 |
|------|---------|
| 1 | `curriculum/week-01/slides.md` |
| 2 | `curriculum/week-02/slides.md` |
| ... | ... |
| 18 | `curriculum/week-18/slides.md` |
