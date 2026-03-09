# 學年度系統設計

**日期**: 2026-03-09
**目標**: 在課程平台加入學年度+學期概念，讓學生可依學期分類查詢

## 格式

`semester` 欄位格式：`"114-2"`（學年度-學期）
- 114-1 = 114 學年度上學期
- 114-2 = 114 學年度下學期

## 變動

### DB: users 表加 semester 欄位
- `semester TEXT DEFAULT ''`
- system_settings 加 `current_semester` 預設值

### Model: Pydantic models 加 semester
- UserCreate, UserUpdate, UserOut

### API
- GET /admin/users 支援 ?semester= 篩選
- GET /analytics/summary 支援 ?semester= 篩選

### 前端
- UserManagement.tsx: 建立帳號選學期、列表篩選
- Sidebar/Layout: 學生顯示學期標籤
- system_settings: 管理員可設定目前學期
