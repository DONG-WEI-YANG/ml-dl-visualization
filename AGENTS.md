# 專案記憶

- 本儲存庫根目錄為 `D:\course\教材教具\ml-dl-visualization`；外層 `D:\course\教材教具` 也是獨立 Git 儲存庫，平台功能應在本儲存庫提交。
- 使用者於 2026-09-08 授權記憶、commit、push、部署學年班級與重修進度功能；真實學生名單之後再匯入，勿自行建立正式學生、匯入測試名單或宣稱正式開通完成。
- 正式前端：`https://dong-wei-yang.github.io/ml-dl-visualization/`，帳號管理路徑 `/admin/users`；後端：`https://kevin19830331-ml-dl-viz-api.hf.space`。
- `master` 推送會依變更路徑觸發 `.github/workflows/deploy-frontend.yml` 與 `deploy-backend.yml`，分別部署 GitHub Pages 和 `kevin19830331/ml-dl-viz-api` Space；部署須等 CI、發布與線上 API 驗證完成。
- `users.semester`、`users.class_name` 是目前歸屬；`enrollments` 保存各學期班級，`learning_events.semester` 固定活動當時學期。重修沿用原 user ID，舊紀錄缺乏學期資訊時保留空字串「未分類」，不得推測填入目前學期。
- 批次匯入停用／軟刪除的學生會重新啟用並重設初始密碼，保留歷史；已啟用重複帳號略過，非學生帳號不可由學生匯入流程恢復。教師僅可查閱 `teacher_students` 已指派學生。
- 教師進度名單 `/api/analytics/roster` 包含零活動學生，按學期分列；總學生數按 ID 去重。測驗週次與已評分作業週次分開，個人分析作業平均與總覽測驗／作業平均不可混稱。
- 詳細操作與升級說明：`docs/enrollment-progress.md`。遷移只加欄位與學籍快照，正式部署前確認資料保全；不要上傳本機 SQLite、初始密碼清單或測試資料。
- 驗證指令：後端 `.venv/Scripts/python.exe -m pytest tests/ -q`；前端 `npm test`、`npm run lint`、`npm run build`。2026-09-08 本機結果：後端 138 passed / 3 skipped（空題庫），前端 131 passed，lint/build 通過。此為當時結果，後續需按變更重新確認。
