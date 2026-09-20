# 2026-09-20 稽核修復實作

依據 [原稽核](audit-2026-09-20.md)，本次完成本機修復、回歸測試與部署準備。正式 HF Space、正式資料庫、學生帳號尚未變更；不將本機測試結果當作已上線證據。

## 驗證結果

- 既有 backend venv：`pytest tests/ -q` **236 passed、0 skipped**，兩項既有 SnowNLP 棄用警告。
- 前端：**23 個測試檔、135 tests passed**；`npm run lint` 無錯誤／警告，`npm run build` 成功，保留既有大型 chunk 提醒。
- `npm audit --json`：**0 vulnerabilities**。
- 新版後端核心依賴另在暫存目錄安裝並與既有其他套件合併測試；詳細套件版本、掃描與剩餘風險列於 [依賴修復報告](dependency-remediation-2026-09-20.md)。原本 venv 未被整批升級。
- 新測試涵蓋撤銷 token、強制改密碼、前端新舊 session 競爭、評分授權、每週題庫、輸入／並行限制與取消請求、共用 DB、備份還原、部署檔案排除及持久路徑檢查。未以 mock 測試結果宣稱正式儲存已完成。
- `git diff --check` 通過；未進行正式站寫入。

## 已實作範圍

| 稽核項目 | 實作與狀態 |
|---|---|
| F01 成績可自行提交 | `/analytics/events` 僅允許無成績、無指定時間的視覺化互動。作業改由 `/analytics/assignments/grade`，驗證教師指派、有效學生與分數，保存評分來源並寫稽核紀錄；學生 ID 正規化避免孤立成績。測驗與助教事件保留伺服器產生路徑。 |
| F02 空題庫 | 加入版本化 54 題基礎題庫，每週 3 題，列明教材來源。啟動驗證後只新增缺漏 ID，保留教師編輯及自訂題。測試不再因空題庫跳過；正式站仍待部署。 |
| F03 預設 JWT | 預設 APP_ENV=production，拒絕公開預設或不足 32 字元的 key。初次建立正式管理員要求至少 12 字元密碼；不改寫既有管理員。開發／測試環境需明確選擇。 |
| F04 強制改密碼 | HTTP 與 WebSocket 均驗證旗標，只允許本人資訊、改密碼與登出。前端先完成改密碼再載入受保護頁面。 |
| F05 token 撤銷 | JWT 含唯一 jti 與帳號 session_version；登出撤銷當次 session，改密碼／重設／實際權限或停用變更撤銷原 session。改密碼回傳新 token，前端直接接收。WebSocket 每次請求及輸出前再驗證。 |
| F06 運算資源 | 限制樣本、特徵、迭代、樹數、深度、解析度、數值與形狀；計算移出事件迴圈。每 process 至多兩個工作，超載 503、錯誤輸入或發散 422。 |
| F07 資料保存 | 主 DB 與 RAG 共用 DATABASE_PATH，新增一致性 SQLite 備份／還原工具及 HF 持久磁碟／實際 DB 路徑啟動檢查。HF 程式、教材、Dockerfile 與 metadata 合併為單次上傳，避免中間版本啟動。正式備份、磁碟供應與遷移尚未執行。 |
| F08 Compose 教材 | 掛載持久資料 volume 與唯讀 curriculum，明確指定 CURRICULUM_DIR。Docker daemon 未可用，仍需實際容器驗收。 |
| F09 final_loss | 使用最後權重重算最終 MSE，history 保持原更新前狀態語意。 |
| F10 測試隔離 | RAG 動態使用共同 DB connection，pytest 僅指向暫存 DB；測試設定不繼承本機正式金鑰。 |
| F11 readiness | 無教材回報 unavailable/degraded；匯入錯誤回報 error/degraded；核心資料庫初始化失敗拒絕啟動。 |
| F12 依賴 | 前端相容更新並升級 Vitest，掃描降為 0。後端核心套件固定至獨立環境驗證版本；未完成所有 Python／正式 image 漏洞清零，見 [依賴修復報告](dependency-remediation-2026-09-20.md)。 |
| F13 外層啟動 | 外層 README 指向真正平台；舊匯入腳本改為明確來源／目的參數的保留原檔複製工具，拒絕覆寫，不再匯入不存在的 catalog。 |

## 行為與相容性

- 此次 JWT 格式升級會拒絕舊 token；部署後既有使用者需重新登入。既有帳號、學籍、成績與密碼雜湊採加欄位遷移保留。
- 改密碼 API 現回傳 `{access_token, token_type, user}`。前端也支援舊後端的成功回應並重新查驗 session，允許前端先行發布。
- 任何角色都不能再用一般事件 API 自行注入 quiz／assignment／llm_chat。外部整合需改用測驗提交或受權限保護的評分 API。
- 題庫為每週短概念測驗，並非完整課程驗收題庫。基礎題刪除後，下一次初始化會補回；既有編輯不會被新版題目文字自動覆寫。
- 離線登出可清除瀏覽器本機登入狀態，但無法保證伺服器收到撤銷；伺服器登出成功才撤銷當次 token。改密碼可撤銷該帳號全部先前 session。
- 模型採有上限的 thread 工作，而非可強制終止的 process；並行名額按 process 計算，多 worker 部署需依總資源調整。

## 部署前尚須執行

1. 依 [儲存操作](storage-operations.md) 對真實資料做備份、還原驗證及持久化遷移；禁止用本機 DB 代替正式資料。完成後才設定部署檢查旗標。
2. 在 Linux / Python 3.11 clean image 重新安裝、測試及掃描全部依賴；本機 Windows / Python 3.14 的既有 venv 與候選套件驗證不等同完整 image 驗證。
3. 實際驗證 Docker 啟動、教材下載、重啟／重建後資料保存，以及正式站 18 週非空題庫。
4. 執行正式登入、改密碼、登出、教師評分與學期進度瀏覽器驗收。現有前端回歸包含真實 App/AuthProvider 的改密碼流程，但沒有宣稱正式瀏覽器全流程驗收已完成。

本報告記錄實作驗證結果；發布狀態以 GitHub Actions 與正式站查核為準。未匯入正式學生資料。
