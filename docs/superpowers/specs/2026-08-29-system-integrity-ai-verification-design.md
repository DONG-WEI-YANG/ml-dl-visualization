# 系統完整性與 AI 真實性強化設計

## 目標

在不重寫既有 React/FastAPI 架構的前提下，完成十輪可驗證強化，使學習資料、測驗結果、登入身份、AI provider 與 UI 顯示採用同一組合約，並讓測試工具本身可穩定量測。

## 現況證據

- 後端基線為 97 passed、3 skipped，整體 line coverage 54%。
- 前端基線為 117 passed，但測試輸出含 React `act`、jsdom navigation/canvas 與 R3F 標籤警告。
- 前端 coverage 指令缺少 `@vitest/coverage-v8`，目前不能產生覆蓋率。
- lint 有六個 Hook/Fast Refresh 警告；production build 有 1 MB 以上 chunk 警告。
- `/api/analytics/events` 接受呼叫端提供的 `student_id`，且分析讀取端沒有身份/角色約束。
- 測驗以送出的答案數計分，缺答不計入分母，亦未一致表示 `selected`/`answer` 與前端期待的 `user_answer`/`correct_answer`。
- AI model-info 回傳設定值，不揭露缺 key 時實際降級到 local 的狀態；個人化 helper 存在，但 chat 沒有傳入登入者 ID。
- WebSocket payload 缺少 Pydantic 邊界驗證，前端也沒有把錯誤內容呈現給使用者。

## 架構選擇

採取漸進式「邊界合約」方案：Pydantic 負責 API 資料約束，服務函式負責交易與計分規則，provider factory/diagnostics 負責 configured/effective 狀態，React hooks 負責 socket 生命週期，頁面只呈現型別化狀態。這比重建資料層或引入新狀態管理器風險低，也能由單元與整合測試逐段證明。

## 資料與權限合約

- `LearningEvent` 僅接受 week 1–18、明確事件類型、非負 duration、0–100 分數與獨立 metadata dict。
- 學生建立事件時，後端以 token 的 user id 覆寫/決定 `student_id`；教師與管理員可讀取授權範圍內的分析資料。
- SQLite 使用 context manager，所有成功、例外與 health 路徑都會關閉連線。
- 測驗總題數來自該週資料庫題目；缺答列入錯誤，未知題號不改變分母，選項索引必須有效。

## AI 真實性合約

- provider diagnostics 同時回傳 `configured_provider/model`、`effective_provider/model`、`status` 與不洩漏秘密的 reason。
- 「available」只代表可建立/具備必要設定；`/api/llm/diagnostics?probe=true` 才會執行最小真實 probe。local probe 必須實際走本地 pipeline；外部 provider 在無 key 時明確 degraded，不假裝成功。
- HTTP 與 WebSocket chat 都把登入學生 ID 傳給 tutor，使個人化資料真正進入 system prompt。
- OpenAI、Anthropic、Ollama provider 對空回覆、HTTP 錯誤與畸形 stream 明確失敗，避免把無內容當成功。

## 前端狀態與視覺設計

沿用現有白底卡片、藍色互動與綠/琥珀狀態語言，不進行換皮。管理頁新增「AI 運作狀態」卡：以 `設定 → 實際執行` 路徑作為唯一識別元素，顯示是否降級、模型與 probe 結果。聊天面板新增可讀的錯誤訊息、重試動作與停止生成按鈕；所有狀態具 aria live、鍵盤 focus 與 mobile wrap。

## 測試策略

- 每項行為先寫失敗測試並確認失敗原因，再作最小實作。
- 後端涵蓋 model validation、connection cleanup、quiz completeness、analytics auth、provider diagnostics、personalization 與 provider edge cases。
- 前端涵蓋 API error payload、socket malformed/error/cancel、quiz error UI、diagnostics card 與 lazy route。
- 最終執行後端全量/coverage、前端全量/coverage、lint、type/build，並檢查 Git diff 與文件一致性。

## 非目標

- 不使用真實付費 API key 執行會產生成本的外部模型請求。
- 不重建資料庫 ORM、不更換視覺框架、不加入與本教學系統無關的功能。
- 不以追求覆蓋率數字為由測試第三方框架或純展示常數。
