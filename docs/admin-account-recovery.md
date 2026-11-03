# 管理員帳號與密碼重設

## 登入方式

- 管理員帳號固定為 **`admin`**（請輸入帳號名稱，不是 Email；登入只比對帳號欄位）。
- `DEFAULT_ADMIN_PASSWORD` 只在資料庫**完全沒有管理員**時建立初始帳號；既有資料庫改它不會生效。

## 忘記密碼：一次性重設（`ADMIN_RESET_PASSWORD`）

1. 到 Hugging Face Space `kevin19830331/ml-dl-viz-api` → **Settings → Variables and secrets**，
   新增 **Secret** `ADMIN_RESET_PASSWORD`，值為 12 字元以上的臨時密碼。
2. Space 會自動重啟（若沒有，按 **Restart**）。啟動時後端會：
   - 把 `admin` 的密碼設為該值，並重新啟用帳號；
   - 強制下次登入必須改密碼；
   - 讓舊的登入 session 失效，並在稽核紀錄寫入 `user.password_reset`（via `ADMIN_RESET_PASSWORD`）。
3. 用 `admin` + 臨時密碼登入，依提示改成新密碼。
4. **回到 Space 設定刪除 `ADMIN_RESET_PASSWORD`**。

同一個值只會套用一次（系統只保存加鹽雜湊作為標記），所以即使忘了刪除，重啟也不會把你改好的密碼蓋回去；
但仍請刪除，避免臨時密碼外洩。值少於 12 字元會被忽略並記錄警告。

## 學生自助申請帳號

- 登入頁「申請帳號」→ 學生填學號、姓名、班級、密碼，建立 **待審核** 學生帳號（無法登入）。
- 管理員在「帳號管理」頂端的「待審核的帳號申請」核准或不通過；兩者都寫入稽核紀錄。
- 申請角色一律為學生，前端無法指定角色。
