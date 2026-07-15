interface Props {
  phase: "connecting" | "waking" | "unavailable";
  onEnterOffline: () => void;
  onRetry: () => void;
  onLogout: () => void;
}

export default function CloudLoadingState({ phase, onEnterOffline, onRetry, onLogout }: Props) {
  const waking = phase === "waking";
  const unavailable = phase === "unavailable";
  return (
    <main className="min-h-screen bg-slate-50 grid place-items-center p-6">
      <section role="status" aria-live="polite" className="w-full max-w-md rounded-3xl border border-slate-200 bg-white p-8 shadow-xl shadow-slate-200/60">
        <div className={`mx-auto mb-6 grid h-16 w-16 place-items-center rounded-2xl text-2xl ${unavailable ? "bg-rose-50 text-rose-600" : waking ? "bg-amber-50 text-amber-600" : "bg-blue-50 text-blue-600"}`}>
          {unavailable ? "!" : waking ? "☁" : "↗"}
        </div>
        <h1 className="text-center text-2xl font-bold text-slate-900">
          {unavailable ? "雲端暫時無法連線" : waking ? "正在喚醒雲端服務" : "正在連線課程平台"}
        </h1>
        <p className="mt-3 text-center leading-7 text-slate-600">
          {unavailable ? "你的登入資料仍安全，可先進入課程瀏覽本地教材。" : waking ? "教材可先瀏覽，登入、測驗與 AI 助教稍後恢復。" : "正在確認登入狀態，通常只需要幾秒鐘。"}
        </p>
        <div className="mt-7 grid gap-3">
          {(waking || unavailable) && <button onClick={onEnterOffline} className="rounded-xl bg-blue-600 px-4 py-3 font-semibold text-white hover:bg-blue-700">{unavailable ? "進入離線課程" : "先瀏覽課程"}</button>}
          {(waking || unavailable) && <button onClick={onRetry} className="rounded-xl bg-slate-100 px-4 py-3 font-semibold text-slate-700 hover:bg-slate-200">重新連線</button>}
          {unavailable && <button onClick={onLogout} className="px-4 py-2 text-sm font-medium text-slate-500 hover:text-slate-800">登出並返回登入</button>}
        </div>
      </section>
    </main>
  );
}
