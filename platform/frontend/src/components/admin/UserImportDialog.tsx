import { useMemo, useState } from "react";
import { fetchAPI } from "../../lib/api";
import { useAuth } from "../../hooks/useAuth";

interface ImportResult {
  created: { username: string; initial_password: string }[];
  skipped: { username: string; reason: string }[];
  restored?: { username: string; initial_password: string }[];
}

interface UserImportDialogProps {
  onDone: () => void;
  onClose: () => void;
}

export default function UserImportDialog({ onDone, onClose }: UserImportDialogProps) {
  const { token } = useAuth();
  const [raw, setRaw] = useState("");
  const [semester, setSemester] = useState("");
  const [result, setResult] = useState<ImportResult | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const rows = useMemo(
    () =>
      raw
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          const [username = "", display_name = "", email = ""] = line.split(",").map((s) => s.trim());
          return { username, display_name, email };
        }),
    [raw],
  );

  const submit = async () => {
    setError("");
    setSubmitting(true);
    try {
      const res = await fetchAPI<ImportResult>(
        "/api/admin/users/import",
        { semester, rows },
        token ?? undefined,
      );
      setResult(res);
      onDone();
    } catch {
      setError("匯入失敗，請稍後再試");
    } finally {
      setSubmitting(false);
    }
  };

  const downloadPasswords = () => {
    if (!result) return;
    const restored = result.restored ?? [];
    const csv = "﻿帳號,初始密碼\n" +
      [...result.created, ...restored].map((c) => `${c.username},${c.initial_password}`).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "初始密碼清單.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" role="dialog" aria-modal="true" aria-label="批次匯入學生">
      <div className="bg-white rounded-xl shadow-xl p-6 w-full max-w-2xl space-y-4 max-h-[85vh] overflow-y-auto">
        <h2 className="text-lg font-bold text-gray-800">批次匯入學生</h2>

        {!result ? (
          <>
            <p className="text-sm text-gray-500">
              每行一位學生：<code className="bg-gray-100 px-1 rounded">學號,姓名,Email</code>（姓名與 Email 可省略）。
              系統會自動產生初始密碼，學生首次登入需更換。
            </p>
            <label className="block text-sm text-gray-600">
              學期（留空使用目前學期）
              <input value={semester} onChange={(e) => setSemester(e.target.value)}
                placeholder="例如 115-1"
                className="mt-1 w-full border border-gray-300 rounded-lg px-3 py-2 text-sm" />
            </label>
            <label className="block text-sm text-gray-600">
              名單內容
              <textarea value={raw} onChange={(e) => setRaw(e.target.value)} rows={8}
                placeholder={"s1150001,王小明,ming@example.com\ns1150002,李小華"}
                className="mt-1 w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono" />
            </label>
            {rows.length > 0 && (
              <div className="border border-gray-200 rounded-lg overflow-hidden">
                <div className="bg-gray-50 px-3 py-1.5 text-xs text-gray-500">預覽（{rows.length} 筆）</div>
                <table className="min-w-full text-sm">
                  <tbody>
                    {rows.slice(0, 10).map((r, i) => (
                      <tr key={i} className="border-t border-gray-100">
                        <td className="px-3 py-1.5 font-mono">{r.username || "（空）"}</td>
                        <td className="px-3 py-1.5">{r.display_name}</td>
                        <td className="px-3 py-1.5 text-gray-500">{r.email}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {rows.length > 10 && (
                  <div className="px-3 py-1.5 text-xs text-gray-400">…其餘 {rows.length - 10} 筆</div>
                )}
              </div>
            )}
            {error && <p className="text-sm text-red-600">{error}</p>}
            <div className="flex justify-end gap-2">
              <button onClick={onClose} className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg">取消</button>
              <button onClick={submit} disabled={submitting || rows.length === 0}
                className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
                開始匯入
              </button>
            </div>
          </>
        ) : (
          <>
            {(() => {
              const restored = result.restored ?? [];
              return (
                <>
                  <p className="text-sm text-gray-600">
                    成功建立 {result.created.length} 筆，還原 {restored.length} 筆，略過 {result.skipped.length} 筆。
                    <span className="text-amber-600">初始密碼僅顯示這一次，請立即下載保存。</span>
                  </p>
                  {result.created.length > 0 && (
                    <div className="border border-gray-200 rounded-lg overflow-hidden">
                      <table className="min-w-full text-sm">
                        <thead className="bg-gray-50 text-left text-gray-600">
                          <tr><th className="px-3 py-1.5">帳號</th><th className="px-3 py-1.5">初始密碼</th></tr>
                        </thead>
                        <tbody>
                          {result.created.map((c) => (
                            <tr key={c.username} className="border-t border-gray-100">
                              <td className="px-3 py-1.5 font-mono">{c.username}</td>
                              <td className="px-3 py-1.5 font-mono">{c.initial_password}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {restored.length > 0 && (
                    <div>
                      <h3 className="text-sm font-semibold text-teal-700">已還原帳號</h3>
                      <p className="text-xs text-gray-500 mb-1">原有學習歷程已保留，僅重設密碼並要求重新設定。</p>
                      <div className="border border-teal-200 rounded-lg overflow-hidden">
                        <table className="min-w-full text-sm">
                          <thead className="bg-teal-50 text-left text-teal-800">
                            <tr><th className="px-3 py-1.5">帳號</th><th className="px-3 py-1.5">初始密碼</th></tr>
                          </thead>
                          <tbody>
                            {restored.map((r) => (
                              <tr key={r.username} className="border-t border-teal-100">
                                <td className="px-3 py-1.5 font-mono">{r.username}</td>
                                <td className="px-3 py-1.5 font-mono">{r.initial_password}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                  {result.skipped.length > 0 && (
                    <ul className="text-sm text-gray-500 list-disc pl-5">
                      {result.skipped.map((s, i) => (
                        <li key={i}>{s.username || "（空）"}：{s.reason}</li>
                      ))}
                    </ul>
                  )}
                  <div className="flex justify-end gap-2">
                    <button onClick={downloadPasswords} disabled={result.created.length === 0 && restored.length === 0}
                      className="px-4 py-2 text-sm border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 disabled:opacity-40">
                      下載初始密碼 CSV
                    </button>
                    <button onClick={onClose} className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700">完成</button>
                  </div>
                </>
              );
            })()}
          </>
        )}
      </div>
    </div>
  );
}
