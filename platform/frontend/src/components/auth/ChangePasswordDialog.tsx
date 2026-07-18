import { useState } from "react";
import { fetchAPI, APIError } from "../../lib/api";
import { useAuth } from "../../hooks/useAuth";

interface ChangePasswordDialogProps {
  forced?: boolean;
  onClose: () => void;
}

export default function ChangePasswordDialog({ forced = false, onClose }: ChangePasswordDialogProps) {
  const { token, retryVerification } = useAuth();
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const submit = async () => {
    setError("");
    if (newPassword.length < 8) {
      setError("新密碼長度至少 8 碼");
      return;
    }
    if (newPassword !== confirm) {
      setError("兩次輸入的新密碼不一致");
      return;
    }
    setSubmitting(true);
    try {
      await fetchAPI("/api/auth/change-password",
        { old_password: oldPassword, new_password: newPassword }, token ?? undefined);
      await retryVerification();
      onClose();
    } catch (e) {
      if (e instanceof APIError && e.status === 401) setError("舊密碼錯誤");
      else setError("變更失敗，請稍後再試");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" role="dialog" aria-modal="true" aria-label="變更密碼">
      <div className="bg-white rounded-xl shadow-xl p-6 w-full max-w-sm space-y-4">
        <h2 className="text-lg font-bold text-gray-800">變更密碼</h2>
        {forced && (
          <p className="text-sm text-amber-600 bg-amber-50 rounded-lg p-2">
            首次登入請更換密碼後再繼續使用。
          </p>
        )}
        <div className="space-y-3">
          <label className="block text-sm text-gray-600">
            舊密碼
            <input type="password" value={oldPassword} onChange={(e) => setOldPassword(e.target.value)}
              className="mt-1 w-full border border-gray-300 rounded-lg px-3 py-2 text-sm" />
          </label>
          <label className="block text-sm text-gray-600">
            新密碼
            <input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)}
              className="mt-1 w-full border border-gray-300 rounded-lg px-3 py-2 text-sm" />
          </label>
          <label className="block text-sm text-gray-600">
            確認新密碼
            <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)}
              className="mt-1 w-full border border-gray-300 rounded-lg px-3 py-2 text-sm" />
          </label>
        </div>
        {error && <p className="text-sm text-red-600">{error}</p>}
        <div className="flex justify-end gap-2">
          {!forced && (
            <button onClick={onClose} className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg">
              取消
            </button>
          )}
          <button onClick={submit} disabled={submitting}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
            確認變更
          </button>
        </div>
      </div>
    </div>
  );
}
