import { useState, useEffect } from "react";
import { useAuth } from "../hooks/useAuth";

interface User {
  id: number;
  username: string;
  display_name: string;
  email: string;
  role: "admin" | "teacher" | "student";
  semester: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

const ROLE_LABELS: Record<string, string> = {
  admin: "管理員",
  teacher: "教師",
  student: "學生",
};

const ROLE_COLORS: Record<string, string> = {
  admin: "bg-red-100 text-red-700",
  teacher: "bg-blue-100 text-blue-700",
  student: "bg-green-100 text-green-700",
};

const CURRENT_YEAR = 114; // 民國年
const SEMESTER_OPTIONS = Array.from({ length: 11 }, (_, i) => {
  const y = CURRENT_YEAR - 5 + i;
  return [`${y}-1`, `${y}-2`];
}).flat();

function formatSemester(s: string): string {
  if (!s) return "-";
  const [year, sem] = s.split("-");
  return `${year} ${sem === "1" ? "上" : "下"}`;
}

export default function UserManagement() {
  const { user: me, token, logout } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [filterRole, setFilterRole] = useState<string>("");
  const [filterSemester, setFilterSemester] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState<{ text: string; type: "ok" | "err" }>({ text: "", type: "ok" });

  // Create form
  const [showCreate, setShowCreate] = useState(false);
  const [createForm, setCreateForm] = useState({ username: "", password: "", display_name: "", email: "", role: "student", semester: "" });

  // Edit form
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [editForm, setEditForm] = useState({ display_name: "", email: "", role: "", password: "", semester: "" });

  // Teacher-student assignment
  const [assignMode, setAssignMode] = useState<{ teacherId: number; teacherName: string } | null>(null);
  const [teacherStudents, setTeacherStudents] = useState<User[]>([]);

  const authFetch = async (path: string, method = "GET", body?: unknown) => {
    const res = await fetch(path, {
      method,
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (res.status === 401 || res.status === 403) { logout(); throw new Error("未授權"); }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Error ${res.status}`);
    }
    return res.json();
  };

  const flash = (text: string, type: "ok" | "err" = "ok") => {
    setMessage({ text, type });
    setTimeout(() => setMessage({ text: "", type: "ok" }), 3000);
  };

  const fetchUsers = async () => {
    try {
      const params = new URLSearchParams();
      if (filterRole) params.set("role", filterRole);
      if (filterSemester) params.set("semester", filterSemester);
      const qs = params.toString();
      const url = `/api/admin/users${qs ? `?${qs}` : ""}`;
      const data = await authFetch(url);
      setUsers(data);
    } catch { flash("無法載入使用者列表", "err"); }
    setLoading(false);
  };

  useEffect(() => { if (token) fetchUsers(); }, [token, filterRole, filterSemester]);

  if (me?.role !== "admin") {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8">
        <h1 className="text-xl font-bold text-gray-900 mb-2">權限不足</h1>
        <p className="text-gray-500">僅管理員可存取此頁面</p>
      </div>
    );
  }

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await authFetch("/api/auth/register", "POST", createForm);
      flash("帳號建立成功");
      setShowCreate(false);
      setCreateForm({ username: "", password: "", display_name: "", email: "", role: "student", semester: "" });
      fetchUsers();
    } catch (err: unknown) {
      flash((err as Error).message || "建立失敗", "err");
    }
  };

  const handleEdit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingUser) return;
    const body: Record<string, unknown> = {};
    if (editForm.display_name !== editingUser.display_name) body.display_name = editForm.display_name;
    if (editForm.email !== editingUser.email) body.email = editForm.email;
    if (editForm.role !== editingUser.role) body.role = editForm.role;
    if (editForm.semester !== (editingUser.semester || "")) body.semester = editForm.semester;
    if (editForm.password) body.password = editForm.password;
    if (Object.keys(body).length === 0) { setEditingUser(null); return; }
    try {
      await authFetch(`/api/admin/users/${editingUser.id}`, "PUT", body);
      flash("帳號已更新");
      setEditingUser(null);
      fetchUsers();
    } catch (err: unknown) {
      flash((err as Error).message || "更新失敗", "err");
    }
  };

  const handleToggleActive = async (u: User) => {
    if (u.id === me?.id) { flash("無法停用自己的帳號", "err"); return; }
    const action = u.is_active ? "停用" : "啟用";
    if (!confirm(`確定要${action} ${u.display_name || u.username}？`)) return;
    try {
      if (u.is_active) {
        await authFetch(`/api/admin/users/${u.id}`, "DELETE");
      } else {
        await authFetch(`/api/admin/users/${u.id}`, "PUT", { is_active: true });
      }
      flash(`已${action} ${u.display_name || u.username}`);
      fetchUsers();
    } catch (err: unknown) {
      flash((err as Error).message || `${action}失敗`, "err");
    }
  };

  const openAssign = async (teacher: User) => {
    setAssignMode({ teacherId: teacher.id, teacherName: teacher.display_name || teacher.username });
    try {
      const data = await authFetch(`/api/admin/teachers/${teacher.id}/students`);
      setTeacherStudents(data);
    } catch { setTeacherStudents([]); }
  };

  const handleAssign = async (studentId: number) => {
    if (!assignMode) return;
    try {
      await authFetch(`/api/admin/teachers/${assignMode.teacherId}/students/${studentId}`, "POST");
      flash("已指派學生");
      openAssign({ id: assignMode.teacherId, display_name: assignMode.teacherName } as User);
    } catch (err: unknown) {
      flash((err as Error).message || "指派失敗", "err");
    }
  };

  const handleUnassign = async (studentId: number) => {
    if (!assignMode) return;
    try {
      await authFetch(`/api/admin/teachers/${assignMode.teacherId}/students/${studentId}`, "DELETE");
      flash("已移除學生");
      openAssign({ id: assignMode.teacherId, display_name: assignMode.teacherName } as User);
    } catch (err: unknown) {
      flash((err as Error).message || "移除失敗", "err");
    }
  };

  const students = users.filter((u) => u.role === "student" && u.is_active);
  const assignedIds = new Set(teacherStudents.map((s) => s.id));

  const counts = {
    total: users.length,
    admin: users.filter((u) => u.role === "admin").length,
    teacher: users.filter((u) => u.role === "teacher").length,
    student: users.filter((u) => u.role === "student").length,
    active: users.filter((u) => u.is_active).length,
  };

  return (
    <div className="max-w-5xl mx-auto p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">帳號管理</h1>
        <button
          onClick={() => setShowCreate(true)}
          className="px-4 py-2 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 transition-colors"
        >
          + 新增帳號
        </button>
      </div>

      {/* Message */}
      {message.text && (
        <div className={`px-4 py-2 rounded-lg text-sm ${message.type === "err" ? "bg-red-50 text-red-700" : "bg-green-50 text-green-700"}`}>
          {message.text}
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-5 gap-3">
        {[
          { label: "全部", value: counts.total, color: "bg-gray-50 text-gray-700" },
          { label: "管理員", value: counts.admin, color: "bg-red-50 text-red-700" },
          { label: "教師", value: counts.teacher, color: "bg-blue-50 text-blue-700" },
          { label: "學生", value: counts.student, color: "bg-green-50 text-green-700" },
          { label: "啟用中", value: counts.active, color: "bg-yellow-50 text-yellow-700" },
        ].map((s) => (
          <div key={s.label} className={`${s.color} rounded-xl p-3 text-center`}>
            <div className="text-2xl font-bold">{s.value}</div>
            <div className="text-xs mt-0.5">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Filter */}
      <div className="flex gap-2">
        {["", "admin", "teacher", "student"].map((r) => (
          <button
            key={r}
            onClick={() => setFilterRole(r)}
            className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
              filterRole === r
                ? "bg-gray-900 text-white"
                : "bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
            }`}
          >
            {r ? ROLE_LABELS[r] : "全部"}
          </button>
        ))}
      </div>

      {/* Semester Filter */}
      <div className="flex items-center gap-2">
        <select
          value={filterSemester}
          onChange={(e) => setFilterSemester(e.target.value)}
          className="px-3 py-1.5 text-xs border border-gray-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-300"
        >
          <option value="">全部學期</option>
          {SEMESTER_OPTIONS.map((s) => (
            <option key={s} value={s}>{formatSemester(s)}</option>
          ))}
        </select>
      </div>

      {/* User Table */}
      {loading ? (
        <div className="text-gray-400 text-center py-8">載入中...</div>
      ) : (
        <div className="border border-gray-200 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-gray-600">
              <tr>
                <th className="text-left px-4 py-3 font-medium">帳號</th>
                <th className="text-left px-4 py-3 font-medium">顯示名稱</th>
                <th className="text-left px-4 py-3 font-medium">Email</th>
                <th className="text-left px-4 py-3 font-medium">角色</th>
                <th className="text-left px-4 py-3 font-medium">學期</th>
                <th className="text-left px-4 py-3 font-medium">狀態</th>
                <th className="text-right px-4 py-3 font-medium">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {users.map((u) => (
                <tr key={u.id} className={`${!u.is_active ? "opacity-50 bg-gray-50" : "hover:bg-gray-50"}`}>
                  <td className="px-4 py-3 font-mono text-gray-900">{u.username}</td>
                  <td className="px-4 py-3 text-gray-700">{u.display_name}</td>
                  <td className="px-4 py-3 text-gray-500">{u.email || "-"}</td>
                  <td className="px-4 py-3">
                    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${ROLE_COLORS[u.role]}`}>
                      {ROLE_LABELS[u.role]}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-500">{formatSemester(u.semester)}</td>
                  <td className="px-4 py-3">
                    <span className={`text-xs ${u.is_active ? "text-green-600" : "text-gray-400"}`}>
                      {u.is_active ? "啟用" : "停用"}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right space-x-1">
                    <button
                      onClick={() => {
                        setEditingUser(u);
                        setEditForm({ display_name: u.display_name, email: u.email, role: u.role, password: "", semester: u.semester || "" });
                      }}
                      className="px-2 py-1 text-xs text-blue-600 hover:bg-blue-50 rounded"
                    >
                      編輯
                    </button>
                    {u.role === "teacher" && u.is_active && (
                      <button
                        onClick={() => openAssign(u)}
                        className="px-2 py-1 text-xs text-purple-600 hover:bg-purple-50 rounded"
                      >
                        學生
                      </button>
                    )}
                    {u.id !== me?.id && (
                      <button
                        onClick={() => handleToggleActive(u)}
                        className={`px-2 py-1 text-xs rounded ${u.is_active ? "text-red-600 hover:bg-red-50" : "text-green-600 hover:bg-green-50"}`}
                      >
                        {u.is_active ? "停用" : "啟用"}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
              {users.length === 0 && (
                <tr><td colSpan={7} className="text-center py-8 text-gray-400">沒有使用者</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Create Modal */}
      {showCreate && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <form onSubmit={handleCreate} className="bg-white rounded-2xl shadow-xl p-6 w-full max-w-md space-y-4">
            <h2 className="text-lg font-bold text-gray-900">新增帳號</h2>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">帳號 *</label>
              <input
                required value={createForm.username}
                onChange={(e) => setCreateForm({ ...createForm, username: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">密碼 *</label>
              <input
                required type="password" value={createForm.password}
                onChange={(e) => setCreateForm({ ...createForm, password: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">顯示名稱</label>
              <input
                value={createForm.display_name}
                onChange={(e) => setCreateForm({ ...createForm, display_name: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
              <input
                type="email" value={createForm.email}
                onChange={(e) => setCreateForm({ ...createForm, email: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">角色 *</label>
              <select
                value={createForm.role}
                onChange={(e) => setCreateForm({ ...createForm, role: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="student">學生</option>
                <option value="teacher">教師</option>
                <option value="admin">管理員</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">學期</label>
              <select
                value={createForm.semester}
                onChange={(e) => setCreateForm({ ...createForm, semester: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">自動（使用系統預設）</option>
                {SEMESTER_OPTIONS.map((s) => (
                  <option key={s} value={s}>{formatSemester(s)}</option>
                ))}
              </select>
            </div>
            <div className="flex gap-2 justify-end pt-2">
              <button type="button" onClick={() => setShowCreate(false)} className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg">取消</button>
              <button type="submit" className="px-4 py-2 text-sm bg-blue-500 text-white rounded-lg hover:bg-blue-600">建立</button>
            </div>
          </form>
        </div>
      )}

      {/* Edit Modal */}
      {editingUser && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <form onSubmit={handleEdit} className="bg-white rounded-2xl shadow-xl p-6 w-full max-w-md space-y-4">
            <h2 className="text-lg font-bold text-gray-900">
              編輯帳號 <span className="font-mono text-gray-500">{editingUser.username}</span>
            </h2>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">顯示名稱</label>
              <input
                value={editForm.display_name}
                onChange={(e) => setEditForm({ ...editForm, display_name: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
              <input
                type="email" value={editForm.email}
                onChange={(e) => setEditForm({ ...editForm, email: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">角色</label>
              <select
                value={editForm.role}
                onChange={(e) => setEditForm({ ...editForm, role: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="student">學生</option>
                <option value="teacher">教師</option>
                <option value="admin">管理員</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">學期</label>
              <select
                value={editForm.semester}
                onChange={(e) => setEditForm({ ...editForm, semester: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">未設定</option>
                {SEMESTER_OPTIONS.map((s) => (
                  <option key={s} value={s}>{formatSemester(s)}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">新密碼 (留空不修改)</label>
              <input
                type="password" value={editForm.password}
                onChange={(e) => setEditForm({ ...editForm, password: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="••••••••"
              />
            </div>
            <div className="flex gap-2 justify-end pt-2">
              <button type="button" onClick={() => setEditingUser(null)} className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg">取消</button>
              <button type="submit" className="px-4 py-2 text-sm bg-blue-500 text-white rounded-lg hover:bg-blue-600">儲存</button>
            </div>
          </form>
        </div>
      )}

      {/* Teacher-Student Assignment Modal */}
      {assignMode && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl shadow-xl p-6 w-full max-w-lg space-y-4">
            <h2 className="text-lg font-bold text-gray-900">
              {assignMode.teacherName} 的學生
            </h2>

            {/* Current students */}
            {teacherStudents.length > 0 ? (
              <div className="space-y-1.5">
                <p className="text-xs font-medium text-gray-500">已指派學生</p>
                {teacherStudents.map((s) => (
                  <div key={s.id} className="flex items-center justify-between px-3 py-2 bg-green-50 rounded-lg">
                    <span className="text-sm text-gray-800">{s.display_name || s.username}</span>
                    <button
                      onClick={() => handleUnassign(s.id)}
                      className="text-xs text-red-500 hover:bg-red-50 px-2 py-1 rounded"
                    >
                      移除
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-400">尚未指派學生</p>
            )}

            {/* Unassigned students */}
            {students.filter((s) => !assignedIds.has(s.id)).length > 0 && (
              <div className="space-y-1.5">
                <p className="text-xs font-medium text-gray-500 mt-3">可指派學生</p>
                {students.filter((s) => !assignedIds.has(s.id)).map((s) => (
                  <div key={s.id} className="flex items-center justify-between px-3 py-2 bg-gray-50 rounded-lg">
                    <span className="text-sm text-gray-600">{s.display_name || s.username}</span>
                    <button
                      onClick={() => handleAssign(s.id)}
                      className="text-xs text-blue-600 hover:bg-blue-50 px-2 py-1 rounded"
                    >
                      + 指派
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="flex justify-end pt-2">
              <button onClick={() => setAssignMode(null)} className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg">關閉</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
