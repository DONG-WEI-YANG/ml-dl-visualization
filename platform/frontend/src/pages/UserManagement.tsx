import { useState, useEffect, useCallback } from "react";
import { useAuth } from "../hooks/useAuth";
import { API_BASE } from "../lib/api";
import UserImportDialog from "../components/admin/UserImportDialog";

interface User {
  id: number;
  username: string;
  display_name: string;
  email: string;
  role: "admin" | "teacher" | "student";
  semester: string;
  class_name: string;
  is_active: boolean;
  approval_status?: "approved" | "pending" | "rejected";
  created_at: string;
  updated_at: string;
}

type BulkAction = "activate" | "deactivate" | "reset_password" | "set_class";

const STATUS_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "全部狀態" },
  { value: "active", label: "啟用" },
  { value: "inactive", label: "停用" },
  { value: "pending", label: "待審核" },
  { value: "rejected", label: "未通過" },
];

function statusLabel(u: User): { text: string; cls: string } {
  if (u.approval_status === "pending") return { text: "待審核", cls: "text-amber-600" };
  if (u.approval_status === "rejected") return { text: "未通過", cls: "text-red-500" };
  return u.is_active ? { text: "啟用", cls: "text-green-600" } : { text: "停用", cls: "text-gray-400" };
}

function downloadPasswordCsv(rows: { username: string; initial_password: string }[], filename: string) {
  const csv = "\uFEFF帳號,初始密碼\n" + rows.map((r) => `${r.username},${r.initial_password}`).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
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

const CURRENT_YEAR = new Date().getFullYear() - 1911 - (new Date().getMonth() < 7 ? 1 : 0);
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
  const [filterClass, setFilterClass] = useState("");
  const [filterYear, setFilterYear] = useState("");
  const [filterSemester, setFilterSemester] = useState<string>("");
  const [filterStatus, setFilterStatus] = useState("");
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [pending, setPending] = useState<User[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [resetPasswords, setResetPasswords] = useState<{ username: string; initial_password: string }[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState<{ text: string; type: "ok" | "err" }>({ text: "", type: "ok" });

  // Batch import
  const [showImport, setShowImport] = useState(false);

  // Create form
  const [showCreate, setShowCreate] = useState(false);
  const [createForm, setCreateForm] = useState({ username: "", password: "", display_name: "", email: "", role: "student", semester: "", class_name: "" });

  // Edit form
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [editForm, setEditForm] = useState({ display_name: "", email: "", role: "", password: "", semester: "", class_name: "" });

  // Teacher-student assignment
  const [assignMode, setAssignMode] = useState<{ teacherId: number; teacherName: string } | null>(null);
  const [teacherStudents, setTeacherStudents] = useState<User[]>([]);

  const authFetch = useCallback(async (path: string, method = "GET", body?: unknown) => {
    const res = await fetch(`${API_BASE}${path}`, {
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
  }, [logout, token]);

  const flash = useCallback((text: string, type: "ok" | "err" = "ok") => {
    setMessage({ text, type });
    setTimeout(() => setMessage({ text: "", type: "ok" }), 3000);
  }, []);

  const fetchUsers = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      if (filterRole) params.set("role", filterRole);
      if (filterSemester) params.set("semester", filterSemester);
      if (filterClass.trim()) params.set("class_name", filterClass.trim());
      if (filterYear.trim()) params.set("academic_year", filterYear.trim());
      if (filterStatus) params.set("status", filterStatus);
      if (debouncedSearch.trim()) params.set("q", debouncedSearch.trim());
      const qs = params.toString();
      const url = `/api/admin/users${qs ? `?${qs}` : ""}`;
      const data = await authFetch(url);
      setUsers(data);
      setSelected(new Set());
      const pend = await authFetch("/api/admin/users?status=pending");
      setPending(pend);
    } catch { flash("無法載入使用者列表", "err"); }
    setLoading(false);
  }, [authFetch, filterRole, filterSemester, filterClass, filterYear, filterStatus, debouncedSearch, flash]);

  useEffect(() => { if (token) void fetchUsers(); }, [token, fetchUsers]);

  useEffect(() => {
    const id = window.setTimeout(() => setDebouncedSearch(search), 300);
    return () => window.clearTimeout(id);
  }, [search]);

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
      setCreateForm({ username: "", password: "", display_name: "", email: "", role: "student", semester: "", class_name: "" });
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
    if (editForm.class_name !== (editingUser.class_name || "")) body.class_name = editForm.class_name;
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

  const handleDelete = async (u: User) => {
    if (u.id === me?.id) { flash("無法刪除自己的帳號", "err"); return; }
    if (!confirm(`確定要停用封存 ${u.display_name || u.username}？學習紀錄會保留，帳號將無法再登入。`)) return;
    try {
      await authFetch(`/api/admin/users/${u.id}`, "DELETE");
      flash(`已封存 ${u.display_name || u.username}`);
      fetchUsers();
    } catch (err: unknown) {
      flash((err as Error).message || "刪除失敗", "err");
    }
  };

  const handleApproval = async (u: User, approve: boolean) => {
    const name = u.display_name || u.username;
    if (!approve && !confirm(`確定不通過 ${name} 的帳號申請？`)) return;
    try {
      await authFetch(`/api/admin/users/${u.id}/${approve ? "approve" : "reject"}`, "POST");
      flash(approve ? `已核准 ${name}` : `已退回 ${name} 的申請`);
      fetchUsers();
    } catch (err: unknown) {
      flash((err as Error).message || "操作失敗", "err");
    }
  };

  const selectable = (u: User) => u.id !== me?.id && (u.approval_status ?? "approved") === "approved";
  const visibleSelectable = users.filter(selectable);
  const allSelected = visibleSelectable.length > 0 && visibleSelectable.every((u) => selected.has(u.id));
  const toggleOne = (id: number) => setSelected((s) => {
    const next = new Set(s);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const toggleAll = () => setSelected(allSelected ? new Set() : new Set(visibleSelectable.map((u) => u.id)));

  const handleBulk = async (action: BulkAction) => {
    const ids = [...selected];
    if (ids.length === 0) return;
    const body: Record<string, unknown> = { ids, action };
    if (action === "set_class") {
      const cls = window.prompt(`將 ${ids.length} 個帳號設定到哪個班級？`);
      if (!cls?.trim()) return;
      body.class_name = cls.trim();
    } else {
      const verb = { activate: "啟用", deactivate: "停用", reset_password: "重設密碼" }[action];
      if (!confirm(`確定要${verb}選取的 ${ids.length} 個帳號？`)) return;
    }
    try {
      const data = await authFetch("/api/admin/users/bulk", "POST", body);
      flash(`已更新 ${data.updated} 個帳號`);
      if (action === "reset_password" && data.passwords?.length) setResetPasswords(data.passwords);
      fetchUsers();
    } catch (err: unknown) {
      flash((err as Error).message || "批次操作失敗", "err");
    }
  };

  const handleArchiveSemester = async () => {
    const semester = window.prompt("輸入要封存的學期（例如 114-2）");
    if (!semester) return;
    try {
      const data = await authFetch(`/api/admin/semesters/${encodeURIComponent(semester)}/archive`, "POST");
      flash(`已封存學期 ${semester}，共停用 ${data.archived} 位學生`);
      fetchUsers();
    } catch (err: unknown) {
      flash((err as Error).message || "封存失敗", "err");
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
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-2xl font-bold text-gray-900">帳號管理</h1>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={handleArchiveSemester}
            className="px-4 py-2 text-sm bg-white border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
          >
            學期封存
          </button>
          <button onClick={() => setShowImport(true)}
            className="px-4 py-2 text-sm bg-white border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50">
            批次匯入
          </button>
          <button
            onClick={() => setShowCreate(true)}
            className="px-4 py-2 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 transition-colors"
          >
            + 新增帳號
          </button>
        </div>
      </div>

      {/* Message */}
      {message.text && (
        <div className={`px-4 py-2 rounded-lg text-sm ${message.type === "err" ? "bg-red-50 text-red-700" : "bg-green-50 text-green-700"}`}>
          {message.text}
        </div>
      )}

      {/* Pending sign-up requests */}
      {pending.length > 0 && (
        <section className="border border-amber-200 bg-amber-50 rounded-xl p-4 space-y-2" aria-label="待審核申請">
          <h2 className="text-sm font-semibold text-amber-800">待審核的帳號申請（{pending.length}）</h2>
          <ul className="divide-y divide-amber-100">
            {pending.map((u) => (
              <li key={u.id} className="flex flex-wrap items-center justify-between gap-2 py-2">
                <div className="text-sm text-gray-800">
                  <span className="font-mono">{u.username}</span>
                  <span className="ml-2">{u.display_name}</span>
                  <span className="ml-2 text-xs text-gray-500">{u.class_name || "未填班級"}{u.email ? ` · ${u.email}` : ""}</span>
                  <span className="ml-2 text-xs text-gray-400">{u.created_at?.slice(0, 16)}</span>
                </div>
                <div className="flex gap-1">
                  <button onClick={() => handleApproval(u, true)}
                    className="px-3 py-1 text-xs bg-green-600 text-white rounded-lg hover:bg-green-700">核准</button>
                  <button onClick={() => handleApproval(u, false)}
                    className="px-3 py-1 text-xs text-red-600 border border-red-200 rounded-lg hover:bg-red-50">不通過</button>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
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

      {/* Search + status */}
      <div className="flex flex-wrap items-center gap-2">
        <input
          type="search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="搜尋學號、姓名、Email、班級"
          aria-label="搜尋使用者"
          className="flex-1 min-w-[220px] px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-300"
        />
        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
          aria-label="狀態篩選"
          className="px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white"
        >
          {STATUS_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
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

      <div className="flex flex-wrap gap-3">
        <label className="text-sm">學年<input aria-label="學年篩選" placeholder="例如 115" value={filterYear} onChange={(e) => setFilterYear(e.target.value)} className="block border rounded px-3 py-2" /></label>
        <label className="text-sm">班級<input aria-label="班級篩選" placeholder="全部班級" value={filterClass} onChange={(e) => setFilterClass(e.target.value)} className="block border rounded px-3 py-2" /></label>
      </div>
      {/* Bulk action bar */}
      {selected.size > 0 && (
        <div className="sticky top-0 z-10 flex flex-wrap items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-xl text-sm" role="toolbar" aria-label="批次操作">
          <span className="text-blue-800 font-medium">已選取 {selected.size} 個帳號</span>
          <button onClick={() => handleBulk("activate")} className="px-3 py-1 text-xs bg-white border border-gray-300 rounded-lg hover:bg-gray-50">啟用</button>
          <button onClick={() => handleBulk("deactivate")} className="px-3 py-1 text-xs bg-white border border-gray-300 rounded-lg hover:bg-gray-50">停用</button>
          <button onClick={() => handleBulk("reset_password")} className="px-3 py-1 text-xs bg-white border border-gray-300 rounded-lg hover:bg-gray-50">重設密碼</button>
          <button onClick={() => handleBulk("set_class")} className="px-3 py-1 text-xs bg-white border border-gray-300 rounded-lg hover:bg-gray-50">設定班級</button>
          <button onClick={() => setSelected(new Set())} className="ml-auto px-3 py-1 text-xs text-gray-600 hover:underline">取消選取</button>
        </div>
      )}

      {/* User Table */}
      {loading ? (
        <div className="text-gray-400 text-center py-8">載入中...</div>
      ) : (
        <div className="border border-gray-200 rounded-xl overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-gray-600">
              <tr>
                <th className="px-3 py-3 w-8">
                  <input type="checkbox" checked={allSelected} onChange={toggleAll} aria-label="全選" disabled={visibleSelectable.length === 0} />
                </th>
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
                <tr key={u.id} className={`${!u.is_active ? "opacity-60 bg-gray-50" : "hover:bg-gray-50"}`}>
                  <td className="px-3 py-3">
                    {selectable(u) && (
                      <input type="checkbox" checked={selected.has(u.id)} onChange={() => toggleOne(u.id)} aria-label={`選取 ${u.username}`} />
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono text-gray-900">{u.username}</td>
                  <td className="px-4 py-3 text-gray-700">{u.display_name}</td>
                  <td className="px-4 py-3 text-gray-500">{u.email || "-"}</td>
                  <td className="px-4 py-3">
                    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${ROLE_COLORS[u.role]}`}>
                      {ROLE_LABELS[u.role]}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-500">{formatSemester(u.semester)}<div>{u.class_name || "未分班"}</div></td>
                  <td className="px-4 py-3">
                    <span className={`text-xs ${statusLabel(u).cls}`}>{statusLabel(u).text}</span>
                  </td>
                  <td className="px-4 py-3 text-right space-x-1">
                    <button
                      onClick={() => {
                        setEditingUser(u);
                        setEditForm({ display_name: u.display_name, email: u.email, role: u.role, password: "", semester: u.semester || "", class_name: u.class_name || "" });
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
                        onClick={() => handleDelete(u)}
                        className="px-2 py-1 text-xs text-red-600 hover:bg-red-50 rounded"
                      >
                        停用封存
                      </button>
                    )}
                  </td>
                </tr>
              ))}
              {users.length === 0 && (
                <tr><td colSpan={8} className="text-center py-8 text-gray-400">沒有符合條件的使用者</td></tr>
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
              <label className="block text-sm mt-3">班級
                <input aria-label="班級" value={createForm.class_name} onChange={(e) => setCreateForm({ ...createForm, class_name: e.target.value })} placeholder="例如 護理一甲" className="mt-1 w-full border rounded-lg px-3 py-2" />
              </label>
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
              <label className="block text-sm mt-3">班級
                <input aria-label="班級" value={editForm.class_name} onChange={(e) => setEditForm({ ...editForm, class_name: e.target.value })} placeholder="例如 護理一甲" className="mt-1 w-full border rounded-lg px-3 py-2" />
              </label>
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

      {resetPasswords && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50" role="dialog" aria-modal="true" aria-label="新密碼">
          <div className="bg-white rounded-2xl shadow-xl p-6 w-full max-w-md space-y-4">
            <h2 className="text-lg font-bold text-gray-900">已重設 {resetPasswords.length} 個帳號的密碼</h2>
            <p className="text-sm text-amber-600">新密碼僅顯示這一次，使用者首次登入需更換。請下載保存。</p>
            <div className="max-h-64 overflow-y-auto border border-gray-200 rounded-lg">
              <table className="w-full text-sm">
                <tbody>
                  {resetPasswords.map((r) => (
                    <tr key={r.username} className="border-t border-gray-100 first:border-0">
                      <td className="px-3 py-1.5 font-mono">{r.username}</td>
                      <td className="px-3 py-1.5 font-mono">{r.initial_password}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex justify-end gap-2">
              <button onClick={() => downloadPasswordCsv(resetPasswords, "重設密碼清單.csv")}
                className="px-4 py-2 text-sm border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50">下載 CSV</button>
              <button onClick={() => setResetPasswords(null)} className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700">完成</button>
            </div>
          </div>
        </div>
      )}

      {showImport && (
        <UserImportDialog onDone={fetchUsers} onClose={() => setShowImport(false)} />
      )}
    </div>
  );
}
