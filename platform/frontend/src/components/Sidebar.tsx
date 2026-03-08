import { NavLink } from "react-router-dom";
import { WEEKS } from "../types";
import { useAuth } from "../hooks/useAuth";

export default function Sidebar() {
  const { user, logout } = useAuth();
  return (
    <nav className="w-64 bg-gray-50 border-r border-gray-200 h-screen overflow-y-auto flex flex-col" aria-label="Course navigation">
      <div className="p-4 border-b border-gray-200">
        <NavLink to="/" className="block">
          <h1 className="text-lg font-bold text-gray-900">ML/DL 視覺化</h1>
          <p className="text-xs text-gray-500">互動教學平台</p>
        </NavLink>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        <div className="space-y-0.5">
          {WEEKS.map((w) => (
            <NavLink
              key={w.id}
              to={`/week/${w.id}`}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? "bg-blue-50 text-blue-700 font-medium"
                    : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                }`
              }
            >
              <span className={`inline-flex items-center justify-center w-6 h-6 rounded text-xs font-medium ${
                w.level === "advanced" ? "bg-purple-100 text-purple-700" : "bg-blue-100 text-blue-700"
              }`}>
                {w.id}
              </span>
              <span className="truncate">{w.title}</span>
            </NavLink>
          ))}
        </div>
      </div>

      <div className="p-3 border-t border-gray-200 space-y-0.5">
        <NavLink
          to="/dashboard"
          className={({ isActive }) =>
            `flex items-center gap-2 px-3 py-2 rounded-lg text-sm ${
              isActive ? "bg-green-50 text-green-700 font-medium" : "text-gray-600 hover:bg-gray-100"
            }`
          }
        >
          <span className="text-base">📊</span>
          學習分析
        </NavLink>
        {user?.role === "admin" && (
          <>
            <NavLink
              to="/admin/users"
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-lg text-sm ${
                  isActive ? "bg-purple-50 text-purple-700 font-medium" : "text-gray-400 hover:bg-gray-100 hover:text-gray-600"
                }`
              }
            >
              <span className="text-base">&#128101;</span>
              帳號管理
            </NavLink>
            <NavLink
              to="/admin"
              end
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-lg text-sm ${
                  isActive ? "bg-gray-200 text-gray-900 font-medium" : "text-gray-400 hover:bg-gray-100 hover:text-gray-600"
                }`
              }
            >
              <span className="text-base">&#9881;</span>
              系統管理
            </NavLink>
          </>
        )}
      </div>

      <div className="p-3 border-t border-gray-200 flex items-center justify-between">
        <span className="text-xs text-gray-500 truncate">
          {user?.display_name || user?.username}
        </span>
        <button
          onClick={logout}
          className="text-xs text-gray-400 hover:text-gray-600"
        >
          登出
        </button>
      </div>
    </nav>
  );
}
