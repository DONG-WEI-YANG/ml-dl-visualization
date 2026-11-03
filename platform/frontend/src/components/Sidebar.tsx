import { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  BarChart3, ClipboardList, KeyRound, LogOut, PanelLeftClose, PanelLeftOpen,
  ScrollText, Settings, Users, type LucideIcon,
} from "lucide-react";
import { WEEKS } from "../types";
import { useAuth } from "../hooks/useAuth";
import ChangePasswordDialog from "./auth/ChangePasswordDialog";
import ThemeToggle from "./ThemeToggle";

export const SIDEBAR_STORAGE_KEY = "sidebar-collapsed";

function readCollapsed(): boolean {
  try {
    const v = localStorage.getItem(SIDEBAR_STORAGE_KEY);
    if (v !== null) return v === "1";
  } catch {
    /* storage blocked */
  }
  // No saved choice: start collapsed on narrow screens so content gets the room.
  return typeof window !== "undefined" && window.innerWidth < 768;
}

interface NavItem {
  to: string;
  label: string;
  Icon: LucideIcon;
  active: string;
  end?: boolean;
}

const ADMIN_ITEMS: NavItem[] = [
  { to: "/admin/users", label: "帳號管理", Icon: Users, active: "bg-purple-50 text-purple-700" },
  { to: "/admin/audit", label: "稽核紀錄", Icon: ScrollText, active: "bg-indigo-50 text-indigo-700" },
  { to: "/admin/quiz", label: "題庫管理", Icon: ClipboardList, active: "bg-orange-50 text-orange-700" },
  { to: "/admin", label: "系統管理", Icon: Settings, active: "bg-gray-200 text-gray-900", end: true },
];

export default function Sidebar() {
  const { user, logout } = useAuth();
  const [showChangePassword, setShowChangePassword] = useState(false);
  const [collapsed, setCollapsed] = useState(readCollapsed);

  const toggle = () => {
    setCollapsed((c) => {
      try {
        localStorage.setItem(SIDEBAR_STORAGE_KEY, c ? "0" : "1");
      } catch {
        /* ignore */
      }
      return !c;
    });
  };

  const navItems: NavItem[] = [
    { to: "/dashboard", label: "學習分析", Icon: BarChart3, active: "bg-green-50 text-green-700" },
    ...(user?.role === "admin" ? ADMIN_ITEMS : []),
  ];

  return (
    <nav
      className={`${collapsed ? "w-16" : "w-64"} shrink-0 bg-gray-50 border-r border-gray-200 h-screen flex flex-col transition-[width] duration-200`}
      aria-label="Course navigation"
      data-collapsed={collapsed}
    >
      <div className={`border-b border-gray-200 flex items-center ${collapsed ? "justify-center p-3" : "justify-between p-4"}`}>
        {!collapsed && (
          <NavLink to="/" className="block min-w-0">
            <h1 className="text-lg font-bold text-gray-900">ML/DL 視覺化</h1>
            <p className="text-xs text-gray-500">互動教學平台</p>
          </NavLink>
        )}
        <button
          type="button"
          onClick={toggle}
          aria-label={collapsed ? "展開側邊欄" : "收合側邊欄"}
          aria-expanded={!collapsed}
          title={collapsed ? "展開側邊欄" : "收合側邊欄"}
          className="p-1.5 rounded-md text-gray-500 hover:bg-gray-200 hover:text-gray-800"
        >
          {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        <div className="space-y-0.5">
          {WEEKS.map((w) => (
            <NavLink
              key={w.id}
              to={`/week/${w.id}`}
              title={collapsed ? `第 ${w.id} 週：${w.title}` : undefined}
              aria-label={collapsed ? `第 ${w.id} 週：${w.title}` : undefined}
              className={({ isActive }) =>
                `flex items-center gap-2 rounded-lg text-sm transition-colors ${collapsed ? "justify-center py-1.5" : "px-3 py-2"} ${
                  isActive
                    ? "bg-blue-50 text-blue-700 font-medium"
                    : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                }`
              }
            >
              <span className={`inline-flex shrink-0 items-center justify-center w-6 h-6 rounded text-xs font-medium ${
                w.level === "advanced" ? "bg-purple-100 text-purple-700" : "bg-blue-100 text-blue-700"
              }`}>
                {w.id}
              </span>
              {!collapsed && <span className="truncate">{w.title}</span>}
            </NavLink>
          ))}
        </div>
      </div>

      <div className="p-2 border-t border-gray-200 space-y-0.5">
        {navItems.map(({ to, label, Icon, active, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            title={collapsed ? label : undefined}
            aria-label={collapsed ? label : undefined}
            className={({ isActive }) =>
              `flex items-center gap-2 rounded-lg text-sm ${collapsed ? "justify-center py-2" : "px-3 py-2"} ${
                isActive ? `${active} font-medium` : "text-gray-500 hover:bg-gray-100 hover:text-gray-800"
              }`
            }
          >
            <Icon size={16} aria-hidden="true" className="shrink-0" />
            {!collapsed && label}
          </NavLink>
        ))}
      </div>

      <div className={`p-3 border-t border-gray-200 ${collapsed ? "flex flex-col items-center gap-2" : "space-y-2"}`}>
        {!collapsed && (
          <div className="flex flex-col min-w-0">
            <span className="text-xs text-gray-500 truncate">{user?.display_name || user?.username}</span>
            {user?.semester && (
              <span className="text-[10px] text-gray-400 truncate">
                {user.semester.replace("-1", " 上學期").replace("-2", " 下學期")}
              </span>
            )}
          </div>
        )}
        <div className={`flex items-center ${collapsed ? "flex-col gap-1" : "justify-between"}`}>
          <ThemeToggle compact={collapsed} />
          <div className={`flex items-center ${collapsed ? "flex-col gap-1" : "gap-1"}`}>
            <button
              onClick={() => setShowChangePassword(true)}
              title="變更密碼"
              aria-label="變更密碼"
              className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs text-gray-500 hover:bg-gray-100 hover:text-gray-800"
            >
              <KeyRound size={14} aria-hidden="true" />
              {!collapsed && "變更密碼"}
            </button>
            <button
              onClick={logout}
              title="登出"
              aria-label="登出"
              className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs text-gray-500 hover:bg-gray-100 hover:text-gray-800"
            >
              <LogOut size={14} aria-hidden="true" />
              {!collapsed && "登出"}
            </button>
          </div>
        </div>
      </div>
      {showChangePassword && (
        <ChangePasswordDialog onClose={() => setShowChangePassword(false)} />
      )}
    </nav>
  );
}
