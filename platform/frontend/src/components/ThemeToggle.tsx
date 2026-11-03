import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme, type ThemePreference } from "../hooks/useTheme";

const META: Record<ThemePreference, { label: string; Icon: typeof Sun }> = {
  light: { label: "淺色", Icon: Sun },
  dark: { label: "深色", Icon: Moon },
  system: { label: "跟隨系統", Icon: Monitor },
};

/** Cycles light → dark → system. `compact` shows only the icon. */
export default function ThemeToggle({ compact = false }: { compact?: boolean }) {
  const { preference, cycle } = useTheme();
  const { label, Icon } = META[preference];
  return (
    <button
      type="button"
      onClick={cycle}
      title={`主題：${label}（點擊切換）`}
      aria-label={`主題：${label}，點擊切換`}
      className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs text-gray-500 hover:bg-gray-100 hover:text-gray-800"
    >
      <Icon size={14} aria-hidden="true" />
      {!compact && <span>{label}</span>}
    </button>
  );
}
