import { useCallback, useEffect, useState } from "react";

export type ThemePreference = "light" | "dark" | "system";

export const THEME_STORAGE_KEY = "theme-preference";
const ORDER: ThemePreference[] = ["light", "dark", "system"];

function readPreference(): ThemePreference {
  try {
    const v = localStorage.getItem(THEME_STORAGE_KEY);
    if (v === "light" || v === "dark" || v === "system") return v;
  } catch {
    /* storage blocked (private mode) */
  }
  return "system";
}

function systemPrefersDark(): boolean {
  return typeof window !== "undefined" && !!window.matchMedia?.("(prefers-color-scheme: dark)").matches;
}

export function applyTheme(pref: ThemePreference): void {
  const dark = pref === "dark" || (pref === "system" && systemPrefersDark());
  document.documentElement.classList.toggle("dark", dark);
}

/** Light / dark / follow-system preference, persisted per browser. */
export function useTheme() {
  const [preference, setPreferenceState] = useState<ThemePreference>(readPreference);

  useEffect(() => {
    applyTheme(preference);
    if (preference !== "system" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => applyTheme("system");
    mq.addEventListener?.("change", onChange);
    return () => mq.removeEventListener?.("change", onChange);
  }, [preference]);

  const setPreference = useCallback((pref: ThemePreference) => {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, pref);
    } catch {
      /* ignore */
    }
    setPreferenceState(pref);
  }, []);

  const cycle = useCallback(() => {
    setPreference(ORDER[(ORDER.indexOf(preference) + 1) % ORDER.length]);
  }, [preference, setPreference]);

  return { preference, setPreference, cycle };
}
