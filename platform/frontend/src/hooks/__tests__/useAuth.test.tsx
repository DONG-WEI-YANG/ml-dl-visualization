import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { AuthProvider, useAuth } from "../useAuth";

vi.mock("../../lib/api", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../../lib/api")>();
  return { ...mod, fetchAPI: vi.fn() };
});

import { fetchAPI } from "../../lib/api";

describe("useAuth logout", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it("calls the logout endpoint with the token and clears state", async () => {
    localStorage.setItem("auth_token", "existing-token");
    (fetchAPI as ReturnType<typeof vi.fn>).mockResolvedValue({ status: "ok" });

    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => {
      result.current.logout();
    });

    expect(fetchAPI).toHaveBeenCalledWith(
      "/api/auth/logout",
      {},
      "existing-token"
    );
    expect(result.current.token).toBeNull();
    expect(result.current.user).toBeNull();
    expect(localStorage.getItem("auth_token")).toBeNull();
  });

  it("still clears state even when the logout call rejects", async () => {
    localStorage.setItem("auth_token", "existing-token");
    (fetchAPI as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("network down"));

    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => {
      result.current.logout();
    });

    expect(result.current.token).toBeNull();
    expect(result.current.user).toBeNull();
    expect(localStorage.getItem("auth_token")).toBeNull();
  });

  it("does not call the logout endpoint when there is no token, but still clears state", async () => {
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => {
      result.current.logout();
    });

    expect(fetchAPI).not.toHaveBeenCalled();
    expect(result.current.token).toBeNull();
    expect(result.current.user).toBeNull();
    expect(localStorage.getItem("auth_token")).toBeNull();
  });
});
