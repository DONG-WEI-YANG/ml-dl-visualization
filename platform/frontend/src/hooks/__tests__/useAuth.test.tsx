import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { AuthProvider, useAuth } from "../useAuth";

vi.mock("../../lib/api", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../../lib/api")>();
  return { ...mod, fetchAPI: vi.fn() };
});

import { APIError, fetchAPI } from "../../lib/api";

describe("useAuth logout", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it('replaces the revoked token and user after password change', async () => {
    localStorage.setItem('auth_token', 'old-token');
    const user = { id: 1, username: 'student', display_name: 'Student', role: 'student' as const,
      semester: '115-1', must_change_password: false };
    vi.mocked(fetchAPI).mockResolvedValue(user);
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await waitFor(() => expect(result.current.loading).toBe(false));
    act(() => result.current.acceptSession({ access_token: 'replacement-token', user }));
    await waitFor(() => expect(result.current.verification).toBe('authenticated'));
    expect(result.current.token).toBe('replacement-token');
    expect(localStorage.getItem('auth_token')).toBe('replacement-token');
    expect(result.current.user?.must_change_password).toBe(false);
    expect(fetchAPI).toHaveBeenLastCalledWith('/api/auth/me', undefined, 'replacement-token', { timeoutMs: 8000 });
  });

  it('ignores an old verification rejection after accepting a new session', async () => {
    localStorage.setItem('auth_token', 'old-token');
    let rejectOld!: (reason: Error) => void;
    const oldVerification = new Promise((_resolve, reject) => { rejectOld = reject; });
    const user = { id: 2, username: 'new-user', display_name: 'New', role: 'student' as const, semester: '115-1' };
    vi.mocked(fetchAPI).mockImplementation((_path, _body, token) =>
      token === 'old-token' ? oldVerification : Promise.resolve(user));
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await waitFor(() => expect(fetchAPI).toHaveBeenCalled());
    act(() => result.current.acceptSession({ access_token: 'new-token', user }));
    await waitFor(() => expect(result.current.verification).toBe('authenticated'));
    await act(async () => { rejectOld(new APIError('unauthorized', 'expired', 401)); });
    expect(result.current.token).toBe('new-token');
    expect(localStorage.getItem('auth_token')).toBe('new-token');
    expect(result.current.user?.username).toBe('new-user');
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
