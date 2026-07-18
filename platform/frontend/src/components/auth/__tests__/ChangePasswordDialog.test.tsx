import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import ChangePasswordDialog from "../ChangePasswordDialog";

vi.mock("../../../lib/api", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../../../lib/api")>();
  return { ...mod, fetchAPI: vi.fn().mockResolvedValue({ status: "ok" }) };
});
vi.mock("../../../hooks/useAuth", () => ({
  useAuth: () => ({ token: "t", retryVerification: vi.fn() }),
}));

import { fetchAPI } from "../../../lib/api";

describe("ChangePasswordDialog", () => {
  beforeEach(() => vi.clearAllMocks());

  it("forced mode shows notice and no close button", () => {
    render(<ChangePasswordDialog forced onClose={() => {}} />);
    expect(screen.getByText(/首次登入請更換密碼/)).toBeDefined();
    expect(screen.queryByRole("button", { name: /取消/ })).toBeNull();
  });

  it("normal mode has a cancel button", () => {
    render(<ChangePasswordDialog onClose={() => {}} />);
    expect(screen.getByRole("button", { name: /取消/ })).toBeDefined();
  });

  it("rejects short new password client-side", async () => {
    render(<ChangePasswordDialog onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/舊密碼/), { target: { value: "oldpass1" } });
    fireEvent.change(screen.getByLabelText(/^新密碼/), { target: { value: "short" } });
    fireEvent.change(screen.getByLabelText(/確認新密碼/), { target: { value: "short" } });
    fireEvent.click(screen.getByRole("button", { name: /確認變更/ }));
    expect(await screen.findByText(/至少 8 碼/)).toBeDefined();
    expect(fetchAPI).not.toHaveBeenCalled();
  });

  it("submits and calls onClose on success", async () => {
    const onClose = vi.fn();
    render(<ChangePasswordDialog onClose={onClose} />);
    fireEvent.change(screen.getByLabelText(/舊密碼/), { target: { value: "oldpass1" } });
    fireEvent.change(screen.getByLabelText(/^新密碼/), { target: { value: "newpassword9" } });
    fireEvent.change(screen.getByLabelText(/確認新密碼/), { target: { value: "newpassword9" } });
    fireEvent.click(screen.getByRole("button", { name: /確認變更/ }));
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    expect(fetchAPI).toHaveBeenCalledWith(
      "/api/auth/change-password",
      { old_password: "oldpass1", new_password: "newpassword9" },
      "t"
    );
  });
});
