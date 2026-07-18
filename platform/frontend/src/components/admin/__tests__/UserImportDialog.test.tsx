import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import UserImportDialog from "../UserImportDialog";

vi.mock("../../../lib/api", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../../../lib/api")>();
  return {
    ...mod,
    fetchAPI: vi.fn().mockResolvedValue({
      created: [{ username: "s001", initial_password: "abcDEF123456" }],
      skipped: [{ username: "dup", reason: "帳號已存在" }],
    }),
  };
});
vi.mock("../../../hooks/useAuth", () => ({
  useAuth: () => ({ token: "t" }),
}));

import { fetchAPI } from "../../../lib/api";

describe("UserImportDialog", () => {
  beforeEach(() => vi.clearAllMocks());

  it("parses pasted CSV into a preview table", () => {
    render(<UserImportDialog onDone={() => {}} onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/名單內容/), {
      target: { value: "s001,王小明,a@x.tw\ndup,李四," },
    });
    expect(screen.getByText("王小明")).toBeDefined();
    expect(screen.getByText(/2 筆/)).toBeDefined();
  });

  it("submits rows and shows created passwords with skipped reasons", async () => {
    render(<UserImportDialog onDone={() => {}} onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/名單內容/), {
      target: { value: "s001,王小明\ndup,李四" },
    });
    fireEvent.click(screen.getByRole("button", { name: /開始匯入/ }));
    await waitFor(() => {
      expect(screen.getByText("abcDEF123456")).toBeDefined();
      expect(screen.getByText(/帳號已存在/)).toBeDefined();
    });
    expect(fetchAPI).toHaveBeenCalledWith(
      "/api/admin/users/import",
      expect.objectContaining({
        rows: [
          { username: "s001", display_name: "王小明", email: "" },
          { username: "dup", display_name: "李四", email: "" },
        ],
      }),
      "t"
    );
  });
});
