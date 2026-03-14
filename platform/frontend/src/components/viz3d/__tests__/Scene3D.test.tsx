import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@react-three/fiber", () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="r3f-canvas">{children}</div>
  ),
}));
vi.mock("@react-three/drei", () => ({
  OrbitControls: () => null,
}));

import Scene3D from "../Scene3D";

describe("Scene3D", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders canvas when WebGL is available", () => {
    // jsdom doesn't support getContext, so we mock createElement to return
    // a canvas whose getContext returns a truthy value
    const origCreate = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag: string, options?: any) => {
      if (tag === "canvas") {
        const el = origCreate(tag);
        el.getContext = () => ({}) as any;
        return el;
      }
      return origCreate(tag, options);
    });
    render(<Scene3D><mesh /></Scene3D>);
    expect(screen.getByTestId("r3f-canvas")).toBeDefined();
  });

  it("renders fallback when WebGL check fails", () => {
    const origCreate = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag: string, options?: any) => {
      if (tag === "canvas") {
        const el = origCreate(tag);
        el.getContext = () => null;
        return el;
      }
      return origCreate(tag, options);
    });
    render(
      <Scene3D fallback={<div data-testid="fallback">No WebGL</div>}><mesh /></Scene3D>
    );
    expect(screen.getByTestId("fallback")).toBeDefined();
  });

  it("has aria-label for accessibility", () => {
    const origCreate = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag: string, options?: any) => {
      if (tag === "canvas") {
        const el = origCreate(tag);
        el.getContext = () => ({}) as any;
        return el;
      }
      return origCreate(tag, options);
    });
    render(<Scene3D ariaLabel="test label"><mesh /></Scene3D>);
    expect(screen.getByRole("img")).toBeDefined();
  });
});
