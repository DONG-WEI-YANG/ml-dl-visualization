import "@testing-library/jest-dom";
import { vi } from "vitest";

// jsdom does not implement scrollIntoView
Element.prototype.scrollIntoView = () => {};

// Visualization components use a canvas in production; jsdom has no renderer.
HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as typeof HTMLCanvasElement.prototype.getContext;
