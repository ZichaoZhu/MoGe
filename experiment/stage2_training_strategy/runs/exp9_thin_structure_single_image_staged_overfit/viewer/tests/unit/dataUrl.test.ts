import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("point-cloud data URL", () => {
  it("keeps relative data paths when no asset origin is configured", async () => {
    vi.stubEnv("NEXT_PUBLIC_POINT_CLOUD_ASSET_ORIGIN", "");
    const { dataUrl } = await import("@/lib/dataUrl");
    expect(dataUrl("/data/exp20/manifest.json")).toBe(
      "/data/exp20/manifest.json",
    );
  });

  it("prefixes relative paths for a separate static asset origin", async () => {
    vi.stubEnv(
      "NEXT_PUBLIC_POINT_CLOUD_ASSET_ORIGIN",
      "https://moge3-exp9-viewer.vercel.app/",
    );
    const { dataUrl } = await import("@/lib/dataUrl");
    expect(dataUrl("/data/exp20/manifest.json")).toBe(
      "https://moge3-exp9-viewer.vercel.app/data/exp20/manifest.json",
    );
    expect(dataUrl("https://example.com/cloud.ply")).toBe(
      "https://example.com/cloud.ply",
    );
  });
});
