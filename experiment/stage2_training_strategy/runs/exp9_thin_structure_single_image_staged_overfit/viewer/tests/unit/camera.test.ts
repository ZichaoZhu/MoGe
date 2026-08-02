import { describe, expect, it } from "vitest";

import { shouldApplyCameraFit } from "@/lib/camera";

describe("camera fitting", () => {
  it("waits until bounds belong to the current scene", () => {
    expect(
      shouldApplyCameraFit({
        boundsSource: "old-scene",
        sceneSource: "new-scene",
        hasFitted: false,
        fitNonce: 0,
        appliedFitNonce: null,
      }),
    ).toBe(false);
  });

  it("fits the first valid scene", () => {
    expect(
      shouldApplyCameraFit({
        boundsSource: "scene",
        sceneSource: "scene",
        hasFitted: false,
        fitNonce: 0,
        appliedFitNonce: null,
      }),
    ).toBe(true);
  });

  it("preserves the camera when only the stage or K asset changes", () => {
    expect(
      shouldApplyCameraFit({
        boundsSource: "new-asset",
        sceneSource: "new-asset",
        hasFitted: true,
        fitNonce: 0,
        appliedFitNonce: 0,
      }),
    ).toBe(false);
  });

  it("fits again after an explicit fit request", () => {
    expect(
      shouldApplyCameraFit({
        boundsSource: "scene",
        sceneSource: "scene",
        hasFitted: true,
        fitNonce: 1,
        appliedFitNonce: 0,
      }),
    ).toBe(true);
  });
});
