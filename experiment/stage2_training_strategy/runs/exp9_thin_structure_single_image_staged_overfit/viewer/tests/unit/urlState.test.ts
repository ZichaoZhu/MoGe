import { describe, expect, it } from "vitest";

import { parseViewerUrl, serializeViewerUrl } from "@/lib/urlState";

describe("viewer URL state", () => {
  it("round-trips the four-level Exp20 browser state", () => {
    const query = serializeViewerUrl({
      experiment: "exp20",
      split: "train",
      sample: 3,
      leftStage: "initial",
      rightStage: "final",
      leftK: 1,
      rightK: 5,
    });
    expect(parseViewerUrl(query)).toEqual({
      experiment: "exp20",
      split: "train",
      sample: 3,
      leftStage: "initial",
      rightStage: "final",
      leftK: 1,
      rightK: 5,
    });
  });

  it("drops invalid split, picture and K values", () => {
    expect(
      parseViewerUrl(
        "?split=other&sample=0&leftStage=middle&rightStage=final&leftK=2&rightK=3",
      ),
    ).toEqual({
      experiment: undefined,
      split: undefined,
      sample: undefined,
      leftStage: undefined,
      rightStage: "final",
      leftK: undefined,
      rightK: 3,
    });
  });

  it("round-trips the Exp30 stage1 checkpoint", () => {
    const query = serializeViewerUrl({
      experiment: "exp30",
      split: "train",
      sample: 8,
      leftStage: "stage1",
      rightStage: "final",
      leftK: 3,
      rightK: 5,
    });
    expect(parseViewerUrl(query)).toMatchObject({
      sample: 8,
      leftStage: "stage1",
      rightStage: "final",
    });
  });
});
