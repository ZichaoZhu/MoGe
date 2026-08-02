import { describe, expect, it } from "vitest";

import { parseViewerUrl, serializeViewerUrl } from "@/lib/urlState";

describe("viewer URL state", () => {
  it("round-trips the four-level Exp20 browser state", () => {
    const query = serializeViewerUrl({
      experiment: "exp20",
      split: "train",
      sample: 3,
      leftK: 1,
      rightK: 5,
    });
    expect(parseViewerUrl(query)).toEqual({
      experiment: "exp20",
      split: "train",
      sample: 3,
      leftK: 1,
      rightK: 5,
    });
  });

  it("drops invalid split, picture and K values", () => {
    expect(
      parseViewerUrl("?split=other&sample=8&leftK=2&rightK=3"),
    ).toEqual({
      experiment: undefined,
      split: undefined,
      sample: undefined,
      leftK: undefined,
      rightK: 3,
    });
  });
});
