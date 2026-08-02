import { describe, expect, it } from "vitest";

import {
  validateExperimentCatalog,
  type ExperimentCatalog,
} from "@/lib/manifest";

const catalog: ExperimentCatalog = {
  version: 1,
  defaultExperiment: "exp12",
  experiments: [
    {
      id: "exp9",
      label: "Exp9",
      shortLabel: "Exp9 单图",
      summary: "单图训练",
      manifestUrl: "/data/manifest.json",
      stageLabels: { initial: "训练前", final: "训练后" },
      stageDetails: { initial: "初始化", final: "最终" },
    },
    {
      id: "exp12",
      label: "Exp12",
      shortLabel: "Exp12 百图",
      summary: "百图训练",
      manifestUrl: "/data/exp12/manifest.json",
      stageLabels: { initial: "联合前", final: "联合后" },
      stageDetails: { initial: "step 3000", final: "step 3300" },
    },
  ],
};

describe("experiment catalog", () => {
  it("accepts a valid multi-experiment directory", () => {
    expect(() => validateExperimentCatalog(catalog)).not.toThrow();
  });

  it("requires the default experiment to exist", () => {
    expect(() =>
      validateExperimentCatalog({
        ...catalog,
        defaultExperiment: "missing",
      }),
    ).toThrow(/默认实验/);
  });

  it("rejects duplicate experiment ids", () => {
    expect(() =>
      validateExperimentCatalog({
        ...catalog,
        experiments: [catalog.experiments[0], catalog.experiments[0]],
      }),
    ).toThrow(/重复/);
  });
});
