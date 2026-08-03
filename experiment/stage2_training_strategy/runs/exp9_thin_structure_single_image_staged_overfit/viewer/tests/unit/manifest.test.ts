import { describe, expect, it } from "vitest";

import {
  defaultManifestSplit,
  defaultStage,
  manifestSplits,
  sampleStages,
  validateExperimentCatalog,
  validateManifest,
  websiteSamples,
  type ExperimentCatalog,
  type PointCloudAsset,
  type PointCloudManifest,
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

const asset: PointCloudAsset = {
  url: "/data/exp20/sample/final_k0.ply",
  pointCount: 4,
  sha256: "asset",
  checkpointSha256: "checkpoint",
  alignment: { scale: 1, zShift: 0 },
  bounds: { min: [0, 0, 1], max: [1, 1, 2] },
  metrics: {
    full: {
      pixels: 4,
      point_rel: 0.1,
      depth_rel: 0.1,
      "depth_delta_1.01": 0.2,
      "depth_delta_1.25": 1,
      boundary_f1: 0.8,
    },
    crop: {
      pixels: 4,
      point_rel: 0.1,
      depth_rel: 0.1,
      "depth_delta_1.01": 0.2,
      "depth_delta_1.25": 1,
      boundary_f1: 0.8,
    },
    structure: {
      pixels: 2,
      point_rel: 0.1,
      depth_rel: 0.1,
      "depth_delta_1.01": 0.2,
      "depth_delta_1.25": 1,
      boundary_f1: 0.8,
    },
  },
};

function v2Manifest(): PointCloudManifest {
  const samples = (["train", "val", "test"] as const).flatMap((split) =>
    Array.from({ length: 5 }, (_, offset) => {
      const id = `${split}-${offset + 1}`;
      return {
        id,
        split,
        label: `图片 ${offset + 1}`,
        order: offset + 1,
        description: id,
        websiteEnabled: true,
        cropXYXY: [0, 0, 2, 2] as [number, number, number, number],
        structureMask: { type: "near_quantile", quantile: 0.5, pixels: 2 },
        rgbUrl: `/data/exp20/${id}/source_rgb.jpg`,
        stages: {
          final: {
            "0": asset,
            "1": asset,
            "3": asset,
            "5": asset,
          },
        },
      };
    }),
  );
  return {
    version: 2,
    experiment: "exp20",
    coordinateSpace: { stored: "raw", aligned: "aligned", threeDisplay: "xyz" },
    voxelization: {
      depthScale: 200,
      spconvOrder: ["batch", "depth", "row", "column"],
      depthCoordinate: "round(200 log Z)",
    },
    resolution: { width: 2, height: 2 },
    steps: [0, 1, 3, 5],
    availableSplits: ["train", "val", "test"],
    defaultSplit: "train",
    websiteSampleOrderBySplit: {
      train: samples.filter((item) => item.split === "train").map((item) => item.id),
      val: samples.filter((item) => item.split === "val").map((item) => item.id),
      test: samples.filter((item) => item.split === "test").map((item) => item.id),
    },
    availableStages: ["final"],
    defaultStages: { left: "final", right: "final" },
    samples,
  };
}

describe("split-aware v2 manifest", () => {
  it("validates five samples per split and resolves the final-only stage", () => {
    const manifest = v2Manifest();
    expect(() => validateManifest(manifest)).not.toThrow();
    expect(manifestSplits(manifest)).toEqual(["train", "val", "test"]);
    expect(defaultManifestSplit(manifest)).toBe("train");
    expect(websiteSamples(manifest, "val")).toHaveLength(5);
    expect(defaultStage(manifest, websiteSamples(manifest, "train")[0], "left")).toBe(
      "final",
    );
  });

  it("rejects an incomplete split", () => {
    const manifest = v2Manifest();
    manifest.websiteSampleOrderBySplit!.test =
      manifest.websiteSampleOrderBySplit!.test!.slice(0, 4);
    expect(() => validateManifest(manifest)).toThrow(/五个/);
  });

  it("validates an optional ground-truth point cloud", () => {
    const manifest = v2Manifest();
    manifest.samples[0].groundTruth = {
      ...asset,
      url: "/data/exp20/train-1/ground_truth.ply",
      validPointCount: 3,
    };
    expect(() => validateManifest(manifest)).not.toThrow();
    manifest.samples[0].groundTruth.pointCount = 3;
    expect(() => validateManifest(manifest)).toThrow(/真实点云点数/);
  });

  it("defaults to training-before versus training-after when both stages exist", () => {
    const manifest = v2Manifest();
    for (const sample of manifest.samples) {
      sample.stages.initial = {
        "0": asset,
        "1": { alias: "initial.0" },
        "3": { alias: "initial.0" },
        "5": { alias: "initial.0" },
      };
    }
    manifest.availableStages = ["initial", "final"];
    manifest.defaultStages = { left: "initial", right: "final" };
    const sample = websiteSamples(manifest, "train")[0];
    expect(() => validateManifest(manifest)).not.toThrow();
    expect(defaultStage(manifest, sample, "left")).toBe("initial");
    expect(defaultStage(manifest, sample, "right")).toBe("final");
  });

  it("supports Exp30's initial, stage1 and final checkpoints", () => {
    const manifest = v2Manifest();
    for (const sample of manifest.samples) {
      sample.stages.initial = {
        "0": asset,
        "1": { alias: "initial.0" },
        "3": { alias: "initial.0" },
        "5": { alias: "initial.0" },
      };
      sample.stages.stage1 = {
        "0": asset,
        "1": asset,
        "3": asset,
        "5": asset,
      };
    }
    manifest.availableStages = ["initial", "stage1", "final"];
    manifest.defaultStages = { left: "initial", right: "final" };
    const sample = websiteSamples(manifest, "train")[0];
    expect(() => validateManifest(manifest)).not.toThrow();
    expect(sampleStages(sample)).toEqual(["initial", "stage1", "final"]);
  });
});
