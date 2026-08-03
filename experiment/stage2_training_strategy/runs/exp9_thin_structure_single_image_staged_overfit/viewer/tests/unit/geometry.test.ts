import { describe, expect, it } from "vitest";

import {
  finiteRasterIndices,
  pointPositions,
  rasterIndices,
  roundHalfToEven,
  transformPoint,
  voxelInstances,
} from "@/lib/geometry";
import {
  resolveAsset,
  websiteSamples,
  type PointCloudAsset,
  type PointCloudManifest,
  type PointCloudSample,
} from "@/lib/manifest";

const asset: PointCloudAsset = {
  url: "/data/sample.ply",
  pointCount: 4,
  sha256: "a",
  checkpointSha256: "b",
  alignment: { scale: 2, zShift: -0.5 },
  bounds: { min: [0, 0, 1], max: [1, 1, 2] },
  metrics: {
    full: {
      pixels: 4,
      point_rel: 0,
      depth_rel: 0,
      "depth_delta_1.01": 1,
      "depth_delta_1.25": 1,
      boundary_f1: 1,
    },
    crop: {
      pixels: 4,
      point_rel: 0,
      depth_rel: 0,
      "depth_delta_1.01": 1,
      "depth_delta_1.25": 1,
      boundary_f1: 1,
    },
    structure: {
      pixels: 2,
      point_rel: 0,
      depth_rel: 0,
      "depth_delta_1.01": 1,
      "depth_delta_1.25": 1,
      boundary_f1: 1,
    },
  },
};

const sample: PointCloudSample = {
  id: "sample",
  label: "样本 05",
  order: 5,
  description: "细杆",
  websiteEnabled: true,
  cropXYXY: [0, 0, 2, 2],
  structureMask: { type: "near_quantile", quantile: 0.5, pixels: 2 },
  rgbUrl: "/rgb.jpg",
  stages: {
    initial: {
      "0": asset,
      "1": { alias: "initial.0" },
      "3": { alias: "initial.0" },
      "5": { alias: "initial.0" },
    },
    final: { "0": asset, "1": asset, "3": asset, "5": asset },
  },
};

const manifest: PointCloudManifest = {
  version: 1,
  experiment: "exp9",
  coordinateSpace: {
    stored: "raw",
    aligned: "aligned",
    threeDisplay: "[x,-y,-z]",
  },
  voxelization: {
    depthScale: 200,
    spconvOrder: ["batch", "depth", "row", "column"],
    depthCoordinate: "round(D log Z)",
  },
  resolution: { width: 2, height: 2 },
  steps: [0, 1, 3, 5],
  websiteSampleOrder: ["sample"],
  archivedInitialSampleIds: ["sample"],
  samples: [sample],
};

describe("Exp9 geometry transforms", () => {
  it("uses full-image and crop raster order", () => {
    expect(Array.from(rasterIndices(3, 2, [1, 0, 3, 2], "full"))).toEqual([
      0, 1, 2, 3, 4, 5,
    ]);
    expect(Array.from(rasterIndices(3, 2, [1, 0, 3, 2], "crop"))).toEqual([
      1, 2, 4, 5,
    ]);
  });

  it("applies scale, optical-axis shift, and display-axis conversion", () => {
    expect(transformPoint(1, 2, 3, asset, "aligned")).toEqual([2, -4, -5.5]);
    expect(transformPoint(1, 2, 3, asset, "raw")).toEqual([1, -2, -3]);
    expect(
      Array.from(
        pointPositions(
          new Float32Array([1, 2, 3]),
          new Uint32Array([0]),
          asset,
          "aligned",
        ),
      ),
    ).toEqual([2, -4, -5.5]);
  });

  it("skips invalid ground-truth pixels without changing raster order", () => {
    const raw = new Float32Array([
      0, 0, 1,
      Number.NaN, Number.NaN, Number.NaN,
      1, 0, 2,
      0, 0, 0,
    ]);
    expect(
      Array.from(finiteRasterIndices(raw, new Uint32Array([0, 1, 2, 3]))),
    ).toEqual([0, 2]);
  });

  it("resolves initial K aliases to the exact K0 asset", () => {
    expect(resolveAsset(sample, "initial", 5)).toBe(asset);
    expect(websiteSamples(manifest)).toEqual([sample]);
  });

  it("constructs SSR voxel bins from round(200 log Z)", () => {
    const positions = new Float32Array([
      0, 0, 1,
      0, 0, Math.exp(0.5),
      0, 0, Math.exp(-0.5),
      0, 0, 1,
    ]);
    const colors = new Float32Array(12).fill(0.5);
    const voxels = voxelInstances(positions, colors, manifest, sample);
    expect(voxels.map((voxel) => voxel.position[2])).toEqual([0, -100, 100, 0]);
    expect(voxels[0].position.slice(0, 2)).toEqual([-0.5, 0.5]);
  });

  it("matches torch and numpy round-to-even behavior at half bins", () => {
    expect(
      [0.5, 1.5, 2.5, -0.5, -1.5, -2.5].map(roundHalfToEven),
    ).toEqual([0, 2, 2, 0, -2, -2]);
  });
});
