export const REFINEMENT_STEPS = [0, 1, 3, 5] as const;

export type RefinementStep = (typeof REFINEMENT_STEPS)[number];
export type StageName = "initial" | "final";
export type CoordinateMode = "aligned" | "raw";
export type RenderMode = "points" | "voxels";
export type ScopeName = "full" | "crop" | "structure";

export type ExperimentCatalogEntry = {
  id: string;
  label: string;
  shortLabel: string;
  summary: string;
  manifestUrl: string;
  stageLabels: Record<StageName, string>;
  stageDetails: Record<StageName, string>;
};

export type ExperimentCatalog = {
  version: 1;
  defaultExperiment: string;
  experiments: ExperimentCatalogEntry[];
};

export type ScopeMetrics = {
  pixels: number;
  point_rel: number;
  depth_rel: number;
  "depth_delta_1.01": number;
  "depth_delta_1.25": number;
  boundary_f1: number;
};

export type PointCloudAsset = {
  url: string;
  pointCount: number;
  sha256: string;
  checkpointSha256: string;
  alignment: { scale: number; zShift: number };
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
  };
  metrics: Record<ScopeName, ScopeMetrics>;
};

export type AssetAlias = { alias: "initial.0" };

export type StageAssets = Record<
  `${RefinementStep}`,
  PointCloudAsset | AssetAlias
>;

export type PointCloudSample = {
  id: string;
  split?: "train" | "val" | "test";
  label: string;
  order: number;
  description: string;
  websiteEnabled: boolean;
  cropXYXY: [number, number, number, number];
  structureMask: {
    type: string;
    quantile?: number;
    minimum_depth_m?: number;
    maximum_depth_m?: number;
    pixels: number;
  };
  rgbUrl: string;
  stages: {
    initial: StageAssets;
    final?: StageAssets;
  };
};

export type PointCloudManifest = {
  version: 1;
  experiment: string;
  coordinateSpace: {
    stored: string;
    aligned: string;
    threeDisplay: string;
  };
  voxelization: {
    depthScale: number;
    spconvOrder: string[];
    depthCoordinate: string;
  };
  resolution: { width: number; height: number };
  steps: RefinementStep[];
  websiteSampleOrder: string[];
  archivedInitialSampleIds: string[];
  samples: PointCloudSample[];
};

export function isAssetAlias(
  value: PointCloudAsset | AssetAlias,
): value is AssetAlias {
  return "alias" in value;
}

export function stageUsesAlias(
  sample: PointCloudSample,
  stage: StageName,
  step: RefinementStep,
): boolean {
  const assets = sample.stages[stage];
  if (!assets) return false;
  return isAssetAlias(assets[String(step) as `${RefinementStep}`]);
}

export function resolveAsset(
  sample: PointCloudSample,
  stage: StageName,
  step: RefinementStep,
): PointCloudAsset {
  const stageAssets = sample.stages[stage];
  if (!stageAssets) {
    throw new Error(`${sample.id} 没有 ${stage} 点云`);
  }
  const selected = stageAssets[String(step) as `${RefinementStep}`];
  if (!selected) {
    throw new Error(`${sample.id} 的 ${stage} 没有 K=${step}`);
  }
  if (!isAssetAlias(selected)) return selected;
  if (selected.alias !== "initial.0") {
    throw new Error(`不支持的点云别名：${selected.alias}`);
  }
  const initial = sample.stages.initial["0"];
  if (isAssetAlias(initial)) {
    throw new Error("初始 K=0 不能是别名");
  }
  return initial;
}

export function websiteSamples(
  manifest: PointCloudManifest,
): PointCloudSample[] {
  const lookup = new Map(manifest.samples.map((sample) => [sample.id, sample]));
  return manifest.websiteSampleOrder.map((id) => {
    const sample = lookup.get(id);
    if (!sample || !sample.websiteEnabled || !sample.stages.final) {
      throw new Error(`网站样本清单无效：${id}`);
    }
    return sample;
  });
}

export function validateManifest(manifest: PointCloudManifest): void {
  if (manifest.version !== 1) throw new Error("不支持的点云清单版本");
  if (
    manifest.resolution.width <= 0 ||
    manifest.resolution.height <= 0 ||
    manifest.samples.length === 0
  ) {
    throw new Error("点云清单缺少有效分辨率或样本");
  }
  const expectedCount =
    manifest.resolution.width * manifest.resolution.height;
  for (const sample of manifest.samples) {
    for (const stage of ["initial", "final"] as const) {
      if (!sample.stages[stage]) continue;
      for (const step of REFINEMENT_STEPS) {
        const asset = resolveAsset(sample, stage, step);
        if (asset.pointCount !== expectedCount) {
          throw new Error(
            `${sample.id} 的 ${stage} K=${step} 点数与图像分辨率不一致`,
          );
        }
        if (!asset.url.startsWith("/data/")) {
          throw new Error(`${sample.id} 的点云 URL 不在 /data/ 下`);
        }
      }
    }
  }
  websiteSamples(manifest);
}

export function validateExperimentCatalog(
  catalog: ExperimentCatalog,
): void {
  if (catalog.version !== 1) throw new Error("不支持的实验目录版本");
  if (!catalog.experiments.length) throw new Error("实验目录为空");
  const ids = catalog.experiments.map((entry) => entry.id);
  if (new Set(ids).size !== ids.length) throw new Error("实验 ID 重复");
  if (!ids.includes(catalog.defaultExperiment)) {
    throw new Error("默认实验不在实验目录中");
  }
  for (const entry of catalog.experiments) {
    if (
      !entry.id ||
      !entry.label ||
      !entry.shortLabel ||
      !entry.summary ||
      !entry.manifestUrl.startsWith("/data/")
    ) {
      throw new Error(`实验目录项无效：${entry.id || "unknown"}`);
    }
  }
}
