export const REFINEMENT_STEPS = [0, 1, 3, 5] as const;

export type RefinementStep = (typeof REFINEMENT_STEPS)[number];
export type StageName = "initial" | "final";
export type DatasetSplit = "train" | "val" | "test";
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
  pointRelReductionFromK0?: number;
};

export type AssetAlias = { alias: "initial.0" };

export type StageAssets = Record<
  `${RefinementStep}`,
  PointCloudAsset | AssetAlias
>;

export type PointCloudSample = {
  id: string;
  split?: DatasetSplit;
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
  selection?: {
    metric: string;
    policy: string;
    rank: number;
    total: number;
    relativeImprovement: number;
    outcome?: "improved" | "degraded";
  };
  stages: Partial<Record<StageName, StageAssets>>;
};

export type PointCloudManifest = {
  version: 1 | 2;
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
  websiteSampleOrder?: string[];
  archivedInitialSampleIds?: string[];
  availableSplits?: DatasetSplit[];
  defaultSplit?: DatasetSplit;
  websiteSampleOrderBySplit?: Partial<Record<DatasetSplit, string[]>>;
  availableStages?: StageName[];
  defaultStages?: { left: StageName; right: StageName };
  provenance?: {
    sourceExperiment: string;
    initialization?: string;
    initializationSeed?: number;
    initializationStateSha256?: string;
    checkpointStep: number;
    checkpointSha256: string;
    inferencePolicy: string;
    selectionMetric: string;
    selectionPolicy?: string;
    smoothLogDepthResidualBound?: number;
    note?: string;
  };
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

export function sampleStages(sample: PointCloudSample): StageName[] {
  return (["initial", "final"] as const).filter(
    (stage) => sample.stages[stage] !== undefined,
  );
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
  const initialStage = sample.stages.initial;
  if (!initialStage) {
    throw new Error(`${sample.id} 的别名缺少 initial 阶段`);
  }
  const initial = initialStage["0"];
  if (isAssetAlias(initial)) {
    throw new Error("初始 K=0 不能是别名");
  }
  return initial;
}

export function websiteSamples(
  manifest: PointCloudManifest,
  split?: DatasetSplit | null,
): PointCloudSample[] {
  const lookup = new Map(manifest.samples.map((sample) => [sample.id, sample]));
  let order: string[];
  if (manifest.version === 2) {
    const activeSplit = split ?? manifest.defaultSplit;
    if (!activeSplit) throw new Error("v2 清单缺少默认数据划分");
    order = manifest.websiteSampleOrderBySplit?.[activeSplit] ?? [];
  } else {
    order = manifest.websiteSampleOrder ?? [];
    if (split) {
      order = order.filter((id) => lookup.get(id)?.split === split);
    }
  }
  return order.map((id) => {
    const sample = lookup.get(id);
    if (!sample || !sample.websiteEnabled || sampleStages(sample).length === 0) {
      throw new Error(`网站样本清单无效：${id}`);
    }
    return sample;
  });
}

export function manifestSplits(
  manifest: PointCloudManifest,
): DatasetSplit[] {
  if (manifest.version === 2) return manifest.availableSplits ?? [];
  const present = new Set(
    manifest.samples
      .filter((sample) => sample.websiteEnabled)
      .map((sample) => sample.split)
      .filter((split): split is DatasetSplit => split !== undefined),
  );
  return (["train", "val", "test"] as const).filter((split) =>
    present.has(split),
  );
}

export function defaultManifestSplit(
  manifest: PointCloudManifest,
): DatasetSplit | null {
  const splits = manifestSplits(manifest);
  if (!splits.length) return null;
  return manifest.defaultSplit && splits.includes(manifest.defaultSplit)
    ? manifest.defaultSplit
    : splits[0];
}

export function defaultStage(
  manifest: PointCloudManifest,
  sample: PointCloudSample,
  pane: "left" | "right",
): StageName {
  const stages = sampleStages(sample);
  if (!stages.length) throw new Error(`${sample.id} 没有可用阶段`);
  const requested = manifest.defaultStages?.[pane];
  if (requested && stages.includes(requested)) return requested;
  const legacy = pane === "left" ? "initial" : "final";
  return stages.includes(legacy) ? legacy : stages[0];
}

export function validateManifest(manifest: PointCloudManifest): void {
  if (manifest.version !== 1 && manifest.version !== 2) {
    throw new Error("不支持的点云清单版本");
  }
  if (
    manifest.resolution.width <= 0 ||
    manifest.resolution.height <= 0 ||
    manifest.samples.length === 0
  ) {
    throw new Error("点云清单缺少有效分辨率或样本");
  }
  if (manifest.version === 1 && !manifest.websiteSampleOrder?.length) {
    throw new Error("v1 清单缺少网站样本顺序");
  }
  if (manifest.version === 2) {
    const splits = manifestSplits(manifest);
    if (
      !splits.length ||
      !manifest.defaultSplit ||
      !splits.includes(manifest.defaultSplit)
    ) {
      throw new Error("v2 清单缺少有效数据划分");
    }
    for (const split of splits) {
      const ids = manifest.websiteSampleOrderBySplit?.[split];
      if (!ids || ids.length !== 5 || new Set(ids).size !== 5) {
        throw new Error(`${split} 必须包含五个不同的网站样本`);
      }
    }
  }
  const expectedCount =
    manifest.resolution.width * manifest.resolution.height;
  for (const sample of manifest.samples) {
    const stages = sampleStages(sample);
    if (!stages.length) throw new Error(`${sample.id} 没有点云阶段`);
    for (const stage of stages) {
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
  const splits = manifestSplits(manifest);
  if (splits.length) {
    for (const split of splits) websiteSamples(manifest, split);
  } else {
    websiteSamples(manifest);
  }
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
