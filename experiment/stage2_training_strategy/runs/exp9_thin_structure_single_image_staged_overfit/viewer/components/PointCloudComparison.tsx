"use client";

import { useEffect, useMemo, useState } from "react";

import {
  CloudScene,
  type CameraSnapshot,
  type InteractionMode,
} from "@/components/CloudScene";
import type { RasterScope } from "@/lib/geometry";
import { dataUrl } from "@/lib/dataUrl";
import {
  REFINEMENT_STEPS,
  defaultManifestSplit,
  defaultStage,
  manifestSplits,
  resolveAsset,
  sampleStages,
  stageUsesAlias,
  validateExperimentCatalog,
  validateManifest,
  websiteSamples,
  type CoordinateMode,
  type DatasetSplit,
  type ExperimentCatalog,
  type ExperimentCatalogEntry,
  type PointCloudManifest,
  type PointCloudSample,
  type RefinementStep,
  type RenderMode,
  type ScopeName,
  type StageName,
} from "@/lib/manifest";
import { parseViewerUrl, serializeViewerUrl } from "@/lib/urlState";

type PaneState = {
  stage: StageName;
  step: RefinementStep;
  fitNonce: number;
  interactionMode: InteractionMode;
};

function percent(value: number, digits = 2): string {
  return `${(100 * value).toFixed(digits)}%`;
}

function splitLabel(split: DatasetSplit): string {
  return {
    train: "Train · 训练集",
    val: "Validation · 验证集",
    test: "Test · 测试集",
  }[split];
}

function Segment<T extends string | number>({
  label,
  value,
  values,
  onChange,
  format = String,
  testId,
}: {
  label: string;
  value: T;
  values: readonly T[];
  onChange: (value: T) => void;
  format?: (value: T) => string;
  testId?: string;
}) {
  return (
    <div className="segment-wrap" data-testid={testId}>
      <span className="segment-label">{label}</span>
      <div className="segment">
        {values.map((item) => (
          <button
            type="button"
            key={item}
            className={item === value ? "active" : ""}
            aria-pressed={item === value}
            onClick={() => onChange(item)}
          >
            {format(item)}
          </button>
        ))}
      </div>
    </div>
  );
}

function Metrics({
  sample,
  stage,
  step,
  scope,
}: {
  sample: PointCloudSample;
  stage: StageName;
  step: RefinementStep;
  scope: ScopeName;
}) {
  const asset = resolveAsset(sample, stage, step);
  const metrics = asset.metrics[scope];
  return (
    <dl className="metrics-grid">
      <div>
        <dt>点图 Rel</dt>
        <dd>{percent(metrics.point_rel, 3)}</dd>
      </div>
      <div>
        <dt>深度 Rel</dt>
        <dd>{percent(metrics.depth_rel, 3)}</dd>
      </div>
      <div>
        <dt>δ1.01</dt>
        <dd>{percent(metrics["depth_delta_1.01"])}</dd>
      </div>
      <div>
        <dt>边界 F1</dt>
        <dd>{percent(metrics.boundary_f1)}</dd>
      </div>
    </dl>
  );
}

function ViewerPane({
  panelId,
  title,
  sample,
  manifest,
  experiment,
  pane,
  setPane,
  coordinateMode,
  renderMode,
  rasterScope,
  syncEnabled,
  cameraSnapshot,
  onCameraChange,
}: {
  panelId: "left" | "right";
  title: string;
  sample: PointCloudSample;
  manifest: PointCloudManifest;
  experiment: ExperimentCatalogEntry;
  pane: PaneState;
  setPane: (next: PaneState) => void;
  coordinateMode: CoordinateMode;
  renderMode: RenderMode;
  rasterScope: RasterScope;
  syncEnabled: boolean;
  cameraSnapshot: CameraSnapshot | null;
  onCameraChange: (snapshot: CameraSnapshot) => void;
}) {
  const asset = resolveAsset(sample, pane.stage, pane.step);
  const metricScope: ScopeName = rasterScope === "full" ? "full" : "crop";
  const isInitialAlias = stageUsesAlias(sample, pane.stage, pane.step);
  const stages = sampleStages(sample);
  const stageLabel = experiment.stageLabels[pane.stage];
  return (
    <section className="viewer-pane" data-testid={`viewer-${panelId}`}>
      <header className="pane-header">
        <div>
          <span className="pane-kicker">{title}</span>
          <h2>
            {stageLabel} · K={pane.step}
          </h2>
          <small className="stage-detail">
            {experiment.stageDetails[pane.stage]}
          </small>
        </div>
        <button
          className="fit-button"
          type="button"
          onClick={() => setPane({ ...pane, fitNonce: pane.fitNonce + 1 })}
        >
          适配视野
        </button>
      </header>

      <div className="pane-controls">
        {stages.length > 1 && (
          <Segment
            label="阶段"
            value={pane.stage}
            values={stages}
            format={(value) => experiment.stageLabels[value]}
            onChange={(stage) => setPane({ ...pane, stage })}
            testId={`${panelId}-stage`}
          />
        )}
        <Segment
          label="精修"
          value={pane.step}
          values={REFINEMENT_STEPS}
          format={(value) => `K=${value}`}
          onChange={(step) => setPane({ ...pane, step })}
          testId={`${panelId}-k`}
        />
        <Segment
          label="鼠标左键"
          value={pane.interactionMode}
          values={["rotate", "pan"] as const}
          format={(value) => (value === "rotate" ? "旋转" : "平移")}
          onChange={(interactionMode) =>
            setPane({ ...pane, interactionMode })
          }
          testId={`${panelId}-interaction`}
        />
      </div>

      {isInitialAlias && (
        <div className="identity-note">
          资源别名：{stageLabel} K={pane.step} 与 K=0
          逐元素完全一致
        </div>
      )}

      <div className="canvas-shell">
        <CloudScene
          panelId={panelId}
          asset={asset}
          manifest={manifest}
          sample={sample}
          coordinateMode={coordinateMode}
          renderMode={renderMode}
          rasterScope={rasterScope}
          syncEnabled={syncEnabled}
          cameraSnapshot={cameraSnapshot}
          onCameraChange={onCameraChange}
          fitNonce={pane.fitNonce}
          interactionMode={pane.interactionMode}
        />
        <div className="canvas-hint">
          左键{pane.interactionMode === "rotate" ? "旋转" : "平移"} ·
          中键拖动/滚轮缩放
        </div>
      </div>

      <Metrics
        sample={sample}
        stage={pane.stage}
        step={pane.step}
        scope={metricScope}
      />
      <footer className="asset-meta">
        <span>{asset.pointCount.toLocaleString("zh-CN")} 点</span>
        {asset.pointRelReductionFromK0 !== undefined && (
          <span
            className={
              asset.pointRelReductionFromK0 >= 0
                ? "metric-improved"
                : "metric-degraded"
            }
          >
            较 K=0{" "}
            {asset.pointRelReductionFromK0 >= 0 ? "改善" : "退化"}{" "}
            {percent(Math.abs(asset.pointRelReductionFromK0))}
          </span>
        )}
        <span>s={asset.alignment.scale.toPrecision(5)}</span>
        <span>t={asset.alignment.zShift.toPrecision(5)}</span>
        <span title={asset.checkpointSha256}>
          ckpt {asset.checkpointSha256.slice(0, 8)}
        </span>
      </footer>
    </section>
  );
}

function LoadingPage() {
  return (
    <main className="boot-screen">
      <div className="boot-mark">M3</div>
      <p>正在读取点云清单…</p>
    </main>
  );
}

export function PointCloudComparison() {
  const initialUrlState = useMemo(
    () =>
      typeof window === "undefined"
        ? {}
        : parseViewerUrl(window.location.search),
    [],
  );
  const [catalog, setCatalog] = useState<ExperimentCatalog | null>(null);
  const [experimentId, setExperimentId] = useState("");
  const [manifest, setManifest] = useState<PointCloudManifest | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [split, setSplit] = useState<DatasetSplit | null>(null);
  const [sampleId, setSampleId] = useState("");
  const [coordinateMode, setCoordinateMode] =
    useState<CoordinateMode>("aligned");
  const [renderMode, setRenderMode] = useState<RenderMode>("points");
  const [rasterScope, setRasterScope] = useState<RasterScope>("crop");
  const [syncEnabled, setSyncEnabled] = useState(true);
  const [cameraSnapshot, setCameraSnapshot] =
    useState<CameraSnapshot | null>(null);
  const [left, setLeft] = useState<PaneState>({
    stage: "initial",
    step: 0,
    fitNonce: 0,
    interactionMode: "rotate",
  });
  const [right, setRight] = useState<PaneState>({
    stage: "final",
    step: 3,
    fitNonce: 0,
    interactionMode: "rotate",
  });

  useEffect(() => {
    const controller = new AbortController();
    fetch(dataUrl("/data/experiments.json"), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`实验目录请求失败：HTTP ${response.status}`);
        }
        return (await response.json()) as ExperimentCatalog;
      })
      .then((loaded) => {
        validateExperimentCatalog(loaded);
        setCatalog(loaded);
        const requested = initialUrlState.experiment;
        setExperimentId(
          requested &&
            loaded.experiments.some((entry) => entry.id === requested)
            ? requested
            : loaded.defaultExperiment,
        );
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setLoadError(error instanceof Error ? error.message : String(error));
      });
    return () => controller.abort();
  }, [initialUrlState.experiment]);

  const experiment = catalog?.experiments.find(
    (entry) => entry.id === experimentId,
  );

  useEffect(() => {
    if (!experiment) return;
    const controller = new AbortController();
    setLoadError(null);
    fetch(dataUrl(experiment.manifestUrl), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(
            `${experiment.shortLabel} 点云清单请求失败：HTTP ${response.status}`,
          );
        }
        return (await response.json()) as PointCloudManifest;
      })
      .then((loaded) => {
        if (loaded.experiment !== experiment.id) {
          throw new Error(
            `实验清单不匹配：期望 ${experiment.id}，收到 ${loaded.experiment}`,
          );
        }
        validateManifest(loaded);
        const availableSplits = manifestSplits(loaded);
        const requestedSplit =
          initialUrlState.split &&
          availableSplits.includes(initialUrlState.split)
            ? initialUrlState.split
            : defaultManifestSplit(loaded);
        const loadedSamples = websiteSamples(loaded, requestedSplit);
        const requestedSample = initialUrlState.sample
          ? loadedSamples.find(
              (candidate) => candidate.order === initialUrlState.sample,
            )
          : undefined;
        const selectedSample = requestedSample ?? loadedSamples[0];
        if (!selectedSample) {
          throw new Error(`${experiment.shortLabel} 没有可显示的样本`);
        }
        setManifest(loaded);
        setSplit(requestedSplit);
        setSampleId(selectedSample.id);
        setCameraSnapshot(null);
        setLeft((value) => ({
          ...value,
          stage: defaultStage(loaded, selectedSample, "left"),
          step: initialUrlState.leftK ?? 0,
          fitNonce: value.fitNonce + 1,
        }));
        setRight((value) => ({
          ...value,
          stage: defaultStage(loaded, selectedSample, "right"),
          step: initialUrlState.rightK ?? 3,
          fitNonce: value.fitNonce + 1,
        }));
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setLoadError(error instanceof Error ? error.message : String(error));
      });
    return () => controller.abort();
  }, [
    experiment,
    initialUrlState.leftK,
    initialUrlState.rightK,
    initialUrlState.sample,
    initialUrlState.split,
  ]);

  useEffect(() => {
    setSyncEnabled(coordinateMode === "aligned");
  }, [coordinateMode]);

  const samples = useMemo(
    () => (manifest ? websiteSamples(manifest, split) : []),
    [manifest, split],
  );
  const sample = samples.find((value) => value.id === sampleId) ?? samples[0];

  useEffect(() => {
    if (
      typeof window === "undefined" ||
      !catalog ||
      !manifest ||
      !sample
    ) {
      return;
    }
    const query = serializeViewerUrl({
      experiment: experimentId,
      split: split ?? undefined,
      sample: sample.order,
      leftK: left.step,
      rightK: right.step,
    });
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${query}${window.location.hash}`,
    );
  }, [
    catalog,
    experimentId,
    left.step,
    manifest,
    right.step,
    sample,
    split,
  ]);

  if (loadError) {
    return (
      <main className="boot-screen error-screen" role="alert">
        <div className="boot-mark">!</div>
        <h1>查看器无法启动</h1>
        <p>{loadError}</p>
      </main>
    );
  }
  if (!catalog || !experiment || !manifest || !sample) return <LoadingPage />;

  const updateGlobalView = (update: () => void) => {
    update();
    setCameraSnapshot(null);
    setLeft((value) => ({ ...value, fitNonce: value.fitNonce + 1 }));
    setRight((value) => ({ ...value, fitNonce: value.fitNonce + 1 }));
  };
  const availableSplits = manifestSplits(manifest);

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <span className="eyebrow">MoGe-3 / INTERACTIVE GEOMETRY LAB</span>
          <h1>多实验点云对比器</h1>
          <p>
            在统一坐标、相机和渲染设置下，比较不同训练实验、阶段与 SSR
            迭代次数。存储的是网络原始 XYZ；默认仅为公平比较应用全图 GT
            仿射对齐。
          </p>
        </div>
        <div className="hero-badge">
          <span>{experiment.shortLabel} · 当前样本</span>
          <strong>{sample.label}</strong>
          <small>{sample.description}</small>
        </div>
      </header>

      <nav className="experiment-switcher" aria-label="实验切换">
        <div className="switcher-heading">
          <span>实验入口</span>
          <strong>{experiment.label}</strong>
        </div>
        {catalog.experiments.map((entry) => (
          <button
            type="button"
            key={entry.id}
            className={entry.id === experiment.id ? "active" : ""}
            aria-pressed={entry.id === experiment.id}
            data-testid={`experiment-${entry.id}`}
            onClick={() => {
              if (entry.id === experiment.id) return;
              setManifest(null);
              setExperimentId(entry.id);
            }}
          >
            <strong>{entry.shortLabel}</strong>
            <small>{entry.summary}</small>
          </button>
        ))}
      </nav>

      <div className="experiment-summary" aria-live="polite">
        <strong>{experiment.label}</strong>
        <span>{experiment.summary}</span>
      </div>

      {manifest.provenance && (
        <section className="provenance-warning" role="status">
          <strong>结果口径</strong>
          <span>
            当前入口使用 {manifest.provenance.sourceExperiment} 的 step{" "}
            {manifest.provenance.checkpointStep} 权重；SSR 采用单图即时
            BatchNorm 统计。这是训练域数值最佳的推理诊断，不是新的训练权重，
            也不代表验证集或测试集泛化成功。
          </span>
        </section>
      )}

      {availableSplits.length > 0 && (
        <nav className="split-switcher" aria-label="数据划分切换">
          <span className="split-heading">数据划分</span>
          {availableSplits.map((item) => (
            <button
              type="button"
              key={item}
              className={item === split ? "active" : ""}
              aria-pressed={item === split}
              data-testid={`split-${item}`}
              onClick={() => {
                if (item === split) return;
                const nextSamples = websiteSamples(manifest, item);
                const nextSample = nextSamples[0];
                updateGlobalView(() => {
                  setSplit(item);
                  setSampleId(nextSample.id);
                  setLeft((value) => ({
                    ...value,
                    stage: defaultStage(manifest, nextSample, "left"),
                  }));
                  setRight((value) => ({
                    ...value,
                    stage: defaultStage(manifest, nextSample, "right"),
                  }));
                });
              }}
            >
              {splitLabel(item)}
            </button>
          ))}
        </nav>
      )}

      <nav className="sample-switcher" aria-label="样本切换">
        {samples.map((item) => (
          <button
            type="button"
            key={item.id}
            className={item.id === sample.id ? "active" : ""}
            aria-pressed={item.id === sample.id}
            data-testid={`sample-${item.order}`}
            onClick={() =>
              updateGlobalView(() => {
                setSampleId(item.id);
                setLeft((value) => ({
                  ...value,
                  stage: defaultStage(manifest, item, "left"),
                }));
                setRight((value) => ({
                  ...value,
                  stage: defaultStage(manifest, item, "right"),
                }));
              })
            }
          >
            <img src={dataUrl(item.rgbUrl)} alt="" />
            <span>
              <strong>{item.label}</strong>
              <small className="sample-id">{item.id}</small>
              {item.selection ? (
                <small
                  className={
                    item.selection.relativeImprovement >= 0
                      ? "sample-gain improved"
                      : "sample-gain degraded"
                  }
                >
                  排名 {item.selection.rank}/{item.selection.total} · K=3{" "}
                  {item.selection.relativeImprovement >= 0
                    ? "改善"
                    : "退化"}{" "}
                  {percent(
                    Math.abs(item.selection.relativeImprovement),
                  )}
                </small>
              ) : (
                <small>{item.description}</small>
              )}
            </span>
          </button>
        ))}
      </nav>

      <section className="global-toolbar">
        <Segment
          label="坐标"
          value={coordinateMode}
          values={["aligned", "raw"] as const}
          format={(value) => (value === "aligned" ? "GT 对齐" : "原始输出")}
          onChange={(value) =>
            updateGlobalView(() => setCoordinateMode(value))
          }
          testId="coordinate-mode"
        />
        <Segment
          label="范围"
          value={rasterScope}
          values={["crop", "full"] as const}
          format={(value) => (value === "crop" ? "细结构裁剪" : "完整场景")}
          onChange={(value) =>
            updateGlobalView(() => setRasterScope(value))
          }
          testId="raster-scope"
        />
        <Segment
          label="渲染"
          value={renderMode}
          values={["points", "voxels"] as const}
          format={(value) => (value === "points" ? "彩色点云" : "SSR 体素壳")}
          onChange={(value) =>
            updateGlobalView(() => setRenderMode(value))
          }
          testId="render-mode"
        />
        <label
          className={`sync-toggle ${syncEnabled ? "active" : ""}`}
          title={
            coordinateMode === "raw"
              ? "原始相对尺度不同，不能同步比较"
              : "同步两个窗口的相机"
          }
        >
          <input
            type="checkbox"
            checked={syncEnabled}
            disabled={coordinateMode === "raw"}
            onChange={(event) => setSyncEnabled(event.target.checked)}
          />
          <span className="toggle-track" />
          相机同步
        </label>
      </section>

      {coordinateMode === "raw" && (
        <div className="raw-warning" role="status">
          原始点图只有相对尺度，不同检查点不能直接比较大小；两个窗口已分别适配视野并关闭相机同步。
        </div>
      )}
      {renderMode === "voxels" && (
        <div className="voxel-note" role="status">
          SSR 体素壳固定显示锁定裁剪；深度轴按 round(200·log Z)
          离散，只减去中位 depth bin 以居中画面。
        </div>
      )}

      <div className="comparison-grid">
        <ViewerPane
          panelId="left"
          title="窗口 A"
          sample={sample}
          manifest={manifest}
          experiment={experiment}
          pane={left}
          setPane={setLeft}
          coordinateMode={coordinateMode}
          renderMode={renderMode}
          rasterScope={renderMode === "voxels" ? "crop" : rasterScope}
          syncEnabled={syncEnabled}
          cameraSnapshot={cameraSnapshot}
          onCameraChange={setCameraSnapshot}
        />
        <ViewerPane
          panelId="right"
          title="窗口 B"
          sample={sample}
          manifest={manifest}
          experiment={experiment}
          pane={right}
          setPane={setRight}
          coordinateMode={coordinateMode}
          renderMode={renderMode}
          rasterScope={renderMode === "voxels" ? "crop" : rasterScope}
          syncEnabled={syncEnabled}
          cameraSnapshot={cameraSnapshot}
          onCameraChange={setCameraSnapshot}
        />
      </div>

      <footer className="page-footer">
        <div>
          <strong>坐标说明</strong>
          <span>
            P′=sP+(0,0,t)；显示轴变换为 [x,−y,−z]。体素顺序遵循
            [batch,depth,row,column]。
          </span>
        </div>
        <div>
          <strong>归档范围</strong>
          <span>
            当前实验共开放{" "}
            {manifest.version === 2
              ? manifest.samples.filter((item) => item.websiteEnabled).length
              : samples.length}{" "}
            张样本；当前划分显示 {samples.length} 张。
          </span>
        </div>
      </footer>
    </main>
  );
}
