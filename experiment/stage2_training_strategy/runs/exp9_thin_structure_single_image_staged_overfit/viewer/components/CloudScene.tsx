"use client";

import { Html, OrbitControls } from "@react-three/drei";
import { Canvas, useLoader, useThree } from "@react-three/fiber";
import {
  Component,
  Suspense,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ComponentRef,
  type ErrorInfo,
  type MutableRefObject,
  type ReactNode,
} from "react";
import * as THREE from "three";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

import {
  finitePositions,
  pointColors,
  pointPositions,
  rasterIndices,
  voxelInstances,
  type RasterScope,
} from "@/lib/geometry";
import { shouldApplyCameraFit } from "@/lib/camera";
import { dataUrl } from "@/lib/dataUrl";
import type {
  CoordinateMode,
  PointCloudAsset,
  PointCloudManifest,
  PointCloudSample,
  RenderMode,
} from "@/lib/manifest";

export type CameraSnapshot = {
  source: string;
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
};

export type InteractionMode = "rotate" | "pan";

type CameraMemory = {
  snapshot: CameraSnapshot | null;
  hasFitted: boolean;
  appliedFitNonce: number | null;
  activeSceneSource: string;
};

type Bounds = {
  center: [number, number, number];
  radius: number;
};

type SceneBounds = Bounds & {
  source: string;
};

type CloudSceneProps = {
  panelId: string;
  asset: PointCloudAsset;
  manifest: PointCloudManifest;
  sample: PointCloudSample;
  coordinateMode: CoordinateMode;
  renderMode: RenderMode;
  rasterScope: RasterScope;
  syncEnabled: boolean;
  cameraSnapshot: CameraSnapshot | null;
  onCameraChange: (snapshot: CameraSnapshot) => void;
  fitNonce: number;
  interactionMode: InteractionMode;
};

function LoadingCloud() {
  return (
    <Html center>
      <div className="canvas-message">点云载入中…</div>
    </Html>
  );
}

class SceneErrorBoundary extends Component<
  { children: ReactNode; resetKey: string },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("点云渲染失败", error, info);
  }

  componentDidUpdate(previous: { children: ReactNode; resetKey: string }) {
    if (previous.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  render() {
    if (this.state.error) {
      return (
        <Html center>
          <div className="canvas-message canvas-error" role="alert">
            <strong>点云渲染失败</strong>
            <span>{this.state.error.message}</span>
          </div>
        </Html>
      );
    }
    return this.props.children;
  }
}

function geometryBounds(geometry: THREE.BufferGeometry): Bounds {
  geometry.computeBoundingSphere();
  const sphere = geometry.boundingSphere;
  if (!sphere || !Number.isFinite(sphere.radius) || sphere.radius <= 0) {
    throw new Error("无法从点云计算有效的相机范围");
  }
  return {
    center: sphere.center.toArray() as [number, number, number],
    radius: sphere.radius,
  };
}

function PointMap({
  asset,
  manifest,
  sample,
  coordinateMode,
  rasterScope,
  onBounds,
}: Pick<
  CloudSceneProps,
  "asset" | "manifest" | "sample" | "coordinateMode" | "rasterScope"
> & { onBounds: (bounds: Bounds) => void }) {
  const source = useLoader(PLYLoader, dataUrl(asset.url));
  const geometry = useMemo(() => {
    const sourcePosition = source.getAttribute("position");
    const sourceColor = source.getAttribute("color");
    if (!sourcePosition || !sourceColor) {
      throw new Error("PLY 缺少 position 或 RGB color 属性");
    }
    if (sourcePosition.count !== asset.pointCount) {
      throw new Error(
        `PLY 点数 ${sourcePosition.count} 与清单 ${asset.pointCount} 不一致`,
      );
    }
    const rawPositions = sourcePosition.array as Float32Array;
    const rawColors = sourceColor.array;
    const indices = rasterIndices(
      manifest.resolution.width,
      manifest.resolution.height,
      sample.cropXYXY,
      rasterScope,
    );
    const positions = pointPositions(
      rawPositions,
      indices,
      asset,
      coordinateMode,
    );
    if (!finitePositions(positions)) {
      throw new Error("坐标变换后出现 NaN 或无穷大");
    }
    const colors = pointColors(
      rawColors,
      indices,
      sourceColor.normalized ? 1 / 255 : 1,
    );
    const result = new THREE.BufferGeometry();
    result.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    result.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return result;
  }, [
    asset,
    coordinateMode,
    manifest.resolution.height,
    manifest.resolution.width,
    rasterScope,
    sample.cropXYXY,
    source,
  ]);

  const bounds = useMemo(() => geometryBounds(geometry), [geometry]);
  useEffect(() => {
    onBounds(bounds);
    return () => geometry.dispose();
  }, [bounds, geometry, onBounds]);

  const pointSize = Math.max(bounds.radius / 420, 0.00001);
  return (
    <points geometry={geometry}>
      <pointsMaterial
        attach="material"
        size={pointSize}
        sizeAttenuation
        vertexColors
        toneMapped={false}
      />
    </points>
  );
}

function VoxelShell({
  asset,
  manifest,
  sample,
  onBounds,
}: Pick<CloudSceneProps, "asset" | "manifest" | "sample"> & {
  onBounds: (bounds: Bounds) => void;
}) {
  const source = useLoader(PLYLoader, dataUrl(asset.url));
  const mesh = useRef<THREE.InstancedMesh>(null);
  const instances = useMemo(() => {
    const sourcePosition = source.getAttribute("position");
    const sourceColor = source.getAttribute("color");
    if (!sourcePosition || !sourceColor) {
      throw new Error("PLY 缺少生成 SSR 体素所需的属性");
    }
    return voxelInstances(
      sourcePosition.array as Float32Array,
      sourceColor.array,
      manifest,
      sample,
      sourceColor.normalized ? 1 / 255 : 1,
    );
  }, [asset.url, manifest, sample, source]);

  const bounds = useMemo(() => {
    const box = new THREE.Box3();
    for (const instance of instances) {
      box.expandByPoint(new THREE.Vector3(...instance.position));
    }
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    return {
      center: sphere.center.toArray() as [number, number, number],
      radius: sphere.radius,
    };
  }, [instances]);

  useLayoutEffect(() => {
    if (!mesh.current) return;
    const matrix = new THREE.Matrix4();
    const color = new THREE.Color();
    instances.forEach((instance, index) => {
      matrix.makeTranslation(...instance.position);
      mesh.current?.setMatrixAt(index, matrix);
      color.setRGB(...instance.color);
      mesh.current?.setColorAt(index, color);
    });
    mesh.current.instanceMatrix.needsUpdate = true;
    if (mesh.current.instanceColor) mesh.current.instanceColor.needsUpdate = true;
    onBounds(bounds);
  }, [bounds, instances, onBounds]);

  return (
    <instancedMesh
      ref={mesh}
      args={[undefined, undefined, instances.length]}
      frustumCulled={false}
    >
      <boxGeometry args={[0.82, 0.82, 0.82]} />
      <meshBasicMaterial vertexColors toneMapped={false} />
    </instancedMesh>
  );
}

function CameraRig({
  panelId,
  bounds,
  sceneSource,
  syncEnabled,
  cameraSnapshot,
  onCameraChange,
  fitNonce,
  interactionMode,
  cameraMemory,
}: Pick<
  CloudSceneProps,
  | "panelId"
  | "syncEnabled"
  | "cameraSnapshot"
  | "onCameraChange"
  | "fitNonce"
  | "interactionMode"
> & {
  bounds: SceneBounds;
  sceneSource: string;
  cameraMemory: MutableRefObject<CameraMemory>;
}) {
  const { camera, gl, invalidate } = useThree();
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const applyingRemote = useRef(false);

  const recordCamera = () => {
    if (
      !controls.current ||
      cameraMemory.current.activeSceneSource !== sceneSource
    ) {
      return null;
    }
    const snapshot: CameraSnapshot = {
      source: panelId,
      position: camera.position.toArray() as [number, number, number],
      target: controls.current.target.toArray() as [number, number, number],
      up: camera.up.toArray() as [number, number, number],
    };
    cameraMemory.current.snapshot = snapshot;
    gl.domElement.dataset.cameraSnapshot = JSON.stringify({
      position: snapshot.position,
      target: snapshot.target,
      up: snapshot.up,
    });
    return snapshot;
  };

  const fit = () => {
    const center = new THREE.Vector3(...bounds.center);
    const radius = Math.max(bounds.radius, 1e-4);
    camera.position.copy(
      center.clone().add(new THREE.Vector3(0.85, 0.32, 1.5).normalize().multiplyScalar(radius * 2.1)),
    );
    camera.up.set(0, 1, 0);
    camera.near = Math.max(radius / 1500, 1e-6);
    camera.far = Math.max(radius * 60, 100);
    camera.updateProjectionMatrix();
    if (controls.current) {
      controls.current.target.copy(center);
      controls.current.update();
      controls.current.saveState();
    }
    invalidate();
  };

  useEffect(() => {
    if (
      !shouldApplyCameraFit({
        boundsSource: bounds.source,
        sceneSource,
        hasFitted: cameraMemory.current.hasFitted,
        fitNonce,
        appliedFitNonce: cameraMemory.current.appliedFitNonce,
      })
    ) {
      const remembered = cameraMemory.current.snapshot;
      if (remembered && controls.current && bounds.source === sceneSource) {
        camera.position.fromArray(remembered.position);
        camera.up.fromArray(remembered.up);
        controls.current.target.fromArray(remembered.target);
        camera.updateProjectionMatrix();
        controls.current.update();
        cameraMemory.current.activeSceneSource = sceneSource;
        recordCamera();
        gl.domElement.dataset.sceneSource = sceneSource;
        invalidate();
      }
      return;
    }
    fit();
    cameraMemory.current.activeSceneSource = sceneSource;
    recordCamera();
    cameraMemory.current.hasFitted = true;
    cameraMemory.current.appliedFitNonce = fitNonce;
    gl.domElement.dataset.sceneSource = sceneSource;
    // Bounds changes caused only by stage/K updates intentionally preserve
    // the camera. fitNonce changes only for explicit/global view resets.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    bounds.center.join(","),
    bounds.radius,
    bounds.source,
    cameraMemory,
    fitNonce,
    sceneSource,
  ]);

  useEffect(() => {
    if (
      !syncEnabled ||
      !cameraSnapshot ||
      cameraSnapshot.source === panelId ||
      !controls.current
    ) {
      return;
    }
    applyingRemote.current = true;
    camera.position.fromArray(cameraSnapshot.position);
    camera.up.fromArray(cameraSnapshot.up);
    controls.current.target.fromArray(cameraSnapshot.target);
    camera.updateProjectionMatrix();
    controls.current.update();
    recordCamera();
    invalidate();
    requestAnimationFrame(() => {
      applyingRemote.current = false;
    });
  }, [camera, cameraSnapshot, invalidate, panelId, syncEnabled]);

  const publishCamera = () => {
    const snapshot = recordCamera();
    if (!syncEnabled || applyingRemote.current || !snapshot) return;
    onCameraChange(snapshot);
  };

  return (
    <OrbitControls
      key={sceneSource}
      ref={controls}
      makeDefault
      enableDamping={false}
      screenSpacePanning
      zoomToCursor
      mouseButtons={{
        LEFT:
          interactionMode === "rotate" ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: undefined,
      }}
      touches={{
        ONE:
          interactionMode === "rotate" ? THREE.TOUCH.ROTATE : THREE.TOUCH.PAN,
        TWO: THREE.TOUCH.DOLLY_PAN,
      }}
      onChange={publishCamera}
    />
  );
}

function Scene({
  renderMode,
  cameraMemory,
  ...props
}: CloudSceneProps & {
  cameraMemory: MutableRefObject<CameraMemory>;
}) {
  const sceneSource = [
    props.sample.id,
    props.asset.url,
    props.coordinateMode,
    renderMode,
    props.rasterScope,
  ].join(":");
  const [bounds, setBounds] = useState<SceneBounds>({
    source: "",
    center: [0, 0, 0],
    radius: 1,
  });
  const updateBounds = useCallback(
    (next: Bounds) => setBounds({ ...next, source: sceneSource }),
    [sceneSource],
  );
  return (
    <>
      <color attach="background" args={["#071017"]} />
      {renderMode === "points" ? (
        <PointMap
          {...props}
          rasterScope={props.rasterScope}
          onBounds={updateBounds}
        />
      ) : (
        <VoxelShell {...props} onBounds={updateBounds} />
      )}
      <CameraRig
        {...props}
        bounds={bounds}
        sceneSource={sceneSource}
        cameraMemory={cameraMemory}
      />
      <axesHelper args={[Math.max(bounds.radius * 0.28, 0.1)]} />
    </>
  );
}

export function CloudScene(props: CloudSceneProps) {
  const cameraMemory = useRef<CameraMemory>({
    snapshot: null,
    hasFitted: false,
    appliedFitNonce: null,
    activeSceneSource: "",
  });
  const resetKey = [
    props.sample.id,
    props.asset.url,
    props.coordinateMode,
    props.renderMode,
    props.rasterScope,
  ].join(":");
  return (
    <Canvas
      className="cloud-canvas"
      dpr={[1, 1.5]}
      camera={{ fov: 45, near: 0.001, far: 1000 }}
      gl={{ antialias: true, alpha: false, preserveDrawingBuffer: true }}
      frameloop="demand"
      data-testid={`canvas-${props.panelId}`}
      onContextMenu={(event) => event.preventDefault()}
    >
      <SceneErrorBoundary resetKey={resetKey}>
        <Suspense fallback={<LoadingCloud />}>
          <Scene {...props} cameraMemory={cameraMemory} />
        </Suspense>
      </SceneErrorBoundary>
    </Canvas>
  );
}
