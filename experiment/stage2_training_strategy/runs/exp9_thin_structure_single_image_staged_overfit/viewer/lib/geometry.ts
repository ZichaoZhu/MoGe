import type {
  CoordinateMode,
  PointCloudAsset,
  PointCloudManifest,
  PointCloudSample,
} from "./manifest";

export type RasterScope = "full" | "crop";

export function rasterIndices(
  width: number,
  height: number,
  crop: [number, number, number, number],
  scope: RasterScope,
): Uint32Array {
  if (scope === "full") {
    return Uint32Array.from({ length: width * height }, (_, index) => index);
  }
  const [x0, y0, x1, y1] = crop;
  const indices = new Uint32Array((x1 - x0) * (y1 - y0));
  let offset = 0;
  for (let row = y0; row < y1; row += 1) {
    for (let column = x0; column < x1; column += 1) {
      indices[offset] = row * width + column;
      offset += 1;
    }
  }
  return indices;
}

export function transformPoint(
  x: number,
  y: number,
  z: number,
  asset: PointCloudAsset,
  coordinateMode: CoordinateMode,
): [number, number, number] {
  const scale = coordinateMode === "aligned" ? asset.alignment.scale : 1;
  const zShift =
    coordinateMode === "aligned" ? asset.alignment.zShift : 0;
  return [scale * x, -scale * y, -(scale * z + zShift)];
}

export function pointPositions(
  raw: Float32Array,
  indices: Uint32Array,
  asset: PointCloudAsset,
  coordinateMode: CoordinateMode,
): Float32Array {
  const output = new Float32Array(indices.length * 3);
  indices.forEach((sourceIndex, outputIndex) => {
    const [x, y, z] = transformPoint(
      raw[3 * sourceIndex],
      raw[3 * sourceIndex + 1],
      raw[3 * sourceIndex + 2],
      asset,
      coordinateMode,
    );
    output[3 * outputIndex] = x;
    output[3 * outputIndex + 1] = y;
    output[3 * outputIndex + 2] = z;
  });
  return output;
}

export function pointColors(
  raw: ArrayLike<number>,
  indices: Uint32Array,
  scale = 1,
): Float32Array {
  const output = new Float32Array(indices.length * 3);
  indices.forEach((sourceIndex, outputIndex) => {
    output[3 * outputIndex] = raw[3 * sourceIndex] * scale;
    output[3 * outputIndex + 1] = raw[3 * sourceIndex + 1] * scale;
    output[3 * outputIndex + 2] = raw[3 * sourceIndex + 2] * scale;
  });
  return output;
}

export type VoxelInstance = {
  position: [number, number, number];
  color: [number, number, number];
};

export function roundHalfToEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

export function voxelInstances(
  rawPositions: Float32Array,
  rawColors: ArrayLike<number>,
  manifest: PointCloudManifest,
  sample: PointCloudSample,
  colorScale = 1,
): VoxelInstance[] {
  const { width, height } = manifest.resolution;
  const indices = rasterIndices(width, height, sample.cropXYXY, "crop");
  const depthBins = Array.from(indices, (index) => {
    const depth = rawPositions[3 * index + 2];
    if (!Number.isFinite(depth) || depth <= 0) {
      throw new Error("SSR 体素化要求 Z 为有限正数");
    }
    return roundHalfToEven(
      manifest.voxelization.depthScale * Math.log(depth),
    );
  });
  const sortedBins = [...depthBins].sort((a, b) => a - b);
  const centerBin = sortedBins[Math.floor(sortedBins.length / 2)];
  const [x0, y0, x1, y1] = sample.cropXYXY;
  const centerColumn = (x0 + x1 - 1) / 2;
  const centerRow = (y0 + y1 - 1) / 2;
  return Array.from(indices, (index, offset) => {
    const row = Math.floor(index / width);
    const column = index % width;
    const centeredDepthBin = depthBins[offset] - centerBin;
    return {
      position: [
        column - centerColumn,
        -(row - centerRow),
        centeredDepthBin === 0 ? 0 : -centeredDepthBin,
      ],
      color: [
        rawColors[3 * index] * colorScale,
        rawColors[3 * index + 1] * colorScale,
        rawColors[3 * index + 2] * colorScale,
      ],
    };
  });
}

export function finitePositions(values: Float32Array): boolean {
  for (const value of values) {
    if (!Number.isFinite(value)) return false;
  }
  return true;
}
