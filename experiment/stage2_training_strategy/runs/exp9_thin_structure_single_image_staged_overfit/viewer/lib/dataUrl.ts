const configuredOrigin =
  process.env.NEXT_PUBLIC_POINT_CLOUD_ASSET_ORIGIN?.replace(/\/+$/, "") ?? "";

export function dataUrl(path: string): string {
  if (!configuredOrigin || /^https?:\/\//.test(path)) return path;
  return `${configuredOrigin}${path.startsWith("/") ? "" : "/"}${path}`;
}
