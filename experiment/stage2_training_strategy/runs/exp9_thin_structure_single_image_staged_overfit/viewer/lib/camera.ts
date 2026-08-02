export function shouldApplyCameraFit({
  boundsSource,
  sceneSource,
  hasFitted,
  fitNonce,
  appliedFitNonce,
}: {
  boundsSource: string;
  sceneSource: string;
  hasFitted: boolean;
  fitNonce: number;
  appliedFitNonce: number | null;
}): boolean {
  if (!boundsSource || boundsSource !== sceneSource) return false;
  return !hasFitted || appliedFitNonce !== fitNonce;
}
