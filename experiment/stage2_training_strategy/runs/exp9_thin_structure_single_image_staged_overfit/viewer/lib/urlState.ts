import {
  REFINEMENT_STEPS,
  type DatasetSplit,
  type RefinementStep,
} from "./manifest";

export type ViewerUrlState = {
  experiment?: string;
  split?: DatasetSplit;
  sample?: number;
  leftK?: RefinementStep;
  rightK?: RefinementStep;
};

function refinementStep(value: string | null): RefinementStep | undefined {
  if (value === null) return undefined;
  const parsed = Number(value);
  return REFINEMENT_STEPS.find((step) => step === parsed);
}

export function parseViewerUrl(search: string): ViewerUrlState {
  const params = new URLSearchParams(search);
  const split = params.get("split");
  const sample = Number(params.get("sample"));
  return {
    experiment: params.get("experiment") || undefined,
    split:
      split === "train" || split === "val" || split === "test"
        ? split
        : undefined,
    sample:
      Number.isInteger(sample) && sample >= 1 && sample <= 5
        ? sample
        : undefined,
    leftK: refinementStep(params.get("leftK")),
    rightK: refinementStep(params.get("rightK")),
  };
}

export function serializeViewerUrl(state: ViewerUrlState): string {
  const params = new URLSearchParams();
  if (state.experiment) params.set("experiment", state.experiment);
  if (state.split) params.set("split", state.split);
  if (state.sample) params.set("sample", String(state.sample));
  if (state.leftK !== undefined) params.set("leftK", String(state.leftK));
  if (state.rightK !== undefined) params.set("rightK", String(state.rightK));
  const query = params.toString();
  return query ? `?${query}` : "";
}
