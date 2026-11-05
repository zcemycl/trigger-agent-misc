export type PathHighlightRange = readonly [number, number];

export function isPathIndexHighlighted(
  index: number,
  range: PathHighlightRange | null | undefined,
): boolean {
  if (range == null) {
    return false;
  }
  return index >= range[0] && index <= range[1];
}

export function pathHighlightRangeForStep(
  steps: readonly {
    iteration: number;
    pathStartIndex?: number;
    pathEndIndex?: number;
  }[],
  step: number | null | undefined,
): PathHighlightRange | null {
  if (step == null) {
    return null;
  }
  const row = steps.find((s) => s.iteration === step);
  if (
    row?.pathStartIndex == null ||
    row.pathEndIndex == null ||
    row.pathEndIndex < row.pathStartIndex
  ) {
    return null;
  }
  return [row.pathStartIndex, row.pathEndIndex];
}
