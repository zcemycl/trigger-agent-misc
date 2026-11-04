import {
  isPathIndexHighlighted,
  type PathHighlightRange,
} from "./pathHighlight";

export function pathPointRadius(
  index: number,
  pathLength: number,
  highlightRange: PathHighlightRange | null | undefined,
): number {
  const isStart = index === 0;
  const isEnd = index === pathLength - 1;
  const highlighted = isPathIndexHighlighted(index, highlightRange);
  if (highlighted) {
    return 7;
  }
  if (isStart || isEnd) {
    return 5;
  }
  return 3;
}

export function pathPointFill(
  index: number,
  pathLength: number,
  highlightRange: PathHighlightRange | null | undefined,
): string {
  const isStart = index === 0;
  const isEnd = index === pathLength - 1;
  const highlighted = isPathIndexHighlighted(index, highlightRange);
  if (highlighted) {
    return "#f472b6";
  }
  if (isStart) {
    return "#fbbf24";
  }
  if (isEnd) {
    return "#34d399";
  }
  return "#818cf8";
}

export function pathPointOpacity(
  index: number,
  highlightRange: PathHighlightRange | null | undefined,
): number {
  if (highlightRange == null) {
    return 1;
  }
  return isPathIndexHighlighted(index, highlightRange) ? 1 : 0.35;
}
