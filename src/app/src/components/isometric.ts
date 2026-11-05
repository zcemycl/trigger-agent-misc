/** Isometric projection for 3D-style SVG charts (no WebGL). */

const ISO_COS = 0.8660254037844386;
const ISO_SIN = 0.5;

export function projectIso(
  x: number,
  y: number,
  z: number,
  scale: number,
): { px: number; py: number } {
  return {
    px: (x - y) * ISO_COS * scale,
    py: (-z + (x + y) * ISO_SIN) * scale,
  };
}

export function fitIsoBounds(
  points: { px: number; py: number }[],
  width: number,
  height: number,
  pad: number,
): (p: { px: number; py: number }) => { sx: number; sy: number } {
  let minPx = Infinity;
  let maxPx = -Infinity;
  let minPy = Infinity;
  let maxPy = -Infinity;
  for (const p of points) {
    minPx = Math.min(minPx, p.px);
    maxPx = Math.max(maxPx, p.px);
    minPy = Math.min(minPy, p.py);
    maxPy = Math.max(maxPy, p.py);
  }
  const spanX = Math.max(maxPx - minPx, 1e-6);
  const spanY = Math.max(maxPy - minPy, 1e-6);
  const innerW = width - 2 * pad;
  const innerH = height - 2 * pad;
  const s = Math.min(innerW / spanX, innerH / spanY);
  const cx = (minPx + maxPx) / 2;
  const cy = (minPy + maxPy) / 2;
  return (p) => ({
    sx: width / 2 + (p.px - cx) * s,
    sy: height / 2 + (p.py - cy) * s,
  });
}
