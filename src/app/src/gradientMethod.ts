/** Gradient optimizers (mirrors trigger_agent_misc.optimization.gradient_method). */

export type Vector = number[];
export type Objective = (x: Vector) => number;
export type GradFn = (x: Vector) => Vector;
export type SampleGradFn = (index: number, x: Vector) => Vector;

export interface GradientIterationStep {
  iteration: number;
  /** Parameter vector x_k before the update. */
  x: Vector;
  /** Update increment Δx_k added to x_k. */
  delta: Vector;
  /** Parameter vector x_{k+1} = x_k + Δx_k. */
  xNext: Vector;
  objective: number;
  detail: string;
}

/** Bracket notation for a parameter or gradient vector (demo is 1D but API is general). */
export function formatVector(v: Vector, digits = 4): string {
  const inner = v.map((c) => c.toFixed(digits)).join(", ");
  return `[${inner}]`;
}

export interface GradientMethodResult {
  x: Vector;
  value: number;
  iterations: number;
  history: Vector[];
  steps: GradientIterationStep[];
}

type StepDelta =
  | Vector
  | null
  | {
      delta: Vector;
      detail: string;
    };

function validateVector(x: Vector): Vector {
  if (x.length === 0) {
    throw new Error("x must be non-empty");
  }
  return x.map((v) => Number(v));
}

function add(a: Vector, b: Vector): Vector {
  return a.map((v, i) => v + b[i]);
}

function sub(a: Vector, b: Vector): Vector {
  return a.map((v, i) => v - b[i]);
}

function scale(s: number, v: Vector): Vector {
  return v.map((x) => s * x);
}

function norm(v: Vector): number {
  return Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));
}

function mean(vectors: Vector[]): Vector {
  if (vectors.length === 0) {
    throw new Error("cannot average empty batch");
  }
  const dim = vectors[0].length;
  const acc = new Array(dim).fill(0);
  for (const vec of vectors) {
    for (let i = 0; i < dim; i += 1) {
      acc[i] += vec[i];
    }
  }
  return acc.map((v) => v / vectors.length);
}

function unpackStep(
  outcome: StepDelta,
): { delta: Vector; detail: string } | null {
  if (outcome == null) {
    return null;
  }
  if (Array.isArray(outcome)) {
    return { delta: outcome, detail: "" };
  }
  return outcome;
}

function runFirstOrder(
  x0: Vector,
  objective: Objective,
  step: (x: Vector, k: number) => StepDelta,
  options: {
    maxIter?: number;
    tol?: number;
    recordSteps?: boolean;
  } = {},
): GradientMethodResult {
  const maxIter = options.maxIter ?? 200;
  const tol = options.tol ?? 1e-9;
  const recordSteps = options.recordSteps ?? true;
  let x = validateVector(x0);
  const history: Vector[] = [x.slice()];
  const steps: GradientIterationStep[] = [];
  for (let k = 0; k < maxIter; k += 1) {
    const unpacked = unpackStep(step(x, k));
    if (unpacked == null) {
      break;
    }
    const { delta, detail } = unpacked;
    const xNext = add(x, delta);
    if (recordSteps) {
      steps.push({
        iteration: k,
        x: x.slice(),
        delta: delta.slice(),
        xNext: xNext.slice(),
        objective: objective(x),
        detail,
      });
    }
    x = xNext;
    history.push(x.slice());
    if (norm(delta) <= tol) {
      break;
    }
  }
  return {
    x,
    value: objective(x),
    iterations: history.length - 1,
    history,
    steps,
  };
}

export function gradientDescent(
  x0: Vector,
  objective: Objective,
  grad: GradFn,
  options: { learningRate?: number; maxIter?: number; tol?: number } = {},
): GradientMethodResult {
  const learningRate = options.learningRate ?? 0.1;
  if (learningRate <= 0) {
    throw new Error("learning_rate must be positive");
  }
  return runFirstOrder(
    x0,
    objective,
    (x) => {
      const g = validateVector(grad(x));
      const delta = scale(-learningRate, g);
      return {
        delta,
        detail: `∇f(x_k) = ${formatVector(
          g,
        )}; Δx_k = −η·∇f(x_k) = ${formatVector(delta)}; x_{k+1} = x_k + Δx_k`,
      };
    },
    options,
  );
}

export function stochasticGradientDescent(
  x0: Vector,
  objective: Objective,
  gradAtIndex: SampleGradFn,
  numSamples: number,
  options: {
    learningRate?: number;
    maxIter?: number;
    tol?: number;
    seed?: number;
  } = {},
): GradientMethodResult {
  const learningRate = options.learningRate ?? 0.1;
  if (numSamples <= 0) {
    throw new Error("num_samples must be positive");
  }
  let state = options.seed ?? 0;
  const rand = () => {
    state = (state * 1664525 + 1013904223) % 2 ** 32;
    return state / 2 ** 32;
  };
  return runFirstOrder(
    x0,
    objective,
    (x) => {
      const idx = Math.floor(rand() * numSamples);
      const g = validateVector(gradAtIndex(idx, x));
      const delta = scale(-learningRate, g);
      return {
        delta,
        detail: `i_k = ${idx}, ∇f_{i_k}(x_k) = ${formatVector(
          g,
        )}; Δx_k = −η·∇f_{i_k} = ${formatVector(delta)}; x_{k+1} = x_k + Δx_k`,
      };
    },
    options,
  );
}

export function minibatchSgd(
  x0: Vector,
  objective: Objective,
  gradAtIndex: SampleGradFn,
  numSamples: number,
  options: {
    batchSize?: number;
    learningRate?: number;
    maxIter?: number;
    tol?: number;
    seed?: number;
  } = {},
): GradientMethodResult {
  const learningRate = options.learningRate ?? 0.1;
  const batchSize = Math.min(options.batchSize ?? 4, numSamples);
  let state = options.seed ?? 0;
  const rand = () => {
    state = (state * 1664525 + 1013904223) % 2 ** 32;
    return state / 2 ** 32;
  };
  return runFirstOrder(
    x0,
    objective,
    (x) => {
      const indices = new Set<number>();
      while (indices.size < batchSize) {
        indices.add(Math.floor(rand() * numSamples));
      }
      const sorted = [...indices].sort((a, b) => a - b);
      const grads = sorted.map((i) => validateVector(gradAtIndex(i, x)));
      const avg = mean(grads);
      const delta = scale(-learningRate, avg);
      return {
        delta,
        detail: `B = {${sorted.join(",")}}, ḡ_B = ${formatVector(
          avg,
        )}; Δx_k = −η·ḡ_B = ${formatVector(delta)}; x_{k+1} = x_k + Δx_k`,
      };
    },
    options,
  );
}

export function momentum(
  x0: Vector,
  objective: Objective,
  grad: GradFn,
  options: {
    learningRate?: number;
    beta?: number;
    maxIter?: number;
    tol?: number;
  } = {},
): GradientMethodResult {
  const learningRate = options.learningRate ?? 0.1;
  const beta = options.beta ?? 0.9;
  let velocity = scale(0, validateVector(x0));
  return runFirstOrder(
    x0,
    objective,
    (x) => {
      const g = validateVector(grad(x));
      velocity = add(scale(beta, velocity), g);
      const delta = scale(-learningRate, velocity);
      return {
        delta,
        detail: `∇f(x_k) = ${formatVector(
          g,
        )}; v_{k+1} = β·v_k + ∇f(x_k) = ${formatVector(
          velocity,
        )}; Δx_k = −η·v_{k+1} = ${formatVector(delta)}; x_{k+1} = x_k + Δx_k`,
      };
    },
    options,
  );
}

export function nesterovAcceleratedGradient(
  x0: Vector,
  objective: Objective,
  grad: GradFn,
  options: {
    learningRate?: number;
    beta?: number;
    maxIter?: number;
    tol?: number;
  } = {},
): GradientMethodResult {
  const learningRate = options.learningRate ?? 0.1;
  const beta = options.beta ?? 0.9;
  let velocity = scale(0, validateVector(x0));
  return runFirstOrder(
    x0,
    objective,
    (x) => {
      const lookAhead = sub(x, scale(beta, velocity));
      const g = validateVector(grad(lookAhead));
      velocity = add(scale(beta, velocity), g);
      const delta = scale(-learningRate, velocity);
      return {
        delta,
        detail: `y_k = x_k − β·v_k = ${formatVector(
          lookAhead,
        )}; ∇f(y_k) = ${formatVector(g)}; v_{k+1} = ${formatVector(
          velocity,
        )}; Δx_k = −η·v_{k+1} = ${formatVector(delta)}; x_{k+1} = x_k + Δx_k`,
      };
    },
    options,
  );
}

export function heavyBallMethod(
  x0: Vector,
  objective: Objective,
  grad: GradFn,
  options: {
    learningRate?: number;
    beta?: number;
    maxIter?: number;
    tol?: number;
  } = {},
): GradientMethodResult {
  const learningRate = options.learningRate ?? 0.1;
  const beta = options.beta ?? 0.9;
  let xPrev = validateVector(x0);
  return runFirstOrder(
    x0,
    objective,
    (x, k) => {
      const g = validateVector(grad(x));
      const delta =
        k === 0
          ? scale(-learningRate, g)
          : add(scale(-learningRate, g), scale(beta, sub(x, xPrev)));
      const detail =
        k === 0
          ? `∇f(x_k) = ${formatVector(g)}; Δx_k = −η·∇f(x_k) = ${formatVector(
              delta,
            )} (no x_{k−1} term yet); x_{k+1} = x_k + Δx_k`
          : `∇f(x_k) = ${formatVector(
              g,
            )}; Δx_k = −η·∇f(x_k) + β·(x_k − x_{k−1}) = ${formatVector(
              delta,
            )}; x_{k+1} = x_k + Δx_k`;
      xPrev = x.slice();
      return { delta, detail };
    },
    options,
  );
}
