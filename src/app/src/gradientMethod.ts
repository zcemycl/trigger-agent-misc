/** Gradient optimizers (mirrors trigger_agent_misc.optimization.gradient_method). */

export type Vector = number[];
export type Objective = (x: Vector) => number;
export type GradFn = (x: Vector) => Vector;
export type SampleGradFn = (index: number, x: Vector) => Vector;

export interface GradientIterationStep {
  iteration: number;
  x: number;
  xNext: number;
  objective: number;
  detail: string;
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
        x: x[0],
        xNext: xNext[0],
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
      return {
        delta: scale(-learningRate, g),
        detail: `∇f(x_k) = ${g[0].toFixed(
          4,
        )}; update x ← x − ${learningRate}·∇f`,
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
      return {
        delta: scale(-learningRate, g),
        detail: `pick i=${idx}, ∇f_i = ${g[0].toFixed(
          4,
        )}; x ← x − ${learningRate}·∇f_i`,
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
      return {
        delta: scale(-learningRate, avg),
        detail: `batch B={${sorted.join(",")}}, avg ∇ = ${avg[0].toFixed(4)}`,
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
      return {
        delta: scale(-learningRate, velocity),
        detail: `∇f = ${g[0].toFixed(
          4,
        )}, v ← ${beta}v + ∇f → ${velocity[0].toFixed(4)}`,
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
      return {
        delta: scale(-learningRate, velocity),
        detail: `y = ${lookAhead[0].toFixed(4)}, ∇f(y) = ${g[0].toFixed(
          4,
        )}, v = ${velocity[0].toFixed(4)}`,
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
          ? `∇f = ${g[0].toFixed(4)}; first step (no momentum term)`
          : `∇f = ${g[0].toFixed(4)}; + ${beta}(x_k − x_{k−1})`;
      xPrev = x.slice();
      return { delta, detail };
    },
    options,
  );
}
