/** Gradient optimizers (mirrors trigger_agent_misc.optimization.gradient_method). */

export type Vector = number[];
export type Objective = (x: Vector) => number;
export type GradFn = (x: Vector) => Vector;
export type SampleGradFn = (index: number, x: Vector) => Vector;

export interface GradientMethodResult {
  x: Vector;
  value: number;
  iterations: number;
  history: Vector[];
}

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

function runFirstOrder(
  x0: Vector,
  objective: Objective,
  step: (x: Vector, k: number) => Vector | null,
  options: { maxIter?: number; tol?: number } = {},
): GradientMethodResult {
  const maxIter = options.maxIter ?? 200;
  const tol = options.tol ?? 1e-9;
  let x = validateVector(x0);
  const history: Vector[] = [x.slice()];
  for (let k = 0; k < maxIter; k += 1) {
    const delta = step(x, k);
    if (delta == null) {
      break;
    }
    x = add(x, delta);
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
    (x) => scale(-learningRate, validateVector(grad(x))),
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
      return scale(-learningRate, validateVector(gradAtIndex(idx, x)));
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
      const grads = [...indices].map((i) => validateVector(gradAtIndex(i, x)));
      return scale(-learningRate, mean(grads));
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
      velocity = add(scale(beta, velocity), validateVector(grad(x)));
      return scale(-learningRate, velocity);
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
      velocity = add(scale(beta, velocity), validateVector(grad(lookAhead)));
      return scale(-learningRate, velocity);
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
      xPrev = x.slice();
      return delta;
    },
    options,
  );
}
