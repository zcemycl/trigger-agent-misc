import { useMemo, useState, type ReactNode } from "react";
import {
  FINITE_SUM_OPTIMUM,
  finiteSumGrad,
  finiteSumGradAtIndex,
  finiteSumObjective,
  SAMPLE_CENTERS,
} from "../finiteSumQuadratic";
import {
  gradientDescent,
  heavyBallMethod,
  minibatchSgd,
  momentum,
  nesterovAcceleratedGradient,
  stochasticGradientDescent,
} from "../gradientMethod";
import { MathBlock } from "./MathBlock";
import { TrajectoryChart } from "./TrajectoryChart";

function Section({
  title,
  description,
  equations,
  children,
}: {
  title: string;
  description: string;
  equations?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
      <h2 className="text-lg font-medium text-white">{title}</h2>
      <p className="mt-2 text-sm leading-relaxed text-slate-400">
        {description}
      </p>
      {equations ? <div className="mt-4 space-y-3">{equations}</div> : null}
      <div className="mt-5 space-y-4">{children}</div>
    </section>
  );
}

function ResultStats({
  x,
  value,
  iterations,
}: {
  x: number;
  value: number;
  iterations: number;
}) {
  return (
    <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
      <div>
        <dt className="text-slate-500">x</dt>
        <dd className="font-mono text-emerald-300">{x.toFixed(4)}</dd>
      </div>
      <div>
        <dt className="text-slate-500">f(x)</dt>
        <dd className="font-mono text-emerald-300">{value.toExponential(3)}</dd>
      </div>
      <div>
        <dt className="text-slate-500">Steps</dt>
        <dd className="font-mono text-emerald-300">{iterations}</dd>
      </div>
      <div>
        <dt className="text-slate-500">Target</dt>
        <dd className="font-mono text-slate-300">
          {FINITE_SUM_OPTIMUM.toFixed(4)}
        </dd>
      </div>
    </dl>
  );
}

function LearningRateSlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1 text-sm text-slate-300">
      Learning rate = {value.toFixed(2)}
      <input
        type="range"
        min={0.01}
        max={0.4}
        step={0.01}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="accent-emerald-500"
      />
    </label>
  );
}

const objective1d = (x: number) => finiteSumObjective([x]);
const x0 = [3.0];
const numSamples = SAMPLE_CENTERS.length;

function GradientDescentDemo() {
  const [lr, setLr] = useState(0.2);
  const result = useMemo(
    () =>
      gradientDescent(x0, finiteSumObjective, finiteSumGrad, {
        learningRate: lr,
        maxIter: 500,
        tol: 1e-8,
      }),
    [lr],
  );
  const historyX = result.history.map((v) => v[0]);

  return (
    <Section
      title="Gradient descent (GD)"
      description="Follow the negative full gradient each step on a smooth finite-sum quadratic."
      equations={
        <>
          <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)" />
          <MathBlock tex="f(x) = \frac{1}{m}\sum_{i=1}^{m} \tfrac{1}{2}(x - c_i)^2" />
        </>
      }
    >
      <LearningRateSlider value={lr} onChange={setLr} />
      <TrajectoryChart objective={objective1d} history={historyX} />
      <ResultStats
        x={result.x[0]}
        value={result.value}
        iterations={result.iterations}
      />
    </Section>
  );
}

function SgdDemo() {
  const [lr, setLr] = useState(0.05);
  const result = useMemo(
    () =>
      stochasticGradientDescent(
        x0,
        finiteSumObjective,
        finiteSumGradAtIndex,
        numSamples,
        { learningRate: lr, maxIter: 800, tol: 1e-8, seed: 0 },
      ),
    [lr],
  );
  const historyX = result.history.map((v) => v[0]);

  return (
    <Section
      title="Stochastic gradient descent (SGD)"
      description="Estimate the gradient from one randomly sampled term per iteration."
      equations={
        <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f_{i_k}(\mathbf{x}_k)" />
      }
    >
      <LearningRateSlider value={lr} onChange={setLr} />
      <TrajectoryChart objective={objective1d} history={historyX} />
      <ResultStats
        x={result.x[0]}
        value={result.value}
        iterations={result.iterations}
      />
    </Section>
  );
}

function MinibatchDemo() {
  const [lr, setLr] = useState(0.1);
  const [batchSize, setBatchSize] = useState(2);
  const result = useMemo(
    () =>
      minibatchSgd(x0, finiteSumObjective, finiteSumGradAtIndex, numSamples, {
        learningRate: lr,
        batchSize,
        maxIter: 400,
        tol: 1e-8,
        seed: 1,
      }),
    [lr, batchSize],
  );
  const historyX = result.history.map((v) => v[0]);

  return (
    <Section
      title="Mini-batch SGD"
      description="Average gradients over a small batch of samples each step."
      equations={
        <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \cdot \frac{1}{|B|}\sum_{i \in B} \nabla f_i(\mathbf{x}_k)" />
      }
    >
      <LearningRateSlider value={lr} onChange={setLr} />
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        Batch size = {batchSize}
        <input
          type="range"
          min={1}
          max={numSamples}
          step={1}
          value={batchSize}
          onChange={(e) => setBatchSize(Number(e.target.value))}
          className="accent-emerald-500"
        />
      </label>
      <TrajectoryChart objective={objective1d} history={historyX} />
      <ResultStats
        x={result.x[0]}
        value={result.value}
        iterations={result.iterations}
      />
    </Section>
  );
}

function MomentumDemo() {
  const [lr, setLr] = useState(0.05);
  const [beta, setBeta] = useState(0.9);
  const result = useMemo(
    () =>
      momentum(x0, finiteSumObjective, finiteSumGrad, {
        learningRate: lr,
        beta,
        maxIter: 400,
        tol: 1e-8,
      }),
    [lr, beta],
  );
  const historyX = result.history.map((v) => v[0]);

  return (
    <Section
      title="Momentum"
      description="Accumulate historical gradients in a velocity vector to damp oscillations."
      equations={
        <>
          <MathBlock tex="\mathbf{v}_{k+1} = \beta \mathbf{v}_k + \nabla f(\mathbf{x}_k)" />
          <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \mathbf{v}_{k+1}" />
        </>
      }
    >
      <LearningRateSlider value={lr} onChange={setLr} />
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        Momentum β = {beta.toFixed(2)}
        <input
          type="range"
          min={0}
          max={0.99}
          step={0.01}
          value={beta}
          onChange={(e) => setBeta(Number(e.target.value))}
          className="accent-emerald-500"
        />
      </label>
      <TrajectoryChart objective={objective1d} history={historyX} />
      <ResultStats
        x={result.x[0]}
        value={result.value}
        iterations={result.iterations}
      />
    </Section>
  );
}

function NesterovDemo() {
  const [lr, setLr] = useState(0.05);
  const [beta, setBeta] = useState(0.9);
  const result = useMemo(
    () =>
      nesterovAcceleratedGradient(x0, finiteSumObjective, finiteSumGrad, {
        learningRate: lr,
        beta,
        maxIter: 400,
        tol: 1e-8,
      }),
    [lr, beta],
  );
  const historyX = result.history.map((v) => v[0]);

  return (
    <Section
      title="Nesterov accelerated gradient"
      description="Look ahead along velocity before evaluating the gradient."
      equations={
        <>
          <MathBlock tex="\mathbf{y}_k = \mathbf{x}_k - \beta \mathbf{v}_k" />
          <MathBlock tex="\mathbf{v}_{k+1} = \beta \mathbf{v}_k + \nabla f(\mathbf{y}_k)" />
        </>
      }
    >
      <LearningRateSlider value={lr} onChange={setLr} />
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        Momentum β = {beta.toFixed(2)}
        <input
          type="range"
          min={0}
          max={0.99}
          step={0.01}
          value={beta}
          onChange={(e) => setBeta(Number(e.target.value))}
          className="accent-emerald-500"
        />
      </label>
      <TrajectoryChart objective={objective1d} history={historyX} />
      <ResultStats
        x={result.x[0]}
        value={result.value}
        iterations={result.iterations}
      />
    </Section>
  );
}

function HeavyBallDemo() {
  const [lr, setLr] = useState(0.05);
  const [beta, setBeta] = useState(0.85);
  const result = useMemo(
    () =>
      heavyBallMethod(x0, finiteSumObjective, finiteSumGrad, {
        learningRate: lr,
        beta,
        maxIter: 400,
        tol: 1e-8,
      }),
    [lr, beta],
  );
  const historyX = result.history.map((v) => v[0]);

  return (
    <Section
      title="Heavy-ball method"
      description="Use momentum on the position update (Polyak heavy-ball)."
      equations={
        <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k) + \beta(\mathbf{x}_k - \mathbf{x}_{k-1})" />
      }
    >
      <LearningRateSlider value={lr} onChange={setLr} />
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        Heavy-ball β = {beta.toFixed(2)}
        <input
          type="range"
          min={0}
          max={0.99}
          step={0.01}
          value={beta}
          onChange={(e) => setBeta(Number(e.target.value))}
          className="accent-emerald-500"
        />
      </label>
      <TrajectoryChart objective={objective1d} history={historyX} />
      <ResultStats
        x={result.x[0]}
        value={result.value}
        iterations={result.iterations}
      />
    </Section>
  );
}

export const GRADIENT_METHOD_CHAPTER = "Gradient methods";

export const GRADIENT_METHOD_PAGES = [
  { id: "gd", title: "Gradient descent (GD)", Demo: GradientDescentDemo },
  { id: "sgd", title: "Stochastic gradient descent (SGD)", Demo: SgdDemo },
  { id: "minibatch", title: "Mini-batch SGD", Demo: MinibatchDemo },
  { id: "momentum", title: "Momentum", Demo: MomentumDemo },
  {
    id: "nesterov",
    title: "Nesterov accelerated gradient",
    Demo: NesterovDemo,
  },
  { id: "heavy-ball", title: "Heavy-ball method", Demo: HeavyBallDemo },
] as const;

export type GradientMethodPageId = (typeof GRADIENT_METHOD_PAGES)[number]["id"];

function ConceptIntro() {
  return (
    <div className="mb-6 space-y-3 text-slate-400">
      <p className="text-sm leading-relaxed">
        Demos minimize the same finite-sum quadratic as{" "}
        <code className="rounded bg-slate-800 px-1.5 py-0.5 text-emerald-300">
          evals/gradient_method_eval.py
        </code>{" "}
        and the Python module{" "}
        <code className="rounded bg-slate-800 px-1.5 py-0.5 text-emerald-300">
          trigger_agent_misc.optimization.gradient_method
        </code>
        .
      </p>
      <MathBlock
        tex={`c_i \\in \\{${SAMPLE_CENTERS.join(", ")}\\}, \\quad x_0 = 3`}
      />
    </div>
  );
}

export function GradientMethodConcepts({
  initialPageId = "gd",
}: {
  initialPageId?: GradientMethodPageId;
}) {
  const [activeId, setActiveId] = useState<GradientMethodPageId>(initialPageId);
  const activePage =
    GRADIENT_METHOD_PAGES.find((p) => p.id === activeId) ??
    GRADIENT_METHOD_PAGES[0];
  const ActiveDemo = activePage.Demo;

  return (
    <div className="flex flex-col gap-8 lg:flex-row lg:items-start">
      <aside className="w-full shrink-0 lg:sticky lg:top-6 lg:w-56">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          {GRADIENT_METHOD_CHAPTER}
        </p>
        <nav
          className="mt-3 flex flex-col gap-1"
          aria-label={`${GRADIENT_METHOD_CHAPTER} concepts`}
        >
          {GRADIENT_METHOD_PAGES.map((page) => {
            const isActive = page.id === activeId;
            return (
              <button
                key={page.id}
                type="button"
                onClick={() => setActiveId(page.id)}
                aria-current={isActive ? "page" : undefined}
                className={`rounded-lg px-3 py-2 text-left text-sm transition ${
                  isActive
                    ? "bg-emerald-500/15 font-medium text-emerald-200"
                    : "text-slate-400 hover:bg-slate-800/80 hover:text-slate-200"
                }`}
              >
                {page.title}
              </button>
            );
          })}
        </nav>
      </aside>
      <div className="min-w-0 flex-1">
        <ConceptIntro />
        <ActiveDemo />
      </div>
    </div>
  );
}
