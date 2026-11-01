import { useMemo, useState, type ReactNode } from "react";
import {
  FINITE_SUM_OPTIMUM,
  finiteSumGrad,
  finiteSumGradAtIndex,
  finiteSumObjective,
  SAMPLE_CENTERS,
} from "../finiteSumQuadratic";
import {
  formatVector,
  gradientDescent,
  heavyBallMethod,
  minibatchSgd,
  momentum,
  nesterovAcceleratedGradient,
  stochasticGradientDescent,
  type Vector,
} from "../gradientMethod";
import { EquationWithWhere, type WhereClause } from "./EquationWithWhere";
import { MathBlock } from "./MathBlock";
import { OptimizationStepTable } from "./OptimizationStepTable";
import { TrajectoryChart } from "./TrajectoryChart";

const WHERE_X_K: WhereClause = {
  tex: "\\mathbf{x}_k",
  description: "parameter vector before the update at step k",
};
const WHERE_ETA: WhereClause = {
  tex: "\\eta",
  description: "learning rate (slider η)",
};
const WHERE_GRAD_F: WhereClause = {
  tex: "\\nabla f(\\mathbf{x}_k)",
  description: "full gradient of the demo objective f",
};
const WHERE_FI: WhereClause = {
  tex: "f_i(\\mathbf{x})",
  description: "i-th term in the finite sum (see objective above)",
};
const WHERE_BETA: WhereClause = {
  tex: "\\beta",
  description: "momentum coefficient in [0, 1) (slider β)",
};
const WHERE_VK: WhereClause = {
  tex: "\\mathbf{v}_k",
  description: "velocity vector carried across steps",
};

function Section({
  title,
  equations,
  children,
}: {
  title: string;
  equations?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
      <h2 className="text-lg font-medium text-white">{title}</h2>
      {equations ? <div className="mt-4 space-y-3">{equations}</div> : null}
      <div className={`space-y-4 ${equations ? "mt-5" : "mt-4"}`}>
        {children}
      </div>
    </section>
  );
}

function ResultStats({
  x,
  value,
  iterations,
}: {
  x: Vector;
  value: number;
  iterations: number;
}) {
  return (
    <p className="font-mono text-sm text-emerald-300">
      x = {formatVector(x)} · f(x) = {value.toExponential(3)} · {iterations} ·
      x* ≈ {formatVector([FINITE_SUM_OPTIMUM])}
    </p>
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
      η = {value.toFixed(2)}
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
      equations={
        <EquationWithWhere where={[WHERE_X_K, WHERE_ETA, WHERE_GRAD_F]}>
          <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)" />
        </EquationWithWhere>
      }
    >
      <LearningRateSlider value={lr} onChange={setLr} />
      <TrajectoryChart objective={objective1d} history={historyX} />
      <OptimizationStepTable
        steps={result.steps}
        caption="Gradient descent step-by-step iterations"
      />
      <ResultStats
        x={result.x}
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
      equations={
        <EquationWithWhere
          where={[
            WHERE_X_K,
            WHERE_ETA,
            {
              tex: "i_k",
              description: "index of one term sampled uniformly per step",
            },
            {
              tex: "\\nabla f_{i_k}(\\mathbf{x}_k)",
              description: "gradient of the sampled term f_{i_k} only",
            },
            WHERE_FI,
          ]}
        >
          <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f_{i_k}(\mathbf{x}_k)" />
        </EquationWithWhere>
      }
    >
      <LearningRateSlider value={lr} onChange={setLr} />
      <TrajectoryChart objective={objective1d} history={historyX} />
      <OptimizationStepTable
        steps={result.steps}
        caption="SGD step-by-step iterations"
      />
      <ResultStats
        x={result.x}
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
      equations={
        <EquationWithWhere
          where={[
            WHERE_X_K,
            WHERE_ETA,
            {
              tex: "B",
              description:
                "random mini-batch of term indices (resampled each step)",
            },
            {
              tex: "|B|",
              description: "batch size (slider |B|)",
            },
            {
              tex: "\\nabla f_i(\\mathbf{x}_k)",
              description: "gradient of term i at the current x_k",
            },
            WHERE_FI,
          ]}
        >
          <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \cdot \frac{1}{|B|}\sum_{i \in B} \nabla f_i(\mathbf{x}_k)" />
        </EquationWithWhere>
      }
    >
      <LearningRateSlider value={lr} onChange={setLr} />
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        |B| = {batchSize}
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
      <OptimizationStepTable
        steps={result.steps}
        caption="Mini-batch SGD step-by-step iterations"
      />
      <ResultStats
        x={result.x}
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
      equations={
        <EquationWithWhere
          where={[WHERE_X_K, WHERE_ETA, WHERE_BETA, WHERE_VK, WHERE_GRAD_F]}
        >
          <MathBlock tex="\mathbf{v}_{k+1} = \beta \mathbf{v}_k + \nabla f(\mathbf{x}_k)" />
          <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \mathbf{v}_{k+1}" />
        </EquationWithWhere>
      }
    >
      <LearningRateSlider value={lr} onChange={setLr} />
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        β = {beta.toFixed(2)}
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
      <OptimizationStepTable
        steps={result.steps}
        caption="Momentum step-by-step iterations"
      />
      <ResultStats
        x={result.x}
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
      equations={
        <EquationWithWhere
          where={[
            WHERE_X_K,
            WHERE_ETA,
            WHERE_BETA,
            WHERE_VK,
            {
              tex: "\\mathbf{y}_k",
              description: "look-ahead point where the gradient is evaluated",
            },
            {
              tex: "\\nabla f(\\mathbf{y}_k)",
              description: "gradient at y_k instead of at x_k",
            },
          ]}
        >
          <MathBlock tex="\mathbf{y}_k = \mathbf{x}_k - \beta \mathbf{v}_k" />
          <MathBlock tex="\mathbf{v}_{k+1} = \beta \mathbf{v}_k + \nabla f(\mathbf{y}_k)" />
          <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \mathbf{v}_{k+1}" />
        </EquationWithWhere>
      }
    >
      <LearningRateSlider value={lr} onChange={setLr} />
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        β = {beta.toFixed(2)}
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
      <OptimizationStepTable
        steps={result.steps}
        caption="Nesterov accelerated gradient step-by-step iterations"
      />
      <ResultStats
        x={result.x}
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
      equations={
        <EquationWithWhere
          where={[
            WHERE_X_K,
            WHERE_ETA,
            WHERE_BETA,
            WHERE_GRAD_F,
            {
              tex: "\\mathbf{x}_{k-1}",
              description: "previous iterate (omitted on the first step)",
            },
          ]}
        >
          <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k) + \beta(\mathbf{x}_k - \mathbf{x}_{k-1})" />
        </EquationWithWhere>
      }
    >
      <LearningRateSlider value={lr} onChange={setLr} />
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        β = {beta.toFixed(2)}
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
      <OptimizationStepTable
        steps={result.steps}
        caption="Heavy-ball step-by-step iterations"
      />
      <ResultStats
        x={result.x}
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
  const centers = SAMPLE_CENTERS.join(", ");
  return (
    <div className="mb-6">
      <EquationWithWhere
        where={[
          {
            tex: "m",
            description: "number of quadratic terms in the sum",
          },
          {
            tex: "f_i(x)",
            description: "single quadratic term ½(x − c_i)²",
          },
          {
            tex: "c_i",
            description: `fixed centers in the demo (${centers})`,
          },
          {
            tex: "\\nabla f(\\mathbf{x})",
            description: "mean of all per-term gradients ∇f_i(x)",
          },
          {
            tex: "\\mathbf{x}_0",
            description: "starting point [3] for every run on this page",
          },
        ]}
      >
        <MathBlock
          tex={`f(\\mathbf{x}) = \\tfrac{1}{m}\\sum_{i=1}^{m} \\tfrac{1}{2}(x - c_i)^2`}
        />
      </EquationWithWhere>
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
        <nav
          className="flex flex-col gap-1"
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
