import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  bfgs,
  conjugateGradient,
  dampedNewtonMethod,
  gaussNewton,
  lbfgs,
  levenbergMarquardt,
  newtonsMethod,
  quasiNewtonBfgs,
  type OptimizeResult,
} from "../secondOrder";
import {
  ConceptChapterSidebar,
  type ConceptViewMode,
} from "./ConceptChapterSidebar";
import {
  LEAST_SQUARES_EXAMPLES,
  SMOOTH_OPTIMIZATION_EXAMPLES,
  type LeastSquaresExampleId,
  type SmoothOptimizationExampleId,
} from "../secondOrderExampleCatalog";
import {
  DemoNoiseControls,
  DemoNoiseProvider,
  useShiftedQuadratic3DProblem,
} from "./DemoNoise";
import {
  LeastSquaresExampleControls,
  SmoothExampleControls,
  useLeastSquaresExampleProblem,
  useLeastSquaresExampleStart,
  useSmoothExampleProblem,
  useSmoothExampleStart,
} from "./secondOrderDemoHelpers";
import {
  ConceptPageOverview,
  type ConceptOverview,
} from "./ConceptPageOverview";
import { ObjectiveLandscapeView } from "./ObjectiveLandscapeView";
import { Path3DChart } from "./Path3DChart";
import { EquationWithWhere, type WhereClause } from "./EquationWithWhere";
import { MathBlock } from "./MathBlock";
import { OptimizationStepTable } from "./OptimizationStepTable";
import { pathHighlightRangeForStep } from "./pathHighlight";

const WHERE_X_K: WhereClause = {
  tex: "\\mathbf{x}_k",
  description: "parameter vector before the update at step k",
};
const WHERE_X0: WhereClause = {
  tex: "\\mathbf{x}_0",
  description: "starting point (sliders x₁, x₂)",
};
const WHERE_GRAD_F: WhereClause = {
  tex: "\\nabla f(\\mathbf{x}_k)",
  description: "gradient of the demo objective f",
};
const WHERE_HESS: WhereClause = {
  tex: "\\mathbf{H}(\\mathbf{x}_k)",
  description: "Hessian of f (exact in Newton demos)",
};
const WHERE_ALPHA: WhereClause = {
  tex: "\\alpha_k",
  description: "Armijo backtracking step length along the search direction",
};
const WHERE_R: WhereClause = {
  tex: "\\mathbf{r}(\\mathbf{x})",
  description: "residual vector in the Rosenbrock least-squares demo",
};
const WHERE_J: WhereClause = {
  tex: "\\mathbf{J}(\\mathbf{x})",
  description: "Jacobian ∂r/∂x",
};
const WHERE_LAM: WhereClause = {
  tex: "\\lambda",
  description: "Levenberg–Marquardt damping on the diagonal of JᵀJ",
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

function ResultPanel({ result }: { result: OptimizeResult }) {
  return (
    <div className="grid gap-2 text-sm text-slate-300 sm:grid-cols-2">
      <p>
        <span className="text-slate-500">Iterations:</span> {result.iterations}
      </p>
      <p>
        <span className="text-slate-500">Converged:</span>{" "}
        {result.converged ? "yes" : "no"}
      </p>
      <p className="sm:col-span-2">
        <span className="text-slate-500">x:</span> (
        {result.x.map((xi) => xi.toFixed(4)).join(", ")})
      </p>
      <p>
        <span className="text-slate-500">f(x):</span>{" "}
        {result.value.toExponential(3)}
      </p>
    </div>
  );
}

function SecondOrderOptimizationTrace({
  result,
  objective,
  caption,
  xRange,
  yRange,
  chart = "landscape",
  TableComponent = OptimizationStepTable,
}: {
  result: OptimizeResult;
  objective?: (x: number[]) => number;
  caption: string;
  xRange?: [number, number];
  yRange?: [number, number];
  chart?: "landscape" | "path3d";
  TableComponent?: typeof OptimizationStepTable;
}) {
  const [highlightedStep, setHighlightedStep] = useState<number | null>(null);
  const highlightPathRange = useMemo(
    () => pathHighlightRangeForStep(result.steps, highlightedStep),
    [highlightedStep, result.steps],
  );

  return (
    <>
      {chart === "path3d" ? (
        <Path3DChart
          path={result.path}
          highlightPathRange={highlightPathRange}
        />
      ) : (
        <ObjectiveLandscapeView
          objective={objective!}
          path={result.path}
          xRange={xRange}
          yRange={yRange}
          highlightPathRange={highlightPathRange}
        />
      )}
      <TableComponent
        steps={result.steps}
        caption={caption}
        highlightedStep={highlightedStep}
        onHighlightedStepChange={setHighlightedStep}
      />
      <ResultPanel result={result} />
    </>
  );
}

function QuadraticStartSliders3D({
  values,
  onChange,
}: {
  values: [number, number, number];
  onChange: (next: [number, number, number]) => void;
}) {
  const labels = ["x₁", "x₂", "x₃"] as const;
  return (
    <div className="flex flex-col gap-3">
      {labels.map((label, i) => (
        <label
          key={label}
          className="flex flex-col gap-1 text-sm text-slate-300"
        >
          Start {label} = {values[i]!.toFixed(2)}
          <input
            type="range"
            min={-1}
            max={4}
            step={0.1}
            value={values[i]}
            onChange={(e) => {
              const next = [...values] as [number, number, number];
              next[i] = Number(e.target.value);
              onChange(next);
            }}
            className="accent-emerald-500"
          />
        </label>
      ))}
    </div>
  );
}

function ProblemDimensionToggle({
  value,
  onChange,
}: {
  value: 2 | 3;
  onChange: (d: 2 | 3) => void;
}) {
  return (
    <div
      className="inline-flex rounded-lg border border-slate-800 p-0.5 text-sm"
      role="group"
      aria-label="Problem dimension"
    >
      {([2, 3] as const).map((d) => (
        <button
          key={d}
          type="button"
          aria-pressed={value === d}
          onClick={() => onChange(d)}
          className={`rounded-md px-3 py-1 transition ${
            value === d
              ? "bg-emerald-500/20 font-medium text-emerald-200"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          {d}D
        </button>
      ))}
    </div>
  );
}

function NewtonDemo() {
  const [exampleId, setExampleId] =
    useState<SmoothOptimizationExampleId>("shifted-quadratic");
  const { start, setStart, meta } = useSmoothExampleStart(exampleId);
  const problem2d = useSmoothExampleProblem(exampleId);
  const problem3d = useShiftedQuadratic3DProblem();
  const [dim, setDim] = useState<2 | 3>(2);
  const [start3, setStart3] = useState<[number, number, number]>([0, 0, 0]);
  const use3d = dim === 3 && exampleId === "shifted-quadratic";
  useEffect(() => {
    if (exampleId !== "shifted-quadratic") {
      setDim(2);
    }
  }, [exampleId]);
  const { objective, gradient, hessian } = use3d ? problem3d : problem2d;
  const result = useMemo(() => {
    const x0 = use3d ? [...start3] : [...start];
    return newtonsMethod(objective, gradient, hessian, x0);
  }, [
    objective,
    gradient,
    hessian,
    use3d,
    start,
    start3[0],
    start3[1],
    start3[2],
  ]);
  return (
    <Section
      title="Newton's method"
      equations={
        <EquationWithWhere where={[WHERE_X_K, WHERE_HESS, WHERE_GRAD_F]}>
          <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{H}(\mathbf{x}_k)^{-1}\nabla f(\mathbf{x}_k)" />
        </EquationWithWhere>
      }
    >
      <SmoothExampleControls
        exampleId={exampleId}
        onExampleIdChange={setExampleId}
        examples={SMOOTH_OPTIMIZATION_EXAMPLES}
        start={start}
        onStartChange={(x, y) => setStart([x, y])}
        hideStartSliders={use3d}
      >
        {exampleId === "shifted-quadratic" ? (
          <ProblemDimensionToggle value={dim} onChange={setDim} />
        ) : null}
        {use3d ? (
          <QuadraticStartSliders3D values={start3} onChange={setStart3} />
        ) : null}
      </SmoothExampleControls>
      <SecondOrderOptimizationTrace
        result={result}
        objective={objective}
        xRange={meta.xRange}
        yRange={meta.yRange}
        chart={use3d ? "path3d" : "landscape"}
        caption="Newton's method step-by-step iterations"
        TableComponent={OptimizationStepTable}
      />
    </Section>
  );
}

function DampedNewtonDemo() {
  const [exampleId, setExampleId] =
    useState<SmoothOptimizationExampleId>("shifted-quadratic");
  const { start, setStart, meta } = useSmoothExampleStart(exampleId);
  const { objective, gradient, hessian } = useSmoothExampleProblem(exampleId);
  const result = useMemo(
    () => dampedNewtonMethod(objective, gradient, hessian, [...start]),
    [objective, gradient, hessian, start],
  );
  return (
    <Section
      title="Damped Newton method"
      equations={
        <EquationWithWhere
          where={[
            WHERE_X_K,
            WHERE_ALPHA,
            {
              tex: "\\mathbf{d}_k",
              description: "Newton direction −H_k⁻¹∇f_k",
            },
            WHERE_HESS,
            WHERE_GRAD_F,
          ]}
        >
          <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{d}_k,\quad \mathbf{d}_k = -\mathbf{H}_k^{-1}\nabla f_k" />
        </EquationWithWhere>
      }
    >
      <SmoothExampleControls
        exampleId={exampleId}
        onExampleIdChange={setExampleId}
        examples={SMOOTH_OPTIMIZATION_EXAMPLES}
        start={start}
        onStartChange={(x, y) => setStart([x, y])}
      />
      <SecondOrderOptimizationTrace
        result={result}
        objective={objective}
        xRange={meta.xRange}
        yRange={meta.yRange}
        caption="Damped Newton step-by-step iterations"
        TableComponent={OptimizationStepTable}
      />
    </Section>
  );
}

function ConjugateGradientDemo() {
  const [exampleId, setExampleId] =
    useState<SmoothOptimizationExampleId>("shifted-quadratic");
  const { start, setStart, meta } = useSmoothExampleStart(exampleId);
  const { objective, gradient } = useSmoothExampleProblem(exampleId);
  const result = useMemo(
    () => conjugateGradient(objective, gradient, [...start]),
    [objective, gradient, start],
  );
  return (
    <Section
      title="Conjugate gradient"
      equations={
        <EquationWithWhere
          where={[
            WHERE_X_K,
            WHERE_ALPHA,
            WHERE_GRAD_F,
            {
              tex: "\\beta_k",
              description:
                "Fletcher–Reeves coefficient from successive gradients",
            },
            { tex: "\\mathbf{d}_k", description: "conjugate search direction" },
          ]}
        >
          <MathBlock tex="\mathbf{d}_{k+1} = -\nabla f_{k+1} + \beta_k \mathbf{d}_k,\quad \beta_k^{\mathrm{FR}} = \frac{\|\nabla f_{k+1}\|^2}{\|\nabla f_k\|^2}" />
        </EquationWithWhere>
      }
    >
      <SmoothExampleControls
        exampleId={exampleId}
        onExampleIdChange={setExampleId}
        examples={SMOOTH_OPTIMIZATION_EXAMPLES}
        start={start}
        onStartChange={(x, y) => setStart([x, y])}
      />
      <SecondOrderOptimizationTrace
        result={result}
        objective={objective}
        xRange={meta.xRange}
        yRange={meta.yRange}
        caption="Conjugate gradient step-by-step iterations"
        TableComponent={OptimizationStepTable}
      />
    </Section>
  );
}

function BfgsDemo() {
  const [exampleId, setExampleId] =
    useState<SmoothOptimizationExampleId>("shifted-quadratic");
  const { start, setStart, meta } = useSmoothExampleStart(exampleId);
  const { objective, gradient } = useSmoothExampleProblem(exampleId);
  const result = useMemo(
    () => bfgs(objective, gradient, [...start]),
    [objective, gradient, start],
  );
  return (
    <Section
      title="BFGS"
      equations={
        <EquationWithWhere
          where={[
            WHERE_X_K,
            WHERE_ALPHA,
            WHERE_GRAD_F,
            {
              tex: "\\mathbf{B}_k",
              description: "inverse Hessian approximation updated each step",
            },
            {
              tex: "\\mathbf{s}_k, \\mathbf{y}_k",
              description: "step and gradient difference for the BFGS update",
            },
          ]}
        >
          <MathBlock tex="\mathbf{B}_{k+1} = (\mathbf{I}-\rho_k \mathbf{s}_k \mathbf{y}_k^\top)\mathbf{B}_k(\mathbf{I}-\rho_k \mathbf{y}_k \mathbf{s}_k^\top)+\rho_k \mathbf{s}_k \mathbf{s}_k^\top" />
        </EquationWithWhere>
      }
    >
      <SmoothExampleControls
        exampleId={exampleId}
        onExampleIdChange={setExampleId}
        examples={SMOOTH_OPTIMIZATION_EXAMPLES}
        start={start}
        onStartChange={(x, y) => setStart([x, y])}
      />
      <SecondOrderOptimizationTrace
        result={result}
        objective={objective}
        xRange={meta.xRange}
        yRange={meta.yRange}
        caption="BFGS step-by-step iterations"
        TableComponent={OptimizationStepTable}
      />
    </Section>
  );
}

function LBfgsDemo() {
  const [exampleId, setExampleId] =
    useState<SmoothOptimizationExampleId>("shifted-quadratic");
  const { start, setStart, meta } = useSmoothExampleStart(exampleId);
  const { objective, gradient } = useSmoothExampleProblem(exampleId);
  const result = useMemo(
    () => lbfgs(objective, gradient, [...start]),
    [objective, gradient, start],
  );
  return (
    <Section
      title="L-BFGS"
      equations={
        <EquationWithWhere
          where={[
            WHERE_X_K,
            WHERE_ALPHA,
            WHERE_GRAD_F,
            {
              tex: "m",
              description: "memory window of stored (s, y) pairs (default 10)",
            },
          ]}
        >
          <MathBlock tex="\mathbf{H}_k \approx \mathbf{H}_0 + \sum_{i=k-m}^{k-1} \cdots \quad \text{(two-loop recursion)}" />
        </EquationWithWhere>
      }
    >
      <SmoothExampleControls
        exampleId={exampleId}
        onExampleIdChange={setExampleId}
        examples={SMOOTH_OPTIMIZATION_EXAMPLES}
        start={start}
        onStartChange={(x, y) => setStart([x, y])}
      />
      <SecondOrderOptimizationTrace
        result={result}
        objective={objective}
        xRange={meta.xRange}
        yRange={meta.yRange}
        caption="L-BFGS step-by-step iterations"
        TableComponent={OptimizationStepTable}
      />
    </Section>
  );
}

function QuasiNewtonDemo() {
  const [exampleId, setExampleId] =
    useState<SmoothOptimizationExampleId>("shifted-quadratic");
  const { start, setStart, meta } = useSmoothExampleStart(exampleId);
  const { objective, gradient } = useSmoothExampleProblem(exampleId);
  const result = useMemo(
    () => quasiNewtonBfgs(objective, gradient, [...start]),
    [objective, gradient, start],
  );
  return (
    <Section
      title="Quasi-Newton methods"
      equations={
        <EquationWithWhere
          where={[
            WHERE_X_K,
            WHERE_ALPHA,
            WHERE_GRAD_F,
            {
              tex: "\\mathbf{p}_k",
              description: "quasi-Newton search direction −B_k∇f_k (BFGS demo)",
            },
          ]}
        >
          <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k,\quad \mathbf{p}_k = -\mathbf{B}_k \nabla f_k" />
        </EquationWithWhere>
      }
    >
      <SmoothExampleControls
        exampleId={exampleId}
        onExampleIdChange={setExampleId}
        examples={SMOOTH_OPTIMIZATION_EXAMPLES}
        start={start}
        onStartChange={(x, y) => setStart([x, y])}
      />
      <SecondOrderOptimizationTrace
        result={result}
        objective={objective}
        xRange={meta.xRange}
        yRange={meta.yRange}
        caption="Quasi-Newton (BFGS) step-by-step iterations"
        TableComponent={OptimizationStepTable}
      />
    </Section>
  );
}

function GaussNewtonDemo() {
  const [exampleId, setExampleId] =
    useState<LeastSquaresExampleId>("rosenbrock");
  const { start, setStart, meta } = useLeastSquaresExampleStart(exampleId);
  const { residual, jacobian, objective } =
    useLeastSquaresExampleProblem(exampleId);
  const result = useMemo(
    () => gaussNewton(residual, jacobian, [...start]),
    [residual, jacobian, start],
  );
  return (
    <Section
      title="Gauss–Newton"
      equations={
        <EquationWithWhere where={[WHERE_X_K, WHERE_R, WHERE_J]}>
          <MathBlock tex="\min_{\mathbf{x}} \tfrac12 \|\mathbf{r}(\mathbf{x})\|^2,\quad \Delta \mathbf{x} = -(\mathbf{J}^\top \mathbf{J})^{-1}\mathbf{J}^\top \mathbf{r}" />
        </EquationWithWhere>
      }
    >
      <LeastSquaresExampleControls
        exampleId={exampleId}
        onExampleIdChange={setExampleId}
        examples={LEAST_SQUARES_EXAMPLES}
        start={start}
        onStartChange={(x, y) => setStart([x, y])}
      />
      <SecondOrderOptimizationTrace
        result={result}
        objective={objective}
        xRange={meta.xRange}
        yRange={meta.yRange}
        caption="Gauss–Newton step-by-step iterations"
        TableComponent={OptimizationStepTable}
      />
    </Section>
  );
}

function LevenbergMarquardtDemo() {
  const [exampleId, setExampleId] =
    useState<LeastSquaresExampleId>("rosenbrock");
  const { start, setStart, meta } = useLeastSquaresExampleStart(exampleId);
  const { residual, jacobian, objective } =
    useLeastSquaresExampleProblem(exampleId);
  const result = useMemo(
    () => levenbergMarquardt(residual, jacobian, [...start]),
    [residual, jacobian, start],
  );
  return (
    <Section
      title="Levenberg–Marquardt"
      equations={
        <EquationWithWhere where={[WHERE_X_K, WHERE_R, WHERE_J, WHERE_LAM]}>
          <MathBlock tex="(\mathbf{J}^\top \mathbf{J} + \lambda \mathbf{I})\,\Delta \mathbf{x} = -\mathbf{J}^\top \mathbf{r}" />
        </EquationWithWhere>
      }
    >
      <LeastSquaresExampleControls
        exampleId={exampleId}
        onExampleIdChange={setExampleId}
        examples={LEAST_SQUARES_EXAMPLES}
        start={start}
        onStartChange={(x, y) => setStart([x, y])}
      />
      <SecondOrderOptimizationTrace
        result={result}
        objective={objective}
        xRange={meta.xRange}
        yRange={meta.yRange}
        caption="Levenberg–Marquardt step-by-step iterations"
        TableComponent={OptimizationStepTable}
      />
    </Section>
  );
}

export const SECOND_ORDER_CHAPTER = "Second-order optimization";

const NEWTON_OVERVIEW: ConceptOverview = {
  purposeTex:
    "\\min_{\\mathbf{x}} f(\\mathbf{x}) \\quad\\text{via local quadratic models}",
  explanationTex:
    "\\mathbf{x}_{k+1} = \\mathbf{x}_k - \\mathbf{H}(\\mathbf{x}_k)^{-1}\\nabla f(\\mathbf{x}_k)",
  references: [
    {
      href: "https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization",
      label: "Newton (1669), quadratic models for roots",
    },
    {
      href: "https://link.springer.com/book/10.1007/978-0-387-40065-5",
      label: "Nocedal & Wright, Ch. 2 & 6",
    },
    {
      href: "https://web.stanford.edu/~boyd/cvxbook/",
      label: "Boyd & Vandenberghe, §9.5",
    },
  ],
};

const DAMPED_NEWTON_OVERVIEW: ConceptOverview = {
  purposeTex: "\\text{Globalize Newton with a line search on the step}",
  explanationTex:
    "\\mathbf{x}_{k+1} = \\mathbf{x}_k + \\alpha_k \\mathbf{d}_k,\\quad \\mathbf{d}_k = -\\mathbf{H}_k^{-1}\\nabla f_k",
  references: [
    {
      href: "https://doi.org/10.1137/0706016",
      label: "Dennis & Schnabel (1983), globalized Newton",
    },
    {
      href: "https://link.springer.com/book/10.1007/978-0-387-40065-5",
      label: "Nocedal & Wright, Ch. 6.3",
    },
  ],
};

const CG_OVERVIEW: ConceptOverview = {
  purposeTex:
    "\\min f(\\mathbf{x}) \\quad\\text{without explicit Hessian storage}",
  explanationTex:
    "\\mathbf{d}_{k+1} = -\\nabla f_{k+1} + \\beta_k \\mathbf{d}_k,\\quad \\beta_k^{\\mathrm{FR}} = \\frac{\\|\\nabla f_{k+1}\\|^2}{\\|\\nabla f_k\\|^2}",
  references: [
    {
      href: "https://doi.org/10.1090/S0025-5718-1952-0062154-5",
      label: "Hestenes & Stiefel (1952), conjugate gradients",
    },
    {
      href: "https://doi.org/10.1093/comjnl/7.2.149",
      label: "Fletcher & Reeves (1964), nonlinear CG",
    },
    {
      href: "https://link.springer.com/book/10.1007/978-0-387-40065-5",
      label: "Nocedal & Wright, Ch. 5–6",
    },
  ],
};

const QUASI_NEWTON_OVERVIEW: ConceptOverview = {
  purposeTex:
    "\\min f(\\mathbf{x}) \\quad\\text{with secant Hessian approximations}",
  explanationTex:
    "\\mathbf{x}_{k+1} = \\mathbf{x}_k + \\alpha_k \\mathbf{p}_k,\\quad \\mathbf{p}_k = -\\mathbf{B}_k \\nabla f_k",
  references: [
    {
      href: "https://doi.org/10.1137/0706016",
      label: "Davidon (1959), variable metric methods",
    },
    {
      href: "https://link.springer.com/book/10.1007/978-0-387-40065-5",
      label: "Nocedal & Wright, Ch. 6",
    },
  ],
};

const BFGS_OVERVIEW: ConceptOverview = {
  purposeTex:
    "\\text{Maintain } \\mathbf{B}_k \\approx \\mathbf{H}^{-1} \\text{ with rank-two updates}",
  explanationTex:
    "\\mathbf{B}_{k+1} = (\\mathbf{I}-\\rho_k \\mathbf{s}_k \\mathbf{y}_k^\\top)\\mathbf{B}_k(\\mathbf{I}-\\rho_k \\mathbf{y}_k \\mathbf{s}_k^\\top)+\\rho_k \\mathbf{s}_k \\mathbf{s}_k^\\top",
  references: [
    {
      href: "https://doi.org/10.1093/imamat/6.1.76",
      label: "Broyden–Fletcher–Goldfarb–Shanno (1970)",
    },
    {
      href: "https://link.springer.com/book/10.1007/978-0-387-40065-5",
      label: "Nocedal & Wright, §6.1–6.2",
    },
  ],
};

const LBFGS_OVERVIEW: ConceptOverview = {
  purposeTex:
    "\\text{Implicit BFGS from the last } m \\text{ pairs } (\\mathbf{s}_i,\\mathbf{y}_i)",
  explanationTex:
    "\\mathbf{p}_k = -\\mathbf{H}_k \\nabla f_k \\quad\\text{(two-loop recursion, memory } O(mn)\\text{)}",
  references: [
    {
      href: "https://doi.org/10.1093/imamat/45.1-3.503",
      label: "Liu & Nocedal (1989), L-BFGS",
    },
    {
      href: "https://link.springer.com/book/10.1007/978-0-387-40065-5",
      label: "Nocedal & Wright, §7.2",
    },
  ],
};

const GAUSS_NEWTON_OVERVIEW: ConceptOverview = {
  purposeTex: "\\min_{\\mathbf{x}} \\tfrac12 \\|\\mathbf{r}(\\mathbf{x})\\|^2",
  explanationTex:
    "\\Delta\\mathbf{x} = -(\\mathbf{J}^\\top\\mathbf{J})^{-1}\\mathbf{J}^\\top\\mathbf{r}",
  references: [
    {
      href: "https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm",
      label: "Gauss (1809), least squares",
    },
    {
      href: "https://link.springer.com/book/10.1007/978-0-387-40065-5",
      label: "Nocedal & Wright, §10.3",
    },
    {
      href: "https://www.siam.org/books/ot51/",
      label: "Björck, Numerical Methods for Least Squares",
    },
  ],
};

const LM_OVERVIEW: ConceptOverview = {
  purposeTex:
    "\\min \\tfrac12\\|\\mathbf{r}(\\mathbf{x})\\|^2 \\quad\\text{with Levenberg–Marquardt damping}",
  explanationTex:
    "(\\mathbf{J}^\\top\\mathbf{J} + \\lambda \\mathbf{I})\\,\\Delta\\mathbf{x} = -\\mathbf{J}^\\top\\mathbf{r}",
  references: [
    {
      href: "https://doi.org/10.1137/0706016",
      label: "Levenberg (1944); Marquardt (1963)",
    },
    {
      href: "https://link.springer.com/book/10.1007/978-0-387-40065-5",
      label: "Nocedal & Wright, §10.3–10.4",
    },
    {
      href: "https://doi.org/10.1137/0706016",
      label: "Moré (1978), trust-region LM analysis",
    },
  ],
};

export const SECOND_ORDER_PAGES = [
  {
    id: "newton",
    title: "Newton's method",
    Demo: NewtonDemo,
    overview: NEWTON_OVERVIEW,
  },
  {
    id: "damped-newton",
    title: "Damped Newton method",
    Demo: DampedNewtonDemo,
    overview: DAMPED_NEWTON_OVERVIEW,
  },
  {
    id: "cg",
    title: "Conjugate gradient",
    Demo: ConjugateGradientDemo,
    overview: CG_OVERVIEW,
  },
  {
    id: "quasi-newton",
    title: "Quasi-Newton methods",
    Demo: QuasiNewtonDemo,
    overview: QUASI_NEWTON_OVERVIEW,
  },
  { id: "bfgs", title: "BFGS", Demo: BfgsDemo, overview: BFGS_OVERVIEW },
  { id: "lbfgs", title: "L-BFGS", Demo: LBfgsDemo, overview: LBFGS_OVERVIEW },
  {
    id: "gauss-newton",
    title: "Gauss–Newton",
    Demo: GaussNewtonDemo,
    overview: GAUSS_NEWTON_OVERVIEW,
  },
  {
    id: "levenberg-marquardt",
    title: "Levenberg–Marquardt",
    Demo: LevenbergMarquardtDemo,
    overview: LM_OVERVIEW,
  },
] as const;

export type SecondOrderPageId = (typeof SECOND_ORDER_PAGES)[number]["id"];

function ConceptIntro() {
  return (
    <div className="mb-6 space-y-4">
      <DemoNoiseControls />
      <EquationWithWhere
        where={[
          WHERE_X0,
          {
            tex: "f(\\mathbf{x})",
            description:
              "selectable smooth objectives (quadratic, Booth, Rosenbrock, …)",
          },
        ]}
      >
        <MathBlock tex="\min_{\mathbf{x}} f(\mathbf{x})" />
      </EquationWithWhere>
      <EquationWithWhere
        where={[
          WHERE_X0,
          WHERE_R,
          WHERE_J,
          {
            tex: "\\tfrac12\\|\\mathbf{r}\\|^2",
            description:
              "selectable nonlinear least-squares models for Gauss–Newton and LM",
          },
        ]}
      >
        <MathBlock tex="\min_{\mathbf{x}} \tfrac12 \|\mathbf{r}(\mathbf{x})\|^2" />
      </EquationWithWhere>
    </div>
  );
}

function SecondOrderConceptsContent({
  initialPageId = "newton",
}: {
  initialPageId?: SecondOrderPageId;
}) {
  const [activeId, setActiveId] = useState<SecondOrderPageId>(initialPageId);
  const [viewMode, setViewMode] = useState<ConceptViewMode>("demo");
  const activePage =
    SECOND_ORDER_PAGES.find((p) => p.id === activeId) ?? SECOND_ORDER_PAGES[0];
  const ActiveDemo = activePage.Demo;

  return (
    <div className="flex flex-col gap-8 lg:flex-row lg:items-start">
      <ConceptChapterSidebar
        chapter={SECOND_ORDER_CHAPTER}
        sectionLabel={SECOND_ORDER_CHAPTER}
        pages={SECOND_ORDER_PAGES}
        activeId={activeId}
        onActiveIdChange={setActiveId}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
      />
      <div className="min-w-0 flex-1">
        {viewMode === "overview" ? (
          <ConceptPageOverview
            title={activePage.title}
            overview={activePage.overview}
          />
        ) : (
          <>
            <ConceptIntro />
            <ActiveDemo />
          </>
        )}
      </div>
    </div>
  );
}

export function SecondOrderConcepts(
  props: { initialPageId?: SecondOrderPageId } = {},
) {
  return (
    <DemoNoiseProvider>
      <SecondOrderConceptsContent {...props} />
    </DemoNoiseProvider>
  );
}
