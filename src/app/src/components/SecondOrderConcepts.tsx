import { useMemo, useState, type ReactNode } from "react";
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
  DemoNoiseControls,
  DemoNoiseProvider,
  useRosenbrockLeastSquares,
  useShiftedQuadratic3DProblem,
  useShiftedQuadraticProblem,
} from "./DemoNoise";
import {
  ConceptPageOverview,
  type ConceptOverview,
} from "./ConceptPageOverview";
import { ObjectiveLandscapeView } from "./ObjectiveLandscapeView";
import { Path3DChart } from "./Path3DChart";
import { EquationWithWhere, type WhereClause } from "./EquationWithWhere";
import { MathBlock } from "./MathBlock";
import { OptimizationStepTable } from "./OptimizationStepTable";

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

function QuadraticStartSliders({
  x0,
  y0,
  onChange,
}: {
  x0: number;
  y0: number;
  onChange: (x: number, y: number) => void;
}) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row">
      <label className="flex flex-1 flex-col gap-1 text-sm text-slate-300">
        Start x₁ = {x0.toFixed(2)}
        <input
          type="range"
          min={-1}
          max={4}
          step={0.1}
          value={x0}
          onChange={(e) => onChange(Number(e.target.value), y0)}
          className="accent-emerald-500"
        />
      </label>
      <label className="flex flex-1 flex-col gap-1 text-sm text-slate-300">
        Start x₂ = {y0.toFixed(2)}
        <input
          type="range"
          min={-1}
          max={4}
          step={0.1}
          value={y0}
          onChange={(e) => onChange(x0, Number(e.target.value))}
          className="accent-emerald-500"
        />
      </label>
    </div>
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
  const problem2d = useShiftedQuadraticProblem();
  const problem3d = useShiftedQuadratic3DProblem();
  const [dim, setDim] = useState<2 | 3>(2);
  const [start2, setStart2] = useState({ x0: 0, y0: 0 });
  const [start3, setStart3] = useState<[number, number, number]>([0, 0, 0]);
  const { objective, gradient, hessian } = dim === 2 ? problem2d : problem3d;
  const result = useMemo(() => {
    const x0 = dim === 2 ? [start2.x0, start2.y0] : [...start3];
    return newtonsMethod(objective, gradient, hessian, x0);
  }, [
    objective,
    gradient,
    hessian,
    dim,
    start2.x0,
    start2.y0,
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
      <ProblemDimensionToggle value={dim} onChange={setDim} />
      {dim === 2 ? (
        <QuadraticStartSliders
          x0={start2.x0}
          y0={start2.y0}
          onChange={(a, b) => setStart2({ x0: a, y0: b })}
        />
      ) : (
        <QuadraticStartSliders3D values={start3} onChange={setStart3} />
      )}
      {dim === 2 ? (
        <ObjectiveLandscapeView objective={objective} path={result.path} />
      ) : (
        <Path3DChart path={result.path} />
      )}
      <OptimizationStepTable
        steps={result.steps}
        caption="Newton's method step-by-step iterations"
      />
      <ResultPanel result={result} />
    </Section>
  );
}

function DampedNewtonDemo() {
  const { objective, gradient, hessian } = useShiftedQuadraticProblem();
  const [x0, setX0] = useState(3);
  const [y0, setY0] = useState(-2);
  const result = useMemo(
    () => dampedNewtonMethod(objective, gradient, hessian, [x0, y0]),
    [objective, gradient, hessian, x0, y0],
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
      <QuadraticStartSliders
        x0={x0}
        y0={y0}
        onChange={(a, b) => {
          setX0(a);
          setY0(b);
        }}
      />
      <ObjectiveLandscapeView objective={objective} path={result.path} />
      <OptimizationStepTable
        steps={result.steps}
        caption="Damped Newton step-by-step iterations"
      />
      <ResultPanel result={result} />
    </Section>
  );
}

function ConjugateGradientDemo() {
  const { objective, gradient } = useShiftedQuadraticProblem();
  const [x0, setX0] = useState(0);
  const [y0, setY0] = useState(0);
  const result = useMemo(
    () => conjugateGradient(objective, gradient, [x0, y0]),
    [objective, gradient, x0, y0],
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
      <QuadraticStartSliders
        x0={x0}
        y0={y0}
        onChange={(a, b) => {
          setX0(a);
          setY0(b);
        }}
      />
      <ObjectiveLandscapeView objective={objective} path={result.path} />
      <OptimizationStepTable
        steps={result.steps}
        caption="Conjugate gradient step-by-step iterations"
      />
      <ResultPanel result={result} />
    </Section>
  );
}

function BfgsDemo() {
  const { objective, gradient } = useShiftedQuadraticProblem();
  const [x0, setX0] = useState(0);
  const [y0, setY0] = useState(0);
  const result = useMemo(
    () => bfgs(objective, gradient, [x0, y0]),
    [objective, gradient, x0, y0],
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
      <QuadraticStartSliders
        x0={x0}
        y0={y0}
        onChange={(a, b) => {
          setX0(a);
          setY0(b);
        }}
      />
      <ObjectiveLandscapeView objective={objective} path={result.path} />
      <OptimizationStepTable
        steps={result.steps}
        caption="BFGS step-by-step iterations"
      />
      <ResultPanel result={result} />
    </Section>
  );
}

function LBfgsDemo() {
  const { objective, gradient } = useShiftedQuadraticProblem();
  const [x0, setX0] = useState(0);
  const [y0, setY0] = useState(0);
  const result = useMemo(
    () => lbfgs(objective, gradient, [x0, y0]),
    [objective, gradient, x0, y0],
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
      <QuadraticStartSliders
        x0={x0}
        y0={y0}
        onChange={(a, b) => {
          setX0(a);
          setY0(b);
        }}
      />
      <ObjectiveLandscapeView objective={objective} path={result.path} />
      <OptimizationStepTable
        steps={result.steps}
        caption="L-BFGS step-by-step iterations"
      />
      <ResultPanel result={result} />
    </Section>
  );
}

function QuasiNewtonDemo() {
  const { objective, gradient } = useShiftedQuadraticProblem();
  const [x0, setX0] = useState(2);
  const [y0, setY0] = useState(2);
  const result = useMemo(
    () => quasiNewtonBfgs(objective, gradient, [x0, y0]),
    [objective, gradient, x0, y0],
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
      <QuadraticStartSliders
        x0={x0}
        y0={y0}
        onChange={(a, b) => {
          setX0(a);
          setY0(b);
        }}
      />
      <ObjectiveLandscapeView objective={objective} path={result.path} />
      <OptimizationStepTable
        steps={result.steps}
        caption="Quasi-Newton (BFGS) step-by-step iterations"
      />
      <ResultPanel result={result} />
    </Section>
  );
}

function LeastSquaresSliders({
  x0,
  y0,
  onChange,
}: {
  x0: number;
  y0: number;
  onChange: (x: number, y: number) => void;
}) {
  return <QuadraticStartSliders x0={x0} y0={y0} onChange={onChange} />;
}

function GaussNewtonDemo() {
  const { residual, jacobian, objective } = useRosenbrockLeastSquares();
  const [x0, setX0] = useState(0);
  const [y0, setY0] = useState(0);
  const result = useMemo(
    () => gaussNewton(residual, jacobian, [x0, y0]),
    [residual, jacobian, x0, y0],
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
      <LeastSquaresSliders
        x0={x0}
        y0={y0}
        onChange={(a, b) => {
          setX0(a);
          setY0(b);
        }}
      />
      <ObjectiveLandscapeView
        objective={objective}
        path={result.path}
        xRange={[-1, 2]}
        yRange={[-1, 2]}
      />
      <OptimizationStepTable
        steps={result.steps}
        caption="Gauss–Newton step-by-step iterations"
      />
      <ResultPanel result={result} />
    </Section>
  );
}

function LevenbergMarquardtDemo() {
  const { residual, jacobian, objective } = useRosenbrockLeastSquares();
  const [x0, setX0] = useState(0);
  const [y0, setY0] = useState(0);
  const result = useMemo(
    () => levenbergMarquardt(residual, jacobian, [x0, y0]),
    [residual, jacobian, x0, y0],
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
      <LeastSquaresSliders
        x0={x0}
        y0={y0}
        onChange={(a, b) => {
          setX0(a);
          setY0(b);
        }}
      />
      <ObjectiveLandscapeView
        objective={objective}
        path={result.path}
        xRange={[-1, 2]}
        yRange={[-1, 2]}
      />
      <OptimizationStepTable
        steps={result.steps}
        caption="Levenberg–Marquardt step-by-step iterations"
      />
      <ResultPanel result={result} />
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
              "smooth objective for Newton, CG, and quasi-Newton demos",
          },
          {
            tex: "(1, 2)",
            description: "unique minimizer of the shifted quadratic",
          },
        ]}
      >
        <MathBlock tex="f(\mathbf{x}) = \tfrac12\bigl((x_1-1)^2 + (x_2-2)^2\bigr)" />
      </EquationWithWhere>
      <EquationWithWhere
        where={[
          WHERE_X0,
          WHERE_R,
          WHERE_J,
          {
            tex: "\\tfrac12\\|\\mathbf{r}\\|^2",
            description:
              "nonlinear least-squares objective for Gauss–Newton and LM",
          },
        ]}
      >
        <MathBlock tex="\min_{\mathbf{x}} \tfrac12 \|\mathbf{r}(\mathbf{x})\|^2" />
        <MathBlock tex="\mathbf{r}(\mathbf{x}) = \begin{pmatrix}x_1-1\\10(x_2-x_1^2)\end{pmatrix}" />
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
