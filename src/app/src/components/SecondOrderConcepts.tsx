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
  type Vec,
} from "../secondOrder";
import {
  leastSquaresValue,
  rosenbrockJacobian,
  rosenbrockResidual,
  shiftedQuadratic,
  shiftedQuadraticGrad,
  shiftedQuadraticHess,
} from "../secondOrderProblems";
import {
  ConceptChapterSidebar,
  type ConceptViewMode,
} from "./ConceptChapterSidebar";
import {
  ConceptPageOverview,
  type ConceptOverview,
} from "./ConceptPageOverview";
import { ContourPathChart } from "./ContourPathChart";
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
      <p>
        <span className="text-slate-500">x:</span> ({result.x[0]!.toFixed(4)},{" "}
        {result.x[1]!.toFixed(4)})
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

function NewtonDemo() {
  const [x0, setX0] = useState(0);
  const [y0, setY0] = useState(0);
  const result = useMemo(
    () =>
      newtonsMethod(
        shiftedQuadratic,
        shiftedQuadraticGrad,
        shiftedQuadraticHess,
        [x0, y0],
      ),
    [x0, y0],
  );
  return (
    <Section
      title="Newton's method"
      equations={
        <EquationWithWhere where={[WHERE_X_K, WHERE_HESS, WHERE_GRAD_F]}>
          <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{H}(\mathbf{x}_k)^{-1}\nabla f(\mathbf{x}_k)" />
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
      <ContourPathChart objective={shiftedQuadratic} path={result.path} />
      <OptimizationStepTable
        steps={result.steps}
        caption="Newton's method step-by-step iterations"
      />
      <ResultPanel result={result} />
    </Section>
  );
}

function DampedNewtonDemo() {
  const [x0, setX0] = useState(3);
  const [y0, setY0] = useState(-2);
  const result = useMemo(
    () =>
      dampedNewtonMethod(
        shiftedQuadratic,
        shiftedQuadraticGrad,
        shiftedQuadraticHess,
        [x0, y0],
      ),
    [x0, y0],
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
      <ContourPathChart objective={shiftedQuadratic} path={result.path} />
      <OptimizationStepTable
        steps={result.steps}
        caption="Damped Newton step-by-step iterations"
      />
      <ResultPanel result={result} />
    </Section>
  );
}

function ConjugateGradientDemo() {
  const [x0, setX0] = useState(0);
  const [y0, setY0] = useState(0);
  const result = useMemo(
    () => conjugateGradient(shiftedQuadratic, shiftedQuadraticGrad, [x0, y0]),
    [x0, y0],
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
      <ContourPathChart objective={shiftedQuadratic} path={result.path} />
      <OptimizationStepTable
        steps={result.steps}
        caption="Conjugate gradient step-by-step iterations"
      />
      <ResultPanel result={result} />
    </Section>
  );
}

function BfgsDemo() {
  const [x0, setX0] = useState(0);
  const [y0, setY0] = useState(0);
  const result = useMemo(
    () => bfgs(shiftedQuadratic, shiftedQuadraticGrad, [x0, y0]),
    [x0, y0],
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
      <ContourPathChart objective={shiftedQuadratic} path={result.path} />
      <OptimizationStepTable
        steps={result.steps}
        caption="BFGS step-by-step iterations"
      />
      <ResultPanel result={result} />
    </Section>
  );
}

function LBfgsDemo() {
  const [x0, setX0] = useState(0);
  const [y0, setY0] = useState(0);
  const result = useMemo(
    () => lbfgs(shiftedQuadratic, shiftedQuadraticGrad, [x0, y0]),
    [x0, y0],
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
      <ContourPathChart objective={shiftedQuadratic} path={result.path} />
      <OptimizationStepTable
        steps={result.steps}
        caption="L-BFGS step-by-step iterations"
      />
      <ResultPanel result={result} />
    </Section>
  );
}

function QuasiNewtonDemo() {
  const [x0, setX0] = useState(2);
  const [y0, setY0] = useState(2);
  const result = useMemo(
    () => quasiNewtonBfgs(shiftedQuadratic, shiftedQuadraticGrad, [x0, y0]),
    [x0, y0],
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
      <ContourPathChart objective={shiftedQuadratic} path={result.path} />
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

function rosenbrockObjective(x: Vec): number {
  return leastSquaresValue(rosenbrockResidual, x);
}

function GaussNewtonDemo() {
  const [x0, setX0] = useState(0);
  const [y0, setY0] = useState(0);
  const result = useMemo(
    () => gaussNewton(rosenbrockResidual, rosenbrockJacobian, [x0, y0]),
    [x0, y0],
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
      <ContourPathChart
        objective={rosenbrockObjective}
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
  const [x0, setX0] = useState(0);
  const [y0, setY0] = useState(0);
  const result = useMemo(
    () => levenbergMarquardt(rosenbrockResidual, rosenbrockJacobian, [x0, y0]),
    [x0, y0],
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
      <ContourPathChart
        objective={rosenbrockObjective}
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
  purpose:
    "Minimize a smooth function using a local quadratic model built from the Hessian.",
  explanation:
    "Each step solves the Newton system H_k Δx = −∇f_k and jumps to x_k + Δx. For a convex quadratic with exact Hessians, one step reaches the minimizer; on general functions it is a fast local method when H_k is well conditioned.",
  references: [
    "Isaac Newton (1669), root finding via quadratics",
    "Nocedal & Wright, Numerical Optimization, Ch. 2 & 6",
    "Boyd & Vandenberghe, Convex Optimization, §9.5",
  ],
};

const DAMPED_NEWTON_OVERVIEW: ConceptOverview = {
  purpose:
    "Retain Newton directions while guaranteeing progress via a line search on the step length.",
  explanation:
    "Pure Newton can overshoot when the quadratic model is poor or H is indefinite. Backtracking Armijo shortens α_k along d_k = −H_k⁻¹∇f_k until sufficient decrease holds, improving global convergence behavior.",
  references: [
    "Dennis & Schnabel (1983), globalized Newton methods",
    "Nocedal & Wright, Ch. 6.3 (line search Newton)",
  ],
};

const CG_OVERVIEW: ConceptOverview = {
  purpose:
    "Solve large smooth minimization problems without forming or storing the Hessian.",
  explanation:
    "Search directions are conjugate with respect to the Hessian on quadratics, so n steps solve an n-dimensional quadratic in exact arithmetic. The nonlinear demo uses Fletcher–Reeves β and restarts for general f.",
  references: [
    "Hestenes & Stiefel (1952), conjugate gradients for linear systems",
    "Fletcher & Reeves (1964), nonlinear CG",
    "Nocedal & Wright, Ch. 5–6",
  ],
};

const QUASI_NEWTON_OVERVIEW: ConceptOverview = {
  purpose:
    "Achieve Newton-like convergence using only gradient information to build curvature.",
  explanation:
    "A matrix B_k (or its inverse) is updated each step so that secant conditions approximate the Hessian. The demo follows the standard quasi-Newton step p_k = −B_k ∇f_k with line search.",
  references: [
    "Davidon (1959), variable metric methods",
    "Nocedal & Wright, Ch. 6 (quasi-Newton methods)",
  ],
};

const BFGS_OVERVIEW: ConceptOverview = {
  purpose:
    "Maintain a positive-definite inverse Hessian approximation with a stable rank-two update.",
  explanation:
    "BFGS enforces the secant equation on B_k and is the workhorse for medium-scale unconstrained optimization. The update uses step s_k and gradient change y_k; the demo tracks B_k explicitly in 2D.",
  references: [
    "Broyden (1970); Fletcher (1970); Goldfarb (1970); Shanno (1970)",
    "Nocedal & Wright, §6.1–6.2",
  ],
};

const LBFGS_OVERVIEW: ConceptOverview = {
  purpose:
    "Scale quasi-Newton ideas to high dimensions by storing only recent (s, y) pairs.",
  explanation:
    "L-BFGS applies the inverse BFGS update implicitly via a two-loop recursion over the last m steps. Memory is O(mn) instead of O(n²), which is essential for large machine-learning models.",
  references: [
    "Liu & Nocedal (1989), limited memory BFGS",
    "Nocedal & Wright, §7.2",
  ],
};

const GAUSS_NEWTON_OVERVIEW: ConceptOverview = {
  purpose:
    "Minimize nonlinear least-squares objectives by linearizing the residual each iteration.",
  explanation:
    "Gauss–Newton solves (JᵀJ) Δx = −Jᵀr using the Jacobian of residuals r(x). It equals Newton on ½‖r‖² when second-order terms in the Hessian are dropped, and is standard for fitting and inverse problems.",
  references: [
    "Gauss (1809), least squares",
    "Nocedal & Wright, §10.3",
    "Björck, Numerical Methods for Least Squares Problems",
  ],
};

const LM_OVERVIEW: ConceptOverview = {
  purpose:
    "Stabilize Gauss–Newton when JᵀJ is singular or the linearization is too aggressive.",
  explanation:
    "Levenberg–Marquardt adds λI to JᵀJ, interpolating between gradient descent (large λ) and Gauss–Newton (small λ). λ adapts from trial-step acceptance on the nonlinear least-squares objective.",
  references: [
    "Levenberg (1944); Marquardt (1963)",
    "Nocedal & Wright, §10.3–10.4",
    "Moré (1978), trust-region LM analysis",
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

export function SecondOrderConcepts({
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
