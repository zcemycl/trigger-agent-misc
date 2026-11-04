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

export const SECOND_ORDER_PAGES = [
  { id: "newton", title: "Newton's method", Demo: NewtonDemo },
  {
    id: "damped-newton",
    title: "Damped Newton method",
    Demo: DampedNewtonDemo,
  },
  { id: "cg", title: "Conjugate gradient", Demo: ConjugateGradientDemo },
  { id: "quasi-newton", title: "Quasi-Newton methods", Demo: QuasiNewtonDemo },
  { id: "bfgs", title: "BFGS", Demo: BfgsDemo },
  { id: "lbfgs", title: "L-BFGS", Demo: LBfgsDemo },
  { id: "gauss-newton", title: "Gauss–Newton", Demo: GaussNewtonDemo },
  {
    id: "levenberg-marquardt",
    title: "Levenberg–Marquardt",
    Demo: LevenbergMarquardtDemo,
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
  const activePage =
    SECOND_ORDER_PAGES.find((p) => p.id === activeId) ?? SECOND_ORDER_PAGES[0];
  const ActiveDemo = activePage.Demo;

  return (
    <div className="flex flex-col gap-8 lg:flex-row lg:items-start">
      <aside className="w-full shrink-0 lg:sticky lg:top-6 lg:w-56">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          {SECOND_ORDER_CHAPTER}
        </p>
        <nav
          className="mt-3 flex flex-col gap-1"
          aria-label={`${SECOND_ORDER_CHAPTER} concepts`}
        >
          {SECOND_ORDER_PAGES.map((page) => {
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
