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
  leastSquaresValue,
  rosenbrockJacobian,
  rosenbrockResidual,
  shiftedQuadratic,
  shiftedQuadraticGrad,
  shiftedQuadraticHess,
} from "../secondOrderProblems";
import { ContourPathChart } from "./ContourPathChart";
import { MathBlock } from "./MathBlock";

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
      description="Full Newton steps using the exact Hessian. On this quadratic, one step reaches the minimizer from any start."
      equations={
        <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{H}(\mathbf{x}_k)^{-1}\nabla f(\mathbf{x}_k)" />
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
      <ResultPanel result={result} />
      <ContourPathChart objective={shiftedQuadratic} path={result.path} />
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
      description="Newton direction with backtracking Armijo line search when a full step is too aggressive."
      equations={
        <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{d}_k,\quad \mathbf{d}_k = -\mathbf{H}_k^{-1}\nabla f_k" />
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
      <ResultPanel result={result} />
      <ContourPathChart objective={shiftedQuadratic} path={result.path} />
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
      description="Fletcher–Reeves nonlinear CG with Armijo line search along conjugate directions."
      equations={
        <MathBlock tex="\mathbf{d}_{k+1} = -\nabla f_{k+1} + \beta_k \mathbf{d}_k,\quad \beta_k^{\mathrm{FR}} = \frac{\|\nabla f_{k+1}\|^2}{\|\nabla f_k\|^2}" />
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
      <ResultPanel result={result} />
      <ContourPathChart objective={shiftedQuadratic} path={result.path} />
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
      description="Quasi-Newton BFGS builds an inverse Hessian approximation from gradient differences."
      equations={
        <MathBlock tex="\mathbf{B}_{k+1} = (\mathbf{I}-\rho_k \mathbf{s}_k \mathbf{y}_k^\top)\mathbf{B}_k(\mathbf{I}-\rho_k \mathbf{y}_k \mathbf{s}_k^\top)+\rho_k \mathbf{s}_k \mathbf{s}_k^\top" />
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
      <ResultPanel result={result} />
      <ContourPathChart objective={shiftedQuadratic} path={result.path} />
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
      description="Limited-memory BFGS stores only recent (s, y) pairs and applies the two-loop recursion."
      equations={
        <MathBlock tex="\mathbf{H}_k \approx \mathbf{H}_0 + \sum_{i=k-m}^{k-1} \cdots \quad \text{(two-loop recursion)}" />
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
      <ResultPanel result={result} />
      <ContourPathChart objective={shiftedQuadratic} path={result.path} />
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
      description="BFGS is the canonical quasi-Newton update; this demo calls the same BFGS implementation via the quasi-Newton alias."
      equations={
        <MathBlock tex="\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k,\quad \mathbf{p}_k = -\mathbf{B}_k \nabla f_k" />
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
      <ResultPanel result={result} />
      <ContourPathChart objective={shiftedQuadratic} path={result.path} />
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

function rosenbrockObjective(x: [number, number]): number {
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
      description="For least squares, approximates the Hessian with JᵀJ and linearizes the model each step."
      equations={
        <MathBlock tex="\min_{\mathbf{x}} \tfrac12 \|\mathbf{r}(\mathbf{x})\|^2,\quad \Delta \mathbf{x} = -(\mathbf{J}^\top \mathbf{J})^{-1}\mathbf{J}^\top \mathbf{r}" />
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
      <ResultPanel result={result} />
      <ContourPathChart
        objective={rosenbrockObjective}
        path={result.path}
        xRange={[-1, 2]}
        yRange={[-1, 2]}
      />
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
      description="Damps Gauss–Newton with λ on the diagonal of JᵀJ, adapting λ when trial steps reduce the residual."
      equations={
        <MathBlock tex="(\mathbf{J}^\top \mathbf{J} + \lambda \mathbf{I})\,\Delta \mathbf{x} = -\mathbf{J}^\top \mathbf{r}" />
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
      <ResultPanel result={result} />
      <ContourPathChart
        objective={rosenbrockObjective}
        path={result.path}
        xRange={[-1, 2]}
        yRange={[-1, 2]}
      />
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
    <div className="mb-6 space-y-3 text-slate-400">
      <p className="text-sm leading-relaxed">
        Demos mirror{" "}
        <code className="rounded bg-slate-800 px-1.5 py-0.5 text-emerald-300">
          trigger_agent_misc.optimization.second_order
        </code>{" "}
        on a shifted quadratic (min at (1, 2)) and Rosenbrock least squares.
      </p>
      <MathBlock tex="f(\mathbf{x}) = \tfrac12\bigl((x_1-1)^2 + (x_2-2)^2\bigr)" />
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
