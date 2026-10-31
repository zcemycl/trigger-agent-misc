import { useMemo, useState, type ReactNode } from "react";
import {
  armijoSufficientDecrease,
  backtrackingArmijo,
  backtrackingTrialAlphas,
  exactLineSearch,
  goldsteinConditions,
  lineSearchGoldstein,
  lineSearchStrongWolfe,
  lineSearchWolfe,
  strongWolfeCurvature,
  wolfeCurvature,
} from "../lineSearch";
import {
  QUADRATIC_PHI0,
  QUADRATIC_PHI_PRIME0,
  quadraticPhi,
  quadraticPhiPrime,
} from "../quadratic";
import { MathBlock } from "./MathBlock";
import { PhiChart } from "./PhiChart";

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

function StatusBadge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-medium ${
        ok
          ? "bg-emerald-500/20 text-emerald-300"
          : "bg-rose-500/20 text-rose-300"
      }`}
    >
      {label}: {ok ? "satisfied" : "not satisfied"}
    </span>
  );
}

function ExactLineSearchDemo() {
  const [bracketHi, setBracketHi] = useState(2);
  const result = useMemo(
    () => exactLineSearch(quadraticPhi, { a: 0, b: bracketHi }),
    [bracketHi],
  );

  return (
    <Section
      title="Exact line search"
      description="Golden-section search finds the minimizer of a unimodal ray on a bracket. Drag the upper bracket bound to see the search interval."
      equations={
        <>
          <MathBlock tex="\varphi(\alpha) = f(\mathbf{x} + \alpha \mathbf{p})" />
          <MathBlock tex="\alpha^* = \arg\min_{\alpha \in [a,\, b]} \varphi(\alpha)" />
        </>
      }
    >
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        Upper bracket b = {bracketHi.toFixed(2)}
        <input
          type="range"
          min={0.5}
          max={2.5}
          step={0.05}
          value={bracketHi}
          onChange={(e) => setBracketHi(Number(e.target.value))}
          className="accent-emerald-500"
        />
      </label>
      <PhiChart
        phi={quadraticPhi}
        bracket={{ a: 0, b: bracketHi }}
        markers={[
          {
            alpha: result.alpha,
            label: `α* ≈ ${result.alpha.toFixed(4)}`,
            color: "#34d399",
          },
        ]}
      />
      <p className="text-sm text-slate-300">
        Optimal step α* ={" "}
        <span className="font-mono text-emerald-300">
          {result.alpha.toFixed(6)}
        </span>
        , φ(α*) = {result.value.toExponential(3)}, iterations:{" "}
        {result.iterations}
      </p>
    </Section>
  );
}

function BacktrackingDemo() {
  const [alpha0, setAlpha0] = useState(1.5);
  const trials = useMemo(
    () => backtrackingTrialAlphas(quadraticPhi, QUADRATIC_PHI_PRIME0, alpha0),
    [alpha0],
  );
  const result = useMemo(
    () => backtrackingArmijo(quadraticPhi, QUADRATIC_PHI_PRIME0, { alpha0 }),
    [alpha0],
  );

  return (
    <Section
      title="Backtracking line search"
      description="Start with a trial step and shrink until a sufficient-decrease test passes."
      equations={
        <>
          <MathBlock tex="\alpha_0 > 0,\quad \rho \in (0,\, 1)" />
          <MathBlock tex="\alpha_{k+1} = \rho\,\alpha_k \quad \text{until Armijo holds}" />
        </>
      }
    >
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        Initial step α₀ = {alpha0.toFixed(2)}
        <input
          type="range"
          min={0.2}
          max={2.5}
          step={0.05}
          value={alpha0}
          onChange={(e) => setAlpha0(Number(e.target.value))}
          className="accent-emerald-500"
        />
      </label>
      <PhiChart
        phi={quadraticPhi}
        markers={trials.map((a, i) => ({
          alpha: a,
          label:
            i === trials.length - 1
              ? `accept α=${a.toFixed(3)}`
              : `try ${a.toFixed(2)}`,
          color: i === trials.length - 1 ? "#34d399" : "#f87171",
        }))}
        armijoLineAtAlpha={result.alpha}
        phi0={QUADRATIC_PHI0}
        phiPrime0={QUADRATIC_PHI_PRIME0}
      />
      <p className="text-sm text-slate-300">
        Accepted α = {result.alpha.toFixed(4)} after {result.iterations}{" "}
        backtrack{result.iterations === 1 ? "" : "s"}.
      </p>
    </Section>
  );
}

function ArmijoDemo() {
  const [alpha, setAlpha] = useState(0.5);
  const phiA = quadraticPhi(alpha);
  const holds = armijoSufficientDecrease(
    QUADRATIC_PHI0,
    phiA,
    alpha,
    QUADRATIC_PHI_PRIME0,
  );

  return (
    <Section
      title="Armijo condition"
      description="Armijo (sufficient decrease) requires the trial value to lie on or below a linear model at α = 0. Move α to compare φ(α) with the bound."
      equations={
        <>
          <MathBlock tex="0 < c_1 < 1" />
          <MathBlock tex="\varphi(\alpha) \le \varphi(0) + c_1\,\alpha\,\varphi'(0)" />
        </>
      }
    >
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        Trial step α = {alpha.toFixed(2)}
        <input
          type="range"
          min={0.05}
          max={2}
          step={0.05}
          value={alpha}
          onChange={(e) => setAlpha(Number(e.target.value))}
          className="accent-emerald-500"
        />
      </label>
      <PhiChart
        phi={quadraticPhi}
        armijoLineAtAlpha={alpha}
        phi0={QUADRATIC_PHI0}
        phiPrime0={QUADRATIC_PHI_PRIME0}
        markers={[
          {
            alpha,
            label: `φ(α)=${phiA.toFixed(3)}`,
            color: holds ? "#34d399" : "#f87171",
          },
        ]}
      />
      <StatusBadge ok={holds} label="Armijo" />
    </Section>
  );
}

function WolfeDemo() {
  const [alpha, setAlpha] = useState(1);
  const phiA = quadraticPhi(alpha);
  const phiPA = quadraticPhiPrime(alpha);
  const armijo = armijoSufficientDecrease(
    QUADRATIC_PHI0,
    phiA,
    alpha,
    QUADRATIC_PHI_PRIME0,
  );
  const curvature = wolfeCurvature(phiPA, QUADRATIC_PHI_PRIME0);
  const run = useMemo(
    () =>
      lineSearchWolfe(quadraticPhi, quadraticPhiPrime, QUADRATIC_PHI_PRIME0, {
        alpha0: 1,
      }),
    [],
  );

  return (
    <Section
      title="Wolfe conditions"
      description="Weak Wolfe line search accepts a step only when both sufficient decrease and a curvature inequality hold."
      equations={
        <>
          <MathBlock tex="0 < c_1 < c_2 < 1" />
          <MathBlock
            tex={`\\begin{aligned}
\\varphi(\\alpha) &\\le \\varphi(0) + c_1\\,\\alpha\\,\\varphi'(0) \\\\
\\varphi'(\\alpha) &\\ge c_2\\,\\varphi'(0)
\\end{aligned}`}
          />
        </>
      }
    >
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        Trial step α = {alpha.toFixed(2)}
        <input
          type="range"
          min={0.05}
          max={1.5}
          step={0.05}
          value={alpha}
          onChange={(e) => setAlpha(Number(e.target.value))}
          className="accent-emerald-500"
        />
      </label>
      <PhiChart
        phi={quadraticPhi}
        markers={[
          {
            alpha,
            label: "trial",
            color: armijo && curvature ? "#34d399" : "#fbbf24",
          },
          {
            alpha: run.alpha,
            label: "Wolfe LS",
            color: "#818cf8",
          },
        ]}
      />
      <div className="flex flex-wrap gap-2">
        <StatusBadge ok={armijo} label="Armijo" />
        <StatusBadge ok={curvature} label="Wolfe curvature" />
      </div>
      <p className="text-sm text-slate-400">
        Line search returned α = {run.alpha.toFixed(4)}, φ′(α) ={" "}
        {quadraticPhiPrime(run.alpha).toFixed(4)}.
      </p>
    </Section>
  );
}

function StrongWolfeDemo() {
  const [alpha, setAlpha] = useState(0.8);
  const phiPA = quadraticPhiPrime(alpha);
  const armijo = armijoSufficientDecrease(
    QUADRATIC_PHI0,
    quadraticPhi(alpha),
    alpha,
    QUADRATIC_PHI_PRIME0,
  );
  const strong = strongWolfeCurvature(phiPA, QUADRATIC_PHI_PRIME0);
  const run = useMemo(
    () =>
      lineSearchStrongWolfe(
        quadraticPhi,
        quadraticPhiPrime,
        QUADRATIC_PHI_PRIME0,
        { alpha0: 1 },
      ),
    [],
  );

  return (
    <Section
      title="Strong Wolfe conditions"
      description="Strong Wolfe replaces the weak curvature inequality with a bound on |φ′(α)|, which is tighter near a minimizer."
      equations={
        <>
          <MathBlock tex="0 < c_1 < c_2 < 1" />
          <MathBlock
            tex={`\\begin{aligned}
\\varphi(\\alpha) &\\le \\varphi(0) + c_1\\,\\alpha\\,\\varphi'(0) \\\\
|\\varphi'(\\alpha)| &\\le c_2\\,|\\varphi'(0)|
\\end{aligned}`}
          />
        </>
      }
    >
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        Trial step α = {alpha.toFixed(2)}
        <input
          type="range"
          min={0.05}
          max={1.5}
          step={0.05}
          value={alpha}
          onChange={(e) => setAlpha(Number(e.target.value))}
          className="accent-emerald-500"
        />
      </label>
      <PhiChart
        phi={quadraticPhi}
        markers={[
          { alpha, label: "trial", color: strong ? "#34d399" : "#f87171" },
          { alpha: run.alpha, label: "strong Wolfe LS", color: "#818cf8" },
        ]}
      />
      <div className="flex flex-wrap gap-2">
        <StatusBadge ok={armijo} label="Armijo" />
        <StatusBadge ok={strong} label="Strong Wolfe curvature" />
      </div>
      <p className="text-sm text-slate-400">
        |φ′(α)| = {Math.abs(phiPA).toFixed(4)}, bound c₂|φ′(0)| ={" "}
        {(0.9 * Math.abs(QUADRATIC_PHI_PRIME0)).toFixed(4)}.
      </p>
    </Section>
  );
}

function GoldsteinDemo() {
  const [alpha, setAlpha] = useState(0.35);
  const c = 0.1;
  const phiA = quadraticPhi(alpha);
  const lower = QUADRATIC_PHI0 + (1 - c) * alpha * QUADRATIC_PHI_PRIME0;
  const upper = QUADRATIC_PHI0 + c * alpha * QUADRATIC_PHI_PRIME0;
  const holds = goldsteinConditions(
    QUADRATIC_PHI0,
    phiA,
    alpha,
    QUADRATIC_PHI_PRIME0,
    c,
  );
  const run = useMemo(
    () =>
      lineSearchGoldstein(quadraticPhi, QUADRATIC_PHI_PRIME0, { alpha0: 1 }),
    [],
  );

  return (
    <Section
      title="Goldstein conditions"
      description="Goldstein bounds φ(α) between two linear models; the yellow segment on the chart is that interval at your trial α."
      equations={
        <>
          <MathBlock tex="0 < c < \tfrac{1}{2}" />
          <MathBlock
            tex={`\\varphi(0) + (1-c)\\,\\alpha\\,\\varphi'(0)
\\;\\le\\; \\varphi(\\alpha)
\\;\\le\\; \\varphi(0) + c\\,\\alpha\\,\\varphi'(0)`}
          />
        </>
      }
    >
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        Trial step α = {alpha.toFixed(2)}
        <input
          type="range"
          min={0.05}
          max={1.2}
          step={0.05}
          value={alpha}
          onChange={(e) => setAlpha(Number(e.target.value))}
          className="accent-emerald-500"
        />
      </label>
      <PhiChart
        phi={quadraticPhi}
        band={{
          alpha,
          yLower: lower,
          yUpper: upper,
          label: "Goldstein band",
        }}
        markers={[
          {
            alpha,
            label: holds ? "in band" : "outside",
            color: holds ? "#34d399" : "#f87171",
          },
          { alpha: run.alpha, label: "Goldstein LS", color: "#818cf8" },
        ]}
      />
      <StatusBadge ok={holds} label="Goldstein" />
    </Section>
  );
}

export const LINE_SEARCH_CHAPTER = "Line search";

export const LINE_SEARCH_PAGES = [
  { id: "exact", title: "Exact line search", Demo: ExactLineSearchDemo },
  {
    id: "backtracking",
    title: "Backtracking line search",
    Demo: BacktrackingDemo,
  },
  { id: "armijo", title: "Armijo condition", Demo: ArmijoDemo },
  { id: "wolfe", title: "Wolfe conditions", Demo: WolfeDemo },
  {
    id: "strong-wolfe",
    title: "Strong Wolfe conditions",
    Demo: StrongWolfeDemo,
  },
  { id: "goldstein", title: "Goldstein conditions", Demo: GoldsteinDemo },
] as const;

export type LineSearchPageId = (typeof LINE_SEARCH_PAGES)[number]["id"];

function ConceptIntro() {
  return (
    <div className="mb-6 space-y-3 text-slate-400">
      <p className="text-sm leading-relaxed">
        Interactive demos use the same quadratic ray as the Python evals in{" "}
        <code className="rounded bg-slate-800 px-1.5 py-0.5 text-emerald-300">
          trigger_agent_misc.optimization
        </code>
        .
      </p>
      <MathBlock
        tex={`\\varphi(\\alpha) = (1-\\alpha)^2, \\qquad \\varphi'(0) = -2`}
      />
    </div>
  );
}

export function LineSearchConcepts({
  initialPageId = "exact",
}: {
  initialPageId?: LineSearchPageId;
}) {
  const [activeId, setActiveId] = useState<LineSearchPageId>(initialPageId);
  const activePage =
    LINE_SEARCH_PAGES.find((p) => p.id === activeId) ?? LINE_SEARCH_PAGES[0];
  const ActiveDemo = activePage.Demo;

  return (
    <div className="flex flex-col gap-8 lg:flex-row lg:items-start">
      <aside className="w-full shrink-0 lg:sticky lg:top-6 lg:w-56">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          {LINE_SEARCH_CHAPTER}
        </p>
        <nav
          className="mt-3 flex flex-col gap-1"
          aria-label={`${LINE_SEARCH_CHAPTER} concepts`}
        >
          {LINE_SEARCH_PAGES.map((page) => {
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
