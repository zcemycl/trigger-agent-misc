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
import {
  ConceptChapterSidebar,
  type ConceptViewMode,
} from "./ConceptChapterSidebar";
import {
  ConceptPageOverview,
  type ConceptOverview,
} from "./ConceptPageOverview";
import { EquationWithWhere } from "./EquationWithWhere";
import { MathBlock } from "./MathBlock";
import { PhiChart } from "./PhiChart";

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
      equations={
        <EquationWithWhere
          where={[
            {
              tex: "\\varphi(\\alpha)",
              description: "objective along the search ray (see chapter intro)",
            },
            {
              tex: "\\alpha",
              description: "step size along direction p from current x",
            },
            {
              tex: "[a, b]",
              description: "bracket interval (slider b sets the upper bound)",
            },
            {
              tex: "\\alpha^*",
              description: "minimizer of φ on the bracket",
            },
          ]}
        >
          <MathBlock tex="\varphi(\alpha) = f(\mathbf{x} + \alpha \mathbf{p})" />
          <MathBlock tex="\alpha^* = \arg\min_{\alpha \in [a,\, b]} \varphi(\alpha)" />
        </EquationWithWhere>
      }
    >
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        b = {bracketHi.toFixed(2)}
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
      <p className="font-mono text-sm text-emerald-300">
        α* = {result.alpha.toFixed(6)} · φ(α*) = {result.value.toExponential(3)}{" "}
        · {result.iterations}
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
      equations={
        <>
          <MathBlock tex="\alpha_0 > 0,\quad \rho \in (0,\, 1)" />
          <MathBlock tex="\alpha_{k+1} = \rho\,\alpha_k \quad \text{until Armijo holds}" />
        </>
      }
    >
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        α₀ = {alpha0.toFixed(2)}
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
      <p className="font-mono text-sm text-emerald-300">
        α = {result.alpha.toFixed(4)} · {result.iterations}
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
      equations={
        <EquationWithWhere
          where={[
            {
              tex: "c_1",
              description: "Armijo constant (fixed 0.1 in the chart bound)",
            },
            {
              tex: "\\varphi(\\alpha)",
              description: "trial value along the ray at step α",
            },
            {
              tex: "\\varphi'(0)",
              description: "slope of φ at α = 0 (chapter intro)",
            },
          ]}
        >
          <MathBlock tex="0 < c_1 < 1" />
          <MathBlock tex="\varphi(\alpha) \le \varphi(0) + c_1\,\alpha\,\varphi'(0)" />
        </EquationWithWhere>
      }
    >
      <label className="flex flex-col gap-1 text-sm text-slate-300">
        α = {alpha.toFixed(2)}
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
        α = {alpha.toFixed(2)}
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
      <p className="font-mono text-sm text-emerald-300">
        α = {run.alpha.toFixed(4)} · φ′(α) ={" "}
        {quadraticPhiPrime(run.alpha).toFixed(4)}
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
        α = {alpha.toFixed(2)}
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
      <p className="font-mono text-sm text-emerald-300">
        |φ′(α)| = {Math.abs(phiPA).toFixed(4)} · c₂|φ′(0)| ={" "}
        {(0.9 * Math.abs(QUADRATIC_PHI_PRIME0)).toFixed(4)}
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
        α = {alpha.toFixed(2)}
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

const EXACT_LS_OVERVIEW: ConceptOverview = {
  purpose:
    "Pick the step size that minimizes the 1D restriction of the objective along the search direction.",
  explanation:
    "Given φ(α) = f(x + αp), exact line search finds α* in a bracket where φ is unimodal. It is ideal when φ is cheap to evaluate and easy to minimize, but is rarely used alone in large-scale optimization.",
  references: [
    "Nocedal & Wright, Numerical Optimization, §3.1",
    "Boyd & Vandenberghe, Convex Optimization, §9.2",
  ],
};

const BACKTRACKING_OVERVIEW: ConceptOverview = {
  purpose:
    "Find an acceptable step length by shrinking α from an initial guess until a simple decrease condition holds.",
  explanation:
    "Backtracking starts at α₀ and multiplies α by ρ ∈ (0, 1) until sufficient decrease is met. It avoids solving for the exact minimizer and is the standard wrapper around Newton and quasi-Newton directions.",
  references: [
    "Armijo (1966), backtracking for Newton methods",
    "Nocedal & Wright, §3.1–3.2",
  ],
};

const ARMIJO_OVERVIEW: ConceptOverview = {
  purpose:
    "Require enough decrease along the search direction to guarantee global convergence of descent methods.",
  explanation:
    "The Armijo (sufficient decrease) condition φ(α) ≤ φ(0) + c₁ α φ′(0) with c₁ ≪ 1 rules out steps that are too short to make progress. Backtracking Armijo is the most common inexact line search in practice.",
  references: ["Armijo (1966)", "Nocedal & Wright, §3.1"],
};

const WOLFE_OVERVIEW: ConceptOverview = {
  purpose:
    "Balance sufficient decrease with a requirement that the slope along φ is less negative at the chosen α.",
  explanation:
    "Wolfe pairs Armijo with a curvature condition φ′(α) ≥ c₂ φ′(0) (0 < c₁ < c₂ < 1). Accepted steps are neither too small nor too large, which helps quasi-Newton updates satisfy the secant equation.",
  references: ["Wolfe (1969, 1971)", "Nocedal & Wright, §3.1, §6.1"],
};

const STRONG_WOLFE_OVERVIEW: ConceptOverview = {
  purpose:
    "Strengthen Wolfe curvature by bounding |φ′(α)| relative to |φ′(0)|.",
  explanation:
    "Strong Wolfe replaces the weak curvature inequality with |φ′(α)| ≤ c₂ |φ′(0)|, ensuring the directional derivative is substantially reduced. Line searches for BFGS and L-BFGS typically use strong Wolfe or close variants.",
  references: [
    "Nocedal & Wright, §3.1",
    "Line search routines in MINPACK / More–Thuente",
  ],
};

const GOLDSTEIN_OVERVIEW: ConceptOverview = {
  purpose:
    "Accept α only if φ(α) lies in a symmetric band around the linear model at α = 0.",
  explanation:
    "Goldstein conditions bracket φ(α) between two lines through φ(0) with slopes c and (1−c) times φ′(0). They imply both sufficient decrease and an upper bound on the step, offering an alternative to Wolfe for some descent proofs.",
  references: ["Goldstein (1962)", "Nocedal & Wright, §3.1"],
};

export const LINE_SEARCH_PAGES = [
  {
    id: "exact",
    title: "Exact line search",
    Demo: ExactLineSearchDemo,
    overview: EXACT_LS_OVERVIEW,
  },
  {
    id: "backtracking",
    title: "Backtracking line search",
    Demo: BacktrackingDemo,
    overview: BACKTRACKING_OVERVIEW,
  },
  {
    id: "armijo",
    title: "Armijo condition",
    Demo: ArmijoDemo,
    overview: ARMIJO_OVERVIEW,
  },
  {
    id: "wolfe",
    title: "Wolfe conditions",
    Demo: WolfeDemo,
    overview: WOLFE_OVERVIEW,
  },
  {
    id: "strong-wolfe",
    title: "Strong Wolfe conditions",
    Demo: StrongWolfeDemo,
    overview: STRONG_WOLFE_OVERVIEW,
  },
  {
    id: "goldstein",
    title: "Goldstein conditions",
    Demo: GoldsteinDemo,
    overview: GOLDSTEIN_OVERVIEW,
  },
] as const;

export type LineSearchPageId = (typeof LINE_SEARCH_PAGES)[number]["id"];

function ConceptIntro() {
  return (
    <div className="mb-6">
      <EquationWithWhere
        where={[
          {
            tex: "\\varphi(\\alpha)",
            description: "1D line-search objective used in every demo here",
          },
          {
            tex: "\\varphi'(0)",
            description:
              "directional derivative at α = 0 (fixed at −2 for this φ)",
          },
          {
            tex: "\\alpha",
            description: "scalar step along the search direction",
          },
        ]}
      >
        <MathBlock tex={`\\varphi(\\alpha) = (1-\\alpha)^2`} />
        <MathBlock tex={`\\varphi'(0) = -2`} />
      </EquationWithWhere>
    </div>
  );
}

export function LineSearchConcepts({
  initialPageId = "exact",
}: {
  initialPageId?: LineSearchPageId;
}) {
  const [activeId, setActiveId] = useState<LineSearchPageId>(initialPageId);
  const [viewMode, setViewMode] = useState<ConceptViewMode>("demo");
  const activePage =
    LINE_SEARCH_PAGES.find((p) => p.id === activeId) ?? LINE_SEARCH_PAGES[0];
  const ActiveDemo = activePage.Demo;

  return (
    <div className="flex flex-col gap-8 lg:flex-row lg:items-start">
      <ConceptChapterSidebar
        chapter={LINE_SEARCH_CHAPTER}
        aria-label={`${LINE_SEARCH_CHAPTER} concepts`}
        pages={LINE_SEARCH_PAGES}
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
