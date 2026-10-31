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
import { PhiChart } from "./PhiChart";

function Section({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
      <h2 className="text-lg font-medium text-white">{title}</h2>
      <p className="mt-2 text-sm leading-relaxed text-slate-400">
        {description}
      </p>
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
      description="Golden-section search finds the minimizer of unimodal φ(α) on a bracket [a, b]. Drag the upper bracket bound, then run search."
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
      description="Start with a trial step α₀; shrink by factor ρ until the Armijo sufficient-decrease condition holds."
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
      description="Sufficient decrease: φ(α) ≤ φ(0) + c₁ α φ′(0). Move α to see when the trial point lies below the Armijo line."
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
      description="Weak Wolfe pairs Armijo with curvature φ′(α) ≥ c₂ φ′(0). Explore α manually or compare with the Wolfe line search result."
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
      description="Strong curvature requires |φ′(α)| ≤ c₂ |φ′(0)|, giving a more restrictive step near the minimizer."
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
      description="φ(α) must lie in [φ(0) + (1−c)αφ′(0), φ(0) + cαφ′(0)]. The yellow segment shows the acceptable interval at your α."
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

export function LineSearchConcepts() {
  return (
    <div className="space-y-8">
      <p className="text-slate-400">
        Interactive demos use the ray φ(α) = (1 − α)² with φ′(0) = −2, matching
        the Python examples in{" "}
        <code className="rounded bg-slate-800 px-1.5 py-0.5 text-emerald-300">
          trigger_agent_misc.optimization
        </code>
        .
      </p>
      <ExactLineSearchDemo />
      <BacktrackingDemo />
      <ArmijoDemo />
      <WolfeDemo />
      <StrongWolfeDemo />
      <GoldsteinDemo />
    </div>
  );
}
