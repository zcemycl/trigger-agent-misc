import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  numericDerivative,
  wrapGradAtIndex,
  wrapGradientVec,
  wrapObjectiveVec,
  wrapPhi,
  wrapResidualVec,
  type DemoNoiseSettings,
} from "../demoNoise";
import {
  finiteSumGrad,
  finiteSumGradAtIndex,
  finiteSumObjective,
} from "../finiteSumQuadratic";
import {
  QUADRATIC_PHI0,
  QUADRATIC_PHI_PRIME0,
  quadraticPhi,
  quadraticPhiPrime,
} from "../quadratic";
import {
  leastSquaresValue,
  rosenbrockJacobian,
  rosenbrockResidual,
  shiftedQuadratic,
  shiftedQuadratic3D,
  shiftedQuadraticGrad,
  shiftedQuadraticHess,
} from "../secondOrderProblems";

type DemoNoiseContextValue = DemoNoiseSettings & {
  setEnabled: (enabled: boolean) => void;
  reshuffle: () => void;
};

const DemoNoiseContext = createContext<DemoNoiseContextValue | null>(null);

export function DemoNoiseProvider({ children }: { children: ReactNode }) {
  const [enabled, setEnabled] = useState(false);
  const [seed, setSeed] = useState(0);
  const value = useMemo(
    () => ({
      enabled,
      seed,
      setEnabled,
      reshuffle: () => setSeed((s) => s + 1),
    }),
    [enabled, seed],
  );
  return (
    <DemoNoiseContext.Provider value={value}>
      {children}
    </DemoNoiseContext.Provider>
  );
}

export function useDemoNoise(): DemoNoiseContextValue {
  const ctx = useContext(DemoNoiseContext);
  if (!ctx) {
    throw new Error("useDemoNoise must be used within DemoNoiseProvider");
  }
  return ctx;
}

export function DemoNoiseControls() {
  const { enabled, setEnabled, reshuffle } = useDemoNoise();
  return (
    <div className="flex flex-wrap items-center gap-4 rounded-lg border border-slate-800 bg-slate-950/60 px-4 py-3 text-sm text-slate-300">
      <label className="flex cursor-pointer items-center gap-2">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => setEnabled(e.target.checked)}
          className="accent-emerald-500"
        />
        Noise mode
      </label>
      {enabled ? (
        <button
          type="button"
          onClick={reshuffle}
          className="rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-400 transition hover:border-slate-500 hover:text-slate-200"
        >
          Reseed noise
        </button>
      ) : null}
    </div>
  );
}

export function useFiniteSumProblem() {
  const noise = useDemoNoise();
  return useMemo(
    () => ({
      objective: wrapObjectiveVec(finiteSumObjective, noise),
      grad: wrapGradientVec(finiteSumGrad, noise),
      gradAtIndex: wrapGradAtIndex(finiteSumGradAtIndex, noise),
    }),
    [noise.enabled, noise.seed],
  );
}

export function useNoisyQuadraticRay() {
  const noise = useDemoNoise();
  return useMemo(() => {
    const phi = wrapPhi(quadraticPhi, noise);
    const phiPrime = (alpha: number) =>
      noise.enabled ? numericDerivative(phi, alpha) : quadraticPhiPrime(alpha);
    const phi0 = noise.enabled ? phi(0) : QUADRATIC_PHI0;
    const phiPrime0 = noise.enabled ? phiPrime(0) : QUADRATIC_PHI_PRIME0;
    return { phi, phiPrime, phi0, phiPrime0 };
  }, [noise.enabled, noise.seed]);
}

export function useShiftedQuadraticProblem() {
  const noise = useDemoNoise();
  return useMemo(
    () => ({
      objective: wrapObjectiveVec(shiftedQuadratic, noise),
      gradient: wrapGradientVec(shiftedQuadraticGrad, noise),
      hessian: shiftedQuadraticHess,
    }),
    [noise.enabled, noise.seed],
  );
}

export function useShiftedQuadratic3DProblem() {
  const noise = useDemoNoise();
  return useMemo(
    () => ({
      objective: wrapObjectiveVec(shiftedQuadratic3D.objective, noise),
      gradient: wrapGradientVec(shiftedQuadratic3D.gradient, noise),
      hessian: shiftedQuadratic3D.hessian,
    }),
    [noise.enabled, noise.seed],
  );
}

export function useRosenbrockLeastSquares() {
  const noise = useDemoNoise();
  return useMemo(() => {
    const residual = wrapResidualVec(rosenbrockResidual, noise);
    return {
      residual,
      jacobian: rosenbrockJacobian,
      objective: (x: number[]) => leastSquaresValue(residual, x),
    };
  }, [noise.enabled, noise.seed]);
}
