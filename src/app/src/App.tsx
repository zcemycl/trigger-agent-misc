import { useState } from "react";
import { GradientMethodConcepts } from "./components/GradientMethodConcepts";
import { LineSearchConcepts } from "./components/LineSearchConcepts";

type Chapter = "line-search" | "gradient-methods";

export default function App() {
  const [chapter, setChapter] = useState<Chapter>("gradient-methods");

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6 py-5 sm:flex-row sm:items-center sm:justify-between">
          <h1 className="text-xl font-semibold tracking-tight">
            Optimization Concepts
          </h1>
          <div className="flex flex-wrap items-center gap-2">
            <nav
              className="flex rounded-lg border border-slate-700 p-0.5"
              aria-label="Optimization chapters"
            >
              <button
                type="button"
                onClick={() => setChapter("line-search")}
                aria-current={chapter === "line-search" ? "page" : undefined}
                className={`rounded-md px-3 py-1.5 text-sm transition ${
                  chapter === "line-search"
                    ? "bg-emerald-500/20 font-medium text-emerald-200"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Line search
              </button>
              <button
                type="button"
                onClick={() => setChapter("gradient-methods")}
                aria-current={
                  chapter === "gradient-methods" ? "page" : undefined
                }
                className={`rounded-md px-3 py-1.5 text-sm transition ${
                  chapter === "gradient-methods"
                    ? "bg-emerald-500/20 font-medium text-emerald-200"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Gradient methods
              </button>
            </nav>
            <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-medium text-emerald-300">
              trigger-agent-misc
            </span>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-10">
        {chapter === "line-search" ? (
          <LineSearchConcepts />
        ) : (
          <GradientMethodConcepts />
        )}
      </main>
    </div>
  );
}
