import { useState } from "react";
import { LineSearchConcepts } from "./components/LineSearchConcepts";
import { SecondOrderConcepts } from "./components/SecondOrderConcepts";

type Chapter = "line-search" | "second-order";

export default function App() {
  const [chapter, setChapter] = useState<Chapter>("second-order");

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-6 py-5">
          <h1 className="text-xl font-semibold tracking-tight">
            Optimization Concepts
          </h1>
          <div className="flex items-center gap-2">
            <nav
              className="flex rounded-lg border border-slate-700 bg-slate-900 p-1"
              aria-label="Optimization chapters"
            >
              <button
                type="button"
                onClick={() => setChapter("line-search")}
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
                onClick={() => setChapter("second-order")}
                className={`rounded-md px-3 py-1.5 text-sm transition ${
                  chapter === "second-order"
                    ? "bg-emerald-500/20 font-medium text-emerald-200"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Second-order
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
          <SecondOrderConcepts />
        )}
      </main>
    </div>
  );
}
