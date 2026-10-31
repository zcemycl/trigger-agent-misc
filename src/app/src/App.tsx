import { LineSearchConcepts } from "./components/LineSearchConcepts";

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-6 py-5">
          <h1 className="text-xl font-semibold tracking-tight">
            Line Search Concepts
          </h1>
          <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-medium text-emerald-300">
            trigger-agent-misc
          </span>
        </div>
      </header>
      <main className="mx-auto max-w-3xl px-6 py-10">
        <LineSearchConcepts />
      </main>
    </div>
  );
}
