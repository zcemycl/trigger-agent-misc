const readings = [
  {
    title: "Line Search Methods",
    summary: "Backtracking Armijo line search for unconstrained optimization.",
    href: "https://github.com/zcemycl/trigger-agent-misc/blob/main/notebooks/line_search_demo.ipynb",
  },
  {
    title: "Optimization Library",
    summary: "Python helpers under trigger_agent_misc.optimization.",
    href: "https://github.com/zcemycl/trigger-agent-misc/tree/main/src/trigger_agent_misc",
  },
];

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-6 py-5">
          <h1 className="text-xl font-semibold tracking-tight">Readings</h1>
          <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-medium text-emerald-300">
            trigger-agent-misc
          </span>
        </div>
      </header>
      <main className="mx-auto max-w-3xl px-6 py-10">
        <p className="mb-8 text-slate-400">
          Static notes and links for ML optimization work in this repository.
        </p>
        <ul className="space-y-4">
          {readings.map((item) => (
            <li key={item.title}>
              <a
                href={item.href}
                className="block rounded-xl border border-slate-800 bg-slate-900 p-5 transition hover:border-emerald-500/40 hover:bg-slate-900/80"
              >
                <h2 className="text-lg font-medium text-white">{item.title}</h2>
                <p className="mt-2 text-sm text-slate-400">{item.summary}</p>
              </a>
            </li>
          ))}
        </ul>
      </main>
    </div>
  );
}
