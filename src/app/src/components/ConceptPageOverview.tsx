export interface ConceptOverview {
  purpose: string;
  explanation: string;
  references: readonly string[];
}

export function ConceptPageOverview({
  title,
  overview,
}: {
  title: string;
  overview: ConceptOverview;
}) {
  return (
    <section
      className="rounded-xl border border-slate-800 bg-slate-900/70 p-6 text-sm leading-relaxed text-slate-400"
      aria-label={`${title} overview`}
    >
      <h2 className="text-lg font-medium text-white">{title}</h2>
      <div className="mt-4 space-y-3">
        <p>
          <span className="font-medium text-slate-300">Purpose. </span>
          {overview.purpose}
        </p>
        <p>{overview.explanation}</p>
        <div>
          <p className="font-medium text-slate-300">References</p>
          <ul className="mt-1 list-inside list-disc space-y-0.5">
            {overview.references.map((ref) => (
              <li key={ref}>{ref}</li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
