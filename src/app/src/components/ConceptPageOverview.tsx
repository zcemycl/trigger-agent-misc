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
    <div
      className="mt-4 space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-3 text-xs leading-relaxed text-slate-400"
      aria-label={`${title} overview`}
    >
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
  );
}
