import { MathBlock } from "./MathBlock";

export interface ConceptReference {
  href: string;
  label: string;
}

export interface ConceptOverview {
  purposeTex: string;
  explanationTex: string;
  references: readonly ConceptReference[];
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
      <div className="mt-4 space-y-4">
        <div>
          <p className="mb-2 font-medium text-slate-300">Purpose.</p>
          <MathBlock tex={overview.purposeTex} />
        </div>
        <div>
          <p className="mb-2 font-medium text-slate-300">Update</p>
          <MathBlock tex={overview.explanationTex} />
        </div>
        <div>
          <p className="font-medium text-slate-300">References</p>
          <ul className="mt-2 list-inside list-disc space-y-1">
            {overview.references.map((ref) => (
              <li key={ref.href}>
                <a
                  href={ref.href}
                  className="text-emerald-400 underline decoration-emerald-500/40 underline-offset-2 hover:text-emerald-300"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {ref.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
