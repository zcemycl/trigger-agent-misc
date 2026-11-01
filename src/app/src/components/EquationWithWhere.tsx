import { type ReactNode } from "react";
import { InlineMath } from "./MathBlock";

export type WhereClause = {
  tex: string;
  description: string;
};

export function EquationWithWhere({
  children,
  where,
}: {
  children: ReactNode;
  where: WhereClause[];
}) {
  if (where.length === 0) {
    return <>{children}</>;
  }

  return (
    <div className="space-y-2">
      {children}
      <div
        className="rounded-lg border border-slate-800/60 bg-slate-950/40 px-4 py-2.5 text-sm leading-relaxed text-slate-400"
        data-testid="math-where"
      >
        <span className="italic text-slate-500">where</span>
        <ul className="mt-1.5 list-none space-y-1">
          {where.map((item) => (
            <li
              key={item.tex}
              className="flex flex-wrap items-baseline gap-x-1.5 gap-y-0.5"
            >
              <InlineMath tex={item.tex} />
              <span>— {item.description}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
