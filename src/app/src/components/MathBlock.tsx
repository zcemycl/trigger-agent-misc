import katex from "katex";
import { useMemo } from "react";

interface MathBlockProps {
  tex: string;
  /** Block (`displayMode`) vs inline math. */
  display?: boolean;
}

export function MathBlock({ tex, display = true }: MathBlockProps) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, {
        throwOnError: true,
        displayMode: display,
        strict: "ignore",
      });
    } catch {
      return katex.renderToString(tex, {
        throwOnError: false,
        displayMode: display,
      });
    }
  }, [tex, display]);

  return (
    <div
      className="overflow-x-auto rounded-lg border border-slate-700/80 bg-slate-950/50 px-4 py-3 text-slate-100 [&_.katex]:text-[1.05rem]"
      data-testid="math-block"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
