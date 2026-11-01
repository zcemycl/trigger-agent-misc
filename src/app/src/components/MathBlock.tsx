import katex from "katex";
import { useMemo } from "react";

interface MathBlockProps {
  tex: string;
  /** Block (`displayMode`) vs inline math. */
  display?: boolean;
}

function renderKatex(tex: string, display: boolean): string {
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
}

export function MathBlock({ tex, display = true }: MathBlockProps) {
  const html = useMemo(() => renderKatex(tex, display), [tex, display]);

  return (
    <div
      className="overflow-x-auto rounded-lg border border-slate-700/80 bg-slate-950/50 px-4 py-3 text-slate-100 [&_.katex]:text-[1.05rem]"
      data-testid="math-block"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function InlineMath({ tex }: { tex: string }) {
  const html = useMemo(() => renderKatex(tex, false), [tex]);

  return (
    <span
      className="inline-block text-slate-200 [&_.katex]:text-[0.95em]"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
