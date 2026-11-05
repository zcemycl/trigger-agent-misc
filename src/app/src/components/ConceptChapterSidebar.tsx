export type ConceptViewMode = "demo" | "overview";

export function ConceptChapterSidebar<T extends string>({
  chapter,
  sectionLabel,
  pages,
  activeId,
  onActiveIdChange,
  viewMode,
  onViewModeChange,
  "aria-label": navAriaLabel,
}: {
  chapter: string;
  sectionLabel?: string;
  pages: readonly { id: T; title: string }[];
  activeId: T;
  onActiveIdChange: (id: T) => void;
  viewMode: ConceptViewMode;
  onViewModeChange: (mode: ConceptViewMode) => void;
  "aria-label"?: string;
}) {
  return (
    <aside className="w-full shrink-0 lg:sticky lg:top-6 lg:w-72">
      {sectionLabel ? (
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          {sectionLabel}
        </p>
      ) : null}
      <nav
        className={`flex flex-col gap-1 ${sectionLabel ? "mt-3" : ""}`}
        aria-label={navAriaLabel ?? `${chapter} concepts`}
      >
        <button
          type="button"
          onClick={() => onViewModeChange("overview")}
          aria-current={viewMode === "overview" ? "page" : undefined}
          className={`rounded-lg border border-slate-800 px-3 py-2 text-left text-sm transition ${
            viewMode === "overview"
              ? "bg-emerald-500/15 font-medium text-emerald-200"
              : "text-slate-400 hover:bg-slate-800/80 hover:text-slate-200"
          }`}
        >
          Overview
        </button>
        {pages.map((page) => {
          const isActive = page.id === activeId && viewMode === "demo";
          return (
            <button
              key={page.id}
              type="button"
              onClick={() => {
                onActiveIdChange(page.id);
                onViewModeChange("demo");
              }}
              aria-current={isActive ? "page" : undefined}
              className={`rounded-lg px-3 py-2 text-left text-sm transition ${
                isActive
                  ? "bg-emerald-500/15 font-medium text-emerald-200"
                  : "text-slate-400 hover:bg-slate-800/80 hover:text-slate-200"
              }`}
            >
              {page.title}
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
