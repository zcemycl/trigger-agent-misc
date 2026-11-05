export function DemoExamplePicker<T extends string>({
  label,
  examples,
  value,
  onChange,
}: {
  label: string;
  examples: readonly { id: T; label: string }[];
  value: T;
  onChange: (id: T) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label={label}>
      {examples.map((example) => {
        const selected = example.id === value;
        return (
          <button
            key={example.id}
            type="button"
            aria-pressed={selected}
            onClick={() => onChange(example.id)}
            className={`rounded-md border px-2.5 py-1 text-xs transition ${
              selected
                ? "border-emerald-500/50 bg-emerald-500/15 font-medium text-emerald-200"
                : "border-slate-700 text-slate-400 hover:border-slate-500 hover:text-slate-200"
            }`}
          >
            {example.label}
          </button>
        );
      })}
    </div>
  );
}
