import type { ReactNode } from "react";

interface Props {
  available: boolean;
  title: string;
  onRetry?: () => void;
  children: ReactNode;
}

export default function CloudFeatureGate({ available, title, onRetry, children }: Props) {
  if (available) return <>{children}</>;
  return (
    <section className="rounded-xl border border-amber-200 bg-amber-50 p-5 text-center" role="status">
      <h3 className="font-semibold text-amber-900">{title}等待雲端服務</h3>
      <p className="mt-1 text-sm text-amber-800">課程內容仍可瀏覽，雲端恢復後即可繼續使用。</p>
      {onRetry && <button onClick={onRetry} className="mt-3 rounded-lg bg-white px-4 py-2 text-sm font-semibold text-amber-900 shadow-sm ring-1 ring-amber-200">重新連線</button>}
    </section>
  );
}
