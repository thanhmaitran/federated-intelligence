import { ReactNode } from "react";

export function Page({ title, sub, children }: { title: string; sub?: string; children: ReactNode }) {
  return (
    <>
      <h1>{title}</h1>
      {sub && <p className="sub">{sub}</p>}
      {children}
    </>
  );
}

export function Card({ title, desc, children }: { title?: string; desc?: string; children: ReactNode }) {
  return (
    <div className="card">
      {title && <h3>{title}</h3>}
      {desc && <p className="desc">{desc}</p>}
      {children}
    </div>
  );
}

export function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="card stat">
      <div className="l">{label}</div>
      <div className="v">{value}</div>
    </div>
  );
}

export function Status({ loading, error, empty }: { loading?: boolean; error?: string | null; empty?: boolean }) {
  if (error) return <div className="err">API error: {error}</div>;
  if (loading) return <p className="note">Loading…</p>;
  if (empty) return <p className="note">No data. Run the experiment suites and import the results (see README).</p>;
  return null;
}

export const PALETTE = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"];
