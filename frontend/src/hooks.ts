import { useEffect, useState } from "react";

export interface Async<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Fetch on mount / when deps change; optional polling interval (ms). */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[], pollMs = 0): Async<T> {
  const [state, setState] = useState<Async<T>>({ data: null, error: null, loading: true });
  useEffect(() => {
    let alive = true;
    const run = () =>
      fn()
        .then((data) => alive && setState({ data, error: null, loading: false }))
        .catch((e: Error) => alive && setState((s) => ({ data: s.data, error: e.message, loading: false })));
    setState((s) => ({ ...s, loading: true }));
    run();
    const t = pollMs ? setInterval(run, pollMs) : undefined;
    return () => {
      alive = false;
      if (t) clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

export const fmt = (v: number | null | undefined, d = 4) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(d);

export function fmtBytes(b: number | null | undefined): string {
  if (b === null || b === undefined) return "–";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let x = b;
  while (x >= 1024 && i < u.length - 1) {
    x /= 1024;
    i++;
  }
  return `${x.toFixed(i === 0 ? 0 : 2)} ${u[i]}`;
}
