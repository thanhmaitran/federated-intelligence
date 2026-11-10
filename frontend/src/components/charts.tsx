import {
  Bar, BarChart, CartesianGrid, ErrorBar, Legend, Line, LineChart, ResponsiveContainer, Scatter,
  ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { ComparisonRow, MetricSeries } from "../api";
import { PALETTE } from "./ui";

const axis = { stroke: "var(--muted)", fontSize: 12 };

/** Round-indexed line chart; `series` maps legend name -> points. */
export function RoundLines({ series, yLabel, domain }: {
  series: Record<string, { round: number; value: number }[]>; yLabel?: string; domain?: [number | "auto", number | "auto"];
}) {
  const rounds = new Set<number>();
  Object.values(series).forEach((s) => s.forEach((p) => rounds.add(p.round)));
  const data = [...rounds].sort((a, b) => a - b).map((r) => {
    const row: Record<string, number> = { round: r };
    for (const [k, s] of Object.entries(series)) {
      const p = s.find((q) => q.round === r);
      if (p) row[k] = p.value;
    }
    return row;
  });
  return (
    <div className="chartbox">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
          <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
          <XAxis dataKey="round" {...axis} label={{ value: "round", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 12 }} />
          <YAxis {...axis} domain={domain ?? ["auto", "auto"]} label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "var(--muted)", fontSize: 12 } : undefined} />
          <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)" }} />
          {Object.keys(series).length > 1 && <Legend />}
          {Object.keys(series).map((k, i) => (
            <Line key={k} type="monotone" dataKey={k} stroke={PALETTE[i % PALETTE.length]} dot={false} strokeWidth={2} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export const pick = (m: MetricSeries, names: string[]) =>
  Object.fromEntries(names.filter((n) => m[n]).map((n) => [n, m[n]]));

/** Grouped bars with 95% CI whiskers. x = first key, series = second key (optional). */
export function CompareBars({ rows, xKey, seriesKey, yLabel, domain }: {
  rows: ComparisonRow[]; xKey: string; seriesKey?: string; yLabel?: string; domain?: [number | "auto", number | "auto"];
}) {
  const xs = [...new Set(rows.map((r) => String(r.key[xKey])))];
  const ss = seriesKey ? [...new Set(rows.map((r) => String(r.key[seriesKey])))] : ["value"];
  const data = xs.map((x) => {
    const d: Record<string, string | number | number[]> = { x };
    for (const s of ss) {
      const r = rows.find((q) => String(q.key[xKey]) === x && (!seriesKey || String(q.key[seriesKey]) === s));
      if (r) {
        d[s] = r.mean;
        d[`${s}_err`] = [r.mean - r.ci_low, r.ci_high - r.mean];
      }
    }
    return d;
  });
  return (
    <div className="chartbox">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
          <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="x" {...axis} />
          <YAxis {...axis} domain={domain ?? ["auto", "auto"]} label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "var(--muted)", fontSize: 12 } : undefined} />
          <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)" }} formatter={(v: number | string) => (typeof v === "number" ? v.toFixed(4) : v)} />
          {ss.length > 1 && <Legend />}
          {ss.map((s, i) => (
            <Bar key={s} dataKey={s} fill={PALETTE[i % PALETTE.length]} isAnimationActive={false} radius={[3, 3, 0, 0]}>
              <ErrorBar dataKey={`${s}_err`} stroke="var(--ink)" width={4} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Scatter2D({ points, xLabel, yLabel, logX }: {
  points: { name: string; x: number; y: number }[]; xLabel: string; yLabel: string; logX?: boolean;
}) {
  return (
    <div className="chartbox">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
          <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
          <XAxis type="number" dataKey="x" name={xLabel} scale={logX ? "log" : "auto"} domain={["auto", "auto"]} {...axis}
            label={{ value: xLabel, position: "insideBottom", offset: -10, fill: "var(--muted)", fontSize: 12 }} />
          <YAxis type="number" dataKey="y" name={yLabel} domain={["auto", "auto"]} {...axis}
            label={{ value: yLabel, angle: -90, position: "insideLeft", fill: "var(--muted)", fontSize: 12 }} />
          <ZAxis range={[90, 90]} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)" }}
            formatter={(v: number | string) => (typeof v === "number" ? v.toFixed(4) : v)} labelFormatter={() => ""} />
          <Scatter data={points} fill="var(--c1)" isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
