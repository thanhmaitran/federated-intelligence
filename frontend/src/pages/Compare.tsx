import { useState } from "react";
import { api } from "../api";
import { CompareBars } from "../components/charts";
import { Card, Page, Status } from "../components/ui";
import { fmt, useAsync } from "../hooks";

const METRICS = ["test_accuracy", "test_macro_f1", "test_loss", "client_acc_min", "rounds_to_target", "total_bytes"];
const PRESETS: Record<string, { by: string; x: string; s?: string }> = {
  "EXP-01": { by: "variant", x: "variant" },
  "EXP-02": { by: "dataset.alpha,federated.algorithm", x: "dataset.alpha", s: "federated.algorithm" },
  "EXP-03": { by: "federated.participation_rate,federated.algorithm", x: "federated.participation_rate", s: "federated.algorithm" },
  "EXP-07": { by: "federated.dropout_rate,federated.algorithm", x: "federated.dropout_rate", s: "federated.algorithm" },
  "EXP-08": { by: "dataset.partition,dataset.alpha,federated.algorithm", x: "dataset.alpha", s: "federated.algorithm" },
};

export default function Compare() {
  const groups = useAsync(api.groups, []);
  const [group, setGroup] = useState("EXP-02");
  const [metric, setMetric] = useState("test_accuracy");
  const p = PRESETS[group] ?? { by: "variant", x: "variant" };
  const cmp = useAsync(() => api.compare(group, metric, p.by), [group, metric]);
  const rows = cmp.data ?? [];
  const keys = p.by.split(",");
  return (
    <Page title="Algorithm comparison" sub="Mean over independent seeds with 95% Student-t confidence intervals (error bars). Differences smaller than the intervals should not be called improvements without a paired test (see the research report).">
      <div className="row">
        <label>Experiment
          <select value={group} onChange={(e) => setGroup(e.target.value)}>
            {(groups.data ?? []).map((g) => <option key={g.group}>{g.group}</option>)}
          </select>
        </label>
        <label>Metric
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>{METRICS.map((m) => <option key={m}>{m}</option>)}</select>
        </label>
      </div>
      <Status loading={cmp.loading && !cmp.data} error={cmp.error} empty={!cmp.loading && !cmp.error && rows.length === 0} />
      {rows.length > 0 && (
        <>
          <Card title={`${group} · ${metric}`} desc={`Grouped by ${p.by}. Bars show the mean, whiskers the 95% CI.`}>
            <CompareBars rows={rows} xKey={p.x} seriesKey={p.s} yLabel={metric}
              domain={metric === "test_accuracy" || metric === "test_macro_f1" ? [0.4, 1] : ["auto", "auto"]} />
          </Card>
          <div style={{ height: 16 }} />
          <Card title="Table" desc="Same numbers as the chart.">
            <div className="scroll"><table>
              <thead><tr>{keys.map((k) => <th key={k}>{k}</th>)}<th className="num">n</th><th className="num">mean</th><th className="num">SD</th><th className="num">CI low</th><th className="num">CI high</th></tr></thead>
              <tbody>{rows.map((r, i) => <tr key={i}>{keys.map((k) => <td key={k}>{String(r.key[k])}</td>)}
                <td className="num">{r.n}</td><td className="num">{fmt(r.mean)}</td><td className="num">{fmt(r.sd)}</td><td className="num">{fmt(r.ci_low)}</td><td className="num">{fmt(r.ci_high)}</td></tr>)}</tbody>
            </table></div>
          </Card>
        </>
      )}
    </Page>
  );
}
