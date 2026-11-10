import { api, ComparisonRow } from "../api";
import { CompareBars, Scatter2D } from "../components/charts";
import { Card, Page, Status } from "../components/ui";
import { fmt, fmtBytes, useAsync } from "../hooks";

export default function Communication() {
  const acc = useAsync(() => api.compare("EXP-06", "test_accuracy", "variant"), []);
  const bytes = useAsync(() => api.compare("EXP-06", "total_bytes", "variant"), []);
  const a = acc.data ?? [];
  const b = (bytes.data ?? []) as ComparisonRow[];
  const base = b.find((r) => r.key.variant === "float32")?.mean;
  const pts = a.flatMap((r) => { const x = b.find((q) => q.key.variant === r.key.variant)?.mean; return x === undefined ? [] : [{ name: String(r.key.variant), x: x / 1e6, y: r.mean }]; });
  return (
    <Page title="Communication analysis" sub="Bytes are measured on the serialized payloads actually exchanged (uplink updates + downlink models), not estimated from tensor sizes. Compression is applied to the client→server update.">
      <Status loading={acc.loading} error={acc.error ?? bytes.error} empty={!acc.loading && !acc.error && a.length === 0} />
      {a.length > 0 && (
        <>
          <div className="grid g2">
            <Card title="EXP-06 · accuracy by codec" desc="Test accuracy of FedAvg with each update codec (mean ± 95% CI).">
              <CompareBars rows={a} xKey="variant" yLabel="test accuracy" domain={[0.4, 1]} />
            </Card>
            <Card title="Bytes vs accuracy" desc="Total traffic over the whole run (MB, x) against mean test accuracy (y). Up and to the left is better.">
              <Scatter2D points={pts} xLabel="total traffic (MB)" yLabel="test accuracy" />
            </Card>
          </div>
          <div style={{ height: 16 }} />
          <Card title="Traffic table" desc="Total uplink+downlink bytes per run and reduction relative to the float32 baseline.">
            <table>
              <thead><tr><th>Codec</th><th className="num">Total bytes</th><th className="num">vs float32</th><th className="num">Test accuracy</th></tr></thead>
              <tbody>{b.map((r) => { const acc1 = a.find((q) => q.key.variant === r.key.variant);
                return <tr key={String(r.key.variant)}><td>{String(r.key.variant)}</td><td className="num">{fmtBytes(r.mean)}</td>
                  <td className="num">{base ? `${(base / r.mean).toFixed(2)}×` : "–"}</td><td className="num">{fmt(acc1?.mean)}</td></tr>; })}</tbody>
            </table>
          </Card>
        </>
      )}
    </Page>
  );
}
