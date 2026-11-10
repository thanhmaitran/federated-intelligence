import { api } from "../api";
import { CompareBars, Scatter2D } from "../components/charts";
import { Card, Page, Status } from "../components/ui";
import { fmt, useAsync } from "../hooks";

export default function Privacy() {
  const acc = useAsync(() => api.compare("EXP-04", "test_accuracy", "variant"), []);
  const eps = useAsync(() => api.compare("EXP-04", "epsilon", "variant"), []);
  const grid = useAsync(() => api.compare("EXP-05", "test_accuracy", "privacy.clip_norm,privacy.noise_multiplier"), []);
  const geps = useAsync(() => api.compare("EXP-05", "epsilon", "privacy.clip_norm,privacy.noise_multiplier"), []);
  const rows = acc.data ?? [];
  const epsOf = (v: string) => (eps.data ?? []).find((r) => r.key.variant === v)?.mean;
  const pts = rows.flatMap((r) => { const e = epsOf(String(r.key.variant)); return e === undefined ? [] : [{ name: String(r.key.variant), x: e, y: r.mean }]; });
  const clips = [...new Set((grid.data ?? []).map((r) => String(r.key["privacy.clip_norm"])))];
  const noises = [...new Set((grid.data ?? []).map((r) => String(r.key["privacy.noise_multiplier"])))];
  const cell = (rs: typeof grid.data, c: string, n: string) => rs?.find((r) => String(r.key["privacy.clip_norm"]) === c && String(r.key["privacy.noise_multiplier"]) === n);
  return (
    <Page title="Privacy analysis" sub="Client-level central DP: the server clips each client update to norm C and adds Gaussian noise N(0, σ²C²I) to the sum. ε is computed by an RDP accountant for the Poisson-sampled Gaussian mechanism (add/remove-one-client adjacency); it is never assumed from σ.">
      <Status loading={acc.loading} error={acc.error ?? eps.error} empty={!acc.loading && !acc.error && rows.length === 0} />
      {rows.length > 0 && (
        <div className="grid g2">
          <Card title="EXP-04 · accuracy vs privacy budget" desc="Test accuracy per target ε (mean ± 95% CI over seeds). 'nodp' and 'clip_only' are the non-private references.">
            <CompareBars rows={rows} xKey="variant" yLabel="test accuracy" domain={[0.4, 1]} />
          </Card>
          <Card title="Privacy–utility frontier" desc="Each point is one configuration: achieved ε (x, log scale) against mean test accuracy (y).">
            <Scatter2D points={pts} xLabel="ε (δ = 1e-5)" yLabel="test accuracy" logX />
          </Card>
        </div>
      )}
      {clips.length > 0 && (
        <>
          <div style={{ height: 16 }} />
          <Card title="EXP-05 · clipping norm × noise multiplier" desc="Mean test accuracy (top) and the ε the accountant reports for the same configuration (bottom, smaller is more private).">
            <table>
              <thead><tr><th>clip C \ σ</th>{noises.map((n) => <th key={n} className="num">σ = {n}</th>)}</tr></thead>
              <tbody>
                {clips.map((c) => <tr key={c}><td>C = {c}</td>{noises.map((n) => { const a = cell(grid.data, c, n); const e = cell(geps.data, c, n);
                  return <td key={n} className="num">{a ? fmt(a.mean, 3) : "–"} <span className="note">· ε {e ? fmt(e.mean, 1) : "–"}</span></td>; })}</tr>)}
              </tbody>
            </table>
          </Card>
        </>
      )}
    </Page>
  );
}
