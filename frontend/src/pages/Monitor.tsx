import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { RoundLines, pick } from "../components/charts";
import { Card, Page, Status } from "../components/ui";
import { fmt, fmtBytes, useAsync } from "../hooks";

export default function Monitor() {
  const [sp, setSp] = useSearchParams();
  const list = useAsync(() => api.experiments({ limit: 2000 }), []);
  const id = sp.get("id") ?? list.data?.find((e) => e.status === "finished")?.id ?? "";
  const det = useAsync(() => (id ? api.experiment(id) : Promise.resolve(null)), [id], 2000);
  const met = useAsync(() => (id ? api.metrics(id) : Promise.resolve(null)), [id], 2000);
  const m = met.data?.metrics ?? {};
  const s = det.data?.summary ?? {};
  const cfg = det.data?.config ?? {};
  return (
    <Page title="Training monitor" sub="Round-by-round metrics of one run. Running experiments refresh every two seconds.">
      <div className="row">
        <label>Run
          <select style={{ minWidth: 360 }} value={id} onChange={(e) => setSp({ id: e.target.value })}>
            {(list.data ?? []).map((e) => <option key={e.id} value={e.id}>{e.id} · {e.status}</option>)}
          </select>
        </label>
        {det.data && <span className={`pill ${det.data.status}`}>{det.data.status}</span>}
      </div>
      <Status loading={list.loading && !list.data} error={list.error ?? det.error ?? met.error} empty={!list.loading && !id} />
      {det.data?.error && <div className="err">{det.data.error}</div>}
      {det.data && (
        <>
          <div className="grid g2">
            <Card title="Global accuracy" desc="Accuracy of the global model on the centralized test set (solid) and the pooled client validation data.">
              <RoundLines series={pick(m, ["test_accuracy", "val_accuracy"])} domain={[0.4, 1]} yLabel="accuracy" />
            </Card>
            <Card title="Global loss" desc="Cross-entropy of the global model after every aggregation.">
              <RoundLines series={pick(m, ["test_loss", "val_loss"])} yLabel="cross-entropy" />
            </Card>
            <Card title="Client fairness" desc="Mean, 10th percentile and worst per-client validation accuracy (evaluated every few rounds).">
              <RoundLines series={pick(m, ["client_acc_mean", "client_acc_p10", "client_acc_min"])} domain={[0, 1]} yLabel="accuracy" />
            </Card>
            <Card title="Run summary" desc="Final values and resource use.">
              <table><tbody>
                <tr><td>Algorithm</td><td className="num">{cfg.federated?.algorithm} (μ={cfg.federated?.mu})</td></tr>
                <tr><td>Final test accuracy</td><td className="num">{fmt(s.final?.test_accuracy)}</td></tr>
                <tr><td>Final macro-F1</td><td className="num">{fmt(s.final?.test_macro_f1)}</td></tr>
                <tr><td>Rounds to target val. accuracy</td><td className="num">{s.rounds_to_target_val_accuracy ?? "not reached"}</td></tr>
                <tr><td>Total communication</td><td className="num">{fmtBytes(s.total_bytes)}</td></tr>
                <tr><td>ε (δ={s.privacy?.delta ?? "–"})</td><td className="num">{s.privacy?.epsilon !== undefined ? fmt(s.privacy.epsilon, 2) : "no DP"}</td></tr>
                <tr><td>Wall-clock seconds</td><td className="num">{fmt(s.wall_seconds, 1)}</td></tr>
              </tbody></table>
              <p className="note"><Link className="link" to="/history">All runs →</Link></p>
            </Card>
          </div>
        </>
      )}
    </Page>
  );
}
