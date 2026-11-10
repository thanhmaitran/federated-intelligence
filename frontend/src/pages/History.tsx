import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { Card, Page, Status } from "../components/ui";
import { fmt, fmtBytes, useAsync } from "../hooks";

export default function History() {
  const [sp] = useSearchParams();
  const [group, setGroup] = useState(sp.get("group") ?? "");
  const [status, setStatus] = useState("");
  const groups = useAsync(api.groups, []);
  const rows = useAsync(() => api.experiments({ group, status }), [group, status], 5000);
  return (
    <Page title="Experiment history" sub="All stored runs. Select a row to open it in the training monitor.">
      <div className="row">
        <label>Group<select value={group} onChange={(e) => setGroup(e.target.value)}>
          <option value="">all</option>{(groups.data ?? []).map((g) => <option key={g.group}>{g.group}</option>)}</select></label>
        <label>Status<select value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="">all</option>{["finished", "running", "queued", "failed"].map((s) => <option key={s}>{s}</option>)}</select></label>
        <span className="note">{rows.data?.length ?? 0} runs</span>
      </div>
      <Status loading={rows.loading && !rows.data} error={rows.error} />
      <Card>
        <div className="scroll" style={{ maxHeight: 640 }}><table>
          <thead><tr><th>Run</th><th>Algorithm</th><th>Status</th><th className="num">Seed</th><th className="num">Test acc.</th><th className="num">ε</th><th className="num">Traffic</th></tr></thead>
          <tbody>{(rows.data ?? []).map((e) => (
            <tr key={e.id}><td><Link className="link" to={`/monitor?id=${encodeURIComponent(e.id)}`}>{e.id}</Link></td><td>{e.algorithm}</td>
              <td><span className={`pill ${e.status}`}>{e.status}</span></td><td className="num">{e.seed}</td><td className="num">{fmt(e.test_accuracy)}</td>
              <td className="num">{e.epsilon !== null ? fmt(e.epsilon, 2) : "–"}</td><td className="num">{fmtBytes(e.total_bytes)}</td></tr>))}</tbody>
        </table></div>
      </Card>
    </Page>
  );
}
