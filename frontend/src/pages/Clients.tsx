import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Card, Page, Status } from "../components/ui";
import { useAsync } from "../hooks";

export default function Clients() {
  const parts = useAsync(api.partitions, []);
  const [pid, setPid] = useState("dirichlet_alpha_0_5_seed_42");
  const cl = useAsync(() => api.clients({ partition_id: pid }), [pid]);
  const rows = cl.data ?? [];
  const total = rows.reduce((s, r) => s + (r.num_samples ?? 0), 0);
  return (
    <Page title="Client management" sub="Per-client sample counts and train/validation/test splits of a partition manifest. Quantity imbalance is visible directly in the chart.">
      <div className="row">
        <label>Partition
          <select value={pid} onChange={(e) => setPid(e.target.value)}>
            {(parts.data ?? []).map((p) => <option key={p.partition_id} value={p.partition_id}>{p.partition_id}</option>)}
          </select>
        </label>
        <span className="note">{rows.length} clients · {total} samples</span>
      </div>
      <Status loading={cl.loading} error={cl.error} empty={!cl.loading && !cl.error && rows.length === 0} />
      {rows.length > 0 && (
        <div className="grid g2">
          <Card title="Samples per client" desc="Total samples assigned to each client before the train/validation/test split.">
            <div className="chartbox"><ResponsiveContainer><BarChart data={rows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="client_id" hide /><YAxis stroke="var(--muted)" fontSize={12} />
              <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)" }} />
              <Bar dataKey="num_samples" fill="var(--c1)" isAnimationActive={false} />
            </BarChart></ResponsiveContainer></div>
          </Card>
          <Card title="Client table" desc="Split sizes per client.">
            <div className="scroll" style={{ maxHeight: 300 }}><table>
              <thead><tr><th>Client</th><th className="num">Total</th><th className="num">Train</th><th className="num">Val</th><th className="num">Test</th></tr></thead>
              <tbody>{rows.map((r) => <tr key={r.client_id}><td>{r.client_id}</td><td className="num">{r.num_samples}</td>
                <td className="num">{r.train_samples}</td><td className="num">{r.validation_samples}</td><td className="num">{r.test_samples}</td></tr>)}</tbody>
            </table></div>
          </Card>
        </div>
      )}
    </Page>
  );
}
