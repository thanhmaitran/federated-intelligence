import { api } from "../api";
import { Card, Page, Status } from "../components/ui";
import { useAsync } from "../hooks";

export default function Datasets() {
  const ds = useAsync(api.datasets, []);
  const parts = useAsync(api.partitions, []);
  return (
    <Page title="Dataset management" sub="Dataset registry and partition manifests. Raw data never leaves the data/ directory; only manifests and metadata are served.">
      <Card title="Datasets" desc="Synthetic scenarios are generated locally; FEMNIST and Shakespeare are downloaded from Hugging Face by scripts/download_datasets.py.">
        <Status loading={ds.loading} error={ds.error} />
        <div className="scroll"><table>
          <thead><tr><th>Name</th><th>Kind</th><th>Status</th><th className="num">Clients</th><th>Description</th></tr></thead>
          <tbody>{(ds.data ?? []).map((d) => (
            <tr key={d.name}><td>{d.name}</td><td>{d.kind}</td><td><span className={`pill ${d.status}`}>{d.status.replace("_", " ")}</span></td>
              <td className="num">{d.num_clients ?? "–"}</td><td style={{ whiteSpace: "normal" }}>{d.description}</td></tr>))}</tbody>
        </table></div>
      </Card>
      <div style={{ height: 16 }} />
      <Card title="Partition manifests" desc="Each manifest stores per-client sample indices, checksum and split policy.">
        <Status loading={parts.loading} error={parts.error} />
        <table>
          <thead><tr><th>Partition</th><th>Method</th><th className="num">α</th><th className="num">Clients</th><th className="num">Samples</th></tr></thead>
          <tbody>{(parts.data ?? []).map((p) => (
            <tr key={p.partition_id}><td>{p.partition_id}</td><td>{p.method}</td><td className="num">{p.alpha ?? "–"}</td>
              <td className="num">{p.num_clients}</td><td className="num">{p.total_samples}</td></tr>))}</tbody>
        </table>
      </Card>
    </Page>
  );
}
