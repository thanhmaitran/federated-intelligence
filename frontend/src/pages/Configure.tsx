import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { Card, Page } from "../components/ui";

type F = Record<string, string>;
const DEFAULTS: F = {
  name: "ui_run", seed: "1", algorithm: "fedavg", mu: "0", partition: "dirichlet", alpha: "0.5", num_clients: "50",
  total_samples: "30000", rounds: "20", participation: "0.5", local_epochs: "2", lr: "0.1", batch: "64",
  model: "mlp", compressor: "none", dp: "off", clip: "1.0", noise: "1.0",
};

function build(f: F): Record<string, unknown> {
  const cfg: Record<string, any> = {
    experiment: { name: f.name, seed: +f.seed },
    dataset: { name: "synthetic_federated_tabular", mode: "pooled", partition: f.partition, alpha: +f.alpha,
      num_clients: +f.num_clients, total_samples: +f.total_samples },
    model: { name: f.model, ...(f.model === "mlp" ? { hidden: [64, 32] } : {}) },
    federated: { algorithm: f.algorithm, mu: f.algorithm === "fedprox" ? +f.mu : 0, num_rounds: +f.rounds,
      participation_rate: +f.participation, selection: f.dp === "on" ? "poisson" : "random" },
    training: { local_epochs: +f.local_epochs, learning_rate: +f.lr, batch_size: +f.batch },
    communication: { compressor: f.compressor },
    privacy: f.dp === "on" ? { enabled: true, clip_norm: +f.clip, noise_multiplier: +f.noise } : { enabled: false },
  };
  return cfg;
}

export default function Configure() {
  const [f, setF] = useState<F>(DEFAULTS);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const nav = useNavigate();
  const set = (k: string) => (e: { target: { value: string } }) => setF({ ...f, [k]: e.target.value });
  const input = (k: string, label: string, type = "text") => (
    <label>{label}<input type={type} value={f[k]} onChange={set(k)} step="any" /></label>
  );
  const select = (k: string, label: string, opts: string[]) => (
    <label>{label}<select value={f[k]} onChange={set(k)}>{opts.map((o) => <option key={o}>{o}</option>)}</select></label>
  );
  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const run = await api.create(build(f));
      nav(`/monitor?id=${encodeURIComponent(run.id)}`);
    } catch (ex) {
      setErr((ex as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Page title="Experiment configuration" sub="Launch a new federated run from the browser. The configuration is validated by the same schema the command-line runner uses; the run executes on the backend and can be followed live in the training monitor.">
      <form onSubmit={submit}>
        <Card>
          <div className="form">
            <h4>Experiment</h4>{input("name", "Name")}{input("seed", "Seed", "number")}{select("model", "Model", ["mlp", "logreg"])}
            <h4>Data</h4>{select("partition", "Partition", ["iid", "dirichlet"])}{input("alpha", "Dirichlet α", "number")}{input("num_clients", "Clients", "number")}
            {input("total_samples", "Total samples", "number")}
            <h4>Federated optimisation</h4>{select("algorithm", "Algorithm", ["fedavg", "fedprox"])}{input("mu", "FedProx μ", "number")}{input("rounds", "Rounds", "number")}
            {input("participation", "Participation rate", "number")}{input("local_epochs", "Local epochs", "number")}{input("lr", "Learning rate", "number")}{input("batch", "Batch size", "number")}
            <h4>Privacy &amp; communication</h4>{select("dp", "Client-level DP", ["off", "on"])}{input("clip", "Clip norm C", "number")}{input("noise", "Noise multiplier σ", "number")}
            {select("compressor", "Uplink compression", ["none", "float16", "int8", "topk"])}
          </div>
          <div className="row" style={{ marginTop: 18, marginBottom: 0 }}>
            <button disabled={busy}>{busy ? "Submitting…" : "Run experiment"}</button>
            <span className="note">A 20-round run on the synthetic dataset takes ~10 s.</span>
          </div>
          {err && <div className="err" style={{ marginTop: 12 }}>{err}</div>}
        </Card>
      </form>
    </Page>
  );
}
