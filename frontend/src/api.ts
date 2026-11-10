// Thin typed client of the FastAPI backend. Every number shown in the dashboard comes from here.
const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 300)}`);
  }
  return res.json() as Promise<T>;
}

export interface Overview {
  num_experiments: number;
  by_status: Record<string, number>;
  by_group: Record<string, number>;
  best_test_accuracy: number | null;
  total_bytes_transferred: number;
  datasets: number;
}
export interface DatasetInfo {
  name: string;
  kind: string;
  description: string;
  status: string;
  num_clients: number | null;
  metadata: Record<string, unknown>;
}
export interface PartitionInfo {
  partition_id: string;
  dataset: string;
  method: string;
  alpha: number | null;
  num_clients: number;
  total_samples: number;
}
export interface ClientInfo {
  client_id: string;
  num_samples: number | null;
  train_samples: number | null;
  validation_samples: number | null;
  test_samples: number | null;
  metrics: Record<string, number>;
}
export interface ExperimentSummary {
  id: string;
  exp_group: string;
  variant: string;
  seed: number;
  status: string;
  source: string;
  algorithm: string | null;
  test_accuracy: number | null;
  total_bytes: number | null;
  epsilon: number | null;
  created_at: string;
  finished_at: string | null;
}
export interface ExperimentDetail extends ExperimentSummary {
  config: Record<string, any>;
  summary: Record<string, any>;
  error: string;
}
export interface ComparisonRow {
  key: Record<string, string | number | null>;
  n: number;
  mean: number;
  sd: number;
  ci_low: number;
  ci_high: number;
}
export type MetricSeries = Record<string, { round: number; value: number }[]>;

const enc = (id: string) => id.split("/").map(encodeURIComponent).join("/");

export const api = {
  overview: () => request<Overview>("/overview"),
  datasets: () => request<DatasetInfo[]>("/datasets"),
  partitions: () => request<PartitionInfo[]>("/partitions"),
  clients: (q: { partition_id?: string; experiment_id?: string }) => {
    const p = new URLSearchParams(Object.entries(q).filter(([, v]) => v) as [string, string][]);
    return request<ClientInfo[]>(`/clients?${p}`);
  },
  groups: () => request<{ group: string; runs: number }[]>("/analysis/groups"),
  experiments: (q: { group?: string; status?: string; limit?: number } = {}) => {
    const p = new URLSearchParams(
      Object.entries({ limit: 2000, ...q }).filter(([, v]) => v !== undefined && v !== "") as [string, string][],
    );
    return request<ExperimentSummary[]>(`/experiments?${p}`);
  },
  experiment: (id: string) => request<ExperimentDetail>(`/experiments/${enc(id)}`),
  metrics: (id: string) =>
    request<{ experiment_id: string; status: string; metrics: MetricSeries }>(`/experiments/${enc(id)}/metrics`),
  compare: (group: string, metric: string, by: string) =>
    request<ComparisonRow[]>(
      `/analysis/compare?group=${encodeURIComponent(group)}&metric=${metric}&by=${encodeURIComponent(by)}`,
    ),
  create: (config: Record<string, unknown>, group = "UI") =>
    request<ExperimentDetail>("/experiments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config, group }),
    }),
};
