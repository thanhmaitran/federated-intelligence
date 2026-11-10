# Methodology

## 1. Evaluation protocol

* **Global test set**: fresh samples drawn from the *population* (mixture of all client distributions, or the pooled population for partitioned data), independent of any client's training data. It is the primary metric.
* **Per-client validation/test**: each client keeps a 70/15/15 train/validation/test split. Per-client accuracy distribution (mean, 10th percentile, worst) quantifies fairness.
* **Unseen clients**: extra clients drawn from the same population but never trained on, to measure generalisation to new participants.
* **Model selection** uses validation data only, on pilot seeds disjoint from the final seeds (`configs/experiments/pilot.yaml`; `PILOT-LR`, `PILOT-MU`). Test metrics are reported once, at the final round.
* **Centralized baseline**: same preprocessing, architecture and test set (`scripts/run_baselines.py`).

## 2. Data-generating process

Per client *k*: `y ~ Bernoulli(p_k)`; `x | y ~ N(±(s/2)·u_k + Δ_k, diag(σ_k²) Σ₀)` with label noise η. Heterogeneity sources are independent switches:

| Scenario | p_k | Δ_k, u_k, σ_k | n_k |
|---|---|---|---|
| iid | 0.5 | shared | equal |
| label_skew | Beta(a, b) | shared | equal |
| feature_skew | 0.5 | client-specific shift and concept rotation | equal |
| quantity_skew | 0.5 | shared | clipped log-normal |
| combined_non_iid | Beta | client-specific | log-normal |

Pooled-then-partitioned mode instead draws one population and applies `IIDPartitioner`, `DirichletLabelPartitioner(α)` or `QuantitySkewPartitioner`. All randomness flows from explicit `numpy.random.Generator`s derived from the seed; manifests store seed, config and SHA-256 checksums.

## 3. Heterogeneity metrics

Label JSD (base 2) between each client and the global label distribution, pairwise JSD matrix, standardised feature-mean distance, normalised covariance Frobenius distance, and sample-count coefficient of variation `CV = σ_n / μ_n`. They are measured on the generated data, so heterogeneity is *quantified*, not just labelled.

## 4. Federated algorithms

* **FedAvg**: `w_{t+1} = w_t + η_s Σ_k p_k Δ_k`, `p_k = n_k / Σ_j n_j` over clients that returned a valid update (equivalent to averaging parameters when `η_s = 1`).
* **FedProx**: local loss `L(w) + (μ/2)‖w − w_t‖²`; with `μ = 0` it is exactly FedAvg (tested). μ is tuned on validation data in a pilot.
* **Validation**: shape, dtype, finiteness and weight checks before aggregation; empty rounds leave the model unchanged and are counted (`skipped_rounds`).

## 5. Differential privacy

| Item | Choice |
|---|---|
| Threat model | Central DP: honest-but-curious *analyst* sees only released models; the **server is trusted** with individual clipped updates. Against an untrusted server, local DP or secure aggregation would be required (not implemented). |
| Privacy unit / adjacency | One **client** (all its data); add/remove one client. This is *not* record-level DP. |
| Sampling | Poisson client sampling with rate q (enforced when DP is on) |
| Mechanism | Clip each update to L2 norm C, sum, add N(0, σ²C²I), divide by the fixed expectation q·K |
| Accountant | RDP of the sampled Gaussian mechanism (Mironov et al. 2019), integer orders, conversion of Balle et al. 2020 |
| Verification | Unit tests compare against Opacus' RDP analysis and check closed-form limits |
| Reporting | (ε, δ), C, σ, q, rounds, unit and assumptions; ε is *computed*, and for target-ε experiments σ is found by bisection on the accountant |

Opacus' sample-level guarantee is not reused as a client-level claim. With only 50 clients, strong client-level privacy is intrinsically hard (EXP-05B shows the benefit of more clients).

## 6. Communication

Update codecs: float32 (baseline), float16, symmetric per-tensor int8, top-k sparsification (index + value payload) with optional per-client error feedback. Bytes are the **length of the serialized payload**; uplink and downlink are accounted separately. The network simulator draws log-normal latency/bandwidth per client and reports *simulated* round time (max over participants, with optional deadline) – distinct from wall-clock time.

## 7. Statistical analysis

≥5 independent seeds (8 for the FedAvg-vs-FedProx study). Reported: mean, sample SD (n−1), 95% Student-t CI. Comparisons use the same seeds in both arms (same data, partition, initial weights) → **paired** t-test with Wilcoxon as a check, Cohen's d_z, CI of the difference, Holm correction within each family. A difference is called an improvement only if it passes these checks; with n = 5–8 low power is stated explicitly.

## 8. Threats to validity

Synthetic Gaussian data favours linear/MLP models and may not transfer to image/text; all clients are simulated in one process; hyper-parameters tuned at one heterogeneity level; DP with 50 clients yields large ε; simulated network ≠ real network. See the research report for the full list.
