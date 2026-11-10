# Experiment matrix and status

| ID | Question | Variables | Algorithms | Seeds | Status |
|---|---|---|---|---|---|
| EXP-00 | Sanity | IID baseline | FedAvg | 5 | done |
| EXP-01 | RQ1 | IID vs Dirichlet(0.5) | FedAvg | 5 | done |
| EXP-01B | RQ1 | label / feature / quantity / combined skew (native) | FedAvg | 5 | done |
| EXP-02 | RQ2 | Dirichlet α ∈ {10, 1, 0.5, 0.1} | FedAvg, FedProx(μ=1) | 8 | done |
| EXP-02B | RQ2 | FedProx μ sweep | FedProx | 5 | done |
| EXP-03 | RQ5 | participation ∈ {0.1 … 1.0} | FedAvg, FedProx | 5 | done |
| EXP-04 | RQ3 | privacy budget ε ∈ {2 … 50}, no-DP, clip-only | FedAvg + DP | 5 | done |
| EXP-05 / 05B | RQ3 | clip norm × noise multiplier; number of clients | FedAvg + DP | 5 | done |
| EXP-06 | RQ4 | codec (fp16, int8, top-k, EF, downlink) | FedAvg | 5 | done |
| EXP-07 | RQ5 | dropout ∈ {0 … 0.7} | FedAvg, FedProx | 5 | done |
| EXP-08 | RQ2+RQ3 | heterogeneity × DP (ε=10) | FedAvg, FedProx | 5 | done |
| EXP-09 | RQ4+RQ5 | slow network + round deadline, fp32 vs int8 | FedAvg | 5 | done |
| FEMNIST / Shakespeare | – | natural partitions | FedAvg | – | **pending: datasets not downloadable in the dev sandbox** |

Execution order followed: smoke tests (`configs/experiments/smoke.yaml`) → pilots on separate seeds (`PILOT-LR`, `PILOT-MU`; validation accuracy only) → full runs. Pilot outcome (4 pilot seeds, validation accuracy): learning rates 0.02–0.2 are within 0.002 of each other (best 0.05: 0.8559; 0.1 used: 0.8550). FedProx on the pilot's strongly non-IID setting: FedAvg 0.7839, μ=0.1 0.7938, μ=1 0.8114, μ=3 0.8186, μ=10 0.7409. μ = 1 was used in the final suites (μ = 3 scored marginally higher in the pilot but converges more slowly; both are covered by EXP-02B).

Result files: `reports/experiment-results/<EXP>/<variant>__seed<k>.json`. Tables: `reports/tables/`. Figures: `reports/figures/experiments/` (captions in `captions.json`).
