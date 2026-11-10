# Auto-generated results summary

All numbers are computed from `reports/experiment-results/` by `scripts/generate_report.py`. `n` is the number of independent seeds; ± is the sample SD; CI = 95% Student-t. Paired tests compare arms on the same seeds (same data, partition and initial weights) and are Holm-corrected within each family.

Centralized reference (pooled training data, same model/optimiser): mlp = 0.8576, logreg = 0.8697 test accuracy.

### EXP-01 – IID vs Dirichlet(0.5), FedAvg

Same population, same seeds; the only difference is the partition.

| partition | n | test acc | macro-F1 | worst-client acc | rounds→0.80 val acc | acc-curve AUC |
|---|---|---|---|---|---|---|
| dirichlet0.5 | 5 | 0.8604 ± 0.0091 | 0.8602 ± 0.0092 | 0.5636 ± 0.1022 | 4.8000 | 0.8439 |
| iid | 5 | 0.8668 ± 0.0099 | 0.8668 ± 0.01 | 0.7867 ± 0.0183 | 4.8000 | 0.8518 |


### EXP-01 – paired IID − Dirichlet(0.5) (test accuracy)



| g | a | b | metric | n | mean_a | mean_b | mean_diff | ci_low | ci_high | cohens_dz | p_value | wilcoxon_p | p_holm | significant_holm_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | iid | dirichlet0.5 | test_accuracy | 5 | 0.8668 | 0.8604 | 0.0065 | -0.0010 | 0.0140 | 1.0709 | 0.0748 | 0.1250 | 0.0748 | False |


### EXP-01B – one heterogeneity source at a time (native generator)

Heterogeneity statistics are measured on seed-42 data (reports/tables/heterogeneity_by_scenario.csv).

| scenario | n | test acc | worst-client acc | label JSD | feature-mean dist | sample CV |
|---|---|---|---|---|---|---|
| iid | 5 | 0.8689 ± 0.0007 | 0.7918 ± 0.0134 | 0.0010 | 0.3290 | 0.0000 |
| label_skew | 5 | 0.6593 ± 0.0219 | 0.4082 ± 0.0214 | 0.2740 | 0.3290 | 0.0000 |
| feature_skew | 5 | 0.7492 ± 0.0068 | 0.4639 ± 0.0242 | 0.0010 | 7.0880 | 0.0000 |
| quantity_skew | 5 | 0.8684 ± 0.0031 | 0.7488 ± 0.0412 | 0.0020 | 0.3970 | 0.3510 |
| combined_non_iid | 5 | 0.6506 ± 0.0256 | 0.0717 ± 0.0565 | 0.2660 | 7.2070 | 0.3170 |


### EXP-01B – paired difference to the IID scenario (test accuracy)

Caveat: each scenario has its own data-generating process, so this contrasts scenarios, not a single treatment.

| scenario | n | mean_diff_vs_iid | ci_low | ci_high | d_z | p | p_holm |
|---|---|---|---|---|---|---|---|
| label_skew | 5 | -0.2096 | -0.2365 | -0.1827 | -9.6786 | 0.0000 | 0.0001 |
| feature_skew | 5 | -0.1197 | -0.1281 | -0.1112 | -17.5749 | 0.0000 | 0.0000 |
| quantity_skew | 5 | -0.0005 | -0.0043 | 0.0034 | -0.1548 | 0.7466 | 0.7466 |
| combined_non_iid | 5 | -0.2183 | -0.2500 | -0.1866 | -8.5509 | 0.0000 | 0.0001 |


### EXP-02 – Dirichlet α × algorithm (FedProx μ = 1.0)



| alpha | algorithm | n | test acc | worst-client acc | rounds→0.80 | curve AUC | volatility |
|---|---|---|---|---|---|---|---|
| 10.0000 | fedavg | 8 | 0.8672 ± 0.0082 | 0.778 ± 0.01 | 4.5000 | 0.8526 | 0.0011 |
| 10.0000 | fedprox | 8 | 0.8662 ± 0.0082 | 0.7811 ± 0.0115 | 7.6000 | 0.8362 | 0.0011 |
| 1.0000 | fedavg | 8 | 0.8627 ± 0.0092 | 0.688 ± 0.0717 | 4.1000 | 0.8505 | 0.0032 |
| 1.0000 | fedprox | 8 | 0.8641 ± 0.0084 | 0.7127 ± 0.0827 | 7.8000 | 0.8314 | 0.0021 |
| 0.5000 | fedavg | 8 | 0.8602 ± 0.0082 | 0.5814 ± 0.096 | 4.4000 | 0.8445 | 0.0067 |
| 0.5000 | fedprox | 8 | 0.8628 ± 0.0078 | 0.5226 ± 0.2325 | 7.9000 | 0.8265 | 0.0048 |
| 0.1000 | fedavg | 8 | 0.8212 ± 0.0526 | 0.1224 ± 0.2342 | 6.2000 | 0.7948 | 0.0477 |
| 0.1000 | fedprox | 8 | 0.8382 ± 0.0352 | 0.0907 ± 0.2565 | 10.6000 | 0.7858 | 0.0262 |


### EXP-02 – paired FedProx − FedAvg, final test accuracy

Holm-corrected across the four α values.

| alpha | a | b | metric | n | mean_a | mean_b | mean_diff | ci_low | ci_high | cohens_dz | p_value | wilcoxon_p | p_holm | significant_holm_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1000 | fedprox | fedavg | test_accuracy | 8 | 0.8382 | 0.8212 | 0.0170 | 0.0018 | 0.0322 | 0.9330 | 0.0335 | 0.0156 | 0.1339 | False |
| 0.5000 | fedprox | fedavg | test_accuracy | 8 | 0.8628 | 0.8602 | 0.0025 | -0.0001 | 0.0052 | 0.7949 | 0.0594 | 0.0781 | 0.1781 | False |
| 10.0000 | fedprox | fedavg | test_accuracy | 8 | 0.8662 | 0.8672 | -0.0009 | -0.0023 | 0.0005 | -0.5529 | 0.1619 | 0.3281 | 0.3202 | False |
| 1.0000 | fedprox | fedavg | test_accuracy | 8 | 0.8641 | 0.8627 | 0.0013 | -0.0007 | 0.0034 | 0.5555 | 0.1601 | 0.2188 | 0.3202 | False |


### EXP-02 – paired FedProx − FedAvg, accuracy-curve AUC (convergence speed)



| alpha | a | b | metric | n | mean_a | mean_b | mean_diff | ci_low | ci_high | cohens_dz | p_value | wilcoxon_p | p_holm | significant_holm_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1000 | fedprox | fedavg | auc_accuracy_curve | 8 | 0.7858 | 0.7948 | -0.0090 | -0.0192 | 0.0012 | -0.7367 | 0.0757 | 0.1094 | 0.0757 | False |
| 0.5000 | fedprox | fedavg | auc_accuracy_curve | 8 | 0.8265 | 0.8445 | -0.0180 | -0.0225 | -0.0135 | -3.3521 | 0.0000 | 0.0078 | 0.0001 | True |
| 10.0000 | fedprox | fedavg | auc_accuracy_curve | 8 | 0.8362 | 0.8526 | -0.0164 | -0.0204 | -0.0124 | -3.4166 | 0.0000 | 0.0078 | 0.0001 | True |
| 1.0000 | fedprox | fedavg | auc_accuracy_curve | 8 | 0.8314 | 0.8505 | -0.0191 | -0.0247 | -0.0135 | -2.8555 | 0.0001 | 0.0078 | 0.0002 | True |


### EXP-02B – FedProx μ sensitivity (μ = 0 is FedAvg)



| alpha | mu | n | test acc | curve AUC | volatility |
|---|---|---|---|---|---|
| 0.1000 | 0.0100 | 5 | 0.8335 ± 0.0307 | 0.8081 | 0.0316 |
| 0.1000 | 0.1000 | 5 | 0.838 ± 0.029 | 0.8118 | 0.0271 |
| 0.1000 | 0.0000 | 5 | 0.8334 ± 0.0306 | 0.8071 | 0.0326 |
| 0.1000 | 10.0000 | 5 | 0.7485 ± 0.0385 | 0.6398 | 0.0217 |
| 0.1000 | 1.0000 | 5 | 0.8458 ± 0.0195 | 0.7959 | 0.0163 |
| 0.1000 | 3.0000 | 5 | 0.8438 ± 0.021 | 0.7547 | 0.0133 |
| 0.5000 | 0.0100 | 5 | 0.8606 ± 0.0089 | 0.8438 | 0.0068 |
| 0.5000 | 0.1000 | 5 | 0.8616 ± 0.0092 | 0.8429 | 0.0061 |
| 0.5000 | 0.0000 | 5 | 0.8604 ± 0.0091 | 0.8439 | 0.0068 |
| 0.5000 | 10.0000 | 5 | 0.7598 ± 0.0368 | 0.6425 | 0.0208 |
| 0.5000 | 1.0000 | 5 | 0.8631 ± 0.0091 | 0.8259 | 0.0049 |
| 0.5000 | 3.0000 | 5 | 0.8611 ± 0.0072 | 0.7785 | 0.0052 |


### EXP-03 – participation rate × algorithm (α = 0.5)



| rate | algorithm | n | test acc | rounds→0.80 | client updates | client compute s | total MB |
|---|---|---|---|---|---|---|---|
| 0.1000 | fedavg | 5 | 0.8282 ± 0.0414 | 4.8000 | 200.0000 | 1.5000 | 6.0300 |
| 0.1000 | fedprox | 5 | 0.8375 ± 0.0208 | 10.4000 | 200.0000 | 3.8000 | 6.0300 |
| 0.2500 | fedavg | 5 | 0.8396 ± 0.0219 | 4.6000 | 480.0000 | 3.1000 | 14.4700 |
| 0.2500 | fedprox | 5 | 0.8478 ± 0.0129 | 9.8000 | 480.0000 | 4.7000 | 14.4700 |
| 0.5000 | fedavg | 5 | 0.8604 ± 0.0091 | 4.8000 | 1000.0000 | 7.8000 | 30.1500 |
| 0.5000 | fedprox | 5 | 0.8631 ± 0.0091 | 7.6000 | 1000.0000 | 10.7000 | 30.1500 |
| 0.7500 | fedavg | 5 | 0.8615 ± 0.0073 | 4.0000 | 1520.0000 | 6.9000 | 45.8300 |
| 0.7500 | fedprox | 5 | 0.864 ± 0.0069 | 8.2000 | 1520.0000 | 14.9000 | 45.8300 |
| 1.0000 | fedavg | 5 | 0.8642 ± 0.0078 | 3.8000 | 2000.0000 | 8.8000 | 60.3000 |
| 1.0000 | fedprox | 5 | 0.8649 ± 0.0095 | 7.6000 | 2000.0000 | 16.1000 | 60.3000 |


### EXP-07 – client dropout (participation 0.5, α = 0.5)



| dropout | algorithm | n | test acc | updates received | skipped rounds | rounds→0.80 |
|---|---|---|---|---|---|---|
| 0.1000 | fedavg | 5 | 0.8607 ± 0.0087 | 894.0000 | 0.0000 | 4.0000 |
| 0.1000 | fedprox | 5 | 0.8629 ± 0.0085 | 894.0000 | 0.0000 | 8.0000 |
| 0.3000 | fedavg | 5 | 0.8578 ± 0.0071 | 694.0000 | 0.0000 | 4.8000 |
| 0.3000 | fedprox | 5 | 0.8623 ± 0.0054 | 694.0000 | 0.0000 | 8.2000 |
| 0.5000 | fedavg | 5 | 0.859 ± 0.0084 | 486.0000 | 0.0000 | 4.4000 |
| 0.5000 | fedprox | 5 | 0.8609 ± 0.0108 | 486.0000 | 0.0000 | 8.2000 |
| 0.7000 | fedavg | 5 | 0.8245 ± 0.0319 | 290.0000 | 0.0000 | 6.2000 |
| 0.7000 | fedprox | 5 | 0.833 ± 0.0232 | 290.0000 | 0.0000 | 9.4000 |
| 0.0000 | fedavg | 5 | 0.8604 ± 0.0091 | 1000.0000 | 0.0000 | 4.8000 |
| 0.0000 | fedprox | 5 | 0.8631 ± 0.0091 | 1000.0000 | 0.0000 | 7.6000 |


### EXP-04 – privacy budget sweep (logistic regression, δ = 1e-5, client-level central DP, RDP accountant)

ε is computed by the accountant for the noise actually used; 'nodp' has ε = ∞, 'clip_only' clips but adds no noise (ε = ∞).

| variant | n | σ (noise mult.) | ε (accountant) | test acc | macro-F1 | test loss |
|---|---|---|---|---|---|---|
| nodp | 5 |  |  | 0.868 ± 0.0092 | 0.8679 ± 0.0093 | 0.2957 ± 0.0126 |
| clip_only | 5 | 0.0000 |  | 0.8699 ± 0.0091 | 0.8699 ± 0.0092 | 0.2953 ± 0.0122 |
| eps50 | 5 | 0.6980 | 49.8900 | 0.8658 ± 0.0074 | 0.8658 ± 0.0074 | 0.2995 ± 0.0116 |
| eps20 | 5 | 1.1540 | 19.9800 | 0.8608 ± 0.0093 | 0.8607 ± 0.0092 | 0.3086 ± 0.0121 |
| eps10 | 5 | 1.8640 | 10.0000 | 0.8493 ± 0.0078 | 0.8492 ± 0.0078 | 0.3327 ± 0.0151 |
| eps5 | 5 | 3.1950 | 5.0000 | 0.8309 ± 0.0102 | 0.8308 ± 0.0102 | 0.4108 ± 0.0291 |
| eps2 | 5 | 6.9700 | 2.0000 | 0.7958 ± 0.0185 | 0.7955 ± 0.0183 | 0.851 ± 0.1133 |


### EXP-05 – clipping norm × noise multiplier

ε depends only on σ, sampling rate and rounds (not on C).

| clip C | σ | n | ε | test acc |
|---|---|---|---|---|
| 0.1000 | 0.5000 | 5 | 116.8100 | 0.8678 ± 0.0112 |
| 0.1000 | 1.0000 | 5 | 24.4200 | 0.8675 ± 0.0107 |
| 0.1000 | 2.0000 | 5 | 9.2000 | 0.8658 ± 0.0105 |
| 0.5000 | 0.5000 | 5 | 116.8100 | 0.8687 ± 0.0088 |
| 0.5000 | 1.0000 | 5 | 24.4200 | 0.8668 ± 0.0079 |
| 0.5000 | 2.0000 | 5 | 9.2000 | 0.8625 ± 0.0087 |
| 1.0000 | 0.5000 | 5 | 116.8100 | 0.8674 ± 0.008 |
| 1.0000 | 1.0000 | 5 | 24.4200 | 0.8633 ± 0.009 |
| 1.0000 | 2.0000 | 5 | 9.2000 | 0.8474 ± 0.0078 |
| 2.0000 | 0.5000 | 5 | 116.8100 | 0.8633 ± 0.009 |
| 2.0000 | 1.0000 | 5 | 24.4200 | 0.8474 ± 0.0078 |
| 2.0000 | 2.0000 | 5 | 9.2000 | 0.8218 ± 0.0111 |


### EXP-05B – number of clients at fixed ε



| clients | n | σ | ε | test acc |
|---|---|---|---|---|
| K200 | 5 | 1.0390 | 5.0000 | 0.8515 ± 0.0087 |
| K50 | 5 | 3.1950 | 5.0000 | 0.8309 ± 0.0102 |


### EXP-08 – heterogeneity + DP (ε = 10), FedAvg vs FedProx



| partition | algorithm | n | ε | test acc |
|---|---|---|---|---|
| alpha0.1 | fedavg | 5 | 10.0000 | 0.8437 ± 0.013 |
| alpha0.1 | fedprox | 5 | 10.0000 | 0.8217 ± 0.0138 |
| alpha0.5 | fedavg | 5 | 10.0000 | 0.8493 ± 0.0078 |
| alpha0.5 | fedprox | 5 | 10.0000 | 0.8338 ± 0.0112 |
| iid | fedavg | 5 | 10.0000 | 0.8532 ± 0.0086 |
| iid | fedprox | 5 | 10.0000 | 0.841 ± 0.012 |


### EXP-08 – paired FedProx − FedAvg



| axis_alpha | a | b | metric | n | mean_a | mean_b | mean_diff | ci_low | ci_high | cohens_dz | p_value | wilcoxon_p | p_holm | significant_holm_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alpha0.1 | fedprox | fedavg | test_accuracy | 5 | 0.8217 | 0.8437 | -0.0220 | -0.0354 | -0.0085 | -2.0279 | 0.0105 | 0.0625 | 0.0316 | True |
| alpha0.5 | fedprox | fedavg | test_accuracy | 5 | 0.8338 | 0.8493 | -0.0155 | -0.0253 | -0.0056 | -1.9526 | 0.0120 | 0.0625 | 0.0316 | True |
| iid | fedprox | fedavg | test_accuracy | 5 | 0.8410 | 0.8532 | -0.0123 | -0.0219 | -0.0027 | -1.5910 | 0.0236 | 0.0625 | 0.0316 | True |


### EXP-06 – communication compression (measured serialized payloads)

Ratios are computed from the byte length of real serialized payloads (header included).

| variant | n | bytes / update | uplink ratio | total MB (up+down) | test acc | curve AUC |
|---|---|---|---|---|---|---|
| float32 | 5 | 15009.0000 | 1.0000 | 30.1500 | 0.8604 ± 0.0091 | 0.8439 |
| float16 | 5 | 7517.0000 | 2.0000 | 22.6600 | 0.8606 ± 0.009 | 0.8439 |
| int8 | 5 | 4028.0000 | 3.7300 | 19.1700 | 0.8604 ± 0.0091 | 0.8439 |
| topk10 | 5 | 3067.0000 | 4.8900 | 18.2100 | 0.8622 ± 0.0078 | 0.8418 |
| topk10_ef | 5 | 3067.0000 | 4.8900 | 18.2100 | 0.8605 ± 0.0083 | 0.8416 |
| topk1 | 5 | 363.0000 | 41.3500 | 15.5000 | 0.8645 ± 0.0084 | 0.8221 |
| topk1_ef | 5 | 363.0000 | 41.3500 | 15.5000 | 0.8629 ± 0.0078 | 0.8252 |
| int8_down16 | 5 | 4028.0000 | 3.7300 | 11.6800 | 0.8605 ± 0.0091 | 0.8439 |


### EXP-06 – paired accuracy difference to float32



| variant | mean_diff_vs_float32 | ci_low | ci_high | p | p_holm |
|---|---|---|---|---|---|
| float32 | 0.0000 |  |  |  |  |
| float16 | 0.0002 | -0.0002 | 0.0006 | 0.1778 | 0.7112 |
| int8 | 0.0001 | -0.0003 | 0.0004 | 0.5415 | 1.0000 |
| topk10 | 0.0018 | -0.0007 | 0.0044 | 0.1122 | 0.5611 |
| topk10_ef | 0.0001 | -0.0016 | 0.0018 | 0.8541 | 1.0000 |
| topk1 | 0.0041 | -0.0011 | 0.0093 | 0.0924 | 0.5542 |
| topk1_ef | 0.0025 | -0.0003 | 0.0054 | 0.0712 | 0.4983 |
| int8_down16 | 0.0002 | -0.0001 | 0.0004 | 0.1778 | 0.7112 |


### EXP-09 – simulated slow network with round deadline

Simulated time is the network model's clock (latency + size/bandwidth + compute); wall-clock is real elapsed time. They are different quantities.

| deadline | codec | n | test acc | updates received | skipped rounds | simulated time (s) | wall-clock (s) |
|---|---|---|---|---|---|---|---|
| ddl0.8 | float32 | 5 | 0.8264 ± 0.03 | 318.0000 | 0.0000 | 59.9000 | 7.6000 |
| ddl0.8 | int8 | 5 | 0.8449 ± 0.0133 | 441.0000 | 0.0000 | 52.2000 | 7.2000 |
| ddl1.2 | float32 | 5 | 0.8508 ± 0.0122 | 553.0000 | 0.0000 | 67.8000 | 8.4000 |
| ddl1.2 | int8 | 5 | 0.8531 ± 0.0162 | 653.0000 | 0.0000 | 61.6000 | 8.5000 |
| ddl2.0 | float32 | 5 | 0.8544 ± 0.0101 | 748.0000 | 0.0000 | 90.1000 | 8.0000 |
| ddl2.0 | int8 | 5 | 0.8553 ± 0.0093 | 817.0000 | 0.0000 | 86.6000 | 8.1000 |
| noddl | float32 | 5 | 0.8604 ± 0.0088 | 953.0000 | 0.0000 | 187.2000 | 8.2000 |
| noddl | int8 | 5 | 0.8605 ± 0.0087 | 953.0000 | 0.0000 | 151.0000 | 8.1000 |

