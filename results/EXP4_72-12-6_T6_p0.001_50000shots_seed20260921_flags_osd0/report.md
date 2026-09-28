# Experiment 4 — [[72, 12, 6]], T=6, p=0.001, 50000 shots

Data file: `EXP4_72-12-6_T6_p0.001_50000shots_seed20260921_flags_osd0.json`  
Exported: 2026-09-28T10:11:10

## Table: metrics

[metrics.csv](metrics.csv) — 5 rows

| code | rounds | p | shots | seed | use_flags | osd_order | workers | x_detectors | policy | failures | ler | ci_low | ci_high | escalation_rate | silent_failures | unconverged_failures | escalated_failures | work_mean | work_p99 | time_p50_us | time_p99_us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [[72, 12, 6]] | 6 | 0.001 | 50000 | 20260921 | True | 0 | 12 | False | never | 9894 | 0.1979 | 0.1944 | 0.2014 | 0 | 0 | 9894 | 0 | 10.05 | 31 | 12.9 | 38.3 |
| [[72, 12, 6]] | 6 | 0.001 | 50000 | 20260921 | True | 0 | 12 | False | always | 124 | 0.00248 | 0.002081 | 0.002956 | 1 | 0 | 0 | 124 | 7.823 | 20 | 832.6 | 5490 |
| [[72, 12, 6]] | 6 | 0.001 | 50000 | 20260921 | True | 0 | 12 | False | primary_fail | 114 | 0.00228 | 0.001898 | 0.002738 | 0.2465 | 0 | 0 | 114 | 13.29 | 50 | 14 | 5224 |
| [[72, 12, 6]] | 6 | 0.001 | 50000 | 20260921 | True | 0 | 12 | False | flag | 124 | 0.00248 | 0.002081 | 0.002956 | 0.713 | 0 | 0 | 124 | 9.146 | 20 | 801.2 | 5494 |
| [[72, 12, 6]] | 6 | 0.001 | 50000 | 20260921 | True | 0 | 12 | False | flag_or_fail | 124 | 0.00248 | 0.002081 | 0.002956 | 0.713 | 0 | 0 | 124 | 9.146 | 20 | 801.2 | 5494 |

## Figure: ablation

![ablation](ablation.png)  
Vector version: [ablation.pdf](ablation.pdf)

## Figure: outcomes

![outcomes](outcomes.png)  
Vector version: [outcomes.pdf](outcomes.pdf)

## Figure: tradeoff

![tradeoff](tradeoff.png)  
Vector version: [tradeoff.pdf](tradeoff.pdf)

## Figure: latency

![latency](latency.png)  
Vector version: [latency.pdf](latency.pdf)
