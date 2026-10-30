# Experiment 9 dynamic prior updating — [[144, 12, 12]], T=12, p=0.001

Data file: `EXP9_144-12-12_T12_p0.001_250000shots_seed20260930_osd0.json`  
Exported: 2026-10-27T17:09:49

**Prior updating recovers 107% of the gap between the blind and sighted arms**, on a check matrix 936 × 11376 rather than 1800 × 11376 — 1.92× fewer entries.  
It matches the sighted arm within 5%, so the flag rows buy nothing that the exact one-shot posterior does not already capture.

## Table: arms

[arms.csv](arms.csv) — 3 rows

| arm | failures | ler | ci_low | ci_high | time_p50_us | time_p99_us | detectors | faults |
|---|---|---|---|---|---|---|---|---|
| blind | 473 | 0.001892 | 0.001729 | 0.00207 | 1.011e+04 | 1.483e+04 | 936 | 11376 |
| updated | 0 | 0 | 0 | 1.537e-05 | 2731 | 1.15e+04 | 936 | 11376 |
| sighted | 31 | 0.000124 | 8.736e-05 | 0.000176 | 1.096e+04 | 1.593e+04 | 1800 | 11376 |

## Figure: prior_update

![prior_update](prior_update.png)  
Vector version: [prior_update.pdf](prior_update.pdf)
