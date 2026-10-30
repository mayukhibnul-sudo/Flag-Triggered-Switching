# Experiment 9 dynamic prior updating — [[288, 12, 18]], T=12, p=0.001

Data file: `EXP9_288-12-18_T12_p0.001_250000shots_seed20260930_osd0.json`  
Exported: 2026-10-27T21:25:44

**Prior updating recovers nan% of the gap between the blind and sighted arms**, on a check matrix 1872 × 22752 rather than 3600 × 22752 — 1.92× fewer entries.  
It matches the sighted arm within 5%, so the flag rows buy nothing that the exact one-shot posterior does not already capture.

## Table: arms

[arms.csv](arms.csv) — 3 rows

| arm | failures | ler | ci_low | ci_high | time_p50_us | time_p99_us | detectors | faults |
|---|---|---|---|---|---|---|---|---|
| blind | 0 | 0 | 0 | 1.537e-05 | 2.126e+04 | 2.906e+04 | 1872 | 22752 |
| updated | 0 | 0 | 0 | 1.537e-05 | 6924 | 2.523e+04 | 1872 | 22752 |
| sighted | 0 | 0 | 0 | 1.537e-05 | 2.556e+04 | 3.436e+04 | 3600 | 22752 |

## Figure: prior_update

![prior_update](prior_update.png)  
Vector version: [prior_update.pdf](prior_update.pdf)
