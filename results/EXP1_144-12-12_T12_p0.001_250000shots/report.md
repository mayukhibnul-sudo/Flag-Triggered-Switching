# Experiment 1 validation — [[144, 12, 12]], T=12, p=0.001

Data file: `EXP1_144-12-12_T12_p0.001_250000shots.json`  
Exported: 2026-09-29T19:57:19

Literature source: Pakhunov (2026), Table II, [[144,12,12]], p=1e-3, T=12

## Table: checks

[checks.csv](checks.csv) — 4 rows

| check | value | detail |
|---|---|---|
| Check 1: fault-model density | 1.4× | 8,784 faults vs 6,192 from n(wT + T/2 + 1) |
| Check 2: reproduces literature | no | ours 0.306 [0.304, 0.308] vs 0.879 (Pakhunov (2026), Table II, [[144,12,12]], p=1e-3, T=12) |
| Check 3: accuracy cost of flags | flags improve accuracy | LER 1.64e-04 with flags vs 9.76e-04 without; +2,592 faults, +864 detectors; flags fire on 98.9% of shots |
| silent failures (weak decoder) | 0 flagged / 0 unflagged | the only failures a flag trigger could ever catch |

## Figure: validation

![validation](validation.png)  
Vector version: [validation.pdf](validation.pdf)
