# Experiment 1 validation — [[72, 12, 6]], T=6, p=0.001

Data file: `EXP1_72-12-6_T6_p0.001_50000shots.json`  
Exported: 2026-09-28T10:16:12

Literature source: Pakhunov (2026), Table II, [[72,12,6]], p=1e-3, T=12

## Table: checks

[checks.csv](checks.csv) — 4 rows

| check | value | detail |
|---|---|---|
| Check 1: fault-model density | 1.4× | 2,232 faults vs 1,584 from n(wT + T/2 + 1) |
| Check 2: reproduces literature | no | ours 0.754 [0.750, 0.757] vs 0.935 (Pakhunov (2026), Table II, [[72,12,6]], p=1e-3, T=12) |
| Check 3: accuracy cost of flags | flags improve accuracy | LER 6.00e-04 with flags vs 1.42e-03 without; +648 faults, +216 detectors; flags fire on 67.5% of shots |
| silent failures (weak decoder) | 0 flagged / 0 unflagged | the only failures a flag trigger could ever catch |

## Figure: validation

![validation](validation.png)  
Vector version: [validation.pdf](validation.pdf)
