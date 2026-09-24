# E0 validation — [[72, 12, 6]], T=12, p=0.001

Data file: `E0_72-12-6_T12_p0.001_22600shots.json`  
Exported: 2026-09-23T18:45:57

Literature source: Pakhunov (2026), Table II, [[72,12,6]], p=1e-3, T=12

## Table: checks

[checks.csv](checks.csv) — 4 rows

| check | value | detail |
|---|---|---|
| V1 fault-model density | 10.8× | 33,552 faults vs 3,096 from n(wT + T/2 + 1) |
| V2 reproduces literature | no | ours 0.363 [0.357, 0.370] vs 0.935 (Pakhunov (2026), Table II, [[72,12,6]], p=1e-3, T=12) |
| V3 cost of flags | flags improve accuracy | LER 1.64e-02 with flags vs 2.21e-02 without; +2,592 faults, +432 detectors; flags fire on 89.1% of shots |
| silent failures (weak decoder) | 0 flagged / 0 unflagged | the only failures a flag trigger could ever catch |

## Figure: validation

![validation](validation.png)  
Vector version: [validation.pdf](validation.pdf)
