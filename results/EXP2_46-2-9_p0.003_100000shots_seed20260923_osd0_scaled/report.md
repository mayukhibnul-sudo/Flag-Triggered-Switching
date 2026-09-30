# Flag trade-off study — [[46, 2, 9]], p = 0.003

Data file: `EXP2_46-2-9_p0.003_100000shots_seed20260923_osd0_scaled.json`  
Exported: 2026-09-30T19:25:56

**No crossover within the fitted range**: on this fit the flagged-and-sighted arm does not reach parity with the unflagged circuit over the T values measured.

## Table: points

[points.csv](points.csv) — 1 rows

| rounds | shots | unflagged failures | flagged, blind failures | flagged, sighted failures | unflagged LER | flagged, blind LER | flagged, sighted LER | ratio_sighted | ratio_blind | flagged_fraction | mean_flags | blind_only_fail | sighted_only_fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 100000 | 56687 | 59728 | 53482 | 0.5669 | 0.5973 | 0.5348 | 0.9435 | 1.054 | 0.9974 | 5.953 | 23312 | 17066 |

## Table: overhead

[overhead.csv](overhead.csv) — 1 rows

| rounds | qubits_unflagged | qubits_flagged | qubit_overhead | gate_overhead | fault_overhead | lambda_unflagged | lambda_flagged |
|---|---|---|---|---|---|---|---|
| 12 | 92 | 115 | 0.25 | 0.1 | 0.1651 | 14.67 | 17.64 |

## Figure: three_arms_vs_rounds

![three_arms_vs_rounds](three_arms_vs_rounds.png)  
Vector version: [three_arms_vs_rounds.pdf](three_arms_vs_rounds.pdf)
