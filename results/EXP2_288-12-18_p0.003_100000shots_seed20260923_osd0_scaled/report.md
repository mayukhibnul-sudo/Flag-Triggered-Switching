# Flag trade-off study — [[288, 12, 18]], p = 0.003

Data file: `EXP2_288-12-18_p0.003_100000shots_seed20260923_osd0_scaled.json`  
Exported: 2026-09-30T18:21:34

**No crossover within the fitted range**: on this fit the flagged-and-sighted arm does not reach parity with the unflagged circuit over the T values measured.

## Table: points

[points.csv](points.csv) — 1 rows

| rounds | shots | unflagged failures | flagged, blind failures | flagged, sighted failures | unflagged LER | flagged, blind LER | flagged, sighted LER | ratio_sighted | ratio_blind | flagged_fraction | mean_flags | blind_only_fail | sighted_only_fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 100000 | 171 | 530 | 75 | 0.00171 | 0.0053 | 0.00075 | 0.4386 | 3.099 | 1 | 26.58 | 488 | 33 |

## Table: overhead

[overhead.csv](overhead.csv) — 1 rows

| rounds | qubits_unflagged | qubits_flagged | qubit_overhead | gate_overhead | fault_overhead | lambda_unflagged | lambda_flagged |
|---|---|---|---|---|---|---|---|
| 12 | 576 | 720 | 0.25 | 0.1667 | 0.2951 | 58.86 | 77.45 |

## Figure: three_arms_vs_rounds

![three_arms_vs_rounds](three_arms_vs_rounds.png)  
Vector version: [three_arms_vs_rounds.pdf](three_arms_vs_rounds.pdf)
