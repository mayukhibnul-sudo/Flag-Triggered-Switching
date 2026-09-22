# Flag-Triggered Decoder Switching for Bivariate Bicycle Codes

Undergraduate thesis project, BRAC University.
**Authors:** TODO · **Supervisor:** TODO

This repository studies whether **flag-qubit outcomes** can serve as a cheap,
*pre-decode* signal for routing each shot of a quantum error-correction experiment
between a fast decoder and an accurate one, for bivariate bicycle (BB) and
generalised bicycle (GB) codes under circuit-level noise.

The central comparison is between two kinds of trigger:

- **`primary_fail`** — the trivial baseline: escalate to the accurate decoder when
  the fast decoder fails to converge. It can never catch a *silent failure*, a shot
  where the fast decoder converges confidently to the wrong answer.
- **flag triggers** — escalate when a flag qubit fires, *before* the fast decoder
  runs. These can catch silent failures in principle; whether they do in practice is
  the research question.

Everything — code, tests, experiments and figures — lives in one
[marimo](https://marimo.io) notebook, `decoderSwitch.py`, which doubles as the
presentation.

---

## Quick start

Tested with Python 3.12.

```bash
pip install "stim>=1.16" "ldpc>=2.4" "numpy>=2.0" "scipy>=1.13" "marimo>=0.24" "matplotlib>=3.9"
marimo edit decoderSwitch.py
```

In the editor, the notebook runs its test suite automatically. Open **§10
Implementation status**: every row should show ✅. The experiments in §11 stay
locked until all tests pass.

| Command | What it does |
|---|---|
| `marimo edit decoderSwitch.py` | Interactive editor; tests re-run when you change code |
| `marimo run decoderSwitch.py` | Read-only app view, code hidden |
| `marimo check decoderSwitch.py` | Lint the notebook |
| `marimo export html decoderSwitch.py -o report.html` | Static HTML snapshot, test results included |

---

## Notebook layout

Each section follows the same pattern — explanation → implementation → tests — so
it reads top to bottom as a presentation.

| § | Contents |
|---|---|
| 1 | Motivation and research question |
| 2 | GF(2) linear algebra: rank, row reduction, nullspace, quotient basis |
| 3 | CSS codes; BB codes (Bravyi et al.) and GB codes (Kovalev & Pryadko; Panteleev & Kalachev) |
| 4 | Stim circuit-level memory experiment with optional flag qubits |
| 5 | Detector error model → check matrix `H`, observable matrix `L`, priors |
| 6 | Decoders: greedy peeling and BP (fast); BP+OSD and BP+LSD (accurate) |
| 7 | Switch policies: `never`, `always`, `primary_fail`, `flag`, `flag_or_fail` |
| 8 | Statistics, benchmark harness, five graph types, and a quick graph preview |
| 9 | Saving and loading results, with provenance |
| 10 | Implementation status dashboard |
| 11 | Experiments: E1 ablation, E2 noise sweep |
| 12 | Results, discussion, conclusion (TODO) |

### Codes

| Family | Presets (n and k verified by tests) |
|---|---|
| BB | [[72, 12, 6]], [[90, 8, 10]], [[108, 8, 10]], [[144, 12, 12]], [[288, 12, 18]] |
| GB | [[46, 2, 9]], [[48, 6, 8]], [[126, 28, 8]], [[254, 28]] |

Distances *d* are the literature values and are **not** verified by the tests.

### What counts as a failure

A shot fails when the decoder's predicted observable flips differ from the true
flips sampled by Stim, on any of the *k* logical observables. Failures that differ
from the truth only by a stabiliser are correctly counted as successes.

Every failure falls into exactly one category:

| Category | Meaning |
|---|---|
| escalated | the accurate decoder ran and still got it wrong |
| **silent** | the fast decoder claimed convergence but was wrong |
| unconverged | the fast decoder gave up and nothing escalated |

Confidence intervals are 95% Wilson intervals, which remain valid at zero failures.
All policies in a comparison decode the **same** sampled shots.

---

## Tests

The notebook contains 67 tests across 10 sections. They run on every execution and
report ✅ pass, ❌ fail or ⬜ not yet implemented.

The tests were checked by deliberately introducing bugs one at a time — including
every real bug found during development — and confirming that each one turns a test
red. Examples: rank computed over the reals instead of GF(2), a wrong code
polynomial, logical errors scored by residual weight, a misplaced flag mask, CNOT
layers that reuse a qubit, and policies scored on different shots.

**Do not edit a test to make it pass.** If a test seems wrong, raise it first: the
tests encode the contracts the rest of the code relies on.

---

## Running experiments

### In the notebook

Set the configuration in §11 (code, rounds *T*, noise *p*, shots, flags, OSD order,
seed) and press **Run E1** or **Run E2**. At the defaults (500 shots, *T* = 6), E1
takes a couple of minutes and E2 roughly ten.

### Long runs, headless

Resolving logical error rates near 10⁻³ needs around 10⁵ shots per point — too
many for the notebook's buttons. Run such experiments as scripts; the notebook's
functions are importable:

```python
import dataclasses
import numpy as np
from decoderSwitch import app

_, nb = app.run()                      # executes the notebook, including all tests
assert nb["core_ready"], "some tests fail: fix them before running experiments"

cfg = dataclasses.replace(nb["ExperimentConfig"](), code="[[144, 12, 12]]",
                          rounds=12, shots=100_000)
ps = np.logspace(np.log10(5e-4), np.log10(6e-3), 6)
sweep = nb["run_sweep"](cfg, ps, progress=False)
nb["save_sweep"](nb["result_path"]("E2", cfg), cfg, sweep)
```

### Results files

Results are written to `results/` as JSON, named after their configuration, e.g.

```
results/E1_72-12-6_T6_p0.001_500shots_seed20260921_flags_osd0.json
```

The same configuration always maps to the same file. Each file records the full
configuration, the creation time, and the versions of stim, ldpc, numpy, scipy and
marimo used, so every figure can be regenerated and traced. Load them with
`load_results` / `load_sweep` and plot with the §8 graph functions.

Decide whether to commit `results/` to git: committing final results makes figures
reproducible from the repository alone; exploratory runs are better left out.

---

## Presenting

The default view scrolls top to bottom. For slides, open the notebook in
`marimo edit`, switch the layout to **Slides** in the app view, and save; marimo
stores the layout alongside the notebook, and `marimo run decoderSwitch.py` then
presents it as slides. Commit the layout file so the slides travel with the code.

---

## Known limitations

These are open issues, not settled results. Resolve them before drawing conclusions.

1. **Wall-clock latency is not a fair comparison yet.** The peeling decoder is pure
   Python; BP+OSD and BP+LSD come from `ldpc` and are compiled C++. Timings compare
   implementations, not algorithms. Planned fix: compile the peeler with `numba`, and
   report an implementation-independent operation count as the primary latency metric.
2. **Noise model not yet validated against the literature.** The circuit uses full
   two-qubit depolarising noise, giving roughly ten times more DEM fault mechanisms
   than Pakhunov (2026)'s model. Reproducing one of his published numbers (E0) is
   required before new results are credible.
3. **CNOT schedule.** All X-check layers run before all Z-check layers: always valid,
   but deeper than Bravyi et al.'s depth-8 schedule, so it injects more noise per round.
4. **`flag` and `flag_or_fail` behave identically.** Making `flag` a flag-only policy
   would isolate what the flag alone catches.
5. **No soft-information trigger yet.** The literature suggests this is the strongest
   competitor to a flag trigger.

---

## Roadmap

- [ ] Compile the peeling decoder; add an operation-count latency metric
- [ ] Settle the §7 design decisions, including a soft-information trigger
- [ ] **E0** — reproduce Pakhunov's peeling success rate under his noise model
- [ ] **E1** — flagged vs unflagged circuits: do flags pay for their extra noise?
- [ ] **E2** — silent-failure rates at scale
- [ ] **E3** — trigger comparison on identical shots
- [ ] **E4** — sweeps over noise, code size and rounds
- [ ] Write §1 and §12; set up the slides layout

---

## Key references

- S. Bravyi et al., "High-threshold and low-overhead fault-tolerant quantum memory,"
  *Nature* **627**, 778 (2024).
- R. Toshio et al., "Decoder Switching: Breaking the Speed-Accuracy Tradeoff in
  Real-Time Quantum Error Correction," arXiv:2510.25222 (2025).
- A. Pakhunov, "Analytical Theory of Greedy Peeling for Bivariate Bicycle Codes and
  Two-Shot Streaming Decoding," arXiv:2604.11352 (2026).
- C. Chamberland et al., "Topological and subsystem codes on low-degree graphs with
  flag qubits," *Phys. Rev. X* **10**, 011022 (2020).
- R. Chao and B. Reichardt, "Quantum error correction with only two extra qubits,"
  *Phys. Rev. Lett.* **121**, 050502 (2018).
- J. Roffe et al., "Decoding across the quantum low-density parity-check code
  landscape," *Phys. Rev. Research* **2**, 043423 (2020).
- T. Hillmann et al., "Localized statistics decoding: A parallel decoding algorithm
  for quantum LDPC codes," *Nature Communications* **16**, 8214 (2025).
- P. Panteleev and G. Kalachev, "Degenerate quantum LDPC codes with good finite
  length performance," *Quantum* **5**, 585 (2021).
- C. Gidney, "Stim: a fast stabilizer circuit simulator," *Quantum* **5**, 497 (2021).

The full reference list is in the notebook.

## License

TODO — choose a license before making the repository public.
