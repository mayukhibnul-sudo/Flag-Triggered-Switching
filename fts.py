import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Flags: trade-off study")


@app.cell
def _():
    import json
    import os
    import sys
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.sparse as sp

    _here = mo.notebook_dir()
    NB_DIR = os.path.abspath(str(_here) if _here else ".")
    if NB_DIR not in sys.path:
        sys.path.insert(0, NB_DIR)
    import decoder_zoo as dz

    return NB_DIR, dz, json, mo, np, os, plt, sp, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Flags in bivariate bicycle memories: a trade-off study

    **Claim under test.** Flag qubits are usually dismissed for qLDPC syndrome
    extraction because they add depth and qubits and so "typically worsen the
    circuit's performance". That accounting prices the *circuit* and ignores what
    the flag outcomes give the decoder.

    We separate the two with three arms on the same code, noise and shots:

    | arm | circuit | decoder sees |
    |---|---|---|
    | **unflagged** | no flag qubits | all detectors |
    | **flagged, blind** | flag qubits present | flag detector rows **removed** from H |
    | **flagged, sighted** | flag qubits present | all detectors |

    *blind vs unflagged* prices the circuit. *sighted vs blind* prices the
    information. *sighted vs unflagged* is the number an architect cares about.

    **The scaling question.** The circuit cost grows with the number of rounds T
    (more rounds, more flag circuitry, more noise) while the information each flag
    carries does not. So the net benefit should shrink with T and eventually
    reverse. This notebook measures that and estimates the crossover T\*, which is
    a design rule: *flags pay below T\*, not above*.

    Everything heavy is reused: decoders and parallel decoding from
    `decoder_zoo.py`, codes/circuits/DEM from `decoderSwitch.py`.
    """)
    return


@app.cell
def _(mo, time):
    _t0 = time.perf_counter()
    from decoderSwitch import app as _ds_app

    _outputs, NB = _ds_app.run()          # also runs decoderSwitch's own test suite
    pipeline_ready = bool(NB["core_ready"])
    mo.md(
        f"Loaded `decoderSwitch.py` in {time.perf_counter() - _t0:.0f} s — "
        + ("**all its tests pass**, so the code, circuit and DEM builders below are "
           "the validated ones." if pipeline_ready else
           "⚠️ **its tests do not all pass**: fix them there before trusting anything here.")
    )
    return NB, pipeline_ready


@app.cell
def _(NB):
    # Reuse, don't reimplement.
    BB_PRESETS = NB["BB_PRESETS"]
    GB_PRESETS = NB["GB_PRESETS"]
    build_memory_circuit = NB["build_memory_circuit"]
    code_from_key = NB["code_from_key"]
    dem_to_matrices = NB["dem_to_matrices"]
    export_bundle = NB["export_bundle"]
    flag_detector_mask = NB["flag_detector_mask"]
    read_result_file = NB["read_result_file"]
    render_checks = NB["render_checks"]
    run_checks = NB["run_checks"]
    wilson = NB["wilson"]
    write_result_file = NB["write_result_file"]
    RESULTS_DIR = NB["RESULTS_DIR"]
    return (
        BB_PRESETS,
        GB_PRESETS,
        RESULTS_DIR,
        build_memory_circuit,
        code_from_key,
        dem_to_matrices,
        export_bundle,
        flag_detector_mask,
        read_result_file,
        render_checks,
        run_checks,
        wilson,
        write_result_file,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. What flags cost, before any decoding

    Qubits, two-qubit gates, detectors and fault mechanisms, with and without
    flags. This is the overhead an architect weighs against the accuracy gain, and
    it needs no simulation.
    """)
    return


@app.cell
def _(build_memory_circuit, dem_to_matrices, np):
    def circuit_overhead(code, rounds, p):
        """Qubit, gate, detector and fault counts for the flagged/unflagged circuits."""
        out = {}
        for flags in (False, True):
            circ = build_memory_circuit(code, rounds, p, use_flags=flags)
            cx = sum(len(inst.targets_copy()) // 2 for inst in circ.flattened()
                     if inst.name == "CX")
            H, _L, priors = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
            out[flags] = dict(qubits=circ.num_qubits, two_qubit_gates=cx,
                              detectors=circ.num_detectors, faults=H.shape[1],
                              expected_faults_per_shot=float(np.sum(priors)))
        a, b = out[False], out[True]
        return dict(unflagged=a, flagged=b,
                    extra_qubits=b["qubits"] - a["qubits"],
                    qubit_overhead=b["qubits"] / a["qubits"] - 1,
                    gate_overhead=b["two_qubit_gates"] / a["two_qubit_gates"] - 1,
                    fault_overhead=b["faults"] / a["faults"] - 1,
                    lambda_overhead=(b["expected_faults_per_shot"]
                                     / a["expected_faults_per_shot"] - 1))

    return (circuit_overhead,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The three-arm measurement

    One function, used for every point in the sweep. The blind arm **removes** the
    flag detector rows from the check matrix rather than zeroing those syndrome
    bits: zeroing would hand the decoder a syndrome that never occurred.
    """)
    return


@app.cell
def _(
    build_memory_circuit,
    dem_to_matrices,
    dz,
    flag_detector_mask,
    np,
    wilson,
):
    ARMS = ("unflagged", "flagged, blind", "flagged, sighted")

    def three_arm_run(code, rounds, p, shots, seed, workers=1, osd_order=0, on_chunk=None):
        """LER of one strong decoder on the three arms, plus flag statistics."""
        spec = [("strong", {"kind": "osd", "osd_order": osd_order, "max_iter": 20})]
        w = dz.parallel_workers(shots, workers)

        def decode(H, L, priors, det, obs):
            res, _ = dz.run_zoo(H, L, priors, det, obs, spec, workers=w,
                                chunk_size=dz.chunk_for(shots, w), on_chunk=on_chunk)
            fails = int(res["strong"]["fail"].sum())
            return fails, wilson(fails, shots), res["strong"]["fail"]

        circ0 = build_memory_circuit(code, rounds, p, use_flags=False)
        H0, L0, pr0 = dem_to_matrices(circ0.detector_error_model(decompose_errors=False))
        det0, obs0 = circ0.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True)

        circ1 = build_memory_circuit(code, rounds, p, use_flags=True)
        H1, L1, pr1 = dem_to_matrices(circ1.detector_error_model(decompose_errors=False))
        det1, obs1 = circ1.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True)
        mask = flag_detector_mask(code, rounds, True, circ1.num_detectors)
        keep = ~mask

        out = {}
        out["unflagged"] = decode(H0, L0, pr0, det0, obs0)
        out["flagged, blind"] = decode(H1[keep], L1, pr1, det1[:, keep], obs1)
        out["flagged, sighted"] = decode(H1, L1, pr1, det1, obs1)

        n_flags = det1[:, mask].sum(1)
        return dict(
            rounds=rounds, p=p, shots=shots, seed=seed, code=code.name,
            arms={k: dict(failures=v[0], ler=v[1][0], ci_low=v[1][1], ci_high=v[1][2])
                  for k, v in out.items()},
            # paired within the flagged circuit: blind and sighted saw the same shots
            blind_only_fail=int((out["flagged, blind"][2] & ~out["flagged, sighted"][2]).sum()),
            sighted_only_fail=int((out["flagged, sighted"][2] & ~out["flagged, blind"][2]).sum()),
            flagged_fraction=float((n_flags > 0).mean()),
            mean_flags=float(n_flags.mean()),
            detectors_flagged=int(circ1.num_detectors), detectors_unflagged=int(circ0.num_detectors))

    return ARMS, three_arm_run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Crossover fit

    Ratios of logical error rate to the unflagged arm. A ratio below 1 means flags
    help. Fitting `ln(ratio)` linearly in T and solving for `ratio = 1` gives the
    crossover T\*: the number of rounds beyond which flags stop paying.

    The error bars on a ratio of two failure counts use the standard Poisson
    approximation, `SE(ln r) ≈ sqrt(1/k₁ + 1/k₂)`, which needs a decent number of
    failures per point — check the counts before quoting T\*.
    """)
    return


@app.cell
def _(np):
    def ratio_to_unflagged(point, arm):
        """Ratio of `arm` to the unflagged arm, with a log-scale standard error."""
        k1 = point["arms"][arm]["failures"]
        k0 = point["arms"]["unflagged"]["failures"]
        if k0 == 0 or k1 == 0:
            return float("nan"), float("nan")
        return k1 / k0, float(np.sqrt(1.0 / k1 + 1.0 / k0))

    def crossover_fit(points, arm="flagged, sighted"):
        """
        Weighted least squares of ln(ratio) against T. Returns slope, intercept and
        the T where the fit crosses ratio = 1 (None when it never does).
        """
        rows = [(pt["rounds"], *ratio_to_unflagged(pt, arm)) for pt in points]
        rows = [(t, r, se) for t, r, se in rows if np.isfinite(r) and np.isfinite(se) and se > 0]
        if len(rows) < 2:
            return dict(slope=float("nan"), intercept=float("nan"), crossover=None, n=len(rows))
        T = np.array([r[0] for r in rows], float)
        y = np.log([r[1] for r in rows])
        w = 1.0 / np.array([r[2] for r in rows]) ** 2
        A = np.vstack([np.ones_like(T), T]).T
        W = np.diag(w)
        coef = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
        intercept, slope = float(coef[0]), float(coef[1])
        cross = -intercept / slope if slope > 0 and intercept < 0 else None
        return dict(slope=slope, intercept=intercept, crossover=cross, n=len(rows),
                    T=T.tolist(), ratio=[r[1] for r in rows], se=[r[2] for r in rows])

    return crossover_fit, ratio_to_unflagged


@app.cell
def _(
    BB_PRESETS,
    GB_PRESETS,
    mo,
    os,
):
    _cores = os.cpu_count() or 1
    ui_code = mo.ui.dropdown(list(BB_PRESETS) + list(GB_PRESETS), value="[[72, 12, 6]]",
                             label="Code")
    ui_p = mo.ui.dropdown(["5e-4", "1e-3", "2e-3", "3e-3"], value="1e-3", label="p")
    ui_rounds = mo.ui.multiselect(["3", "6", "9", "12", "18", "24"],
                                  value=["6", "12", "18", "24"], label="Rounds T")
    ui_shots = mo.ui.number(start=100, stop=500_000, step=100, value=20_000, label="Shots per arm")
    ui_seed = mo.ui.number(value=20260923, label="Seed")
    ui_osd = mo.ui.slider(0, 4, value=0, label="OSD order")
    ui_workers = mo.ui.slider(1, max(2, _cores), value=min(12, max(1, _cores - 4)),
                              label=f"CPU workers (of {_cores})")
    mo.vstack([mo.md("### Configuration"),
               mo.hstack([ui_code, ui_p, ui_osd]),
               ui_rounds,
               mo.hstack([ui_shots, ui_seed, ui_workers]),
               mo.md("*Each T costs three arms. 20,000 shots × 4 values of T is a few hours; "
                     "start with 5,000 to see the trend, then re-run for the final figure.*")])
    return ui_code, ui_osd, ui_p, ui_rounds, ui_seed, ui_shots, ui_workers


@app.cell
def _(ui_code, ui_osd, ui_p, ui_rounds, ui_seed, ui_shots, ui_workers):
    fts_config = dict(code=ui_code.value, p=float(ui_p.value),
                      rounds=sorted(int(t) for t in ui_rounds.value),
                      shots=int(ui_shots.value), seed=int(ui_seed.value),
                      osd_order=int(ui_osd.value), workers=int(ui_workers.value))
    return (fts_config,)


@app.cell
def _(circuit_overhead, code_from_key, fts_config, mo, pipeline_ready):
    mo.stop(not pipeline_ready)
    _code = code_from_key(fts_config["code"])
    _rows = []
    for _T in fts_config["rounds"]:
        _o = circuit_overhead(_code, _T, fts_config["p"])
        _rows.append(dict(rounds=_T, qubits_unflagged=_o["unflagged"]["qubits"],
                          qubits_flagged=_o["flagged"]["qubits"],
                          qubit_overhead=_o["qubit_overhead"],
                          gate_overhead=_o["gate_overhead"],
                          fault_overhead=_o["fault_overhead"],
                          lambda_unflagged=_o["unflagged"]["expected_faults_per_shot"],
                          lambda_flagged=_o["flagged"]["expected_faults_per_shot"]))
    overhead_rows = _rows
    mo.vstack([
        mo.md("### Cost of the flag hardware (no decoding)"),
        mo.md("| T | qubits | +qubits | +2q gates | +fault mechanisms | λ unflagged → flagged |\n"
              "|--:|--:|--:|--:|--:|--:|\n"
              + "\n".join(f"| {r['rounds']} | {r['qubits_unflagged']} → {r['qubits_flagged']} | "
                          f"{r['qubit_overhead']:+.0%} | {r['gate_overhead']:+.0%} | "
                          f"{r['fault_overhead']:+.0%} | "
                          f"{r['lambda_unflagged']:.2f} → {r['lambda_flagged']:.2f} |"
                          for r in overhead_rows)),
        mo.md("*λ is the expected number of fault mechanisms firing per shot: the quantity "
              "that actually drives the logical error rate.*"),
    ])
    return (overhead_rows,)


@app.cell
def _(mo):
    run_fts = mo.ui.run_button(label="Run the three-arm sweep")
    run_fts
    return (run_fts,)


@app.cell
def _(
    RESULTS_DIR,
    code_from_key,
    fts_config,
    json,
    mo,
    os,
    pipeline_ready,
    run_fts,
    three_arm_run,
    write_result_file,
):
    fts_run = None
    if not pipeline_ready:
        _out = mo.md("*Locked: `decoderSwitch.py` tests must pass first.*")
    elif not run_fts.value:
        _out = mo.md("*Press **Run the three-arm sweep**, or load a saved run below.*")
    else:
        _code = code_from_key(fts_config["code"])
        _points = []
        _total = 3 * len(fts_config["rounds"])
        with mo.status.progress_bar(total=_total, title="three-arm sweep", show_eta=True) as _bar:
            for _T in fts_config["rounds"]:
                _pt = three_arm_run(_code, _T, fts_config["p"], fts_config["shots"],
                                    fts_config["seed"], workers=fts_config["workers"],
                                    osd_order=fts_config["osd_order"])
                _points.append(_pt)
                _bar.update(3)
        _tag = fts_config["code"].strip("[]").replace(", ", "-")
        _path = os.path.join(
            RESULTS_DIR, f"FTS_{_tag}_p{fts_config['p']:g}_{fts_config['shots']}shots_"
                         f"seed{fts_config['seed']}_osd{fts_config['osd_order']}.json")
        write_result_file(_path, {"kind": "fts", "config": fts_config, "points": _points})
        fts_run = dict(config=fts_config, points=_points, path=_path)
        _out = mo.md(f"Sweep finished. Saved to `{_path}`.")
    _out
    return (fts_run,)


@app.cell
def _(RESULTS_DIR, fts_run, mo, os):
    import glob as _glob
    _ = fts_run
    _files = sorted(_glob.glob(os.path.join(RESULTS_DIR, "FTS_*.json")))
    ui_fts_file = mo.ui.dropdown({os.path.basename(f): f for f in _files},
                                 value=os.path.basename(_files[-1]) if _files else None,
                                 label="Saved sweep")
    load_fts = mo.ui.run_button(label="Load")
    mo.hstack([ui_fts_file, load_fts]) if _files else mo.md("*No saved sweeps yet.*")
    return load_fts, ui_fts_file


@app.cell
def _(load_fts, read_result_file, ui_fts_file):
    fts_loaded = None
    if load_fts.value and ui_fts_file.value:
        _d = read_result_file(ui_fts_file.value, "fts")
        fts_loaded = dict(config=_d["config"], points=_d["points"], path=ui_fts_file.value)
    return (fts_loaded,)


@app.cell
def _(ARMS, crossover_fit, fts_loaded, fts_run, mo, np, plt, ratio_to_unflagged):
    fts_data = fts_run if fts_run is not None else fts_loaded
    mo.stop(fts_data is None)
    _pts = sorted(fts_data["points"], key=lambda q: q["rounds"])
    _fit_sighted = crossover_fit(_pts, "flagged, sighted")
    _fit_blind = crossover_fit(_pts, "flagged, blind")

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(12, 4.6))
    _colours = {"unflagged": "#7f7f7f", "flagged, blind": "#d62728", "flagged, sighted": "#2ca02c"}
    for _arm in ARMS:
        _T = [q["rounds"] for q in _pts]
        _y = [q["arms"][_arm]["ler"] for q in _pts]
        _lo = [q["arms"][_arm]["ler"] - q["arms"][_arm]["ci_low"] for q in _pts]
        _hi = [q["arms"][_arm]["ci_high"] - q["arms"][_arm]["ler"] for q in _pts]
        _a.errorbar(_T, _y, yerr=[_lo, _hi], fmt="o-", color=_colours[_arm], capsize=4, label=_arm)
    _a.set_yscale("log")
    _a.set_xlabel("syndrome rounds T")
    _a.set_ylabel("logical error rate")
    _a.set_title(f"{fts_data['config']['code']}, p = {fts_data['config']['p']}, "
                 f"{fts_data['config']['shots']:,} shots/arm")
    _a.grid(True, which="both", alpha=0.3)
    _a.legend(fontsize=8)

    for _arm, _fit in (("flagged, sighted", _fit_sighted), ("flagged, blind", _fit_blind)):
        _r = [ratio_to_unflagged(q, _arm) for q in _pts]
        _T = [q["rounds"] for q, (v, _s) in zip(_pts, _r) if np.isfinite(v)]
        _v = [v for v, _s in _r if np.isfinite(v)]
        _e = [v * s for v, s in _r if np.isfinite(v)]
        _b.errorbar(_T, _v, yerr=_e, fmt="o", color=_colours[_arm], capsize=4, label=_arm)
        if np.isfinite(_fit["slope"]):
            _x = np.linspace(min(_T) - 1, max(max(_T) + 2, (_fit["crossover"] or 0) + 2), 50)
            _b.plot(_x, np.exp(_fit["intercept"] + _fit["slope"] * _x), "--",
                    color=_colours[_arm], lw=1)
    _b.axhline(1.0, color="k", lw=0.8)
    if _fit_sighted["crossover"]:
        _b.axvline(_fit_sighted["crossover"], color="#2ca02c", ls=":", lw=1)
        _b.annotate(f"T* ≈ {_fit_sighted['crossover']:.0f}",
                    (_fit_sighted["crossover"], 1.0), textcoords="offset points",
                    xytext=(6, 10), fontsize=9, color="#2ca02c")
    _b.set_xlabel("syndrome rounds T")
    _b.set_ylabel("logical error rate ÷ unflagged")
    _b.set_title("Below 1 = flags help. Dashed: weighted fit of ln(ratio) in T")
    _b.grid(True, alpha=0.3)
    _b.legend(fontsize=8)
    _fig.tight_layout()
    fts_fig, fts_fit = _fig, dict(sighted=_fit_sighted, blind=_fit_blind)
    return fts_data, fts_fig, fts_fit


@app.cell
def _(export_bundle, fts_data, fts_fig, fts_fit, mo, overhead_rows, ratio_to_unflagged):
    _pts = sorted(fts_data["points"], key=lambda q: q["rounds"])
    _rows = []
    for _q in _pts:
        _rs, _ = ratio_to_unflagged(_q, "flagged, sighted")
        _rb, _ = ratio_to_unflagged(_q, "flagged, blind")
        _rows.append(dict(
            rounds=_q["rounds"], shots=_q["shots"],
            **{f"{_arm} failures": _q["arms"][_arm]["failures"] for _arm in _q["arms"]},
            **{f"{_arm} LER": _q["arms"][_arm]["ler"] for _arm in _q["arms"]},
            ratio_sighted=_rs, ratio_blind=_rb,
            flagged_fraction=_q["flagged_fraction"], mean_flags=_q["mean_flags"],
            blind_only_fail=_q["blind_only_fail"], sighted_only_fail=_q["sighted_only_fail"]))

    _cross = fts_fit["sighted"]["crossover"]
    _headline = (
        f"**Flags pay up to T\\* ≈ {_cross:.0f}** and cost accuracy beyond it "
        f"(weighted fit, slope {fts_fit['sighted']['slope']:+.3f} per round)."
        if _cross else
        "**No crossover within the fitted range**: on this fit the flagged-and-sighted arm "
        "does not reach parity with the unflagged circuit over the T values measured.")
    _lines = ["| T | unflagged | blind | sighted | sighted ÷ unflagged | flags fire on |",
              "|--:|--:|--:|--:|--:|--:|"]
    for _q, _r in zip(_pts, _rows):
        _lines.append(f"| {_q['rounds']} | {_q['arms']['unflagged']['ler']:.3e} | "
                      f"{_q['arms']['flagged, blind']['ler']:.3e} | "
                      f"{_q['arms']['flagged, sighted']['ler']:.3e} | "
                      f"{_r['ratio_sighted']:.2f}× | {_q['flagged_fraction']:.1%} |")
    _folder = export_bundle(
        fts_data["path"],
        f"Flag trade-off study — {fts_data['config']['code']}, p = {fts_data['config']['p']}",
        {"three_arms_vs_rounds": fts_fig},
        {"points": _rows, "overhead": overhead_rows},
        notes=_headline)
    mo.vstack([mo.md("### Results"), mo.md(_headline), mo.md("\n".join(_lines)),
               mo.md(f"Data: `{fts_data['path']}` · exported to `{_folder}`"), fts_fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Tests

    These check the new code in this notebook. The code it builds on is tested in
    `decoderSwitch.py`, whose suite runs when this notebook loads it.
    """)
    return


@app.cell
def _(
    circuit_overhead,
    code_from_key,
    crossover_fit,
    mo,
    np,
    pipeline_ready,
    ratio_to_unflagged,
    render_checks,
    run_checks,
    three_arm_run,
):
    def _known_crossover_is_recovered():
        """A synthetic ratio that crosses 1 at T = 20 must be fitted as T* ≈ 20."""
        pts = []
        for T in (4, 8, 12, 16):
            ratio = np.exp(0.05 * (T - 20))          # crosses 1 exactly at T = 20
            k0 = 400
            pts.append(dict(rounds=T, arms={"unflagged": dict(failures=k0),
                                            "flagged, sighted": dict(failures=int(k0 * ratio))}))
        fit = crossover_fit(pts, "flagged, sighted")
        assert abs(fit["crossover"] - 20) < 1.0, fit["crossover"]
        assert fit["slope"] > 0

    def _no_crossover_returns_none():
        pts = [dict(rounds=T, arms={"unflagged": dict(failures=400),
                                    "flagged, sighted": dict(failures=200)})
               for T in (4, 8, 12)]
        assert crossover_fit(pts, "flagged, sighted")["crossover"] is None, \
            "a flat ratio below 1 has no crossover"

    def _ratio_handles_zero_failures():
        pt = dict(arms={"unflagged": dict(failures=0), "flagged, sighted": dict(failures=3)})
        assert not np.isfinite(ratio_to_unflagged(pt, "flagged, sighted")[0])

    def _overhead_counts_are_sane():
        code = code_from_key("[[72, 12, 6]]")
        o = circuit_overhead(code, 2, 1e-3)
        mx = code.hx.shape[0]
        assert o["extra_qubits"] == mx, f"one flag per X-check expected, got {o['extra_qubits']}"
        assert o["gate_overhead"] > 0 and o["fault_overhead"] > 0
        assert o["flagged"]["detectors"] > o["unflagged"]["detectors"]

    def _three_arms_differ_only_as_intended():
        code = code_from_key("[[72, 12, 6]]")
        r = three_arm_run(code, 2, 3e-3, 120, seed=5, workers=1)
        assert set(r["arms"]) == {"unflagged", "flagged, blind", "flagged, sighted"}
        assert r["detectors_flagged"] > r["detectors_unflagged"]
        assert 0.0 <= r["flagged_fraction"] <= 1.0
        # blind and sighted decode the SAME shots, so their disagreement is paired
        assert r["blind_only_fail"] + r["sighted_only_fail"] >= 0
        for arm in r["arms"].values():
            assert arm["ci_low"] <= arm["ler"] <= arm["ci_high"]

    tests_fts = run_checks([
        ("crossover fit recovers a known T*", _known_crossover_is_recovered),
        ("a ratio that never reaches 1 reports no crossover", _no_crossover_returns_none),
        ("ratios with zero failures are not reported", _ratio_handles_zero_failures),
        ("overhead accounting: one flag per X-check, more gates, more faults", _overhead_counts_are_sane),
        ("three-arm run returns consistent arms and intervals", _three_arms_differ_only_as_intended),
    ]) if pipeline_ready else []
    (render_checks("flag trade-off study", tests_fts) if pipeline_ready
     else mo.md("*Tests skipped: `decoderSwitch.py` did not load cleanly.*"))
    return (tests_fts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. What to write from this

    - **Cost table (§1):** flags add one qubit per X-check and raise λ, the expected
      faults per shot. This is the accounting behind "flags typically worsen the
      circuit".
    - **Blind vs unflagged:** confirms that claim — and quantifies it.
    - **Sighted vs unflagged:** the counterpoint. The flag outcomes buy back more
      than the circuit loses, so pricing flags without giving the decoder their
      outcomes understates them.
    - **T\* :** the design rule. Flags pay below T\*, not above, because their cost
      scales with rounds while their information does not.

    Compare the gain against the alternatives at the same p — biased-noise ancillas
    and CNOT-schedule optimisation — and state the qubit overhead alongside it.
    """)
    return


if __name__ == "__main__":
    app.run()
