import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Flagged Decoding — older experiments L1–L4",
)

with app.setup:
    # ---------------------------------------------------------------------------
    # The older experiments reuse FlaggedDecoding.py's codes, circuits, decoders,
    # statistics and result-file handling. Both files must sit in the same folder
    # (they also share its results/ folder).
    # ---------------------------------------------------------------------------
    import math
    import os
    import sys
    from dataclasses import asdict, replace

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as st

    try:
        _here = os.path.dirname(os.path.abspath(__file__))
    except NameError:                        # pragma: no cover - exotic runners
        _here = str(mo.notebook_dir() or os.getcwd())
    if _here not in sys.path:
        sys.path.insert(0, _here)

    from FlaggedDecoding import (
        BB_PRESETS, BpOsdDecoder, ExperimentConfig, GB_PRESETS, Metrics, PeelingDecoder,
        STRONG_MAX_ITER, SwitchPolicy, TABLE_II, TRIGGERS, TRIGGER_DISPLAY, all_pass,
        build_memory_circuit, build_reference_circuit, chunk_for, circuit_identity,
        code_from_key, config_from_file, config_payload, decoder_spec_full, dem_to_matrices,
        derive, derive_seed, flag_detector_mask, load_results, load_sweep, mde_ratio,
        metrics_rows, paired_exact, parallel_workers, plot_ablation, plot_outcomes, plot_sweep,
        provenance_banner, rate_ratio_ci, read_result_file, reference_report, render_checks,
        result_file, run_benchmark, run_checks, run_grid, run_zoo, sample_hash, save_results,
        save_sweep, saved_run_picker, saved_runs, start_run, strong_spec, theory_faults,
        wilson, write_result_file,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # FlaggedDecoding — older exploratory experiments L1–L4

    This companion notebook holds four exploratory experiments that came before
    the experiment programme of `FlaggedDecoding.py`. **No thesis table or figure
    is sourced from them**; each has a better-designed successor in the main
    notebook. They are kept so that their older result files (`EXP1_*`, `EXP2_*`,
    `EXP4_*`, `EXP5_*`) still open with correct provenance, and so the runs can be
    reproduced.

    **How it works.** Everything except the four experiments — codes, circuits,
    decoders, statistics, parallel decoding and result files — is imported from
    `FlaggedDecoding.py`, which must sit in the same folder (both use its
    `results/` folder). Section references such as "§13" point to that notebook.

    | here | what it measured | successor in FlaggedDecoding.py | why superseded |
    |---|---|---|---|
    | L1 | validation: fault count, peeling against Table II, BP+OSD with vs without flags | Experiment 1 (Table 5.1), Experiment 2 (flags vs no flags), law L7 | the peeling check used the experiment circuit rather than the reference circuit; the flags-vs-no-flags arms were compared as if paired; X-check detectors were kept |
    | L2 | three-arm depth sweep over $T$ | Experiment 2 (Table 5.7) on the four-arm runner | unpaired arms; 20 BP iterations; no file for the thesis table |
    | L3 | trigger ablation on one code | Experiment 4's decoder grid (every rule on the same 250k shots) | two trigger names meant the same rule (`flag` ≡ `flag_or_fail`); X-check detectors |
    | L4 | trigger ablation across $p$ | Experiment 5 (census, escalation vs $p$) | X-check detectors; no thesis item |

    **Changes from the original code.** One strong-decoder specification (30
    iterations); `flag>=1` and `flag-only` replace the duplicated trigger; unpaired
    comparisons carry exact rate-ratio intervals and a minimum detectable effect;
    paired comparisons against `primary_fail` and `always` are reported shot by
    shot; the Table II target is keyed by (code, $T$, $p$); every new run writes a
    schema-2 file with its seed, detector convention, decoder settings and timing
    regime.

    **Statistics.** Wilson intervals for every rate; exact McNemar for paired
    comparisons on the same shots (L3/L4 triggers against `primary_fail` and
    `always`); the exact conditional rate-ratio test with its minimum detectable
    ratio for unpaired arms (L1 Check 3, L2); a weighted crossover fit with its
    standard error for L2. All of it is exploratory: no multiplicity correction
    is claimed.

    The experiments run once this notebook's own tests (bottom) pass.
    """)
    return


@app.cell
def _():
    legacy_ui = mo.ui.dictionary(dict(
        code=mo.ui.dropdown(list(BB_PRESETS) + list(GB_PRESETS), value="[[72, 12, 6]]", label="code"),
        rounds=mo.ui.number(start=1, stop=48, value=12, label="rounds T"),
        p=mo.ui.dropdown(["0.0005", "0.001", "0.002", "0.003", "0.005"], value="0.001", label="p"),
        shots=mo.ui.number(start=100, stop=1_000_000, step=100, value=10_000, label="shots"),
        seed=mo.ui.number(value=2026111901, label="seed"),
        use_flags=mo.ui.switch(value=True, label="flag qubits"),
        osd_order=mo.ui.slider(0, 7, value=0, label="OSD order", show_value=True),
        workers=mo.ui.slider(1, max(2, os.cpu_count() or 1), value=1, label="workers", show_value=True),
        x_detectors=mo.ui.switch(value=False, label="X-check detectors (older convention)"),
    ))
    mo.vstack([mo.md("### Configuration (L1, L3, L4)"), legacy_ui])
    return (legacy_ui,)


@app.cell
def _(legacy_ui):
    _v = legacy_ui.value
    legacy_config = ExperimentConfig(code=_v["code"], rounds=int(_v["rounds"]), p=float(_v["p"]),
                                     shots=int(_v["shots"]), seed=int(_v["seed"]),
                                     use_flags=bool(_v["use_flags"]), osd_order=int(_v["osd_order"]),
                                     workers=int(_v["workers"]), x_detectors=bool(_v["x_detectors"]))
    return (legacy_config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## L1 — validation

    Check 1: the fault count against Theorem 1 ("not applicable" for codes whose
    column weight is not uniform). Check 2: peeling on the **reference circuit**
    against Table II's value for this (code, $T$, $p$) — or a custom target,
    recorded as such. Check 3: BP+OSD with and without flag qubits; the two arms
    are **different circuits with independent samples**, so the comparison is an
    unpaired exact rate-ratio interval with its minimum detectable ratio.
    """)
    return


@app.function
def exp1_check2_title(v2):
    """The Check 2 heading, from what the numbers actually show."""
    if v2.get("target") is None:
        return "Check 2 — no published value for this (code, T, p)"
    if "predicted" not in v2:
        return "Check 2 — saved before the reference-circuit fix: not comparable to the paper"
    if not v2["model_agrees"]:
        return "Check 2 — the model does not reproduce the published value"
    if v2["agrees"]:
        return "Check 2 — model and decoder both reproduce the published value"
    return "Check 2 — the model reproduces the published value; our peeler falls short (cause not isolated)"


@app.function
def l1_check3(k_flag, k_unf, n):
    """Unpaired exact comparison of the flagged and unflagged circuits."""
    rr = rate_ratio_ci(k_flag, n, k_unf, n)
    if rr["hi"] < 1:
        effect = "flags improve accuracy"
    elif rr["lo"] > 1:
        effect = "flags cost accuracy"
    else:
        effect = "no resolvable difference"
    return dict(rr, effect=effect, mde_ratio=mde_ratio(k_unf), paired=False)


@app.function
def run_validation(config, target=None, target_source=None, on_chunk=None):
    """
    The three checks of L1. `target` None means "use Table II for this
    (code, T, p)"; a number is a custom target (target_source='manual').
    """
    if config.x_detectors is None:
        raise ValueError("unknown detector convention")
    code = code_from_key(config.code)
    T, p, n = config.rounds, config.p, config.shots
    tab = TABLE_II.get((config.code, int(T), float(p)))
    if target is None:
        target, target_source = tab, ("Pakhunov (2026), Table II" if tab is not None else None)
    else:
        target_source = target_source or "manual"
    spec = strong_spec(config.osd_order)
    seeds = {True: int(config.seed), False: derive_seed(config.seed, "unflagged")}
    arms, samples = {}, {}
    for flags in (True, False):
        circ = build_memory_circuit(code, T, p, use_flags=flags, x_detectors=config.x_detectors)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        det, obs = circ.compile_detector_sampler(seed=seeds[flags]).sample(n, separate_observables=True)
        w = parallel_workers(n, config.workers)
        res, backend, info = run_zoo(H, L, pr, det, obs, [("weak", {"kind": "peel"}), ("strong", spec)],
                                     workers=w, chunk_size=chunk_for(n, w), on_chunk=on_chunk,
                                     checkpoint_dir=False, return_info=True)
        mask = flag_detector_mask(code, T, flags, circ.num_detectors, x_detectors=config.x_detectors)
        samples["flagged" if flags else "unflagged"] = sample_hash(det, obs)
        arms[flags] = dict(faults=int(H.shape[1]), detectors=int(circ.num_detectors),
                           flagged_fraction=float((det[:, mask].sum(1) > 0).mean()) if flags else 0.0,
                           weak_resolved=int(res["weak"]["conv"].sum()),
                           strong_failures=int(res["strong"]["fail"].sum()),
                           silent=int((res["weak"]["conv"] & res["weak"]["fail"]).sum()),
                           workers_effective=info["workers_effective"], backend=backend,
                           seed=seeds[flags])
    rep = reference_report(code, T, p)
    rc = build_reference_circuit(code, T, p)
    rH, rL, rpr = dem_to_matrices(rc.detector_error_model(decompose_errors=False))
    rdet, robs = rc.compile_detector_sampler(seed=seeds[True]).sample(n, separate_observables=True)
    rres, _b = run_zoo(rH, rL, rpr, rdet, robs, ["peel"], workers=1, checkpoint_dir=False)
    k = int(rres["peel"]["conv"].sum())
    rate, lo, hi = wilson(k, n)
    th = theory_faults(code, T)
    out = dict(code=config.code, rounds=T, p=p, shots=n, seed=config.seed, target=target,
               target_source=target_source, osd_order=config.osd_order,
               x_detectors=bool(config.x_detectors),
               detector_convention="x+z" if config.x_detectors else "z-only",
               decoders=dict(weak=decoder_spec_full("peel"), strong=decoder_spec_full(spec)),
               samples=samples, arms={("flagged" if f else "unflagged"): a for f, a in arms.items()})
    out["v1"] = dict(reference_faults=th["total"], our_faults=arms[True]["faults"],
                     our_faults_unflagged=arms[False]["faults"], uniform=th["uniform"],
                     ratio=(arms[False]["faults"] / th["total"]) if th["uniform"] else None,
                     reference_circuit_faults=int(rH.shape[1]),
                     law_l7_added=arms[True]["faults"] - arms[False]["faults"])
    out["v2"] = dict(measured=rate, resolved=k, ci_low=lo, ci_high=hi, target=target,
                     agrees=(lo <= target <= hi) if target is not None else None,
                     predicted=rep["theory_peel"], predicted_uncorrected=rep["theory_peel_uncorrected"],
                     model_agrees=(abs(rep["theory_peel"] - target) < 0.02) if target is not None else None,
                     decoder_gap=rep["theory_peel"] - rate, lam=rep["lam"],
                     mean_degree=rep["mean_degree"],
                     reference_sample_sha256=sample_hash(rdet, robs))
    out["v2"]["title"] = exp1_check2_title(out["v2"])
    out["v3"] = dict(l1_check3(arms[True]["strong_failures"], arms[False]["strong_failures"], n),
                     extra_faults=arms[True]["faults"] - arms[False]["faults"],
                     extra_detectors=arms[True]["detectors"] - arms[False]["detectors"],
                     flagged_fraction=arms[True]["flagged_fraction"],
                     silent_flagged=arms[True]["silent"], silent_unflagged=arms[False]["silent"])
    return out


@app.function
def l1_load(path):
    """A new L1 file, or an older EXP1 'validation' file (with v1)."""
    d = read_result_file(path, ("L1-validation", "validation"))
    if d.get("kind") == "validation":
        cfg, prov = config_from_file(d, path)
        v2 = dict(d["checks"]["v2"])
        v2.setdefault("target", d["config"].get("target"))
        v2["title"] = exp1_check2_title(v2)
        v3 = d["checks"]["v3"]
        out = dict(code=d["config"]["code"], rounds=d["config"]["rounds"], p=d["config"]["p"],
                   shots=d["config"]["shots"], target=v2.get("target"),
                   target_source=d.get("source"), v1=d["checks"]["v1"], v2=v2,
                   v3=dict(v3, effect=v3.get("effect"), legacy=True), provenance=prov,
                   detector_convention=cfg["detector_convention"])
        return dict(path=path, doc=dict(d, status=prov.get("status")), result=out, fresh=False)
    return dict(path=path, doc=d, result=d["result"], fresh=False)


@app.cell
def _():
    l1_target_custom = mo.ui.switch(value=False, label="custom target (else Table II for this code, T, p)")
    l1_target = mo.ui.number(0.0, 1.0, value=0.935, step=0.001, label="custom target")
    l1_run_btn = mo.ui.run_button(label="Run L1")
    mo.hstack([l1_target_custom, l1_target, l1_run_btn])
    return l1_run_btn, l1_target, l1_target_custom


@app.cell
def _(l1_run_btn, l1_target, l1_target_custom, legacy_config, tests_legacy):
    l1_run = None
    if l1_run_btn.value and all_pass(tests_legacy):
        _r = run_validation(legacy_config, target=l1_target.value if l1_target_custom.value else None)
        _stamp = start_run("L1-validation", "L1")
        _path = result_file("L1", code=legacy_config.code, rounds=legacy_config.rounds, p=legacy_config.p,
                            shots=legacy_config.shots, seed=legacy_config.seed, osd=legacy_config.osd_order,
                            convention=_r["detector_convention"])
        write_result_file(_path, dict(
            config=dict(config_payload(legacy_config, decoders=_r["decoders"]), target=_r["target"],
                        target_source=_r["target_source"]),
            timing=dict(workers_requested=legacy_config.workers,
                        workers_effective=sorted({a["workers_effective"] for a in _r["arms"].values()}),
                        backends=sorted({a["backend"].split(";")[0] for a in _r["arms"].values()}),
                        resumed_chunks=0, total_chunks=0),
            samples=_r["samples"], result=_r), _stamp, "complete")
        l1_run = dict(path=_path, doc=read_result_file(_path), result=_r, fresh=True)
    l1_file, _files = saved_run_picker("L1", "Saved L1 run", legacy_prefixes=("EXP1",),
                                       legacy_kinds=("validation",))
    l1_load_btn = mo.ui.run_button(label="Load")
    mo.hstack([l1_file, l1_load_btn])
    return l1_file, l1_load_btn, l1_run


@app.cell
def _(l1_file, l1_load_btn, l1_run):
    l1_data = l1_run if l1_run is not None else (l1_load(l1_file.value)
                                                 if (l1_load_btn.value and l1_file.value) else None)
    mo.stop(l1_data is None, mo.md("*Run L1 or load a saved run.*"))
    _r = l1_data["result"]
    _v1, _v2, _v3 = _r["v1"], _r["v2"], _r["v3"]
    _c1 = ("not applicable (column weight not uniform)" if _v1.get("ratio") is None
           else f"{_v1['ratio']:.3f}× ({_v1['our_faults_unflagged']:,} vs {_v1['reference_faults']:,})")
    _c2 = (f"measured {_v2['measured']:.4f} [{_v2['ci_low']:.4f}, {_v2['ci_high']:.4f}]"
           + (f", model {_v2['predicted']:.4f}" if "predicted" in _v2 else "")
           + (f", target {_v2['target']} ({_r.get('target_source')})" if _v2.get("target") is not None else ""))
    if "ratio" in _v3 and not _v3.get("legacy"):
        _c3 = (f"{_v3['effect']}: ratio {_v3['ratio']:.3g} [{_v3['lo']:.3g}, {_v3['hi']:.3g}] "
               f"(unpaired, exact conditional); minimum detectable ratio {_v3['mde_ratio']:.2f}")
    else:
        _c3 = (f"legacy verdict \"{_v3.get('effect')}\" from overlapping Wilson intervals of unpaired "
               "arms — not a test; rerun for an exact interval")
    mo.vstack([provenance_banner(dict(provenance=_r.get("provenance", {}),
                                      status=l1_data["doc"].get("status"))),
               mo.md(f"#### L1 — {_r['code']}, T = {_r['rounds']}, p = {_r['p']}, {_r['shots']:,} shots\n\n"
                     f"| check | result |\n|:--|:--|\n| Check 1 — faults / Theorem 1 | {_c1} |\n"
                     f"| {_v2['title']} | {_c2} |\n| Check 3 — flags vs no flags | {_c3} |")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## L2 — three-arm depth sweep

    unflagged, blind and sighted across $T$ at one $p$, now through the four-arm
    runner (unflagged unpaired; blind and sighted paired). The ratio to the
    unflagged arm is drawn on a log axis with exact asymmetric intervals; a
    zero-failure ratio is drawn at its upper end. A crossover $T^*$ is printed
    only if the weighted fit's slope is positive with $p < 0.05$ and $T^*$ lies
    inside the measured range.
    """)
    return


@app.cell
def _():
    L2_ARM_MAP = {"unflagged": "unflagged", "flagged, blind": "blind", "flagged, sighted": "sighted"}

    def exp2_shots_for(T, config):
        """Shots at T: the budget at the smallest T, scaled as 1/T (floor 2,000) when asked."""
        base = int(config["shots"])
        if not config.get("scale_shots", False) or not config["rounds"]:
            return base
        t_min = min(config["rounds"])
        return int(min(base, max(2000, round(base * t_min / int(T)))))

    return L2_ARM_MAP, exp2_shots_for


@app.function
def circuit_overhead(code, rounds, p, x_detectors=False):
    """Qubits (touched), two-qubit gates, detectors and faults, flagged vs unflagged."""
    out = {}
    for flags in (False, True):
        circ = build_memory_circuit(code, rounds, p, use_flags=flags, x_detectors=x_detectors)
        cx = sum(len(inst.targets_copy()) // 2 for inst in circ.flattened() if inst.name == "CX")
        H, _L, priors = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        ident = circuit_identity(circ, H, priors)
        out[flags] = dict(qubits=ident["qubits_used"], two_qubit_gates=cx,
                          detectors=circ.num_detectors, faults=H.shape[1],
                          expected_faults_per_shot=float(np.sum(priors)))
    a, b = out[False], out[True]
    return dict(unflagged=a, flagged=b, extra_qubits=b["qubits"] - a["qubits"],
                qubit_overhead=b["qubits"] / a["qubits"] - 1,
                gate_overhead=b["two_qubit_gates"] / a["two_qubit_gates"] - 1,
                fault_overhead=b["faults"] / a["faults"] - 1,
                lambda_overhead=b["expected_faults_per_shot"] / a["expected_faults_per_shot"] - 1)


@app.cell
def _(L2_ARM_MAP):
    def l2_points(doc):
        """Points (one per T) with arms unflagged/blind/sighted, from a new L2 grid
        file or an older EXP2 file."""
        pts = []
        if doc.get("kind") == "exp2":
            for q in doc["points"]:
                arms = {L2_ARM_MAP[k]: dict(v) for k, v in q["arms"].items()}
                pts.append(dict(rounds=q["rounds"], shots=q["shots"], arms=arms,
                                paired={"blind|sighted": dict(a_only=q.get("blind_only_fail"),
                                                              b_only=q.get("sighted_only_fail"))},
                                flagged_fraction=q.get("flagged_fraction")))
        else:
            for r in doc["results"]:
                pts.append(dict(rounds=r["rounds"], shots=r["shots"], arms=r["arms"], paired=r["paired"],
                                flagged_fraction=r.get("overhead", {}).get("flagged_fraction")))
        for q in pts:
            for a in ("blind", "sighted"):
                if a in q["arms"] and "unflagged" in q["arms"]:
                    q.setdefault("ratio", {})[a] = rate_ratio_ci(q["arms"][a]["failures"], q["shots"],
                                                                 q["arms"]["unflagged"]["failures"],
                                                                 q["shots"])
        return sorted(pts, key=lambda q: q["rounds"])

    return (l2_points,)


@app.function
def crossover_fit(points, arm="sighted"):
    """
    Weighted least squares of ln(k_arm / k_unflagged) on T (variance
    1/k1 + 1/k0). Returns slope, its SE and p (t, n - 2 df), T* = -a/b with a
    delta-method SE, whether T* lies in the measured range, and n.
    `crossover` is T* only if slope > 0, p < 0.05 and T* is in range.
    """
    rows = []
    for q in points:
        k1 = q["arms"].get(arm, {}).get("failures", 0)
        k0 = q["arms"].get("unflagged", {}).get("failures", 0)
        if k1 > 0 and k0 > 0:
            rows.append((q["rounds"], math.log(k1 / k0), 1 / k1 + 1 / k0))
    if len(rows) < 2:
        return dict(n=len(rows), slope=None, crossover=None, note="Fit not performed (n < 2)")
    T = np.array([r[0] for r in rows], float)
    y = np.array([r[1] for r in rows])
    w = 1 / np.array([r[2] for r in rows])
    X = np.column_stack([np.ones_like(T), T])
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    a, b = cov @ (XtW @ y)
    df = len(rows) - 2
    if df > 0:
        resid = y - (a + b * T)
        s2 = max(float((w * resid ** 2).sum() / df), 1.0)
        cov = cov * s2
        p = float(2 * st.t.sf(abs(b) / math.sqrt(cov[1, 1]), df))
    else:
        p = float(2 * st.norm.sf(abs(b) / math.sqrt(cov[1, 1])))
    tstar = -a / b if b != 0 else None
    se_t = None
    if tstar is not None:
        g = np.array([-1 / b, a / b ** 2])
        se_t = float(math.sqrt(g @ cov @ g))
    in_range = tstar is not None and T.min() <= tstar <= T.max()
    ok = b > 0 and p < 0.05 and in_range
    return dict(n=len(rows), slope=float(b), slope_se=float(math.sqrt(cov[1, 1])), slope_p=p,
                intercept=float(a), t_star=tstar, t_star_se=se_t, in_range=bool(in_range),
                crossover=float(tstar) if ok else None,
                note="" if ok else "no significant crossover inside the measured range")


@app.function
def plot_l2(points, code, p):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    col = {"unflagged": "#555555", "blind": "#d62728", "sighted": "#2ca02c"}
    for arm in ("unflagged", "blind", "sighted"):
        for q in points:
            m = q["arms"].get(arm)
            if m is None:
                continue
            if m["failures"] > 0:
                a.errorbar(q["rounds"], m["ler"], yerr=[[m["ler"] - m["ci_low"]], [m["ci_high"] - m["ler"]]],
                           fmt="o", color=col[arm], capsize=3)
            else:
                a.plot(q["rounds"], m["ci_high"], "v", ms=8, mfc="none", mec=col[arm])
        a.plot([q["rounds"] for q in points],
               [q["arms"][arm]["ler"] if q["arms"].get(arm, {}).get("failures") else
                q["arms"].get(arm, {}).get("ci_high", np.nan) for q in points], "-", color=col[arm],
               alpha=0.6, label=arm)
    for arm in ("blind", "sighted"):
        for q in points:
            r = q.get("ratio", {}).get(arm)
            if r is None:
                continue
            if r["k1"] > 0 and np.isfinite(r["ratio"]):
                b.errorbar(q["rounds"], r["ratio"], yerr=[[r["ratio"] - r["lo"]], [r["hi"] - r["ratio"]]],
                           fmt="o", color=col[arm], capsize=3)
            elif np.isfinite(r["hi"]):
                b.plot(q["rounds"], r["hi"], "v", ms=8, mfc="none", mec=col[arm])
        b.plot([], [], "o", color=col[arm], label=arm)
    b.axhline(1, color="k", lw=0.8)
    for ax in (a, b):
        ax.set_yscale("log")
        ax.set_xlabel("rounds T")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    shots = ", ".join(f"T={q['rounds']}: {q['shots']:,}" for q in points)
    a.set_ylabel("logical error rate per shot")
    a.set_title(f"{code}, p = {p:g} — shots per arm {shots}", fontsize=9)
    b.set_ylabel("failures ÷ unflagged (unpaired, exact 95%)")
    b.set_title("below 1 = flags help (▽ = zero failures: upper end)", fontsize=9)
    fig.tight_layout()
    return fig


@app.cell
def _():
    l2_ui = mo.ui.dictionary(dict(
        code=mo.ui.dropdown(list(BB_PRESETS) + list(GB_PRESETS), value="[[72, 12, 6]]", label="code"),
        p=mo.ui.dropdown(["0.0005", "0.001", "0.002", "0.003"], value="0.001", label="p"),
        rounds=mo.ui.multiselect(["3", "6", "9", "12", "18", "24"], value=["6", "12", "18", "24"],
                                 label="rounds T"),
        shots=mo.ui.number(start=100, stop=500_000, step=100, value=20_000, label="shots at the smallest T"),
        scale=mo.ui.switch(value=True, label="scale shots as 1/T"),
        seed=mo.ui.number(value=2026111902, label="master seed"),
        workers=mo.ui.slider(1, max(2, os.cpu_count() or 1), value=1, label="workers", show_value=True),
    ))
    l2_run_btn = mo.ui.run_button(label="Run L2")
    mo.vstack([l2_ui, l2_run_btn])
    return l2_run_btn, l2_ui


@app.cell
def _(exp2_shots_for, l2_run_btn, l2_ui, tests_legacy):
    l2_run = None
    if l2_run_btn.value and all_pass(tests_legacy):
        _v = l2_ui.value
        _cfg = dict(code=_v["code"], p=float(_v["p"]), rounds=sorted(int(t) for t in _v["rounds"]),
                    shots=int(_v["shots"]), scale_shots=bool(_v["scale"]))
        _configs = [dict(code=_cfg["code"], rounds=T, p=_cfg["p"], shots=exp2_shots_for(T, _cfg),
                         arms=["unflagged", "blind", "sighted"]) for T in _cfg["rounds"]]
        _path, _doc, _res = run_grid("L2", "L2-depth-sweep", _configs, int(_v["seed"]),
                                     workers=int(_v["workers"]), scaled=_cfg["scale_shots"])
        l2_run = dict(path=_path, doc=_doc, fresh=True)
    l2_file, _files = saved_run_picker("L2", "Saved L2 run", legacy_prefixes=("EXP2",))
    l2_load_btn = mo.ui.run_button(label="Load")
    mo.hstack([l2_file, l2_load_btn])
    return l2_file, l2_load_btn, l2_run


@app.cell
def _(l2_file, l2_load_btn, l2_points, l2_run):
    l2_data = l2_run if l2_run is not None else (
        dict(path=l2_file.value, doc=read_result_file(l2_file.value, ("L2-depth-sweep", "exp2")), fresh=False)
        if (l2_load_btn.value and l2_file.value) else None)
    mo.stop(l2_data is None, mo.md("*Run L2 or load a saved run.*"))
    _doc = l2_data["doc"]
    _cfg, _prov = config_from_file(_doc, l2_data["path"])
    _pts = l2_points(_doc)
    _code = _cfg.get("code") or (_cfg.get("codes") or ["?"])[0]
    _p = _cfg.get("p") or (_cfg.get("ps") or [float("nan")])[0]
    _fit = crossover_fit(_pts, "sighted")
    _head = (f"**Crossover T\\* = {_fit['crossover']:.1f} ± {_fit['t_star_se']:.1f}** "
             f"(slope {_fit['slope']:+.3f} ± {_fit['slope_se']:.3f}, p = {_fit['slope_p']:.2g})"
             if _fit.get("crossover") else f"**No crossover reported:** {_fit['note']}")
    _lines = ["| T | shots | unflagged | blind | sighted | sighted ÷ unflagged [95%] | blind ÷ unflagged |",
              "|--:|--:|--:|--:|--:|:--|:--|"]
    for _q in _pts:
        _f = lambda a: (f"{_q['arms'][a]['ler']:.3e}" if _q["arms"][a]["failures"] else
                        f"≤ {_q['arms'][a]['ci_high']:.1e}")
        _rr = lambda a: (f"{_q['ratio'][a]['ratio']:.3g} [{_q['ratio'][a]['lo']:.3g}, {_q['ratio'][a]['hi']:.3g}]"
                         if _q['ratio'][a]['k1'] else f"0 [0, ≤ {_q['ratio'][a]['hi']:.3g}]")
        _lines.append(f"| {_q['rounds']} | {_q['shots']:,} | {_f('unflagged')} | {_f('blind')} | "
                      f"{_f('sighted')} | {_rr('sighted')} | {_rr('blind')} |")
    mo.vstack([provenance_banner(dict(provenance=_prov, status=_prov.get("status"))), mo.md(_head),
               mo.md("\n".join(_lines)), plot_l2(_pts, _code, _p)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## L3 / L4 — trigger ablation and noise sweep

    Every trigger on the same shots: `never`, `always`, `primary_fail`,
    `flag>=1` (flag, else primary failure) and `flag-only` (flag, never on primary
    failure). Each policy carries its paired comparison with `primary_fail` and
    with `always` (shots only this policy fails : shots only the reference fails,
    exact p); a policy is called better only if it wins with Bonferroni over the
    four non-reference triggers.
    """)
    return


@app.function
def run_ablation(config, keep_samples=True, on_chunk=None, prov=None):
    """
    L3: decode every shot once with peeling and once with the shared strong
    spec, then derive every trigger (FlaggedDecoding.py §13). Returns dict trigger -> Metrics;
    fills `prov` (decoders, timing, samples) when given.
    """
    if config.x_detectors is None:
        raise ValueError("unknown detector convention")
    code = code_from_key(config.code)
    circuit = build_memory_circuit(code, config.rounds, config.p, use_flags=config.use_flags,
                                   x_detectors=config.x_detectors)
    H, L, priors = dem_to_matrices(circuit.detector_error_model(decompose_errors=False))
    mask = flag_detector_mask(code, config.rounds, config.use_flags, circuit.num_detectors,
                              x_detectors=config.x_detectors)
    det, obs = circuit.compile_detector_sampler(seed=config.seed).sample(config.shots,
                                                                         separate_observables=True)
    det, obs = det.astype(np.uint8), obs.astype(np.uint8)
    spec = strong_spec(config.osd_order)
    workers = parallel_workers(config.shots, config.workers)
    results, backend, info = run_zoo(H, L, priors, det, obs, [("weak", {"kind": "peel"}), ("strong", spec)],
                                     workers=workers, chunk_size=chunk_for(config.shots, workers),
                                     on_chunk=on_chunk, checkpoint_dir=False, return_info=True,
                                     label=f"L3 ablation {config.code} T={config.rounds} p={config.p:g}")
    W, S = results["weak"], results["strong"]
    n_flags = det[:, mask].sum(1) if mask.any() else np.zeros(config.shots, dtype=int)
    none, ones = np.zeros(config.shots, bool), np.ones(config.shots, bool)
    triggers = {"never": (none, False), "always": (ones, False), "primary_fail": (none, True)}
    if config.use_flags:
        triggers["flag>=1"] = (n_flags > 0, True)
        triggers["flag-only"] = (n_flags > 0, False)
    pf = derive(W, S, none, True)
    al = derive(W, S, ones, False)
    out = {}
    for name, (pre, fb) in triggers.items():
        P = derive(W, S, pre, fb)
        k = int(P["fail"].sum())
        ler, lo, hi = wilson(k, config.shots)
        x, y, pv = paired_exact(P["fail"], pf["fail"])
        xa, ya, pa = paired_exact(P["fail"], al["fail"])
        out[name] = Metrics(
            shots=config.shots, failures=k, ler=ler, ci_low=lo, ci_high=hi,
            escalation_rate=float(P["esc"].mean()), work_mean=float(P["work"].mean()),
            work_p99=float(np.percentile(P["work"], 99)),
            time_p50_us=float(np.percentile(P["cost"], 50)),
            time_p99_us=float(np.percentile(P["cost"], 99)),
            silent_failures=int(P["silent"].sum()), unconverged_failures=int(P["unconverged"].sum()),
            samples=(dict(time_us=P["cost"], work=P["work"], failed=P["fail"], escalated=P["esc"],
                          converged=W["conv"]) if keep_samples else None),
            vs_primary_fail=dict(worse=x, better=y, p=pv), vs_always=dict(worse=xa, better=ya, p=pa))
    if prov is not None:
        prov.update(decoders=dict(weak=decoder_spec_full("peel"), strong=decoder_spec_full(spec)),
                    timing=dict(workers_requested=config.workers,
                                workers_effective=info["workers_effective"],
                                backends=[backend.split(";")[0]], resumed_chunks=info["resumed_chunks"],
                                total_chunks=info["total_chunks"]),
                    samples=dict(seed=config.seed, shots=config.shots, sha256=sample_hash(det, obs),
                                 circuit=circuit_identity(circuit, H, priors)))
    return out


@app.function
def run_sweep(config, ps, keep_samples=False, progress=True, prov=None):
    """L4: run_ablation at each p. Returns dict p -> dict trigger -> Metrics."""
    ps = [float(p) for p in ps]
    steps = mo.status.progress_bar(ps, title="L4 sweep", show_eta=True) if progress else ps
    out, per = {}, {}
    for p in steps:
        pr = {}
        out[p] = run_ablation(replace(config, p=p), keep_samples=keep_samples, prov=pr)
        per[p] = pr
    if prov is not None:
        first = next(iter(per.values()))
        prov.update(decoders=first["decoders"],
                    timing=dict(first["timing"], resumed_chunks=0,
                                total_chunks=sum(v["timing"]["total_chunks"] for v in per.values())),
                    samples={f"{p:g}": v["samples"] for p, v in per.items()}, per_p=per)
    return out


@app.function
def ablation_table_md(res, m_family=None):
    m_family = m_family or max(1, len(res) - 1)
    lines = ["| trigger | failures | LER [95% CI] | escalated | silent | vs primary_fail (only this : "
             "only pf, p) | vs always | better than pf (Bonf.) |",
             "|:--|--:|:--|--:|--:|:--|:--|:--|"]
    for t, m in res.items():
        a, b = m.vs_primary_fail, m.vs_always
        sa = f"{a['worse']} : {a['better']} ({a['p']:.2g})" if a else "—"
        sb = f"{b['worse']} : {b['better']} ({b['p']:.2g})" if b else "—"
        better = (a and a["better"] > a["worse"] and a["p"] * m_family < 0.05)
        lines.append(f"| {TRIGGER_DISPLAY.get(t, t)} | {m.failures} | {m.ler:.2e} [{m.ci_low:.2e}, "
                     f"{m.ci_high:.2e}] | {m.escalation_rate:.1%} | {m.silent_failures} | {sa} | {sb} | "
                     f"{'yes' if better else 'no'} |")
    return "\n".join(lines)


@app.cell
def _():
    l34_mode = mo.ui.radio({"L3 ablation (one p)": "L3", "L4 noise sweep": "L4"}, value="L3 ablation (one p)")
    l34_run_btn = mo.ui.run_button(label="Run L3 / L4")
    mo.hstack([l34_mode, l34_run_btn])
    return l34_mode, l34_run_btn


@app.cell
def _(l34_mode, l34_run_btn, legacy_config, tests_legacy):
    l34_run = None
    if l34_run_btn.value and all_pass(tests_legacy):
        _prov = {}
        if l34_mode.value == "L3":
            _res = run_ablation(legacy_config, prov=_prov)
            _stamp = start_run("ablation", "L3")
            _path = result_file("L3", code=legacy_config.code, rounds=legacy_config.rounds, p=legacy_config.p,
                                shots=legacy_config.shots, seed=legacy_config.seed,
                                flags=legacy_config.use_flags, osd=legacy_config.osd_order,
                                convention=legacy_config.detector_convention)
            save_results(_path, legacy_config, _res, _stamp, prov=_prov, include_samples=True)
            l34_run = dict(kind="L3", path=_path, results=_res, config=asdict(legacy_config), meta={})
        else:
            _ps = [float(x) for x in np.logspace(np.log10(5e-4), np.log10(6e-3), 6)]
            _sw = run_sweep(legacy_config, _ps, prov=_prov)
            _stamp = start_run("sweep", "L4")
            _path = result_file("L4", code=legacy_config.code, rounds=legacy_config.rounds, p=_ps,
                                shots=legacy_config.shots, seed=legacy_config.seed,
                                flags=legacy_config.use_flags, osd=legacy_config.osd_order,
                                convention=legacy_config.detector_convention)
            save_sweep(_path, legacy_config, _sw, _stamp, prov=_prov)
            l34_run = dict(kind="L4", path=_path, sweep=_sw, config=asdict(legacy_config), meta={})
    _f3 = saved_runs("L3") + saved_runs("EXP4", ext=".json")
    _f4 = saved_runs("L4") + saved_runs("EXP5", ext=".json")
    l34_file = mo.ui.dropdown({os.path.basename(f): f for f in _f3 + _f4}, label="Saved L3/L4 run")
    l34_load_btn = mo.ui.run_button(label="Load")
    mo.hstack([l34_file, l34_load_btn])
    return l34_file, l34_load_btn, l34_run


@app.cell
def _(l34_file, l34_load_btn, l34_run):
    l34_data = l34_run
    if l34_data is None and l34_load_btn.value and l34_file.value:
        _kind = read_result_file(l34_file.value).get("kind")
        if _kind in ("ablation", "L3-ablation"):
            _c, _r, _m = load_results(l34_file.value)
            l34_data = dict(kind="L3", path=l34_file.value, results=_r, config=_c, meta=_m)
        else:
            _c, _s, _m = load_sweep(l34_file.value)
            l34_data = dict(kind="L4", path=l34_file.value, sweep=_s, config=_c, meta=_m)
    mo.stop(l34_data is None, mo.md("*Run L3/L4 or load a saved run.*"))
    _c = l34_data["config"]
    _extra = dict(code=_c.get("code"), rounds=_c.get("rounds"), shots=_c.get("shots"), seed=_c.get("seed"),
                  osd_order=_c.get("osd_order"), use_flags=_c.get("use_flags"),
                  detector_convention=_c.get("detector_convention"))
    if l34_data["kind"] == "L3":
        _res = l34_data["results"]
        _body = [mo.md(f"#### L3 — {_c.get('code')}, T = {_c.get('rounds')}, p = {_c.get('p')}, "
                       f"{_c.get('shots'):,} shots, convention {_c.get('detector_convention')}"),
                 mo.md(ablation_table_md(_res)
                       if all(m.vs_primary_fail for m in _res.values()) else
                       "*Older file: no paired comparisons were stored; LERs only.*\n\n"
                       + "\n".join(f"- {TRIGGER_DISPLAY.get(t, t)}: {m.failures} failures, "
                                   f"{m.ler:.2e} [{m.ci_low:.2e}, {m.ci_high:.2e}]" for t, m in _res.items())),
                 plot_ablation(_res), plot_outcomes(_res)]
        _rows = metrics_rows(_res, p=_c.get("p"), **_extra)
    else:
        _sw = l34_data["sweep"]
        _body = [mo.md(f"#### L4 — {_c.get('code')}, T = {_c.get('rounds')}, {_c.get('shots'):,} shots/point"),
                 plot_sweep(_sw)]
        _rows = [r for _p, _r in _sw.items() for r in metrics_rows(_r, p=_p, **_extra)]
    mo.vstack([provenance_banner(l34_data.get("meta")),
               *(_body), mo.accordion({"metrics (CSV rows)": mo.md(
                   "| " + " | ".join(list(_rows[0])[:12]) + " |\n|" + "---|" * 12 + "\n" + "\n".join(
                       "| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v)
                                         for v in list(r.values())[:12]) + " |" for r in _rows))})])
    return


@app.cell
def _():
    _small = ExperimentConfig(code="[[72, 12, 6]]", rounds=2, p=3e-3, shots=60, seed=3)

    def _ablation_triggers():
        res = run_ablation(_small)
        assert tuple(res) == TRIGGERS, tuple(res)
        assert res["always"].escalation_rate == 1.0 and res["never"].escalation_rate == 0.0
        s = res["flag-only"].samples
        code = code_from_key(_small.code)
        circ = build_memory_circuit(code, 2, 3e-3, use_flags=True, x_detectors=False)
        det = circ.compile_detector_sampler(seed=3).sample(60)
        mask = flag_detector_mask(code, 2, True, circ.num_detectors, x_detectors=False)
        flagged = det[:, mask].any(1)
        assert np.array_equal(s["escalated"], flagged), "flag-only must escalate exactly the flagged shots"
        e1 = res["flag>=1"].samples["escalated"]
        assert np.array_equal(e1, flagged | ~res["flag>=1"].samples["converged"])

    def _ablation_matches_switchpolicy():
        code = code_from_key(_small.code)
        circ = build_memory_circuit(code, 2, 3e-3, use_flags=True, x_detectors=False)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(code, 2, True, circ.num_detectors, x_detectors=False)
        got = run_ablation(_small, keep_samples=False)
        peel, osd = PeelingDecoder(H, pr), BpOsdDecoder(H, pr, osd_order=0)
        ref = run_benchmark(circ, {t: SwitchPolicy(peel, osd, t, mask) for t in TRIGGERS}, L,
                            _small.shots, _small.seed)
        for t, m in ref.items():
            assert got[t].failures == m.failures, (t, got[t].failures, m.failures)
            assert abs(got[t].escalation_rate - m.escalation_rate) < 1e-12, t
            assert got[t].silent_failures == m.silent_failures, t

    def _ablation_provenance_and_comparisons():
        prov = {}
        res = run_ablation(_small, keep_samples=False, prov=prov)
        assert prov["decoders"]["strong"]["max_iter"] == STRONG_MAX_ITER
        assert prov["timing"]["workers_effective"] == 1 and len(prov["samples"]["sha256"]) == 64
        assert res["primary_fail"].vs_primary_fail == dict(worse=0, better=0, p=1.0)
        for m in res.values():
            a = m.vs_primary_fail
            assert a["worse"] - a["better"] == m.failures - res["primary_fail"].failures
        rows = metrics_rows(res, code=_small.code, rounds=2, seed=3, osd_order=0, p=3e-3,
                            detector_convention="z-only")
        assert {"code", "rounds", "seed", "osd_order", "detector_convention", "p"} <= set(rows[0])

    def _no_flags_no_flag_triggers():
        res = run_ablation(replace(_small, use_flags=False), keep_samples=False)
        assert set(res) == {"never", "always", "primary_fail"}

    def _sweep_runs():
        sw = run_sweep(_small, [1e-3, 3e-3], progress=False)
        assert list(sw) == [1e-3, 3e-3] and all(tuple(r) == TRIGGERS for r in sw.values())

    def _check2_titles():
        assert exp1_check2_title(dict(target=None)).startswith("Check 2 — no published")
        assert "both reproduce" in exp1_check2_title(dict(target=0.9, predicted=0.9, model_agrees=True,
                                                          agrees=True))
        assert "falls short" in exp1_check2_title(dict(target=0.9, predicted=0.9, model_agrees=True,
                                                       agrees=False))
        assert "does not reproduce" in exp1_check2_title(dict(target=0.9, predicted=0.7,
                                                              model_agrees=False, agrees=False))

    def _check3_unpaired():
        a = l1_check3(3, 8, 10_000)
        assert a["effect"] == "no resolvable difference" and a["lo"] < 1 < a["hi"] and not a["paired"]
        b = l1_check3(41, 244, 250_000)
        assert b["effect"] == "flags improve accuracy"
        assert TABLE_II.get(("[[72, 12, 6]]", 6, 1e-3)) is None
        assert TABLE_II[("[[72, 12, 6]]", 12, 1e-3)] == 0.935
        assert theory_faults(code_from_key("[[46, 2, 9]]"), 6)["uniform"] is False

    def _validation_runs():
        v = run_validation(replace(_small, shots=40))
        assert v["v1"]["law_l7_added"] == 3 * 36 * 2
        assert v["v3"]["effect"] in ("flags improve accuracy", "flags cost accuracy",
                                     "no resolvable difference")
        assert v["samples"]["flagged"] != v["samples"]["unflagged"]
        assert v["v3"]["paired"] is False and "lo" in v["v3"] and "mde_ratio" in v["v3"]
        assert v["target"] is None and v["v2"]["title"].startswith("Check 2 — no published")

    def _crossover_fit_repaired():
        pts = []
        for T in (12, 16, 20, 24):
            r = math.exp(0.08 * (T - 20))
            pts.append(dict(rounds=T, arms=dict(unflagged=dict(failures=4000),
                                                sighted=dict(failures=int(4000 * r)))))
        f = crossover_fit(pts)
        assert f["crossover"] is not None and abs(f["crossover"] - 20) < 1 and f["slope_p"] < 0.05
        flat = [dict(rounds=T, arms=dict(unflagged=dict(failures=400), sighted=dict(failures=200)))
                for T in (4, 8, 12)]
        assert crossover_fit(flat)["crossover"] is None
        out = [dict(rounds=T, arms=dict(unflagged=dict(failures=4000),
                                        sighted=dict(failures=int(4000 * math.exp(0.05 * (T - 40))))))
               for T in (4, 8, 12)]
        assert crossover_fit(out)["crossover"] is None and crossover_fit(out[:1])["note"].startswith("Fit not")

    def _l2_plot_and_zero_ratio():
        pts = [dict(rounds=6, shots=10_000, arms=dict(unflagged=dict(failures=1, ler=1e-4, ci_low=1.8e-5, ci_high=5.7e-4),
                                                      blind=dict(failures=3, ler=3e-4, ci_low=1e-4, ci_high=8.8e-4),
                                                      sighted=dict(failures=2, ler=2e-4, ci_low=5.5e-5, ci_high=7.3e-4)),
                    paired={}),
               dict(rounds=24, shots=5_000, arms=dict(unflagged=dict(failures=4, ler=8e-4, ci_low=3e-4, ci_high=2e-3),
                                                      blind=dict(failures=6, ler=1.2e-3, ci_low=5e-4, ci_high=2.6e-3),
                                                      sighted=dict(failures=0, ler=0.0, ci_low=0.0, ci_high=7.7e-4)),
                    paired={})]
        for q in pts:
            q["ratio"] = {a: rate_ratio_ci(q["arms"][a]["failures"], q["shots"],
                                           q["arms"]["unflagged"]["failures"], q["shots"])
                          for a in ("blind", "sighted")}
        z = pts[1]["ratio"]["sighted"]
        assert z["ratio"] == 0 and np.isfinite(z["hi"])
        fig = plot_l2(pts, "[[72, 12, 6]]", 1e-3)
        ax = fig.axes[1]
        assert ax.get_yscale() == "log"
        for line in ax.get_lines():
            y = np.asarray(line.get_ydata(), float)
            assert not len(y) or (y[np.isfinite(y)] > 0).all()
        assert "T=24: 5,000" in fig.axes[0].get_title() and "T=6: 10,000" in fig.axes[0].get_title()

    def _overhead_uses_touched_qubits():
        o = circuit_overhead(code_from_key("[[46, 2, 9]]"), 2, 1e-3)
        assert o["unflagged"]["qubits"] == 92, o["unflagged"]["qubits"]
        assert o["extra_qubits"] == code_from_key("[[46, 2, 9]]").hx.shape[0]

    tests_legacy = run_checks([
        ("L3: every trigger; flag-only = flagged shots; flag>=1 = flagged ∪ unconverged", _ablation_triggers),
        ("L3: derived policies equal SwitchPolicy on every trigger", _ablation_matches_switchpolicy),
        ("L3: provenance, paired comparisons and CSV config columns", _ablation_provenance_and_comparisons),
        ("L3: no flag triggers without flags", _no_flags_no_flag_triggers),
        ("L4: one ablation per p", _sweep_runs),
        ("L1: Check 2 titles follow the numbers", _check2_titles),
        ("L1: Check 3 is unpaired and exact; targets keyed by (code, T, p)", _check3_unpaired),
        ("L1: validation runs; independent samples per arm; law L7", _validation_runs),
        ("L2: crossover fit with SE, p and range", _crossover_fit_repaired),
        ("L2: log ratios, zero-failure upper ends, per-T shots in titles", _l2_plot_and_zero_ratio),
        ("L2: overhead counts touched qubits", _overhead_uses_touched_qubits),
    ])
    render_checks("L1–L4", tests_legacy)
    return (tests_legacy,)


if __name__ == "__main__":
    app.run()
