import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="Pakhunov Reference Model")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Pakhunov (2026) reference model

    A validation notebook for `decoderSwitch.py`. It answers one question with
    data: **does our simulation reproduce the literature, or does it deviate
    from it?**

    Experiment 1 check 2 compares our greedy-peeling success rate against
    Table II of Pakhunov, *Analytical Theory of Greedy Peeling for Bivariate
    Bicycle Codes and Two-Shot Streaming Decoding* (2026) — 93.5% for
    $[[72,12,6]]$ at $p = 10^{-3}$, $T = 12$. Our experiment circuit resolves
    about 59%. This notebook locates that gap.

    **It is not the noise channel and it is not the detectors.** It is the
    X-check *extraction gates*:

    | | Theorem | what it says | what we do |
    |:--|:--|:--|:--|
    | 1 | $N = n(wT + T/2 + 1)$ | "each round applies $nw$ CNOT gates" — 216 for $[[72,12,6]]$ | 432, because we extract **both** check families |
    | 2 | $\lambda = \alpha n T p$, $\alpha = 3.505$ | measured $\alpha = 3.56$ with `DEPOLARIZE2` — 1.6% high; a single X per CNOT gives 4.17, 19% high | matches — the channel was never the problem |

    Drop the X-check extraction gates and the fault count becomes *exactly*
    $n(wT + T/2 + 1)$, $\lambda$ lands within 2% of Theorem 2, and the
    Theorem 3/5 prediction evaluated on our own measured graph reproduces
    Table II. What remains is our peeling implementation being less
    aggressive than theirs — a far narrower and more defensible claim than
    "we do not reproduce the literature".

    /// admonition | This is a validation circuit, not a replacement.
    The flag qubits in Experiments 2, 3, 4 and 6 attach to **X-check**
    ancillas, so those experiments need both check families and must keep
    using `build_memory_circuit` from `decoderSwitch.py`. Nothing here
    changes the experiment circuit.
    ///

    Everything below — codes, linear algebra, the peeling decoder, Wilson
    intervals — is imported from `decoderSwitch.py`. This notebook defines no
    physics of its own except the gate-noise variants in §3.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import functools
    import os
    import sys
    import types

    import matplotlib.pyplot as plt
    import numpy as np
    import stim

    import decoder_zoo as dz
    return dz, functools, np, os, plt, stim, sys, types


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Reusing the notebook

    `decoderSwitch.py` is a marimo notebook: a module of `@app.cell`
    functions whose `defs` and `refs` form a DAG. Running the whole app
    executes every test and experiment cell, which takes minutes. Instead we
    resolve that DAG and execute **only the cells needed to define the names
    we ask for**, so this notebook stays in lockstep with it without
    duplicating a line.

    The cells run inside a real `types.ModuleType` registered in
    `sys.modules`, not a bare dict: the notebook uses `@dataclass`, and
    `dataclasses` resolves annotations through `sys.modules[cls.__module__]`.
    """)
    return


@app.cell
def _(functools, sys, types):
    NB_MODULE = "decoderSwitch"

    NB_SUBSET = types.ModuleType(f"{NB_MODULE}_subset")
    sys.modules[NB_SUBSET.__name__] = NB_SUBSET
    NB_NS = NB_SUBSET.__dict__

    @functools.lru_cache(maxsize=None)
    def nb_cells():
        """Every parsable cell of the notebook, as (source, defs, refs)."""
        import importlib

        try:
            nb = importlib.import_module(NB_MODULE)
        except ImportError as exc:                      # pragma: no cover
            raise ImportError(
                f"cannot import {NB_MODULE}.py -- put this notebook in the same "
                f"folder as {NB_MODULE}.py and start marimo from there"
            ) from exc
        out = []
        for cd in nb.app._cell_manager.cell_data():
            if cd.cell is None:                         # unparsable cell
                continue
            out.append(dict(code=cd.code,
                            defs=set(getattr(cd.cell, "defs", ())),
                            refs=set(getattr(cd.cell, "refs", ()))))
        return out

    def nb_plan(names):
        """Cell indices needed to define `names`, in dependency order."""
        cells = nb_cells()
        provider = {}
        for i, c in enumerate(cells):
            for d in c["defs"]:
                provider.setdefault(d, i)

        unknown = [n for n in names if n not in provider]
        if unknown:
            raise ImportError(f"{NB_MODULE}.py defines no {unknown}")

        need, stack = set(), list(names)
        while stack:                                    # transitive closure
            i = provider[stack.pop()]
            if i in need:
                continue
            need.add(i)
            stack += [r for r in cells[i]["refs"] if r in provider]

        order, done = [], set()
        while need:                                     # topological order
            ready = [i for i in sorted(need)
                     if {provider[r] for r in cells[i]["refs"] if r in provider}
                     <= done | {i}]
            if not ready:                               # marimo forbids cycles
                raise ImportError(f"cyclic cell dependency among {sorted(need)}")
            order += ready
            done |= set(ready)
            need -= set(ready)
        return order

    def notebook(*names):
        """
        The named objects from decoderSwitch.py, executing only the cells needed
        to define them and their transitive dependencies.

        >>> make_bb_code, dem_to_matrices = notebook("make_bb_code", "dem_to_matrices")
        """
        todo = [n for n in names if n not in NB_NS]
        if todo:
            cells = nb_cells()
            for i in nb_plan(todo):
                if not cells[i]["defs"] <= set(NB_NS):  # already run
                    exec(cells[i]["code"], NB_NS)       # the notebook is the source
            missing = [n for n in names if n not in NB_NS]
            if missing:
                raise ImportError(
                    f"could not resolve {missing} from {NB_MODULE}.py; its cell "
                    "structure may have changed"
                )
        return tuple(NB_NS[n] for n in names) if len(names) > 1 else NB_NS[names[0]]
    return NB_MODULE, notebook


@app.cell
def _(notebook):
    # Pulled from the notebook rather than redefined, so a change there reaches here.
    check_schedule, dem_to_matrices, run_checks, wilson = notebook(
        "check_schedule", "dem_to_matrices", "run_checks", "wilson")
    BB_PRESETS, GB_PRESETS, bb_from_preset, gb_from_preset = notebook(
        "BB_PRESETS", "GB_PRESETS", "bb_from_preset", "gb_from_preset")
    # Persistence: the same results/ folder, file format and export bundles the
    # experiments use, so a run from here is indistinguishable from one of theirs.
    RESULTS_DIR, export_bundle, read_result_file, saved_run_picker, write_result_file = (
        notebook("RESULTS_DIR", "export_bundle", "read_result_file",
                 "saved_run_picker", "write_result_file"))

    def code_by_label(key):
        """Build a BB or GB preset by its label, using the notebook's builders."""
        if key in BB_PRESETS:
            return bb_from_preset(key, BB_PRESETS)
        if key in GB_PRESETS:
            return gb_from_preset(key, GB_PRESETS)
        raise KeyError(f"unknown code {key!r}")

    CODE_LABELS = list(BB_PRESETS) + list(GB_PRESETS)
    return (CODE_LABELS, RESULTS_DIR, check_schedule, code_by_label,
            dem_to_matrices, export_bundle, read_result_file, run_checks,
            saved_run_picker, wilson, write_result_file)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Theorems 1, 2, 3 and 5

    Four predictions, each a closed form in the code parameters. The point of
    this notebook is to evaluate them on **our own measured graph** and see
    where the measurement lands.

    - **Theorem 1** — $N = n(wT + T/2 + 1)$, as $nwT$ CNOT faults, $(n/2)T$
      measurement faults and $n$ boundary faults. Here $w$ is the **column**
      weight of the check matrix (qubit degree), not the row weight.
    - **Theorem 2** — $\lambda = \alpha n T p$, the expected number of fault
      mechanisms firing per shot, with $\alpha = 3.505$ for $w = 3$.
    - **Theorems 3 and 5** — $P_{\text{peel}} = \exp(-\bar{d}\lambda^2 / 2N)^{A_0}$,
      where $A_0$ is the collision-resolution factor of Table III: the
      fraction of detector-sharing fault pairs that a peeler can actually
      separate. $A_0$ is constant across the Gross family and lower for small
      codes.

    Theorem 1 is stated for BB codes, whose column weight is uniform. For a
    code where it is not — a GB code with $|a| \neq |b|$ — the count is
    nominal only, and `theory_faults` reports `uniform=False` so it is never
    silently compared.
    """)
    return


@app.cell
def _(np):
    ALPHA_W3 = 3.505      # Theorem 2, column weight 3, standard superconducting model

    # Theorem 5 / Table III: |separable| / |all| detector-sharing fault pairs.
    A0 = {"[[18, 4, 4]]": 0.767,
          "[[32, 8, 6]]": 0.764,
          "[[72, 12, 6]]": 0.869,
          "[[144, 12, 12]]": 0.869,
          "[[288, 12, 18]]": 0.869}
    A0_DEFAULT = 0.869

    def theory_faults(code, rounds):
        """Theorem 1: N = n(wT + T/2 + 1), split into its three sources."""
        n = code.n
        cols = np.asarray(code.hz.sum(axis=0)).ravel()
        w = int(cols[0])
        return dict(total=int(n * (w * rounds + rounds / 2 + 1)),
                    cnot=int(n * w * rounds),
                    measurement=int(n // 2 * rounds),
                    boundary=int(n),
                    column_weight=w,
                    uniform=bool((cols == w).all()))

    def theory_lambda(code, rounds, p, alpha=ALPHA_W3):
        """Theorem 2: expected number of fault mechanisms firing per shot."""
        return alpha * code.n * rounds * p

    def theory_peel(faults, lam, mean_degree, a0=A0_DEFAULT):
        """
        Theorems 3 and 5: P_peel = exp(-beta lambda^2) ** A0, beta = d_bar / 2N.

        Evaluated on a circuit's OWN measured graph this is what a perfect
        queue-based peeler should achieve on it -- so the distance between it and
        our decoder's measured rate is attributable to the decoder, not the model.
        """
        return float(np.exp(-mean_degree / (2 * faults) * lam ** 2) ** a0)
    return A0, A0_DEFAULT, ALPHA_W3, theory_faults, theory_lambda, theory_peel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The reference circuit

    A Z-basis memory experiment, with two axes that can be varied one at a
    time. This is the only physics this notebook defines, and it exists so
    that the cause of the gap can be *isolated* rather than asserted.

    **Extraction** is the decisive axis. `z-only` applies $nw$ CNOTs per
    round, as Theorem 1 counts; `both` applies $2nw$, which is what
    §4 of `decoderSwitch.py` does because its flag qubits attach to X-check
    ancillas.

    **Gate noise** is the axis that turns out not to matter. `depolarize2` is
    the notebook's model; the `one-fault` variants apply a single X mechanism
    per CNOT at probability $p$, which is the most literal reading of
    Theorem 2. They differ only in which qubit of the pair carries it, which
    changes the fault's detector weight.

    Detectors are Z-check only in every variant — the observables are Z-type,
    so X-check detectors cannot help predict them (§4 of the notebook) —
    and the detector order is $m_z$ per round then $m_z$ from the final data
    readout, which is `build_memory_circuit`'s contract at
    `x_detectors=False, use_flags=False`.
    """)
    return


@app.cell
def _(check_schedule, np, stim):
    GATE_NOISE = {
        "depolarize2": "the notebook's model: DEPOLARIZE2, 15 components per CNOT",
        "one-fault-control": "one X mechanism per CNOT, on the control",
        "one-fault-target": "one X mechanism per CNOT, on the target",
        "one-fault-both": "one correlated XX mechanism per CNOT",
    }
    EXTRACTION = {
        "z-only": "Z checks only -- nw CNOTs per round, as Theorem 1 counts",
        "both": "the notebook's model: X and Z extraction, 2nw CNOTs per round",
    }

    def build_reference_circuit(code, rounds, p, gate_noise="depolarize2",
                                extraction="z-only", measurement_noise=True,
                                boundary_noise=True):
        """
        Z-basis memory in the Pakhunov reference model.

        The defaults reproduce the paper's counting exactly: Z-check extraction
        only, DEPOLARIZE2 per CNOT, one measurement flip per ancilla per round,
        and boundary faults from the data reset and the final readout.

        Set extraction="both" to recover the notebook's own circuit, so the two
        can be compared on one axis at a time.
        """
        if gate_noise not in GATE_NOISE:
            raise ValueError(f"gate_noise must be one of {sorted(GATE_NOISE)}")
        if extraction not in EXTRACTION:
            raise ValueError(f"extraction must be one of {sorted(EXTRACTION)}")
        if rounds < 1:
            raise ValueError("rounds must be >= 1")

        n, mx, mz = code.n, code.hx.shape[0], code.hz.shape[0]
        data = list(range(n))
        zanc = list(range(n, n + mz))
        xanc = list(range(n + mz, n + mz + mx)) if extraction == "both" else []

        sz = check_schedule(code.hz, code.lattice)
        sx = check_schedule(code.hx, code.lattice) if xanc else None
        hz_rows = [code.hz.indices[code.hz.indptr[i]:code.hz.indptr[i + 1]].tolist()
                   for i in range(mz)]
        lz = np.asarray(code.lz, dtype=np.uint8)

        c = stim.Circuit()
        meas = [0]

        def rec(i):
            return stim.target_rec(i - meas[0])

        def cx_layer(pairs):
            flat = [q for pair in pairs for q in pair]
            c.append("CX", flat)
            if p > 0:
                controls, targets = flat[0::2], flat[1::2]
                if gate_noise == "depolarize2":
                    c.append("DEPOLARIZE2", flat, p)
                elif gate_noise == "one-fault-control":
                    c.append("X_ERROR", controls, p)
                elif gate_noise == "one-fault-target":
                    c.append("X_ERROR", targets, p)
                else:                      # one correlated XX per gate
                    for a, b in zip(controls, targets):
                        c.append("CORRELATED_ERROR",
                                 [stim.target_x(a), stim.target_x(b)], p)
            c.append("TICK")

        def measure_into(name, qubits):
            if not qubits:
                return []
            if p > 0 and measurement_noise:
                c.append(name, qubits, p)
            else:
                c.append(name, qubits)
            start = meas[0]
            meas[0] += len(qubits)
            return list(range(start, meas[0]))

        # ---- reset -----------------------------------------------------------
        c.append("R", data + zanc)
        if xanc:
            c.append("RX", xanc)
        if p > 0 and boundary_noise:
            c.append("X_ERROR", data, p)          # n boundary faults, Theorem 1
        if p > 0:
            c.append("X_ERROR", zanc, p)
        c.append("TICK")

        prev_z = None
        for t in range(rounds):
            if xanc:                              # only to mirror the notebook
                for lay in range(sx.shape[0]):
                    cx_layer([(xanc[r], int(sx[lay, r])) for r in range(mx)])
            for lay in range(sz.shape[0]):
                cx_layer([(int(sz[lay, r]), zanc[r]) for r in range(mz)])

            if xanc:
                measure_into("MRX", xanc)
            z_idx = measure_into("MR", zanc)
            if p > 0 and t < rounds - 1:
                c.append("X_ERROR", zanc, p)

            for i in range(mz):
                tg = [rec(z_idx[i])] + ([rec(prev_z[i])] if t > 0 else [])
                c.append("DETECTOR", tg, [i, t, 0])
            c.append("TICK")
            prev_z = z_idx

        # ---- final data readout ----------------------------------------------
        d_idx = measure_into("M", data)
        for i in range(mz):
            c.append("DETECTOR",
                     [rec(d_idx[q]) for q in hz_rows[i]] + [rec(prev_z[i])],
                     [i, rounds, 3])
        for j in range(lz.shape[0]):
            c.append("OBSERVABLE_INCLUDE",
                     [rec(d_idx[q]) for q in np.flatnonzero(lz[j])], j)
        return c
    return EXTRACTION, GATE_NOISE, build_reference_circuit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Measuring a circuit against the theory

    `measure` builds one variant, extracts its detector error model and
    reports every quantity the theorems predict beside the one they do not:
    the fraction of shots our greedy peeler actually resolves on its own,
    which is what Table II's "Actual" column reports.

    The mean degree $\bar{d}$ is measured on the fault graph
    $G = H^{\top}H$, excluding self-loops, and $\lambda$ is read straight off
    the DEM as $\sum_i p_i$. Neither is assumed from the theory — that is the
    whole point, since $P_{\text{peel}}$ is then a prediction about our
    circuit rather than about the paper's.
    """)
    return


@app.cell
def _(A0, A0_DEFAULT, build_reference_circuit, dem_to_matrices, dz, np,
      theory_faults, theory_lambda, theory_peel, wilson):
    def measure(code, rounds, p, shots=2000, seed=7, workers=1, **kwargs):
        """
        Build one variant and report Theorems 1, 2, 3 and 5 against measurement.

        `peel` is the fraction of shots greedy peeling resolves unaided, which is
        what Table II's "Actual" column measures.
        """
        circ = build_reference_circuit(code, rounds, p, **kwargs)
        H, L, priors = dem_to_matrices(
            circ.detector_error_model(decompose_errors=False))
        th = theory_faults(code, rounds)

        det, obs = circ.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True)
        w = dz.parallel_workers(shots, workers)
        res, _backend = dz.run_zoo(H, L, priors, det, obs,
                                   [("weak", {"kind": "peel"})],
                                   workers=w, chunk_size=dz.chunk_for(shots, w))
        resolved = int(res["weak"]["conv"].sum())
        _, lo, hi = wilson(resolved, shots)

        Hb = (H > 0).astype(np.int8)
        G = (Hb.T @ Hb).tocsr()
        mean_degree = float((np.diff(G.indptr) - 1).mean())
        lam = float(priors.sum())
        a0 = A0.get(getattr(code, "name", ""), A0_DEFAULT)
        return dict(
            detectors=int(H.shape[0]), faults=int(H.shape[1]),
            theory_faults=th["total"], fault_ratio=H.shape[1] / th["total"],
            lam=lam, theory_lambda=theory_lambda(code, rounds, p),
            lambda_ratio=lam / theory_lambda(code, rounds, p),
            mean_degree=mean_degree, a0=a0,
            theory_peel=theory_peel(H.shape[1], lam, mean_degree, a0),
            peel=resolved / shots, peel_lo=lo, peel_hi=hi,
            uniform_weight=th["uniform"], shots=shots,
            column_weight=th["column_weight"],
            cnot=th["cnot"], meas_faults=th["measurement"],
            boundary=th["boundary"])
    return (measure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Table II and the variants

    Table II's "Actual" column, and the four circuits worth putting beside
    it. Reading down the table in §7 isolates the cause: rows 1–3 change the
    **channel** and barely move, row 4 changes the **extraction** and moves
    everything.
    """)
    return


@app.cell
def _():
    # Pakhunov (2026), Table II, "Actual" column: greedy-peeling success rate.
    TABLE_II = {
        ("[[72, 12, 6]]", 12, 0.001): 0.935,
        ("[[144, 12, 12]]", 12, 0.001): 0.879,
        ("[[288, 12, 18]]", 12, 0.001): 0.778,
        ("[[144, 12, 12]]", 12, 0.003): 0.334,
        ("[[288, 12, 18]]", 12, 0.003): 0.119,
    }

    VARIANTS = [
        ("Pakhunov reference",
         dict(gate_noise="depolarize2", extraction="z-only")),
        ("  .. 1 fault/CNOT (control)",
         dict(gate_noise="one-fault-control", extraction="z-only")),
        ("  .. correlated XX",
         dict(gate_noise="one-fault-both", extraction="z-only")),
        ("notebook (both check families)",
         dict(gate_noise="depolarize2", extraction="both")),
    ]
    return TABLE_II, VARIANTS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Controls

    Defaults reproduce the $[[72,12,6]]$ row of Table II. Peeling is cheap,
    so a few thousand shots pin the rate to under a percentage point —
    `shots` matters far less here than in the experiments, because the
    quantity measured is a *per-shot* success rate near 0.9 rather than a
    logical error rate near $10^{-4}$.
    """)
    return


@app.cell
def _(CODE_LABELS, VARIANTS, mo):
    ui_code = mo.ui.dropdown(CODE_LABELS, value="[[72, 12, 6]]", label="code")
    ui_rounds = mo.ui.multiselect(["3", "6", "12", "24"], value=["12"],
                                  label="rounds $T$")
    ui_p = mo.ui.dropdown(["0.0005", "0.001", "0.002", "0.003"], value="0.001",
                          label="physical error rate $p$")
    ui_shots = mo.ui.slider(500, 20000, value=2000, step=500, label="shots",
                            show_value=True)
    ui_workers = mo.ui.slider(1, 16, value=4, label="workers", show_value=True)
    ui_variants = mo.ui.multiselect([n for n, _ in VARIANTS],
                                    value=[n for n, _ in VARIANTS],
                                    label="variants")
    mo.vstack([
        mo.hstack([ui_code, ui_p, ui_rounds], justify="start", gap=2),
        mo.hstack([ui_shots, ui_workers], justify="start", gap=2),
        ui_variants,
    ])
    return ui_code, ui_p, ui_rounds, ui_shots, ui_variants, ui_workers


@app.cell
def _(mo):
    ui_run = mo.ui.run_button(label="Measure")
    ui_run
    return (ui_run,)


@app.cell
def _(VARIANTS, code_by_label, measure, mo, ui_code, ui_p, ui_rounds, ui_run,
      ui_shots, ui_variants, ui_workers):
    pk_rows = None
    if ui_run.value:
        _c = code_by_label(ui_code.value)
        _p = float(ui_p.value)
        _Ts = sorted(int(t) for t in ui_rounds.value) or [12]
        _vs = [v for v in VARIANTS if v[0] in ui_variants.value] or VARIANTS[:1]
        pk_rows = []
        with mo.status.progress_bar(
                total=len(_Ts) * len(_vs), title="measuring variants") as _bar:
            for _T in _Ts:
                for _name, _kw in _vs:
                    _bar.update(subtitle=f"T={_T} · {_name.replace('  .. ', '')}")
                    pk_rows.append(dict(
                        variant=_name, rounds=_T, p=_p, code=ui_code.value,
                        kwargs=_kw,
                        **measure(_c, _T, _p, shots=ui_shots.value,
                                  workers=ui_workers.value, **_kw)))
    return (pk_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Results

    `P_peel` is the Theorem 3/5 prediction evaluated on **that variant's own
    measured graph**; `peeled` is what our decoder actually achieved on it.
    The distance between those two columns is our decoder; the distance
    between `P_peel` and Table II is our *model*. Separating them is the
    reason this notebook exists.

    Every run is written to `results/` as `PK_<code>_T<rounds>_p<p>_<shots>shots.json`,
    in the same schema the experiments use, and exported beside it as figures,
    a CSV and a `report.md`. Pick an earlier run below to read it back without
    re-measuring.
    """)
    return


@app.cell
def _(mo, saved_run_picker):
    ui_pk_file, _saved = saved_run_picker("PK", "load a saved run")
    mo.vstack([ui_pk_file,
               mo.md(f"*{len(_saved)} saved run(s) in `results/`*")])
    return (ui_pk_file,)


@app.cell
def _(read_result_file, ui_pk_file):
    pk_loaded = None
    if ui_pk_file.value:
        pk_loaded = read_result_file(ui_pk_file.value, "pakhunov")["rows"]
    return (pk_loaded,)


@app.cell
def _(RESULTS_DIR, TABLE_II, code_by_label, export_bundle, mo, np, os,
      pk_loaded, pk_rows, plt, theory_faults, theory_lambda, ui_pk_file,
      write_result_file):
    pk_data = pk_rows if pk_rows is not None else pk_loaded
    mo.stop(pk_data is None,
            mo.md("*Set the options in §6 and press **Measure**, or load a saved "
                  "run above.*"))

    _key, _p = pk_data[0]["code"], pk_data[0]["p"]
    _c = code_by_label(_key)
    _Ts = sorted({r["rounds"] for r in pk_data})
    # The headline T is the one Table II actually covers, so a sweep that also
    # measures T = 6 does not report "nothing to reproduce".
    _T0 = next((t for t in _Ts if (_key, t, _p) in TABLE_II), _Ts[-1])
    _names = [r["variant"] for r in pk_data if r["rounds"] == _T0]

    # ---- the theory header, per T -------------------------------------------
    _hdr = []
    for _T in _Ts:
        _th = theory_faults(_c, _T)
        _tgt = TABLE_II.get((_key, _T, _p))
        _hdr.append(
            f"**$T = {_T}$** — Theorem 1: {_th['total']:,} faults = "
            f"{_th['cnot']:,} CNOT + {_th['measurement']:,} measurement + "
            f"{_th['boundary']:,} boundary (column weight $w = {_th['column_weight']}$)"
            f" · Theorem 2: $\\lambda = {theory_lambda(_c, _T, _p):.2f}$"
            + (f" · Table II: **{_tgt:.1%}**" if _tgt else
               " · *no Table II entry for this point*"))

    # ---- the comparison table ------------------------------------------------
    _tbl = ["| variant | $T$ | faults | /thry | $\\lambda$ | /thry | $\\bar{d}$ "
            "| $P_{peel}$ | peeled | Table II |",
            "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for _r in pk_data:
        _tgt = TABLE_II.get((_key, _r["rounds"], _p))
        _tbl.append(
            f"| {_r['variant'].replace('  ..', '&nbsp;&nbsp;↳')} | {_r['rounds']} "
            f"| {_r['faults']:,} | {_r['fault_ratio']:.2f} | {_r['lam']:.2f} "
            f"| {_r['lambda_ratio']:.2f} | {_r['mean_degree']:.1f} "
            f"| {_r['theory_peel']:.1%} "
            f"| {_r['peel']:.1%} [{_r['peel_lo']:.1%}, {_r['peel_hi']:.1%}] "
            f"| {f'{_tgt:.1%}' if _tgt else '—'} |")

    # ---- figure --------------------------------------------------------------
    _fig, (_a, _b, _c3) = plt.subplots(1, 3, figsize=(15, 4.4))
    _x = np.arange(len(_names))
    _first = [r for r in pk_data if r["rounds"] == _T0]
    _short = [n.replace("  .. ", "↳ ").replace(" (both check families)", "\n(both)")
               .replace(" (control)", "\n(control)").replace("Pakhunov ", "Pakhunov\n")
              for n in _names]

    _a.bar(_x - 0.2, [r["fault_ratio"] for r in _first], 0.4,
           color="#1f77b4", label="faults / Theorem 1")
    _a.bar(_x + 0.2, [r["lambda_ratio"] for r in _first], 0.4,
           color="#ff7f0e", label="$\\lambda$ / Theorem 2")
    _a.axhline(1.0, color="k", ls="--", lw=1)
    _a.set_xticks(_x, _short, fontsize=7)
    _a.set_ylabel("measured / predicted")
    _a.set_title(f"Theorems 1 and 2 ($T={_T0}$)")
    _a.legend(fontsize=8)

    _b.bar(_x, [r["mean_degree"] for r in _first], 0.5, color="#8c564b")
    _b.axhline(52.3, color="#d62728", ls="--", lw=1, label="paper: 52.3")
    _b.set_xticks(_x, _short, fontsize=7)
    _b.set_ylabel("mean fault-graph degree $\\bar{d}$")
    _b.set_title("The graph the peeler sees")
    _b.legend(fontsize=8)

    _tgt0 = TABLE_II.get((_key, _T0, _p))
    _c3.bar(_x - 0.2, [r["theory_peel"] for r in _first], 0.4,
            color="#9467bd", label="$P_{peel}$ (Thm 3/5 on our graph)")
    _c3.bar(_x + 0.2, [r["peel"] for r in _first], 0.4,
            color="#2ca02c", label="our peeling decoder")
    _c3.errorbar(_x + 0.2, [r["peel"] for r in _first],
                 yerr=[[r["peel"] - r["peel_lo"] for r in _first],
                       [r["peel_hi"] - r["peel"] for r in _first]],
                 fmt="none", color="k", capsize=4)
    if _tgt0:
        _c3.axhline(_tgt0, color="#d62728", ls="--", lw=1,
                    label=f"Table II: {_tgt0:.3f}")
    _c3.set_xticks(_x, _short, fontsize=7)
    _c3.set_ylim(0, 1)
    _c3.set_ylabel("fraction of shots peeling resolves")
    _c3.set_title("Model vs decoder vs literature")
    _c3.legend(fontsize=7)
    _fig.tight_layout()

    # ---- the sweep, when more than one T was measured ------------------------
    _fig2 = None
    if len(_Ts) > 1:
        _fig2, (_d, _e) = plt.subplots(1, 2, figsize=(11, 4))
        for _n in _names:
            _s = [r for r in pk_data if r["variant"] == _n]
            _d.plot([r["rounds"] for r in _s], [r["fault_ratio"] for r in _s],
                    "o-", label=_n.replace("  .. ", ""))
            _e.plot([r["rounds"] for r in _s], [r["peel"] for r in _s], "o-",
                    label=_n.replace("  .. ", ""))
        _d.axhline(1.0, color="k", ls="--", lw=1)
        _d.set_xlabel("rounds $T$"); _d.set_ylabel("faults / Theorem 1")
        _d.set_title("Theorem 1 across $T$"); _d.legend(fontsize=7)
        _e.set_xlabel("rounds $T$"); _e.set_ylabel("peeling success")
        _e.set_title("What our decoder resolves"); _e.legend(fontsize=7)
        _fig2.tight_layout()

    # ---- the verdict ---------------------------------------------------------
    _ref = _first[0]
    _tgt = _tgt0
    if _tgt is None:
        _verdict = ("No Table II entry for this code, $T$ and $p$, so there is "
                    "nothing to reproduce here — read the ratios instead.")
    elif abs(_ref["theory_peel"] - _tgt) < 0.02:
        _verdict = (
            f"**The model reproduces the literature.** On the reference circuit "
            f"the Theorem 3/5 prediction evaluated on our own measured graph is "
            f"{_ref['theory_peel']:.3f} against Table II's {_tgt:.3f}. Our "
            f"decoder then resolves {_ref['peel']:.3f}, so the residual "
            f"**{_ref['theory_peel'] - _ref['peel']:+.3f}** is our peeling "
            f"implementation, not the physics.")
    else:
        _verdict = (
            f"**The model does not reproduce the literature here.** Prediction "
            f"{_ref['theory_peel']:.3f} against Table II's {_tgt:.3f} — a gap of "
            f"{_ref['theory_peel'] - _tgt:+.3f}, which is a modelling "
            f"discrepancy rather than a decoder one.")
    if not _ref["uniform_weight"]:
        _verdict += ("\n\n*This code's column weight is not uniform, so "
                     "Theorem 1's count is nominal; read $\\lambda$ and the "
                     "peeling rate instead of the fault ratio.*")

    # ---- persistence ---------------------------------------------------------
    # A fresh run is written and exported; a loaded one is only displayed, so
    # reading an old file back never overwrites it.
    if pk_rows is not None:
        _tag = _key.strip("[]").replace(", ", "-")
        _Tstr = "-".join(str(t) for t in _Ts)
        _path = write_result_file(
            os.path.join(RESULTS_DIR,
                         f"PK_{_tag}_T{_Tstr}_p{_p:g}_{_first[0]['shots']}shots.json"),
            {"kind": "pakhunov",
             "config": {"code": _key, "p": _p, "rounds": _Ts,
                        "shots": _first[0]["shots"],
                        "variants": _names,
                        "headline_rounds": _T0,
                        "table_ii": _tgt0},
             "source": "Pakhunov (2026), Table II, 'Actual' column",
             "rows": pk_data})
        _figs = {"variants": _fig} | ({"sweep": _fig2} if _fig2 is not None else {})
        _folder = export_bundle(
            _path, f"Pakhunov reference model — {_key}, p={_p:g}, T={_Tstr}",
            _figs,
            {"variants": [{k: v for k, v in r.items() if k != "kwargs"}
                          for r in pk_data]},
            notes=_verdict)
        _where = mo.md(f"Saved to `{_path}` · exported to `{_folder}`")
    else:
        _where = mo.md(f"Loaded from `{ui_pk_file.value}` — press **Measure** to "
                       "produce a new run.")

    mo.vstack([
        mo.md(f"### {_key}, $p = {_p:g}$, {_first[0]['shots']:,} shots"),
        mo.md("\n\n".join(_hdr)),
        mo.md("\n".join(_tbl)),
        mo.md(_verdict),
        _where,
        _fig,
    ] + ([_fig2] if _fig2 is not None else []))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Self-tests

    These run on every edit and take a few seconds. The ones that matter most
    are 4–6: they assert that this notebook and `decoderSwitch.py` §5.1 build
    the *same* circuit and predict the *same* numbers, so the two can never
    silently drift apart. If §5.1 changes, these fail here.
    """)
    return


@app.cell
def _(TABLE_II, build_reference_circuit, code_by_label, dem_to_matrices,
      notebook, run_checks, theory_faults, theory_lambda, theory_peel):
    _c72 = code_by_label("[[72, 12, 6]]")
    _nb_brc, _nb_tp, _nb_t2, _nb_bmc, _nb_rep = notebook(
        "build_reference_circuit", "theory_peel", "TABLE_II",
        "build_memory_circuit", "reference_report")

    def _dem(circ):
        """(detectors, faults, lambda) for a circuit's undecomposed DEM."""
        H, _L, pr = dem_to_matrices(
            circ.detector_error_model(decompose_errors=False))
        return H.shape[0], H.shape[1], float(pr.sum())

    def _t1_exact():
        assert theory_faults(_c72, 12)["total"] == 3096
        assert theory_faults(_c72, 12)["column_weight"] == 3

    def _circuit_hits_t1():
        for T in (3, 6, 12):
            f = _dem(build_reference_circuit(_c72, T, 1e-3))[1]
            assert f == theory_faults(_c72, T)["total"], (T, f)

    def _lambda_near_t2():
        lam = _dem(build_reference_circuit(_c72, 12, 1e-3))[2]
        assert abs(lam / theory_lambda(_c72, 12, 1e-3) - 1) < 0.05, lam

    def _same_peel_formula():
        assert abs(theory_peel(3096, 3.08, 50.6)
                   - _nb_tp(3096, 3.08, 50.6)) < 1e-12

    def _same_circuit_as_notebook():
        for T in (3, 12):
            a = build_reference_circuit(_c72, T, 1e-3)
            b = _nb_brc(_c72, T, 1e-3)
            assert (str(a.detector_error_model(decompose_errors=False))
                    == str(b.detector_error_model(decompose_errors=False))), \
                f"T={T}: DEM differs from section 5.1"

    def _both_matches_section_4():
        ours = _dem(build_reference_circuit(_c72, 12, 1e-3, extraction="both"))
        theirs = _dem(_nb_bmc(_c72, 12, 1e-3, use_flags=False, x_detectors=False))
        assert ours == theirs, (ours, theirs)

    def _table_ii_agrees():
        # the notebook keys on (code, T); we also carry p, so compare its slice
        ours = {(k, T): v for (k, T, p), v in TABLE_II.items() if p == 1e-3}
        for key, v in _nb_t2.items():
            assert ours.get(key) == v, (key, ours.get(key), v)

    def _prediction_lands_on_table_ii():
        r = _nb_rep(_c72, 12, 1e-3)
        assert abs(r["theory_peel"] - 0.935) < 0.02, r["theory_peel"]

    def _bad_options_rejected():
        for call in (lambda: build_reference_circuit(_c72, 3, 1e-3, gate_noise="nope"),
                     lambda: build_reference_circuit(_c72, 3, 1e-3, extraction="nope"),
                     lambda: build_reference_circuit(_c72, 0, 1e-3)):
            try:
                call()
            except ValueError:
                continue
            raise AssertionError("an invalid argument was accepted")

    def _noise_axis_does_not_move_it():
        f1 = _dem(build_reference_circuit(_c72, 12, 1e-3))[1]
        f2 = _dem(build_reference_circuit(_c72, 12, 1e-3,
                                          gate_noise="one-fault-control"))[1]
        assert f1 == f2, (f1, f2)          # same fault count, different channel

    def _extraction_axis_does():
        f1 = _dem(build_reference_circuit(_c72, 12, 1e-3))[1]
        f2 = _dem(build_reference_circuit(_c72, 12, 1e-3, extraction="both"))[1]
        assert f2 / f1 > 1.35, f2 / f1

    tests_pakhunov = run_checks([
        ("Theorem 1 is exact for [[72,12,6]] at T=12", _t1_exact),
        ("the reference circuit hits Theorem 1 for T in {3,6,12}", _circuit_hits_t1),
        ("lambda is within 5% of Theorem 2", _lambda_near_t2),
        ("P_peel matches decoderSwitch section 5.1", _same_peel_formula),
        ("the circuit is bit-identical to section 5.1's", _same_circuit_as_notebook),
        ("extraction='both' reproduces section 4's circuit", _both_matches_section_4),
        ("Table II agrees with the notebook's copy", _table_ii_agrees),
        ("the prediction lands on Table II's 93.5%", _prediction_lands_on_table_ii),
        ("bad gate_noise, extraction and rounds are rejected", _bad_options_rejected),
        ("the noise channel does not change the fault count", _noise_axis_does_not_move_it),
        ("the extraction axis does, by more than 35%", _extraction_axis_does),
    ])
    return (tests_pakhunov,)


@app.cell(hide_code=True)
def _(mo, tests_pakhunov):
    _bad = [r for r in tests_pakhunov if r["status"] != "PASS"]
    mo.md(
        f"**{len(tests_pakhunov) - len(_bad)}/{len(tests_pakhunov)} checks pass**\n\n"
        + "\n".join(f"- `{r['status']}` {r['name']}"
                    + (f" — {r['detail']}" if r["detail"] else "")
                    for r in tests_pakhunov))
    return


if __name__ == "__main__":
    app.run()
