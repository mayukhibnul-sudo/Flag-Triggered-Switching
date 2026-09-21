import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Decoder Switching for BB Codes")


# =============================================================================
# 0. Setup
# =============================================================================
@app.cell
def _():
    import json
    import time
    from dataclasses import asdict, dataclass, field

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.sparse as sp
    import stim

    return asdict, dataclass, field, json, mo, np, plt, sp, stim, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Flag-Triggered Decoder Switching for Bivariate Bicycle Codes

    **TODO:** authors, supervisor, BRAC University, date

    ---

    *One-sentence claim of the thesis goes here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Outline

    1. Motivation and research question
    2. GF(2) linear algebra
    3. Bivariate bicycle codes
    4. Circuit-level noise and flag qubits
    5. Detector error model
    6. Decoders
    7. Switch policies
    8. Statistics and benchmarking
    9. Reproducibility
    10. Implementation status
    11. Experiments
    12. Results, discussion, conclusion
    """)
    return


@app.cell
def _(mo):
    STATUS_ICON = {"PASS": "✅", "FAIL": "❌", "TODO": "⬜"}

    def run_checks(checks):
        """
        Run a list of (name, zero-arg callable) checks.

        A check PASSES if it returns without raising, FAILS on any exception,
        and is TODO if it hits NotImplementedError -- so an unfinished skeleton
        shows its progress instead of crashing.
        """
        results = []
        for name, fn in checks:
            try:
                fn()
                results.append({"name": name, "status": "PASS", "detail": ""})
            except NotImplementedError as exc:
                results.append({"name": name, "status": "TODO", "detail": str(exc)})
            except Exception as exc:
                results.append({"name": name, "status": "FAIL",
                                "detail": f"{type(exc).__name__}: {exc}"})
        return results

    def render_checks(title, results):
        counts = {s: sum(r["status"] == s for r in results) for s in STATUS_ICON}
        lines = [f"### Tests — {title}",
                 f"{counts['PASS']} passed · {counts['FAIL']} failed · {counts['TODO']} to do",
                 ""]
        for r in results:
            detail = f" — `{r['detail']}`" if r["status"] == "FAIL" and r["detail"] else ""
            lines.append(f"- {STATUS_ICON[r['status']]} {r['name']}{detail}")
        return mo.md("\n".join(lines))

    def all_pass(*result_lists):
        return all(r["status"] == "PASS" for rl in result_lists for r in rl)

    return STATUS_ICON, all_pass, render_checks, run_checks


# =============================================================================
# 1. Motivation
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Motivation and research question

    **TODO — write these in your own words:**

    - **Problem.** Why decoder latency matters for real-time QEC; the backlog problem.
    - **Existing approach.** Fast-primary / accurate-secondary switching that
      decides *after* the primary decoder runs (post-hoc reliability).
    - **Gap.** A trigger available *before* decoding — a hardware flag — could
      skip the wasted primary pass on hard syndromes.
    - **Research question.** Does a pre-decode flag trigger reduce latency or
      escalation rate relative to the trivial "escalate when the primary fails"
      rule, at equal logical error rate?
    - **Hypothesis and what would falsify it.**
    """)
    return


# =============================================================================
# 2. GF(2) linear algebra
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. GF(2) linear algebra

    Everything downstream — code dimension $k$, logical operators, scoring —
    depends on rank and nullspace computations **over GF(2)**, not over the reals.
    """)
    return


@app.cell
def _(np):
    def gf2_rref(M):
        """
        Reduced row echelon form of M over GF(2).

        Args:
            M: 2-D array-like (dense or scipy.sparse) of 0/1 entries.
        Returns:
            (R, pivots): R is a uint8 array the same shape as M in RREF;
            pivots is the list of pivot column indices, in increasing order.
        """
        # TODO
        raise NotImplementedError("gf2_rref")

    def gf2_rank(M):
        """Rank of M over GF(2). Must return 0 for an empty matrix."""
        # TODO
        raise NotImplementedError("gf2_rank")

    def gf2_nullspace(M):
        """
        Basis of {v : M v = 0 (mod 2)}.

        Returns:
            uint8 array of shape (n_cols - rank(M), n_cols); rows are basis vectors.
        """
        # TODO
        raise NotImplementedError("gf2_nullspace")

    def gf2_quotient_basis(subspace, ambient):
        """
        Rows of `ambient` that are linearly independent modulo rowspace(`subspace`).

        Used for logical operators: logicals = ker(...) modulo stabilisers.
        Returns a uint8 array with shape (dim, n_cols); may have zero rows.
        """
        # TODO
        raise NotImplementedError("gf2_quotient_basis")

    return gf2_nullspace, gf2_quotient_basis, gf2_rank, gf2_rref


@app.cell
def _(
    gf2_nullspace,
    gf2_quotient_basis,
    gf2_rank,
    gf2_rref,
    np,
    render_checks,
    run_checks,
    sp,
):
    # Over the reals this matrix has rank 3; over GF(2) the rows sum to zero.
    _M = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)

    def _rank_is_gf2():
        assert gf2_rank(_M) == 2, f"got {gf2_rank(_M)} (real-valued rank is 3)"
        assert gf2_rank(np.eye(5, dtype=np.uint8)) == 5
        assert gf2_rank(np.ones((3, 4), dtype=np.uint8)) == 1
        assert gf2_rank(np.zeros((0, 4), dtype=np.uint8)) == 0

    def _rank_accepts_sparse():
        assert gf2_rank(sp.csr_matrix(_M)) == 2

    def _rref_pivots():
        R, piv = gf2_rref(_M)
        assert list(piv) == [0, 1], f"pivots {piv}"
        assert not R[2].any(), "last row should reduce to zero"
        assert R.dtype == np.uint8

    def _nullspace_is_kernel():
        rng = np.random.default_rng(0)
        for _ in range(20):
            A = rng.integers(0, 2, size=(6, 10), dtype=np.uint8)
            N = gf2_nullspace(A)
            assert N.shape == (10 - gf2_rank(A), 10), f"shape {N.shape}"
            assert not ((A @ N.T) % 2).any(), "basis vector not in kernel"
            assert gf2_rank(N) == N.shape[0], "basis vectors not independent"

    def _quotient_basis():
        sub = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        amb = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.uint8)
        Q = gf2_quotient_basis(sub, amb)
        assert Q.shape[0] == 1, f"expected 1 independent row, got {Q.shape[0]}"

    tests_gf2 = run_checks([
        ("rank is computed over GF(2), not the reals", _rank_is_gf2),
        ("rank accepts scipy.sparse input", _rank_accepts_sparse),
        ("rref returns correct pivots", _rref_pivots),
        ("nullspace basis spans the kernel (20 random matrices)", _nullspace_is_kernel),
        ("quotient basis drops rows already in the subspace", _quotient_basis),
    ])
    render_checks("GF(2)", tests_gf2)
    return (tests_gf2,)


# =============================================================================
# 3. Bivariate bicycle codes
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Bivariate bicycle codes

    On an $L \times M$ torus with $x = S_L \otimes I_M$ and $y = I_L \otimes S_M$,
    choose polynomials $A, B$ in $x, y$. Then

    $$H_X = [A \mid B], \qquad H_Z = [B^\top \mid A^\top],$$

    and $H_X H_Z^\top = AB + BA = 0$ because $A$ and $B$ commute.

    For a CSS code, $k = n - \operatorname{rank} H_X - \operatorname{rank} H_Z$ and

    - $L_X$ = basis of $\ker H_Z$ modulo $\operatorname{rowspace} H_X$
    - $L_Z$ = basis of $\ker H_X$ modulo $\operatorname{rowspace} H_Z$

    A residual X error is a **logical** error iff it is undetected and
    anticommutes with some row of $L_Z$. Residuals equal to a stabiliser are
    **successes** — BB codes are highly degenerate, so this matters.
    """)
    return


@app.cell
def _(dataclass, np):
    @dataclass
    class CSSCode:
        """Container for a CSS code. Fill it via make_css_code, not by hand."""
        name: str
        hx: object          # scipy.sparse csr, X-type stabilisers (m_x, n)
        hz: object          # scipy.sparse csr, Z-type stabilisers (m_z, n)
        n: int
        k: int
        rank_x: int
        rank_z: int
        lx: np.ndarray      # (k, n) logical X operators
        lz: np.ndarray      # (k, n) logical Z operators

    # Reference data. k and n are verified by the tests; d is the literature value
    # (Bravyi et al., Nature 627, 778 (2024), Table 3) and is NOT tested.
    BB_PRESETS = {
        "[[72, 12, 6]]":   dict(L=6,  M=6,  a_terms=[(3, 0), (0, 1), (0, 2)],
                                b_terms=[(0, 3), (1, 0), (2, 0)], n=72,  k=12, d=6),
        "[[90, 8, 10]]":   dict(L=15, M=3,  a_terms=[(9, 0), (0, 1), (0, 2)],
                                b_terms=[(0, 0), (2, 0), (7, 0)], n=90,  k=8,  d=10),
        "[[108, 8, 10]]":  dict(L=9,  M=6,  a_terms=[(3, 0), (0, 1), (0, 2)],
                                b_terms=[(0, 3), (1, 0), (2, 0)], n=108, k=8,  d=10),
        "[[144, 12, 12]]": dict(L=12, M=6,  a_terms=[(3, 0), (0, 1), (0, 2)],
                                b_terms=[(0, 3), (1, 0), (2, 0)], n=144, k=12, d=12),
        "[[288, 12, 18]]": dict(L=12, M=12, a_terms=[(3, 0), (0, 2), (0, 7)],
                                b_terms=[(0, 3), (1, 0), (2, 0)], n=288, k=12, d=18),
    }
    # a_terms / b_terms: each (i, j) is the monomial x^i y^j.
    return BB_PRESETS, CSSCode


@app.cell
def _(CSSCode, gf2_nullspace, gf2_quotient_basis, gf2_rank, np, sp):
    def make_css_code(hx, hz, name="code"):
        """
        Build a CSSCode from two check matrices.

        Must:
          - store hx, hz as uint8 csr matrices
          - raise ValueError if hx @ hz.T != 0 (mod 2)
          - compute rank_x, rank_z, k
          - compute lx, lz (see the section text) with exactly k rows each
        """
        # TODO
        raise NotImplementedError("make_css_code")

    def make_bb_code(L, M, a_terms, b_terms, name=None):
        """
        Bivariate bicycle code on an L x M torus.

        Args:
            a_terms, b_terms: lists of (i, j) exponent pairs, monomial x^i y^j.
        Returns:
            CSSCode with hx = [A | B], hz = [B^T | A^T].
        Pitfall: after reducing entries mod 2, call .eliminate_zeros() on sparse
        matrices, otherwise stored zeros corrupt row/column weight counts.
        """
        # TODO
        raise NotImplementedError("make_bb_code")

    def bb_from_preset(key, presets):
        """Convenience: build a preset from BB_PRESETS by its label."""
        s = presets[key]
        return make_bb_code(s["L"], s["M"], s["a_terms"], s["b_terms"], name=key)

    return bb_from_preset, make_bb_code, make_css_code


@app.cell
def _(
    BB_PRESETS,
    bb_from_preset,
    gf2_rank,
    make_css_code,
    np,
    render_checks,
    run_checks,
):
    def _code72():
        return bb_from_preset("[[72, 12, 6]]", BB_PRESETS)

    def _presets_n_k():
        for key, spec in BB_PRESETS.items():
            c = bb_from_preset(key, BB_PRESETS)
            assert (c.n, c.k) == (spec["n"], spec["k"]), \
                f"{key}: built [[{c.n}, {c.k}]], expected [[{spec['n']}, {spec['k']}]]"

    def _css_condition():
        for key in BB_PRESETS:
            c = bb_from_preset(key, BB_PRESETS)
            assert not ((c.hx @ c.hz.T).toarray() % 2).any(), f"{key}"

    def _weights():
        c = _code72()
        for H in (c.hx, c.hz):
            assert set(np.asarray(H.sum(axis=1)).ravel()) == {6}, "stabiliser weight != 6"
            assert set(np.asarray(H.sum(axis=0)).ravel()) == {3}, "qubit degree != 3"

    def _rejects_non_css():
        hx = np.array([[1, 0]], dtype=np.uint8)
        hz = np.array([[1, 0]], dtype=np.uint8)
        try:
            make_css_code(hx, hz)
        except ValueError:
            return
        raise AssertionError("make_css_code accepted hx @ hz.T != 0")

    def _logicals_commute():
        c = _code72()
        assert c.lx.shape == c.lz.shape == (c.k, c.n)
        assert not ((c.hx @ c.lz.T) % 2).any(), "lz must commute with X stabilisers"
        assert not ((c.hz @ c.lx.T) % 2).any(), "lx must commute with Z stabilisers"

    def _logicals_nontrivial():
        c = _code72()
        assert gf2_rank(np.vstack([c.hz.toarray(), c.lz])) == c.rank_z + c.k
        assert gf2_rank(np.vstack([c.hx.toarray(), c.lx])) == c.rank_x + c.k

    def _logicals_pair():
        c = _code72()
        assert gf2_rank((c.lx @ c.lz.T) % 2) == c.k, "lx·lz^T must be full rank"

    def _degenerate_is_success():
        c = _code72()
        e_true = np.zeros(c.n, dtype=np.uint8)
        e_true[7] = 1
        e_hat = e_true ^ c.hx.toarray()[0].astype(np.uint8)   # differ by an X stabiliser
        diff = e_true ^ e_hat
        assert diff.sum() == 6
        assert not ((c.hz @ diff) % 2).any(), "difference should be undetected"
        assert not ((c.lz @ diff) % 2).any(), "stabiliser difference scored as logical error"

    tests_codes = run_checks([
        ("every preset builds with the expected [[n, k]]", _presets_n_k),
        ("CSS condition hx·hz^T = 0 for every preset", _css_condition),
        ("stabiliser weight 6, qubit degree 3", _weights),
        ("make_css_code rejects non-commuting check matrices", _rejects_non_css),
        ("logical operators commute with opposite-type stabilisers", _logicals_commute),
        ("logical operators are independent of same-type stabilisers", _logicals_nontrivial),
        ("lx and lz pair non-degenerately", _logicals_pair),
        ("stabiliser-equivalent correction scores as success", _degenerate_is_success),
    ])
    render_checks("codes", tests_codes)
    return (tests_codes,)


# =============================================================================
# 4. Circuit-level noise
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Circuit-level noise and flag qubits

    Hook errors are a property of the **syndrome-extraction circuit**: a fault on
    an X-check ancilla (the CNOT control) mid-extraction propagates to every data
    qubit it has not yet touched. They cannot appear in a code-capacity model.

    **Experiment:** Z-basis memory, $T$ rounds, circuit-level depolarising noise
    with strength $p$ on resets, single- and two-qubit gates, and measurements.

    **Flags:** one flag qubit per X-check ancilla; flag CNOTs bracket the middle
    of the extraction, so a mid-circuit ancilla fault flips the flag.

    **TODO:** a circuit diagram for one X-check with its flag.
    """)
    return


@app.cell
def _(np, stim):
    def build_memory_circuit(code, rounds, p, use_flags=False, idle_noise=False):
        """
        Z-basis memory experiment for a CSS code under circuit-level noise.

        Detector ORDER is part of the contract (flag_detector_mask depends on it):
          for each round t = 0 .. rounds-1:
            1. m_z Z-check detectors          (round 0: raw outcome; later: XOR with previous)
            2. m_x X-check detectors          (only for t >= 1; X outcomes are random at t = 0)
            3. m_x flag detectors             (only if use_flags)
          then m_z final detectors from the data-qubit readout.
        Observables: one per row of code.lz, from the final data readout.
        Qubits: data qubits are 0 .. code.n - 1; the circuit starts with a reset
        layer followed by a TICK.

        Pitfall: the CNOT schedule must keep every detector deterministic. If
        X- and Z-check CNOTs are interleaved carelessly, Stim raises when you call
        circuit.detector_error_model(). Running all X-check CNOT layers before all
        Z-check layers is always safe; Bravyi et al.'s depth-8 schedule is better.
        """
        # TODO
        raise NotImplementedError("build_memory_circuit")

    def flag_detector_mask(code, rounds, use_flags, num_detectors):
        """Boolean array of length num_detectors, True exactly at flag detectors."""
        # TODO
        raise NotImplementedError("flag_detector_mask")

    def expected_num_detectors(code, rounds, use_flags):
        """Detector count implied by the ordering contract above."""
        mx, mz = code.hx.shape[0], code.hz.shape[0]
        per_round_flags = mx if use_flags else 0
        return mz * rounds + mx * (rounds - 1) + per_round_flags * rounds + mz

    return build_memory_circuit, expected_num_detectors, flag_detector_mask


@app.cell
def _(
    BB_PRESETS,
    bb_from_preset,
    build_memory_circuit,
    expected_num_detectors,
    flag_detector_mask,
    np,
    render_checks,
    run_checks,
    stim,
):
    _T = 3

    def _code():
        return bb_from_preset("[[72, 12, 6]]", BB_PRESETS)

    def _noiseless_is_silent():
        c = _code()
        for flags in (False, True):
            circ = build_memory_circuit(c, _T, 0.0, use_flags=flags)
            det, obs = circ.compile_detector_sampler(seed=1).sample(
                32, separate_observables=True)
            assert not det.any(), f"flags={flags}: detectors fire with p = 0"
            assert not obs.any(), f"flags={flags}: observables flip with p = 0"

    def _dem_builds():
        c = _code()
        for flags in (False, True):
            circ = build_memory_circuit(c, _T, 1e-3, use_flags=flags)
            circ.detector_error_model(decompose_errors=False)   # raises if non-deterministic

    def _counts():
        c = _code()
        for flags in (False, True):
            circ = build_memory_circuit(c, _T, 1e-3, use_flags=flags)
            assert circ.num_detectors == expected_num_detectors(c, _T, flags), \
                f"flags={flags}: {circ.num_detectors} detectors"
            assert circ.num_observables == c.k

    def _flag_mask():
        c = _code()
        mx = c.hx.shape[0]
        circ = build_memory_circuit(c, _T, 1e-3, use_flags=True)
        m = flag_detector_mask(c, _T, True, circ.num_detectors)
        assert m.dtype == bool and m.shape == (circ.num_detectors,)
        assert m.sum() == mx * _T
        off = flag_detector_mask(c, _T, False, circ.num_detectors)
        assert not off.any()

    def _flags_see_faults():
        c = _code()
        circ = build_memory_circuit(c, _T, 1e-3, use_flags=True)
        m = flag_detector_mask(c, _T, True, circ.num_detectors)
        flag_ids = set(np.flatnonzero(m).tolist())
        dem = circ.detector_error_model(decompose_errors=False)
        seen = any(
            any(t.is_relative_detector_id() and t.val in flag_ids for t in inst.targets_copy())
            for inst in dem.flattened() if inst.type == "error")
        assert seen, "no fault mechanism triggers any flag detector"

    def _flags_are_quiet_at_zero_noise():
        c = _code()
        circ = build_memory_circuit(c, _T, 0.0, use_flags=True)
        m = flag_detector_mask(c, _T, True, circ.num_detectors)
        det = circ.compile_detector_sampler(seed=2).sample(32)
        assert not det[:, m].any()

    def _data_errors_never_flag():
        # A data-qubit X error cannot reach an X-check ancilla (the data qubit is
        # the CNOT target), so it must fire Z-check detectors and never a flag.
        c = _code()
        clean = build_memory_circuit(c, _T, 0.0, use_flags=True)
        first_tick = next(i for i, op in enumerate(clean) if op.name == "TICK")
        noisy = (clean[:first_tick + 1]
                 + stim.Circuit(f"X_ERROR(0.01) {' '.join(map(str, range(c.n)))}")
                 + clean[first_tick + 1:])
        m = flag_detector_mask(c, _T, True, noisy.num_detectors)
        dem = noisy.detector_error_model(decompose_errors=False)
        fired = set()
        for inst in dem.flattened():
            if inst.type == "error":
                fired |= {t.val for t in inst.targets_copy() if t.is_relative_detector_id()}
        assert fired, "data errors fired no detectors at all"
        hit = fired & set(np.flatnonzero(m).tolist())
        assert not hit, f"data X errors fire {len(hit)} 'flag' detectors: mask is misplaced"

    tests_circuit = run_checks([
        ("noiseless circuit: no detector fires, no observable flips", _noiseless_is_silent),
        ("noisy circuit yields a valid DEM (deterministic detectors)", _dem_builds),
        ("detector and observable counts match the ordering contract", _counts),
        ("flag mask has the right shape and count", _flag_mask),
        ("some fault mechanism fires a flag detector", _flags_see_faults),
        ("flags never fire without noise", _flags_are_quiet_at_zero_noise),
        ("data-qubit X errors never fire a flag detector", _data_errors_never_flag),
    ])
    render_checks("circuit", tests_circuit)
    return (tests_circuit,)


# =============================================================================
# 5. Detector error model
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Detector error model

    Stim reduces the noisy circuit to independent fault mechanisms. Each fault
    $j$ has a probability $p_j$, a detector signature (column $j$ of $H$), and a
    set of observables it flips (column $j$ of $L$). Decoding means: given the
    detector syndrome $s$, find $\hat e$ with $H\hat e = s$; predict $L\hat e$.
    """)
    return


@app.cell
def _(np, sp):
    def dem_to_matrices(dem):
        """
        Convert a stim.DetectorErrorModel into (H, L, priors).

        H:      csr uint8, (num_detectors, num_faults)
        L:      csr uint8, (num_observables, num_faults)
        priors: float array, (num_faults,)
        Fault j must be the j-th `error` instruction of dem.flattened() -- the
        tests check this against Stim's own sampler.
        """
        # TODO
        raise NotImplementedError("dem_to_matrices")

    return (dem_to_matrices,)


@app.cell
def _(
    BB_PRESETS,
    bb_from_preset,
    build_memory_circuit,
    dem_to_matrices,
    np,
    render_checks,
    run_checks,
):
    def _dem():
        c = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        circ = build_memory_circuit(c, 3, 2e-3, use_flags=True)
        return circ.detector_error_model(decompose_errors=False)

    def _shapes():
        dem = _dem()
        H, L, pr = dem_to_matrices(dem)
        assert H.shape == (dem.num_detectors, dem.num_errors)
        assert L.shape == (dem.num_observables, dem.num_errors)
        assert pr.shape == (dem.num_errors,)
        assert ((pr > 0) & (pr <= 0.5)).all()

    def _matches_stim_sampler():
        dem = _dem()
        H, L, _ = dem_to_matrices(dem)
        det, obs, err = dem.compile_sampler(seed=5).sample(300, return_errors=True)
        err = err.astype(np.uint8).T
        assert np.array_equal((H @ err % 2).T, det.astype(np.uint8)), "H disagrees with Stim"
        assert np.array_equal((L @ err % 2).T, obs.astype(np.uint8)), "L disagrees with Stim"

    tests_dem = run_checks([
        ("H, L, priors have consistent shapes; priors in (0, 0.5]", _shapes),
        ("H·e and L·e reproduce Stim's own sampled detectors/observables", _matches_stim_sampler),
    ])
    render_checks("DEM", tests_dem)
    return (tests_dem,)


# =============================================================================
# 6. Decoders
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Decoders

    Every decoder exposes one method, `decode(syndrome) -> DecodeResult`.

    - **Primary (fast):** greedy peeling (Pakhunov, 2026) — peel a fault only
      when its whole signature is active *and* it shares no detector with
      another fully-active fault.
    - **Secondary (accurate):** BP+OSD (`ldpc` package).

    `work` is an implementation-independent cost (e.g. peel passes, BP
    iterations). Report it alongside wall-clock time: Python timings are only a
    relative proxy and cannot support hardware latency claims.
    """)
    return


@app.cell
def _(dataclass, np):
    @dataclass
    class DecodeResult:
        correction: np.ndarray      # uint8, length num_faults
        converged: bool             # True iff H @ correction == syndrome (mod 2)
        work: int = 0               # decoder-specific operation count
        escalated: bool = False     # set by SwitchPolicy when the secondary ran

    return (DecodeResult,)


@app.cell
def _(DecodeResult, np, sp):
    class PeelingDecoder:
        """
        Greedy peeling over the DEM.

        Contract:
          - decode(zeros) returns an all-zero correction with converged=True
          - converged=True only if the returned correction reproduces the syndrome
          - work = number of peeling passes used
        """

        def __init__(self, H, priors=None, max_passes=20):
            # TODO: precompute whatever makes decode fast (column supports, H^T, ...)
            raise NotImplementedError("PeelingDecoder.__init__")

        def decode(self, syndrome):
            # TODO
            raise NotImplementedError("PeelingDecoder.decode")

    class BpOsdDecoder:
        """
        BP+OSD wrapper around ldpc.bposd_decoder.BpOsdDecoder.

        Contract: the correction always reproduces the syndrome (OSD guarantees
        this), so converged is always True. work = BP iterations used.
        """

        def __init__(self, H, priors, max_iter=20, osd_order=0):
            # TODO
            raise NotImplementedError("BpOsdDecoder.__init__")

        def decode(self, syndrome):
            # TODO
            raise NotImplementedError("BpOsdDecoder.decode")

    return BpOsdDecoder, PeelingDecoder


@app.cell
def _(
    BB_PRESETS,
    BpOsdDecoder,
    PeelingDecoder,
    bb_from_preset,
    build_memory_circuit,
    dem_to_matrices,
    np,
    render_checks,
    run_checks,
):
    def _setup(p=2e-3):
        c = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        circ = build_memory_circuit(c, 3, p, use_flags=True)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        det = circ.compile_detector_sampler(seed=9).sample(40).astype(np.uint8)
        return H, pr, det

    def _consistent(H, s, r):
        return np.array_equal((H @ np.asarray(r.correction, dtype=np.uint8)) % 2, s)

    def _peel_zero():
        H, pr, _ = _setup()
        r = PeelingDecoder(H, pr).decode(np.zeros(H.shape[0], dtype=np.uint8))
        assert r.converged and not np.asarray(r.correction).any()

    def _peel_single_faults():
        H, pr, _ = _setup()
        dec = PeelingDecoder(H, pr)
        rng = np.random.default_rng(3)
        for j in rng.choice(H.shape[1], size=50, replace=False):
            s = H[:, [j]].toarray().ravel().astype(np.uint8)
            r = dec.decode(s)
            assert r.converged and _consistent(H, s, r), f"single fault {j} not decoded"

    def _peel_honest():
        H, pr, det = _setup()
        dec = PeelingDecoder(H, pr)
        for s in det:
            r = dec.decode(s)
            assert r.converged == _consistent(H, s, r), "converged flag disagrees with H·e = s"

    def _osd_consistent():
        H, pr, det = _setup()
        dec = BpOsdDecoder(H, pr)
        for s in det:
            r = dec.decode(s)
            assert _consistent(H, s, r), "BP+OSD correction does not reproduce the syndrome"
            assert r.converged

    tests_decoders = run_checks([
        ("peeling: zero syndrome gives zero correction", _peel_zero),
        ("peeling: decodes 50 random single faults exactly", _peel_single_faults),
        ("peeling: converged flag is honest on sampled shots", _peel_honest),
        ("BP+OSD: correction always reproduces the syndrome", _osd_consistent),
    ])
    render_checks("decoders", tests_decoders)
    return (tests_decoders,)


# =============================================================================
# 7. Switch policies
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Switch policies

    | trigger | escalate to secondary when |
    |---|---|
    | `never` | never (primary only) |
    | `always` | always (secondary only) |
    | `primary_fail` | the primary does not converge — the trivial baseline |
    | `flag` | a flag fired (**before** the primary runs), else on primary failure |
    | `flag_or_fail` | same as `flag`; kept separate so ablations can differ |

    **Invariant:** no policy can have a lower logical error rate than `always`
    — the secondary sees the same syndrome. Beating it means a bug or an unfair
    baseline, never a result.
    """)
    return


@app.cell
def _(DecodeResult, np):
    TRIGGERS = ("never", "always", "primary_fail", "flag", "flag_or_fail")

    class SwitchPolicy:
        """
        Contract:
          - unknown trigger -> ValueError; flag triggers without a flag mask -> ValueError
          - `always` never calls the primary; `never` never calls the secondary
          - for flag triggers, a fired flag escalates WITHOUT calling the primary
          - the returned DecodeResult has escalated=True iff the secondary ran
        """

        def __init__(self, primary, secondary, trigger, flag_mask=None, name=None):
            # TODO
            raise NotImplementedError("SwitchPolicy.__init__")

        def decode(self, syndrome):
            # TODO
            raise NotImplementedError("SwitchPolicy.decode")

    return SwitchPolicy, TRIGGERS


@app.cell
def _(DecodeResult, SwitchPolicy, np, render_checks, run_checks):
    class _Mock:
        """Stub decoder: fixed convergence, counts its calls."""
        def __init__(self, converged):
            self.ok, self.calls = converged, 0

        def decode(self, s):
            self.calls += 1
            return DecodeResult(np.zeros(4, np.uint8), self.ok)

    _mask = np.array([False, False, True, True])
    _quiet = np.array([1, 0, 0, 0], np.uint8)     # no flag fired
    _flagged = np.array([1, 0, 1, 0], np.uint8)   # flag fired

    def _run(trigger, primary_ok, syndrome, mask=_mask):
        p, s = _Mock(primary_ok), _Mock(True)
        r = SwitchPolicy(p, s, trigger, mask).decode(syndrome)
        return r, p.calls, s.calls

    def _never():
        r, pc, sc = _run("never", False, _quiet)
        assert (pc, sc, r.escalated) == (1, 0, False)

    def _always():
        r, pc, sc = _run("always", True, _quiet)
        assert (pc, sc, r.escalated) == (0, 1, True)

    def _primary_fail():
        r, pc, sc = _run("primary_fail", True, _flagged)
        assert (pc, sc, r.escalated) == (1, 0, False), "flags must be ignored here"
        r, pc, sc = _run("primary_fail", False, _quiet)
        assert (pc, sc, r.escalated) == (1, 1, True)

    def _flag_skips_primary():
        for trig in ("flag", "flag_or_fail"):
            r, pc, sc = _run(trig, True, _flagged)
            assert (pc, sc, r.escalated) == (0, 1, True), f"{trig}: primary ran on a flagged shot"

    def _flag_fallback():
        for trig in ("flag", "flag_or_fail"):
            r, pc, sc = _run(trig, True, _quiet)
            assert (pc, sc, r.escalated) == (1, 0, False)
            r, pc, sc = _run(trig, False, _quiet)
            assert (pc, sc, r.escalated) == (1, 1, True)

    def _validation():
        for bad in (dict(trigger="sometimes", flag_mask=_mask),
                    dict(trigger="flag", flag_mask=None),
                    dict(trigger="flag", flag_mask=np.zeros(4, bool))):
            try:
                SwitchPolicy(_Mock(True), _Mock(True), **bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted invalid config {bad}")

    tests_switch = run_checks([
        ("never: secondary is never called", _never),
        ("always: primary is never called", _always),
        ("primary_fail: escalates only on primary failure, ignores flags", _primary_fail),
        ("flag triggers: a fired flag skips the primary pass", _flag_skips_primary),
        ("flag triggers: unflagged shots fall back to primary_fail", _flag_fallback),
        ("invalid trigger or missing flag mask raises ValueError", _validation),
    ])
    render_checks("switch policies", tests_switch)
    return (tests_switch,)


# =============================================================================
# 8. Statistics and benchmarking
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Statistics and benchmarking

    - **Logical error** = predicted observable flips $L\hat e$ differ from the
      true flips sampled by Stim, on *any* of the $k$ observables.
    - **Confidence intervals:** Wilson score interval (valid at zero failures).
    - **Paired comparison:** every policy decodes the *same* sampled shots.
    """)
    return


@app.cell
def _(dataclass):
    @dataclass
    class Metrics:
        shots: int
        failures: int
        ler: float
        ci_low: float
        ci_high: float
        escalation_rate: float
        work_mean: float
        work_p99: float
        time_p50_us: float
        time_p99_us: float

    return (Metrics,)


@app.cell
def _(Metrics, np, time):
    def wilson(k, n, z=1.96):
        """Wilson score interval for k successes in n trials -> (p_hat, low, high)."""
        # TODO
        raise NotImplementedError("wilson")

    def run_benchmark(circuit, policies, L, shots, seed):
        """
        Sample `shots` from `circuit` ONCE, decode them with every policy.

        Args:
            policies: dict name -> object with decode(syndrome) -> DecodeResult
            L:        observable matrix from dem_to_matrices
        Returns:
            dict name -> Metrics
        """
        # TODO
        raise NotImplementedError("run_benchmark")

    return run_benchmark, wilson


@app.cell
def _(
    BB_PRESETS,
    DecodeResult,
    bb_from_preset,
    build_memory_circuit,
    dem_to_matrices,
    np,
    render_checks,
    run_benchmark,
    run_checks,
    wilson,
):
    def _wilson_zero():
        ph, lo, hi = wilson(0, 100)
        assert ph == 0 and lo == 0
        assert abs(hi - 1.96**2 / (100 + 1.96**2)) < 1e-9, f"upper bound {hi}"

    def _wilson_symmetric():
        ph, lo, hi = wilson(50, 100)
        assert abs(ph - 0.5) < 1e-12 and abs((0.5 - lo) - (hi - 0.5)) < 1e-12

    def _wilson_contains():
        for k, n in [(1, 10), (3, 1000), (999, 1000), (1000, 1000)]:
            ph, lo, hi = wilson(k, n)
            assert 0 <= lo <= ph <= hi <= 1, f"({k}, {n}) -> {lo}, {ph}, {hi}"

    class _ZeroDecoder:
        def __init__(self, nf):
            self.nf = nf

        def decode(self, s):
            return DecodeResult(np.zeros(self.nf, np.uint8), True)

    def _setup():
        c = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        circ = build_memory_circuit(c, 3, 3e-3)
        H, L, _ = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        return circ, H, L

    def _zero_decoder_ler():
        circ, H, L = _setup()
        res = run_benchmark(circ, {"zero": _ZeroDecoder(H.shape[1])}, L, shots=200, seed=11)
        _, obs = circ.compile_detector_sampler(seed=11).sample(200, separate_observables=True)
        expected = int(obs.any(axis=1).sum())
        assert res["zero"].failures == expected, \
            f"{res['zero'].failures} failures, expected {expected} (check seeding/scoring)"

    def _paired():
        circ, H, L = _setup()
        res = run_benchmark(circ, {"a": _ZeroDecoder(H.shape[1]),
                                   "b": _ZeroDecoder(H.shape[1])}, L, shots=100, seed=4)
        assert res["a"].failures == res["b"].failures, "policies did not see the same shots"

    tests_stats = run_checks([
        ("wilson(0, n): lower bound 0, upper z²/(n+z²)", _wilson_zero),
        ("wilson is symmetric at p̂ = 1/2", _wilson_symmetric),
        ("wilson interval brackets p̂ and stays in [0, 1]", _wilson_contains),
        ("benchmark: zero decoder fails exactly when an observable flips", _zero_decoder_ler),
        ("benchmark: policies are scored on identical shots", _paired),
    ])
    render_checks("statistics", tests_stats)
    return (tests_stats,)


# =============================================================================
# 9. Reproducibility
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Reproducibility

    Every result is saved together with the exact configuration that produced it,
    so any figure in the thesis can be regenerated from its JSON file.
    """)
    return


@app.cell
def _(dataclass):
    @dataclass(frozen=True)
    class ExperimentConfig:
        code: str = "[[72, 12, 6]]"
        rounds: int = 6
        p: float = 1e-3
        shots: int = 1000
        seed: int = 20260921
        use_flags: bool = True
        osd_order: int = 0

    return (ExperimentConfig,)


@app.cell
def _(ExperimentConfig, Metrics, asdict, json):
    def save_results(path, config, results):
        """Write {'config': ..., 'results': {policy: Metrics}} as JSON."""
        # TODO
        raise NotImplementedError("save_results")

    def load_results(path):
        """Inverse of save_results -> (ExperimentConfig, dict name -> Metrics)."""
        # TODO
        raise NotImplementedError("load_results")

    return load_results, save_results


@app.cell
def _(
    ExperimentConfig,
    Metrics,
    load_results,
    render_checks,
    run_checks,
    save_results,
):
    def _roundtrip():
        import os
        import tempfile
        cfg = ExperimentConfig(p=2e-3, shots=10)
        res = {"always": Metrics(10, 1, 0.1, 0.02, 0.4, 1.0, 5.0, 9.0, 100.0, 200.0)}
        path = os.path.join(tempfile.mkdtemp(), "r.json")
        save_results(path, cfg, res)
        cfg2, res2 = load_results(path)
        assert cfg2 == cfg and res2 == res

    tests_repro = run_checks([
        ("save_results / load_results round-trip exactly", _roundtrip),
    ])
    render_checks("reproducibility", tests_repro)
    return (tests_repro,)


# =============================================================================
# 10. Status dashboard
# =============================================================================
@app.cell(hide_code=True)
def _(
    STATUS_ICON,
    all_pass,
    mo,
    tests_circuit,
    tests_codes,
    tests_decoders,
    tests_dem,
    tests_gf2,
    tests_repro,
    tests_stats,
    tests_switch,
):
    _sections = [
        ("2. GF(2)", tests_gf2), ("3. Codes", tests_codes),
        ("4. Circuit", tests_circuit), ("5. DEM", tests_dem),
        ("6. Decoders", tests_decoders), ("7. Switch policies", tests_switch),
        ("8. Statistics", tests_stats), ("9. Reproducibility", tests_repro),
    ]
    _rows = ["| section | " + " | ".join(STATUS_ICON.values()) + " |",
             "|:--|--:|--:|--:|"]
    for _name, _res in _sections:
        _c = [sum(r["status"] == s for r in _res) for s in STATUS_ICON]
        _rows.append(f"| {_name} | " + " | ".join(map(str, _c)) + " |")

    core_ready = all_pass(*[r for _, r in _sections])
    mo.vstack([
        mo.md("## 10. Implementation status"),
        mo.md("\n".join(_rows)),
        mo.md("**All tests pass — experiments unlocked.**" if core_ready else
              "Experiments stay locked until every test above passes."),
    ])
    return (core_ready,)


# =============================================================================
# 11. Experiments
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Experiments

    **E1 — Ablation** at fixed $p$: compare all five triggers on the same shots.
    The thesis claim lives in the gap between `primary_fail` and `flag`.

    **E2 — Noise sweep:** LER and escalation rate vs $p$ (log-spaced).

    **E3 — TODO:** your own, e.g. code size, rounds $T$, OSD order.
    """)
    return


@app.cell
def _(BB_PRESETS, mo):
    ui_code = mo.ui.dropdown(list(BB_PRESETS), value="[[72, 12, 6]]", label="Code")
    ui_rounds = mo.ui.slider(1, 12, value=6, label="Rounds T")
    ui_p = mo.ui.dropdown(["5e-4", "1e-3", "2e-3", "3e-3", "5e-3"], value="1e-3", label="p")
    ui_shots = mo.ui.slider(100, 5000, step=100, value=500, label="Shots")
    ui_flags = mo.ui.switch(value=True, label="Flag qubits")
    ui_osd = mo.ui.slider(0, 4, value=0, label="OSD order")
    ui_seed = mo.ui.number(value=20260921, label="Seed")
    mo.vstack([mo.md("### Experiment configuration"),
               mo.hstack([ui_code, ui_p, ui_flags]),
               mo.hstack([ui_rounds, ui_shots]),
               mo.hstack([ui_osd, ui_seed])])
    return ui_code, ui_flags, ui_osd, ui_p, ui_rounds, ui_seed, ui_shots


@app.cell
def _(
    ExperimentConfig,
    ui_code,
    ui_flags,
    ui_osd,
    ui_p,
    ui_rounds,
    ui_seed,
    ui_shots,
):
    config = ExperimentConfig(
        code=ui_code.value, rounds=int(ui_rounds.value), p=float(ui_p.value),
        shots=int(ui_shots.value), seed=int(ui_seed.value),
        use_flags=bool(ui_flags.value), osd_order=int(ui_osd.value))
    return (config,)


@app.cell
def _(
    BB_PRESETS,
    BpOsdDecoder,
    PeelingDecoder,
    SwitchPolicy,
    TRIGGERS,
    bb_from_preset,
    build_memory_circuit,
    dem_to_matrices,
    flag_detector_mask,
    run_benchmark,
):
    def run_ablation(config):
        """
        E1. Build the code, circuit and DEM from `config`; construct one
        SwitchPolicy per trigger over a shared PeelingDecoder / BpOsdDecoder pair
        (skip flag triggers when config.use_flags is False); run_benchmark.

        Returns dict trigger -> Metrics.
        """
        # TODO
        raise NotImplementedError("run_ablation")

    def run_sweep(config, ps):
        """E2. run_ablation at each p in `ps`. Returns dict p -> (dict trigger -> Metrics)."""
        # TODO
        raise NotImplementedError("run_sweep")

    return run_ablation, run_sweep


@app.cell
def _(np, plt):
    def plot_ablation(results):
        """Bar chart of LER with Wilson error bars, one bar per trigger. Returns a Figure."""
        # TODO
        raise NotImplementedError("plot_ablation")

    def plot_sweep(sweep):
        """
        Two panels: (a) LER vs p, log-log, Wilson error bars; (b) escalation rate vs p.
        Zero-failure points should be drawn at their upper bound, not dropped.
        Returns a Figure.
        """
        # TODO
        raise NotImplementedError("plot_sweep")

    return plot_ablation, plot_sweep


@app.cell
def _(mo):
    run_e1 = mo.ui.run_button(label="Run E1 — ablation")
    run_e2 = mo.ui.run_button(label="Run E2 — sweep (slow)")
    mo.hstack([run_e1, run_e2])
    return run_e1, run_e2


@app.cell
def _(config, core_ready, mo, plot_ablation, run_ablation, run_e1):
    mo.stop(not core_ready, mo.md("*E1 locked: finish the implementation (see §10).*"))
    mo.stop(not run_e1.value, mo.md("*Press **Run E1** to start.*"))

    e1_results = run_ablation(config)

    _always = e1_results.get("always")
    _suspicious = [t for t, m in e1_results.items()
                   if _always is not None and t != "always" and m.ci_high < _always.ci_low]
    _table = "\n".join(
        ["| trigger | LER | 95% CI | escalated | work (mean) | p99 time |",
         "|:--|--:|:--|--:|--:|--:|"]
        + [f"| `{t}` | {m.ler:.4f} | [{m.ci_low:.4f}, {m.ci_high:.4f}] | "
           f"{m.escalation_rate:.1%} | {m.work_mean:.1f} | {m.time_p99_us:.0f} µs |"
           for t, m in e1_results.items()])
    mo.vstack([
        mo.md(f"### E1 — {config.code}, T={config.rounds}, p={config.p}, {config.shots} shots"),
        mo.md(_table),
        mo.md("> ⚠️ " + ", ".join(_suspicious) + " beat `always` — suspect a bug, "
              "not a result.") if _suspicious else mo.md(""),
        plot_ablation(e1_results),
    ])
    return (e1_results,)


@app.cell
def _(config, core_ready, mo, np, plot_sweep, run_e2, run_sweep):
    mo.stop(not core_ready, mo.md("*E2 locked: finish the implementation (see §10).*"))
    mo.stop(not run_e2.value, mo.md("*Press **Run E2** to start.*"))

    e2_ps = np.logspace(np.log10(5e-4), np.log10(6e-3), 6)
    e2_sweep = run_sweep(config, e2_ps)
    mo.vstack([mo.md(f"### E2 — {config.code}, T={config.rounds}, {config.shots} shots/point"),
               plot_sweep(e2_sweep)])
    return (e2_sweep,)


# =============================================================================
# 12. Results, discussion, conclusion
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Results

    **TODO:** state each finding in one sentence, pointing at the figure/table
    that supports it. Include the numbers and their confidence intervals.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Discussion and limitations

    **TODO.** Prompts:

    - Did the flag trigger beat `primary_fail`? By how much, and is it significant?
    - At what $p$ does escalation approach 100% and the latency argument fail?
    - Cost of flags: extra qubits, extra CNOTs, extra noise.
    - Schedule used vs. Bravyi et al.'s depth-8 schedule.
    - Python timings vs. real decoder hardware.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion and future work

    **TODO.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    1. S. Bravyi et al., "High-threshold and low-overhead fault-tolerant quantum
       memory," *Nature* **627**, 778 (2024).
    2. C. Gidney, "Stim: a fast stabilizer circuit simulator," *Quantum* **5**, 497 (2021).
    3. J. Roffe et al., "Decoding across the quantum low-density parity-check code
       landscape," *Phys. Rev. Research* **2**, 043423 (2020).
    4. R. Chao and B. Reichardt, "Quantum error correction with only two extra
       qubits," *Phys. Rev. Lett.* **121**, 050502 (2018).
    5. Pakhunov (2026), "Analytical Theory of Greedy Peeling for Bivariate Bicycle
       Codes and Two-Shot Streaming Decoding." **TODO:** complete citation.
    6. Sahay et al. (2026), "A matching decoder for bivariate bicycle codes."
       **TODO:** complete citation.
    7. **TODO:** the decoder-switching reference your proposal builds on.
    """)
    return


if __name__ == "__main__":
    app.run()
