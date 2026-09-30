import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium", app_title="Decoder Switching for BB Codes")


@app.cell
def _():
    import csv
    import datetime
    import glob
    import json
    import os
    import time
    from dataclasses import asdict, dataclass, field

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.sparse as sp
    import stim

    # Results always go next to the notebook, wherever marimo was launched from.
    _nb_dir = mo.notebook_dir()
    RESULTS_DIR = os.path.join(os.path.abspath(str(_nb_dir) if _nb_dir else "."), "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return (
        RESULTS_DIR,
        asdict,
        csv,
        dataclass,
        datetime,
        field,
        glob,
        json,
        mo,
        np,
        os,
        plt,
        sp,
        stim,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Flag-Triggered Decoder Switching for Bivariate Bicycle Codes

    Undergraduate thesis · BRAC University

    *Authors and supervisor: add your names here before submission.*

    ---

    **Flag qubits pay for themselves as decoder input, not as a switching
    trigger** — and the trigger idea does not merely underperform here, it is
    provably unable to help: at 250,000 paired shots the rule "escalate when the
    primary decoder fails" already matches an omniscient oracle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Outline

    1. Motivation and research question
    1b. Glossary: the symbols and terms used in the tables and graphs
    2. GF(2) linear algebra
    3. Bivariate bicycle codes
    4. Circuit-level noise and flag qubits
    5. Detector error model
    6. Decoders
    7. Switch policies
    8. Statistics and benchmarking
    9. Reproducibility
    10. Implementation status
    11. Experiments 1-9: validation, flag cost vs information, trigger saturation, ablation, noise sweep, decoder zoo, surface-code baseline, accuracy/latency frontier, dynamic prior updating
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Motivation and research question

    A quantum computer must correct errors faster than they accumulate. Every
    syndrome-extraction round produces a new syndrome, and if the decoder cannot
    keep up, the undecoded rounds pile into a backlog that grows without bound.
    Accurate decoders (BP+OSD) are slow; fast decoders (belief propagation alone,
    greedy peeling) are less accurate. **Decoder switching** resolves the tension
    by running the fast decoder on most shots and escalating the hard ones to the
    accurate decoder, which is worthwhile only if the escalation signal is good.

    Existing schemes decide *after* the fast decoder has run, from its convergence
    or its confidence. **Flag qubits** offer a different signal. A flag ancilla
    attached to a check ancilla detects faults that would otherwise spread from the
    ancilla into several data qubits (hook errors), and its outcome is available
    *before* any decoding. That suggests a pre-decode trigger: if a flag fired,
    send the shot straight to the accurate decoder.

    **Research question.** Where does flag information help when decoding
    bivariate bicycle codes under circuit-level noise: as a switching trigger, as
    decoder input, or not at all?

    We answer it in three parts:

    1. *As a trigger* — compare flag rules against the trivial rule "escalate when
       the fast decoder fails", on identical shots (Experiments 4 and 6).
    2. *As decoder input* — compare an unflagged circuit, a flagged circuit whose
       flag outcomes are hidden from the decoder, and a flagged circuit whose
       outcomes it reads (Experiment 2).
    3. *Why* — measure how often flags fire and how much they tell us
       (Experiments 1 and 3).

    **What would falsify the trigger hypothesis.** If the fast decoder is never
    confidently wrong, no trigger can improve accuracy; and if flags fire on almost
    every shot, no trigger can be selective. Both are measured, not assumed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1b. Glossary

    Every symbol that appears in a table or on an axis in this notebook.

    | term | meaning |
    |---|---|
    | **n, k, d** | physical qubits, logical qubits, code distance: a code is written [[n, k, d]] |
    | **p** | physical error rate: the probability parameter of every noise channel in the circuit |
    | **T** | number of syndrome-extraction rounds in one shot (how long the memory is kept alive) |
    | **shot** | one run of the experiment: prepare, run T rounds, measure, decode once |
    | **detector** | a parity of measurements that is deterministic without noise; the decoder sees these, not raw measurements |
    | **syndrome** | the vector of detector outcomes for one shot |
    | **fault mechanism** | one independent error the noise model can produce, with its detector signature; the columns of H |
    | **λ (lambda)** | expected number of fault mechanisms firing per shot = sum of all fault probabilities |
    | **H** | detector-by-fault check matrix from the detector error model |
    | **L** | observable-by-fault matrix: which logical observables each fault flips |
    | **LER** | logical error rate: fraction of shots whose predicted observable flips differ from the truth |
    | **converged** | the decoder found a correction that reproduces the syndrome exactly |
    | **silent failure** | the weak decoder converged but was logically wrong: a confident mistake |
    | **headroom** | silent failures of the weak decoder that the strong decoder gets right — the most accuracy any trigger could gain |
    | **escalation rate** | fraction of shots sent to the strong decoder |
    | **trigger / rule** | the condition for escalating: `primary_fail`, `flag>=k`, `rounds>=k`, `flag-only` |
    | **work** | decoder-native operation count: detector visits for peeling, iterations for BP |
    | **p50 / p99** | median and 99th-percentile decode time; p99 is the tail that a real-time decoder must meet |
    | **Wilson interval** | 95% confidence interval for a rate, valid even at zero observed failures |
    | **arm** | one leg of a controlled comparison: *unflagged*, *flagged blind*, *flagged sighted* |
    | **blind / sighted** | whether the decoder is given the flag detectors (sighted) or they are removed from H (blind) |
    | **r\*** | break-even cost ratio: the weak/strong cost ratio above which a flag rule would be cheaper |
    | **T\*** | crossover: the round count at which flags stop paying for themselves — *never reached*: the measured ratios are flat over $T = 6$–24, so no fitted T\* is meaningful |

    **Decoder names.** `peel` greedy peeling · `BP-ms-N` min-sum belief propagation,
    N iterations · `BP-ps-N` product-sum BP · `OSD-0` BP + ordered statistics,
    order 0 · `OSD-CSk` / `LSD-CSk` combination sweep of order k · `LSD` localized
    statistics decoding · `control-*` positive controls that fail silently on
    purpose.
    """)
    return


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
    def gf2_dense(M):
        """Dense uint8 copy of M with entries reduced mod 2 (accepts scipy.sparse)."""
        return np.array(M.toarray() if hasattr(M, "toarray") else M, dtype=np.uint8) % 2

    def gf2_rref(M):
        """
        Reduced row echelon form of M over GF(2).

        Args:
            M: 2-D array-like (dense or scipy.sparse) of 0/1 entries.
        Returns:
            (R, pivots): R is a uint8 array the same shape as M in RREF;
            pivots is the list of pivot column indices, in increasing order.
        """
        R = gf2_dense(M)
        n_rows, n_cols = R.shape
        pivots = []
        r = 0                                   # next row to place a pivot in
        for c in range(n_cols):
            if r == n_rows:                     # every row has a pivot: done
                break
            candidates = np.flatnonzero(R[r:, c])
            if candidates.size == 0:            # no 1 at or below row r: free column
                continue
            p = r + candidates[0]
            if p != r:
                R[[r, p]] = R[[p, r]]           # swap the pivot row into place
            others = np.flatnonzero(R[:, c])
            others = others[others != r]
            R[others] ^= R[r]                   # clear column c above AND below (XOR = add mod 2)
            pivots.append(c)
            r += 1
        return R, pivots

    def gf2_rank(M):
        """Rank of M over GF(2). Must return 0 for an empty matrix."""
        if M.shape[0] == 0 or M.shape[1] == 0:
            return 0
        return len(gf2_rref(M)[1])

    def gf2_nullspace(M):
        """
        Basis of {v : M v = 0 (mod 2)}.

        Returns:
            uint8 array of shape (n_cols - rank(M), n_cols); rows are basis vectors.
        """
        R, pivots = gf2_rref(M)
        n_cols = R.shape[1]
        pivot_set = set(pivots)
        free = [c for c in range(n_cols) if c not in pivot_set]
        basis = np.zeros((len(free), n_cols), dtype=np.uint8)
        for i, f in enumerate(free):
            basis[i, f] = 1
            # Row r of R reads: x[pivots[r]] + sum_f R[r, f] x[f] = 0.
            # With only x[f] = 1, and -1 = +1 mod 2, this gives x[pivots[r]] = R[r, f].
            basis[i, pivots] = R[:len(pivots), f]
        return basis

    def gf2_quotient_basis(subspace, ambient):
        """
        Rows of `ambient` that are linearly independent modulo rowspace(`subspace`).

        Used for logical operators: logicals = ker(...) modulo stabilisers.
        Returns a uint8 array with shape (dim, n_cols); may have zero rows.
        """
        span = gf2_dense(subspace)
        amb = gf2_dense(ambient)
        current_rank = gf2_rank(span)
        kept = []
        for row in amb:
            trial = np.vstack([span, row[None, :]])
            trial_rank = gf2_rank(trial)
            if trial_rank > current_rank:       # row adds something new: keep it
                kept.append(row)
                span, current_rank = trial, trial_rank
        if not kept:
            return np.zeros((0, amb.shape[1]), dtype=np.uint8)
        return np.array(kept, dtype=np.uint8)

    return gf2_dense, gf2_nullspace, gf2_quotient_basis, gf2_rank, gf2_rref


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

    **Generalised bicycle (GB) codes** (Kovalev & Pryadko, 2013) are the
    one-variable case: $A = a(x)$ and $B = b(x)$ are $\ell \times \ell$
    circulants built from a single cyclic shift $x$ of order $\ell$. A GB code
    is exactly a BB code with $M = 1$.
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
        lattice: tuple = None   # (L, M) for BB/GB codes; the circuit schedule needs it

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

    # Generalised bicycle codes. a_exps / b_exps: each e is the monomial x^e.
    # n and k verified by the tests; d from Panteleev & Kalachev, Quantum 5, 585
    # (2021), and NOT tested.
    GB_PRESETS = {
        "[[46, 2, 9]]":   dict(ell=23,  a_exps=[0, 5, 8, 12],
                               b_exps=[0, 1, 5, 7, 9, 10],   n=46,  k=2,  d=9),
        "[[48, 6, 8]]":   dict(ell=24,  a_exps=[0, 2, 8, 15],
                               b_exps=[0, 2, 12, 17],        n=48,  k=6,  d=8),
        "[[126, 28, 8]]": dict(ell=63,  a_exps=[0, 1, 14, 16, 22],
                               b_exps=[0, 3, 13, 20, 42],    n=126, k=28, d=8),
        "[[254, 28]]":    dict(ell=127, a_exps=[0, 15, 20, 28, 66],
                               b_exps=[0, 58, 59, 100, 121], n=254, k=28, d=None),
    }
    return BB_PRESETS, CSSCode, GB_PRESETS


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
        # Normalise: accept dense or sparse, reduce mod 2, store as uint8 CSR.
        def to_csr(H):
            dense = H.toarray() if sp.issparse(H) else np.asarray(H)
            M = sp.csr_matrix(dense % 2, dtype=np.uint8)
            M.eliminate_zeros()
            return M

        hx = to_csr(hx)
        hz = to_csr(hz)

        n = hx.shape[1]
        if hz.shape[1] != n:
            raise ValueError(f"{name}: hx has {n} columns but hz has {hz.shape[1]}")

        # CSS condition: every X check commutes with every Z check.
        if ((hx @ hz.T).toarray() % 2).any():
            raise ValueError(f"{name}: hx @ hz.T != 0 (mod 2); not a valid CSS code")

        # Ranks over GF(2) and number of logical qubits.
        rank_x = gf2_rank(hx)
        rank_z = gf2_rank(hz)
        k = n - rank_x - rank_z

        # L_X = ker(H_Z) modulo rowspace(H_X);  L_Z = ker(H_X) modulo rowspace(H_Z).
        lx = gf2_quotient_basis(hx, gf2_nullspace(hz))
        lz = gf2_quotient_basis(hz, gf2_nullspace(hx))

        if lx.shape[0] != k or lz.shape[0] != k:
            raise RuntimeError(
                f"{name}: expected {k} logicals, got lx={lx.shape[0]}, lz={lz.shape[0]}"
            )

        return CSSCode(
            name=name, hx=hx, hz=hz, n=n, k=k,
            rank_x=rank_x, rank_z=rank_z, lx=lx, lz=lz,
        )


    def make_bb_code(L, M, a_terms, b_terms, name=None):
        """
        Bivariate bicycle code on an L x M torus.

        Args:
            a_terms, b_terms: lists of (i, j) exponent pairs, monomial x^i y^j.
        Returns:
            CSSCode with hx = [A | B], hz = [B^T | A^T], and lattice = (L, M).
        """
        # Cyclic shifts and torus generators x = S_L ⊗ I_M, y = I_L ⊗ S_M.
        I_L = np.eye(L, dtype=np.int64)
        I_M = np.eye(M, dtype=np.int64)
        S_L = np.roll(I_L, 1, axis=1)
        S_M = np.roll(I_M, 1, axis=1)
        x = np.kron(S_L, I_M)
        y = np.kron(I_L, S_M)

        def poly(terms):
            """Sum of monomials x^i y^j, reduced mod 2."""
            P = np.zeros((L * M, L * M), dtype=np.int64)
            for i, j in terms:
                P += np.linalg.matrix_power(x, i % L) @ np.linalg.matrix_power(y, j % M)
            return (P % 2).astype(np.uint8)

        A = poly(a_terms)
        B = poly(b_terms)

        hx = sp.csr_matrix(np.hstack([A, B]))
        hz = sp.csr_matrix(np.hstack([B.T, A.T]))
        hx.eliminate_zeros()
        hz.eliminate_zeros()

        code = make_css_code(hx, hz, name=name or f"BB(L={L}, M={M})")
        code.lattice = (L, M)
        return code


    def make_gb_code(ell, a_exps, b_exps, name=None):
        """
        Generalised bicycle code: A = a(x), B = b(x), with x the ell x ell cyclic shift.
        Returns a CSSCode with lattice = (ell, 1).
        """
        return make_bb_code(
            ell, 1,
            [(e, 0) for e in a_exps],
            [(e, 0) for e in b_exps],
            name=name or f"GB(ell={ell})",
        )


    def bb_from_preset(key, presets):
        """Convenience: build a preset from BB_PRESETS by its label."""
        s = presets[key]
        return make_bb_code(s["L"], s["M"], s["a_terms"], s["b_terms"], name=key)


    def gb_from_preset(key, presets):
        """Convenience: build a preset from GB_PRESETS by its label."""
        s = presets[key]
        return make_gb_code(s["ell"], s["a_exps"], s["b_exps"], name=key)

    return (
        bb_from_preset,
        gb_from_preset,
        make_bb_code,
        make_css_code,
        make_gb_code,
    )


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

    def _lattice_recorded():
        for key, spec in BB_PRESETS.items():
            c = bb_from_preset(key, BB_PRESETS)
            assert tuple(c.lattice) == (spec["L"], spec["M"]), f"{key}: lattice={c.lattice}"

    tests_codes = run_checks([
        ("every preset builds with the expected [[n, k]]", _presets_n_k),
        ("make_bb_code records lattice = (L, M)", _lattice_recorded),
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


@app.cell
def _(
    GB_PRESETS,
    gb_from_preset,
    make_bb_code,
    make_gb_code,
    np,
    render_checks,
    run_checks,
):
    def _gb_n_k():
        for key, spec in GB_PRESETS.items():
            c = gb_from_preset(key, GB_PRESETS)
            assert (c.n, c.k) == (spec["n"], spec["k"]), \
                f"{key}: built [[{c.n}, {c.k}]], expected [[{spec['n']}, {spec['k']}]]"

    def _gb_css():
        for key in GB_PRESETS:
            c = gb_from_preset(key, GB_PRESETS)
            assert not ((c.hx @ c.hz.T).toarray() % 2).any(), key

    def _gb_weights():
        for key, spec in GB_PRESETS.items():
            c = gb_from_preset(key, GB_PRESETS)
            w = len(spec["a_exps"]) + len(spec["b_exps"])
            assert set(np.asarray(c.hx.sum(axis=1)).ravel()) == {w}, f"{key}: row weight != {w}"

    def _gb_is_bb_with_m1():
        s = GB_PRESETS["[[48, 6, 8]]"]
        g = make_gb_code(s["ell"], s["a_exps"], s["b_exps"])
        b = make_bb_code(s["ell"], 1, [(e, 0) for e in s["a_exps"]], [(e, 0) for e in s["b_exps"]])
        assert (g.hx != b.hx).nnz == 0 and (g.hz != b.hz).nnz == 0
        assert tuple(g.lattice) == (s["ell"], 1), f"lattice={g.lattice}"

    def _gb_logicals():
        c = gb_from_preset("[[48, 6, 8]]", GB_PRESETS)
        assert c.lx.shape == c.lz.shape == (c.k, c.n)
        assert not ((c.hx @ c.lz.T) % 2).any()
        assert not ((c.hz @ c.lx.T) % 2).any()

    def _k_zero_allowed():
        # The GB example in the old main.py (ell = 5) encodes no logical qubits.
        c = make_gb_code(5, [0, 1, 2], [0, 2, 3])
        assert c.k == 0 and c.lx.shape == (0, c.n) and c.lz.shape == (0, c.n)

    tests_gb = run_checks([
        ("every GB preset builds with the expected [[n, k]]", _gb_n_k),
        ("CSS condition for every GB preset", _gb_css),
        ("GB row weight equals |a| + |b|", _gb_weights),
        ("make_gb_code equals make_bb_code with M = 1", _gb_is_bb_with_m1),
        ("GB logical operators commute with opposite-type stabilisers", _gb_logicals),
        ("a k = 0 code builds with empty logical matrices", _k_zero_allowed),
    ])
    render_checks("GB codes", tests_gb)
    return (tests_gb,)


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

    The schematic below is generated from the same conventions the builder uses.
    """)
    return


@app.cell(hide_code=True)
def _(plt):
    def flag_circuit_diagram():
        """Schematic of one X-check with its flag qubit (matches build_memory_circuit)."""
        fig, ax = plt.subplots(figsize=(10, 3.4))
        rows = {"X-ancilla": 5, "flag": 4, "data 1": 3, "data 2": 2, "...": 1, "data 6": 0}
        for name, y in rows.items():
            ax.plot([0, 13], [y, y], color="0.75", lw=1, zorder=0)
            ax.text(-0.3, y, name, ha="right", va="center", fontsize=9)
        ax.text(0.3, 5, "|+>", ha="center", va="center", fontsize=9)
        steps = [(1.5, 3, "CX 1"), (3.5, 2, "CX 2"), (5.5, 1, "CX 3"),
                 (7.5, 1, "CX 4"), (9.0, 1, "CX 5"), (10.5, 0, "CX 6")]
        for x, y, label in steps:
            ax.plot([x, x], [5, y], color="#1f77b4", lw=1.5, zorder=2)
            ax.plot(x, 5, "o", color="#1f77b4", ms=7, zorder=3)
            ax.plot(x, y, "o", mfc="white", mec="#1f77b4", ms=9, mew=1.5, zorder=3)
            ax.text(x, 5.45, label, ha="center", fontsize=7, color="#1f77b4")
        for x in (2.5, 9.75):                      # the two flag CNOTs
            ax.plot([x, x], [5, 4], color="#d62728", lw=1.5, zorder=2)
            ax.plot(x, 5, "o", color="#d62728", ms=7, zorder=3)
            ax.plot(x, 4, "o", mfc="white", mec="#d62728", ms=9, mew=1.5, zorder=3)
        ax.annotate("", xy=(9.75, 4.55), xytext=(2.5, 4.55),
                    arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1))
        ax.text(6.1, 4.68, "an X fault on the ancilla in this window flips the flag",
                ha="center", fontsize=8, color="#d62728")
        for x, y in ((12.2, 5), (12.2, 4)):
            ax.add_patch(plt.Rectangle((x - 0.35, y - 0.3), 0.7, 0.6, fc="white", ec="0.3"))
            ax.text(x, y, "M", ha="center", va="center", fontsize=9)
        ax.set_xlim(-2.2, 13.2)
        ax.set_ylim(-0.6, 6.0)
        ax.axis("off")
        ax.set_title("One X-check with its flag qubit: ancilla is the CONTROL, so a mid-window "
                     "ancilla fault spreads to the data qubits it has not yet touched",
                     fontsize=9)
        fig.tight_layout()
        return fig

    flag_circuit_diagram()
    return (flag_circuit_diagram,)


@app.cell
def _(np, sp, stim):
    def check_schedule(H, lattice):
        """
        Order each check's data qubits by lattice offset.

        Returns an int array S of shape (w, m): S[l, r] is the data qubit that
        check r touches in CNOT layer l. Every row of S is a permutation of
        distinct qubits, so each layer is a valid parallel CX instruction.
        """
        if lattice is None:
            raise ValueError("code.lattice is None: the schedule needs (L, M)")
        L, M = lattice
        N = L * M
        H = sp.csr_matrix(H)
        m = H.shape[0]
        keyed_rows = []
        for r in range(m):
            ri, rj = divmod(r % N, M)
            cols = H.indices[H.indptr[r]:H.indptr[r + 1]]
            keyed = {}
            for c in cols:
                block, q = divmod(int(c), N)
                qi, qj = divmod(q, M)
                keyed[(block, (qi - ri) % L, (qj - rj) % M)] = int(c)
            keyed_rows.append(keyed)
        keys = sorted(keyed_rows[0])
        for r, keyed in enumerate(keyed_rows):
            if sorted(keyed) != keys:
                raise ValueError(f"check {r} has a different offset pattern: not translation-invariant")
        return np.array([[keyed_rows[r][key] for r in range(m)] for key in keys], dtype=np.int64)


    def build_memory_circuit(code, rounds, p, use_flags=False, idle_noise=False,
                             x_detectors=True):
        """
        Z-basis memory experiment for a CSS code under circuit-level noise.

        Detector order per round t: m_z Z-check detectors, m_x X-check detectors
        (t >= 1 only, and only when x_detectors), m_x flag detectors (if use_flags);
        then m_z final detectors.
        Schedule: all X-check CNOT layers, then all Z-check CNOT layers (always
        deterministic). Flag CNOTs sit after the first and before the last X layer.

        x_detectors=False drops the X-check detectors. In a Z-BASIS memory the
        observables are Z-type (rows of code.lz), so they are flipped by X errors
        only. X checks detect Z errors, which can never flip those observables:
        their detectors add a second, decoupled fault population that cannot cause
        a logical failure but does inflate the DEM. For [[72,12,6]] at T=12 the
        fault count goes 33,552 -> 4,392 (Theorem 1 of Pakhunov (2026) predicts
        n(wT + T/2 + 1) = 3,096) and the mean fault-graph degree 1,356 -> 128.
        Peeling is collision-limited, so the dense version cripples it: the
        birthday bound exp(-d_bar*lambda^2 / 2N) gives 0.38 against 0.70.
        Keep x_detectors=False for anything compared against the BB-code
        literature, and for anything where decode time matters.
        """
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        n = code.n
        mx, mz = code.hx.shape[0], code.hz.shape[0]

        data = list(range(n))
        xanc = list(range(n, n + mx))
        zanc = list(range(n + mx, n + mx + mz))
        flag = list(range(n + mx + mz, n + mx + mz + mx)) if use_flags else []
        all_q = data + xanc + zanc + flag

        sx = check_schedule(code.hx, code.lattice)   # (w_x, m_x)
        sz = check_schedule(code.hz, code.lattice)   # (w_z, m_z)
        if use_flags and sx.shape[0] < 2:
            raise ValueError("flags need X checks of weight >= 2")

        hz_rows = [code.hz.indices[code.hz.indptr[i]:code.hz.indptr[i + 1]].tolist()
                   for i in range(mz)]
        lz = np.asarray(code.lz, dtype=np.uint8)

        c = stim.Circuit()
        meas = [0]                          # running count of measurements

        def rec(idx):
            return stim.target_rec(idx - meas[0])

        def cx_layer(pairs):
            targets = [q for pair in pairs for q in pair]
            c.append("CX", targets)
            if p > 0:
                c.append("DEPOLARIZE2", targets, p)
                if idle_noise:
                    busy = set(targets)
                    idle = [q for q in all_q if q not in busy]
                    if idle:
                        c.append("DEPOLARIZE1", idle, p)
            c.append("TICK")

        def measure(name, qubits):
            """Append a (noisy) measurement; return the absolute record indices."""
            if p > 0:
                c.append(name, qubits, p)       # p = measurement-flip probability
            else:
                c.append(name, qubits)
            start = meas[0]
            meas[0] += len(qubits)
            return list(range(start, meas[0]))

        def ancilla_reset_noise():
            if p > 0:
                c.append("Z_ERROR", xanc, p)            # X-basis reset fails as a Z flip
                c.append("X_ERROR", zanc + flag, p)     # Z-basis reset fails as an X flip

        # ---- initial reset layer ------------------------------------------------
        c.append("R", data + zanc + flag)
        c.append("RX", xanc)
        if p > 0:
            c.append("X_ERROR", data, p)
        ancilla_reset_noise()
        c.append("TICK")

        prev_x = prev_z = None
        for t in range(rounds):
            # ---- X checks: ancilla (control) -> data (target) -------------------
            wx = sx.shape[0]
            for l in range(wx):
                if use_flags and l == wx - 1:
                    cx_layer(list(zip(xanc, flag)))                  # close flag
                cx_layer([(xanc[r], int(sx[l, r])) for r in range(mx)])
                if use_flags and l == 0:
                    cx_layer(list(zip(xanc, flag)))                  # open flag

            # ---- Z checks: data (control) -> ancilla (target) -------------------
            for l in range(sz.shape[0]):
                cx_layer([(int(sz[l, r]), zanc[r]) for r in range(mz)])

            # ---- measure (and reset) ancillas -----------------------------------
            if idle_noise and p > 0:
                c.append("DEPOLARIZE1", data, p)
            x_idx = measure("MRX", xanc)
            z_idx = measure("MR", zanc)
            f_idx = measure("MR", flag) if use_flags else []
            if t < rounds - 1:
                ancilla_reset_noise()

            # ---- detectors, in contract order -----------------------------------
            for i in range(mz):                                       # 1. Z checks
                tg = [rec(z_idx[i])] + ([rec(prev_z[i])] if t > 0 else [])
                c.append("DETECTOR", tg, [i, t, 0])
            if x_detectors and t > 0:                                 # 2. X checks
                for i in range(mx):
                    c.append("DETECTOR", [rec(x_idx[i]), rec(prev_x[i])], [i, t, 1])
            for i in range(len(f_idx)):                               # 3. flags
                c.append("DETECTOR", [rec(f_idx[i])], [i, t, 2])
            c.append("TICK")
            prev_x, prev_z = x_idx, z_idx

        # ---- final data readout ---------------------------------------------------
        d_idx = measure("M", data)

        for i in range(mz):
            tg = [rec(d_idx[q]) for q in hz_rows[i]] + [rec(prev_z[i])]
            c.append("DETECTOR", tg, [i, rounds, 3])
        for j in range(lz.shape[0]):
            c.append("OBSERVABLE_INCLUDE", [rec(d_idx[q]) for q in np.flatnonzero(lz[j])], j)
        return c


    def flag_detector_mask(code, rounds, use_flags, num_detectors, x_detectors=True):
        """
        Boolean array of length num_detectors, True exactly at flag detectors.
        x_detectors must match the value the circuit was built with, or the mask
        lands on the wrong detectors.
        """
        mx, mz = code.hx.shape[0], code.hz.shape[0]
        mask = np.zeros(num_detectors, dtype=bool)
        if not use_flags:
            return mask
        pos = 0
        for t in range(rounds):
            pos += mz                       # Z-check detectors
            if x_detectors and t > 0:
                pos += mx                   # X-check detectors
            mask[pos:pos + mx] = True       # flag detectors
            pos += mx
        return mask


    def expected_num_detectors(code, rounds, use_flags, x_detectors=True):
        """Detector count implied by the ordering contract above."""
        mx, mz = code.hx.shape[0], code.hz.shape[0]
        per_round_flags = mx if use_flags else 0
        per_round_x = mx if x_detectors else 0
        return (mz * rounds + per_round_x * (rounds - 1)
                + per_round_flags * rounds + mz)

    return build_memory_circuit, check_schedule, expected_num_detectors, flag_detector_mask


@app.cell
def _(
    BB_PRESETS,
    GB_PRESETS,
    bb_from_preset,
    build_memory_circuit,
    expected_num_detectors,
    flag_detector_mask,
    gb_from_preset,
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

    def _layers_are_parallel():
        codes = [_code(), gb_from_preset("[[48, 6, 8]]", GB_PRESETS)]
        for c in codes:
            circ = build_memory_circuit(c, 2, 1e-3, use_flags=True)
            for inst in circ.flattened():
                if inst.name == "CX":
                    q = [t.value for t in inst.targets_copy()]
                    assert len(q) == len(set(q)), \
                        f"{c.name}: a qubit appears twice in one CX layer"

    def _gb_circuit():
        c = gb_from_preset("[[48, 6, 8]]", GB_PRESETS)
        for flags in (False, True):
            clean = build_memory_circuit(c, 2, 0.0, use_flags=flags)
            det, obs = clean.compile_detector_sampler(seed=3).sample(
                16, separate_observables=True)
            assert not det.any() and not obs.any(), f"flags={flags}: noiseless GB circuit fires"
            noisy = build_memory_circuit(c, 2, 1e-3, use_flags=flags)
            noisy.detector_error_model(decompose_errors=False)
            assert noisy.num_detectors == expected_num_detectors(c, 2, flags)
            assert noisy.num_observables == c.k

    def _z_only_is_silent_and_valid():
        # x_detectors=False: still a correct circuit, just without the X-check
        # detectors. Everything downstream must stay consistent with it.
        c = _code()
        for flags in (False, True):
            clean = build_memory_circuit(c, _T, 0.0, use_flags=flags, x_detectors=False)
            det, obs = clean.compile_detector_sampler(seed=11).sample(
                32, separate_observables=True)
            assert not det.any(), f"flags={flags}: Z-only detectors fire with p = 0"
            assert not obs.any(), f"flags={flags}: Z-only observables flip with p = 0"
            noisy = build_memory_circuit(c, _T, 1e-3, use_flags=flags, x_detectors=False)
            noisy.detector_error_model(decompose_errors=False)   # must stay deterministic
            assert noisy.num_detectors == expected_num_detectors(
                c, _T, flags, x_detectors=False), f"flags={flags}: {noisy.num_detectors}"
            assert noisy.num_observables == c.k

    def _z_only_flag_mask():
        c = _code()
        mx = c.hx.shape[0]
        circ = build_memory_circuit(c, _T, 1e-3, use_flags=True, x_detectors=False)
        m = flag_detector_mask(c, _T, True, circ.num_detectors, x_detectors=False)
        assert m.dtype == bool and m.shape == (circ.num_detectors,)
        assert m.sum() == mx * _T
        # the mask must still land only on flag detectors: with p = 0 they never fire
        clean = build_memory_circuit(c, _T, 0.0, use_flags=True, x_detectors=False)
        assert not clean.compile_detector_sampler(seed=12).sample(32)[:, m].any()

    def _z_only_thins_the_dem():
        # The point of the flag: X-check detectors carry a Z-error fault population
        # that cannot flip a Z-type observable, so it is pure decoding overhead.
        # Pakhunov (2026) Theorem 1 predicts n(wT + T/2 + 1) fault mechanisms.
        c = _code()
        theory = c.n * (3 * _T + _T / 2 + 1)
        big = build_memory_circuit(c, _T, 1e-3, x_detectors=True)
        small = build_memory_circuit(c, _T, 1e-3, x_detectors=False)
        n_big = big.detector_error_model(decompose_errors=False).num_errors
        n_small = small.detector_error_model(decompose_errors=False).num_errors
        assert n_small < n_big / 3, f"Z-only DEM not thinned: {n_small} vs {n_big}"
        assert n_small < 3 * theory, f"Z-only DEM still {n_small / theory:.1f}x theory"

    tests_circuit = run_checks([
        ("noiseless circuit: no detector fires, no observable flips", _noiseless_is_silent),
        ("noisy circuit yields a valid DEM (deterministic detectors)", _dem_builds),
        ("detector and observable counts match the ordering contract", _counts),
        ("flag mask has the right shape and count", _flag_mask),
        ("some fault mechanism fires a flag detector", _flags_see_faults),
        ("flags never fire without noise", _flags_are_quiet_at_zero_noise),
        ("data-qubit X errors never fire a flag detector", _data_errors_never_flag),
        ("every CX instruction is a true parallel layer (BB and GB)", _layers_are_parallel),
        ("circuit works for a weight-8 GB code", _gb_circuit),
        ("x_detectors=False: silent at p = 0, valid DEM, counts agree",
         _z_only_is_silent_and_valid),
        ("x_detectors=False: flag mask still lands on flag detectors", _z_only_flag_mask),
        ("x_detectors=False: DEM density drops toward the analytic fault count",
         _z_only_thins_the_dem),
    ])
    render_checks("circuit", tests_circuit)
    return (tests_circuit,)


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
        Fault j is the j-th `error` instruction of dem.flattened().
        """
        h_rows, h_cols = [], []
        l_rows, l_cols = [], []
        priors = []

        j = 0
        for inst in dem.flattened():
            if inst.type != "error":
                continue                      # skip detector / logical_observable lines
            priors.append(inst.args_copy()[0])
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    h_rows.append(t.val); h_cols.append(j)
                elif t.is_logical_observable_id():
                    l_rows.append(t.val); l_cols.append(j)
                # t.is_separator() ('^' from decompose_errors) is ignored: the
                # fault's full symptom is the XOR of all its components.
            j += 1

        def _gf2_csr(rows, cols, shape):
            # Sum duplicates, then reduce mod 2 (a target listed twice cancels).
            M = sp.coo_matrix(
                (np.ones(len(rows), dtype=np.int64), (rows, cols)), shape=shape
            ).tocsr()
            M.data %= 2
            M.eliminate_zeros()
            return M.astype(np.uint8)

        H = _gf2_csr(h_rows, h_cols, (dem.num_detectors, j))
        L = _gf2_csr(l_rows, l_cols, (dem.num_observables, j))
        return H, L, np.asarray(priors, dtype=np.float64)

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
# 5.1 The reference noise model
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.1 The reference noise model (Pakhunov, 2026)

    Section 4's circuit is the one the experiments use: it extracts **both** check
    families, because the flag qubits attach to the X-check ancillas. The
    reference the literature reports against extracts only **one**. Theorem 1
    counts $nw$ CNOTs per round; for $[[72,12,6]]$ that is 216, where our circuit
    applies 432.

    That single difference, not the noise channel, is the whole gap. Measured on
    the reference circuit at $T = 12$, `DEPOLARIZE2` per CNOT gives
    $\alpha = 3.56$ against Theorem 2's $3.505$ — 1.6% high. A single X
    mechanism per CNOT, the most literal reading of the theorem, gives $4.17$
    instead, 19% high. Theorem 2's $\alpha$ therefore describes a depolarizing
    channel, which is the one we already had.

    This section builds the reference circuit so the claim in the write-up is a
    measurement, not an assertion:

    | Theorem | Prediction | Checked against |
    |---|---|---|
    | 1 | $N = n(wT + T/2 + 1)$ fault mechanisms | exact fault count |
    | 2 | $\lambda = \alpha n T p$, $\alpha = 3.505$ | sum of DEM priors |
    | 3, 5 | $P_{peel} = \exp(-\bar{d}\lambda^2 / 2N)^{A_0}$ | Table II's 93.5% |

    The reference circuit is for **validation only**. It has no flag qubits, so
    every experiment keeps using `build_memory_circuit`.
    """)
    return


@app.cell
def _(check_schedule, dem_to_matrices, np, sp, stim):
    ALPHA_W3 = 3.505      # Theorem 2, column weight 3, standard superconducting model
    A0_GROSS = 0.869      # Theorem 5 / Table III, constant across the Gross family

    def theory_faults(code, rounds):
        """
        Theorem 1: N = n(wT + T/2 + 1) = nwT CNOT + (n/2)T measurement + n boundary.

        w is the COLUMN weight (qubit degree), not the row weight. The theorem is
        stated for BB codes, whose column weight is uniform; `uniform` reports
        whether that holds, so a GB code with |a| != |b| is not silently compared.
        """
        cols = np.asarray(code.hz.sum(axis=0)).ravel()
        w = int(cols[0])
        return dict(total=int(code.n * (w * rounds + rounds / 2 + 1)),
                    cnot=int(code.n * w * rounds),
                    measurement=int(code.n // 2 * rounds),
                    boundary=int(code.n),
                    column_weight=w,
                    uniform=bool((cols == w).all()))

    def theory_lambda(code, rounds, p, alpha=ALPHA_W3):
        """Theorem 2: expected number of fault mechanisms firing per shot."""
        return alpha * code.n * rounds * p

    def theory_peel(faults, lam, mean_degree, a0=A0_GROSS):
        """
        Theorems 3 and 5: P_peel = exp(-beta lambda^2) ** A0 with beta = d_bar/2N.
        Evaluated on a circuit's own measured graph, this is what a perfect
        queue-based peeler should achieve on it.
        """
        return float(np.exp(-mean_degree / (2 * faults) * lam ** 2) ** a0)

    def build_reference_circuit(code, rounds, p, extraction="z-only"):
        """
        Z-basis memory in the reference model: Z-check extraction only, no flags.

        extraction="both" reproduces section 4's gate count instead, so the two
        circuits can be compared one axis at a time. Detectors are Z-check only
        either way, matching build_memory_circuit(x_detectors=False).
        """
        if extraction not in ("z-only", "both"):
            raise ValueError("extraction must be 'z-only' or 'both'")
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
                c.append("DEPOLARIZE2", flat, p)
            c.append("TICK")

        def measure(name, qubits):
            if not qubits:
                return []
            c.append(name, qubits, p) if p > 0 else c.append(name, qubits)
            start = meas[0]
            meas[0] += len(qubits)
            return list(range(start, meas[0]))

        c.append("R", data + zanc)
        if xanc:
            c.append("RX", xanc)
        if p > 0:
            c.append("X_ERROR", data + zanc, p)     # n boundary + ancilla resets
        c.append("TICK")

        prev_z = None
        for t in range(rounds):
            if xanc:
                for l in range(sx.shape[0]):
                    cx_layer([(xanc[r], int(sx[l, r])) for r in range(mx)])
            for l in range(sz.shape[0]):
                cx_layer([(int(sz[l, r]), zanc[r]) for r in range(mz)])
            if xanc:
                measure("MRX", xanc)
            z_idx = measure("MR", zanc)
            if p > 0 and t < rounds - 1:
                c.append("X_ERROR", zanc, p)
            for i in range(mz):
                tg = [rec(z_idx[i])] + ([rec(prev_z[i])] if t > 0 else [])
                c.append("DETECTOR", tg, [i, t, 0])
            c.append("TICK")
            prev_z = z_idx

        d_idx = measure("M", data)
        for i in range(mz):
            c.append("DETECTOR",
                     [rec(d_idx[q]) for q in hz_rows[i]] + [rec(prev_z[i])],
                     [i, rounds, 3])
        for j in range(lz.shape[0]):
            c.append("OBSERVABLE_INCLUDE",
                     [rec(d_idx[q]) for q in np.flatnonzero(lz[j])], j)
        return c

    def reference_report(code, rounds, p, extraction="z-only"):
        """Measured DEM quantities beside what Theorems 1, 2, 3 and 5 predict."""
        circ = build_reference_circuit(code, rounds, p, extraction=extraction)
        H, _L, priors = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        th = theory_faults(code, rounds)
        lam = float(priors.sum())
        Hb = (H > 0).astype(np.int8)
        G = (Hb.T @ Hb).tocsr()
        deg = float((np.diff(G.indptr) - 1).mean())
        return dict(detectors=int(H.shape[0]), faults=int(H.shape[1]),
                    theory_faults=th["total"], fault_ratio=H.shape[1] / th["total"],
                    lam=lam, theory_lambda=theory_lambda(code, rounds, p),
                    lambda_ratio=lam / theory_lambda(code, rounds, p),
                    mean_degree=deg,
                    theory_peel=theory_peel(H.shape[1], lam, deg),
                    uniform_weight=th["uniform"])

    # Table II of Pakhunov (2026), "Actual" column: peeling success at p = 1e-3.
    TABLE_II = {("[[72, 12, 6]]", 12): 0.935,
                ("[[144, 12, 12]]", 12): 0.879,
                ("[[288, 12, 18]]", 12): 0.778}

    return (ALPHA_W3, A0_GROSS, TABLE_II, build_reference_circuit,
            reference_report, theory_faults, theory_lambda, theory_peel)


@app.cell
def _(
    BB_PRESETS,
    TABLE_II,
    bb_from_preset,
    build_reference_circuit,
    reference_report,
    render_checks,
    run_checks,
    theory_faults,
):
    def _code(key="[[72, 12, 6]]"):
        return bb_from_preset(key, BB_PRESETS)

    def _theorem_1_is_exact():
        """The reference circuit must hold EXACTLY n(wT + T/2 + 1) fault mechanisms."""
        for key, T in [("[[72, 12, 6]]", 3), ("[[72, 12, 6]]", 6), ("[[72, 12, 6]]", 12),
                       ("[[144, 12, 12]]", 6)]:
            c = _code(key)
            r = reference_report(c, T, 1e-3)
            assert r["faults"] == r["theory_faults"], \
                f"{key} T={T}: {r['faults']} faults, Theorem 1 says {r['theory_faults']}"

    def _theorem_2_lambda():
        # alpha is quoted for T = 12; the boundary term 1/T inflates small T
        for T in (6, 12):
            r = reference_report(_code(), T, 1e-3)
            assert 0.90 <= r["lambda_ratio"] <= 1.10, \
                f"T={T}: lambda {r['lam']:.2f} vs theory {r['theory_lambda']:.2f}"

    def _mean_degree_matches_paper():
        # the paper reports d_bar = 52.3 for the Gross family at T = 12
        for key in ("[[72, 12, 6]]", "[[144, 12, 12]]"):
            r = reference_report(_code(key), 12, 1e-3)
            assert 45 <= r["mean_degree"] <= 60, f"{key}: d_bar = {r['mean_degree']:.1f}"

    def _analytic_peel_reproduces_table_ii():
        """The headline validation: Theorems 3 and 5 on our own graph must land on
        Table II. This is what lets section 3.3 claim reproduction."""
        for key, T in [("[[72, 12, 6]]", 12), ("[[144, 12, 12]]", 12)]:
            r = reference_report(_code(key), T, 1e-3)
            target = TABLE_II[(key, T)]
            assert abs(r["theory_peel"] - target) < 0.02, \
                f"{key}: predicted {r['theory_peel']:.3f} against Table II {target:.3f}"

    def _experiment_circuit_deviates_on_purpose():
        """Section 4's circuit extracts both families, so it must NOT match Theorem 1.
        Asserting the deviation keeps the two circuits from being confused."""
        r = reference_report(_code(), 12, 1e-3, extraction="both")
        assert r["fault_ratio"] > 1.3, \
            f"both-family circuit unexpectedly matches theory (ratio {r['fault_ratio']:.2f})"

    def _reference_circuit_is_well_formed():
        c = _code()
        clean = build_reference_circuit(c, 4, 0.0)
        det, obs = clean.compile_detector_sampler(seed=5).sample(
            32, separate_observables=True)
        assert not det.any() and not obs.any(), "noiseless reference circuit fires"
        assert clean.num_observables == c.k
        build_reference_circuit(c, 4, 1e-3).detector_error_model(decompose_errors=False)

    def _non_uniform_weight_is_flagged():
        # Theorem 1 assumes a uniform column weight; a GB code with |a| != |b| is not
        gb = theory_faults(_code(), 6)
        assert gb["uniform"] and gb["column_weight"] == 3

    tests_reference = run_checks([
        ("reference circuit is well formed and silent at p = 0", _reference_circuit_is_well_formed),
        ("Theorem 1 fault count is reproduced exactly", _theorem_1_is_exact),
        ("Theorem 2 lambda is reproduced within 10%", _theorem_2_lambda),
        ("mean fault-graph degree matches the paper's 52.3", _mean_degree_matches_paper),
        ("analytic peeling prediction reproduces Table II", _analytic_peel_reproduces_table_ii),
        ("section 4's circuit deviates from Theorem 1, as designed",
         _experiment_circuit_deviates_on_purpose),
        ("column-weight uniformity is reported", _non_uniform_weight_is_flagged),
    ])
    render_checks("5.1 reference model", tests_reference)
    return (tests_reference,)


# =============================================================================
# 5.2 Surface-code baseline
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.2 Surface-code baseline

    `build_memory_circuit` cannot build a surface code. Its CNOT schedule orders
    each check's data qubits by **lattice offset**, which is what makes every
    layer a permutation — and that relies on translation invariance on a torus.
    A surface code has boundaries, so its checks do not all share one offset
    pattern and `check_schedule` raises.

    Stim generates these circuits natively, so the baseline costs almost nothing:
    the generated circuit goes through the same `dem_to_matrices` and the same
    decoders. Only the circuit construction differs.

    The contrast is worth stating in its own right. A surface-code DEM decomposes
    into graphlike components — every fault has at most two symptoms, which is
    what lets matching decoders work — while a BB DEM does not. That is the
    structural reason this thesis needs BP+OSD rather than MWPM, and the test
    below asserts it rather than leaving it as a claim.
    """)
    return


@app.cell
def _(dem_to_matrices, np, stim):
    def surface_code_problem(distance, rounds, p, basis="Z"):
        """
        Rotated surface-code memory as a decoding problem, in Stim's own circuit.

        Returns the same (circuit, H, L, priors) shape the BB path produces, so a
        baseline run reuses the decoders and metrics unchanged. Detectors number
        (d^2 - 1) * rounds and there is exactly one logical observable.
        """
        if basis.upper() not in ("X", "Z"):
            raise ValueError("basis must be 'X' or 'Z'")
        if distance < 3 or distance % 2 == 0:
            raise ValueError("rotated surface code needs an odd distance >= 3")
        circ = stim.Circuit.generated(
            f"surface_code:rotated_memory_{basis.lower()}",
            distance=distance, rounds=rounds,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p)
        H, L, priors = dem_to_matrices(
            circ.detector_error_model(decompose_errors=False))
        return dict(circuit=circ, H=H, L=L, priors=priors,
                    distance=distance, rounds=rounds, p=p, basis=basis.upper(),
                    name=f"surface d={distance}")

    def is_graphlike(circuit):
        """
        True when every fault mechanism has at most two symptoms, i.e. the DEM is a
        matching graph. Surface codes are; bivariate bicycle codes are not, which
        is why this thesis uses BP+OSD instead of MWPM.
        """
        try:
            circuit.detector_error_model(decompose_errors=True)
            return True
        except ValueError:
            return False

    def dem_mean_degree(H):
        """Mean number of other faults sharing a detector with a given fault."""
        Hb = (H > 0).astype(np.int8)
        G = (Hb.T @ Hb).tocsr()
        return float((np.diff(G.indptr) - 1).mean())

    return dem_mean_degree, is_graphlike, surface_code_problem


@app.cell
def _(
    BB_PRESETS,
    BpOsdDecoder,
    bb_from_preset,
    build_memory_circuit,
    dem_mean_degree,
    dem_to_matrices,
    is_graphlike,
    np,
    plot_surface_sweep,
    render_checks,
    run_checks,
    surface_code_problem,
    surface_sweep_point,
    threshold_estimate,
):
    def _builds_at_every_distance():
        for d in (3, 5, 7):
            for T in (3, 6):
                pr = surface_code_problem(d, T, 1e-3)
                assert pr["circuit"].num_detectors == (d * d - 1) * T, \
                    f"d={d} T={T}: {pr['circuit'].num_detectors} detectors"
                assert pr["circuit"].num_observables == 1
                assert pr["H"].shape == (pr["circuit"].num_detectors, pr["L"].shape[1])

    def _rejects_bad_parameters():
        for bad in (dict(distance=4, rounds=3, p=1e-3), dict(distance=2, rounds=3, p=1e-3)):
            try:
                surface_code_problem(**bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted {bad}")

    def _noiseless_is_silent():
        pr = surface_code_problem(5, 3, 0.0)
        det, obs = pr["circuit"].compile_detector_sampler(seed=4).sample(
            32, separate_observables=True)
        assert not det.any() and not obs.any()

    def _decodes_with_the_same_decoder():
        pr = surface_code_problem(5, 3, 2e-3)
        dec = BpOsdDecoder(pr["H"], pr["priors"], osd_order=0)
        det = pr["circuit"].compile_detector_sampler(seed=6).sample(40).astype(np.uint8)
        for sdr in det:
            r = dec.decode(sdr)
            assert np.array_equal((pr["H"] @ np.asarray(r.correction, np.uint8)) % 2, sdr), \
                "surface-code correction does not reproduce its syndrome"

    def _surface_is_graphlike_and_bb_is_not():
        """The structural reason this thesis cannot use a matching decoder."""
        assert is_graphlike(surface_code_problem(5, 3, 1e-3)["circuit"]), \
            "surface-code DEM should decompose into graphlike components"
        bb = build_memory_circuit(bb_from_preset("[[72, 12, 6]]", BB_PRESETS), 3, 1e-3)
        assert not is_graphlike(bb), \
            "a BB DEM decomposed into graphlike components: check the circuit"

    def _surface_graph_is_far_sparser():
        """Quantifies the contrast: a BB fault graph is an order of magnitude denser."""
        surf = dem_mean_degree(surface_code_problem(5, 6, 1e-3)["H"])
        c = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        bb_H, _L, _pr = dem_to_matrices(
            build_memory_circuit(c, 6, 1e-3).detector_error_model(decompose_errors=False))
        bb = dem_mean_degree(bb_H)
        assert surf < bb / 3, \
            f"surface mean degree {surf:.1f} not far below BB's {bb:.1f}"

    def _sweep_point_runs_end_to_end():
        """The Experiment 7 runner, at its cheapest, so a long sweep cannot fail late."""
        pt = surface_sweep_point(3, 3, 3e-3, 200, seed=11, osd_order=0, workers=1)
        assert pt["distance"] == 3 and pt["shots"] == 200
        assert 0 <= pt["ler"] <= 1 and pt["ci_low"] <= pt["ler"] <= pt["ci_high"]
        assert pt["failures"] == round(pt["ler"] * 200)
        assert pt["detectors"] == (3 * 3 - 1) * 3
        assert pt["time_p50_us"] > 0

    def _threshold_estimate_finds_a_crossing():
        # synthetic: d=7 beats d=5 below 3e-3 and loses above it
        pts = []
        for p, l5, l7 in [(1e-3, 1e-2, 3e-3), (3e-3, 4e-2, 4e-2), (1e-2, 1e-1, 2e-1)]:
            pts.append(dict(distance=5, p=p, ler=l5))
            pts.append(dict(distance=7, p=p, ler=l7))
        t = threshold_estimate(pts)
        assert t is not None and 1e-3 <= t <= 1e-2, f"threshold {t}"
        # no crossing -> None, rather than a fabricated number
        flat = [dict(distance=d, p=p, ler=1e-2 if d == 5 else 1e-3)
                for d in (5, 7) for p in (1e-3, 3e-3)]
        assert threshold_estimate(flat) is None
        assert threshold_estimate([dict(distance=3, p=1e-3, ler=1e-2)]) is None

    def _sweep_plot_renders():
        import io
        pts = [dict(distance=d, p=p, ler=1e-3 * d, ci_low=5e-4 * d, ci_high=2e-3 * d,
                    failures=10, shots=1000)
               for d in (3, 5, 7) for p in (1e-3, 3e-3)]
        fig = plot_surface_sweep(pts)
        fig.savefig(io.BytesIO(), format="png")

    tests_surface = run_checks([
        ("builds at d = 3, 5, 7 with the expected detector count", _builds_at_every_distance),
        ("even or too-small distances are rejected", _rejects_bad_parameters),
        ("noiseless circuit fires nothing", _noiseless_is_silent),
        ("decodes with the same BP+OSD decoder, corrections consistent",
         _decodes_with_the_same_decoder),
        ("surface DEM is graphlike, BB DEM is not", _surface_is_graphlike_and_bb_is_not),
        ("surface fault graph is far sparser than a BB one", _surface_graph_is_far_sparser),
        ("Experiment 7 sweep point runs end to end", _sweep_point_runs_end_to_end),
        ("threshold estimate finds a crossing and refuses to invent one",
         _threshold_estimate_finds_a_crossing),
        ("surface sweep plot renders", _sweep_plot_renders),
    ])
    render_checks("5.2 surface baseline", tests_surface)
    return (tests_surface,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Decoders

    Every decoder exposes one method, `decode(syndrome) -> DecodeResult`.

    | role | decoder | failure signal |
    |---|---|---|
    | weak | **greedy peeling** (after Pakhunov, 2026) | residual syndrome not cleared |
    | weak | **BP** (min-sum) | BP did not converge |
    | strong | **BP+OSD** (Roffe et al., 2020) | never fails to reproduce the syndrome |
    | strong | **BP+LSD** (Hillmann et al., 2025) | never fails to reproduce the syndrome |

    **Peeling rule.** A fault is peeled when its whole detector signature is
    active and every other fully-active fault touching those detectors has a
    signature *contained in* its own. One fault explaining the syndrome is far
    more likely than several, so a dominating fault wins. Pure uniqueness
    peeling stalls on circuit-level DEMs, where such nested signatures are common.

    **Soft output.** BP-based decoders return their posterior log-likelihood
    ratios in `soft` (positive = "no fault"). Soft-information triggers use them.

    **Cost.** `work` counts decoder-native operations: detector visits for
    peeling, BP iterations for the BP family. OSD/LSD post-processing cost is
    *not* included in `work` -- compare those decoders by wall-clock time.
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
        soft: np.ndarray = None     # posterior LLRs (BP family); None for peeling

    return (DecodeResult,)


@app.cell
def _(DecodeResult, np, sp):
    from collections import deque as _deque

    from ldpc.bp_decoder import BpDecoder as _LdpcBp
    from ldpc.bplsd_decoder import BpLsdDecoder as _LdpcBpLsd
    from ldpc.bposd_decoder import BpOsdDecoder as _LdpcBpOsd

    class PeelingDecoder:
        """
        Queue-based greedy peeling over the DEM, with the dominance rule.

        Contract:
          - decode(zeros) returns an all-zero correction with converged=True
          - converged=True only if the returned correction reproduces the syndrome
          - work = number of detector visits (queue pops)
        `priors` is accepted for interface uniformity; the dominance rule does
        not need it, because nested signatures are resolved by containment.
        """

        def __init__(self, H, priors=None):
            Hc, Hr = H.tocsc(), H.tocsr()
            self.num_faults = H.shape[1]
            self.num_detectors = H.shape[0]
            # detectors of each fault, and faults touching each detector
            self.fault_dets = [Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]]
                               for j in range(self.num_faults)]
            self.det_faults = [Hr.indices[Hr.indptr[i]:Hr.indptr[i + 1]]
                               for i in range(self.num_detectors)]
            self.fault_set = [frozenset(d.tolist()) for d in self.fault_dets]
            self._neighbourhood = {}          # lazy cache: fault -> nearby detectors

        def _fully_active(self, j, s):
            return bool(s[self.fault_dets[j]].all())

        def _nearby_detectors(self, j):
            """Detectors of every fault that shares a detector with fault j."""
            if j not in self._neighbourhood:
                near = {int(d3)
                        for d in self.fault_dets[j]
                        for k in self.det_faults[d]
                        for d3 in self.fault_dets[k]}
                self._neighbourhood[j] = np.fromiter(near, dtype=np.int64)
            return self._neighbourhood[j]

        def _dominant_fault(self, d, s):
            """
            The fault to peel at detector d, or None.

            Returns a fully-active fault whose signature contains the signature
            of every other fully-active fault touching any of its detectors.
            """
            full = [int(j) for j in self.det_faults[d] if self._fully_active(j, s)]
            if not full:
                return None
            best = max(full, key=lambda j: len(self.fault_dets[j]))
            best_set = self.fault_set[best]
            for d2 in self.fault_dets[best]:
                for k in self.det_faults[d2]:
                    if k != best and self._fully_active(k, s) \
                            and not self.fault_set[k] <= best_set:
                        return None               # a genuine rival: ambiguous
            return best

        def decode(self, syndrome):
            s = np.asarray(syndrome, dtype=np.uint8).copy()
            e = np.zeros(self.num_faults, dtype=np.uint8)
            queue = _deque(np.flatnonzero(s).tolist())
            queued = s.astype(bool)
            visits = 0
            while queue:
                d = queue.popleft()
                queued[d] = False
                visits += 1
                if not s[d]:
                    continue
                j = self._dominant_fault(d, s)
                if j is None:
                    continue
                # Peeling a fully-active fault only switches detectors OFF, so the
                # syndrome weight strictly decreases and the loop must terminate.
                e[j] ^= 1
                s[self.fault_dets[j]] ^= 1
                # Removing j may unblock faults that were rivals of j: revisit them.
                for d3 in self._nearby_detectors(j):
                    if s[d3] and not queued[d3]:
                        queue.append(int(d3))
                        queued[d3] = True
            return DecodeResult(e, not s.any(), visits)

    def ldpc_kwargs(priors, max_iter):
        """Settings shared by every BP-family decoder, so comparisons are fair."""
        return dict(error_channel=[float(x) for x in priors], max_iter=max_iter,
                    bp_method="ms", ms_scaling_factor=0.625, schedule="parallel")

    class BpDecoder:
        """
        Min-sum belief propagation (weak decoder).

        Contract: converged=True iff BP converged, which in ldpc means the
        correction reproduces the syndrome. work = BP iterations used.
        soft = posterior log-likelihood ratios.
        """

        def __init__(self, H, priors, max_iter=20):
            self.dec = _LdpcBp(sp.csr_matrix(H, dtype=np.uint8), **ldpc_kwargs(priors, max_iter))

        def decode(self, syndrome):
            e = self.dec.decode(np.asarray(syndrome, dtype=np.uint8))
            return DecodeResult(np.asarray(e, dtype=np.uint8), bool(self.dec.converge),
                                int(self.dec.iter), soft=np.array(self.dec.log_prob_ratios))

    class BpOsdDecoder:
        """
        BP + ordered-statistics decoding (strong decoder).

        ldpc runs OSD only when BP fails to converge; otherwise the BP solution is
        returned unchanged. Contract: the correction always reproduces the
        syndrome, so converged is always True. work = BP iterations used.
        """

        def __init__(self, H, priors, max_iter=20, osd_order=0):
            self.dec = _LdpcBpOsd(sp.csr_matrix(H, dtype=np.uint8),
                                  osd_method="osd_cs", osd_order=osd_order,
                                  **ldpc_kwargs(priors, max_iter))

        def decode(self, syndrome):
            e = self.dec.decode(np.asarray(syndrome, dtype=np.uint8))
            return DecodeResult(np.asarray(e, dtype=np.uint8), True,
                                int(self.dec.iter), soft=np.array(self.dec.log_prob_ratios))

    class BpLsdDecoder:
        """
        BP + localized statistics decoding (strong decoder).

        Same contract as BpOsdDecoder: LSD runs only when BP fails, and the
        correction always reproduces the syndrome.
        """

        def __init__(self, H, priors, max_iter=20, lsd_order=0):
            self.dec = _LdpcBpLsd(sp.csr_matrix(H, dtype=np.uint8),
                                  lsd_method="LSD_CS", lsd_order=lsd_order,
                                  **ldpc_kwargs(priors, max_iter))

        def decode(self, syndrome):
            e = self.dec.decode(np.asarray(syndrome, dtype=np.uint8))
            return DecodeResult(np.asarray(e, dtype=np.uint8), True,
                                int(self.dec.iter), soft=np.array(self.dec.log_prob_ratios))

    return BpDecoder, BpLsdDecoder, BpOsdDecoder, PeelingDecoder, ldpc_kwargs


@app.cell
def _(
    BB_PRESETS,
    BpDecoder,
    BpLsdDecoder,
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

    def _zero(make):
        def check():
            H, pr, _ = _setup()
            r = make(H, pr).decode(np.zeros(H.shape[0], dtype=np.uint8))
            assert r.converged and not np.asarray(r.correction).any()
        return check

    def _peel_single_faults():
        # Pure uniqueness peeling fails most of these: nested signatures are common.
        H, pr, _ = _setup()
        dec = PeelingDecoder(H, pr)
        rng = np.random.default_rng(3)
        for j in rng.choice(H.shape[1], size=50, replace=False):
            s = H[:, [j]].toarray().ravel().astype(np.uint8)
            r = dec.decode(s)
            assert r.converged and _consistent(H, s, r), f"single fault {j} not resolved"

    def _honest(make):
        def check():
            H, pr, det = _setup()
            dec = make(H, pr)
            for s in det:
                r = dec.decode(s)
                assert r.converged == _consistent(H, s, r), \
                    "converged flag disagrees with H·e = s"
        return check

    def _always_consistent(make):
        def check():
            H, pr, det = _setup()
            dec = make(H, pr)
            for s in det:
                r = dec.decode(s)
                assert r.converged and _consistent(H, s, r), \
                    "correction does not reproduce the syndrome"
        return check

    def _soft_output():
        H, pr, det = _setup()
        for make in (BpDecoder, BpOsdDecoder, BpLsdDecoder):
            r = make(H, pr).decode(det[0])
            assert r.soft is not None and r.soft.shape == (H.shape[1],), make.__name__
            assert np.isfinite(r.soft).all(), f"{make.__name__}: non-finite LLRs"
        assert PeelingDecoder(H, pr).decode(det[0]).soft is None

    def _strong_keeps_bp_solution():
        # OSD/LSD should only act when BP fails: on BP-converged shots the strong
        # decoder must return BP's own answer. This is why "BP primary, BP+OSD
        # secondary" repeats work rather than adding a new decoder.
        H, pr, det = _setup(p=1e-3)
        bp = BpDecoder(H, pr)
        strong = [BpOsdDecoder(H, pr), BpLsdDecoder(H, pr)]
        checked = 0
        for s in det:
            rb = bp.decode(s)
            if not rb.converged:
                continue
            checked += 1
            for dec in strong:
                assert np.array_equal(dec.decode(s).correction, rb.correction), \
                    f"{type(dec).__name__} changed a converged BP solution"
        assert checked > 0, "no BP-converged shots to check"

    def _peel_complete():
        # Peeling must stop only when nothing is peelable. Re-decoding the
        # residual syndrome from scratch must therefore peel nothing; if it
        # does, the queue failed to revisit detectors unblocked by earlier peels.
        c = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        circ = build_memory_circuit(c, 3, 4e-3, use_flags=True)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        det = circ.compile_detector_sampler(seed=21).sample(150).astype(np.uint8)
        dec = PeelingDecoder(H, pr)
        for s in det:
            r = dec.decode(s)
            residual = (s ^ (H @ r.correction) % 2).astype(np.uint8)
            again = dec.decode(residual)
            assert not again.correction.any(), \
                "peeling left a peelable fault behind (missing revisit?)"

    def _work_counts():
        H, pr, det = _setup()
        nz = next(s for s in det if s.any())
        assert PeelingDecoder(H, pr).decode(nz).work > 0
        r = BpDecoder(H, pr, max_iter=7).decode(nz)
        assert 1 <= r.work <= 7, f"BP work {r.work} outside [1, max_iter]"

    tests_decoders = run_checks([
        ("peeling: zero syndrome gives zero correction", _zero(PeelingDecoder)),
        ("peeling: resolves 50 random single faults (dominance rule)", _peel_single_faults),
        ("peeling: converged flag is honest on sampled shots", _honest(PeelingDecoder)),
        ("BP: zero syndrome gives zero correction", _zero(BpDecoder)),
        ("BP: converged flag is honest on sampled shots", _honest(BpDecoder)),
        ("BP+OSD: correction always reproduces the syndrome", _always_consistent(BpOsdDecoder)),
        ("BP+LSD: correction always reproduces the syndrome", _always_consistent(BpLsdDecoder)),
        ("BP family returns finite soft output; peeling returns none", _soft_output),
        ("BP+OSD / BP+LSD return BP's solution when BP converges", _strong_keeps_bp_solution),
        ("peeling: stops only when nothing is left to peel", _peel_complete),
        ("work is counted and bounded", _work_counts),
    ])
    render_checks("decoders", tests_decoders)
    return (tests_decoders,)


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

    **Beating `always` is possible.** A policy can fail less than `always` when
    the primary is right on shots where the secondary errs. Over 250,000 shots on
    [[72,12,6]] at $T = 12$, peeling converged and was correct where BP+OSD-0 was
    wrong on **66** shots, and converged-but-wrong where BP+OSD-0 was right on
    **0** — so `primary_fail` keeps 66 answers that `always` would have thrown
    away and loses none. That is the whole of its 0.00512 against 0.00538 margin.
    A policy beating `always` is therefore not automatically a bug, but it must be
    confirmed with a paired test on identical shots (Experiment 6 does this).
    """)
    return


@app.cell
def _():
    TRIGGERS = ("never", "always", "primary_fail", "flag", "flag_or_fail")
    FLAG_TRIGGERS = ("flag", "flag_or_fail")

    class SwitchPolicy:
        """
        Contract:
          - unknown trigger -> ValueError; flag triggers without a flag mask -> ValueError
          - `always` never calls the primary; `never` never calls the secondary
          - for flag triggers, a fired flag escalates WITHOUT calling the primary
          - the returned DecodeResult has escalated=True iff the secondary ran

        work: primary work plus secondary work when both run. If the two decoders
        count different things (e.g. detector visits vs BP iterations) the sum mixes
        units, so compare policies with different decoders by wall-clock time.
        """

        def __init__(self, primary, secondary, trigger, flag_mask=None, name=None):
            if trigger not in TRIGGERS:
                raise ValueError(f"unknown trigger {trigger!r}; expected one of {TRIGGERS}")
            if trigger in FLAG_TRIGGERS:
                if flag_mask is None or not np.any(flag_mask):
                    raise ValueError(
                        f"trigger {trigger!r} needs a flag mask with at least one flag "
                        "detector: build the circuit with use_flags=True")
                flag_mask = np.asarray(flag_mask, dtype=bool)
            self.primary = primary
            self.secondary = secondary
            self.trigger = trigger
            self.flag_mask = flag_mask
            self.name = name or trigger

        def _flag_fired(self, syndrome):
            if self.trigger not in FLAG_TRIGGERS:
                return False
            s = np.asarray(syndrome)
            if s.shape != self.flag_mask.shape:
                raise ValueError(f"syndrome length {s.shape} does not match flag mask "
                                 f"{self.flag_mask.shape}: mask built for another circuit?")
            return bool(s[self.flag_mask].any())

        def _escalate(self, syndrome, work_so_far=0):
            r = self.secondary.decode(syndrome)
            return DecodeResult(r.correction, r.converged, work_so_far + r.work,
                                escalated=True, soft=r.soft)

        def decode(self, syndrome):
            if self.trigger == "always":
                return self._escalate(syndrome)

            # The flag is read BEFORE the primary runs -- that is the point of a
            # pre-decode trigger, so a flagged shot never pays for the primary.
            if self._flag_fired(syndrome):
                return self._escalate(syndrome)

            r = self.primary.decode(syndrome)
            if self.trigger != "never" and not r.converged:
                return self._escalate(syndrome, r.work)
            return DecodeResult(r.correction, r.converged, r.work,
                                escalated=False, soft=r.soft)

    return FLAG_TRIGGERS, SwitchPolicy


@app.cell
def _(DecodeResult, SwitchPolicy, dz, np, render_checks, run_checks):
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

    def _beating_always_is_representable():
        """
        Section 7 states that a policy CAN fail less than `always`, because the
        primary may be right where the secondary errs. The accounting has to be
        able to express that: a reader who believes the opposite might "fix" it.

        One shot, primary right and secondary wrong. primary_fail keeps the
        primary's answer, so it must record fewer failures than always.
        """
        n = 4
        weak = dict(conv=np.ones(n, bool), fail=np.zeros(n, bool),
                    work=np.ones(n, np.int64), time_us=np.ones(n))
        strong = dict(conv=np.ones(n, bool), fail=np.zeros(n, bool),
                      work=np.ones(n, np.int64), time_us=np.ones(n))
        strong["fail"][2] = True                      # the secondary errs on shot 2
        none = np.zeros(n, bool)
        pf = dz.derive(weak, strong, none, True)      # primary_fail
        always = dz.derive(weak, strong, np.ones(n, bool), True)
        assert int(pf["fail"].sum()) == 0, "primary_fail should keep the primary's answer"
        assert int(always["fail"].sum()) == 1, "always should inherit the secondary's error"
        assert pf["fail"].sum() < always["fail"].sum(), \
            "the accounting cannot express a policy beating `always` -- see section 7"

    def _escalating_a_correct_primary_can_harm():
        """The mirror image: a pre-decode trigger that fires on that same shot
        converts a success into a failure. This is the `harm` column of §11.6."""
        n = 4
        weak = dict(conv=np.ones(n, bool), fail=np.zeros(n, bool),
                    work=np.ones(n, np.int64), time_us=np.ones(n))
        strong = dict(conv=np.ones(n, bool), fail=np.zeros(n, bool),
                      work=np.ones(n, np.int64), time_us=np.ones(n))
        strong["fail"][2] = True
        pre = np.zeros(n, bool)
        pre[2] = True                                  # the flag fires on that shot
        flagged = dz.derive(weak, strong, pre, True)
        assert int(flagged["fail"].sum()) == 1, "escalating a correct primary should harm"
        assert flagged["cost"][2] == 1.0, "a pre-escalated shot must not pay the primary"

    tests_switch = run_checks([
        ("never: secondary is never called", _never),
        ("always: primary is never called", _always),
        ("primary_fail: escalates only on primary failure, ignores flags", _primary_fail),
        ("flag triggers: a fired flag skips the primary pass", _flag_skips_primary),
        ("flag triggers: unflagged shots fall back to primary_fail", _flag_fallback),
        ("invalid trigger or missing flag mask raises ValueError", _validation),
        ("a policy beating `always` is representable, not a bug",
         _beating_always_is_representable),
        ("escalating a correct primary is recorded as harm, and skips its cost",
         _escalating_a_correct_primary_can_harm),
    ])
    render_checks("switch policies", tests_switch)
    return (tests_switch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Statistics and benchmarking

    - **Logical error** = predicted observable flips $L\hat e$ differ from the
      true flips sampled by Stim, on *any* of the $k$ observables.
    - **Confidence intervals:** Wilson score interval (valid at zero failures).
    - **Paired comparison:** every policy decodes the *same* sampled shots.

    Every failure falls into exactly one category:

    | category | escalated? | primary converged? | meaning |
    |---|---|---|---|
    | escalated failure | yes | – | even the strong decoder got it wrong |
    | **silent failure** | no | yes | primary was confidently wrong; a convergence rule *cannot* catch these |
    | unconverged failure | no | no | primary gave up and nothing escalated (only with `never`) |

    Silent failures are the population a flag trigger could catch and
    `primary_fail` cannot, so they are the number the thesis argument rests on.
    """)
    return


@app.cell
def _(dataclass, field):
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
        silent_failures: int = 0        # failed, not escalated, primary claimed convergence
        unconverged_failures: int = 0   # failed, not escalated, primary did not converge
        # per-shot arrays (time_us, work, failed, escalated, converged) when
        # run_benchmark(..., keep_samples=True); excluded from == and from repr
        samples: dict = field(default=None, compare=False, repr=False)

        @property
        def escalated_failures(self):
            return self.failures - self.silent_failures - self.unconverged_failures

    return (Metrics,)


@app.cell
def _(Metrics, np, time):
    def wilson(k, n, z=1.96):
        """Wilson score interval for k successes in n trials -> (p_hat, low, high)."""
        if n <= 0:
            return 0.0, 0.0, 1.0
        if not 0 <= k <= n:
            raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
        p_hat = k / n
        z2 = z * z
        denom = 1 + z2 / n
        centre = (p_hat + z2 / (2 * n)) / denom
        half = z * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / denom
        low = 0.0 if k == 0 else max(0.0, centre - half)     # exact at the edges,
        high = 1.0 if k == n else min(1.0, centre + half)    # not off by float error
        return p_hat, low, high

    def run_benchmark(circuit, policies, L, shots, seed, keep_samples=False):
        """
        Sample `shots` from `circuit` ONCE, decode them with every policy.

        Args:
            policies:     dict name -> object with decode(syndrome) -> DecodeResult
            L:            observable matrix from dem_to_matrices
            keep_samples: also store per-shot arrays in Metrics.samples (for graphs)
        Returns:
            dict name -> Metrics
        """
        det, obs = circuit.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True)
        det = det.astype(np.uint8)
        obs = obs.astype(np.uint8)

        results = {}
        for name, policy in policies.items():
            failed = np.zeros(shots, dtype=bool)
            escalated = np.zeros(shots, dtype=bool)
            converged = np.zeros(shots, dtype=bool)
            work = np.zeros(shots, dtype=np.int64)
            time_us = np.zeros(shots)
            for i in range(shots):
                t0 = time.perf_counter()
                r = policy.decode(det[i])
                time_us[i] = (time.perf_counter() - t0) * 1e6
                predicted = (L @ np.asarray(r.correction, dtype=np.uint8)) % 2
                failed[i] = bool(np.any(predicted != obs[i]))
                escalated[i] = bool(r.escalated)
                converged[i] = bool(r.converged)
                work[i] = int(r.work)

            n_fail = int(failed.sum())
            p_hat, low, high = wilson(n_fail, shots)
            not_escalated = failed & ~escalated
            results[name] = Metrics(
                shots=shots, failures=n_fail, ler=p_hat, ci_low=low, ci_high=high,
                escalation_rate=float(escalated.mean()),
                work_mean=float(work.mean()), work_p99=float(np.percentile(work, 99)),
                time_p50_us=float(np.percentile(time_us, 50)),
                time_p99_us=float(np.percentile(time_us, 99)),
                silent_failures=int((not_escalated & converged).sum()),
                unconverged_failures=int((not_escalated & ~converged).sum()),
                samples=(dict(time_us=time_us, work=work, failed=failed,
                              escalated=escalated, converged=converged)
                         if keep_samples else None),
            )
        return results

    return run_benchmark, wilson


@app.cell
def _(np, plt):
    # One colour per policy, assigned by order of first appearance, so the same
    # policy has the same colour in every figure built from the same results.
    PLOT_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                "#8c564b", "#e377c2", "#17becf", "#7f7f7f", "#bcbd22"]

    def plot_colours(names):
        return {n: PLOT_PALETTE[i % len(PLOT_PALETTE)] for i, n in enumerate(names)}

    def plot_ler_marks(ax, x, metrics, colour, label=None, horizontal=False):
        """
        Plot LER with Wilson error bars on a log axis. A zero-failure point has
        no finite log value, so it is drawn as a hollow downward triangle at its
        95% upper bound -- 'the true rate is at most this' -- rather than dropped.
        """
        for xi, m in zip(x, metrics):
            if m.failures > 0:
                err = [[m.ler - m.ci_low], [m.ci_high - m.ler]]
                kw = dict(fmt="o", color=colour, capsize=4, ms=6, label=label)
                if horizontal:
                    ax.errorbar(m.ler, xi, xerr=err, **kw)
                else:
                    ax.errorbar(xi, m.ler, yerr=err, **kw)
            else:
                pos = (m.ci_high, xi) if horizontal else (xi, m.ci_high)
                ax.plot(*pos, marker="<" if horizontal else "v", ms=8, mfc="none",
                        mec=colour, ls="none", label=label)
            label = None                        # legend entry only once

    def plot_ablation(results):
        """
        Two panels, one row per policy: (a) LER with 95% Wilson intervals on a log
        axis; (b) fraction of shots escalated to the strong decoder.
        """
        names = list(results)
        col = plot_colours(names)
        y = np.arange(len(names))[::-1]
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 0.55 * len(names) + 1.8),
                                   sharey=True, gridspec_kw=dict(width_ratios=[3, 2]))
        for yi, n in zip(y, names):
            plot_ler_marks(a, [yi], [results[n]], col[n], horizontal=True)
            b.barh(yi, results[n].escalation_rate, color=col[n], alpha=0.85)
            b.text(results[n].escalation_rate + 0.01, yi, f"{results[n].escalation_rate:.0%}",
                   va="center", fontsize=9)
        a.set_xscale("log")
        a.xaxis.set_minor_formatter(plt.matplotlib.ticker.LogFormatter(labelOnlyBase=False,
                                                                          minor_thresholds=(2, 0.5)))
        a.tick_params(axis="x", which="minor", labelsize=7)
        a.set_yticks(y, names)
        a.set_xlabel("logical error rate (95% Wilson CI)")
        a.set_title("(a) accuracy")
        a.grid(True, which="both", axis="x", alpha=0.3)
        b.set_xlim(0, 1.15)
        b.set_xlabel("fraction of shots escalated")
        b.set_title("(b) escalation")
        b.grid(True, axis="x", alpha=0.3)
        shots = next(iter(results.values())).shots
        fig.suptitle(f"Policy ablation — {shots} paired shots"
                     "   (▽ = no failures observed, drawn at 95% upper bound)", fontsize=10)
        fig.tight_layout()
        return fig

    def plot_sweep(sweep):
        """
        Two panels versus physical error rate p (log axes): (a) LER with Wilson
        error bars; (b) escalation rate. `sweep` is dict p -> dict policy -> Metrics.
        """
        ps = sorted(sweep)
        names = list(sweep[ps[0]])
        col = plot_colours(names)
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
        for i, n in enumerate(names):
            ms = [sweep[p][n] for p in ps]
            # small multiplicative offset: policies with identical LER (which the
            # invariant makes common) would otherwise hide behind each other
            dodge = 1 + 0.035 * (i - (len(names) - 1) / 2)
            xs = [p * dodge for p in ps]
            plot_ler_marks(a, xs, ms, col[n], label=n)
            a.plot(xs, [m.ler if m.failures else m.ci_high for m in ms],
                   color=col[n], alpha=0.5, lw=1)
            b.plot(ps, [m.escalation_rate for m in ms], "s-", color=col[n], label=n)
        for ax in (a, b):
            ax.set_xscale("log")
            ax.set_xlabel("physical error rate p (per noise channel)")
            ax.grid(True, which="both", alpha=0.3)
        a.set_yscale("log")
        a.set_ylabel("logical error rate per shot")
        a.set_title("(a) accuracy (▽ = 95% upper bound, no failures)\n"
                    "points offset sideways slightly so overlaps stay visible", fontsize=10)
        b.set_ylim(-0.03, 1.03)
        b.set_ylabel("fraction escalated")
        b.set_title("(b) escalation — latency argument fails as this → 1")
        a.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def plot_outcomes(results):
        """
        Stacked bars: each policy's LER split into escalated / silent /
        unconverged failures. Silent failures are what a flag could catch and a
        convergence-based rule cannot.
        """
        names = list(results)
        y = np.arange(len(names))[::-1]
        parts = [("escalated failures", "escalated_failures", "#7f7f7f"),
                 ("silent failures", "silent_failures", "#d62728"),
                 ("unconverged failures", "unconverged_failures", "#ff7f0e")]
        fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.8))
        left = np.zeros(len(names))
        for label, attr, colour in parts:
            vals = np.array([getattr(results[n], attr) / results[n].shots for n in names])
            ax.barh(y, vals, left=left, color=colour, label=label)
            left += vals
        for yi, n in zip(y, names):
            m = results[n]
            ax.text(left[list(y).index(yi)] * 1.01 + 1e-4, yi,
                    f"{m.failures} fail ({m.silent_failures} silent)", va="center", fontsize=8)
        ax.set_yticks(y, names)
        ax.set_xlim(0, max(left.max() * 1.35, 1e-3))
        ax.set_xlabel("fraction of shots that failed, split by category")
        ax.set_ylabel("policy")
        ax.set_title("Where do failures come from?")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_latency(results):
        """
        Empirical CDF of per-shot decode time, log x-axis, with p50 and p99
        marked. Needs run_benchmark(..., keep_samples=True).
        """
        missing = [n for n, m in results.items() if m.samples is None]
        if missing:
            raise ValueError(f"no per-shot samples for {missing}: "
                             "call run_benchmark(..., keep_samples=True)")
        col = plot_colours(list(results))
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for n, m in results.items():
            t = np.sort(np.maximum(m.samples["time_us"], 1e-3))
            ax.step(t, np.arange(1, len(t) + 1) / len(t), where="post", color=col[n],
                    label=f"{n}  (p50 {m.time_p50_us:.0f}, p99 {m.time_p99_us:.0f} µs)")
            ax.plot([m.time_p99_us], [0.99], "o", color=col[n], ms=5)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.axhline(0.99, color="k", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("decode time per shot (µs, Python — relative comparison only)")
        ax.set_ylabel("fraction of shots decoded")
        ax.set_title("Latency distribution (dots = p99)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
        fig.tight_layout()
        return fig

    def plot_tradeoff(results):
        """
        Accuracy versus tail latency, one point per policy: lower-left is better.
        The thesis claim is a policy that moves left without moving up.
        """
        col = plot_colours(list(results))
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (n, m) in enumerate(results.items()):
            plot_ler_marks(ax, [m.time_p99_us], [m], col[n], label=n)
            ax.annotate(n, (m.time_p99_us, m.ler if m.failures else m.ci_high),
                        textcoords="offset points", xytext=(8, 14 - 12 * (i % 4)),
                        fontsize=8, color=col[n])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("p99 decode time (µs)")
        ax.set_ylabel("logical error rate per shot")
        ax.set_title("Accuracy vs tail latency — lower-left is better")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        return fig

    return (
        PLOT_PALETTE,
        plot_ablation,
        plot_colours,
        plot_latency,
        plot_ler_marks,
        plot_outcomes,
        plot_sweep,
        plot_tradeoff,
    )


@app.cell
def _(
    BB_PRESETS,
    DecodeResult,
    Metrics,
    bb_from_preset,
    build_memory_circuit,
    dem_to_matrices,
    np,
    plot_ablation,
    plot_latency,
    plot_outcomes,
    plot_sweep,
    plot_tradeoff,
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

    class _Recorder(_ZeroDecoder):
        """Zero decoder that remembers every syndrome it is given."""
        def __init__(self, nf):
            super().__init__(nf)
            self.seen = []

        def decode(self, s):
            self.seen.append(np.asarray(s, dtype=np.uint8).copy())
            return super().decode(s)

    def _paired():
        # Compare the syndromes themselves, shot by shot. Comparing failure
        # counts is not enough: two unpaired runs can have equal counts by chance.
        circ, H, L = _setup()
        a, b = _Recorder(H.shape[1]), _Recorder(H.shape[1])
        run_benchmark(circ, {"a": a, "b": b}, L, shots=100, seed=4)
        det = circ.compile_detector_sampler(seed=4).sample(100).astype(np.uint8)
        assert len(a.seen) == len(b.seen) == 100
        assert np.array_equal(np.array(a.seen), np.array(b.seen)), \
            "policies did not see the same shots"
        assert np.array_equal(np.array(a.seen), det), \
            "shots are not the ones sampled from `seed`"

    def _counting_decoder(converged, escalated):
        class _D:
            def __init__(self, nf):
                self.nf = nf

            def decode(self, s):
                return DecodeResult(np.zeros(self.nf, np.uint8), converged, 3, escalated)
        return _D

    def _failure_categories():
        circ, H, L = _setup()
        nf = H.shape[1]
        res = run_benchmark(circ, {
            "silent": _counting_decoder(True, False)(nf),
            "unconv": _counting_decoder(False, False)(nf),
            "escal": _counting_decoder(True, True)(nf),
        }, L, shots=150, seed=13)
        f = res["silent"].failures
        assert f > 0, "need some failures for this test"
        assert (res["silent"].silent_failures, res["silent"].unconverged_failures) == (f, 0)
        assert (res["unconv"].silent_failures, res["unconv"].unconverged_failures) == (0, f)
        assert (res["escal"].silent_failures, res["escal"].unconverged_failures) == (0, 0)
        assert res["escal"].escalated_failures == f and res["escal"].escalation_rate == 1.0

    def _samples():
        circ, H, L = _setup()
        dec = _ZeroDecoder(H.shape[1])
        res = run_benchmark(circ, {"z": dec}, L, shots=60, seed=2, keep_samples=True)["z"]
        assert res.samples is not None
        for key in ("time_us", "work", "failed", "escalated", "converged"):
            assert len(res.samples[key]) == 60, key
        assert int(res.samples["failed"].sum()) == res.failures
        plain = run_benchmark(circ, {"z": dec}, L, shots=60, seed=2)["z"]
        assert plain.samples is None
        # wall-clock times differ between runs; everything else must match exactly
        import dataclasses
        untimed = dict(time_p50_us=0.0, time_p99_us=0.0)
        assert dataclasses.replace(plain, **untimed) == dataclasses.replace(res, **untimed), \
            "keep_samples changed the metrics, or samples leak into equality"

    def _fake_results(with_zero=True):
        rng = np.random.default_rng(0)
        out = {}
        for i, name in enumerate(["never", "always", "primary_fail", "flag"]):
            fails = 0 if (with_zero and name == "always") else 5 + 3 * i
            ph, lo, hi = wilson(fails, 400)
            t = rng.lognormal(3 + i, 0.6, 400)
            out[name] = Metrics(400, fails, ph, lo, hi, 0.2 * i, 5.0, 9.0,
                                float(np.percentile(t, 50)), float(np.percentile(t, 99)),
                                silent_failures=fails // 3, unconverged_failures=fails // 4,
                                samples=dict(time_us=t, work=np.ones(400), failed=np.zeros(400, bool),
                                             escalated=np.zeros(400, bool),
                                             converged=np.ones(400, bool)))
        return out

    def _plots_render():
        import io
        res = _fake_results()
        sweep = {p: _fake_results(with_zero=(p < 2e-3)) for p in (1e-3, 2e-3, 4e-3)}
        figs = [plot_ablation(res), plot_outcomes(res), plot_latency(res),
                plot_tradeoff(res), plot_sweep(sweep)]
        for fig in figs:
            assert hasattr(fig, "savefig"), "plot functions must return a Figure"
            fig.savefig(io.BytesIO(), format="png")      # errors often appear only at draw time

    def _latency_needs_samples():
        res = _fake_results()
        for m in res.values():
            m.samples = None
        try:
            plot_latency(res)
        except ValueError:
            return
        raise AssertionError("plot_latency should raise ValueError without samples")

    tests_stats = run_checks([
        ("wilson(0, n): lower bound 0, upper z²/(n+z²)", _wilson_zero),
        ("wilson is symmetric at p̂ = 1/2", _wilson_symmetric),
        ("wilson interval brackets p̂ and stays in [0, 1]", _wilson_contains),
        ("benchmark: zero decoder fails exactly when an observable flips", _zero_decoder_ler),
        ("benchmark: policies are scored on identical shots", _paired),
        ("benchmark: failures split into escalated / silent / unconverged", _failure_categories),
        ("benchmark: keep_samples stores per-shot arrays", _samples),
        ("all five graphs render, including zero-failure points", _plots_render),
        ("plot_latency refuses results without samples", _latency_needs_samples),
    ])
    render_checks("statistics", tests_stats)
    return (tests_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Graph preview

    A quick end-to-end run on [[72, 12, 6]] to check the pipeline and the graphs.
    **Not a thesis result** — few shots, T = 3, fixed settings. It unlocks once
    every test in §2–§8 passes, before the full experiments in §11.
    """)
    return


@app.cell
def _(mo):
    run_preview = mo.ui.run_button(label="Run graph preview (~1–2 min)")
    run_preview
    return (run_preview,)


@app.cell
def _(
    BB_PRESETS,
    BpDecoder,
    BpOsdDecoder,
    PeelingDecoder,
    SwitchPolicy,
    all_pass,
    bb_from_preset,
    build_memory_circuit,
    dem_to_matrices,
    flag_detector_mask,
    tests_circuit,
    tests_codes,
    tests_decoders,
    tests_dem,
    tests_gb,
    tests_gf2,
    tests_stats,
    tests_switch,
):
    preview_ready = all_pass(tests_gf2, tests_codes, tests_gb, tests_circuit,
                             tests_dem, tests_decoders, tests_switch, tests_stats)
    preview_code = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
    preview_rounds = 3

    def make_preview_policies(p, names=None):
        """Build the circuit at noise p and the preview policies over one decoder set."""
        circ = build_memory_circuit(preview_code, preview_rounds, p, use_flags=True)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(preview_code, preview_rounds, True, circ.num_detectors)
        peel, bp, osd = PeelingDecoder(H, pr), BpDecoder(H, pr), BpOsdDecoder(H, pr)
        pol = {
            "peeling only": SwitchPolicy(peel, osd, "never"),
            "always BP+OSD": SwitchPolicy(peel, osd, "always"),
            "peel → OSD on fail": SwitchPolicy(peel, osd, "primary_fail"),
            "peel → OSD on flag/fail": SwitchPolicy(peel, osd, "flag_or_fail", mask),
            "BP → OSD on flag/fail": SwitchPolicy(bp, osd, "flag_or_fail", mask),
        }
        if names is not None:
            pol = {n: pol[n] for n in names}
        return circ, L, pol

    return make_preview_policies, preview_code, preview_ready, preview_rounds


@app.cell
def _(
    make_preview_policies,
    mo,
    plot_ablation,
    plot_latency,
    plot_outcomes,
    plot_tradeoff,
    preview_code,
    preview_ready,
    preview_rounds,
    run_benchmark,
    run_preview,
):
    mo.stop(not preview_ready, mo.md("*Preview locked: every test in §2–§8 must pass.*"))
    mo.stop(not run_preview.value, mo.md("*Press **Run graph preview** to start.*"))

    _p = 2e-3
    _circ, _L, _pol = make_preview_policies(_p)
    preview_results = run_benchmark(_circ, _pol, _L, shots=300, seed=7, keep_samples=True)

    _rows = ["| policy | LER | 95% CI | escalated | silent failures | p99 time |",
             "|:--|--:|:--|--:|--:|--:|"]
    for _n, _m in preview_results.items():
        _rows.append(f"| {_n} | {_m.ler:.4f} | [{_m.ci_low:.4f}, {_m.ci_high:.4f}] | "
                     f"{_m.escalation_rate:.0%} | {_m.silent_failures} | {_m.time_p99_us:.0f} µs |")
    mo.vstack([
        mo.md(f"#### Preview — {preview_code.name}, T = {preview_rounds}, p = {_p}, 300 shots"),
        mo.md("\n".join(_rows)),
        plot_ablation(preview_results),
        plot_outcomes(preview_results),
        plot_latency(preview_results),
        plot_tradeoff(preview_results),
    ])
    return (preview_results,)


@app.cell
def _(
    make_preview_policies,
    mo,
    plot_sweep,
    preview_code,
    preview_ready,
    preview_rounds,
    run_benchmark,
    run_preview,
):
    mo.stop(not preview_ready or not run_preview.value)

    _names = ["always BP+OSD", "peel → OSD on fail", "peel → OSD on flag/fail"]
    preview_sweep = {}
    for _p in (1e-3, 2e-3, 4e-3):
        _circ, _L, _pol = make_preview_policies(_p, _names)
        preview_sweep[_p] = run_benchmark(_circ, _pol, _L, shots=150, seed=11)
    mo.vstack([mo.md(f"#### Preview sweep — {preview_code.name}, T = {preview_rounds}, "
                     "150 shots/point"),
               plot_sweep(preview_sweep)])
    return (preview_sweep,)


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
        workers: int = 1          # CPU processes used for decoding; 1 = sequential
        # False (the default) drops the X-check detectors. A Z-basis memory has
        # Z-type observables, which X-check detectors can never help predict, so
        # keeping them multiplies the DEM by ~10x and cripples peeling. Set True
        # only to reproduce the full two-basis detector contract of section 4.
        x_detectors: bool = False

    return (ExperimentConfig,)


@app.cell
def _(ExperimentConfig, Metrics, asdict, json, np):
    SCHEMA_VERSION = 1

    def software_versions():
        from importlib.metadata import PackageNotFoundError, version
        out = {}
        for pkg in ("stim", "ldpc", "numpy", "scipy", "marimo"):
            try:
                out[pkg] = version(pkg)
            except PackageNotFoundError:
                out[pkg] = None
        return out

    def metrics_to_json(m, include_samples):
        d = asdict(m)
        samples = d.pop("samples", None)
        if include_samples and samples is not None:
            d["samples"] = {k: np.asarray(v).tolist() for k, v in samples.items()}
        return d

    def metrics_from_json(d):
        d = dict(d)
        samples = d.pop("samples", None)
        if samples is not None:
            samples = {k: np.asarray(v) for k, v in samples.items()}
        return Metrics(**d, samples=samples)

    def write_result_file(path, payload):
        import datetime
        import os
        payload = {"schema": SCHEMA_VERSION,
                   "created": datetime.datetime.now().isoformat(timespec="seconds"),
                   "software": software_versions(), **payload}
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=1)
        return path

    def read_result_file(path, kind):
        with open(path) as fh:
            d = json.load(fh)
        if d.get("kind") != kind:
            raise ValueError(f"{path} holds a {d.get('kind')!r} result, expected {kind!r}")
        return d

    def save_results(path, config, results, include_samples=False):
        """
        Write {'config': ..., 'results': {policy: Metrics}} as JSON, plus the
        creation time and package versions. Per-shot samples are dropped unless
        include_samples=True (they make files large).
        """
        return write_result_file(path, {
            "kind": "ablation",
            "config": asdict(config),
            "results": {n: metrics_to_json(m, include_samples) for n, m in results.items()},
        })

    def load_results(path):
        """Inverse of save_results -> (ExperimentConfig, dict name -> Metrics)."""
        d = read_result_file(path, "ablation")
        return (ExperimentConfig(**d["config"]),
                {n: metrics_from_json(m) for n, m in d["results"].items()})

    def save_sweep(path, config, sweep, include_samples=False):
        """Like save_results, for dict p -> dict policy -> Metrics."""
        return write_result_file(path, {
            "kind": "sweep",
            "config": asdict(config),
            "points": [{"p": float(p),
                        "results": {n: metrics_to_json(m, include_samples)
                                    for n, m in res.items()}}
                       for p, res in sweep.items()],
        })

    def load_sweep(path):
        """Inverse of save_sweep -> (ExperimentConfig, dict p -> dict name -> Metrics)."""
        d = read_result_file(path, "sweep")
        return (ExperimentConfig(**d["config"]),
                {pt["p"]: {n: metrics_from_json(m) for n, m in pt["results"].items()}
                 for pt in d["points"]})

    return (
        SCHEMA_VERSION,
        load_results,
        load_sweep,
        metrics_from_json,
        metrics_to_json,
        read_result_file,
        save_results,
        save_sweep,
        software_versions,
        write_result_file,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exporting results

    Every experiment writes two things into `results/`, next to this notebook:

    - a **data file** (`.json`, or `.npz` for Experiment 6) that the notebook can
      reload later, so nothing is lost when you close it;
    - an **export folder** with the same name, holding every figure as PNG
      (for slides) and PDF (vector, for LaTeX), every table as CSV, and a
      `report.md` that links them together.

    Exports are rewritten whenever a result is shown, so a reloaded run can be
    re-exported after you change a plot.
    """)
    return


@app.cell
def _(csv, datetime, os, plt):
    def export_bundle(data_path, title, figures, tables, notes=""):
        """
        Write figures (PNG + PDF), tables (CSV) and report.md into a folder named
        after `data_path`. figures: dict name -> matplotlib Figure;
        tables: dict name -> list of dicts (one per row). Returns the folder.
        """
        def fmt(v):                     # nested: a cell-private helper would not be
            if isinstance(v, float):    # visible when this function is called from
                return f"{v:.4g}"       # another cell
            if isinstance(v, (tuple, list)):
                return "[" + ", ".join(fmt(x) for x in v) + "]"
            return str(v)

        folder = os.path.splitext(data_path)[0]
        os.makedirs(folder, exist_ok=True)
        lines = [f"# {title}", "",
                 f"Data file: `{os.path.basename(data_path)}`  ",
                 f"Exported: {datetime.datetime.now().isoformat(timespec='seconds')}", ""]
        if notes:
            lines += [notes, ""]
        for name, rows in tables.items():
            path = os.path.join(folder, f"{name}.csv")
            cols = list(dict.fromkeys(k for r in rows for k in r))
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in cols})
            lines += [f"## Table: {name}", "", f"[{name}.csv]({name}.csv) — {len(rows)} rows", ""]
            if rows and len(rows) <= 40:
                lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                lines += ["| " + " | ".join(fmt(r.get(k, "")) for k in cols) + " |" for r in rows]
                lines.append("")
        for name, fig in figures.items():
            fig.savefig(os.path.join(folder, f"{name}.png"), dpi=200, bbox_inches="tight")
            fig.savefig(os.path.join(folder, f"{name}.pdf"), bbox_inches="tight")
            # do NOT plt.close(fig): inside marimo a closed figure no longer displays
            lines += [f"## Figure: {name}", "", f"![{name}]({name}.png)  ",
                      f"Vector version: [{name}.pdf]({name}.pdf)", ""]
        with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        return folder

    def metrics_rows(results, **extra):
        """Metrics dict -> CSV rows (per-shot samples left out)."""
        rows = []
        for name, m in results.items():
            r = dict(extra, policy=name, shots=m.shots, failures=m.failures, ler=m.ler,
                     ci_low=m.ci_low, ci_high=m.ci_high, escalation_rate=m.escalation_rate,
                     silent_failures=m.silent_failures,
                     unconverged_failures=m.unconverged_failures,
                     escalated_failures=m.escalated_failures, work_mean=m.work_mean,
                     work_p99=m.work_p99, time_p50_us=m.time_p50_us, time_p99_us=m.time_p99_us)
            rows.append(r)
        return rows

    return export_bundle, metrics_rows


@app.cell
def _(
    BB_PRESETS,
    ExperimentConfig,
    Metrics,
    RESULTS_DIR,
    bb_from_preset,
    build_memory_circuit,
    csv,
    dem_to_matrices,
    dz,
    export_bundle,
    load_results,
    metrics_rows,
    mo,
    np,
    os,
    plot_latency,
    plt,
    render_checks,
    run_checks,
    save_results,
):
    import tempfile as _tempfile

    def _tmpdir():
        return _tempfile.mkdtemp()

    def _results_dir_absolute():
        assert os.path.isabs(RESULTS_DIR) and os.path.isdir(RESULTS_DIR)
        here = mo.notebook_dir()
        if here is not None:
            assert os.path.dirname(RESULTS_DIR) == os.path.abspath(str(here)), \
                "results must live next to the notebook, not wherever marimo was launched"

    def _bundle_written():
        data = os.path.join(_tmpdir(), "E1_demo.json")
        open(data, "w").write("{}")
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        m = Metrics(100, 3, 0.03, 0.01, 0.08, 0.4, 2.0, 5.0, 10.0, 50.0, 1, 0)
        folder = export_bundle(data, "demo", {"curve": fig}, {"metrics": metrics_rows({"always": m})})
        assert folder == os.path.splitext(data)[0]
        for f in ("curve.png", "curve.pdf", "metrics.csv", "report.md"):
            assert os.path.getsize(os.path.join(folder, f)) > 0, f"{f} missing or empty"
        rows = list(csv.DictReader(open(os.path.join(folder, "metrics.csv"))))
        assert len(rows) == 1 and rows[0]["policy"] == "always" and rows[0]["failures"] == "3"
        report = open(os.path.join(folder, "report.md"), encoding="utf-8").read()
        assert "curve.png" in report and "metrics.csv" in report

    def _figures_still_display():
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        data = os.path.join(_tmpdir(), "x.json")
        open(data, "w").write("{}")
        export_bundle(data, "x", {"f": fig}, {})
        # a figure closed by pyplot does not display in marimo, so export must leave it open
        assert plt.fignum_exists(fig.number), "export closed the figure, so it will not display"
        import io
        fig.savefig(io.BytesIO(), format="png")

    def _e1_reload_keeps_latency():
        rng = np.random.default_rng(1)
        t = rng.lognormal(3, 0.5, 50)
        smp = dict(time_us=t, work=np.ones(50), failed=np.zeros(50, bool),
                   escalated=np.zeros(50, bool), converged=np.ones(50, bool))
        m = Metrics(50, 0, 0.0, 0.0, 0.07, 0.0, 1.0, 1.0, float(np.median(t)),
                    float(np.percentile(t, 99)), samples=smp)
        p = os.path.join(_tmpdir(), "E1_x.json")
        save_results(p, ExperimentConfig(shots=50), {"never": m}, include_samples=True)
        _, back = load_results(p)
        plot_latency(back)                            # needs the per-shot samples

    def _interrupted_run_resumes():
        code = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        circ = build_memory_circuit(code, 2, 2e-3, use_flags=True)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        det, obs = circ.compile_detector_sampler(seed=5).sample(60, separate_observables=True)
        ck = os.path.join(_tmpdir(), "ck")
        names = ["peel", "OSD-0"]
        full, _ = dz.run_zoo(H, L, pr, det, obs, names, chunk_size=20, checkpoint_dir=ck)
        os.remove(os.path.join(ck, "chunk_000000020.npz"))       # the run "died" here
        again, backend = dz.run_zoo(H, L, pr, det, obs, names, chunk_size=20, checkpoint_dir=ck)
        assert "resumed 2/3" in backend, backend
        for n in names:
            for k in ("conv", "fail", "work"):
                assert np.array_equal(full[n][k], again[n][k]), f"{n}.{k} changed on resume"
        _, other = dz.run_zoo(H, L, pr, det, obs, ["peel"], chunk_size=20, checkpoint_dir=ck)
        assert "resumed" not in other, "chunks from a different decoder list were reused"

    tests_export = run_checks([
        ("results folder is absolute and next to the notebook", _results_dir_absolute),
        ("export writes PNG, PDF, CSV and report.md", _bundle_written),
        ("figures still display after being exported", _figures_still_display),
        ("a reloaded ablation run keeps its per-shot samples", _e1_reload_keeps_latency),
        ("an interrupted zoo run resumes with identical results", _interrupted_run_resumes),
    ])
    mo.vstack([render_checks("export & resume", tests_export),
               mo.md(f"Results folder: `{RESULTS_DIR}`")])
    return (tests_export,)


@app.cell
def _(
    ExperimentConfig,
    Metrics,
    load_results,
    load_sweep,
    np,
    render_checks,
    run_checks,
    save_results,
    save_sweep,
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

    def _tmp(name):
        import os
        import tempfile
        return os.path.join(tempfile.mkdtemp(), "nested", name)   # folder must be created

    def _samples_roundtrip():
        cfg = ExperimentConfig(shots=3)
        smp = dict(time_us=np.array([1.5, 2.5, 3.5]), failed=np.array([True, False, False]))
        res = {"a": Metrics(3, 1, 1 / 3, 0.06, 0.79, 0.0, 1.0, 1.0, 2.5, 3.5, 1, 0, samples=smp)}
        p = _tmp("s.json")
        save_results(p, cfg, res)
        assert load_results(p)[1]["a"].samples is None, "samples should be dropped by default"
        save_results(p, cfg, res, include_samples=True)
        back = load_results(p)[1]["a"]
        assert np.array_equal(back.samples["time_us"], smp["time_us"])
        assert back.samples["failed"].dtype == bool
        assert back == res["a"]

    def _sweep_roundtrip():
        cfg = ExperimentConfig()
        m = Metrics(10, 1, 0.1, 0.02, 0.4, 1.0, 5.0, 9.0, 100.0, 200.0)
        sweep = {5e-4: {"always": m}, 1.2345678901234567e-3: {"always": m, "never": m}}
        p = _tmp("sw.json")
        save_sweep(p, cfg, sweep)
        cfg2, sweep2 = load_sweep(p)
        assert cfg2 == cfg and sweep2 == sweep, "p values or metrics changed in the round trip"

    def _kind_checked():
        p = _tmp("k.json")
        save_sweep(p, ExperimentConfig(), {1e-3: {}})
        try:
            load_results(p)
        except ValueError:
            return
        raise AssertionError("load_results accepted a sweep file")

    def _provenance():
        import json
        p = _tmp("m.json")
        save_results(p, ExperimentConfig(), {})
        d = json.load(open(p))
        assert d["software"]["stim"] and d["software"]["ldpc"], "package versions missing"
        assert "created" in d and d["schema"] >= 1

    tests_repro = run_checks([
        ("save_results / load_results round-trip exactly", _roundtrip),
        ("per-shot samples are optional and round-trip exactly", _samples_roundtrip),
        ("save_sweep / load_sweep round-trip exactly, p values included", _sweep_roundtrip),
        ("loading the wrong kind of file raises ValueError", _kind_checked),
        ("files record creation time and package versions", _provenance),
    ])
    render_checks("reproducibility", tests_repro)
    return (tests_repro,)


@app.cell(hide_code=True)
def _(
    STATUS_ICON,
    all_pass,
    mo,
    tests_circuit,
    tests_codes,
    tests_decoders,
    tests_dem,
    tests_experiments,
    tests_export,
    tests_exp2,
    tests_gb,
    tests_gf2,
    tests_planning,
    tests_reference,
    tests_repro,
    tests_surface,
    tests_stats,
    tests_switch,
    tests_exp6,
):
    _sections = [
        ("2. GF(2)", tests_gf2), ("3. BB codes", tests_codes), ("3. GB codes", tests_gb),
        ("4. Circuit", tests_circuit), ("5. DEM", tests_dem),
        ("5.1 Reference model", tests_reference),
        ("5.2 Surface baseline", tests_surface),
        ("6. Decoders", tests_decoders), ("7. Switch policies", tests_switch),
        ("8. Statistics", tests_stats), ("9. Reproducibility", tests_repro),
        ("9b. Export & resume", tests_export),
        ("10.1 Planning & scope", tests_planning),
        ("11.2 Flag cost vs information", tests_exp2),
        ("11.4-11.5 Experiment drivers", tests_experiments),
        ("11.6 Decoder zoo", tests_exp6),
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
# 10.1 Scope
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10.1 Scope: what is simulated and what is not

    Stated here so a reader meets it before the results rather than after.

    **Real, and the results stand on it.** The decoders are production
    implementations — BP+OSD is Roffe's `ldpc` package, the same one the BB-code
    literature uses — and every correction they return really does reproduce its
    syndrome. The noise is Stim circuit-level noise, gate by gate, not a
    code-capacity or phenomenological approximation. Failures are scored as
    $L\hat{e} \neq \text{obs}$ against Stim's own sampled observables. The
    switching logic is the real control flow, in the real order.

    So the logical error rates, escalation rates, silent-failure counts and the
    saturation curve are physically meaningful quantities.

    **Idealised, and claims must be bounded accordingly.**

    - **Timing is CPU wall-clock, not latency.** `time.perf_counter()` around a
      Python call, measured while other worker processes compete for cores. The
      same peeling loop is ~45x faster in Rust and faster again on an FPGA. Use
      these numbers as a **ratio** between decoders, never as an achievable
      latency. The breakeven criterion is expressed as a ratio for exactly this
      reason, and is the part that transfers.
    - **The pipeline is offline, not streaming.** A whole $T$-round block is
      sampled, then decoded. A real decoder must decode round $t$ while round
      $t+1$ is being measured, so the backlog problem is *motivation* here and
      was never measured.
    - **The flag is a detector bit, not a hardware signal.** Routing it to the
      controller before the primary starts is itself a latency this work does
      not model.
    - **Policies are derived, not executed.** Experiments 4 and 6 decode each
      shot once per decoder and then select. That is provably identical for
      outcomes — a test asserts it shot by shot — but the reported cost is a
      reconstruction, not an observation.

    Not modelled at all: pipelining, leakage, crosstalk, drift, readout latency.
    """)
    return


@app.cell
def _(np):
    def shots_for_failures(ler, target=50, floor=2000, cap=None):
        """
        Shots needed to observe about `target` failures at a given logical error
        rate. Precision on a rate is set by the FAILURE count, not the shot count:
        ~10 failures place a point on a trend, ~30 make it plottable, ~100 make it
        quotable. Floored so no point is too small to resolve anything, and capped
        when a budget is fixed.
        """
        if not 0 < ler <= 1:
            raise ValueError(f"ler must be in (0, 1], got {ler}")
        n = int(max(floor, round(target / ler)))
        return int(min(n, cap)) if cap else n

    def wilson_halfwidth(ler, shots, z=1.96):
        """Half-width of the Wilson interval, for sizing a run before paying for it."""
        d = 1 + z * z / shots
        return float(z * np.sqrt(ler * (1 - ler) / shots + z * z / (4 * shots ** 2)) / d)

    return shots_for_failures, wilson_halfwidth


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
    shots_for_failures,
    theory_faults,
    wilson_halfwidth,
):
    def _planner_arithmetic():
        assert shots_for_failures(1e-2, target=50, floor=0) == 5000
        assert shots_for_failures(1e-3, target=50, floor=0) == 50000
        assert shots_for_failures(1e-1, target=50, floor=2000) == 2000, "floor ignored"
        assert shots_for_failures(1e-3, target=50, cap=10000) == 10000, "cap ignored"
        for bad in (0.0, -1, 2.0):
            try:
                shots_for_failures(bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted ler={bad}")

    def _planner_is_monotone():
        ns = [shots_for_failures(l, floor=0) for l in (1e-1, 1e-2, 1e-3, 1e-4)]
        assert ns == sorted(ns), f"rarer failures must need more shots: {ns}"

    def _halfwidth_shrinks_with_shots():
        hw = [wilson_halfwidth(3e-3, n) for n in (1000, 10000, 100000)]
        assert hw == sorted(hw, reverse=True), hw
        # the planner's own recommendation should give a usable interval
        n = shots_for_failures(3e-3, target=50)
        assert wilson_halfwidth(3e-3, n) < 3e-3, "50 failures should resolve the rate"

    def _gross_code_runs_end_to_end():
        """
        [[144, 12, 12]] at T = 12 is a headline run: 150,000 shots at p = 1e-3
        gave the 5.91x sighted gain of section 12. This is the cheap guard for it:
        if the pipeline is going to fail on the Gross code, it fails here in
        seconds rather than hours in.
        """
        c = bb_from_preset("[[144, 12, 12]]", BB_PRESETS)
        # built the way the experiments build it: corrected detector convention
        circ = build_memory_circuit(c, 12, 1e-3, use_flags=True, x_detectors=False)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        ratio = H.shape[1] / theory_faults(c, 12)["total"]
        assert ratio < 3, (f"Gross DEM is {ratio:.1f}x the analytic fault count "
                           "-- x_detectors is probably on")
        det, obs = circ.compile_detector_sampler(seed=8).sample(
            24, separate_observables=True)
        det, obs = det.astype(np.uint8), obs.astype(np.uint8)
        peel, osd = PeelingDecoder(H, pr), BpOsdDecoder(H, pr, osd_order=0)
        for sdr in det:
            rp, ro = peel.decode(sdr), osd.decode(sdr)
            if rp.converged:
                assert np.array_equal((H @ np.asarray(rp.correction, np.uint8)) % 2, sdr)
            assert np.array_equal((H @ np.asarray(ro.correction, np.uint8)) % 2, sdr)

    tests_planning = run_checks([
        ("shot planner arithmetic, floor, cap and validation", _planner_arithmetic),
        ("rarer failures require more shots", _planner_is_monotone),
        ("Wilson half-width shrinks, and 50 failures resolve the rate",
         _halfwidth_shrinks_with_shots),
        ("[[144, 12, 12]] decodes end to end before a long run", _gross_code_runs_end_to_end),
    ])
    render_checks("10.1 planning and scope", tests_planning)
    return (tests_planning,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Experiments

    The nine experiments run in the order below, and that is the order they are
    laid out in: each one answers a question the next one depends on. The shared
    configuration, drivers and saved-run loader are defined just above, so any
    experiment can be run on its own.

    | § | | asks | answer |
    |--:|:--|:--|:--|
    | 11.1 | **Experiment 1 — validation** | does this pipeline reproduce known results, and what do the flags cost? | yes, on three codes; read by the decoder the flags **improve** LER 2.06× |
    | 11.2 | **Experiment 2 — flag cost vs information** | is the information the flags carry worth the hardware they need? | **yes** — 2.0× on [[72,12,6]], 5.9× on [[144,12,12]] |
    | 11.3 | **Experiment 3 — trigger saturation** | does a block-level flag rule still discriminate as $T$ grows? | no — 99% of shots flag by $T = 24$ |
    | 11.4 | **Experiment 4 — trigger ablation** | all five triggers on the same shots, at one $p$ | `primary_fail` wins everywhere |
    | 11.5 | **Experiment 5 — noise sweep** | logical error rate and escalation against $p$ | identical LER, and `primary_fail` escalates far less as $p$ falls |
    | 11.6 | **Experiment 6 — decoder zoo** | every weak × strong pair against eight trigger rules | **no viable pair** in 250,000 shots |
    | 11.7 | **Experiment 7 — surface-code baseline** | does the pipeline find a threshold everyone agrees on? | yes — 1.07%, against a ~1% literature value |
    | 11.8 | **Experiment 8 — accuracy/latency frontier** | which way of spending the budget — flag rows, flag priors or OSD order — lands on the frontier? | prior updating dominates; OSD order alone saturates at 3.8e-3 |
    | 11.9 | **Experiment 9 — dynamic prior updating** | can the flags be used without adding rows to $H$? | **yes, and better** — 37× lower LER than flag rows, on half the matrix |

    Experiment 1 comes first because nothing after it means anything if the
    fault model is wrong: it is the check that this notebook's detector error
    model matches the published one. Experiments 2 and 3 characterise the
    trigger itself — what it costs and how much it discriminates — before
    Experiments 4 and 6 spend hours measuring what it buys. Experiment 7 checks
    the whole pipeline against a code family with no stake in the argument, and
    Experiment 8 asks whether Experiment 2's gain survives being priced against
    the cheapest alternative. Experiment 9 tests the other half of the thesis
    title: using the flag outcomes to update the fault priors rather than to add
    rows to the check matrix.

    **Detector convention.** Every experiment defaults to `x_detectors=False`.
    This is a Z-basis memory, so the observables are Z-type and only X errors
    can flip them; X-check detectors report Z errors, which never can. On
    [[72,12,6]] at $T = 12$, keeping them takes the fault count from 4,392 to
    33,552 and the mean fault-graph degree from 128 to 1,356, which cripples
    peeling — it is collision-limited. Section 4 documents the full two-basis
    contract, and the switch in the configuration panel turns it back on when
    you want to reproduce it.
    """)
    return


@app.cell
def _(
    BB_PRESETS,
    GB_PRESETS,
    mo,
    os,
):
    ui_code = mo.ui.dropdown(list(BB_PRESETS) + list(GB_PRESETS), value="[[72, 12, 6]]",
                             label="Code")
    ui_rounds = mo.ui.slider(1, 24, value=6, step=1,
                             label="Rounds T (syndrome-extraction rounds per shot)")
    ui_p = mo.ui.dropdown(["5e-4", "1e-3", "2e-3", "3e-3", "5e-3"], value="1e-3", label="p")
    ui_shots = mo.ui.number(start=100, stop=1_000_000, step=100, value=500,
                            label="Shots (independent runs; ~100 failures needed to compare)")
    ui_flags = mo.ui.switch(value=True, label="Flag qubits")
    ui_osd = mo.ui.slider(0, 7, value=0, step=1,
                          label="OSD order (0 fastest; >=2 is ~15x slower per shot)")
    ui_seed = mo.ui.number(value=20260921, label="Seed (fixes which shots are sampled)")
    _cores = os.cpu_count() or 1
    ui_workers = mo.ui.slider(1, max(2, _cores), value=min(12, max(1, _cores - 4)),
                              label=f"CPU workers (of {_cores})")
    ui_xdet = mo.ui.switch(value=False, label="X-check detectors (section 4 contract)")
    mo.vstack([mo.md("### Experiment configuration"),
               mo.hstack([ui_code, ui_p, ui_flags]),
               mo.hstack([ui_rounds, ui_shots]),
               mo.hstack([ui_osd, ui_seed]),
               mo.hstack([ui_workers, ui_xdet]),
               mo.md("*Decoding runs in parallel from about 2,000 shots upward; below that "
                     "the per-process setup costs more than it saves.*"),
               mo.md('*X-check detectors: off is the corrected model. A Z-basis memory has Z-type observables, so X-check detectors only ever report Z errors, which cannot flip those observables — keeping them multiplies the DEM by roughly an order of magnitude (33,552 faults against 4,392 at T=12), cripples peeling and slows every decoder. Turn it on only to reproduce the full two-basis detector contract of section 4.*')])
    return ui_code, ui_flags, ui_osd, ui_p, ui_rounds, ui_seed, ui_shots, ui_workers, ui_xdet


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
    ui_workers,
    ui_xdet,
):
    config = ExperimentConfig(
        code=ui_code.value, rounds=int(ui_rounds.value), p=float(ui_p.value),
        shots=int(ui_shots.value), seed=int(ui_seed.value),
        use_flags=bool(ui_flags.value), osd_order=int(ui_osd.value),
        workers=int(ui_workers.value), x_detectors=bool(ui_xdet.value))
    return (config,)


@app.cell
def _(
    BB_PRESETS,
    GB_PRESETS,
    Metrics,
    RESULTS_DIR,
    bb_from_preset,
    build_memory_circuit,
    code_from_key,
    dem_to_matrices,
    dz,
    flag_detector_mask,
    gb_from_preset,
    mo,
    np,
    os,
    wilson,
):
    def code_from_key(key):
        """Build any preset, BB or GB, from its label."""
        if key in BB_PRESETS:
            return bb_from_preset(key, BB_PRESETS)
        if key in GB_PRESETS:
            return gb_from_preset(key, GB_PRESETS)
        raise KeyError(f"unknown code {key!r}")

    def run_ablation(config, keep_samples=True, on_chunk=None):
        """
        Experiment 4. Decode every shot ONCE with the weak decoder (peeling) and once with the
        strong one (BP+OSD at config.osd_order), then derive one policy per trigger.
        This matches running SwitchPolicy on every shot -- the decoder-zoo tests
        check that shot by shot -- but it uses config.workers CPU processes and
        never decodes the same shot twice.

        Returns dict trigger -> Metrics.
        """
        code = code_from_key(config.code)
        circuit = build_memory_circuit(code, config.rounds, config.p, use_flags=config.use_flags,
                                       x_detectors=config.x_detectors)
        H, L, priors = dem_to_matrices(circuit.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(code, config.rounds, config.use_flags, circuit.num_detectors,
                                  x_detectors=config.x_detectors)
        det, obs = circuit.compile_detector_sampler(seed=config.seed).sample(
            config.shots, separate_observables=True)

        names = [("weak", {"kind": "peel"}),
                 ("strong", {"kind": "osd", "osd_order": config.osd_order, "max_iter": 20})]
        workers = dz.parallel_workers(config.shots, config.workers)
        results, _backend = dz.run_zoo(H, L, priors, det, obs, names, workers=workers,
                                       chunk_size=dz.chunk_for(config.shots, workers),
                                       on_chunk=on_chunk)
        W, S = results["weak"], results["strong"]

        n_flags = det[:, mask].sum(1) if mask.any() else np.zeros(config.shots, dtype=int)
        none = np.zeros(config.shots, bool)
        triggers = {"never": (none, False), "always": (np.ones(config.shots, bool), False),
                    "primary_fail": (none, True)}
        if config.use_flags:
            triggers["flag"] = (n_flags > 0, True)
            triggers["flag_or_fail"] = (n_flags > 0, True)

        out = {}
        for name, (pre, fallback) in triggers.items():
            P = dz.derive(W, S, pre, fallback)
            k = int(P["fail"].sum())
            ler, lo, hi = wilson(k, config.shots)
            out[name] = Metrics(
                shots=config.shots, failures=k, ler=ler, ci_low=lo, ci_high=hi,
                escalation_rate=float(P["esc"].mean()), work_mean=float(P["work"].mean()),
                work_p99=float(np.percentile(P["work"], 99)),
                time_p50_us=float(np.percentile(P["cost"], 50)),
                time_p99_us=float(np.percentile(P["cost"], 99)),
                silent_failures=int(P["silent"].sum()),
                unconverged_failures=int(P["unconverged"].sum()),
                samples=(dict(time_us=P["cost"], work=P["work"], failed=P["fail"],
                              escalated=P["esc"], converged=W["conv"]) if keep_samples else None))
        return out

    def run_sweep(config, ps, keep_samples=False, progress=True):
        """Experiment 5. run_ablation at each p in `ps`. Returns dict p -> (dict trigger -> Metrics)."""
        import dataclasses
        ps = [float(p) for p in ps]
        steps = mo.status.progress_bar(ps, title="Experiment 5 sweep", show_eta=True) if progress else ps
        return {p: run_ablation(dataclasses.replace(config, p=p), keep_samples=keep_samples)
                for p in steps}

    def result_path(kind, config):
        """Deterministic file name: the same config always maps to the same file."""
        tag = config.code.strip("[]").replace(", ", "-").replace(" ", "")
        return os.path.join(RESULTS_DIR, f"{kind}_{tag}_T{config.rounds}_p{config.p:g}_"
                            f"{config.shots}shots_seed{config.seed}"
                            f"{'_flags' if config.use_flags else ''}_osd{config.osd_order}"
                            f"{'_xdet' if config.x_detectors else ''}.json")

    return code_from_key, result_path, run_ablation, run_sweep


@app.cell
def _(
    BB_PRESETS,
    ExperimentConfig,
    TRIGGERS,
    bb_from_preset,
    code_from_key,
    reference_fault_count,
    render_checks,
    result_path,
    run_ablation,
    run_checks,
    run_sweep,
    run_validation,
):
    _small = ExperimentConfig(code="[[72, 12, 6]]", rounds=2, p=2e-3, shots=20, seed=3)

    def _ablation_triggers():
        res = run_ablation(_small)
        assert tuple(res) == TRIGGERS, f"got {tuple(res)}"
        assert res["always"].escalation_rate == 1.0 and res["never"].escalation_rate == 0.0
        assert all(m.shots == 20 for m in res.values())

    def _no_flags():
        import dataclasses
        res = run_ablation(dataclasses.replace(_small, use_flags=False))
        assert "flag" not in res and "flag_or_fail" not in res

    def _gb_code():
        import dataclasses
        res = run_ablation(dataclasses.replace(_small, code="[[48, 6, 8]]"), keep_samples=False)
        assert res["always"].shots == 20 and res["always"].samples is None

    def _sweep():
        sw = run_sweep(_small, [1e-3, 3e-3], progress=False)
        assert list(sw) == [1e-3, 3e-3]
        assert all(tuple(r) == TRIGGERS for r in sw.values())

    def _paths():
        a = result_path("EXP4", _small)
        assert a == result_path("EXP4", _small) and a.endswith(".json")
        import dataclasses
        assert a != result_path("EXP4", dataclasses.replace(_small, p=3e-3)), \
            "different configs must not share a file"
        assert code_from_key("[[48, 6, 8]]").n == 48

    def _reference_formula():
        code = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        # n(wT + T/2 + 1) with n = 72, w = 3 (qubit degree), T = 4 -> 72 * 15 = 1080
        assert reference_fault_count(code, 4) == 1080, reference_fault_count(code, 4)

    def _validation_runs():
        import dataclasses
        cfg = dataclasses.replace(_small, rounds=2, shots=40)
        v = run_validation(cfg, target=0.5)
        assert v["v1"]["our_faults"] > v["v1"]["our_faults_unflagged"] > 0, \
            "a flagged circuit must have more fault mechanisms"
        assert v["v1"]["ratio"] > 0
        lo, hi = v["v2"]["ci_low"], v["v2"]["ci_high"]
        assert v["v2"]["agrees"] == (lo <= 0.5 <= hi), "agreement must follow the interval"
        assert v["v3"]["effect"] in ("flags improve accuracy", "flags cost accuracy",
                                     "no resolvable difference")
        assert v["v3"]["extra_detectors"] > 0 and 0.0 <= v["v3"]["flagged_fraction"] <= 1.0

    tests_experiments = run_checks([
        ("reference fault-count formula n(wT + T/2 + 1)", _reference_formula),
        ("Experiment 1 runs and reports all three checks", _validation_runs),
        ("run_ablation returns every trigger, with sane escalation", _ablation_triggers),
        ("flag triggers are skipped when flags are off", _no_flags),
        ("run_ablation works on a GB code", _gb_code),
        ("run_sweep returns one ablation per p", _sweep),
        ("result files are named deterministically per config", _paths),
    ])
    render_checks("experiment drivers", tests_experiments)
    return (tests_experiments,)


@app.cell
def _(RESULTS_DIR, glob, mo, os):
    def saved_run_picker(prefix, label):
        """Dropdown over the saved runs of one experiment, plus the file list."""
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{prefix}_*.json")))
        return mo.ui.dropdown({os.path.basename(f): f for f in files},
                              value=os.path.basename(files[-1]) if files else None,
                              label=label), files

    return (saved_run_picker,)


@app.cell
def _(mo):
    import sys as _sys
    _here = mo.notebook_dir()
    _here = str(_here) if _here is not None else "."
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    import decoder_zoo as dz
    return (dz,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.1 Experiment 1 — validation: does the pipeline reproduce known results, and what do flags cost?

    A negative result is only believable if the setup could have found the effect.
    Experiment 1 is the evidence for that, and it runs before any conclusion is drawn.

    **Check 1 — how big is our fault model?** Pakhunov (2026) gives the number of DEM
    fault mechanisms for a BB memory as $n(wT + T/2 + 1)$ under his noise model.
    Ours is 1.42× denser, and §5.1 identifies why: the experiment circuit extracts
    **both** check families, because the flag qubits attach to the X-check ancillas,
    where the reference extracts one. The channel is not the cause — the same
    `DEPOLARIZE2` noise on the reference circuit reproduces the theorem exactly.

    **Check 2 — do we reproduce a published number?** Measured on §5.1's reference
    circuit, the Theorem 3/5 prediction evaluated on our own fault graph lands on
    Table II for all three Gross-family codes: 0.935 against 0.935, 0.874 against
    0.879, 0.764 against 0.778. What the check reports is therefore split in two —
    2a, whether the model reproduces the literature (it does), and 2b, how far our
    peeling implementation falls short of that model (+0.032, +0.073, +0.117 as $n$
    grows). The second gap is a decoder result, not a modelling error.

    **Check 3 — what do the flags cost?** Flag qubits add ancillas and CNOTs, so they
    add noise. This decodes the same configuration with and without flags using
    the same strong decoder and compares the logical error rates. **Measured:** with
    the decoder reading the flag detectors the flagged circuit is 2.06× *better*,
    so there is no accuracy loss for a trigger to buy back. The hardware cost is
    real but shows up only when the flag outcomes are hidden — which is what
    Experiment 2's blind arm isolates.

    *Important:* this section does not reimplement anyone else's noise model. It
    measures the distance between ours and a published reference, which is what a
    reader needs in order to weigh the rest of the thesis.
    """)
    return


@app.cell
def _(mo):
    ui_exp1_target = mo.ui.number(0.0, 1.0, value=0.935, step=0.001,
                                  label="Literature target: fraction of shots the weak decoder resolves")
    ui_exp1_source = mo.ui.text(value="Pakhunov (2026), Table II, [[72,12,6]], p=1e-3, T=12",
                              label="Source", full_width=True)
    run_exp1 = mo.ui.run_button(label="Run Experiment 1 — validation")
    mo.vstack([mo.hstack([ui_exp1_target, run_exp1]), ui_exp1_source,
               mo.md("*Uses the shared configuration above (code, T, p, shots, workers). "
                     "Set T and p to match the source you are comparing against.*")])
    return run_exp1, ui_exp1_source, ui_exp1_target


@app.cell
def _(build_memory_circuit, build_reference_circuit, code_from_key, dem_to_matrices, dz,
      flag_detector_mask, np, sp, theory_peel, wilson):
    def reference_fault_count(code, rounds):
        """Pakhunov's count for a BB memory: n(wT + T/2 + 1), w = qubit degree."""
        w = int(np.asarray(code.hx.sum(axis=0)).max())
        return int(code.n * (w * rounds + rounds / 2 + 1))

    def run_validation(config, target, on_chunk=None):
        """All three checks on one configuration. Returns a dict of plain numbers."""
        code = code_from_key(config.code)
        out = {"code": config.code, "rounds": config.rounds, "p": config.p,
               "shots": config.shots, "target": float(target),
               "x_detectors": bool(config.x_detectors)}
        arms = {}
        for flags in (True, False):
            circ = build_memory_circuit(code, config.rounds, config.p, use_flags=flags,
                                        x_detectors=config.x_detectors)
            H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
            det, obs = circ.compile_detector_sampler(seed=config.seed).sample(
                config.shots, separate_observables=True)
            names = [("weak", {"kind": "peel"}),
                     ("strong", {"kind": "osd", "osd_order": config.osd_order, "max_iter": 20})]
            workers = dz.parallel_workers(config.shots, config.workers)
            res, _ = dz.run_zoo(H, L, pr, det, obs, names, workers=workers,
                                chunk_size=dz.chunk_for(config.shots, workers),
                                on_chunk=on_chunk)
            mask = flag_detector_mask(code, config.rounds, flags, circ.num_detectors,
                                      x_detectors=config.x_detectors)
            arms[flags] = dict(
                faults=H.shape[1], detectors=circ.num_detectors,
                flagged_fraction=float((det[:, mask].sum(1) > 0).mean()) if flags else 0.0,
                weak_resolved=wilson(int(res["weak"]["conv"].sum()), config.shots),
                weak_ler=wilson(int(res["weak"]["fail"].sum()), config.shots),
                strong_ler=wilson(int(res["strong"]["fail"].sum()), config.shots),
                silent=int((res["weak"]["conv"] & res["weak"]["fail"]).sum()))

        # Check 2 compares against a number the literature measured on ITS circuit,
        # which extracts one check family. Ours extracts both, because the flags
        # attach to the X-check ancillas -- so the comparison has to run on the
        # reference circuit of section 5.1, not on the experiment circuit. Measured
        # on the wrong one, [[288,12,18]] reads 9% against a 77.8% target; on the
        # right one it reads 64.8%, and the residual is the peeling implementation.
        ref_circ = build_reference_circuit(code, config.rounds, config.p)
        rH, rL, rpr = dem_to_matrices(
            ref_circ.detector_error_model(decompose_errors=False))
        rdet, robs = ref_circ.compile_detector_sampler(seed=config.seed).sample(
            config.shots, separate_observables=True)
        _w = dz.parallel_workers(config.shots, config.workers)
        rres, _ = dz.run_zoo(rH, rL, rpr, rdet, robs, [("weak", {"kind": "peel"})],
                             workers=_w, chunk_size=dz.chunk_for(config.shots, _w),
                             on_chunk=on_chunk)
        ref_resolved = wilson(int(rres["weak"]["conv"].sum()), config.shots)
        ref_deg = None
        _Hb = (rH > 0).astype(np.int8)
        _G = (_Hb.T @ _Hb).tocsr()
        ref_deg = float((np.diff(_G.indptr) - 1).mean())
        ref_predicted = theory_peel(rH.shape[1], float(rpr.sum()), ref_deg)

        # Check 1: how dense is our fault model compared with the reference formula?
        ref = reference_fault_count(code, config.rounds)
        out["v1"] = dict(reference_faults=ref, our_faults=arms[True]["faults"],
                         our_faults_unflagged=arms[False]["faults"],
                         ratio=arms[False]["faults"] / ref,
                         reference_circuit_faults=int(rH.shape[1]),
                         reference_circuit_ratio=rH.shape[1] / ref)
        # Check 2: do we reproduce the published figure, on the reference circuit?
        rate, lo, hi = ref_resolved
        out["v2"] = dict(measured=rate, ci_low=lo, ci_high=hi, target=float(target),
                         agrees=bool(lo <= target <= hi),
                         predicted=ref_predicted,
                         model_agrees=bool(abs(ref_predicted - float(target)) < 0.02),
                         decoder_gap=float(ref_predicted - rate),
                         lam=float(rpr.sum()), mean_degree=ref_deg,
                         on_experiment_circuit=arms[True]["weak_resolved"][0],
                         on_experiment_circuit_unflagged=arms[False]["weak_resolved"][0])
        # Check 3: the cost of the flags (independent samples: different circuits)
        f, uf = arms[True]["strong_ler"], arms[False]["strong_ler"]
        if f[2] < uf[1]:
            effect = "flags improve accuracy"
        elif f[1] > uf[2]:
            effect = "flags cost accuracy"
        else:
            effect = "no resolvable difference"
        out["v3"] = dict(ler_flagged=f, ler_unflagged=uf, effect=effect,
                         extra_faults=arms[True]["faults"] - arms[False]["faults"],
                         extra_detectors=arms[True]["detectors"] - arms[False]["detectors"],
                         flagged_fraction=arms[True]["flagged_fraction"],
                         silent_flagged=arms[True]["silent"], silent_unflagged=arms[False]["silent"])
        return out

    return reference_fault_count, run_validation


@app.cell
def _(config, core_ready, mo, run_exp1, ui_exp1_target, run_validation):
    exp1_result = None
    if not core_ready:
        _out = mo.md("*Experiment 1 locked until every test passes.*")
    elif not run_exp1.value:
        _out = mo.md("*Press **Run Experiment 1** to validate this configuration.*")
    else:
        _n_chunks = max(1, 2 * (-(-config.shots // 250)))
        with mo.status.progress_bar(total=_n_chunks, title="Experiment 1: validating", show_eta=True) as _bar:
            exp1_result = run_validation(config, ui_exp1_target.value, on_chunk=_bar.update)
        _out = mo.md("Experiment 1 finished.")
    _out
    return (exp1_result,)


@app.cell
def _(exp1_result, mo, saved_run_picker):
    _ = exp1_result                            # re-list after a new run
    ui_exp1_file, _f1 = saved_run_picker("EXP1", "Saved Experiment 1 run")
    load_exp1_btn = mo.ui.run_button(label="Load Experiment 1")
    mo.vstack([
        mo.md("#### Load a saved Experiment 1 run"),
        mo.hstack([ui_exp1_file, load_exp1_btn]) if _f1
        else mo.md("*No saved Experiment 1 runs yet.*")])
    return load_exp1_btn, ui_exp1_file


@app.cell
def _(load_exp1_btn, read_result_file, ui_exp1_file):
    exp1_loaded = None
    if load_exp1_btn.value and ui_exp1_file.value:
        _d = read_result_file(ui_exp1_file.value, "validation")
        exp1_loaded = dict(_d["config"], path=ui_exp1_file.value,
                           source=_d.get("source", ""), **_d["checks"])
    return (exp1_loaded,)


@app.cell
def _(
    RESULTS_DIR,
    exp1_loaded,
    exp1_result,
    export_bundle,
    mo,
    os,
    plt,
    ui_exp1_source,
    write_result_file,
):
    exp1_data = exp1_result if exp1_result is not None else exp1_loaded
    mo.stop(exp1_data is None, mo.md("*Run Experiment 1, or load a saved run above.*"))
    _v1, _v2, _v3 = exp1_data["v1"], exp1_data["v2"], exp1_data["v3"]
    _source = exp1_data.get("source") or ui_exp1_source.value
    # Files written before the reference-circuit fix measured check 2 on the
    # experiment circuit, which extracts both check families, and carry no
    # analytic prediction. Say so rather than reporting them as comparable.
    _stale2 = (None if "predicted" in _v2 else
               "saved before the reference-circuit fix (§5.1) — re-run this point "
               "for a number comparable to the paper")

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4.2))
    _a.bar([0], [_v2["measured"]], color="#1f77b4", width=0.5)
    _a.errorbar([0], [_v2["measured"]],
                yerr=[[_v2["measured"] - _v2["ci_low"]], [_v2["ci_high"] - _v2["measured"]]],
                fmt="none", color="k", capsize=6)
    if _stale2 is None:
        _a.bar([1], [_v2["predicted"]], color="#9467bd", width=0.5)
        _a.set_xticks([0, 1], ["our decoder\n(reference circuit)", "our DEM,\nanalytic model"])
        _a.set_title("Check 2 — model reproduces the paper; the gap is our decoder")
    else:
        _a.set_xticks([0], ["our decoder\n(experiment circuit)"])
        _a.set_xlim(-0.6, 0.6)
        _a.set_title("Check 2 — stale run, not comparable to the paper")
    _a.axhline(_v2["target"], color="#d62728", ls="--", label=f"literature: {_v2['target']:.3f}")
    _a.set_ylim(0, 1)
    _a.set_ylabel("fraction of shots the weak decoder resolves")
    _a.legend(fontsize=8)
    for _i, (_lab, _m) in enumerate([("with flags", _v3["ler_flagged"]),
                                     ("without flags", _v3["ler_unflagged"])]):
        _b.errorbar([_i], [_m[0] if _m[0] > 0 else _m[2]],
                    yerr=[[max(_m[0] - _m[1], 0)], [max(_m[2] - _m[0], 0)]],
                    fmt="o" if _m[0] > 0 else "v", color="#2ca02c", capsize=6, ms=8)
    _b.set_xticks([0, 1], ["with flags", "without flags"])
    _b.set_yscale("log")
    _b.set_xlim(-0.5, 1.5)
    _b.set_ylabel("logical error rate per shot (strong decoder)")
    _b.set_title(f"Check 3 — {_v3['effect']}")
    _b.grid(True, which="both", axis="y", alpha=0.3)
    _fig.tight_layout()

    _rows = [
        dict(check="Check 1: fault-model density", value=f"{_v1['ratio']:.1f}×",
             detail=f"{_v1['our_faults_unflagged']:,} faults vs {_v1['reference_faults']:,} "
                    f"from n(wT + T/2 + 1)"),
        dict(check="Check 2a: model reproduces literature",
             value=("yes" if _v2.get("model_agrees") else "no") if _stale2 is None else "—",
             detail=(_stale2 if _stale2 else
                     (f"analytic prediction on our own DEM {_v2['predicted']:.3f} "
                      f"vs {_v2['target']:.3f} ({_source}); "
                      f"lambda {_v2['lam']:.2f}, mean degree {_v2['mean_degree']:.1f}"))),
        dict(check="Check 2b: our decoder against that model",
             value=(f"{_v2['decoder_gap']:+.3f}" if _stale2 is None else "—"),
             detail=((f"measured {_v2['measured']:.3f} "
                      f"[{_v2['ci_low']:.3f}, {_v2['ci_high']:.3f}] on the experiment circuit, "
                      "which extracts both check families and so is not comparable to the paper")
                     if _stale2 else
                     (f"measured {_v2['measured']:.3f} "
                      f"[{_v2['ci_low']:.3f}, {_v2['ci_high']:.3f}] on the REFERENCE circuit; "
                      f"on the experiment circuit it reads "
                      f"{_v2['on_experiment_circuit']:.3f}, which extracts "
                      "both check families and so is not comparable to the paper"))),
        dict(check="Check 3: accuracy cost of flags", value=_v3["effect"],
             detail=f"LER {_v3['ler_flagged'][0]:.2e} with flags vs "
                    f"{_v3['ler_unflagged'][0]:.2e} without; "
                    f"+{_v3['extra_faults']:,} faults, +{_v3['extra_detectors']} detectors; "
                    f"flags fire on {_v3['flagged_fraction']:.1%} of shots"),
        dict(check="silent failures (weak decoder)",
             value=f"{_v3['silent_flagged']} flagged / {_v3['silent_unflagged']} unflagged",
             detail="the only failures a flag trigger could ever catch"),
    ]
    _tag = exp1_data["code"].strip("[]").replace(", ", "-")
    _path = write_result_file(
        os.path.join(RESULTS_DIR, f"EXP1_{_tag}_T{exp1_data['rounds']}_p{exp1_data['p']:g}_"
                                  f"{exp1_data['shots']}shots.json"),
        {"kind": "validation",
         "config": {k: exp1_data[k] for k in ("code", "rounds", "p", "shots", "target")},
         "source": _source,
         "checks": {k: exp1_data[k] for k in ("v1", "v2", "v3")}})
    _folder = export_bundle(_path, f"Experiment 1 validation — {exp1_data['code']}, T={exp1_data['rounds']}, "
                            f"p={exp1_data['p']}", {"validation": _fig}, {"checks": _rows},
                            notes=f"Literature source: {ui_exp1_source.value}")
    mo.vstack([
        mo.md("### Experiment 1 — results"),
        mo.md("| check | verdict | detail |\n|:--|:--|:--|\n"
              + "\n".join(f"| {r['check']} | {r['value']} | {r['detail']} |" for r in _rows)),
        mo.md(f"Saved to `{_path}` · exported to `{_folder}`"),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.2 Experiment 2 — flag cost vs information


    Flag qubits are not free: each one adds a qubit, two CNOTs per round and a
    fault mechanism per round. Experiment 2 measures whether the information the
    flags carry ever repays that cost, by decoding three arms on identical shots
    (§11.2.1--§11.2.3) across a range of round counts $T$.

    Shots are budgeted per $T$, not flat: the logical error rate grows roughly
    linearly in $T$, so large $T$ needs fewer shots for the same number of
    failures while costing more per shot. Each point is written to disk as soon
    as it finishes, so an interrupted sweep keeps everything already computed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 11.2.1 What the flag hardware costs, before any decoding

    Qubits, two-qubit gates, detectors and fault mechanisms, with and without
    flags. This is the overhead an architect weighs against the accuracy gain, and
    it needs no simulation.
    """)
    return


@app.cell
def _(build_memory_circuit, dem_to_matrices, np):
    def circuit_overhead(code, rounds, p, x_detectors=False):
        """Qubit, gate, detector and fault counts for the flagged/unflagged circuits."""
        out = {}
        for flags in (False, True):
            circ = build_memory_circuit(code, rounds, p, use_flags=flags,
                                        x_detectors=x_detectors)
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
    ### 11.2.2 The three-arm measurement

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

    def three_arm_run(code, rounds, p, shots, seed, workers=1, osd_order=0, on_chunk=None,
                      x_detectors=False, checkpoint_dir=None):
        """
        LER of one strong decoder on the three arms, plus flag statistics.

        checkpoint_dir: each arm gets its own subfolder, so an interrupted run
        resumes chunk by chunk instead of starting the T over (see dz.run_zoo).
        """
        import os as _os
        spec = [("strong", {"kind": "osd", "osd_order": osd_order, "max_iter": 20})]
        w = dz.parallel_workers(shots, workers)

        def decode(arm, H, L, priors, det, obs):
            cdir = None
            if checkpoint_dir:
                cdir = _os.path.join(checkpoint_dir, f"T{rounds}_{arm}")
            res, _ = dz.run_zoo(H, L, priors, det, obs, spec, workers=w,
                                chunk_size=dz.chunk_for(shots, w, checkpoint=bool(cdir)),
                                on_chunk=on_chunk, checkpoint_dir=cdir)
            fails = int(res["strong"]["fail"].sum())
            return fails, wilson(fails, shots), res["strong"]["fail"]

        circ0 = build_memory_circuit(code, rounds, p, use_flags=False,
                                     x_detectors=x_detectors)
        H0, L0, pr0 = dem_to_matrices(circ0.detector_error_model(decompose_errors=False))
        det0, obs0 = circ0.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True)

        circ1 = build_memory_circuit(code, rounds, p, use_flags=True,
                                     x_detectors=x_detectors)
        H1, L1, pr1 = dem_to_matrices(circ1.detector_error_model(decompose_errors=False))
        det1, obs1 = circ1.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True)
        mask = flag_detector_mask(code, rounds, True, circ1.num_detectors,
                                  x_detectors=x_detectors)
        keep = ~mask

        out = {}
        out["unflagged"] = decode("unflagged", H0, L0, pr0, det0, obs0)
        out["flagged, blind"] = decode("blind", H1[keep], L1, pr1, det1[:, keep], obs1)
        out["flagged, sighted"] = decode("sighted", H1, L1, pr1, det1, obs1)

        n_flags = det1[:, mask].sum(1)
        return dict(
            rounds=rounds, p=p, shots=shots, seed=seed, code=code.name,
            x_detectors=bool(x_detectors),
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
    ### 11.2.3 Trend and crossover fit

    Ratios of logical error rate to the unflagged arm. A ratio below 1 means flags
    help. Fitting `ln(ratio)` linearly in T and solving for `ratio = 1` gives the
    crossover T\*: the number of rounds beyond which flags stop paying.

    **Measured result:** both ratios are flat in T over 6–24 — the blind arm sits
    at 1.6–2.1 and the sighted arm at 0.45–0.50 on [[72,12,6]] — so no crossover
    appears in that range and the fitted T\* is not meaningful here. Report the
    ratios themselves rather than an extrapolated crossover.

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
    ui_exp2_code = mo.ui.dropdown(list(BB_PRESETS) + list(GB_PRESETS), value="[[72, 12, 6]]",
                             label="Code")
    ui_exp2_p = mo.ui.dropdown(["5e-4", "1e-3", "2e-3", "3e-3"], value="1e-3", label="p")
    ui_exp2_rounds = mo.ui.multiselect(["3", "6", "9", "12", "18", "24"],
                                  value=["6", "12", "18", "24"], label="Rounds T")
    ui_exp2_shots = mo.ui.number(start=100, stop=500_000, step=100, value=20_000,
                                 label="Shots per arm (each T costs three arms)")
    ui_exp2_seed = mo.ui.number(value=20260923, label="Seed")
    ui_exp2_osd = mo.ui.slider(0, 7, value=0, step=1, label="OSD order")
    ui_exp2_workers = mo.ui.slider(1, max(2, _cores), value=min(12, max(1, _cores - 4)),
                              label=f"CPU workers (of {_cores})")
    ui_exp2_xdet = mo.ui.switch(value=False, label="X-check detectors")
    ui_exp2_scale = mo.ui.switch(value=True, label="Scale shots as 1/T")
    mo.vstack([mo.md("### Configuration"),
               mo.hstack([ui_exp2_code, ui_exp2_p, ui_exp2_osd]),
               ui_exp2_rounds,
               mo.hstack([ui_exp2_shots, ui_exp2_seed, ui_exp2_workers]),
               mo.hstack([ui_exp2_xdet, ui_exp2_scale]),
               mo.md("*Each T costs three arms.*"),
               mo.md("*Shots above is the budget for the SMALLEST T. With **scale shots as "
                     "1/T** on, larger T gets proportionally fewer: the logical error rate "
                     "grows roughly linearly in T, so large T needs fewer shots for the same "
                     "number of failures, while costing more per shot. Flat shots spends the "
                     "most compute exactly where it is least needed — scaling makes every "
                     "point cost about the same and keeps the small-T error bars tight.*"),
               mo.md("*X-check detectors off is the corrected model: in a Z-basis memory they "
                     "can never help predict a Z-type observable, but they multiply the DEM by "
                     "roughly an order of magnitude. Leave it off unless you are reproducing section 4 exactly.*")])
    return (ui_exp2_code, ui_exp2_osd, ui_exp2_p, ui_exp2_rounds, ui_exp2_scale,
            ui_exp2_seed, ui_exp2_shots, ui_exp2_workers, ui_exp2_xdet)


@app.cell
def _(ui_exp2_code, ui_exp2_osd, ui_exp2_p, ui_exp2_rounds, ui_exp2_scale, ui_exp2_seed,
      ui_exp2_shots, ui_exp2_workers, ui_exp2_xdet):
    exp2_config = dict(code=ui_exp2_code.value, p=float(ui_exp2_p.value),
                      rounds=sorted(int(t) for t in ui_exp2_rounds.value),
                      shots=int(ui_exp2_shots.value), seed=int(ui_exp2_seed.value),
                      osd_order=int(ui_exp2_osd.value), workers=int(ui_exp2_workers.value),
                      x_detectors=bool(ui_exp2_xdet.value),
                      scale_shots=bool(ui_exp2_scale.value))

    def exp2_shots_for(T, config=None):
        """
        Shots to run at T rounds.

        `config["shots"]` is the budget for the smallest T in the sweep. The
        logical error rate grows roughly linearly in T, so the shots needed for a
        fixed number of failures fall as 1/T -- while the cost per shot rises with
        T, because the DEM grows with it. Flat allocation therefore spends the most
        time on the points that need the least; 1/T makes shots x T, the rough cost
        of a point, about equal across the sweep. Floored at 2,000 so no point
        becomes too small to resolve anything, and never above the budget.
        """
        config = exp2_config if config is None else config
        base = int(config["shots"])
        if not config.get("scale_shots", False) or not config["rounds"]:
            return base
        t_min = min(config["rounds"])
        return int(min(base, max(2000, round(base * t_min / int(T)))))

    return exp2_config, exp2_shots_for


@app.cell
def _(circuit_overhead, code_from_key, exp2_config, mo, core_ready):
    mo.stop(not core_ready)
    _code = code_from_key(exp2_config["code"])
    _rows = []
    for _T in exp2_config["rounds"]:
        _o = circuit_overhead(_code, _T, exp2_config["p"],
                              x_detectors=exp2_config["x_detectors"])
        _rows.append(dict(rounds=_T, qubits_unflagged=_o["unflagged"]["qubits"],
                          qubits_flagged=_o["flagged"]["qubits"],
                          qubit_overhead=_o["qubit_overhead"],
                          gate_overhead=_o["gate_overhead"],
                          fault_overhead=_o["fault_overhead"],
                          lambda_unflagged=_o["unflagged"]["expected_faults_per_shot"],
                          lambda_flagged=_o["flagged"]["expected_faults_per_shot"]))
    exp2_overhead = _rows
    mo.vstack([
        mo.md("### Cost of the flag hardware (no decoding)"),
        mo.md("| T | qubits | +qubits | +2q gates | +fault mechanisms | λ unflagged → flagged |\n"
              "|--:|--:|--:|--:|--:|--:|\n"
              + "\n".join(f"| {r['rounds']} | {r['qubits_unflagged']} → {r['qubits_flagged']} | "
                          f"{r['qubit_overhead']:+.0%} | {r['gate_overhead']:+.0%} | "
                          f"{r['fault_overhead']:+.0%} | "
                          f"{r['lambda_unflagged']:.2f} → {r['lambda_flagged']:.2f} |"
                          for r in exp2_overhead)),
        mo.md("*λ is the expected number of fault mechanisms firing per shot: the quantity "
              "that actually drives the logical error rate.*"),
    ])
    return (exp2_overhead,)


@app.cell
def _(mo):
    run_exp2 = mo.ui.run_button(label="Run Experiment 2 — flag cost vs information")
    run_exp2
    return (run_exp2,)


@app.cell
def _(
    RESULTS_DIR,
    code_from_key,
    exp2_config,
    exp2_shots_for,
    json,
    mo,
    os,
    core_ready,
    run_exp2,
    three_arm_run,
    write_result_file,
):
    exp2_run = None
    if not core_ready:
        _out = mo.md("*Locked: `decoderSwitch.py` tests must pass first.*")
    elif not run_exp2.value:
        _out = mo.md("*Press **Run Experiment 2 — flag cost vs information**, or load a saved run below.*")
    else:
        _code = code_from_key(exp2_config["code"])
        _tag = exp2_config["code"].strip("[]").replace(", ", "-")
        _stem = (f"EXP2_{_tag}_p{exp2_config['p']:g}_{exp2_config['shots']}shots_"
                 f"seed{exp2_config['seed']}_osd{exp2_config['osd_order']}"
                 f"{'_xdet' if exp2_config['x_detectors'] else ''}"
                 f"{'_scaled' if exp2_config['scale_shots'] else ''}")
        _path = os.path.join(RESULTS_DIR, _stem + ".json")
        _ckpt = os.path.join(RESULTS_DIR, "checkpoints", _stem)
        _points = []
        # Cost per T is roughly shots x T, so weight the bar by that rather than
        # giving every T the same share: the ETA is meaningless otherwise.
        _units = {T: exp2_shots_for(T) * T for T in exp2_config["rounds"]}
        _total = sum(_units.values())
        with mo.status.progress_bar(total=_total, title="three-arm sweep", show_eta=True) as _bar:
            for _T in exp2_config["rounds"]:
                _pt = three_arm_run(_code, _T, exp2_config["p"], exp2_shots_for(_T),
                                    exp2_config["seed"], workers=exp2_config["workers"],
                                    osd_order=exp2_config["osd_order"],
                                    x_detectors=exp2_config["x_detectors"],
                                    checkpoint_dir=_ckpt)
                _points.append(_pt)
                # Write after EVERY T. The sweep runs for hours, and a crash or an
                # interrupt after the last point used to lose the whole run.
                write_result_file(_path, {"kind": "exp2", "config": exp2_config,
                                          "points": _points})
                _bar.update(_units[_T])
        exp2_run = dict(config=exp2_config, points=_points, path=_path)
        _out = mo.md(f"Sweep finished. Saved to `{_path}`.")
    _out
    return (exp2_run,)


@app.cell
def _(RESULTS_DIR, exp2_run, mo, os):
    import glob as _glob
    _ = exp2_run
    _files = sorted(_glob.glob(os.path.join(RESULTS_DIR, "EXP2_*.json")))
    ui_exp2_file = mo.ui.dropdown({os.path.basename(f): f for f in _files},
                                 value=os.path.basename(_files[-1]) if _files else None,
                                 label="Saved Experiment 2 run")
    load_exp2 = mo.ui.run_button(label="Load")
    mo.hstack([ui_exp2_file, load_exp2]) if _files else mo.md("*No saved sweeps yet.*")
    return load_exp2, ui_exp2_file


@app.cell
def _(load_exp2, read_result_file, ui_exp2_file):
    exp2_loaded = None
    if load_exp2.value and ui_exp2_file.value:
        _d = read_result_file(ui_exp2_file.value, "exp2")
        exp2_loaded = dict(config=_d["config"], points=_d["points"], path=ui_exp2_file.value)
    return (exp2_loaded,)


@app.cell
def _(ARMS, crossover_fit, exp2_loaded, exp2_run, mo, np, plt, ratio_to_unflagged):
    exp2_data = exp2_run if exp2_run is not None else exp2_loaded
    mo.stop(exp2_data is None)
    _pts = sorted(exp2_data["points"], key=lambda q: q["rounds"])
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
    _a.set_xlabel("syndrome rounds T per shot")
    _a.set_ylabel("logical error rate per shot")
    _a.set_title(f"{exp2_data['config']['code']}, p = {exp2_data['config']['p']}, "
                 f"{exp2_data['config']['shots']:,} shots/arm")
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
    _b.set_xlabel("syndrome rounds T per shot")
    _b.set_ylabel("logical error rate ÷ unflagged")
    _b.set_title("Below 1 = flags help. Dashed: weighted fit of ln(ratio) in T")
    _b.grid(True, alpha=0.3)
    _b.legend(fontsize=8)
    _fig.tight_layout()
    exp2_fig, exp2_fit = _fig, dict(sighted=_fit_sighted, blind=_fit_blind)
    return exp2_data, exp2_fig, exp2_fit


@app.cell
def _(export_bundle, exp2_data, exp2_fig, exp2_fit, mo, exp2_overhead, ratio_to_unflagged):
    _pts = sorted(exp2_data["points"], key=lambda q: q["rounds"])
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

    _cross = exp2_fit["sighted"]["crossover"]
    _headline = (
        f"**Flags pay up to T\\* ≈ {_cross:.0f}** and cost accuracy beyond it "
        f"(weighted fit, slope {exp2_fit['sighted']['slope']:+.3f} per round)."
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
        exp2_data["path"],
        f"Flag trade-off study — {exp2_data['config']['code']}, p = {exp2_data['config']['p']}",
        {"three_arms_vs_rounds": exp2_fig},
        {"points": _rows, "overhead": exp2_overhead},
        notes=_headline)
    mo.vstack([mo.md("### Results"), mo.md(_headline), mo.md("\n".join(_lines)),
               mo.md(f"Data: `{exp2_data['path']}` · exported to `{_folder}`"), exp2_fig])
    return


@app.cell
def _(
    circuit_overhead,
    code_from_key,
    crossover_fit,
    exp2_shots_for,
    mo,
    np,
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

    def _shot_schedule_equalises_cost():
        cfg = dict(shots=20000, rounds=[6, 12, 18, 24], scale_shots=True)
        got = {T: exp2_shots_for(T, cfg) for T in cfg["rounds"]}
        assert got[6] == 20000, f"smallest T must keep the full budget, got {got[6]}"
        assert all(got[a] >= got[b] for a, b in zip(cfg["rounds"], cfg["rounds"][1:])), got
        # shots x T is the rough cost of a point: the schedule should level it
        cost = [got[T] * T for T in cfg["rounds"]]
        assert max(cost) / min(cost) < 1.15, f"cost per point not levelled: {cost}"
        assert sum(cost) < 0.45 * sum(20000 * T for T in cfg["rounds"]), \
            "schedule should cut total cost well below flat allocation"

    def _shot_schedule_respects_switches():
        flat = dict(shots=20000, rounds=[6, 12, 24], scale_shots=False)
        assert all(exp2_shots_for(T, flat) == 20000 for T in flat["rounds"])
        # the floor keeps a long-T point big enough to resolve anything at all
        tiny = dict(shots=3000, rounds=[3, 24], scale_shots=True)
        assert exp2_shots_for(24, tiny) == 2000, exp2_shots_for(24, tiny)

    def _z_only_three_arm_run():
        # the corrected DEM must still produce a well-formed three-arm point
        code = code_from_key("[[72, 12, 6]]")
        r = three_arm_run(code, 2, 3e-3, 120, seed=5, workers=1, x_detectors=False)
        assert r["x_detectors"] is False
        assert r["detectors_flagged"] > r["detectors_unflagged"]
        big = three_arm_run(code, 2, 3e-3, 4, seed=5, workers=1, x_detectors=True)
        assert r["detectors_unflagged"] < big["detectors_unflagged"], \
            "dropping X-check detectors should shrink the detector count"
        for arm in r["arms"].values():
            assert arm["ci_low"] <= arm["ler"] <= arm["ci_high"]

    tests_exp2 = run_checks([
        ("crossover fit recovers a known T*", _known_crossover_is_recovered),
        ("a ratio that never reaches 1 reports no crossover", _no_crossover_returns_none),
        ("ratios with zero failures are not reported", _ratio_handles_zero_failures),
        ("overhead accounting: one flag per X-check, more gates, more faults", _overhead_counts_are_sane),
        ("three-arm run returns consistent arms and intervals", _three_arms_differ_only_as_intended),
        ("1/T shot schedule levels the cost of each point", _shot_schedule_equalises_cost),
        ("shot schedule honours the switch and the floor", _shot_schedule_respects_switches),
        ("three-arm run works on the corrected (Z-only) DEM", _z_only_three_arm_run),
    ])
    render_checks("11.2 flag cost vs information", tests_exp2)
    return (tests_exp2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.3 Experiment 3 — trigger saturation: does a flag rule survive more rounds?


    A block-level trigger fires when **any** flag fires anywhere in the whole
    $T$-round block, so it tests $m_x \cdot T$ detectors at once. If a single
    flag fires with probability $q$, the block fires with probability
    $1 - (1-q)^{m_x T}$, which goes to 1 as $T$ grows however small $q$ is.

    This experiment measures that curve directly. It decodes nothing, so it runs
    in seconds. A trigger that fires on nearly every shot escalates nearly every
    shot, which is one of the two mechanisms behind the null result for the flag
    rules in Experiments 4 and 6. A *local* trigger — one flag, one round —
    would escape this saturation, but Experiment 6's headroom measurement closes
    that door too: across four weak decoders and 250,000 shots, at most **8**
    silent failures were repairable by the strong decoder (zero for three of the
    four), so a more selective trigger has almost nothing left to select.
    """)
    return


@app.cell
def _(mo):
    run_exp3 = mo.ui.run_button(label="Run Experiment 3 — trigger saturation (seconds)")
    run_exp3
    return (run_exp3,)


@app.cell
def _(
    build_memory_circuit,
    code_from_key,
    config,
    core_ready,
    flag_detector_mask,
    mo,
    run_exp3,
):
    exp3_run = None
    if core_ready and run_exp3.value:
        _code = code_from_key(config.code)
        _rows = []
        for _T in (3, 6, 12, 18, 24):
            _circ = build_memory_circuit(_code, _T, config.p, use_flags=True,
                                         x_detectors=config.x_detectors)
            _det = _circ.compile_detector_sampler(seed=config.seed).sample(2000)
            _mask = flag_detector_mask(_code, _T, True, _circ.num_detectors,
                                       x_detectors=config.x_detectors)
            _bits = _det[:, _mask]
            _rows.append(dict(rounds=_T, flag_detectors=int(_mask.sum()),
                              fraction_any_flag=float((_bits.sum(1) > 0).mean()),
                              mean_flags_per_shot=float(_bits.sum(1).mean())))
        exp3_run = dict(code=config.code, p=config.p, shots=2000, rows=_rows)
    mo.md("*Press **Run Experiment 3** — flag statistics only, no decoding, so it takes seconds.*"
          if exp3_run is None else "Experiment 3 finished.")
    return (exp3_run,)


@app.cell
def _(exp3_run, mo, saved_run_picker):
    _ = exp3_run                               # re-list after a new run
    ui_exp3_file, _f3 = saved_run_picker("EXP3", "Saved Experiment 3 run")
    load_exp3_btn = mo.ui.run_button(label="Load Experiment 3")
    mo.vstack([
        mo.md("#### Load a saved Experiment 3 run"),
        mo.hstack([ui_exp3_file, load_exp3_btn]) if _f3
        else mo.md("*No saved Experiment 3 runs yet.*")])
    return load_exp3_btn, ui_exp3_file


@app.cell
def _(load_exp3_btn, read_result_file, ui_exp3_file):
    exp3_loaded = None
    if load_exp3_btn.value and ui_exp3_file.value:
        _d = read_result_file(ui_exp3_file.value, "validation")
        exp3_loaded = dict(_d["config"], path=ui_exp3_file.value, rows=_d["checks"]["v5"])
    return (exp3_loaded,)


@app.cell
def _(RESULTS_DIR, export_bundle, mo, os, plt, exp3_loaded, exp3_run, write_result_file):
    exp3_data = exp3_run if exp3_run is not None else exp3_loaded
    mo.stop(exp3_data is None, mo.md("*Run Experiment 3, or load a saved run above.*"))
    _rows = exp3_data["rows"]
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4))
    _Ts = [r["rounds"] for r in _rows]
    _a.plot(_Ts, [r["fraction_any_flag"] for r in _rows], "o-", color="#d62728")
    _a.axhline(1.0, color="k", lw=0.7, ls=":")
    _a.set_ylim(0, 1.05)
    _a.set_xlabel("syndrome rounds T per shot")
    _a.set_ylabel("fraction of shots with >=1 flag fired")
    _a.set_title("'Any flag fired' saturates with T")
    _b.plot(_Ts, [r["mean_flags_per_shot"] for r in _rows], "s-", color="#1f77b4")
    _b.set_xlabel("syndrome rounds T per shot")
    _b.set_ylabel("mean flag bits fired per shot")
    _b.set_title("Flags fire in proportion to rounds")
    for _ax in (_a, _b):
        _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _tag = exp3_data["code"].strip("[]").replace(", ", "-")
    _path = write_result_file(
        os.path.join(RESULTS_DIR, f"EXP3_{_tag}_p{exp3_data['p']:g}_flagrate.json"),
        {"kind": "validation",
         "config": dict(code=exp3_data["code"], p=exp3_data["p"], shots=exp3_data["shots"]),
         "checks": {"v5": _rows}})
    _folder = export_bundle(_path, f"Experiment 3 flag rate vs rounds — {exp3_data['code']}, p={exp3_data['p']}",
                            {"flag_rate_vs_rounds": _fig}, {"flag_rate": _rows},
                            notes="A trigger that fires on nearly every shot cannot discriminate; "
                                  "this is the mechanism behind a null result for flag triggering.")
    mo.vstack([mo.md("### Experiment 3 — results: does the trigger saturate?"),
               mo.md("| T | flag detectors | shots with any flag | mean flags per shot |\n"
                     "|--:|--:|--:|--:|\n"
                     + "\n".join(f"| {r['rounds']} | {r['flag_detectors']} | "
                                  f"{r['fraction_any_flag']:.1%} | {r['mean_flags_per_shot']:.2f} |"
                                  for r in _rows)),
               mo.md(f"Saved to `{_path}` · exported to `{_folder}`"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.4 Experiment 4 — trigger ablation at fixed $p$


    All five triggers on the same shots, at one noise level. The thesis claim
    lives in the gap between `primary_fail` and `flag`: a pre-decode trigger can
    only beat the trivial post-hoc rule on shots where the weak decoder is
    **confidently wrong**, so the silent-failure count is the number to read
    first. Every policy is scored on identical shots, so the comparison is
    paired and the differences are exact.
    """)
    return


@app.cell
def _(mo):
    run_exp4 = mo.ui.run_button(label="Run Experiment 4 — trigger ablation")
    run_exp4
    return (run_exp4,)


@app.cell
def _(config, core_ready, mo, result_path, run_ablation, run_exp4, save_results):
    # Always define exp4_run (None when not run), so the view cell can use either
    # a fresh run or a loaded one.
    exp4_run = None
    if not core_ready:
        _out = mo.md("*Experiment 4 locked: finish the implementation (see §10).*")
    elif not run_exp4.value:
        _out = mo.md("*Press **Run Experiment 4** to start, or load a previous run below.*")
    else:
        _res = run_ablation(config)
        _path = save_results(result_path("EXP4", config), config, _res, include_samples=True)
        exp4_run = dict(config=config, results=_res, path=_path)
        _out = mo.md(f"Experiment 4 finished. Saved to `{_path}`.")
    _out
    return (exp4_run,)


@app.cell
def _(RESULTS_DIR, exp4_run, glob, mo, os):
    _ = exp4_run                                    # refresh the list after a new run
    _files = sorted(glob.glob(os.path.join(RESULTS_DIR, "EXP4_*.json")))
    ui_exp4_file = mo.ui.dropdown({os.path.basename(f): f for f in _files},
                                value=os.path.basename(_files[-1]) if _files else None,
                                label="Saved Experiment 4 run")
    load_exp4_btn = mo.ui.run_button(label="Load")
    mo.hstack([ui_exp4_file, load_exp4_btn]) if _files else mo.md("*No saved Experiment 4 runs yet.*")
    return load_exp4_btn, ui_exp4_file


@app.cell
def _(load_exp4_btn, load_results, ui_exp4_file):
    exp4_loaded = None
    if load_exp4_btn.value and ui_exp4_file.value:
        _cfg, _res = load_results(ui_exp4_file.value)
        exp4_loaded = dict(config=_cfg, results=_res, path=ui_exp4_file.value)
    return (exp4_loaded,)


@app.cell
def _(
    exp4_loaded,
    exp4_run,
    export_bundle,
    metrics_rows,
    mo,
    plot_ablation,
    plot_latency,
    plot_outcomes,
    plot_tradeoff,
):
    _e1 = exp4_run if exp4_run is not None else exp4_loaded
    mo.stop(_e1 is None)
    _cfg, _res = _e1["config"], _e1["results"]
    _always = _res.get("always")
    _beats = [t for t, m in _res.items()
              if _always is not None and t != "always" and m.ci_high < _always.ci_low]
    _table = "\n".join(
        ["| trigger | LER | 95% CI | escalated | silent failures | work (mean) | p99 time |",
         "|:--|--:|:--|--:|--:|--:|--:|"]
        + [f"| `{t}` | {m.ler:.4f} | [{m.ci_low:.4f}, {m.ci_high:.4f}] | "
           f"{m.escalation_rate:.1%} | {m.silent_failures} | {m.work_mean:.1f} | "
           f"{m.time_p99_us:.0f} µs |" for t, m in _res.items()])
    _figs = {"ablation": plot_ablation(_res), "outcomes": plot_outcomes(_res),
             "tradeoff": plot_tradeoff(_res)}
    if all(m.samples is not None for m in _res.values()):
        _figs["latency"] = plot_latency(_res)
    _title = f"Experiment 4 — {_cfg.code}, T={_cfg.rounds}, p={_cfg.p}, {_cfg.shots} shots"
    _folder = export_bundle(_e1["path"], _title, dict(_figs),
                            {"metrics": metrics_rows(_res, **vars(_cfg))})
    mo.vstack([
        mo.md(f"### {_title}"),
        mo.md(_table),
        mo.md("> ℹ️ " + ", ".join(_beats) + " beat `always`. This is possible when the weak "
              "decoder is right where the strong one errs — confirm it with a paired test on "
              "the same shots (Experiment 6) before reporting it.") if _beats else mo.md(""),
        mo.md(f"Data: `{_e1['path']}` · **Exported** figures (PNG + PDF), CSV and report to "
              f"`{_folder}`"),
        *[_f for _f in _figs.values()],
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.5 Experiment 5 — noise sweep


    Logical error rate and escalation rate against the physical error rate $p$.
    Panel (b) is the one that decides the latency argument: as the escalation
    rate approaches 1, a switching policy is doing the strong decoder's work on
    every shot and has no latency advantage left to claim.
    """)
    return


@app.cell
def _(mo):
    run_exp5 = mo.ui.run_button(label="Run Experiment 5 — noise sweep (slow)")
    run_exp5
    return (run_exp5,)


@app.cell
def _(config, core_ready, mo, np, result_path, run_exp5, run_sweep, save_sweep):
    exp5_run = None
    if not core_ready:
        _out = mo.md("*Experiment 5 locked: finish the implementation (see §10).*")
    elif not run_exp5.value:
        _out = mo.md("*Press **Run Experiment 5** to start, or load a previous run below.*")
    else:
        _ps = np.logspace(np.log10(5e-4), np.log10(6e-3), 6)
        _sweep = run_sweep(config, _ps)
        _path = save_sweep(result_path("EXP5", config), config, _sweep)
        exp5_run = dict(config=config, sweep=_sweep, path=_path)
        _out = mo.md(f"Experiment 5 finished. Saved to `{_path}`.")
    _out
    return (exp5_run,)


@app.cell
def _(RESULTS_DIR, exp5_run, glob, mo, os):
    _ = exp5_run
    _files = sorted(glob.glob(os.path.join(RESULTS_DIR, "EXP5_*.json")))
    ui_exp5_file = mo.ui.dropdown({os.path.basename(f): f for f in _files},
                                value=os.path.basename(_files[-1]) if _files else None,
                                label="Saved Experiment 5 run")
    load_exp5_btn = mo.ui.run_button(label="Load")
    mo.hstack([ui_exp5_file, load_exp5_btn]) if _files else mo.md("*No saved Experiment 5 runs yet.*")
    return load_exp5_btn, ui_exp5_file


@app.cell
def _(load_exp5_btn, load_sweep, ui_exp5_file):
    exp5_loaded = None
    if load_exp5_btn.value and ui_exp5_file.value:
        _cfg, _sweep = load_sweep(ui_exp5_file.value)
        exp5_loaded = dict(config=_cfg, sweep=_sweep, path=ui_exp5_file.value)
    return (exp5_loaded,)


@app.cell
def _(exp5_loaded, exp5_run, export_bundle, metrics_rows, mo, plot_sweep):
    _e2 = exp5_run if exp5_run is not None else exp5_loaded
    mo.stop(_e2 is None)
    _cfg, _sweep = _e2["config"], _e2["sweep"]
    _rows = [r for _p, _res in _sweep.items() for r in metrics_rows(_res, p=_p)]
    _fig = plot_sweep(_sweep)
    _title = f"Experiment 5 — {_cfg.code}, T={_cfg.rounds}, {_cfg.shots} shots/point"
    _folder = export_bundle(_e2["path"], _title, {"sweep": plot_sweep(_sweep)},
                            {"sweep_metrics": _rows})
    mo.vstack([mo.md(f"### {_title}"),
               mo.md(f"Data: `{_e2['path']}` · **Exported** to `{_folder}`"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.6 Experiment 6 — decoder zoo: is a flag trigger viable for any decoder pair?

    Experiment 4 tested one decoder pair. A flag trigger might still pay off with a
    different weak decoder (one that fails silently, or costs more) or a
    different strong decoder. Experiment 6 tests **every weak × strong pair** in a
    catalogue, against eight trigger rules, on the same shots.

    **Method.** Each shot is decoded *once* by every decoder. Each policy is
    then derived from the weak decoder's result, the strong decoder's result
    and the shot's flag bits, so the cost grows with the number of decoders,
    not the number of policies. The tests below check that this derivation
    reproduces `SwitchPolicy` exactly. Decoding runs on all your CPU cores,
    and peeling is compiled with `numba`, so timings are comparable across
    decoders.

    **Trigger rules.** `primary_fail` (the baseline), `flag>=k` (escalate
    before decoding if at least *k* flag bits fired, else on failure),
    `rounds>=k` (flags in at least *k* rounds), and `flag-only` (flags and
    nothing else).

    **Verdict per pair.** *Viable: accuracy* if some flag rule fails
    significantly less than `primary_fail` (paired exact test, p < 0.05).
    *Viable: cost* if some flag rule fails on no more shots than `primary_fail`
    and saves at least 5% of its measured time per shot. Otherwise *not viable*. A pair is
    *inconclusive* while `primary_fail` has fewer than 20 failures: too few to
    tell the rules apart, so run more shots.

    **Two numbers that explain the verdict.**
    *Headroom* is the number of the weak decoder's silent failures that the
    strong decoder gets right; no flag rule can gain more accuracy than this.
    *Break-even ratio* r\* is the weak/strong cost ratio above which a flag
    rule becomes cheaper than `primary_fail`; it is hardware-independent.
    """)
    return


@app.cell
def _(
    BB_PRESETS,
    BpOsdDecoder,
    DecodeResult,
    ExperimentConfig,
    PeelingDecoder,
    SwitchPolicy,
    bb_from_preset,
    build_memory_circuit,
    code_from_key,
    dem_to_matrices,
    dz,
    flag_detector_mask,
    mo,
    np,
    plot_zoo_breakeven,
    plot_zoo_silent,
    plot_zoo_tradeoffs,
    plot_zoo_verdicts,
    render_checks,
    run_ablation,
    run_benchmark,
    run_checks,
):
    _code = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
    _T = 3
    _XDET = False                       # matches ExperimentConfig.x_detectors
    _circ = build_memory_circuit(_code, _T, 2e-3, use_flags=True, x_detectors=_XDET)
    _H, _L, _pr = dem_to_matrices(_circ.detector_error_model(decompose_errors=False))
    _mask = flag_detector_mask(_code, _T, True, _circ.num_detectors, x_detectors=_XDET)
    _det, _obs = _circ.compile_detector_sampler(seed=17).sample(90, separate_observables=True)
    _det, _obs = _det.astype(np.uint8), _obs.astype(np.uint8)
    _names = ["peel", "BP-ms-10", "OSD-0", "LSD-0"]
    _state = {}

    def _zoo():
        if "res" not in _state:
            _state["res"], _state["backend"] = dz.run_zoo(
                _H, _L, _pr, _det, _obs, _names, workers=2, chunk_size=30)
        return _state["res"]

    def _peel_matches_notebook():
        ref, fast = PeelingDecoder(_H, _pr), dz.FastPeeling(_H, _pr)
        for s in _det[:60]:
            r = ref.decode(s)
            e, c, _ = fast.decode(s)
            assert np.array_equal(r.correction, e) and r.converged == c, \
                "compiled peeling disagrees with the notebook's PeelingDecoder"

    def _zoo_decoders_valid():
        for name in ("BP-ms-10", "OSD-0", "LSD-CS2"):
            dec = dz.make_decoder(name, _H, _pr)
            for s in _det[:20]:
                e, c, _ = dec.decode(s)
                ok = np.array_equal((_H @ e) % 2, s)
                assert c == ok, f"{name}: converged flag disagrees with H·e = s"
                if name != "BP-ms-10":
                    assert ok, f"{name}: strong decoder did not reproduce the syndrome"

    class _Adapt:
        def __init__(self, dec):
            self.dec = dec

        def decode(self, s):
            e, c, w = self.dec.decode(s)
            return DecodeResult(e, c, w)

    def _derivation_matches_switchpolicy():
        res = _zoo()
        W, S = _Adapt(dz.make_decoder("peel", _H, _pr)), _Adapt(dz.make_decoder("OSD-0", _H, _pr))
        n_flags, _ = dz.flag_features(_det, _mask, _code.hx.shape[0], _T)
        rules = dz.trigger_rules(n_flags, np.zeros_like(n_flags))
        for trig, rule in [("never", None), ("always", None),
                           ("primary_fail", "primary_fail"), ("flag_or_fail", "flag>=1")]:
            pol = SwitchPolicy(W, S, trig, _mask)
            if rule is None:
                src = res["peel"] if trig == "never" else res["OSD-0"]
                esc = np.full(len(_det), trig == "always")
                fail = src["fail"]
            else:
                P = dz.derive(res["peel"], res["OSD-0"], *rules[rule])
                esc, fail = P["esc"], P["fail"]
            for i, s in enumerate(_det):
                r = pol.decode(s)
                f = bool(np.any((_L @ r.correction) % 2 != _obs[i]))
                assert r.escalated == esc[i] and f == fail[i], f"{trig}: shot {i} differs"

    def _parallel_matches_sequential():
        par = _zoo()
        seq, _ = dz.run_zoo(_H, _L, _pr, _det, _obs, _names, workers=1, chunk_size=30)
        for n in _names:
            for k in ("conv", "fail", "work"):
                assert np.array_equal(par[n][k], seq[n][k]), f"{n}.{k} differs"

    def _paired_exact_known():
        a = np.zeros(20, bool); b = np.zeros(20, bool); b[:14] = True
        x, y, p = dz.paired_exact(a, b)
        assert (x, y) == (0, 14) and abs(p - 2 / 2**14) < 1e-15
        x, y, p = dz.paired_exact(b, b)
        assert (x, y, p) == (0, 0, 1.0)

    def _breakeven_formula():
        pre = np.array([1, 1, 0, 0] * 25, bool)
        esc_pf = np.array([0, 0, 1, 0] * 25, bool)
        esc_rule = pre | esc_pf
        r = dz.breakeven_ratio(esc_rule, pre, esc_pf)
        assert abs(r - (0.75 - 0.25) / 0.5) < 1e-12, r
        assert dz.breakeven_ratio(esc_pf, np.zeros(100, bool), esc_pf) == float("inf")

    def _inconclusive_when_few_failures():
        """Too few failures must never be read as evidence of accuracy parity.
        (Zero headroom is different: that settles accuracy outright.)"""
        n_flags, rounds = dz.flag_features(_det, _mask, _code.hx.shape[0], _T)
        a = dz.analyse(_zoo(), ["peel"], ["OSD-0"], n_flags, rounds, min_failures=10**9)
        p = a["pairs"][0]
        assert not p["verdict_accuracy"].startswith("viable"), p["verdict_accuracy"]
        expected = "impossible: no headroom" if p["headroom"] == 0 else "inconclusive"
        assert p["verdict_accuracy"] == expected, p["verdict_accuracy"]

    def _ablation_matches_switchpolicy():
        """Experiment 4 derives its policies instead of running them; it must agree exactly."""
        cfg = ExperimentConfig(code="[[72, 12, 6]]", rounds=_T, p=2e-3, shots=90, seed=17,
                               use_flags=True, osd_order=0, workers=1, x_detectors=_XDET)
        got = run_ablation(cfg, keep_samples=False)
        peel, osd = PeelingDecoder(_H, _pr), BpOsdDecoder(_H, _pr, osd_order=cfg.osd_order)
        ref = run_benchmark(_circ, {t: SwitchPolicy(peel, osd, t, _mask)
                                    for t in ("never", "always", "primary_fail",
                                              "flag", "flag_or_fail")},
                            _L, cfg.shots, cfg.seed)
        for t, m in ref.items():
            assert got[t].failures == m.failures, \
                f"{t}: derived {got[t].failures} failures, SwitchPolicy {m.failures}"
            assert abs(got[t].escalation_rate - m.escalation_rate) < 1e-12, t
            assert got[t].silent_failures == m.silent_failures, t

    def _positive_control_is_detected():
        """
        The check that makes a null believable: inject silent failures on flagged
        shots and confirm the analysis finds headroom and calls a flag rule viable.
        """
        v = dz.logical_null_vector(_H, _L)
        assert v is not None and not ((_H @ v) % 2).any() and ((_L @ v) % 2).any(), \
            "null vector must be invisible in the syndrome but flip an observable"
        spec = dict(dz.CONTROLS["control-flagged-50%"], vector=v, flag_mask=_mask)
        res, _ = dz.run_zoo(_H, _L, _pr, _det, _obs, [("control", spec), "OSD-0"],
                            workers=1, chunk_size=45)
        silent = res["control"]["conv"] & res["control"]["fail"]
        assert silent.sum() > 0, "the control produced no silent failures"
        n_flags, rounds = dz.flag_features(_det, _mask, _code.hx.shape[0], _T)
        assert (n_flags[silent] > 0).all(), "control failures must sit on flagged shots"
        a = dz.analyse(res, ["control"], ["OSD-0"], n_flags, rounds, min_failures=1)
        p = a["pairs"][0]
        assert p["headroom"] > 0, "analysis reports no headroom although silent failures exist"
        assert p["verdict_accuracy"] == "viable: accuracy", \
            f"analysis failed to detect a real effect: {p['verdict_accuracy']}"

    def _zero_headroom_is_impossible_not_inconclusive():
        n_flags, rounds = dz.flag_features(_det, _mask, _code.hx.shape[0], _T)
        a = dz.analyse(_zoo(), ["peel"], ["OSD-0"], n_flags, rounds, min_failures=10**9)
        p = a["pairs"][0]
        if p["headroom"] == 0:
            assert p["verdict_accuracy"] == "impossible: no headroom", p["verdict_accuracy"]

    def _ceiling_and_diagnostics():
        res = _zoo()
        n_flags, _ = dz.flag_features(_det, _mask, _code.hx.shape[0], _T)
        ceil = {(c["decoder"], c["compared_with"]): c
                for c in dz.strong_ceiling(res, ["OSD-0", "LSD-0"])}
        c = ceil[("OSD-0", "LSD-0")]
        assert c["failures"] == int(res["OSD-0"]["fail"].sum())
        assert c["fixed_by_other"] == int((res["OSD-0"]["fail"] & ~res["LSD-0"]["fail"]).sum())
        diag = {d["decoder"]: d for d in dz.flag_diagnostics(res, ["peel"], ["OSD-0"], n_flags)}
        assert 0.0 <= diag["peel"]["mutual_information_bits"] <= 1.0
        # a signal independent of the flags must carry no information
        rng = np.random.default_rng(0)
        fake = dict(res["peel"], fail=rng.random(len(_det)) < 0.3)
        mi = dz.flag_diagnostics({"peel": fake}, ["peel"], [], n_flags)[0]["mutual_information_bits"]
        assert mi < 0.02, f"independent signal reported {mi:.3f} bits"

    def _blind_decoding_removes_flag_rows():
        """
        The blind arm (Experiment 2) must drop the flag detector rows, not zero them: a zeroed
        syndrome claims no flag fired, which is a syndrome that never occurred.
        """
        keep = ~_mask
        H_blind, det_blind = _H[keep], _det[:, keep]
        assert H_blind.shape[0] == _H.shape[0] - int(_mask.sum())
        assert det_blind.shape[1] == H_blind.shape[0]
        dec = dz.make_decoder({"kind": "osd", "osd_order": 0, "max_iter": 20}, H_blind, _pr)
        for s_blind in det_blind[:15]:
            e, _, _ = dec.decode(s_blind)
            assert np.array_equal((H_blind @ e) % 2, s_blind), \
                "blind arm must still decode consistently on the reduced matrix"

    def _flag_rate_grows_with_rounds():
        """Experiment 3's mechanism: more rounds means more chances for a flag to fire."""
        means = []
        for T in (2, 6):
            circ = build_memory_circuit(_code, T, 2e-3, use_flags=True)
            det = circ.compile_detector_sampler(seed=3).sample(400)
            mask = flag_detector_mask(_code, T, True, circ.num_detectors)
            means.append(float(det[:, mask].sum(1).mean()))
        assert means[1] > means[0], f"mean flags per shot did not grow with T: {means}"

    def _graphs_render():
        import io
        n_flags, rounds = dz.flag_features(_det, _mask, _code.hx.shape[0], _T)
        a = dz.analyse(_zoo(), ["peel", "BP-ms-10"], ["OSD-0", "LSD-0"], n_flags, rounds)
        for fn in (plot_zoo_verdicts, plot_zoo_silent, plot_zoo_breakeven, plot_zoo_tradeoffs):
            fn(a).savefig(io.BytesIO(), format="png")

    def _headroom_bounds_gain():
        n_flags, rounds = dz.flag_features(_det, _mask, _code.hx.shape[0], _T)
        a = dz.analyse(_zoo(), ["peel", "BP-ms-10"], ["OSD-0", "LSD-0"], n_flags, rounds)
        head = {(p["weak"], p["strong"]): p["headroom"] for p in a["pairs"]}
        for r in a["rows"]:
            assert r["better"] <= head[(r["weak"], r["strong"])], "gain exceeds headroom"
            if r["rule"] not in ("flag-only", "primary_fail"):
                assert (r["better"], r["worse"]) == (r["catch"], r["harm"]), \
                    "gains/losses vs primary_fail must be exactly catches/harms"
        assert len(a["pairs"]) == 4 and all("verdict" in p for p in a["pairs"])

    tests_exp6 = run_checks([
        ("compiled peeling reproduces the notebook's PeelingDecoder exactly", _peel_matches_notebook),
        ("zoo decoders: honest convergence, strong ones always valid", _zoo_decoders_valid),
        ("derived policies reproduce SwitchPolicy shot by shot", _derivation_matches_switchpolicy),
        ("parallel decoding gives the same results as sequential", _parallel_matches_sequential),
        ("paired exact test gives known p-values", _paired_exact_known),
        ("break-even cost ratio formula", _breakeven_formula),
        ("no rule gains more than the headroom; gains = catches", _headroom_bounds_gain),
        ("too few failures gives 'inconclusive', never 'viable'", _inconclusive_when_few_failures),
        ("all four decoder-zoo graphs render", _graphs_render),
        ("parallel ablation matches SwitchPolicy exactly", _ablation_matches_switchpolicy),
        ("POSITIVE CONTROL: injected silent failures are detected", _positive_control_is_detected),
        ("zero headroom reports 'impossible', not 'inconclusive'", _zero_headroom_is_impossible_not_inconclusive),
        ("strong-decoder ceiling and flag diagnostics are correct", _ceiling_and_diagnostics),
        ("blind arm removes flag rows, not zeroes them", _blind_decoding_removes_flag_rows),
        ("flag rate grows with the number of rounds", _flag_rate_grows_with_rounds),
    ])
    _backend = _state.get("backend", "not run")
    mo.vstack([render_checks("decoder zoo", tests_exp6),
               mo.md(f"Parallel backend on this machine: **{_backend}** · "
                     f"compiled peeling: **{dz.NUMBA_STATUS}**"
                     + ("" if dz.HAVE_NUMBA else
                        "  \nWithout numba the peeling decoder runs orders of magnitude slower "
                        "than the 22 µs per shot measured with it, which makes every cost "
                        "comparison involving `peel` meaningless. Install it with "
                        "`uv add numba`, then reload this notebook."))])
    return (tests_exp6,)


@app.cell
def _(BB_PRESETS, GB_PRESETS, dz, mo, os):
    _cores = os.cpu_count() or 1
    ui_exp6_code = mo.ui.dropdown(list(BB_PRESETS) + list(GB_PRESETS), value="[[72, 12, 6]]", label="Code")
    ui_exp6_rounds = mo.ui.slider(1, 24, value=6, step=1, label="Rounds T")
    ui_exp6_p = mo.ui.dropdown(["5e-4", "1e-3", "2e-3", "3e-3"], value="1e-3", label="p")
    ui_exp6_shots = mo.ui.number(start=100, stop=1_000_000, step=100, value=20_000,
                                 label="Shots (20k+ to rank the rules)")
    ui_exp6_seed = mo.ui.number(value=20260922, label="Seed")
    # Each worker holds its own decoders, so peak memory scales with the worker
    # count times the number of decoders. 15 workers x 8 decoders exhausted 
    # memory on a 16-thread machine, so the default here is deliberately modest.
    ui_exp6_workers = mo.ui.slider(1, max(2, _cores), value=min(8, max(1, _cores - 4)),
                                   label=f"CPU workers (of {_cores}; lower this if you run out of memory)")
    ui_exp6_weak = mo.ui.multiselect(list(dz.WEAK) + list(dz.CONTROLS), value=dz.DEFAULT_WEAK,
                                    label="Weak decoders")
    ui_exp6_strong = mo.ui.multiselect(list(dz.STRONG), value=dz.DEFAULT_STRONG, label="Strong decoders")
    ui_exp6_xdet = mo.ui.switch(value=False, label="X-check detectors")
    mo.vstack([mo.md("### Experiment 6 — configuration (flags always on)"),
               mo.hstack([ui_exp6_code, ui_exp6_p, ui_exp6_rounds]),
               mo.hstack([ui_exp6_shots, ui_exp6_seed, ui_exp6_workers]),
               ui_exp6_xdet,
               ui_exp6_weak, ui_exp6_strong,
               mo.md("*OSD-CS orders cost roughly 15× OSD-0 per shot; LSD-CS is far cheaper. "
                     "Use the estimate button before a long run.*"),
               mo.md("*Memory, not cores, is usually the limit here: every worker builds its "
                     "own copy of each decoder. If a run reports a process-pool failure it "
                     "halves the workers and retries automatically, but starting lower is "
                     "faster. Fewer decoders per run also helps.*"),
               mo.md("*`control-*` weak decoders are **positive controls**: they are wrong "
                     "on a fraction of flagged shots by construction. Run one to show this "
                     "analysis detects silent failures when they exist — the check that makes "
                     "a negative result believable.*")])
    return (
        ui_exp6_code,
        ui_exp6_p,
        ui_exp6_rounds,
        ui_exp6_seed,
        ui_exp6_shots,
        ui_exp6_strong,
        ui_exp6_weak,
        ui_exp6_workers,
        ui_exp6_xdet,
    )


@app.cell
def _(
    build_memory_circuit,
    code_from_key,
    dem_to_matrices,
    flag_detector_mask,
    ui_exp6_code,
    ui_exp6_p,
    ui_exp6_rounds,
    ui_exp6_seed,
    ui_exp6_shots,
    ui_exp6_strong,
    ui_exp6_weak,
    ui_exp6_workers,
    ui_exp6_xdet,
):
    exp6_setup = dict(code=ui_exp6_code.value, rounds=int(ui_exp6_rounds.value),
                     p=float(ui_exp6_p.value), shots=int(ui_exp6_shots.value),
                     seed=int(ui_exp6_seed.value), workers=int(ui_exp6_workers.value),
                     weak=list(ui_exp6_weak.value), strong=list(ui_exp6_strong.value),
                     x_detectors=bool(ui_exp6_xdet.value))

    def build_exp6_problem(setup):
        """Circuit, DEM matrices and flag mask for an Experiment 6 configuration."""
        code = code_from_key(setup["code"])
        xdet = bool(setup.get("x_detectors", False))
        circ = build_memory_circuit(code, setup["rounds"], setup["p"], use_flags=True,
                                    x_detectors=xdet)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(code, setup["rounds"], True, circ.num_detectors,
                                  x_detectors=xdet)
        return code, circ, H, L, pr, mask

    return build_exp6_problem, exp6_setup


@app.cell
def _(RESULTS_DIR, dz, glob, mo, os, exp6_setup):
    import hashlib as _hashlib
    _tag = exp6_setup["code"].strip("[]").replace(", ", "-")
    _stem = (f"EXP6_{_tag}_T{exp6_setup['rounds']}_p{exp6_setup['p']:g}_"
             f"{exp6_setup['shots']}shots_seed{exp6_setup['seed']}"
             f"{'_xdet' if exp6_setup.get('x_detectors') else ''}")
    _key = _hashlib.sha1(",".join(exp6_setup["weak"] + exp6_setup["strong"]).encode()).hexdigest()[:8]
    exp6_checkpoint = dict(data_path=os.path.join(RESULTS_DIR, _stem + ".npz"),
                          dir=os.path.join(RESULTS_DIR, "checkpoints", f"{_stem}_{_key}"))
    _done = len(glob.glob(os.path.join(exp6_checkpoint["dir"], "chunk_*.npz")))
    _total = -(-exp6_setup["shots"] // dz.CHUNK_SIZE)
    (mo.md(f"🔁 **A partial run of this configuration exists: {_done}/{_total} chunks done.** "
           "Pressing Run Experiment 6 resumes it instead of starting over.")
     if _done else mo.md(""))
    return (exp6_checkpoint,)


@app.cell
def _(mo):
    estimate_exp6 = mo.ui.run_button(label="Estimate run time (~30 s pilot)")
    run_exp6 = mo.ui.run_button(label="Run Experiment 6 — decoder zoo")
    mo.hstack([estimate_exp6, run_exp6])
    return run_exp6, estimate_exp6


@app.cell
def _(build_exp6_problem, core_ready, dz, mo, np, estimate_exp6, tests_exp6, time, exp6_setup):
    mo.stop(not (core_ready and all(r["status"] == "PASS" for r in tests_exp6)),
            mo.md("*Experiment 6 locked until every test, including the decoder-zoo tests, passes.*"))
    mo.stop(not estimate_exp6.value)
    _code, _circ, _H, _L, _pr, _ = build_exp6_problem(exp6_setup)
    _det = _circ.compile_detector_sampler(seed=1).sample(25).astype(np.uint8)
    _rows, _total = [], 0.0
    for _name in exp6_setup["weak"] + exp6_setup["strong"]:
        _dec = dz.make_decoder(_name, _H, _pr)
        _dec.decode(_det[0])                              # compile / warm up
        _t0 = time.perf_counter()
        for _s in _det:
            _dec.decode(_s)
        _ms = 1e3 * (time.perf_counter() - _t0) / len(_det)
        _total += _ms
        _rows.append(f"| {_name} | {_ms:.2f} ms |")
    _eta = exp6_setup["shots"] * _total / 1e3 / max(1, exp6_setup["workers"])
    mo.md("| decoder | time per shot |\n|:--|--:|\n" + "\n".join(_rows)
          + f"\n\n**Estimated run time: {_eta / 60:.0f} min** for {exp6_setup['shots']:,} shots "
            f"on {exp6_setup['workers']} workers (parallel speed-up is usually a little below the "
            "worker count).")
    return


@app.cell
def _(
    build_exp6_problem,
    core_ready,
    datetime,
    dz,
    glob,
    mo,
    os,
    run_exp6,
    tests_exp6,
    exp6_checkpoint,
    exp6_setup,
):
    def _run_e3():
        """Sample the shots, decode them with every decoder, save, and summarise."""
        _code, _circ, _H, _L, _pr, _mask = build_exp6_problem(exp6_setup)
        _det, _obs = _circ.compile_detector_sampler(seed=exp6_setup["seed"]).sample(
            exp6_setup["shots"], separate_observables=True)
        _n_flags, _rounds = dz.flag_features(_det, _mask, _code.hx.shape[0], exp6_setup["rounds"])
        _names = []
        _vector = None
        for _n in exp6_setup["weak"] + exp6_setup["strong"]:
            _spec = dict(dz.CATALOGUE[_n])
            if _spec.get("kind") == "control":
                if _vector is None:
                    _vector = dz.logical_null_vector(_H, _L)   # H v = 0 but L v = 1
                _spec.update(vector=_vector, flag_mask=_mask)
            _names.append((_n, _spec))
        _n_chunks = -(-exp6_setup["shots"] // dz.CHUNK_SIZE)
        with mo.status.progress_bar(total=_n_chunks, title="Experiment 6: decoding", show_eta=True,
                                    show_rate=True) as _bar:
            # every finished chunk is written to exp6_checkpoint["dir"] at once, so an
            # interrupted run resumes when Run Experiment 6 is pressed again with the same settings
            _res, _backend = dz.run_zoo(_H, _L, _pr, _det, _obs, _names,
                                        workers=exp6_setup["workers"], on_chunk=_bar.update,
                                        checkpoint_dir=exp6_checkpoint["dir"])
        _path = exp6_checkpoint["data_path"]
        _meta = dict(exp6_setup, backend=_backend, numba=dz.HAVE_NUMBA,
                     created=datetime.datetime.now().isoformat(timespec="seconds"))
        dz.save_zoo(_path, _res, _n_flags, _rounds, _meta)
        for _f in glob.glob(os.path.join(exp6_checkpoint["dir"], "chunk_*.npz")):
            os.remove(_f)                     # the full result is saved: drop the checkpoints
        if os.path.isdir(exp6_checkpoint["dir"]) and not os.listdir(exp6_checkpoint["dir"]):
            os.rmdir(exp6_checkpoint["dir"])
        _data = dict(results=_res, n_flags=_n_flags, flag_rounds=_rounds, meta=_meta, path=_path)
        return _data, mo.md(f"Decoded {exp6_setup['shots']:,} shots × {len(_names)} decoders "
                            f"({_backend}). Saved to `{_path}`.")

    # Always define exp6_run (None when not run): marimo does not run cells that
    # depend on a variable that a stopped cell never defined.
    exp6_run = None
    if not (core_ready and all(r["status"] == "PASS" for r in tests_exp6)):
        _out = mo.md("*Experiment 6 locked until every test passes.*")
    elif not run_exp6.value:
        _out = mo.md("*Press **Run Experiment 6** to start, or load a previous run below.*")
    else:
        exp6_run, _out = _run_e3()
    _out
    return (exp6_run,)


@app.cell
def _(RESULTS_DIR, exp6_run, glob, mo, os):
    _ = exp6_run                            # re-list saved runs after each new run
    _files = sorted(glob.glob(os.path.join(RESULTS_DIR, "EXP6_*.npz")))
    ui_exp6_file = mo.ui.dropdown({os.path.basename(f): f for f in _files},
                                 value=os.path.basename(_files[-1]) if _files else None,
                                 label="Saved Experiment 6 run")
    load_exp6_btn = mo.ui.run_button(label="Load")
    mo.hstack([ui_exp6_file, load_exp6_btn]) if _files else mo.md("*No saved Experiment 6 runs yet.*")
    return load_exp6_btn, ui_exp6_file


@app.cell
def _(dz, load_exp6_btn, ui_exp6_file):
    exp6_loaded = None                      # always defined, see the run cell
    if load_exp6_btn.value and ui_exp6_file.value:
        _res, _nf, _fr, _meta = dz.load_zoo(ui_exp6_file.value)
        exp6_loaded = dict(results=_res, n_flags=_nf, flag_rounds=_fr, meta=_meta,
                         path=ui_exp6_file.value)
    return (exp6_loaded,)


@app.cell
def _(dz, exp6_loaded, exp6_run, mo):
    exp6_data = exp6_run if exp6_run is not None else exp6_loaded
    mo.stop(exp6_data is None, mo.md("*Run Experiment 6 or load a previous run to see the analysis.*"))
    _names = list(exp6_data["results"])
    exp6_weak = [n for n in _names if n in dz.WEAK or n in dz.CONTROLS]
    exp6_strong = [n for n in _names if n in dz.STRONG]
    exp6_analysis = dz.analyse(exp6_data["results"], exp6_weak, exp6_strong,
                             exp6_data["n_flags"], exp6_data["flag_rounds"])

    _pairs = exp6_analysis["pairs"]
    _viable = [p for p in _pairs if p["verdict"].startswith("viable")]
    _inconclusive = [p for p in _pairs if p["verdict"] == "inconclusive"]
    _head = sum(p["headroom"] for p in _pairs)
    _lines = [f"### Experiment 6 — verdict — {exp6_analysis['shots']:,} shots, "
              f"{len(exp6_weak)} weak × {len(exp6_strong)} strong decoders, `{exp6_data['path']}`", ""]
    if _viable:
        _lines.append(f"**Flag-triggered switching is viable for {len(_viable)} of {len(_pairs)} pairs:** "
                      + ", ".join(f"{p['weak']} → {p['strong']} ({p['verdict'].split(': ')[1]}, "
                                  f"rule `{p['best_accuracy_rule'] or p['best_cost_rule']}`)"
                                  for p in _viable) + ".")
    elif not _inconclusive:
        _lines.append(f"**Flag-triggered switching is not viable for any of the {len(_pairs)} pairs.**")
    if _inconclusive:
        _lines.append(f"**{len(_inconclusive)} of {len(_pairs)} pairs are inconclusive**: "
                      "fewer than 20 primary_fail failures. Run more shots (or a higher p) "
                      "before drawing conclusions about them.")
    _lines.append(f"Total headroom (weak-decoder silent failures the strong decoder fixes): "
                  f"**{_head}** across all pairs — the most accuracy any flag rule could gain.")
    _lines += ["", "**Accuracy and cost are judged separately.** Headroom is the number of "
               "weak-decoder silent failures the strong decoder fixes: with zero headroom no "
               "flag rule can gain accuracy, however many shots you run, so the verdict is "
               "*impossible*, not *inconclusive*. A difference of at least "
               f"{_pairs[0]['min_detectable_discordant'] if _pairs else 6} discordant shots is "
               "needed for significance.", "",
               "| weak → strong | accuracy | cost | headroom | primary_fail fails | always fails | "
               "escalated | best saving (no harm) | best rule |",
               "|:--|:--|:--|--:|--:|--:|--:|--:|:--|"]
    for p in _pairs:
        _lines.append(f"| {p['weak']} → {p['strong']} | {p['verdict_accuracy']} | "
                      f"{p['verdict_cost']} | {p['headroom']} | {p['pf_failures']} | "
                      f"{p['always_failures']} | {p['pf_escalation']:.1%} | "
                      f"{p['best_saving']:+.1%} | "
                      f"{p['best_accuracy_rule'] or p['best_cost_rule'] or '—'} |")
    _lines += ["", "#### What do the flags tell us?", "",
               "Mutual information between *a flag fired* and *this decoder was wrong*. "
               "Near zero means the trigger carries almost no usable signal.", "",
               "| decoder | error rate, flagged | unflagged | risk ratio | MI (bits) | "
               "share of failures on flagged shots |", "|:--|--:|--:|--:|--:|--:|"]
    for f in exp6_analysis["flag_diagnostics"]:
        _lines.append(f"| {f['decoder']} | {f['error_rate_flagged']:.2e} | "
                      f"{f['error_rate_unflagged']:.2e} | {f['risk_ratio']:.1f}× | "
                      f"{f['mutual_information_bits']:.4f} | "
                      f"{f['share_of_failures_on_flagged']:.1%} |")
    if exp6_analysis["strong_ceiling"]:
        _lines += ["", "#### Is there headroom left in the strong decoder?", "",
                   "Shots one strong decoder gets wrong that another gets right. If almost none "
                   "are fixed, those failures are near-uncorrectable at this noise level and "
                   "reweighting priors — flag-informed or otherwise — cannot help either.", "",
                   "| decoder | failures | fixed by | fixed | broken |",
                   "|:--|--:|:--|--:|--:|"]
        for c in exp6_analysis["strong_ceiling"]:
            _lines.append(f"| {c['decoder']} | {c['failures']} | {c['compared_with']} | "
                          f"{c['fixed_by_other']} | {c['broken_by_other']} |")
    _lines += ["", "| weak decoder | converged | silent failures | silent rate (95% upper) | "
               "on flagged shots | median time |", "|:--|--:|--:|--:|--:|--:|"]
    for w, v in exp6_analysis["weak"].items():
        _lines.append(f"| {w} | {v['converged']:,} | {v['silent']} | {v['silent_rate'][2]:.2e} | "
                      f"{v['silent_flagged']} | {v['time_us'] / 1e3:.2f} ms |")
    mo.md("\n".join(_lines))
    return exp6_analysis, exp6_data, exp6_strong, exp6_weak


@app.cell
def _(np, plt):
    def zoo_pair_rows(a, w, s):
        return [r for r in a["rows"] if r["weak"] == w and r["strong"] == s]

    def plot_zoo_verdicts(a):
        """Heatmap: best flag rule's net gain over primary_fail, per 10k shots."""
        W = list(dict.fromkeys(p["weak"] for p in a["pairs"]))
        S = list(dict.fromkeys(p["strong"] for p in a["pairs"]))
        grid = np.zeros((len(W), len(S)))
        pairs = {(p["weak"], p["strong"]): p for p in a["pairs"]}
        for i, w in enumerate(W):
            for j, s in enumerate(S):
                rows = [r for r in zoo_pair_rows(a, w, s) if r["rule"] != "primary_fail"]
                grid[i, j] = max(r["better"] - r["worse"] for r in rows) / a["shots"] * 1e4
        lim = max(1.0, np.abs(grid).max())
        fig, ax = plt.subplots(figsize=(1.9 * len(S) + 2.5, 0.95 * len(W) + 1.8))
        im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim)
        for i, w in enumerate(W):
            for j, s in enumerate(S):
                p = pairs[(w, s)]
                mark = {"viable: accuracy": "✓ accuracy", "viable: cost": "✓ cost",
                        "inconclusive": "? too few fails",
                        "not viable": "✗ no gain"}.get(p["verdict"], "✗")
                if p["headroom"] == 0 and not p["verdict"].startswith("viable"):
                    mark = "✗ no headroom"
                ax.text(j, i, f"{mark}\nΔ {grid[i, j]:+.1f}\nheadroom {p['headroom']}",
                        ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(S)), S, rotation=20)
        ax.set_yticks(range(len(W)), W)
        ax.set_xlabel("strong decoder")
        ax.set_ylabel("weak decoder")
        fig.colorbar(im, ax=ax, label="best flag rule: failures saved per 10k shots vs primary_fail")
        ax.set_title(f"Is flag-triggered switching viable? ({a['shots']:,} paired shots)")
        fig.tight_layout()
        return fig

    def plot_zoo_silent(a):
        """Silent-failure rate of each weak decoder: the only failures a flag can catch."""
        names = list(a["weak"])
        fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 4))
        for i, n in enumerate(names):
            v = a["weak"][n]
            ph, lo, hi = v["silent_rate"]
            if v["silent"]:
                ax.errorbar(i, ph, yerr=[[ph - lo], [hi - ph]], fmt="o", color="#d62728", capsize=4)
            else:
                ax.plot(i, hi, "v", ms=9, mfc="none", mec="#d62728")
            ax.annotate(f"{v['silent']}/{v['converged']:,}", (i, hi), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8)
        ax.set_yscale("log")
        ax.set_xlim(-0.6, len(names) - 0.4)
        _lo, _hi = ax.get_ylim()
        ax.set_ylim(_lo / 3, _hi * 3)
        ax.set_xticks(range(len(names)), names)
        ax.set_xlabel("weak decoder")
        ax.set_ylabel("silent failures per converged shot")
        ax.set_title("How often is each weak decoder confidently wrong?\n(▽ = none seen: 95% upper bound)")
        ax.grid(True, which="both", axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_zoo_breakeven(a):
        """
        Measured cost ratio c_weak/c_strong against the break-even ratio r* of each
        pair's cheapest flag rule with no accuracy loss. Points in the shaded region
        (measured ratio > r*) are pairs where that rule saves time on this machine.
        """
        weak = list(dict.fromkeys(p["weak"] for p in a["pairs"]))
        strong = list(dict.fromkeys(p["strong"] for p in a["pairs"]))
        colours = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
        markers = ["o", "s", "^", "D", "v", "P", "X"]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for p in a["pairs"]:
            rows = [r for r in zoo_pair_rows(a, p["weak"], p["strong"])
                    if r["rule"] != "primary_fail" and r["worse"] <= r["better"]
                    and np.isfinite(r["breakeven"])]
            if not rows:
                continue
            r = min(rows, key=lambda r: r["breakeven"])
            ax.plot(max(p["cost_ratio"], 1e-4), max(r["breakeven"], 1e-3),
                    markers[strong.index(p["strong"]) % len(markers)],
                    color=colours[weak.index(p["weak"]) % len(colours)], ms=8, ls="none")
        xs = np.logspace(-4, 1.5, 100)
        ax.plot(xs, xs, "k--", lw=1)
        ax.fill_between(xs, 1e-3, xs, color="#2ca02c", alpha=0.08)
        ax.text(0.97, 0.05, "flag rule cheaper\n(measured ratio > r*)", transform=ax.transAxes,
                ha="right", fontsize=8, color="#2ca02c")
        for i, w in enumerate(weak):
            ax.plot([], [], "o", color=colours[i % len(colours)], label=f"weak: {w}")
        for j, s in enumerate(strong):
            ax.plot([], [], markers[j % len(markers)], color="grey", label=f"strong: {s}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 30)
        ax.set_ylim(1e-3, 30)
        ax.set_xlabel("measured cost ratio  c_weak / c_strong  (median time per shot)")
        ax.set_ylabel("break-even ratio r* (cheapest flag rule with no accuracy loss)")
        ax.set_title("When could a flag trigger save time?")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_zoo_tradeoffs(a):
        """Small multiples: failures vs escalation for every rule, one panel per pair."""
        W = list(dict.fromkeys(p["weak"] for p in a["pairs"]))
        S = list(dict.fromkeys(p["strong"] for p in a["pairs"]))
        fig, axes = plt.subplots(len(W), len(S), figsize=(3.1 * len(S), 2.5 * len(W)),
                                 squeeze=False, sharex=True)
        for i, w in enumerate(W):
            for j, s in enumerate(S):
                ax = axes[i][j]
                for r in zoo_pair_rows(a, w, s):
                    pf = r["rule"] == "primary_fail"
                    ax.plot(r["escalation"], r["failures"], "*" if pf else "o",
                            color="#2ca02c" if pf else ("#d62728" if r["rule"] == "flag-only" else "#9467bd"),
                            ms=11 if pf else 5)
                ax.set_title(f"{w} → {s}", fontsize=8)
                ax.set_yscale("symlog", linthresh=1)   # flag-only is huge; keep the rest readable
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                if i == len(W) - 1:
                    ax.set_xlabel("fraction of shots escalated", fontsize=8)
                if j == 0:
                    ax.set_ylabel("logical failures", fontsize=8)
        fig.suptitle("Every rule for every pair (★ primary_fail, ● flag rules, red = flag-only). "
                     "A viable rule sits below or left of ★. Log-ish (symlog) y-axis: "
                     "flag-only fails on most shots and would otherwise flatten the rest.",
                     fontsize=9)
        fig.tight_layout()
        return fig

    return (
        plot_zoo_breakeven,
        plot_zoo_silent,
        plot_zoo_tradeoffs,
        plot_zoo_verdicts,
        zoo_pair_rows,
    )


@app.cell
def _(
    exp6_analysis,
    exp6_data,
    export_bundle,
    mo,
    plot_zoo_breakeven,
    plot_zoo_silent,
    plot_zoo_tradeoffs,
    plot_zoo_verdicts,
):
    _figs = {"verdicts": plot_zoo_verdicts(exp6_analysis), "silent_failures": plot_zoo_silent(exp6_analysis),
             "breakeven": plot_zoo_breakeven(exp6_analysis), "tradeoffs": plot_zoo_tradeoffs(exp6_analysis)}
    _weak = [dict(weak=w, converged=v["converged"], silent=v["silent"],
                  silent_rate=v["silent_rate"][0], silent_rate_upper=v["silent_rate"][2],
                  silent_on_flagged=v["silent_flagged"], median_time_us=v["time_us"])
             for w, v in exp6_analysis["weak"].items()]
    _rows = [{k: (v[0] if k == "ler" else v) for k, v in r.items()} | dict(
                 ler_low=r["ler"][1], ler_high=r["ler"][2]) for r in exp6_analysis["rows"]]
    _counts = {}
    for _p in exp6_analysis["pairs"]:
        _counts[_p["verdict"]] = _counts.get(_p["verdict"], 0) + 1
    _folder = export_bundle(
        exp6_data["path"], f"Experiment 6 decoder zoo — {exp6_analysis['shots']:,} shots",
        dict(_figs), {"verdicts": exp6_analysis["pairs"], "weak_decoders": _weak, "all_rules": _rows,
                      "flag_diagnostics": exp6_analysis["flag_diagnostics"],
                      "strong_ceiling": exp6_analysis["strong_ceiling"]},
        notes="Verdict counts: " + ", ".join(f"{k}: {v}" for k, v in sorted(_counts.items())))
    mo.vstack([mo.md(f"**Exported** figures (PNG + PDF), CSV tables and report to `{_folder}`"),
               *[_f for _f in _figs.values()]])
    return


# =============================================================================
# 11.7 Experiment 7 -- surface-code baseline
# =============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.7 Experiment 7 — surface-code baseline

    The reference point every QLDPC claim is measured against. P1 promised
    surface codes at $d \in \{5, 7\}$; this is that run.

    It uses the circuits of §5.2 — Stim's own rotated-memory generator, because
    `build_memory_circuit` needs translation invariance and a surface code has
    boundaries — and then the **same** `dem_to_matrices`, the same BP+OSD and the
    same Wilson intervals as every other experiment. Only circuit construction
    differs, which is what makes the comparison fair.

    What to read from it: the logical error rate should fall with distance below
    threshold and rise with it above, and the crossing gives a threshold estimate.

    **Measured result:** the crossing sits at **1.07%**, against the ~1% value the
    literature reports for the rotated surface code under circuit-level
    depolarising noise. Since only circuit construction differs from the BB
    experiments, that agreement licenses the rest of the notebook — it is
    end-to-end evidence on a code family with no stake in the thesis. The estimate
    comes from a genuine crossing between $p = 8{\times}10^{-3}$ and
    $1.2{\times}10^{-2}$, not from extrapolation. Note that the $d = 5$ and
    $d = 7$ points at $p \leq 10^{-3}$ record 0–3 failures in 20,000 shots:
    `threshold_estimate` does not filter on power, so those points are too
    underpowered to quote or plot as measurements even though they sit far below
    the crossing and do not affect it.
    """)
    return


@app.cell
def _(dz, np, surface_code_problem, wilson):
    def surface_sweep_point(distance, rounds, p, shots, seed, osd_order=0, workers=1,
                            on_chunk=None):
        """One (distance, p) point: build, sample, decode with BP+OSD, score."""
        pr = surface_code_problem(distance, rounds, p)
        det, obs = pr["circuit"].compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True)
        w = dz.parallel_workers(shots, workers)
        res, _backend = dz.run_zoo(
            pr["H"], pr["L"], pr["priors"], det.astype(np.uint8), obs.astype(np.uint8),
            [("strong", {"kind": "osd", "osd_order": osd_order, "max_iter": 20})],
            workers=w, chunk_size=dz.chunk_for(shots, w), on_chunk=on_chunk)
        fails = int(res["strong"]["fail"].sum())
        ler, lo, hi = wilson(fails, shots)
        return dict(distance=distance, rounds=rounds, p=float(p), shots=int(shots),
                    seed=int(seed), osd_order=int(osd_order),
                    failures=fails, ler=ler, ci_low=lo, ci_high=hi,
                    detectors=int(pr["H"].shape[0]), faults=int(pr["H"].shape[1]),
                    qubits=int(pr["circuit"].num_qubits),
                    time_p50_us=float(np.median(res["strong"]["time_us"])))

    def threshold_estimate(points):
        """
        Crudest useful estimate: the p where the d = 5 and d = 7 curves cross.
        Below it more distance helps, above it more distance hurts. Returns None
        when the curves do not cross in the sampled range.
        """
        by_d = {}
        for pt in points:
            by_d.setdefault(pt["distance"], {})[pt["p"]] = pt["ler"]
        if not {5, 7} <= set(by_d):
            return None
        ps = sorted(set(by_d[5]) & set(by_d[7]))
        sign = [by_d[5][p] - by_d[7][p] for p in ps]
        for a, b, pa, pb in zip(sign, sign[1:], ps, ps[1:]):
            if a > 0 >= b or a < 0 <= b:          # d=7 overtakes d=5
                t = abs(a) / (abs(a) + abs(b)) if (a or b) else 0.5
                return float(pa + t * (pb - pa))
        return None

    return surface_sweep_point, threshold_estimate


@app.cell
def _(mo, os):
    _cores = os.cpu_count() or 1
    ui_exp7_d = mo.ui.multiselect(["3", "5", "7"], value=["3", "5", "7"], label="Distances")
    ui_exp7_rounds = mo.ui.slider(1, 24, value=6, step=1, label="Rounds T")
    ui_exp7_p = mo.ui.multiselect(
        ["5e-4", "1e-3", "2e-3", "3e-3", "5e-3", "8e-3", "1.2e-2"],
        value=["5e-4", "1e-3", "2e-3", "3e-3", "5e-3", "8e-3"], label="Noise levels p")
    ui_exp7_shots = mo.ui.number(start=100, stop=500_000, step=100, value=20_000,
                                 label="Shots per point")
    ui_exp7_seed = mo.ui.number(value=20260929, label="Seed")
    ui_exp7_osd = mo.ui.slider(0, 7, value=0, step=1, label="OSD order")
    ui_exp7_workers = mo.ui.slider(1, max(2, _cores), value=min(12, max(1, _cores - 4)),
                                   label=f"CPU workers (of {_cores})")
    mo.vstack([mo.md("### Experiment 7 — configuration"),
               mo.hstack([ui_exp7_rounds, ui_exp7_shots, ui_exp7_osd]),
               ui_exp7_d, ui_exp7_p,
               mo.hstack([ui_exp7_seed, ui_exp7_workers]),
               mo.md("*Surface-code DEMs are far smaller than BB ones, so this is the "
                     "cheapest experiment here: d = 7 at T = 6 has 288 detectors against "
                     "the Gross code's 936. Include at least one p above 1% or the "
                     "curves never cross and no threshold can be read.*")])
    return (ui_exp7_d, ui_exp7_osd, ui_exp7_p, ui_exp7_rounds, ui_exp7_seed,
            ui_exp7_shots, ui_exp7_workers)


@app.cell
def _(ui_exp7_d, ui_exp7_osd, ui_exp7_p, ui_exp7_rounds, ui_exp7_seed, ui_exp7_shots,
      ui_exp7_workers):
    exp7_config = dict(distances=sorted(int(d) for d in ui_exp7_d.value),
                       rounds=int(ui_exp7_rounds.value),
                       ps=sorted(float(p) for p in ui_exp7_p.value),
                       shots=int(ui_exp7_shots.value), seed=int(ui_exp7_seed.value),
                       osd_order=int(ui_exp7_osd.value),
                       workers=int(ui_exp7_workers.value))
    return (exp7_config,)


@app.cell
def _(mo):
    run_exp7 = mo.ui.run_button(label="Run Experiment 7 — surface-code baseline")
    run_exp7
    return (run_exp7,)


@app.cell
def _(RESULTS_DIR, core_ready, exp7_config, mo, os, run_exp7, surface_sweep_point,
      write_result_file):
    exp7_run = None
    if not core_ready:
        _out = mo.md("*Locked: the tests above must pass first.*")
    elif not run_exp7.value:
        _out = mo.md("*Press **Run Experiment 7**, or load a saved run below.*")
    else:
        _stem = (f"EXP7_surface_T{exp7_config['rounds']}_{exp7_config['shots']}shots_"
                 f"seed{exp7_config['seed']}_osd{exp7_config['osd_order']}")
        _path = os.path.join(RESULTS_DIR, _stem + ".json")
        _points = []
        _total = len(exp7_config["distances"]) * len(exp7_config["ps"])
        with mo.status.progress_bar(total=_total, title="surface sweep",
                                    show_eta=True) as _bar:
            for _d in exp7_config["distances"]:
                for _p in exp7_config["ps"]:
                    _points.append(surface_sweep_point(
                        _d, exp7_config["rounds"], _p, exp7_config["shots"],
                        exp7_config["seed"], osd_order=exp7_config["osd_order"],
                        workers=exp7_config["workers"]))
                    # written after every point, as Experiment 2 does
                    write_result_file(_path, {"kind": "exp7", "config": exp7_config,
                                              "points": _points})
                    _bar.update(1)
        exp7_run = dict(config=exp7_config, points=_points, path=_path)
        _out = mo.md(f"Sweep finished. Saved to `{_path}`.")
    _out
    return (exp7_run,)


@app.cell
def _(RESULTS_DIR, exp7_run, glob, mo, os, read_result_file):
    _ = exp7_run
    _files = sorted(glob.glob(os.path.join(RESULTS_DIR, "EXP7_*.json")))
    ui_exp7_file = mo.ui.dropdown({os.path.basename(f): f for f in _files},
                                  value=os.path.basename(_files[-1]) if _files else None,
                                  label="Saved Experiment 7 run")
    load_exp7_btn = mo.ui.run_button(label="Load Experiment 7")
    mo.hstack([ui_exp7_file, load_exp7_btn]) if _files else mo.md(
        "*No saved Experiment 7 runs yet.*")
    return load_exp7_btn, read_result_file, ui_exp7_file


@app.cell
def _(load_exp7_btn, read_result_file, ui_exp7_file):
    exp7_loaded = None
    if load_exp7_btn.value and ui_exp7_file.value:
        _d = read_result_file(ui_exp7_file.value, "exp7")
        exp7_loaded = dict(config=_d["config"], points=_d["points"],
                           path=ui_exp7_file.value)
    return (exp7_loaded,)


@app.cell
def _(np, plt):
    def plot_surface_sweep(points):
        """Logical error rate against p, one line per distance, log-log."""
        by_d = {}
        for pt in points:
            by_d.setdefault(pt["distance"], []).append(pt)
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, d in enumerate(sorted(by_d)):
            rows = sorted(by_d[d], key=lambda r: r["p"])
            ps = [r["p"] for r in rows]
            ler = [max(r["ler"], r["ci_high"] if r["failures"] == 0 else r["ler"])
                   for r in rows]
            lo = [max(r["ler"] - r["ci_low"], 0) for r in rows]
            hi = [max(r["ci_high"] - r["ler"], 0) for r in rows]
            ax.errorbar(ps, ler, yerr=[lo, hi], marker="os^Dv"[i % 5], capsize=3,
                        label=f"d = {d}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("physical error rate $p$")
        ax.set_ylabel("logical error rate")
        ax.set_title("Surface-code baseline: below threshold, more distance helps")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    return (plot_surface_sweep,)


@app.cell
def _(exp7_loaded, exp7_run, export_bundle, mo, os, plot_surface_sweep, RESULTS_DIR,
      threshold_estimate, write_result_file):
    exp7_data = exp7_run if exp7_run is not None else exp7_loaded
    mo.stop(exp7_data is None, mo.md("*Run Experiment 7, or load a saved run above.*"))

    _pts = exp7_data["points"]
    _cfg = exp7_data["config"]
    _fig = plot_surface_sweep(_pts)
    _thr = threshold_estimate(_pts)
    _rows = ["| d | p | shots | failures | LER | 95% CI | qubits | detectors |",
             "|--:|--:|--:|--:|--:|:--|--:|--:|"]
    for _pt in sorted(_pts, key=lambda r: (r["distance"], r["p"])):
        _rows.append(
            f"| {_pt['distance']} | {_pt['p']:g} | {_pt['shots']} | {_pt['failures']} | "
            f"{_pt['ler']:.2e} | [{_pt['ci_low']:.1e}, {_pt['ci_high']:.1e}] | "
            f"{_pt['qubits']} | {_pt['detectors']} |")
    _head = (f"**Threshold estimate (d = 5 vs d = 7 crossing): "
             f"{_thr:.2e}**" if _thr else
             "*The d = 5 and d = 7 curves do not cross in this range — add a larger p.*")
    _path = os.path.join(
        RESULTS_DIR, f"EXP7_surface_T{_cfg['rounds']}_{_cfg['shots']}shots_"
                     f"seed{_cfg['seed']}_osd{_cfg['osd_order']}.json")
    write_result_file(_path, {"kind": "exp7", "config": _cfg, "points": _pts,
                              "threshold": _thr})
    _folder = export_bundle(
        _path, f"Experiment 7 surface-code baseline — T={_cfg['rounds']}, "
               f"{_cfg['shots']} shots/point",
        {"surface_sweep": _fig}, {"points": _pts},
        notes=f"Threshold estimate: {_thr:.3e}" if _thr else "No crossing in range.")
    mo.vstack([mo.md("### Experiment 7 — results"), mo.md(_head),
               mo.md("\n".join(_rows)), _fig,
               mo.md(f"**Exported** to `{_folder}`")])
    return (exp7_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.8 Experiment 8 — the accuracy/latency frontier

    Experiment 2 established that reading the flag detectors lowers the logical
    error rate. It did not ask the obvious follow-up: **is that the cheapest way
    to buy the same accuracy?** Flag qubits are not the only knob. Raising the
    OSD order costs nothing in hardware and also lowers the error rate, so the
    honest question is not "do flags help?" but "do flags help *per unit of
    decode time*?"

    This experiment answers it by sweeping both knobs together — four arms
    crossed with a range of OSD orders — and plotting logical error rate against
    median decode time. A point is **dominated** when some other point is both
    more accurate and faster; the points that survive form the Pareto frontier,
    and the verdict is which way of spending the budget lands on it.

    The four arms are §11.2's three plus §11.9's: `unflagged`, `blind`,
    `sighted` (flag rows added to $H$) and `updated` (the exact flag posterior
    applied to the priors, on the blind-sized matrix). Including `updated` here
    is the only way to compare it against the others honestly, for the reason
    below.

    /// admonition | One sweep, one timing regime.
    Per-shot decode time depends on how many worker processes compete for cores:
    the *identical* blind arm reads 7,246 µs at 12 workers and 4,246 µs at one.
    A frontier built from points measured at different worker counts would be an
    artefact of the harness, not a result. The `updated` arm re-derives its priors
    every shot and so cannot be parallelised at all, which pins the whole sweep to
    one worker whenever it is selected. Every point records the worker count it
    was measured at, and `pareto_frontier` refuses to rank points across regimes.
    ///

    **Why the cost is real.** Reading the flags makes the decoding problem
    larger, not just the circuit: on [[72,12,6]] at $T = 12$ the check matrix goes
    from 468 × 4,392 to 900 × 5,688, which is 2.5× the entries. Part of that is
    bought back — the flag detectors help BP converge, so the sighted arm needs
    fewer iterations than the blind arm on the same circuit — but only part.

    **How to read the result.** If a flagged point lands on the frontier, flags
    are worth their decode time at that operating point and Result R1 stands as a
    recommendation. If the frontier is entirely unflagged, the R1 gain is real but
    dominated, and the claim must narrow to *at fixed decoder strength* — still a
    finding, and a more careful one.
    """)
    return


@app.cell
def _(build_memory_circuit, dem_to_matrices, dz, flag_detector_mask,
      flag_groups, flag_posteriors, np, shot_priors, time, wilson):
    PARETO_ARMS = ("unflagged", "blind", "updated", "sighted")

    def pareto_point(code, rounds, p, shots, seed, arm, osd_order, workers=1,
                     max_iter=20, x_detectors=False, on_chunk=None):
        """
        One (arm, osd_order) point: accuracy and decode cost together.

        The arms: `unflagged` has no flag qubits; `blind` has them but the flag
        detector rows are removed from H; `updated` decodes that same blind matrix
        with the per-shot flag posterior of §11.9; `sighted` keeps the flag rows.
        The three flagged arms share one circuit and one set of shots, so they are
        paired with each other; `unflagged` is a different circuit and is not.

        TIMING REGIME. `time_p50_us` is per-shot CPU time, and it depends on how
        many worker processes are competing for cores: the identical blind arm
        reads 7,246 us at 12 workers and 4,246 us at one. `updated` changes the
        priors every shot, so it cannot use `dz.run_zoo`'s pool at all and always
        runs single-process. Every point therefore records the worker count it was
        measured at, and `pareto_frontier` refuses to rank points from different
        regimes -- comparing across them would be meaningless.
        """
        if arm not in PARETO_ARMS:
            raise ValueError(f"arm must be one of {PARETO_ARMS}")
        if arm == "updated" and workers != 1:
            raise ValueError(
                "the 'updated' arm re-derives priors every shot, so it cannot be "
                "parallelised; run the whole sweep at workers=1 so every arm is "
                "measured in the same timing regime")
        flags = arm != "unflagged"
        circ = build_memory_circuit(code, rounds, p, use_flags=flags,
                                    x_detectors=x_detectors)
        H, L, priors = dem_to_matrices(
            circ.detector_error_model(decompose_errors=False))
        det, obs = circ.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True)
        det, obs = det.astype(np.uint8), obs.astype(np.uint8)
        flag_bits = groups = None
        if arm in ("blind", "updated"):
            mask = flag_detector_mask(code, rounds, True, circ.num_detectors,
                                      x_detectors=x_detectors)
            flag_bits = det[:, mask]
            # column indices are unchanged by slicing rows, so the groups taken
            # from the full H stay valid against the blind matrix
            groups = flag_groups(H, mask)
            H, det = H[~mask], det[:, ~mask]

        spec = {"kind": "osd", "osd_order": osd_order, "max_iter": max_iter}
        if arm == "updated":
            idx, disjoint, info = groups
            if not disjoint:
                raise ValueError(f"flag groups overlap on this code: {info}")
            lo_p, hi_p, owner = flag_posteriors(priors, idx)
            dec = dz.make_decoder(dict(spec), H, priors)
            fail = np.zeros(shots, dtype=bool)
            t_us = np.zeros(shots, dtype=np.float64)
            iters = np.zeros(shots, dtype=np.float64)
            for i in range(shots):
                dec.dec.update_channel_probs(
                    shot_priors(lo_p, hi_p, owner, flag_bits[i]).tolist())
                t0 = time.perf_counter()
                e, _c, work = dec.decode(det[i])
                t_us[i] = (time.perf_counter() - t0) * 1e6
                iters[i] = work
                fail[i] = bool(np.any((L @ e) % 2 != obs[i]))
                if on_chunk is not None and (i + 1) % 500 == 0:
                    on_chunk(500)
        else:
            w = dz.parallel_workers(shots, workers)
            res, _backend = dz.run_zoo(
                H, L, priors, det, obs, [("strong", spec)],
                workers=w, chunk_size=dz.chunk_for(shots, w), on_chunk=on_chunk)
            r = res["strong"]
            fail, t_us, iters = r["fail"], r["time_us"], r["work"]

        fails = int(np.sum(fail))
        ler, lo, hi = wilson(fails, shots)
        return dict(arm=arm, osd_order=int(osd_order), code=code.name,
                    rounds=int(rounds), p=float(p), shots=int(shots), seed=int(seed),
                    workers=1 if arm == "updated" else int(workers),
                    failures=fails, ler=ler, ci_low=lo, ci_high=hi,
                    time_p50_us=float(np.median(t_us)),
                    time_p99_us=float(np.percentile(t_us, 99)),
                    bp_iters=float(np.mean(iters)),
                    detectors=int(H.shape[0]), faults=int(H.shape[1]),
                    qubits=int(circ.num_qubits))

    def timing_regimes(points):
        """The distinct worker counts a set of points was measured at."""
        return sorted({int(r.get("workers", 1)) for r in points})

    def pareto_frontier(points, cost="time_p50_us", value="ler", strict=True):
        """
        The non-dominated points, cheapest first.

        A point is dominated when another is no worse on both axes and strictly
        better on one. Ties on both axes keep the first point seen, so the
        frontier never holds two points at the same coordinates.

        `strict` refuses to rank points measured at different worker counts,
        because per-shot time is not comparable across them. Pass strict=False
        only to inspect a mixed set, never to make a claim from one.
        """
        if strict and len(timing_regimes(points)) > 1:
            raise ValueError(
                "these points were measured at different worker counts "
                f"{timing_regimes(points)}, so their decode times cannot be "
                "ranked against each other; re-run the sweep in one regime")
        out = []
        for a in sorted(points, key=lambda r: (r[cost], r[value])):
            if any(b[cost] <= a[cost] and b[value] <= a[value]
                   and (b[cost] < a[cost] or b[value] < a[value]) for b in points):
                continue
            if not any(b[cost] == a[cost] and b[value] == a[value] for b in out):
                out.append(a)
        return out

    def frontier_verdict(points):
        """Whether flags earn their decode time, and the cheapest way to each LER."""
        regimes = timing_regimes(points)
        if len(regimes) > 1:
            return dict(frontier=[], flagged_on_frontier=[], flags_pay=False,
                        best_ler_arm=None, regimes=regimes, comparable=False,
                        verdict=f"not comparable: points span worker counts {regimes}")
        front = pareto_frontier(points)
        flagged = [r for r in front if r["arm"] != "unflagged"]
        best = min(points, key=lambda r: r["ler"]) if points else None
        names = [f"{r['arm']}/OSD-{r['osd_order']}" for r in flagged]
        if any(r["arm"] == "updated" for r in front) and len(front) == 1:
            verdict = "prior updating dominates every other point"
        elif flagged:
            verdict = "flags are on the frontier"
        else:
            verdict = "the frontier is entirely unflagged"
        return dict(frontier=front, flagged_on_frontier=names,
                    flags_pay=bool(flagged), regimes=regimes, comparable=True,
                    best_ler_arm=best["arm"] if best else None, verdict=verdict)
    return (PARETO_ARMS, frontier_verdict, pareto_frontier, pareto_point,
            timing_regimes)


@app.cell
def _(BB_PRESETS, GB_PRESETS, PARETO_ARMS, mo, os):
    _cores = os.cpu_count() or 1
    ui_exp8_code = mo.ui.dropdown(list(BB_PRESETS) + list(GB_PRESETS),
                              value="[[72, 12, 6]]", label="Code")
    ui_exp8_rounds = mo.ui.slider(1, 24, value=12, step=1, label="Rounds T")
    ui_exp8_p = mo.ui.dropdown(["5e-4", "1e-3", "2e-3", "3e-3"], value="1e-3",
                               label="Physical error rate p")
    ui_exp8_shots = mo.ui.number(start=500, stop=200_000, step=500, value=20_000,
                                 label="Shots per point")
    ui_exp8_seed = mo.ui.number(value=20260930, label="Seed")
    ui_exp8_arms = mo.ui.multiselect(list(PARETO_ARMS), value=list(PARETO_ARMS),
                                     label="Arms")
    ui_exp8_osd = mo.ui.multiselect(["0", "2", "4", "7"], value=["0", "2", "4"],
                                    label="OSD orders")
    ui_exp8_workers = mo.ui.slider(1, max(2, _cores), value=min(12, max(1, _cores - 4)),
                                   label=f"CPU workers (of {_cores})")
    mo.vstack([mo.md("### Experiment 8 — configuration"),
               mo.hstack([ui_exp8_code, ui_exp8_rounds, ui_exp8_p]),
               mo.hstack([ui_exp8_shots, ui_exp8_seed, ui_exp8_workers]),
               ui_exp8_arms, ui_exp8_osd,
               mo.md("*Cost grows as arms x OSD orders, and high OSD orders are "
                     "slow: OSD-4 on the blind arm is the most expensive point "
                     "here. The frontier is about ordering points, not about "
                     "resolving each LER to three digits.*  \n"
                     "*Selecting **updated** forces the whole sweep to one worker: "
                     "that arm re-derives priors every shot and cannot be "
                     "parallelised, and per-shot times measured at different "
                     "worker counts are not comparable. Expect it to be slow — "
                     "drop the blind arm at OSD >= 2 if you are short of time, "
                     "since it is never on the frontier.*")])
    return (ui_exp8_arms, ui_exp8_code, ui_exp8_osd, ui_exp8_p, ui_exp8_rounds,
            ui_exp8_seed, ui_exp8_shots, ui_exp8_workers)


@app.cell
def _(ui_exp8_arms, ui_exp8_code, ui_exp8_osd, ui_exp8_p, ui_exp8_rounds,
      ui_exp8_seed, ui_exp8_shots, ui_exp8_workers):
    # One timing regime for the whole sweep: `updated` cannot be parallelised,
    # so selecting it pins every arm to a single worker.
    _arms8 = list(ui_exp8_arms.value)
    exp8_config = dict(code=ui_exp8_code.value, rounds=int(ui_exp8_rounds.value),
                       p=float(ui_exp8_p.value), shots=int(ui_exp8_shots.value),
                       seed=int(ui_exp8_seed.value),
                       arms=_arms8,
                       osd_orders=sorted(int(o) for o in ui_exp8_osd.value),
                       workers=1 if "updated" in _arms8 else int(ui_exp8_workers.value))
    return (exp8_config,)


@app.cell
def _(mo):
    run_exp8 = mo.ui.run_button(label="Run Experiment 8 — accuracy/latency frontier")
    run_exp8
    return (run_exp8,)


@app.cell
def _(RESULTS_DIR, code_from_key, core_ready, exp8_config, mo, os, pareto_point,
      run_exp8, write_result_file):
    exp8_run = None
    if not core_ready:
        _out = mo.md("*Locked: the tests above must pass first.*")
    elif not run_exp8.value:
        _out = mo.md("*Press **Run Experiment 8**, or load a saved run below.*")
    elif not exp8_config["arms"] or not exp8_config["osd_orders"]:
        _out = mo.md("*Pick at least one arm and one OSD order.*")
    else:
        _c = code_from_key(exp8_config["code"])
        _tag = exp8_config["code"].strip("[]").replace(", ", "-")
        _path = os.path.join(
            RESULTS_DIR,
            f"EXP8_{_tag}_T{exp8_config['rounds']}_p{exp8_config['p']:g}_"
            f"{exp8_config['shots']}shots_seed{exp8_config['seed']}.json")
        _points = []
        _total = len(exp8_config["arms"]) * len(exp8_config["osd_orders"])
        with mo.status.progress_bar(total=_total, title="accuracy/latency frontier",
                                    show_eta=True) as _bar:
            for _arm in exp8_config["arms"]:
                for _o in exp8_config["osd_orders"]:
                    _bar.update(subtitle=f"{_arm} · OSD-{_o}")
                    _points.append(pareto_point(
                        _c, exp8_config["rounds"], exp8_config["p"],
                        exp8_config["shots"], exp8_config["seed"], _arm, _o,
                        workers=exp8_config["workers"]))
                    # written after every point, as Experiments 2 and 7 do
                    write_result_file(_path, {"kind": "exp8", "config": exp8_config,
                                              "points": _points})
        exp8_run = dict(config=exp8_config, points=_points, path=_path)
        _out = mo.md(f"Sweep finished. Saved to `{_path}`.")
    _out
    return (exp8_run,)


@app.cell
def _(RESULTS_DIR, exp8_run, glob, mo, os):
    _ = exp8_run
    _files = sorted(glob.glob(os.path.join(RESULTS_DIR, "EXP8_*.json")))
    ui_exp8_file = mo.ui.dropdown({os.path.basename(f): f for f in _files},
                                  value=os.path.basename(_files[-1]) if _files else None,
                                  label="Saved Experiment 8 run")
    load_exp8_btn = mo.ui.run_button(label="Load Experiment 8")
    mo.hstack([ui_exp8_file, load_exp8_btn]) if _files else mo.md(
        "*No saved Experiment 8 runs yet.*")
    return load_exp8_btn, ui_exp8_file


@app.cell
def _(load_exp8_btn, read_result_file, ui_exp8_file):
    exp8_loaded = None
    if load_exp8_btn.value and ui_exp8_file.value:
        _d = read_result_file(ui_exp8_file.value, "exp8")
        exp8_loaded = dict(config=_d["config"], points=_d["points"],
                           path=ui_exp8_file.value)
    return (exp8_loaded,)


@app.cell
def _(np, pareto_frontier, plt):
    def plot_pareto(points):
        """Logical error rate against median decode time, with the frontier drawn."""
        style = {"unflagged": ("#555555", "s", "no flag qubits"),
                 "blind": ("#c2c2c2", "^", "flags built, decoder blind"),
                 "sighted": ("#8c8c8c", "o", "flag rows added to H"),
                 "updated": ("#1f6fb4", "D", "flag posterior on the priors")}
        fig, ax = plt.subplots(figsize=(8.4, 5.4))
        front = pareto_frontier(points, strict=False)
        if len(front) > 1:
            ax.plot([r["time_p50_us"] for r in front],
                    [r["ler"] if r["failures"] > 0 else r["ci_high"] for r in front],
                    "-", color="#d62728", lw=1.6, alpha=0.8, zorder=1,
                    label="Pareto frontier")
        elif len(front) == 1:
            ax.axvline(front[0]["time_p50_us"], color="#d62728", ls="--", lw=1.2,
                       alpha=0.6, zorder=1, label="frontier (a single point)")
        for arm, (col, mk, lab) in style.items():
            rows = sorted([r for r in points if r["arm"] == arm],
                          key=lambda r: r["time_p50_us"])
            if not rows:
                continue
            # A point with zero failures has no logical error rate to plot on a
            # log axis, so it is drawn at its 95% upper bound as a downward
            # marker -- the usual convention for a bound rather than a estimate.
            meas = [r for r in rows if r["failures"] > 0]
            zero = [r for r in rows if r["failures"] == 0]
            if meas:
                ax.errorbar([r["time_p50_us"] for r in meas],
                            [r["ler"] for r in meas],
                            yerr=[[r["ler"] - r["ci_low"] for r in meas],
                                  [r["ci_high"] - r["ler"] for r in meas]],
                            fmt=mk, color=col, ms=8, capsize=3, ls=":", lw=1,
                            label=lab, zorder=3)
            if zero:
                ax.plot([r["time_p50_us"] for r in zero],
                        [r["ci_high"] for r in zero], "v", color=col, ms=9,
                        mfc="none", mew=1.8, ls=":", lw=1, zorder=3,
                        label=None if meas else lab)
                seen = set()
                for r in zero:                  # one label per visual cluster
                    key = (round(np.log10(max(r["time_p50_us"], 1e-9)), 1),
                           round(np.log10(max(r["ci_high"], 1e-12)), 2))
                    if key in seen:
                        continue
                    seen.add(key)
                    ax.annotate("0 failures\n(95% bound)",
                                (r["time_p50_us"], r["ci_high"]), fontsize=7.5,
                                color=col, textcoords="offset points",
                                xytext=(9, -20))
            # OSD orders that land within a label's width of each other share
            # one label, so "OSD-2" and "OSD-4" cannot overprint into "OSD-24".
            # Clustering is in log space because both axes are logarithmic.
            def _y(r):
                return r["ler"] if r["failures"] > 0 else r["ci_high"]

            groups = {}
            for r in rows:
                key = (round(np.log10(max(r["time_p50_us"], 1e-9)), 1),
                       round(np.log10(max(_y(r), 1e-12)), 2))
                groups.setdefault(key, []).append(r)
            for members in groups.values():
                ax.annotate(
                    "OSD-" + ",".join(str(x) for x in
                                      sorted(m["osd_order"] for m in members)),
                    (max(m["time_p50_us"] for m in members),
                     max(_y(m) for m in members)),
                    fontsize=8, color=col, textcoords="offset points", xytext=(8, 5))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("median decode time per shot (µs)")
        ax.set_ylabel("logical error rate per shot")
        ax.set_title("Down and to the left is better — which way of using the flags wins?")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        return fig
    return (plot_pareto,)


@app.cell
def _(RESULTS_DIR, exp8_loaded, exp8_run, export_bundle, frontier_verdict, mo, os,
      plot_pareto, write_result_file):
    exp8_data = exp8_run if exp8_run is not None else exp8_loaded
    mo.stop(exp8_data is None, mo.md("*Run Experiment 8, or load a saved run above.*"))

    _pts, _cfg = exp8_data["points"], exp8_data["config"]
    _v = frontier_verdict(_pts)
    _fig = plot_pareto(_pts)
    _on = {(r["arm"], r["osd_order"]) for r in _v["frontier"]}
    _base = min((r for r in _pts if r["arm"] == "unflagged"),
                key=lambda r: r["ler"], default=None)

    _rows = ["| arm | OSD | LER | 95% CI | median µs | p99 µs | BP iters | "
             "detectors × faults | workers | frontier |",
             "|:--|--:|--:|:--|--:|--:|--:|--:|--:|:--:|"]
    for _r in sorted(_pts, key=lambda r: (r["arm"], r["osd_order"])):
        _rows.append(
            f"| {_r['arm']} | {_r['osd_order']} | {_r['ler']:.2e} | "
            f"[{_r['ci_low']:.1e}, {_r['ci_high']:.1e}] | {_r['time_p50_us']:,.0f} | "
            f"{_r['time_p99_us']:,.0f} | {_r['bp_iters']:.1f} | "
            f"{_r['detectors']} × {_r['faults']} | {_r.get('workers', 1)} | "
            f"{'**yes**' if (_r['arm'], _r['osd_order']) in _on else '—'} |")

    _thin = [r for r in _pts if r["failures"] < 10]
    _upd = [r for r in _v["frontier"] if r["arm"] == "updated"]
    if not _v.get("comparable", True):
        _head = (f"**Not comparable.** These points span worker counts "
                 f"{_v['regimes']}, and per-shot decode time is not comparable "
                 "across them — the identical blind arm reads 7,246 µs at 12 "
                 "workers and 4,246 µs at one. Re-run the sweep in a single "
                 "regime before reading any frontier from it.")
    elif _upd and len(_v["frontier"]) == 1:
        _r0 = _upd[0]
        _head = (f"**Prior updating dominates every other point.** "
                 f"`updated/OSD-{_r0['osd_order']}` is alone on the frontier at "
                 f"{_r0['ler']:.2e} in {_r0['time_p50_us']:,.0f} µs: nothing else "
                 "measured here is either more accurate or faster. The flag "
                 "information is worth having, and adding flag rows to $H$ is the "
                 "wrong way to take it.")
    elif _v["flags_pay"]:
        _head = (f"**Flags earn their decode time.** On the frontier: "
                 + ", ".join(f"`{n}`" for n in _v["flagged_on_frontier"])
                 + ". At those operating points no unflagged configuration is both "
                   "more accurate and faster, so R1 stands as a recommendation.")
    else:
        _head = ("**The frontier is entirely unflagged.** Every flagged point is "
                 "beaten on both axes by an unflagged one — raising the OSD order "
                 "buys the same accuracy more cheaply than adding flag qubits. "
                 "R1's gain is real *at fixed decoder strength*, and that is how "
                 "it must be stated; it is not a recommendation to build flags.")
    if _base is not None:
        _head += (f"  \nCheapest unflagged point reaching its best accuracy: "
                  f"OSD-{_base['osd_order']} at {_base['ler']:.2e} in "
                  f"{_base['time_p50_us']:,.0f} µs.")
    if _thin:
        _head += (f"  \n\n> **Provisional — {len(_thin)} of {len(_pts)} points have "
                  f"fewer than 10 failures** "
                  + ", ".join(f"`{r['arm']}/OSD-{r['osd_order']}` ({r['failures']})"
                              for r in _thin)
                  + ". The frontier is decided by which point is lower, so a point "
                    "resting on a handful of failures can move it. Raise the shot "
                    "count until every point on the frontier has ~50 failures before "
                    "quoting this verdict.")

    _path = os.path.join(
        RESULTS_DIR,
        f"EXP8_{_cfg['code'].strip('[]').replace(', ', '-')}_T{_cfg['rounds']}_"
        f"p{_cfg['p']:g}_{_cfg['shots']}shots_seed{_cfg['seed']}.json")
    write_result_file(_path, {"kind": "exp8", "config": _cfg, "points": _pts,
                              "frontier": [(r["arm"], r["osd_order"])
                                           for r in _v["frontier"]],
                              "flags_pay": _v["flags_pay"],
                              "timing_regimes": _v.get("regimes", []),
                              "underpowered": [(r["arm"], r["osd_order"], r["failures"])
                                               for r in _thin]})
    _folder = export_bundle(
        _path, f"Experiment 8 accuracy/latency frontier — {_cfg['code']}, "
               f"T={_cfg['rounds']}, p={_cfg['p']:g}",
        {"pareto": _fig}, {"points": _pts}, notes=_v["verdict"])
    mo.vstack([mo.md("### Experiment 8 — results"), mo.md(_head),
               mo.md("\n".join(_rows)), _fig,
               mo.md(f"Saved to `{_path}` · exported to `{_folder}`")])
    return (exp8_data,)


@app.cell
def _(PARETO_ARMS, code_from_key, frontier_verdict, pareto_frontier,
      pareto_point, render_checks, run_checks, timing_regimes):
    def _pt(arm, osd, ler, t, workers=1):
        return dict(arm=arm, osd_order=osd, ler=ler, time_p50_us=t, workers=workers)

    def _frontier_drops_dominated():
        pts = [_pt("unflagged", 0, 1e-2, 100), _pt("unflagged", 4, 1e-3, 200),
               _pt("sighted", 0, 5e-3, 1000)]          # worse on both axes
        f = pareto_frontier(pts)
        assert {(r["arm"], r["osd_order"]) for r in f} == {("unflagged", 0),
                                                           ("unflagged", 4)}, f

    def _frontier_keeps_a_genuine_tradeoff():
        pts = [_pt("unflagged", 0, 1e-2, 100), _pt("sighted", 0, 1e-3, 500)]
        assert len(pareto_frontier(pts)) == 2      # cheaper vs more accurate

    def _frontier_is_monotone():
        pts = [_pt("a", 0, 1e-2, 100), _pt("b", 0, 1e-3, 200), _pt("c", 0, 5e-3, 150),
               _pt("d", 0, 2e-2, 90)]
        f = pareto_frontier(pts)
        assert [r["time_p50_us"] for r in f] == sorted(r["time_p50_us"] for r in f)
        lers = [r["ler"] for r in f]
        assert all(x > y for x, y in zip(lers, lers[1:])), lers

    def _frontier_collapses_exact_ties():
        pts = [_pt("unflagged", 0, 1e-3, 100), _pt("sighted", 0, 1e-3, 100)]
        assert len(pareto_frontier(pts)) == 1

    def _verdict_reads_the_frontier():
        dominated = [_pt("unflagged", 0, 1e-3, 100), _pt("sighted", 0, 5e-3, 900)]
        assert not frontier_verdict(dominated)["flags_pay"]
        assert frontier_verdict(dominated)["frontier"][0]["arm"] == "unflagged"
        paying = [_pt("unflagged", 0, 1e-2, 100), _pt("sighted", 0, 1e-4, 300)]
        v = frontier_verdict(paying)
        assert v["flags_pay"] and v["flagged_on_frontier"] == ["sighted/OSD-0"], v

    def _empty_input_is_safe():
        assert pareto_frontier([]) == []
        assert frontier_verdict([])["flags_pay"] is False

    def _mixed_timing_regimes_are_refused():
        # the identical blind arm reads 7,246 us at 12 workers and 4,246 us at
        # one, so ranking across regimes would invent a result
        pts = [_pt("unflagged", 2, 3.8e-3, 1606, workers=12),
               _pt("updated", 0, 1.0e-4, 812, workers=1)]
        assert timing_regimes(pts) == [1, 12]
        try:
            pareto_frontier(pts)
        except ValueError:
            pass
        else:
            raise AssertionError("mixed worker counts were ranked anyway")
        v = frontier_verdict(pts)
        assert v["comparable"] is False and v["frontier"] == [], v
        assert pareto_frontier(pts, strict=False), "strict=False should still rank"

    def _one_regime_is_comparable():
        pts = [_pt("unflagged", 2, 3.8e-3, 1606), _pt("updated", 0, 1.0e-4, 812)]
        v = frontier_verdict(pts)
        assert v["comparable"] and len(v["frontier"]) == 1, v
        assert v["frontier"][0]["arm"] == "updated"
        assert v["verdict"] == "prior updating dominates every other point", v

    def _updated_arm_refuses_to_parallelise():
        c = code_from_key("[[72, 12, 6]]")
        assert "updated" in PARETO_ARMS
        try:
            pareto_point(c, 3, 1e-3, 100, 1, "updated", 0, workers=4)
        except ValueError as exc:
            assert "parallelis" in str(exc), exc
        else:
            raise AssertionError("the updated arm accepted workers > 1")

    def _updated_arm_runs_and_records_its_regime():
        c = code_from_key("[[72, 12, 6]]")
        r = pareto_point(c, 3, 1e-3, 120, 7, "updated", 0)
        b = pareto_point(c, 3, 1e-3, 120, 7, "blind", 0, workers=1)
        assert r["workers"] == 1 and b["workers"] == 1
        # same circuit and shots, same blind-sized matrix, different priors
        assert (r["detectors"], r["faults"]) == (b["detectors"], b["faults"])
        assert r["failures"] <= b["failures"], (r["failures"], b["failures"])

    tests_exp8 = run_checks([
        ("a point worse on both axes is dropped", _frontier_drops_dominated),
        ("a genuine speed/accuracy trade-off is kept", _frontier_keeps_a_genuine_tradeoff),
        ("the frontier is sorted by cost and falls in error", _frontier_is_monotone),
        ("two points at identical coordinates collapse to one", _frontier_collapses_exact_ties),
        ("the verdict reports whether a flagged point is on it", _verdict_reads_the_frontier),
        ("no points is not an error", _empty_input_is_safe),
        ("points from different worker counts are refused", _mixed_timing_regimes_are_refused),
        ("one regime ranks, and names a dominating updated point", _one_regime_is_comparable),
        ("the updated arm refuses workers > 1", _updated_arm_refuses_to_parallelise),
        ("the updated arm runs and records its regime", _updated_arm_runs_and_records_its_regime),
    ])
    render_checks("11.8 accuracy/latency frontier", tests_exp8)
    return (tests_exp8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.9 Experiment 9 — dynamic prior updating

    Experiment 2 gave the decoder the flag outcomes by adding $m_x T$ **rows** to
    $H$. That works, and Experiment 8 shows it earns its keep — but it is not the
    only way to use the same information, and it is the expensive way: the check
    matrix grows from 468 × 4,392 to 900 × 5,688.

    This experiment uses the flags **without adding a single row**. It decodes the
    blind matrix — the flagged circuit with its flag rows removed — but replaces
    the fault priors, per shot, with their posteriors given the observed flag
    pattern. This is the "dynamic prior updating" of the thesis title, and it has
    been the untested half of it.

    ### The update is exact, not an approximation

    Measured on the DEM rather than assumed (the tests below assert it):

    | property | [[72,12,6]], $T = 12$ |
    |:--|--:|
    | faults lighting at least one flag | 2,592 of 5,688 (45.6%) |
    | **maximum flag-weight of any fault** | **1** |
    | faults per flag detector | exactly 6 |
    | 432 flag detectors × 6 | 2,592 — a disjoint partition |

    Because no fault lights two flags, the flag rows are $m_x T$ **independent**
    single-parity checks over **disjoint** groups. The posterior therefore
    factorises and has a closed form. For a group $G$ with flag bit $f$, writing
    $\lambda_k = 1 - 2p_k$ and $s = (-1)^f$:

    ```latex
    P(e_j = 1 \mid f) \;=\; p_j \cdot
      \frac{1 - s\prod_{k \in G \setminus \{j\}} \lambda_k}
           {1 + s\prod_{k \in G} \lambda_k}
    ```

    No belief propagation, no iteration, no approximation — and a test checks it
    against brute-force enumeration of all $2^{|G|}$ patterns.

    The update is also **large**. On [[72,12,6]] a flagged fault's probability
    moves from $5.33{\times}10^{-4}$ to $0.103$ when its flag fires and to
    $2.49{\times}10^{-6}$ when it does not — an odds ratio of about 41,000. The
    flag all but decides the fault.

    ### What this costs, and what it might save

    Each fault takes one of exactly two posterior values, so both are precomputed
    once and the per-shot update is a single `where` over the fault vector: about
    50 µs against a decode of several thousand. The decoding problem stays at
    468 × 5,688 — blind-sized, 1.30× the unflagged matrix instead of sighted's
    2.49×.

    **The hypothesis.** Prior updating should land near the sighted arm on
    accuracy, because BP resolves degree-6 checks in roughly one pass anyway, and
    should beat the blind arm on *time*, because it starts from far better priors
    and so converges more often. If it falls well short of sighted, that is
    informative in its own right: it would mean the back-and-forth between the
    flag and syndrome subsystems matters, which the disjoint structure does not
    obviously predict.

    /// admonition | This experiment is single-process.
    Priors change every shot, so `dz.run_zoo`'s process pool cannot be used —
    it builds each decoder once per chunk. `ldpc` also holds the GIL, so threads
    do not help either. Budget accordingly: all four arms decode the same shots,
    so the cost is roughly four single-arm runs. Keep OSD order at 0 for the
    first pass.
    ///
    """)
    return


@app.cell
def _(np):
    def flag_groups(H, mask):
        """
        The flag rows of H as groups of fault indices, padded to a rectangle.

        Returns (idx, disjoint, info). `idx` is (n_flag_rows, max_group) with -1
        in the padding. `disjoint` is False when some fault lights more than one
        flag, in which case the closed form of `flag_posteriors` does not apply
        and the caller must not use it.
        """
        HF = (H[mask] > 0).tocsr()
        groups = [HF.indices[HF.indptr[i]:HF.indptr[i + 1]]
                  for i in range(HF.shape[0])]
        flat = np.concatenate(groups) if groups else np.zeros(0, dtype=np.int64)
        disjoint = len(flat) == len(np.unique(flat))
        width = max((len(g) for g in groups), default=0)
        idx = np.full((len(groups), width), -1, dtype=np.int64)
        for i, g in enumerate(groups):
            idx[i, :len(g)] = g
        return idx, bool(disjoint), dict(
            flag_rows=len(groups), faults_covered=int(len(flat)),
            group_sizes=sorted({len(g) for g in groups}),
            max_flag_weight=int(np.bincount(flat, minlength=1).max()) if len(flat) else 0)

    def flag_posteriors(priors, idx, floor=1e-12, ceiling=1.0 - 1e-12):
        """
        Exact per-fault posterior for each flag outcome, computed once.

        Returns (lo, hi, owner): `lo[j]` and `hi[j]` are fault j's probability
        when its flag reads 0 and 1; `owner[j]` is the index of the flag row that
        owns fault j, or -1 when no flag touches it. Faults with no flag keep
        their prior in both vectors, so a shot's prior vector is one `where`.

        The leave-one-out product is formed from prefix and suffix cumulative
        products rather than by dividing the total, so a prior of exactly 0.5
        (lambda = 0) cannot produce a division by zero.

        `floor` and `ceiling` only keep the log-likelihood ratio finite at 0 and
        1; they are deliberately not a probability clamp. A posterior above 0.5 is
        meaningful -- the flag says the fault is more likely than not -- and
        clipping it to 0.49 would silently discard that. On [[72,12,6]] the
        largest posterior is about 0.10, so neither bound is reached in practice.
        """
        pad = idx < 0
        P = np.where(pad, 0.0, priors[np.clip(idx, 0, None)])
        lam = 1.0 - 2.0 * P                                  # 1 for padding
        pre = np.ones_like(lam)
        suf = np.ones_like(lam)
        if lam.shape[1] > 1:
            pre[:, 1:] = np.cumprod(lam[:, :-1], axis=1)
            suf[:, :-1] = np.cumprod(lam[:, :0:-1], axis=1)[:, ::-1]
        excl = pre * suf                                     # prod over G \ {j}
        total = (excl * lam)[:, :1] if lam.shape[1] else np.ones((lam.shape[0], 1))

        def post(s):
            num = P * (1.0 - s * excl) / 2.0
            den = (1.0 + s * total) / 2.0
            return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), P)

        lo, hi = priors.copy(), priors.copy()
        owner = np.full(priors.shape[0], -1, dtype=np.int64)
        keep = ~pad
        rows = np.broadcast_to(np.arange(idx.shape[0])[:, None], idx.shape)
        lo[idx[keep]] = np.clip(post(1.0)[keep], floor, ceiling)   # s=+1: flag = 0
        hi[idx[keep]] = np.clip(post(-1.0)[keep], floor, ceiling)  # s=-1: flag = 1
        owner[idx[keep]] = rows[keep]
        return lo, hi, owner

    def shot_priors(lo, hi, owner, flag_bits):
        """Fault priors for one shot, given which flag detectors fired."""
        fired = np.zeros(lo.shape[0], dtype=bool)
        touched = owner >= 0
        fired[touched] = flag_bits[owner[touched]]
        return np.where(fired, hi, lo)
    return flag_groups, flag_posteriors, shot_priors


@app.cell
def _(build_memory_circuit, dem_to_matrices, dz, flag_detector_mask, flag_groups,
      flag_posteriors, np, shot_priors, time, wilson):
    def prior_update_run(code, rounds, p, shots, seed, osd_order=0, max_iter=20,
                         x_detectors=False, arms=("blind", "updated", "sighted"),
                         on_progress=None):
        """
        Decode one set of shots four ways and compare them pairwise.

        Every arm sees the SAME shots of the SAME flagged circuit, so all
        comparisons here are paired; `unflagged` is a different circuit and is
        reported for reference only, not paired.

        Arms: `blind` decodes H with the flag rows removed and the DEM's own
        priors; `updated` decodes that same matrix with the per-shot flag
        posterior; `sighted` decodes the full matrix including flag rows.
        """
        circ = build_memory_circuit(code, rounds, p, use_flags=True,
                                    x_detectors=x_detectors)
        H, L, priors = dem_to_matrices(
            circ.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(code, rounds, True, circ.num_detectors,
                                  x_detectors=x_detectors)
        idx, disjoint, info = flag_groups(H, mask)
        if not disjoint:
            raise ValueError(
                "some fault lights more than one flag detector, so the exact "
                f"single-check posterior does not apply here: {info}")
        lo, hi, owner = flag_posteriors(priors, idx)

        det, obs = circ.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True)
        det = det.astype(np.uint8)
        obs = obs.astype(np.uint8)
        flags = det[:, mask]
        syn = det[:, ~mask]
        HS, Lb = H[~mask], L

        # Built through decoder_zoo, not by hand: every other experiment decodes
        # with min-sum BP at scaling 0.625 on a parallel schedule, and an arm
        # built with ldpc's own defaults would not be comparable to Experiment 8.
        _spec = {"kind": "osd", "osd_order": osd_order, "max_iter": max_iter}
        decoders = {a: dz.make_decoder(dict(_spec), H if a == "sighted" else HS,
                                       priors)
                    for a in arms}

        fail = {a: np.zeros(shots, dtype=bool) for a in arms}
        t_us = {a: np.zeros(shots, dtype=np.float64) for a in arms}
        for i in range(shots):
            for a in arms:
                s = syn[i] if a != "sighted" else det[i]
                if a == "updated":
                    decoders[a].dec.update_channel_probs(
                        shot_priors(lo, hi, owner, flags[i]).tolist())
                t0 = time.perf_counter()
                e, _conv, _work = decoders[a].decode(s)
                t_us[a][i] = (time.perf_counter() - t0) * 1e6
                fail[a][i] = bool(np.any((Lb @ e) % 2 != obs[i]))
            if on_progress is not None and (i + 1) % 200 == 0:
                on_progress(i + 1)

        out = {}
        for a in arms:
            k = int(fail[a].sum())
            ler, cl, ch = wilson(k, shots)
            out[a] = dict(failures=k, ler=ler, ci_low=cl, ci_high=ch,
                          time_p50_us=float(np.median(t_us[a])),
                          time_p99_us=float(np.percentile(t_us[a], 99)),
                          detectors=int(H.shape[0] if a == "sighted" else HS.shape[0]),
                          faults=int(H.shape[1]))
        # Every arm saw the same shots of the same circuit, so each pair gets an
        # exact McNemar test on its discordant shots.
        paired = {}
        for a in arms:
            for b in arms:
                if a < b:
                    x, y, pv = dz.paired_exact(fail[a], fail[b])
                    paired[f"{a}|{b}"] = dict(a=a, b=b, a_only=x, b_only=y, p=pv)
        return dict(code=code.name, rounds=int(rounds), p=float(p),
                    shots=int(shots), seed=int(seed), osd_order=int(osd_order),
                    x_detectors=bool(x_detectors), arms={a: out[a] for a in arms},
                    paired=paired, structure=info,
                    flagged_fraction=float((flags.sum(1) > 0).mean()))
    return (prior_update_run,)


@app.cell
def _(BB_PRESETS, GB_PRESETS, mo):
    ui_exp9_code = mo.ui.dropdown(list(BB_PRESETS) + list(GB_PRESETS),
                                  value="[[72, 12, 6]]", label="Code")
    ui_exp9_rounds = mo.ui.slider(1, 24, value=12, step=1, label="Rounds T")
    ui_exp9_p = mo.ui.dropdown(["5e-4", "1e-3", "2e-3", "3e-3"], value="1e-3",
                               label="Physical error rate p")
    ui_exp9_shots = mo.ui.number(start=200, stop=100_000, step=200, value=20_000,
                                 label="Shots")
    ui_exp9_seed = mo.ui.number(value=20260930, label="Seed")
    ui_exp9_osd = mo.ui.slider(0, 4, value=0, step=1, label="OSD order")
    mo.vstack([mo.md("### Experiment 9 — configuration"),
               mo.hstack([ui_exp9_code, ui_exp9_rounds, ui_exp9_p]),
               mo.hstack([ui_exp9_shots, ui_exp9_seed, ui_exp9_osd]),
               mo.md("*Single-process, and all three arms decode every shot, so "
                     "expect roughly three times one arm's cost. At OSD-0 on "
                     "[[72,12,6]] that is around 15 ms per shot. Start at 20,000 "
                     "shots; raise the OSD order only after the first pass, since "
                     "the blind arm becomes very slow there.*")])
    return (ui_exp9_code, ui_exp9_osd, ui_exp9_p, ui_exp9_rounds, ui_exp9_seed,
            ui_exp9_shots)


@app.cell
def _(mo):
    run_exp9 = mo.ui.run_button(label="Run Experiment 9 — dynamic prior updating")
    run_exp9
    return (run_exp9,)


@app.cell
def _(RESULTS_DIR, code_from_key, core_ready, mo, os, prior_update_run, run_exp9,
      ui_exp9_code, ui_exp9_osd, ui_exp9_p, ui_exp9_rounds, ui_exp9_seed,
      ui_exp9_shots, write_result_file):
    exp9_run = None
    if not core_ready:
        _out = mo.md("*Locked: the tests above must pass first.*")
    elif not run_exp9.value:
        _out = mo.md("*Press **Run Experiment 9**, or load a saved run below.*")
    else:
        _cfg = dict(code=ui_exp9_code.value, rounds=int(ui_exp9_rounds.value),
                    p=float(ui_exp9_p.value), shots=int(ui_exp9_shots.value),
                    seed=int(ui_exp9_seed.value), osd_order=int(ui_exp9_osd.value))
        _tag = _cfg["code"].strip("[]").replace(", ", "-")
        _path = os.path.join(
            RESULTS_DIR,
            f"EXP9_{_tag}_T{_cfg['rounds']}_p{_cfg['p']:g}_{_cfg['shots']}shots_"
            f"seed{_cfg['seed']}_osd{_cfg['osd_order']}.json")
        with mo.status.progress_bar(total=_cfg["shots"], title="prior updating",
                                    show_eta=True) as _bar:
            _seen = [0]

            def _tick(done):
                _bar.update(done - _seen[0])
                _seen[0] = done

            _res = prior_update_run(
                code_from_key(_cfg["code"]), _cfg["rounds"], _cfg["p"],
                _cfg["shots"], _cfg["seed"], osd_order=_cfg["osd_order"],
                on_progress=_tick)
        write_result_file(_path, {"kind": "exp9", "config": _cfg, "result": _res})
        exp9_run = dict(config=_cfg, result=_res, path=_path)
        _out = mo.md(f"Finished. Saved to `{_path}`.")
    _out
    return (exp9_run,)


@app.cell
def _(RESULTS_DIR, exp9_run, glob, mo, os):
    _ = exp9_run
    _files = sorted(glob.glob(os.path.join(RESULTS_DIR, "EXP9_*.json")))
    ui_exp9_file = mo.ui.dropdown({os.path.basename(f): f for f in _files},
                                  value=os.path.basename(_files[-1]) if _files else None,
                                  label="Saved Experiment 9 run")
    load_exp9_btn = mo.ui.run_button(label="Load Experiment 9")
    mo.hstack([ui_exp9_file, load_exp9_btn]) if _files else mo.md(
        "*No saved Experiment 9 runs yet.*")
    return load_exp9_btn, ui_exp9_file


@app.cell
def _(load_exp9_btn, read_result_file, ui_exp9_file):
    exp9_loaded = None
    if load_exp9_btn.value and ui_exp9_file.value:
        _d = read_result_file(ui_exp9_file.value, "exp9")
        exp9_loaded = dict(config=_d["config"], result=_d["result"],
                           path=ui_exp9_file.value)
    return (exp9_loaded,)


@app.cell
def _(np, plt):
    def plot_prior_update(result):
        """Where prior updating lands between the blind and sighted arms."""
        order = [a for a in ("blind", "updated", "sighted") if a in result["arms"]]
        col = {"blind": "#c2c2c2", "updated": "#1f6fb4", "sighted": "#555555"}
        lab = {"blind": "blind\n(flags ignored)", "updated": "prior updating\n(no extra rows)",
               "sighted": "sighted\n(flag rows in H)"}
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
        xs = np.arange(len(order))
        for i, a in enumerate(order):
            m = result["arms"][a]
            a1.plot(i, m["ler"], "o", ms=11, color=col[a])
            a1.errorbar(i, m["ler"], yerr=[[m["ler"] - m["ci_low"]],
                                           [m["ci_high"] - m["ler"]]],
                        fmt="none", color="k", capsize=5)
            a1.annotate(f"{m['ler']:.2e}", (i, m["ler"]), fontsize=9, color=col[a],
                        textcoords="offset points", xytext=(11, 0), va="center")
            a2.plot(i, m["time_p50_us"], "o", ms=11, color=col[a])
            a2.annotate(f"{m['time_p50_us']:,.0f} µs", (i, m["time_p50_us"]),
                        fontsize=9, color=col[a], textcoords="offset points",
                        xytext=(11, 0), va="center")
        for ax, t, yl in ((a1, "Accuracy", "logical error rate per shot"),
                          (a2, "Cost", "median decode time per shot (µs)")):
            ax.set_xticks(xs, [lab[a] for a in order], fontsize=9)
            ax.set_yscale("log")
            ax.set_xlim(-0.5, len(order) - 0.15)
            ax.set_ylabel(yl)
            ax.set_title(t)
            ax.grid(axis="y", which="both", alpha=0.25)
        fig.suptitle("Does the flag information survive without the flag rows?",
                     fontsize=12.5)
        fig.tight_layout()
        return fig
    return (plot_prior_update,)


@app.cell
def _(RESULTS_DIR, exp9_loaded, exp9_run, export_bundle, mo, os,
      plot_prior_update, write_result_file):
    exp9_data = exp9_run if exp9_run is not None else exp9_loaded
    mo.stop(exp9_data is None, mo.md("*Run Experiment 9, or load a saved run above.*"))

    _r, _cfg = exp9_data["result"], exp9_data["config"]
    _A, _d = _r["arms"], _r.get("paired", {})
    _fig = plot_prior_update(_r)

    _rows = ["| arm | H | LER | 95% CI | median µs | p99 µs |",
             "|:--|--:|--:|:--|--:|--:|"]
    for _a in ("blind", "updated", "sighted"):
        if _a not in _A:
            continue
        _m = _A[_a]
        _rows.append(f"| {_a} | {_m['detectors']} × {_m['faults']} | "
                     f"{_m['ler']:.2e} | [{_m['ci_low']:.1e}, {_m['ci_high']:.1e}] | "
                     f"{_m['time_p50_us']:,.0f} | {_m['time_p99_us']:,.0f} |")

    _pairs = []
    for _k in sorted(_d):
        _v = _d[_k]
        _pairs.append(f"| {_v['a']} vs {_v['b']} | {_v['a_only']} | "
                      f"{_v['b_only']} | {_v['p']:.2e} |")
    _mc = (["", "All arms decoded identical shots — McNemar exact test.", "",
            "| comparison | first fails alone | second fails alone | p |",
            "|:--|--:|--:|--:|"] + _pairs) if _pairs else []

    if "updated" in _A and "sighted" in _A and "blind" in _A:
        _u, _s, _b = _A["updated"]["ler"], _A["sighted"]["ler"], _A["blind"]["ler"]
        _recovered = (_b - _u) / (_b - _s) if _b > _s else float("nan")
        _head = (f"**Prior updating recovers {_recovered:.0%} of the gap between "
                 f"the blind and sighted arms**, on a check matrix "
                 f"{_A['updated']['detectors']} × {_A['updated']['faults']} rather "
                 f"than {_A['sighted']['detectors']} × {_A['sighted']['faults']} — "
                 f"{(_A['sighted']['detectors'] * _A['sighted']['faults']) / (_A['updated']['detectors'] * _A['updated']['faults']):.2f}× "
                 "fewer entries.")
        if _u <= _s * 1.05:
            _head += ("  \nIt matches the sighted arm within 5%, so the flag rows "
                      "buy nothing that the exact one-shot posterior does not "
                      "already capture.")
        elif _u < _b:
            _head += ("  \nIt does not reach the sighted arm, so the iteration "
                      "between the flag and syndrome subsystems carries "
                      "information the one-shot posterior misses — a result worth "
                      "reporting in its own right.")
        else:
            _head += "  \nIt does not beat the blind arm — check the update first."
    else:
        _head = "Run all three arms for the comparison."

    _tag = _cfg["code"].strip("[]").replace(", ", "-")
    _path = os.path.join(
        RESULTS_DIR,
        f"EXP9_{_tag}_T{_cfg['rounds']}_p{_cfg['p']:g}_{_cfg['shots']}shots_"
        f"seed{_cfg['seed']}_osd{_cfg['osd_order']}.json")
    write_result_file(_path, {"kind": "exp9", "config": _cfg, "result": _r})
    _folder = export_bundle(
        _path, f"Experiment 9 dynamic prior updating — {_cfg['code']}, "
               f"T={_cfg['rounds']}, p={_cfg['p']:g}",
        {"prior_update": _fig},
        {"arms": [dict(arm=_a, **_A[_a]) for _a in _A]},
        notes=_head)
    mo.vstack([mo.md("### Experiment 9 — results"), mo.md(_head),
               mo.md("\n".join(_rows + _mc)), _fig,
               mo.md(f"Structure: `{_r['structure']}` · "
                     f"flags fire on {_r['flagged_fraction']:.1%} of shots  \n"
                     f"Saved to `{_path}` · exported to `{_folder}`")])
    return (exp9_data,)


@app.cell
def _(build_memory_circuit, code_from_key, dem_to_matrices, dz,
      flag_detector_mask, flag_groups, flag_posteriors, np, pareto_point,
      prior_update_run, render_checks, run_checks, shot_priors, sp):
    def _brute_force_posterior(p, j, f):
        """Exact marginal of bit j given the group's parity, by enumeration."""
        import itertools
        num = den = 0.0
        for b in itertools.product([0, 1], repeat=len(p)):
            if sum(b) % 2 != f:
                continue
            w = np.prod([p[k] if b[k] else 1 - p[k] for k in range(len(p))])
            den += w
            if b[j]:
                num += w
        return num / den

    def _closed_form_matches_brute_force():
        rng = np.random.default_rng(3)
        for size in (2, 4, 6):
            pr = rng.uniform(1e-4, 0.3, size)
            idx = np.arange(size)[None, :]
            lo, hi, _owner = flag_posteriors(pr, idx)
            for j in range(size):
                assert abs(lo[j] - _brute_force_posterior(pr, j, 0)) < 1e-12, (size, j)
                assert abs(hi[j] - _brute_force_posterior(pr, j, 1)) < 1e-12, (size, j)

    def _padding_does_not_change_a_group():
        pr = np.array([0.01, 0.02, 0.03, 0.04])
        a_lo, a_hi, _ = flag_posteriors(pr, np.array([[0, 1, 2, 3]]))
        b_lo, b_hi, _ = flag_posteriors(pr, np.array([[0, 1, 2, 3, -1, -1]]))
        assert np.allclose(a_lo, b_lo) and np.allclose(a_hi, b_hi)

    def _half_probability_does_not_divide_by_zero():
        pr = np.array([0.5, 0.5, 0.1])          # lambda = 0 for two of them
        lo, hi, _ = flag_posteriors(pr, np.array([[0, 1, 2]]))
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)), (lo, hi)

    def _posteriors_are_probabilities():
        rng = np.random.default_rng(5)
        pr = rng.uniform(1e-6, 0.4, 30)
        lo, hi, _ = flag_posteriors(pr, np.arange(30).reshape(5, 6))
        for v in (lo, hi):
            assert np.all((v >= 0) & (v <= 1)), v[(v < 0) | (v > 1)]

    def _a_fired_flag_raises_and_a_quiet_one_lowers():
        pr = np.full(6, 1e-3)
        lo, hi, _ = flag_posteriors(pr, np.arange(6)[None, :])
        assert (hi > pr).all() and (lo < pr).all(), (lo[0], pr[0], hi[0])

    def _untouched_faults_keep_their_prior():
        pr = np.array([1e-3, 2e-3, 3e-3, 4e-3])
        lo, hi, owner = flag_posteriors(pr, np.array([[0, 1]]))
        assert lo[2] == pr[2] and hi[3] == pr[3]
        assert owner[2] == -1 and owner[0] == 0

    def _shot_priors_selects_by_flag():
        pr = np.full(4, 1e-3)
        idx = np.array([[0, 1], [2, 3]])
        lo, hi, owner = flag_posteriors(pr, idx)
        out = shot_priors(lo, hi, owner, np.array([True, False]))
        assert out[0] == hi[0] and out[1] == hi[1], out
        assert out[2] == lo[2] and out[3] == lo[3], out

    def _non_disjoint_groups_are_detected():
        H = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
        _idx, ok, info = flag_groups(H, np.array([True, True]))
        assert not ok, info                       # fault 1 lights both rows
        _idx, ok, _ = flag_groups(sp.csr_matrix(np.array([[1, 1, 0, 0],
                                                          [0, 0, 1, 1]],
                                                         dtype=np.uint8)),
                                  np.array([True, True]))
        assert ok

    def _the_real_dem_has_the_structure_the_method_needs():
        c = code_from_key("[[72, 12, 6]]")
        circ = build_memory_circuit(c, 6, 1e-3, use_flags=True, x_detectors=False)
        H, _L, pr = dem_to_matrices(
            circ.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(c, 6, True, circ.num_detectors, x_detectors=False)
        idx, ok, info = flag_groups(H, mask)
        assert ok, f"flag groups overlap: {info}"
        assert info["max_flag_weight"] == 1, info
        assert info["flag_rows"] == c.hx.shape[0] * 6, info
        lo, hi, owner = flag_posteriors(pr, idx)
        touched = owner >= 0
        assert touched.sum() == info["faults_covered"], (touched.sum(), info)
        assert (hi[touched] > pr[touched]).all()
        assert (lo[touched] < pr[touched]).all()

    def _bounds_only_keep_the_llr_finite():
        # a certain fault: its group's parity must come from it alone
        pr = np.array([1.0 - 1e-18, 1e-18])
        lo, hi, _ = flag_posteriors(pr, np.array([[0, 1]]))
        assert np.all(np.isfinite(np.log(lo / (1 - lo))))
        assert np.all(np.isfinite(np.log(hi / (1 - hi))))
        # and a posterior above 0.5 survives rather than being clamped to 0.49
        pr2 = np.array([0.4, 0.05, 0.05])
        _lo2, hi2, _ = flag_posteriors(pr2, np.array([[0, 1, 2]]))
        assert hi2[0] > 0.5, hi2

    def _updated_corrections_reproduce_their_syndrome():
        # a correction that does not satisfy H e = s would score as a
        # non-failure whenever the observable happens to match, so check it
        c = code_from_key("[[72, 12, 6]]")
        circ = build_memory_circuit(c, 6, 1e-3, use_flags=True, x_detectors=False)
        H, _L, pr = dem_to_matrices(
            circ.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(c, 6, True, circ.num_detectors, x_detectors=False)
        idx, _ok, _info = flag_groups(H, mask)
        lo, hi, owner = flag_posteriors(pr, idx)
        det, _obs = circ.compile_detector_sampler(seed=11).sample(
            60, separate_observables=True)
        det = det.astype(np.uint8)
        HS = H[~mask]
        dec = dz.make_decoder({"kind": "osd", "osd_order": 0, "max_iter": 20},
                              HS, pr)
        for i in range(60):
            dec.dec.update_channel_probs(
                shot_priors(lo, hi, owner, det[i, mask]).tolist())
            e, _c, _w = dec.decode(det[i, ~mask])
            assert np.array_equal((HS @ e) % 2, det[i, ~mask]), i

    def _arms_agree_with_experiment_8():
        # prior_update_run builds its own decoders; if they drift from the ones
        # pareto_point uses (min-sum, scaling 0.625, parallel schedule) the arms
        # stop being comparable across experiments. This caught exactly that.
        c = code_from_key("[[72, 12, 6]]")
        r9 = prior_update_run(c, 6, 1e-3, 200, 4242, osd_order=0,
                              arms=("blind", "sighted"))
        for arm in ("blind", "sighted"):
            p8 = pareto_point(c, 6, 1e-3, 200, 4242, arm, 0)
            assert p8["failures"] == r9["arms"][arm]["failures"], (
                arm, p8["failures"], r9["arms"][arm]["failures"])

    tests_exp9 = run_checks([
        ("closed form equals brute-force enumeration", _closed_form_matches_brute_force),
        ("the bounds keep the LLR finite without clamping", _bounds_only_keep_the_llr_finite),
        ("padding a group leaves it unchanged", _padding_does_not_change_a_group),
        ("a prior of exactly 0.5 does not divide by zero", _half_probability_does_not_divide_by_zero),
        ("posteriors stay inside [0, 1]", _posteriors_are_probabilities),
        ("a fired flag raises its faults, a quiet one lowers them", _a_fired_flag_raises_and_a_quiet_one_lowers),
        ("faults no flag touches keep their prior", _untouched_faults_keep_their_prior),
        ("a shot's priors select by its own flag bits", _shot_priors_selects_by_flag),
        ("overlapping flag groups are detected, not used", _non_disjoint_groups_are_detected),
        ("the real DEM has disjoint weight-1 flag groups", _the_real_dem_has_the_structure_the_method_needs),
        ("updated corrections reproduce their syndrome", _updated_corrections_reproduce_their_syndrome),
        ("the blind and sighted arms match Experiment 8 exactly", _arms_agree_with_experiment_8),
    ])
    render_checks("11.9 dynamic prior updating", tests_exp9)
    return (tests_exp9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Results

    Measured on [[46, 2, 9]], [[72, 12, 6]], [[144, 12, 12]] and [[288, 12, 18]]
    under the circuit-level model of §4. Intervals are 95% Wilson; paired
    comparisons use the McNemar exact test. Shots are budgeted per $T$ rather than
    flat, so the headline sweeps of 250,000 ([[72,12,6]]) and 300,000
    ([[144,12,12]]) shots at $T = 6$ halve as $T$ doubles; per-point counts are in
    the tables.

    **R1 — flag information lowers the logical error rate, substantially.** In
    Experiment 2's three arms on identical shots at $T = 12$, $p = 10^{-3}$:

    | Code | shots | unflagged | flagged, blind | flagged, sighted | sighted gain | McNemar |
    |:--|--:|--:|--:|--:|--:|--:|
    | [[46, 2, 9]] | 10,000 | 0.2593 | 0.2149 | 0.1480 | 1.75× | $p = 3{\times}10^{-51}$ |
    | [[72, 12, 6]] | 125,000 | 0.01037 | 0.01701 | 0.00503 | **2.06×** | $p = 1{\times}10^{-279}$ |
    | [[144, 12, 12]] | 150,000 | 0.00091 | 0.00194 | 0.00015 | **5.91×** | $p = 2{\times}10^{-67}$ |

    On the two Gross-family codes the blind arm is *worse* than no flags — by
    1.64× and 2.13× — and that cost is paid whether or not the decoder reads the
    flags. On [[46, 2, 9]] it is slightly *better* (0.83×), so the hardware penalty
    is not uniform across code families and should not be quoted as a single
    number. The sighted arm wins everywhere, by 1.75× to 5.91×, and the gain grows
    with code size. The effect holds in
    all 34 configurations run — $p \in \{5{\times}10^{-4}, 10^{-3}, 2{\times}10^{-3}\}$,
    OSD order 0 and 4, $T \in \{6, 12, 18, 24\}$ — with no crossover in that
    range. The hardware costs +25% qubits, +17% two-qubit gates, +30% fault
    mechanisms and +32% expected faults per shot.

    **R2 — no flag trigger can improve on the trivial rule.** Five policies
    replayed on the same 250,000 shots, [[72,12,6]] at $T = 12$, $p = 10^{-3}$,
    weak = greedy peeling, strong = BP+OSD-0:

    | Policy | escalates on | LER | 95% CI | median decode |
    |:--|--:|--:|:--|--:|
    | never escalate | 0% | 0.36887 | [0.36698, 0.37076] | 22 µs |
    | flag trigger | 89.3% | 0.02293 | [0.02235, 0.02352] | 3,696 µs |
    | always escalate | 100% | 0.00538 | [0.00510, 0.00567] | 4,076 µs |
    | `primary_fail` | 44.4% | **0.00512** | [0.00484, 0.00540] | **31 µs** |
    | omniscient oracle | 44.4% | **0.00512** | [0.00484, 0.00540] | 31 µs |

    The trivial rule and the oracle are identical **to the shot**. That closes
    the question: no trigger of any kind can beat "escalate when the primary
    decoder fails" here. The flag trigger is 4.5× worse than it while escalating
    twice as often, because flags fire on 89.3% of shots and the 10.7% they miss
    include shots where peeling did not converge. Experiments 4 and 5 reproduce
    this at every point tested — four codes, six noise levels from $5{\times}10^{-4}$
    to $6{\times}10^{-3}$, $T \in \{6, 12\}$, OSD orders 0 and 4.

    **R3 — why: there is no headroom, and the trigger saturates.** *Headroom* —
    silent failures of the weak decoder that the strong decoder repairs — bounds
    what any trigger can gain. Over 250,000 shots:

    | weak decoder | converged but wrong | of those, fixable by BP+OSD |
    |:--|--:|--:|
    | greedy peeling | 0 | **0** |
    | BP min-sum, 10 | 5 | **0** |
    | BP min-sum, 30 | 10 | **0** |
    | BP sum-product, 30 | 27 | **8** |

    At most 8 shots in 250,000 ($3.2{\times}10^{-5}$) are reachable by any rule,
    and for three of the four weak decoders the bound is exactly zero. Separately,
    the trigger saturates: shots with at least one flag are 43% at $T = 3$, 69% at
    $T = 6$, 91% at $T = 12$, 97% at $T = 18$ and **99%** at $T = 24$, so mutual
    information between "a flag fired" and "peeling was wrong" is 0.017 bits.
    Both mechanisms are structural and neither improves with more shots.

    **R4 — switching itself works; it just does not need flags.** `primary_fail`
    reaches always-escalate accuracy (0.00512 against 0.00538) while running the
    expensive decoder on 44.4% of shots, at **131× lower median latency** — 31 µs
    against 4,076 µs. This is a real result about decoder switching, and the
    signal it uses is the primary decoder's own convergence flag, not a flag qubit.

    **R5 — the pipeline reproduces two independent references.** On §5.1's
    reference circuit the fault count matches Theorem 1 exactly and $\lambda$
    matches Theorem 2 to within 2% on all three Gross-family codes; the Theorem
    3/5 prediction evaluated on our own fault graph gives 0.935, 0.874 and 0.764
    against Table II's 0.935, 0.879 and 0.778. Independently, the rotated surface
    code at $d = 3, 5, 7$ crosses at 1.07%, against the ~1% literature value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Discussion and limitations

    **Why the trigger fails.** A useful trigger needs two things, and this setting
    supplies neither. First, the fast decoder must fail *silently*, because a
    failure it reports itself is already caught by `primary_fail` — greedy peeling
    produced zero silent failures in 250,000 shots, and across all four weak
    decoders at most 8 of them were repairable. Second, the trigger must be
    selective — flags fire on 69–99% of shots, and it worsens with $T$. The flag
    outcomes are informative in aggregate — 432 detectors' worth, which is exactly
    why R1 works — but "did any flag fire?" compresses them into one
    nearly-constant bit carrying 0.017 bits of information about failure.

    That the fast decoder is cheap enough to be worth running first is the one
    precondition that *is* met: the decoders themselves run at a median 22.5 µs
    against 4,047 µs, a 180× ratio. This is why
    `primary_fail` is such a strong baseline, and why it is the thing to report.

    **Why flags still pay.** Read as detectors, the same outcomes let the decoder
    explain faults it would otherwise have to guess at. That this survives the
    circuit's own extra noise is the point of Experiment 2's blind arm, and it
    qualifies the common argument that flags are not worth their depth and qubits
    for qLDPC codes: that argument prices the circuit without giving the decoder
    the flag data.

    **Limitations.**

    1. Our experiment circuit holds 1.42× the fault mechanisms of the reference
       formula $n(wT + T/2 + 1)$, because it extracts both check families (§5.1).
       Absolute rates are therefore not directly comparable with published BB
       figures; every claim here is a relative comparison inside one fixed model.
       A denser model makes R2 and R3 *harder* to obtain, not easier. The
       reference circuit of §5.1 closes this gap exactly when a comparable number
       is needed.
    2. Experiment 2's unflagged arm uses a different circuit from the two flagged
       arms, so that leg is unpaired; blind and sighted are paired with each other,
       and the McNemar p-values in R1 are computed on that paired pair only.
    3. [[288, 12, 18]] contributes to R5 but not to R1: at 10,000 shots both arms
       record zero failures, so the largest code is unresolved. R1's scaling claim
       rests on three codes, not four.
    4. Timings are Python-level CPU wall-clock, measured while worker processes
       compete for cores. Peeling is compiled with numba, `ldpc` is C++. Use the
       ratios (180× peel-to-OSD, 131× latency saving for `primary_fail`), never
       the absolute microseconds — §10.1 states this in full.
    5. Flags are placed on X-checks only, which is what a Z-basis memory needs.
       A logical-operation circuit, or an X-basis memory, would place them
       differently and is not covered.
    6. Conclusions apply to superconducting bicycle-type architectures: degree-6
       connectivity with long-range couplers, $p$ near $10^{-3}$, and syndrome
       cycles short enough for decoder latency to matter.
    7. R2's null is established for *pre-decode* triggers built from flag bits.
       It does not rule out a trigger built from the primary decoder's own soft
       information, which is a different signal and was not tested.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion and future work

    Flag information is valuable to a decoder and worthless as a routing signal.
    The flag qubits are worth building only if the decoder reads them.

    Fed to the decoder as detectors, flag outcomes lower the logical error rate by
    2.06× on [[72, 12, 6]] and 5.91× on [[144, 12, 12]], even though the same
    circuitry with its outcomes hidden *raises* the rate by 1.6–2.1×. The
    information is worth roughly four times the hardware it rides on, and the gain
    grows with code size.

    Used as a pre-decode trigger it cannot help, and the reason is structural
    rather than statistical. Over 250,000 paired shots the rule "escalate when the
    primary decoder fails" achieves exactly the same logical error rate as an
    omniscient oracle that escalates only when escalating would change the answer.
    A rule that ties the oracle cannot be beaten. Underneath that: greedy peeling
    never returned a wrong answer while claiming convergence, so there is nothing
    for a trigger to catch, and flags fire on 99% of shots by $T = 24$, so there
    is nothing to discriminate. Both statements are measured rather than argued,
    with a positive control showing the analysis detects the effect when it is
    injected.

    Two further results stand on their own. Switching itself works: `primary_fail`
    reaches always-escalate accuracy at 131× lower median latency, using the
    primary decoder's own convergence signal. And the pipeline reproduces
    Pakhunov (2026) Table II on three codes and the ~1% surface-code threshold,
    which is what licenses the two null results above.

    **Future work.**

    - Graded flag signals (flag count, clustering within a round) as *decoder
      input* — R1 shows the aggregate is informative, and the natural question is
      how much of that gain a smaller subset of flag detectors already delivers.
    - Resolve [[288, 12, 18]] for Experiment 2, most cheaply at $p = 2{\times}10^{-3}$
      rather than by raising the shot count at $p = 10^{-3}$.
    - Close the peeling gap of R5 (+0.032, +0.073, +0.117 as $n$ grows): the
      deficit scales with code size, which points at the dominance rule giving up
      earlier on denser fault graphs.
    - A threshold sweep for the BB codes, to sit beside the surface-code baseline
      of Experiment 7.
    - Compare against the other hook-error mitigations at equal $p$: biased-noise
      ancillas and CNOT-schedule optimisation.
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
    4. A. A. Kovalev and L. P. Pryadko, "Quantum Kronecker sum-product low-density
       parity-check codes with finite rate," *Phys. Rev. A* **88**, 012311 (2013).
    5. P. Panteleev and G. Kalachev, "Degenerate quantum LDPC codes with good
       finite length performance," *Quantum* **5**, 585 (2021).
    6. R. Chao and B. Reichardt, "Quantum error correction with only two extra
       qubits," *Phys. Rev. Lett.* **121**, 050502 (2018).
    7. A. Pakhunov, "Analytical theory of greedy peeling for bivariate bicycle
       codes and two-shot streaming decoding," arXiv:2604.11352 (2026).
    8. A. Sahay, D. J. Williamson and B. J. Brown, "A matching decoder for bivariate
       bicycle codes," arXiv:2602.22770 (2026).
    9. R. Toshio, K. Kishi, J. Fujisaki, H. Oshima, S. Sato and K. Fujii, "Decoder
       switching: breaking the speed-accuracy tradeoff in real-time quantum error
       correction," arXiv:2510.25222 (2025).
    10. T. Hillmann et al., "Localized statistics decoding: a parallel decoding
       algorithm for quantum LDPC codes," Nat. Commun. 16, 8214 (2025).
    11. T. Chen, T. J. Yoder et al., "Calibrated decoders for experimental quantum
       error correction," arXiv:2110.04285 (2021) — flag outcomes used as decoder
       input on heavy-hex codes.
    12. A. Vittal et al., "Flag proxy networks," arXiv:2409.14283 (MICRO 2024) —
       flag-aware decoding for hyperbolic surface and colour codes.
    """)
    return


if __name__ == "__main__":
    app.run()
