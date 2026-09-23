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
    11. Experiments: E0 validation and flag cost, E1 ablation, E2 sweep, E3 decoder zoo
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

    **TODO:** a circuit diagram for one X-check with its flag.
    """)
    return


@app.cell
def _(np, sp, stim):
    def _check_schedule(H, lattice):
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


    def build_memory_circuit(code, rounds, p, use_flags=False, idle_noise=False):
        """
        Z-basis memory experiment for a CSS code under circuit-level noise.

        Detector order per round t: m_z Z-check detectors, m_x X-check detectors
        (t >= 1 only), m_x flag detectors (if use_flags); then m_z final detectors.
        Schedule: all X-check CNOT layers, then all Z-check CNOT layers (always
        deterministic). Flag CNOTs sit after the first and before the last X layer.
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

        sx = _check_schedule(code.hx, code.lattice)   # (w_x, m_x)
        sz = _check_schedule(code.hz, code.lattice)   # (w_z, m_z)
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
            if t > 0:                                                 # 2. X checks
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


    def flag_detector_mask(code, rounds, use_flags, num_detectors):
        """Boolean array of length num_detectors, True exactly at flag detectors."""
        mx, mz = code.hx.shape[0], code.hz.shape[0]
        mask = np.zeros(num_detectors, dtype=bool)
        if not use_flags:
            return mask
        pos = 0
        for t in range(rounds):
            pos += mz                       # Z-check detectors
            if t > 0:
                pos += mx                   # X-check detectors
            mask[pos:pos + mx] = True       # flag detectors
            pos += mx
        return mask


    def expected_num_detectors(code, rounds, use_flags):
        """Detector count implied by the ordering contract above."""
        mx, mz = code.hx.shape[0], code.hz.shape[0]
        per_round_flags = mx if use_flags else 0
        return mz * rounds + mx * (rounds - 1) + per_round_flags * rounds + mz

    return build_memory_circuit, expected_num_detectors, flag_detector_mask


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
    the primary is right on shots where the secondary errs: E1 at 25,000 shots
    found peeling correct and BP+OSD-0 wrong on 14 shots, and never the reverse.
    So a policy beating `always` is not automatically a bug, but it must be
    confirmed with a paired test on identical shots (E3 does this).
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
            ax.set_xlabel("physical error rate $p$")
            ax.grid(True, which="both", alpha=0.3)
        a.set_yscale("log")
        a.set_ylabel("logical error rate")
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
        ax.set_xlabel("fraction of shots")
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
        ax.set_ylabel("logical error rate")
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

    - a **data file** (`.json` for E1/E2, `.npz` for E3) that the notebook can
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
        ("a reloaded E1 run keeps its per-shot samples", _e1_reload_keeps_latency),
        ("an interrupted E3 run resumes with identical results", _interrupted_run_resumes),
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
    tests_gb,
    tests_gf2,
    tests_repro,
    tests_stats,
    tests_switch,
    tests_zoo,
):
    _sections = [
        ("2. GF(2)", tests_gf2), ("3. BB codes", tests_codes), ("3. GB codes", tests_gb),
        ("4. Circuit", tests_circuit), ("5. DEM", tests_dem),
        ("6. Decoders", tests_decoders), ("7. Switch policies", tests_switch),
        ("8. Statistics", tests_stats), ("9. Reproducibility", tests_repro),
        ("9b. Export & resume", tests_export),
        ("11. Experiment drivers", tests_experiments),
        ("11b. Decoder zoo", tests_zoo),
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
def _(
    BB_PRESETS,
    GB_PRESETS,
    mo,
    os,
):
    ui_code = mo.ui.dropdown(list(BB_PRESETS) + list(GB_PRESETS), value="[[72, 12, 6]]",
                             label="Code")
    ui_rounds = mo.ui.slider(1, 12, value=6, label="Rounds T")
    ui_p = mo.ui.dropdown(["5e-4", "1e-3", "2e-3", "3e-3", "5e-3"], value="1e-3", label="p")
    ui_shots = mo.ui.slider(100, 200_000, step=100, value=500, label="Shots")
    ui_flags = mo.ui.switch(value=True, label="Flag qubits")
    ui_osd = mo.ui.slider(0, 4, value=0, label="OSD order")
    ui_seed = mo.ui.number(value=20260921, label="Seed")
    _cores = os.cpu_count() or 1
    ui_workers = mo.ui.slider(1, max(2, _cores), value=min(12, max(1, _cores - 4)),
                              label=f"CPU workers (of {_cores})")
    mo.vstack([mo.md("### Experiment configuration"),
               mo.hstack([ui_code, ui_p, ui_flags]),
               mo.hstack([ui_rounds, ui_shots]),
               mo.hstack([ui_osd, ui_seed]),
               ui_workers,
               mo.md("*Decoding runs in parallel from about 2,000 shots upward; below that "
                     "the per-process setup costs more than it saves.*")])
    return ui_code, ui_flags, ui_osd, ui_p, ui_rounds, ui_seed, ui_shots, ui_workers


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
):
    config = ExperimentConfig(
        code=ui_code.value, rounds=int(ui_rounds.value), p=float(ui_p.value),
        shots=int(ui_shots.value), seed=int(ui_seed.value),
        use_flags=bool(ui_flags.value), osd_order=int(ui_osd.value),
        workers=int(ui_workers.value))
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
        E1. Decode every shot ONCE with the weak decoder (peeling) and once with the
        strong one (BP+OSD at config.osd_order), then derive one policy per trigger.
        This matches running SwitchPolicy on every shot -- the decoder-zoo tests
        check that shot by shot -- but it uses config.workers CPU processes and
        never decodes the same shot twice.

        Returns dict trigger -> Metrics.
        """
        code = code_from_key(config.code)
        circuit = build_memory_circuit(code, config.rounds, config.p, use_flags=config.use_flags)
        H, L, priors = dem_to_matrices(circuit.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(code, config.rounds, config.use_flags, circuit.num_detectors)
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
        """E2. run_ablation at each p in `ps`. Returns dict p -> (dict trigger -> Metrics)."""
        import dataclasses
        ps = [float(p) for p in ps]
        steps = mo.status.progress_bar(ps, title="E2 sweep", show_eta=True) if progress else ps
        return {p: run_ablation(dataclasses.replace(config, p=p), keep_samples=keep_samples)
                for p in steps}

    def result_path(kind, config):
        """Deterministic file name: the same config always maps to the same file."""
        tag = config.code.strip("[]").replace(", ", "-").replace(" ", "")
        return os.path.join(RESULTS_DIR, f"{kind}_{tag}_T{config.rounds}_p{config.p:g}_"
                            f"{config.shots}shots_seed{config.seed}"
                            f"{'_flags' if config.use_flags else ''}_osd{config.osd_order}.json")

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
    validation_run,
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
        a = result_path("E1", _small)
        assert a == result_path("E1", _small) and a.endswith(".json")
        import dataclasses
        assert a != result_path("E1", dataclasses.replace(_small, p=3e-3)), \
            "different configs must not share a file"
        assert code_from_key("[[48, 6, 8]]").n == 48

    def _reference_formula():
        code = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        # n(wT + T/2 + 1) with n = 72, w = 3 (qubit degree), T = 4 -> 72 * 15 = 1080
        assert reference_fault_count(code, 4) == 1080, reference_fault_count(code, 4)

    def _validation_runs():
        import dataclasses
        cfg = dataclasses.replace(_small, rounds=2, shots=40)
        v = validation_run(cfg, target=0.5)
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
        ("E0 validation runs and reports V1, V2, V3", _validation_runs),
        ("run_ablation returns every trigger, with sane escalation", _ablation_triggers),
        ("flag triggers are skipped when flags are off", _no_flags),
        ("run_ablation works on a GB code", _gb_code),
        ("run_sweep returns one ablation per p", _sweep),
        ("result files are named deterministically per config", _paths),
    ])
    render_checks("experiment drivers", tests_experiments)
    return (tests_experiments,)


@app.cell
def _(mo):
    run_e1 = mo.ui.run_button(label="Run E1 — ablation")
    run_e2 = mo.ui.run_button(label="Run E2 — sweep (slow)")
    mo.hstack([run_e1, run_e2])
    return run_e1, run_e2


@app.cell
def _(config, core_ready, mo, result_path, run_ablation, run_e1, save_results):
    # Always define e1_run (None when not run), so the view cell can use either
    # a fresh run or a loaded one.
    e1_run = None
    if not core_ready:
        _out = mo.md("*E1 locked: finish the implementation (see §10).*")
    elif not run_e1.value:
        _out = mo.md("*Press **Run E1** to start, or load a previous run below.*")
    else:
        _res = run_ablation(config)
        _path = save_results(result_path("E1", config), config, _res, include_samples=True)
        e1_run = dict(config=config, results=_res, path=_path)
        _out = mo.md(f"E1 finished. Saved to `{_path}`.")
    _out
    return (e1_run,)


@app.cell
def _(RESULTS_DIR, e1_run, glob, mo, os):
    _ = e1_run                                    # refresh the list after a new run
    _files = sorted(glob.glob(os.path.join(RESULTS_DIR, "E1_*.json")))
    ui_e1_file = mo.ui.dropdown({os.path.basename(f): f for f in _files},
                                value=os.path.basename(_files[-1]) if _files else None,
                                label="Previous E1 run")
    load_e1_btn = mo.ui.run_button(label="Load E1")
    mo.hstack([ui_e1_file, load_e1_btn]) if _files else mo.md("*No saved E1 runs yet.*")
    return load_e1_btn, ui_e1_file


@app.cell
def _(load_e1_btn, load_results, ui_e1_file):
    e1_loaded = None
    if load_e1_btn.value and ui_e1_file.value:
        _cfg, _res = load_results(ui_e1_file.value)
        e1_loaded = dict(config=_cfg, results=_res, path=ui_e1_file.value)
    return (e1_loaded,)


@app.cell
def _(
    e1_loaded,
    e1_run,
    export_bundle,
    metrics_rows,
    mo,
    plot_ablation,
    plot_latency,
    plot_outcomes,
    plot_tradeoff,
):
    _e1 = e1_run if e1_run is not None else e1_loaded
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
    _title = f"E1 — {_cfg.code}, T={_cfg.rounds}, p={_cfg.p}, {_cfg.shots} shots"
    _folder = export_bundle(_e1["path"], _title, dict(_figs),
                            {"metrics": metrics_rows(_res, **vars(_cfg))})
    mo.vstack([
        mo.md(f"### {_title}"),
        mo.md(_table),
        mo.md("> ℹ️ " + ", ".join(_beats) + " beat `always`. This is possible when the weak "
              "decoder is right where the strong one errs — confirm it with a paired test on "
              "the same shots (E3) before reporting it.") if _beats else mo.md(""),
        mo.md(f"Data: `{_e1['path']}` · **Exported** figures (PNG + PDF), CSV and report to "
              f"`{_folder}`"),
        *[_f for _f in _figs.values()],
    ])
    return


@app.cell
def _(config, core_ready, mo, np, result_path, run_e2, run_sweep, save_sweep):
    e2_run = None
    if not core_ready:
        _out = mo.md("*E2 locked: finish the implementation (see §10).*")
    elif not run_e2.value:
        _out = mo.md("*Press **Run E2** to start, or load a previous run below.*")
    else:
        _ps = np.logspace(np.log10(5e-4), np.log10(6e-3), 6)
        _sweep = run_sweep(config, _ps)
        _path = save_sweep(result_path("E2", config), config, _sweep)
        e2_run = dict(config=config, sweep=_sweep, path=_path)
        _out = mo.md(f"E2 finished. Saved to `{_path}`.")
    _out
    return (e2_run,)


@app.cell
def _(RESULTS_DIR, e2_run, glob, mo, os):
    _ = e2_run
    _files = sorted(glob.glob(os.path.join(RESULTS_DIR, "E2_*.json")))
    ui_e2_file = mo.ui.dropdown({os.path.basename(f): f for f in _files},
                                value=os.path.basename(_files[-1]) if _files else None,
                                label="Previous E2 run")
    load_e2_btn = mo.ui.run_button(label="Load E2")
    mo.hstack([ui_e2_file, load_e2_btn]) if _files else mo.md("*No saved E2 runs yet.*")
    return load_e2_btn, ui_e2_file


@app.cell
def _(load_e2_btn, load_sweep, ui_e2_file):
    e2_loaded = None
    if load_e2_btn.value and ui_e2_file.value:
        _cfg, _sweep = load_sweep(ui_e2_file.value)
        e2_loaded = dict(config=_cfg, sweep=_sweep, path=ui_e2_file.value)
    return (e2_loaded,)


@app.cell
def _(e2_loaded, e2_run, export_bundle, metrics_rows, mo, plot_sweep):
    _e2 = e2_run if e2_run is not None else e2_loaded
    mo.stop(_e2 is None)
    _cfg, _sweep = _e2["config"], _e2["sweep"]
    _rows = [r for _p, _res in _sweep.items() for r in metrics_rows(_res, p=_p)]
    _fig = plot_sweep(_sweep)
    _title = f"E2 — {_cfg.code}, T={_cfg.rounds}, {_cfg.shots} shots/point"
    _folder = export_bundle(_e2["path"], _title, {"sweep": plot_sweep(_sweep)},
                            {"sweep_metrics": _rows})
    mo.vstack([mo.md(f"### {_title}"),
               mo.md(f"Data: `{_e2['path']}` · **Exported** to `{_folder}`"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11a. E0 — validation, and what the flags cost

    A negative result is only believable if the setup could have found the effect.
    E0 is the evidence for that, and it runs before any conclusion is drawn.

    **V1 — how big is our fault model?** Pakhunov (2026) gives the number of DEM
    fault mechanisms for a BB memory as $n(wT + T/2 + 1)$ under his noise model.
    Ours uses full two-qubit depolarising noise, so it is denser. This check
    reports the ratio, which tells you how far the two models are apart and
    therefore how much of a published number you should expect to reproduce.

    **V2 — do we reproduce a published number?** Enter a literature figure (for
    example, peeling resolving 93.5% of shots on [[72, 12, 6]] at $p=10^{-3}$,
    $T=12$) and this reports ours with a 95% interval and whether the two agree.
    A disagreement is informative, not fatal: with V1 in hand you can say *why*.

    **V3 — what do the flags cost?** Flag qubits add ancillas and CNOTs, so they
    add noise. This decodes the same configuration with and without flags using
    the same strong decoder and compares the logical error rates. If flags make
    accuracy worse, a flag trigger has to buy back that loss before it can help.

    *Important:* this section does not reimplement anyone else's noise model. It
    measures the distance between ours and a published reference, which is what a
    reader needs in order to weigh the rest of the thesis.
    """)
    return


@app.cell
def _(mo):
    ui_v2_target = mo.ui.number(0.0, 1.0, value=0.935, step=0.001,
                                label="Literature target: fraction of shots the weak decoder resolves")
    ui_v2_source = mo.ui.text(value="Pakhunov (2026), Table II, [[72,12,6]], p=1e-3, T=12",
                              label="Source", full_width=True)
    run_e0 = mo.ui.run_button(label="Run E0 — validation")
    mo.vstack([mo.hstack([ui_v2_target, run_e0]), ui_v2_source,
               mo.md("*Uses the E1 configuration above (code, T, p, shots, workers). "
                     "Set T and p to match the source you are comparing against.*")])
    return run_e0, ui_v2_source, ui_v2_target


@app.cell
def _(build_memory_circuit, code_from_key, dem_to_matrices, dz, flag_detector_mask, np, wilson):
    def reference_fault_count(code, rounds):
        """Pakhunov's count for a BB memory: n(wT + T/2 + 1), w = qubit degree."""
        w = int(np.asarray(code.hx.sum(axis=0)).max())
        return int(code.n * (w * rounds + rounds / 2 + 1))

    def validation_run(config, target, on_chunk=None):
        """V1, V2 and V3 on one configuration. Returns a dict of plain numbers."""
        code = code_from_key(config.code)
        out = {"code": config.code, "rounds": config.rounds, "p": config.p,
               "shots": config.shots, "target": float(target)}
        arms = {}
        for flags in (True, False):
            circ = build_memory_circuit(code, config.rounds, config.p, use_flags=flags)
            H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
            det, obs = circ.compile_detector_sampler(seed=config.seed).sample(
                config.shots, separate_observables=True)
            names = [("weak", {"kind": "peel"}),
                     ("strong", {"kind": "osd", "osd_order": config.osd_order, "max_iter": 20})]
            workers = dz.parallel_workers(config.shots, config.workers)
            res, _ = dz.run_zoo(H, L, pr, det, obs, names, workers=workers,
                                chunk_size=dz.chunk_for(config.shots, workers),
                                on_chunk=on_chunk)
            mask = flag_detector_mask(code, config.rounds, flags, circ.num_detectors)
            arms[flags] = dict(
                faults=H.shape[1], detectors=circ.num_detectors,
                flagged_fraction=float((det[:, mask].sum(1) > 0).mean()) if flags else 0.0,
                weak_resolved=wilson(int(res["weak"]["conv"].sum()), config.shots),
                weak_ler=wilson(int(res["weak"]["fail"].sum()), config.shots),
                strong_ler=wilson(int(res["strong"]["fail"].sum()), config.shots),
                silent=int((res["weak"]["conv"] & res["weak"]["fail"]).sum()))

        # V1: how dense is our fault model compared with the reference formula?
        ref = reference_fault_count(code, config.rounds)
        out["v1"] = dict(reference_faults=ref, our_faults=arms[True]["faults"],
                         our_faults_unflagged=arms[False]["faults"],
                         ratio=arms[False]["faults"] / ref)
        # V2: do we reproduce the published figure?
        rate, lo, hi = arms[True]["weak_resolved"]
        out["v2"] = dict(measured=rate, ci_low=lo, ci_high=hi, target=float(target),
                         agrees=bool(lo <= target <= hi),
                         measured_unflagged=arms[False]["weak_resolved"][0])
        # V3: the cost of the flags (independent samples: different circuits)
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

    return reference_fault_count, validation_run


@app.cell
def _(config, core_ready, mo, run_e0, ui_v2_target, validation_run):
    e0_result = None
    if not core_ready:
        _out = mo.md("*E0 locked until every test passes.*")
    elif not run_e0.value:
        _out = mo.md("*Press **Run E0** to validate this configuration.*")
    else:
        _n_chunks = max(1, 2 * (-(-config.shots // 250)))
        with mo.status.progress_bar(total=_n_chunks, title="E0: validating", show_eta=True) as _bar:
            e0_result = validation_run(config, ui_v2_target.value, on_chunk=_bar.update)
        _out = mo.md("E0 finished.")
    _out
    return (e0_result,)


@app.cell
def _(RESULTS_DIR, e0_result, glob, mo, os, v4_result):
    _ = (e0_result, v4_result)               # re-list after a new run

    def _picker(prefix, label):
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{prefix}_*.json")))
        return mo.ui.dropdown({os.path.basename(f): f for f in files},
                              value=os.path.basename(files[-1]) if files else None,
                              label=label), files

    ui_e0_file, _f0 = _picker("E0", "Saved E0 run")
    ui_v4_file, _f4 = _picker("V4", "Saved V4 run")
    ui_v5_file, _f5 = _picker("V5", "Saved V5 run")
    load_e0_btn = mo.ui.run_button(label="Load E0")
    load_v4_btn = mo.ui.run_button(label="Load V4")
    load_v5_btn = mo.ui.run_button(label="Load V5")
    mo.vstack([mo.md("#### Load a saved validation run"),
               mo.hstack([ui_e0_file, load_e0_btn]) if _f0 else mo.md("*No saved E0 runs yet.*"),
               mo.hstack([ui_v4_file, load_v4_btn]) if _f4 else mo.md("*No saved V4 runs yet.*"),
               mo.hstack([ui_v5_file, load_v5_btn]) if _f5 else mo.md("*No saved V5 runs yet.*")])
    return (
        load_e0_btn,
        load_v4_btn,
        load_v5_btn,
        ui_e0_file,
        ui_v4_file,
        ui_v5_file,
    )


@app.cell
def _(load_e0_btn, load_v4_btn, load_v5_btn, read_result_file, ui_e0_file, ui_v4_file, ui_v5_file):
    # Each loader returns None until its button is pressed, so the display cells can
    # take whichever of (fresh run, loaded file) exists.
    e0_loaded = v4_loaded = v5_loaded = None
    if load_e0_btn.value and ui_e0_file.value:
        _d = read_result_file(ui_e0_file.value, "validation")
        e0_loaded = dict(_d["config"], path=ui_e0_file.value, source=_d.get("source", ""),
                         **_d["checks"])
    if load_v4_btn.value and ui_v4_file.value:
        _d = read_result_file(ui_v4_file.value, "validation")
        v4_loaded = dict(_d["config"], path=ui_v4_file.value, arms=_d["checks"]["v4"])
    if load_v5_btn.value and ui_v5_file.value:
        _d = read_result_file(ui_v5_file.value, "validation")
        v5_loaded = dict(_d["config"], path=ui_v5_file.value, rows=_d["checks"]["v5"])
    return e0_loaded, v4_loaded, v5_loaded


@app.cell
def _(
    RESULTS_DIR,
    e0_loaded,
    e0_result,
    export_bundle,
    mo,
    os,
    plt,
    ui_v2_source,
    write_result_file,
):
    e0_data = e0_result if e0_result is not None else e0_loaded
    mo.stop(e0_data is None, mo.md("*Run E0, or load a saved run above.*"))
    _v1, _v2, _v3 = e0_data["v1"], e0_data["v2"], e0_data["v3"]
    _source = e0_data.get("source") or ui_v2_source.value

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4.2))
    _a.bar([0], [_v2["measured"]], color="#1f77b4", width=0.5)
    _a.errorbar([0], [_v2["measured"]],
                yerr=[[_v2["measured"] - _v2["ci_low"]], [_v2["ci_high"] - _v2["measured"]]],
                fmt="none", color="k", capsize=6)
    _a.axhline(_v2["target"], color="#d62728", ls="--", label=f"literature: {_v2['target']:.3f}")
    _a.set_xticks([0], ["this notebook"])
    _a.set_ylim(0, 1)
    _a.set_ylabel("fraction of shots resolved by the weak decoder")
    _a.set_title(f"V2 — {'agrees with' if _v2['agrees'] else 'differs from'} the published value")
    _a.legend(fontsize=8)
    for _i, (_lab, _m) in enumerate([("with flags", _v3["ler_flagged"]),
                                     ("without flags", _v3["ler_unflagged"])]):
        _b.errorbar([_i], [_m[0] if _m[0] > 0 else _m[2]],
                    yerr=[[max(_m[0] - _m[1], 0)], [max(_m[2] - _m[0], 0)]],
                    fmt="o" if _m[0] > 0 else "v", color="#2ca02c", capsize=6, ms=8)
    _b.set_xticks([0, 1], ["with flags", "without flags"])
    _b.set_yscale("log")
    _b.set_xlim(-0.5, 1.5)
    _b.set_ylabel("logical error rate (strong decoder)")
    _b.set_title(f"V3 — {_v3['effect']}")
    _b.grid(True, which="both", axis="y", alpha=0.3)
    _fig.tight_layout()

    _rows = [
        dict(check="V1 fault-model density", value=f"{_v1['ratio']:.1f}×",
             detail=f"{_v1['our_faults_unflagged']:,} faults vs {_v1['reference_faults']:,} "
                    f"from n(wT + T/2 + 1)"),
        dict(check="V2 reproduces literature",
             value="yes" if _v2["agrees"] else "no",
             detail=f"ours {_v2['measured']:.3f} [{_v2['ci_low']:.3f}, {_v2['ci_high']:.3f}] "
                    f"vs {_v2['target']:.3f} ({_source})"),
        dict(check="V3 cost of flags", value=_v3["effect"],
             detail=f"LER {_v3['ler_flagged'][0]:.2e} with flags vs "
                    f"{_v3['ler_unflagged'][0]:.2e} without; "
                    f"+{_v3['extra_faults']:,} faults, +{_v3['extra_detectors']} detectors; "
                    f"flags fire on {_v3['flagged_fraction']:.1%} of shots"),
        dict(check="silent failures (weak decoder)",
             value=f"{_v3['silent_flagged']} flagged / {_v3['silent_unflagged']} unflagged",
             detail="the only failures a flag trigger could ever catch"),
    ]
    _tag = e0_data["code"].strip("[]").replace(", ", "-")
    _path = write_result_file(
        os.path.join(RESULTS_DIR, f"E0_{_tag}_T{e0_data['rounds']}_p{e0_data['p']:g}_"
                                  f"{e0_data['shots']}shots.json"),
        {"kind": "validation",
         "config": {k: e0_data[k] for k in ("code", "rounds", "p", "shots", "target")},
         "source": _source,
         "checks": {k: e0_data[k] for k in ("v1", "v2", "v3")}})
    _folder = export_bundle(_path, f"E0 validation — {e0_data['code']}, T={e0_data['rounds']}, "
                            f"p={e0_data['p']}", {"validation": _fig}, {"checks": _rows},
                            notes=f"Literature source: {ui_v2_source.value}")
    mo.vstack([
        mo.md("### E0 — validation results"),
        mo.md("| check | verdict | detail |\n|:--|:--|:--|\n"
              + "\n".join(f"| {r['check']} | {r['value']} | {r['detail']} |" for r in _rows)),
        mo.md(f"Saved to `{_path}` · exported to `{_folder}`"),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### V4 — why do flags help? Hardening or information?

    If E0's V3 shows flags improving accuracy, there are two possible reasons and
    they mean very different things:

    - **Circuit hardening.** The flag CNOTs shorten the window in which an ancilla
      fault can spread, so fewer hook errors reach the data. This is the classic
      flag-qubit mechanism (Chao & Reichardt).
    - **Extra information.** The flagged circuit simply hands the decoder 432 more
      detectors, and a decoder with more syndrome bits does better.

    Three arms separate them, all decoded with the same strong decoder:

    | arm | circuit | decoder sees |
    |---|---|---|
    | unflagged | no flag qubits | all detectors |
    | flagged, blind | with flag qubits | flag detector rows **removed** from H |
    | flagged, sighted | with flag qubits | all detectors |

    Blind ≈ sighted means hardening. Blind ≈ unflagged means information. The
    syndrome bits are removed from the decoding problem rather than zeroed: zeroing
    would feed the decoder a syndrome that never occurred.
    """)
    return


@app.cell
def _(mo):
    run_v4 = mo.ui.run_button(label="Run V4 — hardening vs information")
    run_v5 = mo.ui.run_button(label="Run V5 — flagged fraction vs rounds")
    mo.hstack([run_v4, run_v5])
    return run_v4, run_v5


@app.cell
def _(
    build_memory_circuit,
    code_from_key,
    config,
    core_ready,
    dem_to_matrices,
    dz,
    flag_detector_mask,
    mo,
    np,
    run_v4,
    wilson,
):
    def _arm(H, L, priors, det, obs, workers):
        names = [("strong", {"kind": "osd", "osd_order": config.osd_order, "max_iter": 20})]
        res, _ = dz.run_zoo(H, L, priors, det, obs, names, workers=workers,
                            chunk_size=dz.chunk_for(len(det), workers))
        k = int(res["strong"]["fail"].sum())
        return dict(failures=k, ler=wilson(k, len(det)), detectors=H.shape[0], faults=H.shape[1])

    v4_result = None
    if not core_ready:
        _out = mo.md("*V4 locked until every test passes.*")
    elif not run_v4.value:
        _out = mo.md("*Press **Run V4** (uses the E1 configuration; about twice an E0 run).*")
    else:
        _code = code_from_key(config.code)
        _workers = dz.parallel_workers(config.shots, config.workers)
        _arms = {}
        # unflagged circuit
        _c0 = build_memory_circuit(_code, config.rounds, config.p, use_flags=False)
        _H0, _L0, _p0 = dem_to_matrices(_c0.detector_error_model(decompose_errors=False))
        _d0, _o0 = _c0.compile_detector_sampler(seed=config.seed).sample(
            config.shots, separate_observables=True)
        _arms["unflagged"] = _arm(_H0, _L0, _p0, _d0, _o0, _workers)
        # flagged circuit, decoded with and without the flag detectors
        _c1 = build_memory_circuit(_code, config.rounds, config.p, use_flags=True)
        _H1, _L1, _p1 = dem_to_matrices(_c1.detector_error_model(decompose_errors=False))
        _d1, _o1 = _c1.compile_detector_sampler(seed=config.seed).sample(
            config.shots, separate_observables=True)
        _mask = flag_detector_mask(_code, config.rounds, True, _c1.num_detectors)
        _arms["flagged, sighted"] = _arm(_H1, _L1, _p1, _d1, _o1, _workers)
        _keep = ~_mask
        _arms["flagged, blind"] = _arm(_H1[_keep], _L1, _p1, _d1[:, _keep], _o1, _workers)
        v4_result = dict(arms=_arms, shots=config.shots, code=config.code,
                         rounds=config.rounds, p=config.p)
        _out = mo.md("V4 finished.")
    _out
    return (v4_result,)


@app.cell
def _(RESULTS_DIR, export_bundle, mo, os, plt, v4_loaded, v4_result, write_result_file):
    v4_data = v4_result if v4_result is not None else v4_loaded
    mo.stop(v4_data is None, mo.md("*Run V4, or load a saved run above.*"))
    _order = ["unflagged", "flagged, blind", "flagged, sighted"]
    _a = v4_data["arms"]
    _fig, _ax = plt.subplots(figsize=(7.5, 4.2))
    for _i, _n in enumerate(_order):
        _m = _a[_n]["ler"]
        _ax.errorbar([_i], [_m[0] if _m[0] > 0 else _m[2]],
                     yerr=[[max(_m[0] - _m[1], 0)], [max(_m[2] - _m[0], 0)]],
                     fmt="o" if _m[0] > 0 else "v", ms=9, capsize=6, color="#1f77b4")
        _ax.annotate(f"{_a[_n]['failures']} fails", (_i, _m[2]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)
    _ax.set_xticks(range(3), _order)
    _ax.set_yscale("log")
    _ax.set_xlim(-0.5, 2.5)
    _ax.set_ylabel("logical error rate (strong decoder)")
    _ax.set_title(f"V4 — {v4_data['code']}, T={v4_data['rounds']}, "
                  f"{v4_data['shots']:,} shots per arm")
    _ax.grid(True, which="both", axis="y", alpha=0.3)
    _fig.tight_layout()

    # Which explanation do the numbers support? Compare the three intervals.
    _u, _b, _s = (_a[n]["ler"] for n in _order)

    def _overlap(x, y):
        return not (x[2] < y[1] or y[2] < x[1])

    def _better(x, y):            # x significantly lower (better) than y
        return x[2] < y[1]

    if _overlap(_b, _s) and _better(_b, _u):
        _verdict = ("**Circuit hardening.** Blind decoding of the flagged circuit is as good as "
                    "sighted and beats the unflagged circuit, so the flag CNOTs themselves reduce "
                    "the damage; the flag outcomes add little.")
    elif _overlap(_b, _u) and _better(_s, _b):
        _verdict = ("**Extra information.** Blind decoding falls back to the unflagged rate, so the "
                    "gain comes from the decoder reading the flag detectors, not from the circuit.")
    elif _better(_u, _b) and _better(_s, _u):
        _verdict = ("**Information, against a noisier circuit.** The flag qubits make the circuit "
                    "worse — blind decoding is significantly poorer than no flags at all — but the "
                    "flag outcomes more than pay that back when the decoder can read them. Flags "
                    "here are decoder input, not circuit hardening.")
    elif _better(_u, _b) and not _better(_s, _u):
        _verdict = ("**Flags cost accuracy.** The extra flag circuitry adds more error than its "
                    "outcomes recover, even with the decoder reading them.")
    elif _overlap(_b, _u) and _overlap(_s, _u):
        _verdict = "**Inconclusive** — the three arms overlap. Run more shots."
    else:
        _verdict = ("**Mixed.** Ordering: " + ", ".join(
            f"{n} {_a[n]['ler'][0]:.2e}" for n in sorted(_order, key=lambda n: _a[n]["ler"][0]))
            + ". Read the intervals in the table below.")
    _rows = [dict(arm=n, failures=_a[n]["failures"], ler=_a[n]["ler"][0],
                  ci_low=_a[n]["ler"][1], ci_high=_a[n]["ler"][2],
                  detectors=_a[n]["detectors"], faults=_a[n]["faults"]) for n in _order]
    _tag = v4_data["code"].strip("[]").replace(", ", "-")
    _path = write_result_file(
        os.path.join(RESULTS_DIR, f"V4_{_tag}_T{v4_data['rounds']}_p{v4_data['p']:g}_"
                                  f"{v4_data['shots']}shots.json"),
        {"kind": "validation", "config": {k: v4_data[k] for k in ("code", "rounds", "p", "shots")},
         "checks": {"v4": {n: _a[n] for n in _order}}})
    _folder = export_bundle(_path, f"V4 hardening vs information — {v4_data['code']}",
                            {"v4_arms": _fig}, {"arms": _rows}, notes=_verdict)
    mo.vstack([mo.md("### V4 — hardening or information?"), mo.md(_verdict),
               mo.md("| arm | failures | LER | 95% CI | detectors |\n|:--|--:|--:|:--|--:|\n"
                     + "\n".join(f"| {r['arm']} | {r['failures']} | {r['ler']:.3e} | "
                                  f"[{r['ci_low']:.3e}, {r['ci_high']:.3e}] | {r['detectors']} |"
                                  for r in _rows)),
               mo.md(f"Saved to `{_path}` · exported to `{_folder}`"), _fig])
    return


@app.cell
def _(
    build_memory_circuit,
    code_from_key,
    config,
    core_ready,
    flag_detector_mask,
    mo,
    run_v5,
):
    v5_run = None
    if core_ready and run_v5.value:
        _code = code_from_key(config.code)
        _rows = []
        for _T in (3, 6, 12, 18, 24):
            _circ = build_memory_circuit(_code, _T, config.p, use_flags=True)
            _det = _circ.compile_detector_sampler(seed=config.seed).sample(2000)
            _mask = flag_detector_mask(_code, _T, True, _circ.num_detectors)
            _bits = _det[:, _mask]
            _rows.append(dict(rounds=_T, flag_detectors=int(_mask.sum()),
                              fraction_any_flag=float((_bits.sum(1) > 0).mean()),
                              mean_flags_per_shot=float(_bits.sum(1).mean())))
        v5_run = dict(code=config.code, p=config.p, shots=2000, rows=_rows)
    mo.md("*Press **Run V5** — flag statistics only, no decoding, so it takes seconds.*"
          if v5_run is None else "V5 finished.")
    return (v5_run,)


@app.cell
def _(RESULTS_DIR, export_bundle, mo, os, plt, v5_loaded, v5_run, write_result_file):
    v5_data = v5_run if v5_run is not None else v5_loaded
    mo.stop(v5_data is None, mo.md("*Run V5, or load a saved run above.*"))
    _rows = v5_data["rows"]
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4))
    _Ts = [r["rounds"] for r in _rows]
    _a.plot(_Ts, [r["fraction_any_flag"] for r in _rows], "o-", color="#d62728")
    _a.axhline(1.0, color="k", lw=0.7, ls=":")
    _a.set_ylim(0, 1.05)
    _a.set_xlabel("syndrome rounds T")
    _a.set_ylabel("fraction of shots with at least one flag")
    _a.set_title("'Any flag fired' saturates with T")
    _b.plot(_Ts, [r["mean_flags_per_shot"] for r in _rows], "s-", color="#1f77b4")
    _b.set_xlabel("syndrome rounds T")
    _b.set_ylabel("mean flag bits per shot")
    _b.set_title("Flags fire in proportion to rounds")
    for _ax in (_a, _b):
        _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _tag = v5_data["code"].strip("[]").replace(", ", "-")
    _path = write_result_file(
        os.path.join(RESULTS_DIR, f"V5_{_tag}_p{v5_data['p']:g}_flagrate.json"),
        {"kind": "validation",
         "config": dict(code=v5_data["code"], p=v5_data["p"], shots=v5_data["shots"]),
         "checks": {"v5": _rows}})
    _folder = export_bundle(_path, f"V5 flag rate vs rounds — {v5_data['code']}, p={v5_data['p']}",
                            {"v5_flag_rate": _fig}, {"flag_rate": _rows},
                            notes="A trigger that fires on nearly every shot cannot discriminate; "
                                  "this is the mechanism behind a null result for flag triggering.")
    mo.vstack([mo.md("### V5 — does the trigger saturate?"),
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
    ## 11b. E3 — Decoder zoo: is flag-triggered switching viable at all?

    E1 tested one decoder pair. A flag trigger might still pay off with a
    different weak decoder (one that fails silently, or costs more) or a
    different strong decoder. E3 tests **every weak × strong pair** in a
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
def _(mo):
    import sys as _sys
    _here = mo.notebook_dir()
    _here = str(_here) if _here is not None else "."
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    import decoder_zoo as dz
    return (dz,)


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
    _circ = build_memory_circuit(_code, _T, 2e-3, use_flags=True)
    _H, _L, _pr = dem_to_matrices(_circ.detector_error_model(decompose_errors=False))
    _mask = flag_detector_mask(_code, _T, True, _circ.num_detectors)
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
        """E1 derives its policies instead of running them; it must agree exactly."""
        cfg = ExperimentConfig(code="[[72, 12, 6]]", rounds=_T, p=2e-3, shots=90, seed=17,
                               use_flags=True, osd_order=0, workers=1)
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
        V4's 'blind' arm must drop the flag detector rows, not zero them: a zeroed
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
        """V5's mechanism: more rounds means more chances for a flag to fire."""
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

    tests_zoo = run_checks([
        ("compiled peeling reproduces the notebook's PeelingDecoder exactly", _peel_matches_notebook),
        ("zoo decoders: honest convergence, strong ones always valid", _zoo_decoders_valid),
        ("derived policies reproduce SwitchPolicy shot by shot", _derivation_matches_switchpolicy),
        ("parallel decoding gives the same results as sequential", _parallel_matches_sequential),
        ("paired exact test gives known p-values", _paired_exact_known),
        ("break-even cost ratio formula", _breakeven_formula),
        ("no rule gains more than the headroom; gains = catches", _headroom_bounds_gain),
        ("too few failures gives 'inconclusive', never 'viable'", _inconclusive_when_few_failures),
        ("all four E3 graphs render", _graphs_render),
        ("parallel E1 (run_ablation) matches SwitchPolicy exactly", _ablation_matches_switchpolicy),
        ("POSITIVE CONTROL: injected silent failures are detected", _positive_control_is_detected),
        ("zero headroom reports 'impossible', not 'inconclusive'", _zero_headroom_is_impossible_not_inconclusive),
        ("strong-decoder ceiling and flag diagnostics are correct", _ceiling_and_diagnostics),
        ("V4 blind arm removes flag rows rather than zeroing them", _blind_decoding_removes_flag_rows),
        ("V5 flag rate grows with the number of rounds", _flag_rate_grows_with_rounds),
    ])
    _backend = _state.get("backend", "not run")
    mo.vstack([render_checks("decoder zoo", tests_zoo),
               mo.md(f"Parallel backend on this machine: **{_backend}** · "
                     f"compiled peeling (numba): **{'yes' if dz.HAVE_NUMBA else 'no — run `uv add numba`'}**")])
    return (tests_zoo,)


@app.cell
def _(BB_PRESETS, GB_PRESETS, dz, mo, os):
    _cores = os.cpu_count() or 1
    ui_zoo_code = mo.ui.dropdown(list(BB_PRESETS) + list(GB_PRESETS), value="[[72, 12, 6]]", label="Code")
    ui_zoo_rounds = mo.ui.slider(1, 12, value=6, label="Rounds T")
    ui_zoo_p = mo.ui.dropdown(["5e-4", "1e-3", "2e-3", "3e-3"], value="1e-3", label="p")
    ui_zoo_shots = mo.ui.number(start=100, stop=1_000_000, step=100, value=20_000, label="Shots")
    ui_zoo_seed = mo.ui.number(value=20260922, label="Seed")
    ui_zoo_workers = mo.ui.slider(1, max(2, _cores), value=max(1, _cores - 1),
                                  label=f"CPU workers (of {_cores})")
    ui_zoo_weak = mo.ui.multiselect(list(dz.WEAK) + list(dz.CONTROLS), value=dz.DEFAULT_WEAK,
                                    label="Weak decoders")
    ui_zoo_strong = mo.ui.multiselect(list(dz.STRONG), value=dz.DEFAULT_STRONG, label="Strong decoders")
    mo.vstack([mo.md("### E3 configuration (flags always on)"),
               mo.hstack([ui_zoo_code, ui_zoo_p, ui_zoo_rounds]),
               mo.hstack([ui_zoo_shots, ui_zoo_seed, ui_zoo_workers]),
               ui_zoo_weak, ui_zoo_strong,
               mo.md("*OSD-CS orders cost roughly 15× OSD-0 per shot; LSD-CS is far cheaper. "
                     "Use the estimate button before a long run.*"),
               mo.md("*`control-*` weak decoders are **positive controls**: they are wrong "
                     "on a fraction of flagged shots by construction. Run one to show this "
                     "analysis detects silent failures when they exist — the check that makes "
                     "a negative result believable.*")])
    return (
        ui_zoo_code,
        ui_zoo_p,
        ui_zoo_rounds,
        ui_zoo_seed,
        ui_zoo_shots,
        ui_zoo_strong,
        ui_zoo_weak,
        ui_zoo_workers,
    )


@app.cell
def _(
    build_memory_circuit,
    code_from_key,
    dem_to_matrices,
    flag_detector_mask,
    ui_zoo_code,
    ui_zoo_p,
    ui_zoo_rounds,
    ui_zoo_seed,
    ui_zoo_shots,
    ui_zoo_strong,
    ui_zoo_weak,
    ui_zoo_workers,
):
    zoo_setup = dict(code=ui_zoo_code.value, rounds=int(ui_zoo_rounds.value),
                     p=float(ui_zoo_p.value), shots=int(ui_zoo_shots.value),
                     seed=int(ui_zoo_seed.value), workers=int(ui_zoo_workers.value),
                     weak=list(ui_zoo_weak.value), strong=list(ui_zoo_strong.value))

    def build_zoo_problem(setup):
        """Circuit, DEM matrices and flag mask for an E3 configuration."""
        code = code_from_key(setup["code"])
        circ = build_memory_circuit(code, setup["rounds"], setup["p"], use_flags=True)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(code, setup["rounds"], True, circ.num_detectors)
        return code, circ, H, L, pr, mask

    return build_zoo_problem, zoo_setup


@app.cell
def _(RESULTS_DIR, dz, glob, mo, os, zoo_setup):
    import hashlib as _hashlib
    _tag = zoo_setup["code"].strip("[]").replace(", ", "-")
    _stem = (f"E3_{_tag}_T{zoo_setup['rounds']}_p{zoo_setup['p']:g}_"
             f"{zoo_setup['shots']}shots_seed{zoo_setup['seed']}")
    _key = _hashlib.sha1(",".join(zoo_setup["weak"] + zoo_setup["strong"]).encode()).hexdigest()[:8]
    zoo_checkpoint = dict(data_path=os.path.join(RESULTS_DIR, _stem + ".npz"),
                          dir=os.path.join(RESULTS_DIR, "checkpoints", f"{_stem}_{_key}"))
    _done = len(glob.glob(os.path.join(zoo_checkpoint["dir"], "chunk_*.npz")))
    _total = -(-zoo_setup["shots"] // dz.CHUNK_SIZE)
    (mo.md(f"🔁 **A partial run of this configuration exists: {_done}/{_total} chunks done.** "
           "Pressing Run E3 resumes it instead of starting over.")
     if _done else mo.md(""))
    return (zoo_checkpoint,)


@app.cell
def _(mo):
    run_zoo_estimate = mo.ui.run_button(label="Estimate run time (~30 s pilot)")
    run_zoo = mo.ui.run_button(label="Run E3 — decoder zoo")
    mo.hstack([run_zoo_estimate, run_zoo])
    return run_zoo, run_zoo_estimate


@app.cell
def _(build_zoo_problem, core_ready, dz, mo, np, run_zoo_estimate, tests_zoo, time, zoo_setup):
    mo.stop(not (core_ready and all(r["status"] == "PASS" for r in tests_zoo)),
            mo.md("*E3 locked until every test, including the decoder-zoo tests, passes.*"))
    mo.stop(not run_zoo_estimate.value)
    _code, _circ, _H, _L, _pr, _ = build_zoo_problem(zoo_setup)
    _det = _circ.compile_detector_sampler(seed=1).sample(25).astype(np.uint8)
    _rows, _total = [], 0.0
    for _name in zoo_setup["weak"] + zoo_setup["strong"]:
        _dec = dz.make_decoder(_name, _H, _pr)
        _dec.decode(_det[0])                              # compile / warm up
        _t0 = time.perf_counter()
        for _s in _det:
            _dec.decode(_s)
        _ms = 1e3 * (time.perf_counter() - _t0) / len(_det)
        _total += _ms
        _rows.append(f"| {_name} | {_ms:.2f} ms |")
    _eta = zoo_setup["shots"] * _total / 1e3 / max(1, zoo_setup["workers"])
    mo.md("| decoder | time per shot |\n|:--|--:|\n" + "\n".join(_rows)
          + f"\n\n**Estimated run time: {_eta / 60:.0f} min** for {zoo_setup['shots']:,} shots "
            f"on {zoo_setup['workers']} workers (parallel speed-up is usually a little below the "
            "worker count).")
    return


@app.cell
def _(
    build_zoo_problem,
    core_ready,
    datetime,
    dz,
    glob,
    mo,
    os,
    run_zoo,
    tests_zoo,
    zoo_checkpoint,
    zoo_setup,
):
    def _run_e3():
        """Sample the shots, decode them with every decoder, save, and summarise."""
        _code, _circ, _H, _L, _pr, _mask = build_zoo_problem(zoo_setup)
        _det, _obs = _circ.compile_detector_sampler(seed=zoo_setup["seed"]).sample(
            zoo_setup["shots"], separate_observables=True)
        _n_flags, _rounds = dz.flag_features(_det, _mask, _code.hx.shape[0], zoo_setup["rounds"])
        _names = []
        _vector = None
        for _n in zoo_setup["weak"] + zoo_setup["strong"]:
            _spec = dict(dz.CATALOGUE[_n])
            if _spec.get("kind") == "control":
                if _vector is None:
                    _vector = dz.logical_null_vector(_H, _L)   # H v = 0 but L v = 1
                _spec.update(vector=_vector, flag_mask=_mask)
            _names.append((_n, _spec))
        _n_chunks = -(-zoo_setup["shots"] // dz.CHUNK_SIZE)
        with mo.status.progress_bar(total=_n_chunks, title="E3: decoding", show_eta=True,
                                    show_rate=True) as _bar:
            # every finished chunk is written to zoo_checkpoint["dir"] at once, so an
            # interrupted run resumes when Run E3 is pressed again with the same settings
            _res, _backend = dz.run_zoo(_H, _L, _pr, _det, _obs, _names,
                                        workers=zoo_setup["workers"], on_chunk=_bar.update,
                                        checkpoint_dir=zoo_checkpoint["dir"])
        _path = zoo_checkpoint["data_path"]
        _meta = dict(zoo_setup, backend=_backend, numba=dz.HAVE_NUMBA,
                     created=datetime.datetime.now().isoformat(timespec="seconds"))
        dz.save_zoo(_path, _res, _n_flags, _rounds, _meta)
        for _f in glob.glob(os.path.join(zoo_checkpoint["dir"], "chunk_*.npz")):
            os.remove(_f)                     # the full result is saved: drop the checkpoints
        if os.path.isdir(zoo_checkpoint["dir"]) and not os.listdir(zoo_checkpoint["dir"]):
            os.rmdir(zoo_checkpoint["dir"])
        _data = dict(results=_res, n_flags=_n_flags, flag_rounds=_rounds, meta=_meta, path=_path)
        return _data, mo.md(f"Decoded {zoo_setup['shots']:,} shots × {len(_names)} decoders "
                            f"({_backend}). Saved to `{_path}`.")

    # Always define e3_run (None when not run): marimo does not run cells that
    # depend on a variable that a stopped cell never defined.
    e3_run = None
    if not (core_ready and all(r["status"] == "PASS" for r in tests_zoo)):
        _out = mo.md("*E3 locked until every test passes.*")
    elif not run_zoo.value:
        _out = mo.md("*Press **Run E3** to start, or load a previous run below.*")
    else:
        e3_run, _out = _run_e3()
    _out
    return (e3_run,)


@app.cell
def _(RESULTS_DIR, e3_run, glob, mo, os):
    _ = e3_run                            # re-list saved runs after each new run
    _files = sorted(glob.glob(os.path.join(RESULTS_DIR, "E3_*.npz")))
    ui_zoo_file = mo.ui.dropdown({os.path.basename(f): f for f in _files},
                                 value=os.path.basename(_files[-1]) if _files else None,
                                 label="Previous E3 run")
    load_zoo_btn = mo.ui.run_button(label="Load")
    mo.hstack([ui_zoo_file, load_zoo_btn]) if _files else mo.md("*No saved E3 runs yet.*")
    return load_zoo_btn, ui_zoo_file


@app.cell
def _(dz, load_zoo_btn, ui_zoo_file):
    e3_loaded = None                      # always defined, see the run cell
    if load_zoo_btn.value and ui_zoo_file.value:
        _res, _nf, _fr, _meta = dz.load_zoo(ui_zoo_file.value)
        e3_loaded = dict(results=_res, n_flags=_nf, flag_rounds=_fr, meta=_meta,
                         path=ui_zoo_file.value)
    return (e3_loaded,)


@app.cell
def _(dz, e3_loaded, e3_run, mo):
    e3_data = e3_run if e3_run is not None else e3_loaded
    mo.stop(e3_data is None, mo.md("*Run E3 or load a previous run to see the analysis.*"))
    _names = list(e3_data["results"])
    e3_weak = [n for n in _names if n in dz.WEAK or n in dz.CONTROLS]
    e3_strong = [n for n in _names if n in dz.STRONG]
    e3_analysis = dz.analyse(e3_data["results"], e3_weak, e3_strong,
                             e3_data["n_flags"], e3_data["flag_rounds"])

    _pairs = e3_analysis["pairs"]
    _viable = [p for p in _pairs if p["verdict"].startswith("viable")]
    _inconclusive = [p for p in _pairs if p["verdict"] == "inconclusive"]
    _head = sum(p["headroom"] for p in _pairs)
    _lines = [f"### E3 verdict — {e3_analysis['shots']:,} shots, "
              f"{len(e3_weak)} weak × {len(e3_strong)} strong decoders, `{e3_data['path']}`", ""]
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
    for f in e3_analysis["flag_diagnostics"]:
        _lines.append(f"| {f['decoder']} | {f['error_rate_flagged']:.2e} | "
                      f"{f['error_rate_unflagged']:.2e} | {f['risk_ratio']:.1f}× | "
                      f"{f['mutual_information_bits']:.4f} | "
                      f"{f['share_of_failures_on_flagged']:.1%} |")
    if e3_analysis["strong_ceiling"]:
        _lines += ["", "#### Is there headroom left in the strong decoder?", "",
                   "Shots one strong decoder gets wrong that another gets right. If almost none "
                   "are fixed, those failures are near-uncorrectable at this noise level and "
                   "reweighting priors — flag-informed or otherwise — cannot help either.", "",
                   "| decoder | failures | fixed by | fixed | broken |",
                   "|:--|--:|:--|--:|--:|"]
        for c in e3_analysis["strong_ceiling"]:
            _lines.append(f"| {c['decoder']} | {c['failures']} | {c['compared_with']} | "
                          f"{c['fixed_by_other']} | {c['broken_by_other']} |")
    _lines += ["", "| weak decoder | converged | silent failures | silent rate (95% upper) | "
               "on flagged shots | median time |", "|:--|--:|--:|--:|--:|--:|"]
    for w, v in e3_analysis["weak"].items():
        _lines.append(f"| {w} | {v['converged']:,} | {v['silent']} | {v['silent_rate'][2]:.2e} | "
                      f"{v['silent_flagged']} | {v['time_us'] / 1e3:.2f} ms |")
    mo.md("\n".join(_lines))
    return e3_analysis, e3_data, e3_strong, e3_weak


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
        ax.set_ylabel("silent failures / converged shots")
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
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                if i == len(W) - 1:
                    ax.set_xlabel("escalated", fontsize=8)
                if j == 0:
                    ax.set_ylabel("failures", fontsize=8)
        fig.suptitle("Every rule for every pair (★ primary_fail, ● flag rules, red = flag-only). "
                     "A viable rule sits below ★.", fontsize=9)
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
    e3_analysis,
    e3_data,
    export_bundle,
    mo,
    plot_zoo_breakeven,
    plot_zoo_silent,
    plot_zoo_tradeoffs,
    plot_zoo_verdicts,
):
    _figs = {"verdicts": plot_zoo_verdicts(e3_analysis), "silent_failures": plot_zoo_silent(e3_analysis),
             "breakeven": plot_zoo_breakeven(e3_analysis), "tradeoffs": plot_zoo_tradeoffs(e3_analysis)}
    _weak = [dict(weak=w, converged=v["converged"], silent=v["silent"],
                  silent_rate=v["silent_rate"][0], silent_rate_upper=v["silent_rate"][2],
                  silent_on_flagged=v["silent_flagged"], median_time_us=v["time_us"])
             for w, v in e3_analysis["weak"].items()]
    _rows = [{k: (v[0] if k == "ler" else v) for k, v in r.items()} | dict(
                 ler_low=r["ler"][1], ler_high=r["ler"][2]) for r in e3_analysis["rows"]]
    _counts = {}
    for _p in e3_analysis["pairs"]:
        _counts[_p["verdict"]] = _counts.get(_p["verdict"], 0) + 1
    _folder = export_bundle(
        e3_data["path"], f"E3 decoder zoo — {e3_analysis['shots']:,} shots",
        dict(_figs), {"verdicts": e3_analysis["pairs"], "weak_decoders": _weak, "all_rules": _rows,
                      "flag_diagnostics": e3_analysis["flag_diagnostics"],
                      "strong_ceiling": e3_analysis["strong_ceiling"]},
        notes="Verdict counts: " + ", ".join(f"{k}: {v}" for k, v in sorted(_counts.items())))
    mo.vstack([mo.md(f"**Exported** figures (PNG + PDF), CSV tables and report to `{_folder}`"),
               *[_f for _f in _figs.values()]])
    return


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
    4. A. A. Kovalev and L. P. Pryadko, "Quantum Kronecker sum-product low-density
       parity-check codes with finite rate," *Phys. Rev. A* **88**, 012311 (2013).
    5. P. Panteleev and G. Kalachev, "Degenerate quantum LDPC codes with good
       finite length performance," *Quantum* **5**, 585 (2021).
    6. R. Chao and B. Reichardt, "Quantum error correction with only two extra
       qubits," *Phys. Rev. Lett.* **121**, 050502 (2018).
    7. Pakhunov (2026), "Analytical Theory of Greedy Peeling for Bivariate Bicycle
       Codes and Two-Shot Streaming Decoding." **TODO:** complete citation.
    8. Sahay et al. (2026), "A matching decoder for bivariate bicycle codes."
       **TODO:** complete citation.
    9. **TODO:** the decoder-switching reference your proposal builds on.
    """)
    return


if __name__ == "__main__":
    app.run()
