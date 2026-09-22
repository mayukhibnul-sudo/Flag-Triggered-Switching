import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium", app_title="Decoder Switching for BB Codes")


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
    def _as_gf2(M):
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
        R = _as_gf2(M)
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
        span = _as_gf2(subspace)
        amb = _as_gf2(ambient)
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

    def _ldpc_kwargs(priors, max_iter):
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
            self.dec = _LdpcBp(sp.csr_matrix(H, dtype=np.uint8), **_ldpc_kwargs(priors, max_iter))

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
                                  **_ldpc_kwargs(priors, max_iter))

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
                                  **_ldpc_kwargs(priors, max_iter))

        def decode(self, syndrome):
            e = self.dec.decode(np.asarray(syndrome, dtype=np.uint8))
            return DecodeResult(np.asarray(e, dtype=np.uint8), True,
                                int(self.dec.iter), soft=np.array(self.dec.log_prob_ratios))

    return BpDecoder, BpLsdDecoder, BpOsdDecoder, PeelingDecoder


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

    **Invariant:** no policy can have a lower logical error rate than `always`
    — the secondary sees the same syndrome. Beating it means a bug or an unfair
    baseline, never a result.
    """)
    return


@app.cell
def _():
    TRIGGERS = ("never", "always", "primary_fail", "flag", "flag_or_fail")
    _FLAG_TRIGGERS = ("flag", "flag_or_fail")

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
            if trigger in _FLAG_TRIGGERS:
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
            if self.trigger not in _FLAG_TRIGGERS:
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

    return (SwitchPolicy,)


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
    _PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                "#8c564b", "#e377c2", "#17becf", "#7f7f7f", "#bcbd22"]

    def _colours(names):
        return {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(names)}

    def _ler_marks(ax, x, metrics, colour, label=None, horizontal=False):
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
        col = _colours(names)
        y = np.arange(len(names))[::-1]
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 0.55 * len(names) + 1.8),
                                   sharey=True, gridspec_kw=dict(width_ratios=[3, 2]))
        for yi, n in zip(y, names):
            _ler_marks(a, [yi], [results[n]], col[n], horizontal=True)
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
        col = _colours(names)
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
        for i, n in enumerate(names):
            ms = [sweep[p][n] for p in ps]
            # small multiplicative offset: policies with identical LER (which the
            # invariant makes common) would otherwise hide behind each other
            dodge = 1 + 0.035 * (i - (len(names) - 1) / 2)
            xs = [p * dodge for p in ps]
            _ler_marks(a, xs, ms, col[n], label=n)
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
        col = _colours(list(results))
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
        col = _colours(list(results))
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (n, m) in enumerate(results.items()):
            _ler_marks(ax, [m.time_p99_us], [m], col[n], label=n)
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

    return plot_ablation, plot_latency, plot_outcomes, plot_sweep, plot_tradeoff


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

    return (ExperimentConfig,)


@app.cell
def _(ExperimentConfig, Metrics, asdict, json, np):
    _SCHEMA_VERSION = 1

    def _software_versions():
        from importlib.metadata import PackageNotFoundError, version
        out = {}
        for pkg in ("stim", "ldpc", "numpy", "scipy", "marimo"):
            try:
                out[pkg] = version(pkg)
            except PackageNotFoundError:
                out[pkg] = None
        return out

    def _metrics_to_json(m, include_samples):
        d = asdict(m)
        samples = d.pop("samples", None)
        if include_samples and samples is not None:
            d["samples"] = {k: np.asarray(v).tolist() for k, v in samples.items()}
        return d

    def _metrics_from_json(d):
        d = dict(d)
        samples = d.pop("samples", None)
        if samples is not None:
            samples = {k: np.asarray(v) for k, v in samples.items()}
        return Metrics(**d, samples=samples)

    def _write(path, payload):
        import datetime
        import os
        payload = {"schema": _SCHEMA_VERSION,
                   "created": datetime.datetime.now().isoformat(timespec="seconds"),
                   "software": _software_versions(), **payload}
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=1)
        return path

    def _read(path, kind):
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
        return _write(path, {
            "kind": "ablation",
            "config": asdict(config),
            "results": {n: _metrics_to_json(m, include_samples) for n, m in results.items()},
        })

    def load_results(path):
        """Inverse of save_results -> (ExperimentConfig, dict name -> Metrics)."""
        d = _read(path, "ablation")
        return (ExperimentConfig(**d["config"]),
                {n: _metrics_from_json(m) for n, m in d["results"].items()})

    def save_sweep(path, config, sweep, include_samples=False):
        """Like save_results, for dict p -> dict policy -> Metrics."""
        return _write(path, {
            "kind": "sweep",
            "config": asdict(config),
            "points": [{"p": float(p),
                        "results": {n: _metrics_to_json(m, include_samples)
                                    for n, m in res.items()}}
                       for p, res in sweep.items()],
        })

    def load_sweep(path):
        """Inverse of save_sweep -> (ExperimentConfig, dict p -> dict name -> Metrics)."""
        d = _read(path, "sweep")
        return (ExperimentConfig(**d["config"]),
                {pt["p"]: {n: _metrics_from_json(m) for n, m in pt["results"].items()}
                 for pt in d["points"]})

    return load_results, load_sweep, save_results, save_sweep


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
    tests_gb,
    tests_gf2,
    tests_repro,
    tests_stats,
    tests_switch,
):
    _sections = [
        ("2. GF(2)", tests_gf2), ("3. BB codes", tests_codes), ("3. GB codes", tests_gb),
        ("4. Circuit", tests_circuit), ("5. DEM", tests_dem),
        ("6. Decoders", tests_decoders), ("7. Switch policies", tests_switch),
        ("8. Statistics", tests_stats), ("9. Reproducibility", tests_repro),
        ("11. Experiment drivers", tests_experiments),
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
):
    ui_code = mo.ui.dropdown(list(BB_PRESETS) + list(GB_PRESETS), value="[[72, 12, 6]]",
                             label="Code")
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
    GB_PRESETS,
    PeelingDecoder,
    SwitchPolicy,
    TRIGGERS,
    bb_from_preset,
    build_memory_circuit,
    dem_to_matrices,
    flag_detector_mask,
    gb_from_preset,
    mo,
    run_benchmark,
):
    _FLAG_ONLY = ("flag", "flag_or_fail")

    def code_from_key(key):
        """Build any preset, BB or GB, from its label."""
        if key in BB_PRESETS:
            return bb_from_preset(key, BB_PRESETS)
        if key in GB_PRESETS:
            return gb_from_preset(key, GB_PRESETS)
        raise KeyError(f"unknown code {key!r}")

    def run_ablation(config, keep_samples=True):
        """
        E1. Build the code, circuit and DEM from `config`; construct one
        SwitchPolicy per trigger over a shared PeelingDecoder / BpOsdDecoder pair
        (skip flag triggers when config.use_flags is False); run_benchmark.

        Returns dict trigger -> Metrics. Per-shot samples are kept by default so
        the latency graph can be drawn.
        """
        code = code_from_key(config.code)
        circuit = build_memory_circuit(code, config.rounds, config.p, use_flags=config.use_flags)
        H, L, priors = dem_to_matrices(circuit.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(code, config.rounds, config.use_flags, circuit.num_detectors)
        primary = PeelingDecoder(H, priors)
        secondary = BpOsdDecoder(H, priors, osd_order=config.osd_order)
        policies = {t: SwitchPolicy(primary, secondary, t, mask)
                    for t in TRIGGERS if config.use_flags or t not in _FLAG_ONLY}
        return run_benchmark(circuit, policies, L, config.shots, config.seed,
                             keep_samples=keep_samples)

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
        return (f"results/{kind}_{tag}_T{config.rounds}_p{config.p:g}_"
                f"{config.shots}shots_seed{config.seed}"
                f"{'_flags' if config.use_flags else ''}_osd{config.osd_order}.json")

    return code_from_key, result_path, run_ablation, run_sweep
@app.cell
def _(
    ExperimentConfig,
    TRIGGERS,
    code_from_key,
    render_checks,
    result_path,
    run_ablation,
    run_checks,
    run_sweep,
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

    tests_experiments = run_checks([
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
def _(
    config,
    core_ready,
    mo,
    plot_ablation,
    plot_latency,
    plot_outcomes,
    plot_tradeoff,
    result_path,
    run_ablation,
    run_e1,
    save_results,
):
    mo.stop(not core_ready, mo.md("*E1 locked: finish the implementation (see §10).*"))
    mo.stop(not run_e1.value, mo.md("*Press **Run E1** to start.*"))

    e1_results = run_ablation(config)
    e1_path = save_results(result_path("E1", config), config, e1_results)

    _always = e1_results.get("always")
    _suspicious = [t for t, m in e1_results.items()
                   if _always is not None and t != "always" and m.ci_high < _always.ci_low]
    _table = "\n".join(
        ["| trigger | LER | 95% CI | escalated | silent failures | work (mean) | p99 time |",
         "|:--|--:|:--|--:|--:|--:|--:|"]
        + [f"| `{t}` | {m.ler:.4f} | [{m.ci_low:.4f}, {m.ci_high:.4f}] | "
           f"{m.escalation_rate:.1%} | {m.silent_failures} | {m.work_mean:.1f} | "
           f"{m.time_p99_us:.0f} µs |"
           for t, m in e1_results.items()])
    mo.vstack([
        mo.md(f"### E1 — {config.code}, T={config.rounds}, p={config.p}, {config.shots} shots"),
        mo.md(_table),
        mo.md("> ⚠️ " + ", ".join(_suspicious) + " beat `always` — suspect a bug, "
              "not a result.") if _suspicious else mo.md(""),
        mo.md(f"Saved to `{e1_path}`. *Wall-clock times compare pure-Python peeling "
              "with compiled BP+OSD; use `work` or a compiled peeler for latency claims.*"),
        plot_ablation(e1_results),
        plot_outcomes(e1_results),
        plot_latency(e1_results),
        plot_tradeoff(e1_results),
    ])
    return (e1_results,)


@app.cell
def _(
    config,
    core_ready,
    mo,
    np,
    plot_sweep,
    result_path,
    run_e2,
    run_sweep,
    save_sweep,
):
    mo.stop(not core_ready, mo.md("*E2 locked: finish the implementation (see §10).*"))
    mo.stop(not run_e2.value, mo.md("*Press **Run E2** to start.*"))

    e2_ps = np.logspace(np.log10(5e-4), np.log10(6e-3), 6)
    e2_sweep = run_sweep(config, e2_ps)
    e2_path = save_sweep(result_path("E2", config), config, e2_sweep)
    mo.vstack([mo.md(f"### E2 — {config.code}, T={config.rounds}, {config.shots} shots/point"),
               mo.md(f"Saved to `{e2_path}`."),
               plot_sweep(e2_sweep)])
    return (e2_sweep,)


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
