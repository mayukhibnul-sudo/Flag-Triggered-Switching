"""
decoder_zoo.py — compute layer for experiment E3 in decoderSwitch.py.

Question: is flag-triggered switching viable for ANY combination of weak and
strong decoder, or only for none?

Method. Every sampled shot is decoded ONCE by every decoder in the zoo. Each
switching policy (weak W, strong S, trigger rule R) is then a deterministic
combination of W's result, S's result and the shot's flag bits, so all
|weak| x |strong| x |rules| policies are derived from |weak| + |strong| decodes
per shot. The notebook's tests check that this derivation reproduces
SwitchPolicy exactly.

This lives in a plain module, not in notebook cells, because worker processes
must be able to import it (a hard requirement on Windows).
"""
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import comb
import multiprocessing as mp

import numpy as np
import scipy.sparse as sp

try:
    import numba
    HAVE_NUMBA = True
except ImportError:          # peeling still works, just slower
    numba = None
    HAVE_NUMBA = False


# =============================================================================
# Decoder catalogue
# =============================================================================
# Weak decoders run first; their `converged` flag drives the primary_fail rule.
WEAK = {
    "peel":      dict(kind="peel"),
    # Pure uniqueness peeling: stalls whenever a rival fault is fully active.
    # NOT a positive control -- measurement shows it fails to converge more often
    # but is still never confidently wrong. Use "control-*" below for that.
    "peel-naive": dict(kind="peel", dominance=False),
    "BP-ms-5":   dict(kind="bp", bp_method="ms", max_iter=5),
    "BP-ms-10":  dict(kind="bp", bp_method="ms", max_iter=10),
    "BP-ms-30":  dict(kind="bp", bp_method="ms", max_iter=30),
    "BP-ms-100": dict(kind="bp", bp_method="ms", max_iter=100),
    "BP-ps-30":  dict(kind="bp", bp_method="ps", max_iter=30),
}
# Strong decoders always return a correction that reproduces the syndrome.
STRONG = {
    "OSD-0":   dict(kind="osd", osd_order=0, max_iter=30),
    "OSD-CS2": dict(kind="osd", osd_order=2, max_iter=30),
    "OSD-CS4": dict(kind="osd", osd_order=4, max_iter=30),
    "OSD-CS7": dict(kind="osd", osd_order=7, max_iter=30),
    "LSD-0":   dict(kind="lsd", lsd_order=0, max_iter=30),
    "LSD-CS2": dict(kind="lsd", lsd_order=2, max_iter=30),
    "LSD-CS4": dict(kind="lsd", lsd_order=4, max_iter=30),
}
DEFAULT_WEAK = ["peel", "BP-ms-10", "BP-ms-30", "BP-ps-30"]
# Positive controls: a weak decoder that IS confidently wrong on a fraction of
# flagged shots, by construction (see SilentControl). Run one of these to show the
# analysis finds silent failures and calls a flag rule viable when the effect is
# really there -- the check that makes a negative result believable.
CONTROLS = {
    "control-flagged-20%": dict(kind="control", rate=0.2, flagged_only=True),
    "control-flagged-50%": dict(kind="control", rate=0.5, flagged_only=True),
    "control-any-20%":     dict(kind="control", rate=0.2, flagged_only=False),
}
DEFAULT_STRONG = ["OSD-0", "OSD-CS4", "LSD-0", "LSD-CS4"]
CATALOGUE = {**WEAK, **STRONG, **CONTROLS}


# =============================================================================
# Greedy peeling, compiled when numba is available
# =============================================================================
def _peel_kernel_py(syndrome, row_ptr, row_idx, col_ptr, col_idx, n_faults, dominance):
    """
    Queue-based peeling with the dominance rule; same rule as the notebook's
    PeelingDecoder. A fully-active fault is peeled when every other fully-active
    fault touching its detectors has a signature contained in its own.
    Returns (correction, converged, detector_visits).
    """
    m = syndrome.shape[0]
    s = syndrome.copy()
    e = np.zeros(n_faults, dtype=np.uint8)
    queued = np.zeros(m, dtype=np.uint8)
    mark = np.zeros(m, dtype=np.uint8)          # scratch: detectors of the current best
    queue = np.empty(m, dtype=np.int64)         # ring buffer; a detector is queued at most once
    head = 0
    size = 0
    for d in range(m):
        if s[d]:
            queue[(head + size) % m] = d
            size += 1
            queued[d] = 1
    visits = 0
    while size > 0:
        d = queue[head]
        head = (head + 1) % m
        size -= 1
        queued[d] = 0
        visits += 1
        if s[d] == 0:
            continue
        # the largest fully-active fault at d (first one wins ties)
        best = -1
        best_len = -1
        n_full = 0
        for t in range(row_ptr[d], row_ptr[d + 1]):
            k = row_idx[t]
            full = True
            for u in range(col_ptr[k], col_ptr[k + 1]):
                if s[col_idx[u]] == 0:
                    full = False
                    break
            if full:
                n_full += 1
                if col_ptr[k + 1] - col_ptr[k] > best_len:
                    best = k
                    best_len = col_ptr[k + 1] - col_ptr[k]
        if best < 0:
            continue
        if dominance == 0 and n_full != 1:
            continue           # pure uniqueness rule: refuse when there is a rival at all
        for u in range(col_ptr[best], col_ptr[best + 1]):
            mark[col_idx[u]] = 1
        blocked = False
        for u in range(col_ptr[best], col_ptr[best + 1]):
            if blocked:
                break
            d2 = col_idx[u]
            for t in range(row_ptr[d2], row_ptr[d2 + 1]):
                k = row_idx[t]
                if k == best:
                    continue
                full = True
                subset = True
                for v in range(col_ptr[k], col_ptr[k + 1]):
                    dd = col_idx[v]
                    if s[dd] == 0:
                        full = False
                        break
                    if mark[dd] == 0:
                        subset = False
                if full and (not subset or dominance == 0):
                    blocked = True
                    break
        for u in range(col_ptr[best], col_ptr[best + 1]):
            mark[col_idx[u]] = 0
        if blocked:
            continue
        e[best] ^= 1
        for u in range(col_ptr[best], col_ptr[best + 1]):
            s[col_idx[u]] ^= 1
        # removing `best` may unblock rivals of it: revisit their detectors
        for u in range(col_ptr[best], col_ptr[best + 1]):
            d2 = col_idx[u]
            for t in range(row_ptr[d2], row_ptr[d2 + 1]):
                k = row_idx[t]
                for v in range(col_ptr[k], col_ptr[k + 1]):
                    d3 = col_idx[v]
                    if s[d3] and not queued[d3]:
                        queue[(head + size) % m] = d3
                        size += 1
                        queued[d3] = 1
    converged = True
    for d in range(m):
        if s[d]:
            converged = False
            break
    return e, converged, visits


_peel_kernel = (numba.njit(cache=True, nogil=True)(_peel_kernel_py)
                if HAVE_NUMBA else _peel_kernel_py)


class FastPeeling:
    """Compiled peeling decoder: decode(s) -> (correction, converged, work)."""

    def __init__(self, H, priors=None, dominance=True):
        self.dominance = np.uint8(1 if dominance else 0)
        Hr, Hc = sp.csr_matrix(H), sp.csc_matrix(H)
        self.row_ptr = Hr.indptr.astype(np.int64)
        self.row_idx = Hr.indices.astype(np.int64)
        self.col_ptr = Hc.indptr.astype(np.int64)
        self.col_idx = Hc.indices.astype(np.int64)
        self.n_faults = H.shape[1]

    def decode(self, syndrome):
        return _peel_kernel(np.ascontiguousarray(syndrome, dtype=np.uint8), self.row_ptr,
                            self.row_idx, self.col_ptr, self.col_idx, self.n_faults,
                            self.dominance)


class LdpcDecoder:
    """ldpc BP / BP+OSD / BP+LSD with shared settings: decode(s) -> (e, converged, work)."""

    def __init__(self, H, priors, kind, max_iter, bp_method="ms", osd_order=0, lsd_order=0):
        from ldpc.bp_decoder import BpDecoder
        from ldpc.bplsd_decoder import BpLsdDecoder
        from ldpc.bposd_decoder import BpOsdDecoder
        kw = dict(error_channel=[float(x) for x in priors], max_iter=max_iter,
                  bp_method=bp_method, schedule="parallel")
        if bp_method == "ms":
            kw["ms_scaling_factor"] = 0.625
        H = sp.csr_matrix(H, dtype=np.uint8)
        self.kind = kind
        if kind == "bp":
            self.dec = BpDecoder(H, **kw)
        elif kind == "osd":
            self.dec = BpOsdDecoder(H, osd_method="osd_cs", osd_order=osd_order, **kw)
        elif kind == "lsd":
            self.dec = BpLsdDecoder(H, lsd_method="LSD_CS", lsd_order=lsd_order, **kw)
        else:
            raise ValueError(kind)

    def decode(self, syndrome):
        e = np.asarray(self.dec.decode(np.asarray(syndrome, dtype=np.uint8)), dtype=np.uint8)
        converged = bool(self.dec.converge) if self.kind == "bp" else True
        return e, converged, int(self.dec.iter)


def gf2_solve_bitpacked(rows, b, n_cols):
    """
    Solve M v = b over GF(2) for a wide M given as packed uint64 rows.
    Returns a particular solution v (uint8, length n_cols) or None.
    Bit-packing keeps a 649 x 17136 elimination well under a second.
    """
    R = rows.copy()
    rhs = np.array(b, dtype=np.uint8) % 2
    m = R.shape[0]
    pivots = []
    r = 0
    for c in range(n_cols):
        word, bit = divmod(c, 64)
        col = (R[:, word] >> np.uint64(bit)) & np.uint64(1)
        cand = np.flatnonzero(col[r:].astype(np.uint8))
        if cand.size == 0:
            continue
        p = r + int(cand[0])
        if p != r:
            R[[r, p]] = R[[p, r]]
            rhs[[r, p]] = rhs[[p, r]]
        mask = ((R[:, word] >> np.uint64(bit)) & np.uint64(1)).astype(bool)
        mask[r] = False
        R[mask] ^= R[r]
        rhs[mask] ^= rhs[r]
        pivots.append(c)
        r += 1
        if r == m:
            break
    if np.any(rhs[r:]):
        return None                      # inconsistent: no solution
    v = np.zeros(n_cols, dtype=np.uint8)
    for i, c in enumerate(pivots):
        v[c] = rhs[i]
    return v


def logical_null_vector(H, L, observable=0):
    """
    A fault vector v with H v = 0 and L[observable] v = 1: invisible in the
    syndrome, but it flips a logical observable. Adding it to any correction turns
    that correction into a SILENT failure -- still syndrome-consistent, still
    "converged", but logically wrong.
    """
    H = sp.csr_matrix(H)
    L = sp.csr_matrix(L)
    M = sp.vstack([H, L[observable]]).tocsr().astype(np.uint8).toarray()
    n_cols = M.shape[1]
    packed = np.packbits(M, axis=1, bitorder="little")
    pad = (-packed.shape[1]) % 8
    if pad:
        packed = np.hstack([packed, np.zeros((packed.shape[0], pad), np.uint8)])
    rows = packed.view(np.uint64)
    b = np.zeros(M.shape[0], np.uint8)
    b[-1] = 1
    return gf2_solve_bitpacked(rows, b, n_cols)


class SilentControl:
    """
    POSITIVE CONTROL decoder. Decodes with peeling, then on a chosen fraction of
    shots adds `vector` to the correction. The result still reproduces the syndrome
    and still reports converged=True, but it flips a logical observable -- a silent
    failure by construction. With flagged_only=True those failures sit exactly on
    flagged shots, so a working analysis must report non-zero headroom and a viable
    flag rule. If it does not, the analysis cannot detect the effect it is looking
    for, and a null result from it means nothing.
    """

    def __init__(self, H, priors, vector, rate=0.2, flagged_only=True, flag_mask=None,
                 seed=1234):
        self.base = FastPeeling(H, priors)
        self.vector = np.asarray(vector, dtype=np.uint8)
        self.rate = float(rate)
        self.flagged_only = bool(flagged_only)
        self.flag_mask = None if flag_mask is None else np.asarray(flag_mask, bool)
        self.seed = int(seed)

    def decode(self, syndrome):
        e, converged, work = self.base.decode(syndrome)
        if not converged:
            return e, converged, work
        if self.flagged_only:
            if self.flag_mask is None or not np.asarray(syndrome)[self.flag_mask].any():
                return e, converged, work
        # deterministic per syndrome, so repeated runs agree
        h = (int(np.asarray(syndrome, dtype=np.uint8).sum()) * 2654435761 + self.seed) & 0xFFFFFFFF
        h ^= int(np.dot(np.flatnonzero(syndrome)[:8], [1, 3, 5, 7, 11, 13, 17, 19][:len(np.flatnonzero(syndrome)[:8])])) if syndrome.any() else 0
        if (h % 1000) / 1000.0 < self.rate:
            return (e ^ self.vector) % 2, True, work
        return e, converged, work


def make_decoder(spec, H, priors):
    """`spec` is a catalogue name, or a dict like {"kind": "osd", "osd_order": 4}."""
    spec = dict(CATALOGUE[spec]) if isinstance(spec, str) else dict(spec)
    kind = spec.pop("kind")
    if kind == "peel":
        return FastPeeling(H, priors, **spec)
    if kind == "control":
        vector = spec.pop("vector", None)
        if vector is None:
            vector = logical_null_vector(H, spec.pop("L"))
        return SilentControl(H, priors, vector, **spec)
    return LdpcDecoder(H, priors, kind, **spec)


def normalise_names(names):
    """Accept ["peel", ...] or [("name", spec_dict), ...]; return [(name, spec), ...]."""
    out = []
    for n in names:
        if isinstance(n, str):
            out.append((n, dict(CATALOGUE[n])))
        else:
            name, spec = n
            out.append((name, dict(CATALOGUE[spec]) if isinstance(spec, str) else dict(spec)))
    return out


def parallel_workers(shots, workers, min_shots=2000):
    """
    Each worker process builds its own decoders, which costs about a second. That
    only pays off for runs of at least `min_shots` shots.
    """
    return workers if shots >= min_shots else 1


def chunk_for(shots, workers, checkpoint=False):
    """Chunk size: fixed when checkpointing (so a resumed run splits identically),
    otherwise a few chunks per worker so the load balances."""
    if checkpoint:
        return CHUNK_SIZE
    return max(25, -(-shots // max(1, workers * 3)))


# =============================================================================
# Decoding every shot with every decoder, in parallel
# =============================================================================
def decode_chunk(task):
    """Worker: decode one chunk of shots with every requested decoder."""
    H, L, priors, det, obs, pairs = task
    L = sp.csr_matrix(L)
    out = {}
    for name, spec in pairs:
        dec = make_decoder(spec, H, priors)
        n = len(det)
        conv = np.zeros(n, bool)
        fail = np.zeros(n, bool)
        work = np.zeros(n, np.int32)
        t_us = np.zeros(n, np.float32)
        for i in range(n):
            t0 = time.perf_counter()
            e, c, w = dec.decode(det[i])
            t_us[i] = (time.perf_counter() - t0) * 1e6
            conv[i], work[i] = c, w
            fail[i] = bool(np.any((L @ e) % 2 != obs[i]))
        out[name] = dict(conv=conv, fail=fail, work=work, time_us=t_us)
    return out


CHUNK_SIZE = 250     # fixed, so a resumed run cuts the shots into the same chunks


def _chunk_path(checkpoint_dir, start):
    import os
    return os.path.join(checkpoint_dir, f"chunk_{start:09d}.npz")


def _save_chunk(path, part, names):
    import os
    tmp = path + ".tmp.npz"
    np.savez(tmp, names=np.array(list(names)),
             **{f"{n}__{k}": v for n, r in part.items() for k, v in r.items()})
    os.replace(tmp, path)            # atomic: a half-written chunk never looks complete


def _load_chunk(path, names):
    d = np.load(path, allow_pickle=False)
    if list(d["names"]) != list(names):
        return None                  # written for a different decoder list
    part = {}
    for key in d.files:
        if "__" in key:
            n, k = key.split("__", 1)
            part.setdefault(n, {})[k] = d[key]
    return part


def run_zoo(H, L, priors, det, obs, names, workers=1, chunk_size=CHUNK_SIZE, on_chunk=None,
            checkpoint_dir=None):
    """
    Decode all shots with every decoder in `names`.

    workers > 1 uses a process pool (spawn, so it behaves the same on Windows,
    macOS and Linux). If the pool cannot start, falls back to sequential.

    checkpoint_dir: if given, every finished chunk is saved there immediately,
    and chunks already present are loaded instead of decoded. An interrupted run
    therefore resumes where it stopped when started again with the same shots,
    decoders and chunk size.

    Returns (results, backend) with results[name][field] -> per-shot array; the
    backend string also reports how many chunks were resumed.
    """
    import os
    det = np.asarray(det, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    pairs = normalise_names(names)
    names = [n for n, _ in pairs]
    starts = list(range(0, len(det), chunk_size))
    parts = [None] * len(starts)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        for i, a in enumerate(starts):
            p = _chunk_path(checkpoint_dir, a)
            if os.path.exists(p):
                parts[i] = _load_chunk(p, names)
    resumed = sum(p is not None for p in parts)
    for _ in range(resumed):
        if on_chunk:
            on_chunk()

    def task(i):
        a = starts[i]
        return (H, L, priors, det[a:a + chunk_size], obs[a:a + chunk_size], pairs)

    def finished(i, part):
        parts[i] = part
        if checkpoint_dir:
            _save_chunk(_chunk_path(checkpoint_dir, starts[i]), part, names)
        if on_chunk:
            on_chunk()

    todo = [i for i, p in enumerate(parts) if p is None]
    backend = "sequential"
    if workers > 1 and len(todo) > 1:
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                futures = {pool.submit(decode_chunk, task(i)): i for i in todo}
                for f in as_completed(futures):
                    finished(futures[f], f.result())
            backend = f"parallel ({workers} processes)"
        except Exception as exc:                 # noqa: BLE001 -- fall back, but say why
            backend = f"sequential (process pool failed: {type(exc).__name__}: {exc})"
    for i in [i for i, p in enumerate(parts) if p is None]:
        finished(i, decode_chunk(task(i)))
    if resumed:
        backend += f"; resumed {resumed}/{len(starts)} chunks from checkpoint"
    results = {n: {k: np.concatenate([p[n][k] for p in parts]) for k in parts[0][n]}
               for n in names}
    return results, backend


def save_zoo(path, results, n_flags, flag_rounds, meta):
    arrays = {f"{n}__{k}": v for n, r in results.items() for k, v in r.items()}
    np.savez_compressed(path, n_flags=n_flags, flag_rounds=flag_rounds,
                        meta=np.array(repr(meta)), **arrays)
    return path


def load_zoo(path):
    d = np.load(path, allow_pickle=False)
    results = {}
    for key in d.files:
        if "__" in key:
            n, k = key.split("__", 1)
            results.setdefault(n, {})[k] = d[key]
    return results, d["n_flags"], d["flag_rounds"], str(d["meta"])


# =============================================================================
# Deriving policies and deciding viability
# =============================================================================
def flag_features(det, flag_mask, mx, rounds):
    """Per shot: number of flag bits fired, and number of rounds with any flag."""
    bits = np.asarray(det)[:, np.asarray(flag_mask, bool)].astype(bool)
    return bits.sum(1), bits.reshape(len(bits), rounds, mx).any(2).sum(1)


def trigger_rules(n_flags, flag_rounds):
    """
    name -> (pre, fallback). `pre` escalates before the weak decoder runs;
    `fallback` also escalates when the weak decoder fails to converge.
    """
    none = np.zeros(len(n_flags), bool)
    rules = {"primary_fail": (none, True)}
    for k in (1, 2, 3, 4):
        rules[f"flag>={k}"] = (n_flags >= k, True)
    for k in (2, 3):
        rules[f"rounds>={k}"] = (flag_rounds >= k, True)
    rules["flag-only"] = (n_flags >= 1, False)
    return rules


def derive(weak, strong, pre, fallback):
    """
    One policy's per-shot outcome from its weak and strong decoder results.
    A flagged shot skips the weak decoder, so it pays neither its time nor its work.
    """
    esc = pre | (~weak["conv"] if fallback else False)
    fail = np.where(esc, strong["fail"], weak["fail"])
    cost = (np.where(pre, 0.0, weak["time_us"].astype(float))
            + np.where(esc, strong["time_us"].astype(float), 0.0))
    work = (np.where(pre, 0, weak["work"].astype(np.int64))
            + np.where(esc, strong["work"].astype(np.int64), 0))
    return dict(esc=esc, fail=fail, cost=cost, work=work,
                silent=~esc & weak["conv"] & weak["fail"],
                unconverged=~esc & ~weak["conv"] & weak["fail"])


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return ph, (0.0 if k == 0 else c - h), (1.0 if k == n else c + h)


def paired_exact(fail_a, fail_b):
    """
    Exact two-sided sign test on discordant shots (McNemar).
    x = shots only A fails, y = shots only B fails.
    """
    x = int((fail_a & ~fail_b).sum())
    y = int((~fail_a & fail_b).sum())
    n = x + y
    if n == 0:
        return x, y, 1.0
    tail = sum(comb(n, i) for i in range(min(x, y) + 1))
    return x, y, min(1.0, 2 * tail / 2 ** n)


def breakeven_ratio(esc_rule, pre, esc_pf):
    """
    A pre-decode rule is cheaper than primary_fail iff  c_weak / c_strong > r*,
    with r* = (P(escalate | rule) - P(escalate | primary_fail)) / P(pre).
    Hardware-independent: compare r* with your decoders' real cost ratio.
    """
    p_pre = float(np.mean(pre))
    if p_pre == 0:
        return float("inf")
    return (float(np.mean(esc_rule)) - float(np.mean(esc_pf))) / p_pre


def strong_ceiling(results, strong_names):
    """
    For each ordered pair of strong decoders: how many shots does A get wrong that
    B gets right? This is the headroom left in the STRONG decoder. If a much
    stronger decoder fixes almost none of A's failures, those failures are
    essentially uncorrectable at this noise level, and no reweighting of A's priors
    (flag-informed or otherwise) can help.
    """
    out = []
    for a in strong_names:
        for b in strong_names:
            if a == b:
                continue
            fa, fb = results[a]["fail"], results[b]["fail"]
            out.append(dict(decoder=a, compared_with=b, failures=int(fa.sum()),
                            fixed_by_other=int((fa & ~fb).sum()),
                            broken_by_other=int((~fa & fb).sum())))
    return out


def flag_diagnostics(results, weak_names, strong_names, n_flags):
    """
    What do the flags actually tell us? Mutual information (in bits) between
    'a flag fired' and 'this decoder was wrong'. Near zero means the trigger
    carries no usable signal, which is the mechanism behind a null result.
    """
    flagged = np.asarray(n_flags) > 0

    def mutual_information(x, y):
        n = len(x)
        total = 0.0
        for xv in (False, True):
            for yv in (False, True):
                both = float(np.mean((x == xv) & (y == yv)))
                px, py = float(np.mean(x == xv)), float(np.mean(y == yv))
                if both > 0 and px > 0 and py > 0:
                    total += both * np.log2(both / (px * py))
        return total

    out = []
    for name in list(weak_names) + list(strong_names):
        wrong = results[name]["fail"]
        rate_f = float(wrong[flagged].mean()) if flagged.any() else float("nan")
        rate_u = float(wrong[~flagged].mean()) if (~flagged).any() else float("nan")
        out.append(dict(decoder=name, flagged_fraction=float(flagged.mean()),
                        error_rate_flagged=rate_f, error_rate_unflagged=rate_u,
                        risk_ratio=(rate_f / rate_u) if rate_u > 0 else float("inf"),
                        mutual_information_bits=mutual_information(flagged, wrong),
                        share_of_failures_on_flagged=(float((wrong & flagged).sum() / wrong.sum())
                                                      if wrong.any() else float("nan"))))
    return out


def min_detectable_discordant(alpha=0.05):
    """Fewest one-sided discordant shots that the paired exact test can call
    significant: with x = 0, p = 2 / 2**y, so y must satisfy 2/2**y < alpha."""
    y = 1
    while 2 / 2 ** y >= alpha:
        y += 1
    return y


def analyse(results, weak_names, strong_names, n_flags, flag_rounds, alpha=0.05,
            min_failures=20, min_saving=0.05):
    """
    Evaluate every (weak, strong, rule). Returns dict with:
      rows    : one dict per (weak, strong, rule)
      pairs   : one verdict per (weak, strong)
      weak    : silent-failure summary per weak decoder
    Verdict for a pair:
      'inconclusive'     -- primary_fail has fewer than `min_failures` failures, too few
                            to resolve accuracy differences: run more shots
      'viable: accuracy' -- some flag rule fails significantly LESS than primary_fail
                            (paired exact test, p < alpha)
      'viable: cost'     -- some flag rule fails on NO MORE shots than primary_fail
                            (paired) and saves at least `min_saving` (default 5%) of
                            primary_fail's measured time per shot
      'not viable'       -- neither
    "Not significantly worse" is deliberately NOT enough for parity: with few
    failures nothing is significant, and that would make every rule look viable.
    `headroom` = silent failures of the weak decoder that the strong decoder gets
    right: no rule can gain more accuracy over primary_fail than this.
    """
    rules = trigger_rules(n_flags, flag_rounds)
    shots = len(n_flags)
    rows, pairs, weak_summary = [], [], {}
    for w in weak_names:
        W = results[w]
        n_conv = int(W["conv"].sum())
        n_sil = int((W["conv"] & W["fail"]).sum())
        weak_summary[w] = dict(converged=n_conv, silent=n_sil,
                               silent_rate=wilson(n_sil, n_conv),
                               silent_flagged=int((W["conv"] & W["fail"] & (n_flags > 0)).sum()),
                               time_us=float(np.median(W["time_us"])))
    for w in weak_names:
        W = results[w]
        for s in strong_names:
            S = results[s]
            pf = derive(W, S, *rules["primary_fail"])
            headroom = int((W["conv"] & W["fail"] & ~S["fail"]).sum())
            best_acc, best_cost = None, None
            for rname, (pre, fb) in rules.items():
                P = derive(W, S, pre, fb)
                x, y, pval = paired_exact(P["fail"], pf["fail"])
                k = int(P["fail"].sum())
                row = dict(weak=w, strong=s, rule=rname, failures=k,
                           ler=wilson(k, shots), escalation=float(P["esc"].mean()),
                           silent=int(P["silent"].sum()), unconverged=int(P["unconverged"].sum()),
                           catch=int((pre & W["conv"] & W["fail"] & ~S["fail"]).sum()),
                           harm=int((pre & W["conv"] & ~W["fail"] & S["fail"]).sum()),
                           worse=x, better=y, p=pval, cost_us=float(P["cost"].mean()),
                           breakeven=(breakeven_ratio(P["esc"], pre, pf["esc"])
                                      if rname != "primary_fail" else float("nan")))
                row.setdefault("saving", 0.0)
                rows.append(row)
                if rname == "primary_fail":
                    continue
                if y > x and pval < alpha and (best_acc is None or y - x > best_acc["better"] - best_acc["worse"]):
                    best_acc = row
                no_loss = x <= y and row["harm"] == 0
                pf_cost = float(pf["cost"].mean())
                row["saving"] = 1 - row["cost_us"] / pf_cost if pf_cost > 0 else 0.0
                if no_loss and row["saving"] >= min_saving and \
                        (best_cost is None or row["cost_us"] < best_cost["cost_us"]):
                    best_cost = row
            # Accuracy and cost are judged separately. Zero headroom settles accuracy
            # outright -- no rule can gain what is not there, however many shots you
            # run -- so it is NOT "inconclusive". Cost needs no failure count at all:
            # it only requires that no rule harmed a shot the weak decoder got right.
            pf_fail = int(pf["fail"].sum())
            if headroom == 0:
                verdict_accuracy = "impossible: no headroom"
            elif best_acc:
                verdict_accuracy = "viable: accuracy"
            elif pf_fail < min_failures:
                verdict_accuracy = "inconclusive"
            else:
                verdict_accuracy = "not viable"
            verdict_cost = "viable: cost" if best_cost else "not viable"
            verdict = ("viable: accuracy" if verdict_accuracy == "viable: accuracy" else
                       "viable: cost" if best_cost else
                       "inconclusive" if verdict_accuracy == "inconclusive" else "not viable")
            pairs.append(dict(weak=w, strong=s, verdict=verdict,
                              verdict_accuracy=verdict_accuracy, verdict_cost=verdict_cost,
                              headroom=headroom,
                              min_detectable_discordant=min_detectable_discordant(alpha),
                              pf_failures=int(pf["fail"].sum()),
                              pf_escalation=float(pf["esc"].mean()),
                              pf_cost_us=float(pf["cost"].mean()),
                              always_failures=int(S["fail"].sum()),
                              best_accuracy_rule=best_acc and best_acc["rule"],
                              best_cost_rule=best_cost and best_cost["rule"],
                              best_saving=max((r["saving"] for r in rows
                                               if r["weak"] == w and r["strong"] == s
                                               and r["rule"] != "primary_fail"
                                               and r["worse"] <= r["better"]), default=0.0),
                              cost_ratio=float(np.median(W["time_us"]) / max(np.median(S["time_us"]), 1e-9))))
    return dict(rows=rows, pairs=pairs, weak=weak_summary, shots=shots,
                strong_ceiling=strong_ceiling(results, strong_names),
                flag_diagnostics=flag_diagnostics(results, weak_names, strong_names, n_flags))
