import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Flagged Decoding")

with app.setup:
    # ---------------------------------------------------------------------------
    # Everything here runs once when the notebook opens AND whenever a worker
    # process (or the companion notebook FlaggedDecoding_legacy.py) loads this
    # file. Only imports and constants belong here; every function lives in its
    # own cell below and is importable as FlaggedDecoding.<name>.
    # ---------------------------------------------------------------------------
    import ast
    import csv
    import datetime
    import glob
    import hashlib
    import io
    import json
    import math
    import multiprocessing as mp
    import os
    import platform
    import re
    import shutil
    import subprocess
    import sys
    import time
    import warnings
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from dataclasses import asdict, dataclass, field, replace
    from math import comb

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.sparse as sp
    import scipy.stats as st
    import stim
    from ldpc.bp_decoder import BpDecoder as LdpcBpBase
    from ldpc.bplsd_decoder import BpLsdDecoder as LdpcBpLsdBase
    from ldpc.bposd_decoder import BpOsdDecoder as LdpcBpOsdBase

    try:
        import numba
        HAVE_NUMBA = True
        NUMBA_STATUS = f"numba {numba.__version__}"
    except Exception as _numba_exc:          # peeling and Relay-BP still work, far slower
        numba = None
        HAVE_NUMBA = False
        NUMBA_STATUS = f"not available ({type(_numba_exc).__name__}: {_numba_exc})"

    try:                                     # optional: pinning to cores, power state
        import psutil
    except Exception:
        psutil = None

    # ---- where this file lives ---------------------------------------------------
    try:
        NOTEBOOK_FILE = os.path.abspath(__file__)
    except NameError:                        # pragma: no cover - exotic runners
        _nb_dir = mo.notebook_dir()
        NOTEBOOK_FILE = os.path.join(str(_nb_dir) if _nb_dir else os.getcwd(),
                                     "FlaggedDecoding.py")
    NOTEBOOK_DIR = os.path.dirname(NOTEBOOK_FILE)
    # On Windows the path can arrive lower-cased (C:\...\flaggeddecoding.py) while
    # the file on disk is FlaggedDecoding.py; Python's import is case-sensitive
    # even there, so take the spelling from the directory listing.
    try:
        _real = {f.lower(): f for f in os.listdir(NOTEBOOK_DIR)}
        NOTEBOOK_FILE = os.path.join(NOTEBOOK_DIR,
                                     _real.get(os.path.basename(NOTEBOOK_FILE).lower(),
                                               os.path.basename(NOTEBOOK_FILE)))
    except OSError:
        pass
    MODULE_NAME = os.path.splitext(os.path.basename(NOTEBOOK_FILE))[0]
    if NOTEBOOK_DIR not in sys.path:
        sys.path.insert(0, NOTEBOOK_DIR)
    RESULTS_DIR = os.path.join(NOTEBOOK_DIR, "results")

    # Worker processes load this file BY PATH (never by module name), so the run
    # works whatever the file is called and whatever the current directory is.
    # The expression is evaluated in the worker with N = module name, P = path.
    WORKER_LOADER = (
        "(lambda s, u: s.modules.get(N) or (lambda spec: (lambda m: ("
        "s.modules.__setitem__(N, m), spec.loader.exec_module(m), m)[-1])"
        "(u.module_from_spec(spec)))(u.spec_from_file_location(N, P)))"
        "(__import__('sys'), __import__('importlib.util', fromlist=['util']))"
    )

    # ---- experiment numbering ------------------------------------------------------
    NUMBERING = "FlaggedDecoding-v1"
    EXPERIMENTS = {
        "E01": "Experiment 1 - noise-model validation and peeling baseline",
        "E02": "Experiment 2 - syndrome augmentation (four-arm grid, Table 5.7)",
        "E03": "Experiment 3 - flag rate against law L6",
        "E04": "Experiment 4 - flag-triggered switching: decoder grid",
        "E05": "Experiment 5 - silent-failure census",
        "E06": "Experiment 6 - surface-code baseline (BP+OSD-0 and MWPM)",
        "E07": "Experiment 7 - accuracy-latency frontier",
        "E08": "Experiment 8 - exact flag-conditioned prior updating",
        "E09": "Experiment 9 - check-weight family (w = 4..12)",
        "E10": "Experiment 10 - prior updating with BP+LSD",
        "E11": "Experiment 11 - prior updating with Relay-BP",
        "E12": "Experiment 12 - inference gap: cycles and min-sum vs sum-product",
    }
    # Files written by earlier versions of this notebook keep their names. They
    # are read where their experiment now lives; EXP1/2/4/5 belong to the
    # companion notebook FlaggedDecoding_legacy.py.
    LEGACY_PREFIX_MAP = {"PK": "E01", "EXP3": "E03", "EXP6": "E04", "EXP7": "E06",
                         "EXP1": "L1 (legacy file)", "EXP2": "L2 (legacy file)", "EXP4": "L3 (legacy file)",
                         "EXP5": "L4 (legacy file)"}

    # ---- decoders: one specification for the whole study -----------------------
    STRONG_MAX_ITER = 30
    LDPC_FIXED = dict(schedule="parallel", ms_scaling_factor=0.625,
                      osd_method="osd_cs", lsd_method="LSD_CS")
    WEAK = {
        "peel":        dict(kind="peel"),
        "peel-naive":  dict(kind="peel", dominance=False),        # uniqueness rule only
        "peel-single": dict(kind="peel", dominance=False, requeue=False),  # one removal pass
        "BP-ms-5":   dict(kind="bp", bp_method="ms", max_iter=5),
        "BP-ms-10":  dict(kind="bp", bp_method="ms", max_iter=10),
        "BP-ms-30":  dict(kind="bp", bp_method="ms", max_iter=30),
        "BP-ms-100": dict(kind="bp", bp_method="ms", max_iter=100),
        "BP-ps-30":  dict(kind="bp", bp_method="ps", max_iter=30),
    }
    STRONG = {
        "OSD-0":   dict(kind="osd", osd_order=0, max_iter=STRONG_MAX_ITER),
        "OSD-0-ps": dict(kind="osd", osd_order=0, max_iter=STRONG_MAX_ITER, bp_method="ps"),
        "OSD-CS2": dict(kind="osd", osd_order=2, max_iter=STRONG_MAX_ITER),
        "OSD-CS4": dict(kind="osd", osd_order=4, max_iter=STRONG_MAX_ITER),
        "OSD-CS7": dict(kind="osd", osd_order=7, max_iter=STRONG_MAX_ITER),
        "LSD-0":   dict(kind="lsd", lsd_order=0, max_iter=STRONG_MAX_ITER),
        "LSD-CS2": dict(kind="lsd", lsd_order=2, max_iter=STRONG_MAX_ITER),
        "LSD-CS4": dict(kind="lsd", lsd_order=4, max_iter=STRONG_MAX_ITER),
        "relay":   dict(kind="relay"),
    }
    CONTROLS = {
        "control-flagged-20%": dict(kind="control", rate=0.2, flagged_only=True),
        "control-flagged-50%": dict(kind="control", rate=0.5, flagged_only=True),
        "control-any-20%":     dict(kind="control", rate=0.2, flagged_only=False),
    }
    CATALOGUE = {**WEAK, **STRONG, **CONTROLS}
    DEFAULT_WEAK = ["peel", "BP-ms-10", "BP-ms-30", "BP-ps-30"]
    DEFAULT_STRONG = ["OSD-0", "OSD-CS4", "LSD-0", "LSD-CS4"]

    # ---- process-local state ------------------------------------------------------
    DECODER_CACHE = {}          # per process: (run key, decoder name) -> decoder
    COMPILED = {}               # per process: compiled numba kernels

    # ---- checkpoints (power-cut protection) -------------------------------------
    # Finished work reaches the disk at most CHECKPOINT_MINUTES apart; a power cut
    # loses at most that much. 0 = save every 250-shot chunk the moment it is done.
    # After a cut: reopen, set the SAME configuration and press the same Run
    # button; finished chunks load from disk.
    CHECKPOINT_MINUTES = 30
    CHUNK_SIZE = 250
    CHECKPOINT_FORMAT = 3          # folders of any other format are ignored
    DECODE_CHUNK_VERSION = 1       # bump when decode_chunk's per-shot output changes
    CACHE_ALL_DECODERS = True      # keep every decoder of the current run per process
    CKPT = {"root": os.path.join(RESULTS_DIR, "checkpoints"),
            "minutes": float(CHECKPOINT_MINUTES),
            "min_shots": 2000}         # smaller runs are not checkpointed

    # ---- result files ----------------------------------------------------------------
    SCHEMA_VERSION = 2
    DETECTOR_CONVENTIONS = ("z-only", "x+z", "surface-all", "surface-z-only",
                            "reference-z-only", "reference-both", "mixed",
                            "not applicable")
    REGENERATION_COMMANDS = ("tables", "figures", "regenerate", "selftest")
    EXPORTED_RUNS = set()       # run ids already exported in this session (export once per run)
    STATUS_ICON = {"PASS": "✅", "FAIL": "❌", "TODO": "⬜"}

    # ---- codes (§3) -------------------------------------------------------------------
    # n and k are verified by the tests; d is the literature value (Bravyi et al.,
    # Nature 627, 778 (2024), Table 3) and is not tested.
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
    # Generalised bicycle codes: each e in a_exps / b_exps is the monomial x^e.
    # d from Panteleev & Kalachev, Quantum 5, 585 (2021); not tested.
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
    # Experiment 9's check-weight family: nine GB codes on ell = 24 (n = 48) with
    # X-check weight w = |a| + |b| = 4..12. Distances are not known in closed
    # form; Experiment 9 estimates an upper bound before it samples.
    W_FAMILY_PRESETS = {
        "GB24 w=4 [[48, 6]]":  dict(ell=24, a_exps=[0, 3], b_exps=[0, 6], n=48, k=6, d=None, w=4),
        "GB24 w=5 [[48, 4]]":  dict(ell=24, a_exps=[0, 13, 23], b_exps=[0, 6],
                                    n=48, k=4, d=None, w=5),
        "GB24 w=6 [[48, 2]]":  dict(ell=24, a_exps=[0, 19], b_exps=[0, 8, 10, 13],
                                    n=48, k=2, d=None, w=6),
        "GB24 w=7 [[48, 4]]":  dict(ell=24, a_exps=[0, 8, 13], b_exps=[0, 9, 11, 20],
                                    n=48, k=4, d=None, w=7),
        "GB24 w=8 [[48, 2]]":  dict(ell=24, a_exps=[0, 19], b_exps=[0, 4, 11, 14, 16, 22],
                                    n=48, k=2, d=None, w=8),
        "GB24 w=9 [[48, 4]]":  dict(ell=24, a_exps=[0, 10, 14, 20, 23], b_exps=[0, 11, 17, 21],
                                    n=48, k=4, d=None, w=9),
        "GB24 w=10 [[48, 2]]": dict(ell=24, a_exps=[0, 2, 8, 15], b_exps=[0, 12, 13, 15, 22, 23],
                                    n=48, k=2, d=None, w=10),
        "GB24 w=11 [[48, 4]]": dict(ell=24, a_exps=[0, 3, 10, 14, 19, 20], b_exps=[0, 2, 4, 9, 21],
                                    n=48, k=4, d=None, w=11),
        "GB24 w=12 [[48, 2]]": dict(ell=24, a_exps=[0, 1, 4, 7, 15, 19],
                                    b_exps=[0, 9, 10, 15, 19, 22], n=48, k=2, d=None, w=12),
    }
    GB_PRESETS.update(W_FAMILY_PRESETS)
    W_FAMILY_KEYS = list(W_FAMILY_PRESETS)

    # ---- the reference noise model (§5.1; Pakhunov 2026) --------------------------
    ALPHA_W3 = 3.505      # Theorem 2, column weight 3
    # Theorem 5 / Table III: |separable| / |all| detector-sharing fault pairs.
    A0_BY_CODE = {"[[18, 4, 4]]": 0.767, "[[32, 8, 6]]": 0.764, "[[72, 12, 6]]": 0.869,
                  "[[144, 12, 12]]": 0.869, "[[288, 12, 18]]": 0.869}
    A0_GROSS = 0.869      # constant across the Gross family
    DENSITY_B = 3.0       # Eq. (2) density correction exp(-B T p) of the exponent
    # Table II, "Actual" column: greedy-peeling success, keyed (code, T, p).
    TABLE_II = {("[[72, 12, 6]]", 12, 0.001): 0.935,
                ("[[144, 12, 12]]", 12, 0.001): 0.879,
                ("[[288, 12, 18]]", 12, 0.001): 0.778,
                ("[[144, 12, 12]]", 12, 0.003): 0.334,
                ("[[288, 12, 18]]", 12, 0.003): 0.119}
    GATE_NOISE = {
        "depolarize2": "DEPOLARIZE2, 15 components per CNOT (this notebook's model)",
        "one-fault-control": "one X mechanism per CNOT, on the control",
        "one-fault-target": "one X mechanism per CNOT, on the target",
        "one-fault-both": "one correlated XX mechanism per CNOT",
    }
    EXTRACTION = {
        "z-only": "Z checks only -- nw CNOTs per round, as Theorem 1 counts",
        "both": "X and Z extraction, 2nw CNOTs per round (section 4's gate count)",
    }
    REFERENCE_VARIANTS = [
        ("reference circuit", dict(gate_noise="depolarize2", extraction="z-only")),
        ("  .. 1 fault/CNOT (control)", dict(gate_noise="one-fault-control", extraction="z-only")),
        ("  .. correlated XX", dict(gate_noise="one-fault-both", extraction="z-only")),
        ("both check families", dict(gate_noise="depolarize2", extraction="both")),
    ]

    # ---- switch policies (§8) -------------------------------------------------------
    TRIGGERS = ("never", "always", "primary_fail", "flag>=1", "flag-only")
    FLAG_TRIGGERS = ("flag>=1", "flag-only")
    TRIGGER_DISPLAY = {"never": "never escalate", "always": "always escalate",
                       "primary_fail": "escalate on primary failure",
                       "flag>=1": "flag or failure (flag ≥ 1)", "flag-only": "flag only",
                       "oracle": "oracle (unimplementable)"}
    # Older ablation files carried `flag` (= flag>=1) and `flag_or_fail`, an
    # identical copy of it.
    LEGACY_TRIGGER_MAP = {"flag": "flag>=1", "flag_or_fail": None}

    # ---- files written by earlier versions of this notebook (§12) -----------------
    LEGACY_KIND_ALIASES = {"validation": "L1-validation", "exp2": "L2-depth-sweep",
                           "ablation": "L3-ablation", "sweep": "L4-sweep",
                           "exp7": "E06-surface", "pakhunov": "E01-peeling"}
    # What some older files cannot say about themselves, with the evidence.
    LEGACY_PROVENANCE = {
        "EXP4_72-12-6_T6_p0.001_25000shots_seed64260921_flags_osd0.json": dict(
            x_detectors=True, status="superseded by the seed-20260921 runs",
            evidence="stored never-escalate LER 0.2714 agrees with X-check detectors on "
                     "(rerun 0.2782 [0.265, 0.292]) and excludes them off (0.2023 [0.190, 0.215])"),
        "EXP5_72-12-6_T6_p0.001_25000shots_seed64260921_flags_osd0.json": dict(
            x_detectors=True, status="superseded by the seed-20260921 runs",
            evidence="same configuration and seed as the matching EXP4 file"),
        "EXP4_144-12-12_T6_p0.002_25000shots_seed64260921_flags_osd0.json": dict(
            x_detectors=True, status="never repeated with Z-only detectors",
            evidence="stored never-escalate LER 0.8342 agrees with X-check detectors on "
                     "(rerun 0.8317 [0.818, 0.845]) and excludes them off (0.7457 [0.730, 0.761])"),
        "EXP5_144-12-12_T6_p0.002_25000shots_seed64260921_flags_osd0.json": dict(
            x_detectors=True, status="never repeated with Z-only detectors",
            evidence="same configuration and seed as the matching EXP4 file"),
        "EXP2_72-12-6_p0.001_20000shots_seed20260923_osd0.json": dict(
            x_detectors=True, status="superseded",
            evidence="detector counts 432/648 at T = 6 and 3.488 expected faults per shot "
                     "occur only with X-check detectors on"),
        "EXP1_72-12-6_T6_p0.001_25000shots.json": dict(
            x_detectors=True, status="superseded",
            evidence="fault counts 17,136 / 15,840 are reproduced exactly with X-check detectors on"),
        "EXP7_surface_T6_20000shots_seed20260929_osd0.json": dict(
            x_detectors=True, detector_convention="surface-all", status="superseded by Experiment 6",
            evidence="48/144/288 detectors = every Stim detector (Z-only gives 28/84/168)"),
        "EXP1_288-12-18_T12_p0.001_10000shots.json": dict(
            status="stale export (its report names a code not in the study)",
            evidence="report.md and the JSON disagree on the code"),
        "EXP2_72-12-6_p0.001_30000shots_seed20260923_osd0_scaled.json": dict(
            status="aborted after T = 6", evidence="only one point; replaced by a later run"),
        "EXP1_72-12-6_T12_p0.001_10000shots.json": dict(
            status="no export bundle", evidence="no report folder was written"),
    }

    # ---- policies, figures and the four-arm runner (§13–§15) ----------------------
    COST_CONVENTIONS = {"derive": "weak time charged only when weak runs",
                        "charge-primary": "weak time charged on every shot (older convention)"}
    PLOT_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                    "#8c564b", "#e377c2", "#17becf", "#7f7f7f", "#bcbd22"]
    ZERO_LABEL = "▽ = no failures: 97.5% one-sided upper bound"
    ARM_STYLE = {"unflagged": ("#555555", "s", "no flag qubits"),
                 "blind": ("#c2c2c2", "^", "flags built, decoder blind"),
                 "sighted": ("#8c8c8c", "o", "flag rows added to H"),
                 "updated": ("#1f6fb4", "D", "flag posterior on the priors")}
    FOUR_ARMS = ("unflagged", "blind", "sighted", "updated")
    PAIRED_COMPARISONS = (("blind", "sighted"), ("sighted", "updated"), ("blind", "updated"))
    UNPAIRED_COMPARISONS = (("sighted", "unflagged"), ("blind", "unflagged"),
                            ("updated", "unflagged"))
    SHOT_SCALES = {"1 (registered design)": 1.0, "0.1 (rehearsal)": 0.1, "0.01 (smoke test)": 0.01}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Flagged Decoding for Bivariate Bicycle Codes

    Undergraduate thesis · BRAC University

    *Authors and supervisor: add your names here before submission.*

    ---

    **Flag qubits pay for themselves as decoder input, not as a switching
    trigger.** On 250,000 paired shots of [[72,12,6]], no flag-derived rule fails
    on fewer shots than the rule "escalate when the primary decoder fails" (1,279
    failures). The oracle that escalates exactly the shots peeling gets wrong
    fails on 1,216 (36.9% escalated): it escalates a strict *subset* of
    primary_fail's shots, and no implementable flag rule escalates less often than
    primary_fail; peeling made no silent error in 8.75 million shots, so there is
    nothing for a trigger to catch. As decoder input the same bits pay on every
    code tested. Read as extra checks they lower the failure rate by up to 2.6×;
    turned into exact per-shot priors (Gallager's parity-check posterior,
    conditioned flag by flag) they lower it 8.5–69× further with BP+OSD-0 — on
    [[144,12,12]] to zero failures in 150,000 shots — while making the decode
    2.3× ([[72,12,6]]) to 4.5× ([[144,12,12]]) faster than the sighted arm's. The advantage over the flag rows is an inference gap of
    min-sum BP, which Relay-BP closes. Twelve of fourteen pre-registered tests
    are confirmed (§17, §19).

    **How to use this notebook.** Open it with `marimo edit FlaggedDecoding.py`.
    Every section ends with its self-tests; the experiments unlock once all of
    them pass (§16). Each experiment has a **Run** button, which is the only
    thing that writes a result file, and a picker that loads a saved run for
    display. `python FlaggedDecoding.py tables|figures` rebuilds the thesis
    tables and figures from the saved result files (§18). The four experiments
    that the thesis no longer uses live in the companion notebook
    `FlaggedDecoding_legacy.py`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Outline

    **Foundations** — 1. motivation · 1b. glossary · 1c. related work · 2. GF(2) linear algebra ·
    3. codes (BB, GB and the w-family) · 4. circuit-level noise and flag qubits ·
    5. detector error model and the reference noise model · 6. surface-code
    problem · 7. decoders (peeling variants, BP, BP+OSD/LSD, MWPM, Relay-BP) ·
    8. switch policies · 9. statistics · 10. parallel decoding and checkpoints ·
    11. flag structure (groups, posteriors, laws L1–L7) · 12. result files and
    provenance · 13. policy analysis · 14. plotting · 15. shared four-arm runner ·
    16. planning, scope and implementation status.

    **Experiments** (result files start with the prefix):

    | # | prefix | experiment | thesis items | main statistics |
    |---|---|---|---|---|
    | 1 | `E01` | noise-model validation and peeling baseline | Table 5.1 | Wilson intervals; analytic predictions |
    | 2 | `E02` | syndrome augmentation: unflagged / blind / sighted / updated | Table 5.7, Fig 5.3, Table 4.1 | exact rate-ratio (unpaired); McNemar (paired); Holm T4, T7–T9 |
    | 3 | `E03` | flag rate against law L6 | Table 5.6 | Wilson intervals; descriptive χ² |
    | 4 | `E04` | flag-triggered switching: decoder grid | Tables 5.2–5.5, Fig 5.2 | McNemar with Bonferroni; placebo triggers; chunk-block cost intervals; mutual information |
    | 5 | `E05` | silent-failure census | §5.2.1 | one-sided Clopper–Pearson bounds |
    | 6 | `E06` | surface-code baseline, BP+OSD-0 and MWPM | §5.1, Fig 5.1 | Wilson; McNemar per point; bootstrap threshold interval |
    | 7 | `E07` | accuracy–latency frontier | Table 5.9, Fig 5.5 | Pareto dominance; Wilson |
    | 8 | `E08` | exact flag-conditioned prior updating | Table 5.8, Fig 5.4 | McNemar; Wilcoxon on times; WLS slope; Holm T1–T3, T5, T6, T11, T14 |
    | 9 | `E09` | check-weight family, w = 4…12 | laws chapter | WLS slope (Holm T10) |
    | 10 | `E10` | prior updating with BP+LSD | §6.3 | McNemar (Holm T12) |
    | 11 | `E11` | prior updating with Relay-BP | §6.3 | McNemar (Holm T13) |
    | 12 | `E12` | inference gap: Tanner-graph cycles and min-sum vs sum-product | §6.2 | exact cycle counts; McNemar within and across runs (descriptive) |

    **After the experiments** — 17. pre-registered confirmatory family (Holm,
    m = 14) · 18. regenerating the thesis tables and figures · 19. results and
    conclusions.

    **Older result files** keep their names and load where their experiment now
    lives: `PK_*` → Experiment 1, `EXP3_*` → 3, `EXP6_*.npz` → 4, `EXP7_*` → 6;
    `EXP1_*`, `EXP2_*`, `EXP4_*` and `EXP5_*` open in `FlaggedDecoding_legacy.py`.
    """)
    return


@app.cell(hide_code=True)
def _():
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
       the fast decoder fails", on identical shots (Experiments 4 and 5).
    2. *As decoder input* — compare an unflagged circuit, a flagged circuit whose
       flag outcomes are hidden from the decoder, a flagged circuit whose outcomes
       it reads as extra checks, and the same flagged circuit decoded with
       flag-conditioned priors (Experiments 2, 7, 8, 9, 10, 11).
    3. *Why* — measure how often flags fire and how much they tell us
       (Experiments 3 and 4), against closed-form laws (§11).

    **What would falsify the trigger hypothesis.** If the fast decoder is never
    confidently wrong, no trigger can improve accuracy; and if flags fire on almost
    every shot, no trigger can be selective. Both are measured, not assumed.

    **Pairing.** Arms decoded on the *same* circuit and samples (blind, sighted,
    updated; every trigger rule) are compared shot by shot with the exact McNemar
    test. Arms on *different* circuits (unflagged vs flagged) are independent
    samples even when they share a seed; they are compared with an unpaired exact
    rate-ratio interval and never with McNemar.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1b. Glossary

    Every symbol that appears in a table or on an axis in this notebook.

    | term | meaning |
    |---|---|
    | **n, k, d** | physical qubits, logical qubits, code distance: a code is written [[n, k, d]] |
    | **w** | X-check weight = number of monomials in the two defining polynomials |
    | **p** | physical error rate: the probability parameter of every noise channel in the circuit |
    | **T** | number of syndrome-extraction rounds in one shot |
    | **shot** | one run of the experiment: prepare, run T rounds, measure, decode once |
    | **detector** | a parity of measurements that is deterministic without noise; the decoder sees these, not raw measurements |
    | **syndrome** | the vector of detector outcomes for one shot |
    | **fault mechanism** | one independent error the noise model can produce, with its detector signature; the columns of H |
    | **λ (lambda)** | expected number of fault mechanisms firing per shot = sum of all fault probabilities (§5) |
    | **λ_k, soft value** | in §11 only: $1 - 2p_k = \tanh(L_k/2)$ for fault $k$ with prior LLR $L_k$ — Gallager's quantity; a group's parity bias is the product of its soft values |
    | **H, L** | detector-by-fault check matrix and observable-by-fault matrix of the detector error model |
    | **LER** | logical error rate: fraction of shots whose predicted observable flips differ from the truth |
    | **converged** | the decoder found a correction that reproduces the syndrome exactly |
    | **silent failure** | the weak decoder converged but was logically wrong: a confident mistake |
    | **headroom** | silent failures of the weak decoder that the strong decoder gets right — the most accuracy any trigger could gain |
    | **escalation rate** | fraction of shots sent to the strong decoder |
    | **never / always** | reference policies: run only the weak decoder / only the strong decoder |
    | **primary_fail** | escalate exactly when the weak decoder does not converge (no flag information) |
    | **flag>=k** | escalate before decoding when at least k flag detectors fired, and also on primary failure ("flag or failure" for k = 1) |
    | **rounds>=k** | escalate when flags fired in at least k distinct rounds, and on primary failure |
    | **flag-only** | escalate when any flag fired and *never* on primary failure |
    | **oracle** | escalate exactly the shots the weak decoder gets wrong (unimplementable bound) |
    | **work** | decoder-native operation count: detector visits for peeling, iterations for BP |
    | **p50 / p99** | median and 99th-percentile decode time; p99 is the tail that a real-time decoder must meet |
    | **Wilson interval** | 95% two-sided interval for a rate; at zero failures its upper end is a 97.5% one-sided bound |
    | **arm** | one leg of a controlled comparison: *unflagged*, *blind*, *sighted*, *updated* |
    | **blind / sighted / updated** | flag detectors removed from H / kept in H as checks / removed from H and used to compute per-shot priors |
    | **r\*** | break-even cost ratio: the weak/strong conditional cost ratio above which a pre-decode rule is cheaper |
    | **T\*** | crossover round count at which flags stop paying; printed only when the fitted slope is significant and T\* lies inside the measured range |

    **Decoder names.** `peel` queue-based greedy peeling with the dominance rule ·
    `peel-naive` uniqueness rule · `peel-single` one removal pass · `BP-ms-N`
    min-sum belief propagation, N iterations · `BP-ps-N` product-sum BP · `OSD-0`
    BP + ordered statistics, order 0 (min-sum, scaling 0.625, parallel schedule,
    30 iterations everywhere) · `OSD-0-ps` the same with exact sum-product BP (Experiment 12) · `OSD-CSk` / `LSD-CSk` combination sweep of order k
    · `MWPM` minimum-weight perfect matching (surface code only) · `relay` Relay-BP
    · `control-*` positive controls that fail silently on purpose.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1c. Related work, and what is new here

    **The flag posterior is classical.** The per-shot prior update of
    Experiments 8–11 applies Gallager's parity-check posterior (1962; 1963,
    Lemma 4.1 and Theorem 4.1) — the same identity as the sum-product check-node
    update and the piling-up lemma — and it has already been stated for quantum
    decoding (Gong, Cammerer and Renes, 2023, Lemma 3.1). What this notebook adds
    is the *model mapping* on flagged BB-code circuits (each flag detector is the
    XOR of exactly $w$ identifiable DEM fault mechanisms, law L1), the observation
    that disjoint flag groups (Condition 4.1) make the posterior exact for every
    flag simultaneously, and the closed-form laws L1–L7 that follow (§11).

    **Flags as decoder input.**

    * *J. Chen, Yi, Liang and Wang (2025)*, "Improved belief propagation decoding
      algorithms for surface codes" (*IEEE TQE*, arXiv:2407.11523). Of their
      three methods only EWAInit-BP touches
      the priors, adapting them from BP's own dynamics; it is demonstrated under
      code-capacity noise (no measurement errors), with BB codes as preliminary
      evidence. It does not use flags.
    * *Takada and Fujii (2024)*, "Improving threshold for fault-tolerant color
      code quantum computing by flagged weight optimization" (*PRX Quantum* **5**,
      030352; arXiv:2402.13958), estimate fault probabilities conditioned on the
      local flag pattern from $10^6$ Monte Carlo samples, store them as a lookup
      table and use them as $-\log$ weights in an integer-programming decoder. No
      closed form is used; their flags overlap (flag-matrix column weight up to
      3), so Condition 4.1 does not hold there. The often-quoted threshold gain
      (0.14% → 0.27%) is for the whole protocol, including cat-state gadgets and
      a deflagging step; their flag-weight-only ablation gives 0.183(2)% →
      0.268(9)%.
      In their conclusions they consider flags as extra parity checks and write
      that "that method could achieve lower logical error rates than FWO",
      defending their method on complexity and decoder portability, not
      accuracy.
    * *Vittal et al. (2024)*, "Flag proxy networks" (arXiv:2409.14283), reweight
      decoder priors from flag outcomes on hyperbolic surface and colour QLDPC
      codes, with heuristic (underived, unnormalised) weights. BB codes are not
      studied there.
    * *Anker and Marvian (2022)*, "Flag gadgets based on classical codes"
      (arXiv:2212.10738), view flag measurements as a classical code over fault
      locations; in that language Condition 4.1 says that code's parity-check
      matrix has column weight ≤ 1.

    We are not aware of earlier work that derives exact flag-conditioned priors
    for BB codes, states the disjointness condition that makes them exact, or
    compares them on paired samples with the flag rows given to BP as checks.

    **The comparison Takada and Fujii anticipated.** Read as extra checks, the
    flag rows give BP the same bits as the prior update and *more* structure (the
    within-group correlation that a prior vector drops, §11). On paper the
    sighted arm should therefore do at least as well. With min-sum BP+OSD-0 it
    does far worse — 254 against 7 failures on [[72,12,6]] at $10^{-3}$
    (Experiment 8). That is an **inference** gap, not an information gap:
    Experiment 12 and Experiment 11 show the sighted arm improving to 75 failures
    with exact sum-product BP and to 6 with Relay-BP on the same shots, while the
    updated arm stays at 7–8. Short cycles are not the cause (a flag row lies on
    far fewer 4-cycles than a detector row). Takada and Fujii's expectation is
    thus borne out for a strong decoder and fails for min-sum BP+OSD, the decoder
    class in common use for BB codes (§19, R7).

    **Beyond the simulated noise model.** The posterior is exact for the DEM's
    noise model. The update needs only per-mechanism probabilities, so priors
    estimated from device data could be conditioned on the flags in the same
    closed form; this is left to future work.
    """)
    return


@app.function
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


@app.function
def render_checks(title, results):
    counts = {s: sum(r["status"] == s for r in results) for s in STATUS_ICON}
    lines = [f"### Tests — {title}",
             f"{counts['PASS']} passed · {counts['FAIL']} failed · {counts['TODO']} to do",
             ""]
    for r in results:
        detail = f" — `{r['detail']}`" if r["status"] == "FAIL" and r["detail"] else ""
        lines.append(f"- {STATUS_ICON[r['status']]} {r['name']}{detail}")
    return mo.md("\n".join(lines))


@app.function
def all_pass(*result_lists):
    return all(r["status"] == "PASS" for rl in result_lists for r in rl)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. GF(2) linear algebra

    Everything downstream — code dimension $k$, logical operators, scoring —
    depends on rank and nullspace computations **over GF(2)**, not over the reals.
    """)
    return


@app.function
def gf2_dense(M):
    """Dense uint8 copy of M with entries reduced mod 2 (accepts scipy.sparse)."""
    return np.array(M.toarray() if hasattr(M, "toarray") else M, dtype=np.uint8) % 2


@app.function
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


@app.function
def gf2_rank(M):
    """Rank of M over GF(2). Must return 0 for an empty matrix."""
    if M.shape[0] == 0 or M.shape[1] == 0:
        return 0
    return len(gf2_rref(M)[1])


@app.function
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


@app.function
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


@app.cell
def _():
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
def _():
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


@app.class_definition
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


@app.function
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


@app.function
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


@app.function
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


@app.function
def bb_from_preset(key, presets):
    """Convenience: build a preset from BB_PRESETS by its label."""
    s = presets[key]
    return make_bb_code(s["L"], s["M"], s["a_terms"], s["b_terms"], name=key)


@app.function
def gb_from_preset(key, presets):
    """Convenience: build a preset from GB_PRESETS by its label."""
    s = presets[key]
    return make_gb_code(s["ell"], s["a_exps"], s["b_exps"], name=key)


@app.cell
def _():
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
def _():
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


@app.function
def code_from_key(key):
    """Build any preset, BB or GB (including the w-family), from its label."""
    if key in BB_PRESETS:
        return bb_from_preset(key, BB_PRESETS)
    if key in GB_PRESETS:
        return gb_from_preset(key, GB_PRESETS)
    raise KeyError(f"unknown code {key!r}")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Circuit-level noise and flag qubits

    Hook errors are a property of the **syndrome-extraction circuit**: a fault on
    an X-check ancilla (the CNOT control) mid-extraction propagates to every data
    qubit it has not yet touched. They cannot appear in a code-capacity model.

    **Experiment:** Z-basis memory, $T$ rounds, circuit-level depolarising noise
    with strength $p$ on resets, single- and two-qubit gates, and measurements.

    **Noise model.** `DEPOLARIZE2(p)` after every CNOT, a flip with probability
    $p$ on every reset and every measurement, and `X_ERROR(p)` on the data after
    initialisation. **Idle qubits are not depolarised**: `idle_noise` defaults to
    False and every experiment uses the default (no `DEPOLARIZE1` anywhere), so
    all runs, old and new, share one noise model.

    **Flags:** one flag qubit per X-check ancilla. The flag CNOTs open after the
    first data layer and close before the last one, so an X fault on the ancilla
    inside that window reaches the flag once and flips it, while a fault before
    the window reaches it twice (the copies cancel) and a fault after it never.
    The window holds $w - 2$ data layers, which is why every flag group has
    $|G| = w$ members (law L1, §11).

    The diagram below is read from `build_memory_circuit`'s own output for check
    0 of [[72,12,6]]; a self-test fails if the schedule it draws drifts from the
    builder.

    **X-basis memory.** A Z-basis memory decodes X errors, which only X-check
    ancillas spread into hooks — so that is where the flags sit. The mirror
    experiment is an X-basis memory with flags on the Z-check ancillas. It needs
    no second builder: conjugating every qubit by a Hadamard turns a Z-basis
    memory of the code with its check families **swapped** into an X-basis memory
    of the original code with flags on its Z checks. The noise model is invariant
    (two-qubit depolarising is symmetric; a Z-basis reset or measurement flip
    becomes an X-basis one), and detectors and observables are the same parities,
    so the detector error models coincide. `x_basis_mirror(code)` returns that
    swapped code; `four_arm_run(..., basis="X")` uses it. Within a round the
    mirrored circuit measures the flagged (here: Z) checks first.
    """)
    return


@app.function(hide_code=True)
def flag_circuit_diagram(code, build_memory_circuit, check_schedule):
    """
    X-check 0 and its flag qubit in one round, read from the REAL circuit.

    The gate order is taken from build_memory_circuit's own output (not drawn
    by convention), so the figure cannot drift from the builder: the flag
    opens after data layer 0 and closes before the last data layer, and an
    X fault on the ancilla flips the flag only between the two flag CNOTs.
    """
    from matplotlib.patches import Circle, Rectangle
    n, mx, mz = code.n, code.hx.shape[0], code.hz.shape[0]
    xa0, f0 = n, n + mx + mz
    circ = build_memory_circuit(code, 1, 0.0, use_flags=True, x_detectors=False)
    seq = []                                      # (kind, target) for ancilla 0
    for inst in circ.flattened():
        if inst.name != "CX":
            continue
        t = [x.value for x in inst.targets_copy()]
        for a, b in zip(t[0::2], t[1::2]):
            if a == xa0:
                seq.append(("flag" if b == f0 else "data", b))
    data = [q for k, q in seq if k == "data"]
    assert data == list(check_schedule(code.hx, code.lattice)[:, 0]), "schedule drift"
    fig, ax = plt.subplots(figsize=(10, 4.2))
    rows = {f0: 1.25, xa0: 0.0}
    rows.update({q: -1.0 - i for i, q in enumerate(data)})
    xs = [1.0 + i for i in range(len(seq))]
    k_open = next(i for i, (k, _) in enumerate(seq) if k == "flag")
    k_close = max(i for i, (k, _) in enumerate(seq) if k == "flag")
    ax.add_patch(Rectangle((xs[k_open], -0.45), xs[k_close] - xs[k_open], 2.1,
                           fc="#fbe9e6", ec="none", zorder=0))
    for q, y in rows.items():
        lab = (r"flag $f_0$ $|0\rangle$" if q == f0 else
               r"ancilla $a_0$ $|{+}\rangle$" if q == xa0 else f"data q{q}")
        ax.plot([0.3, xs[-1] + 1.2], [y, y], color="0.6", lw=1, zorder=1)
        ax.text(0.2, y, lab, ha="right", va="center", fontsize=8)
    for i, ((kind, q), x) in enumerate(zip(seq, xs)):
        col = "#c0392b" if kind == "flag" else "#1f4e9c"
        ax.plot([x, x], [rows[xa0], rows[q]], color=col, lw=1.4, zorder=2)
        ax.add_patch(Circle((x, rows[xa0]), 0.07, color=col, zorder=3))
        ax.add_patch(Circle((x, rows[q]), 0.15, fc="white", ec=col, lw=1.4, zorder=3))
        ax.plot([x - 0.15, x + 0.15], [rows[q]] * 2, color=col, lw=1.2, zorder=4)
        ax.plot([x, x], [rows[q] - 0.15, rows[q] + 0.15], color=col, lw=1.2, zorder=4)
        if kind == "flag":
            ax.text(x, rows[f0] + 0.3, "open" if i == k_open else "close",
                    ha="center", fontsize=8, color=col)
    for s in range(len(seq) - 1):                 # fault slots on the ancilla
        flips = sum(1 for k, _ in seq[s + 1:] if k == "flag") % 2 == 1
        x = (xs[s] + xs[s + 1]) / 2
        ax.text(x, -0.3, str(s), ha="center", va="center", fontsize=7,
                color="#c0392b" if flips else "0.5")
    ax.text(xs[-1] + 1.3, rows[f0], r"$M_Z$ → flag detector", fontsize=8, va="center")
    ax.text(xs[-1] + 1.3, rows[xa0], r"$M_X$ (off in Z memory)", fontsize=8, va="center")
    ax.set_xlim(-1.6, xs[-1] + 3.4)
    ax.set_ylim(-len(data) - 0.6, 2.0)
    ax.axis("off")
    ax.set_title(f"{code.name}: X-check 0 with its flag, one round. Ancilla X faults in "
                 "red-numbered slots flip the flag; data layers inside the window "
                 f"= {k_close - k_open - 1}, so |G| = {k_close - k_open + 1} = w",
                 fontsize=9)
    fig.tight_layout()
    return fig


@app.function
def x_basis_mirror(code):
    """
    The code with its X and Z check families (and logicals) swapped. A Z-basis
    memory of it is, up to a global Hadamard, an X-basis memory of `code` with
    the flags on its Z-check ancillas (§4); the detector error model is the same.
    """
    if code.name.endswith("(X basis)"):
        raise ValueError(f"{code.name} is already a mirror")
    return replace(code, name=f"{code.name} (X basis)", hx=code.hz, hz=code.hx,
                   rank_x=code.rank_z, rank_z=code.rank_x, lx=code.lz, lz=code.lx)


@app.function
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


@app.function
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


@app.function
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


@app.function
def expected_num_detectors(code, rounds, use_flags, x_detectors=True):
    """Detector count implied by the ordering contract above."""
    mx, mz = code.hx.shape[0], code.hz.shape[0]
    per_round_flags = mx if use_flags else 0
    per_round_x = mx if x_detectors else 0
    return (mz * rounds + per_round_x * (rounds - 1)
            + per_round_flags * rounds + mz)


@app.cell
def _():
    flag_circuit_diagram(bb_from_preset("[[72, 12, 6]]", BB_PRESETS), build_memory_circuit,
                         check_schedule)
    return


@app.cell
def _():
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

    def _x_basis_mirror_is_a_valid_flagged_circuit():
        """The mirror is a valid CSS code; its flagged circuit is silent at p = 0
        and has the same flag structure (L1, L2, L7) as the Z-basis circuit."""
        c = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        m = x_basis_mirror(c)
        assert (m.n, m.k) == (c.n, c.k) and m.hx.shape == c.hz.shape
        assert not ((m.hx @ m.hz.T).toarray() % 2).any()
        circ = build_memory_circuit(m, 2, 0.0, use_flags=True, x_detectors=False)
        det, obs = circ.compile_detector_sampler(seed=1).sample(20, separate_observables=True)
        assert not det.any() and not obs.any()
        rz, rx = flag_structure_report(c, 2, 1e-3), flag_structure_report(m, 2, 1e-3)
        for k in ("group_sizes", "disjoint", "added_mechanisms", "faults_total", "separated_by_flags"):
            assert rz[k] == rx[k], (k, rz[k], rx[k])

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
        ("X-basis mirror: valid code, silent circuit, same flag structure", _x_basis_mirror_is_a_valid_flagged_circuit),
        ("x_detectors=False: DEM density drops toward the analytic fault count",
         _z_only_thins_the_dem),
    ])
    render_checks("circuit", tests_circuit)
    return (tests_circuit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Detector error model

    Stim reduces the noisy circuit to independent fault mechanisms. Each fault
    $j$ has a probability $p_j$, a detector signature (column $j$ of $H$), and a
    set of observables it flips (column $j$ of $L$). Decoding means: given the
    detector syndrome $s$, find $\hat e$ with $H\hat e = s$; predict $L\hat e$.
    """)
    return


@app.function
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


@app.cell
def _():
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
def _():
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

    **What this reference validates, and what it does not.** Theorems 1 and 2
    predict the *size and density of the fault model* — how many fault
    mechanisms a circuit has and how many fire per shot. That is a property of
    the noise model and the DEM extraction, which **every** experiment shares,
    whichever decoder it runs; so this part of the reference matters for the
    whole thesis. Theorems 3 and 5 predict how often greedy *peeling* succeeds;
    that part only concerns the peeling primary of the switching experiments
    (4 and 5). The decoder-input experiments (2, 7–11) rest on BP+OSD, BP+LSD and
    Relay-BP, whose pipeline is validated separately against a decoder-
    independent benchmark: the rotated surface code, decoded by both BP+OSD-0 and
    MWPM on identical samples (Experiment 6).

    This section builds the reference circuit so the claims are measurements:

    | Theorem | Prediction | Checked against |
    |---|---|---|
    | 1 | $N = n(wT + T/2 + 1)$ fault mechanisms | exact fault count |
    | 2 | $\lambda = \alpha n T p$, $\alpha = 3.505$ | sum of DEM priors |
    | 3, 5 | $P_{peel} = \exp(-\bar{d}\lambda^2 e^{-BTp} / 2N)^{A_0}$, $B = 3$ | Table II's 93.5% (0.937 with Eq. (2)'s correction, 0.935 without) |

    The reference circuit is for **validation only**. It has no flag qubits, so
    every experiment keeps using `build_memory_circuit`.

    The model reproduces Table II; our **peeler does not** (Experiment 1). All
    three peeling variants — queue-based with the dominance rule (`peel`, used
    for every reported number), the uniqueness rule (`peel-naive`) and a single
    removal pass (`peel-single`) — resolve the same shots on [[72,12,6]], and all
    fall below even the single-pass bound $\exp(-\beta\lambda^2)$. The cause of
    that residual is not isolated: it is not a single-pass versus queue effect.
    """)
    return


@app.function
def theory_faults(code, rounds):
    """
    Theorem 1: N = n(wT + T/2 + 1) = nwT CNOT + (n/2)T measurement + n boundary.

    w is the COLUMN weight (qubit degree), not the row weight. The theorem is
    stated for BB codes, whose column weight is uniform; `uniform` reports
    whether that holds, so a GB code with |a| != |b| is never silently
    compared (its count is nominal and must not be quoted as a check).
    """
    cols = np.asarray(code.hz.sum(axis=0)).ravel()
    w = int(cols[0])
    return dict(total=int(code.n * (w * rounds + rounds / 2 + 1)),
                cnot=int(code.n * w * rounds),
                measurement=int(code.n // 2 * rounds),
                boundary=int(code.n),
                column_weight=w,
                uniform=bool((cols == w).all()))


@app.function
def theory_lambda(code, rounds, p, alpha=ALPHA_W3):
    """Theorem 2: expected number of fault mechanisms firing per shot."""
    return alpha * code.n * rounds * p


@app.function
def theory_peel(faults, lam, mean_degree, a0=A0_GROSS, rounds=None, p=None, B=DENSITY_B):
    """
    Theorems 3 and 5: P_peel = exp(-beta lambda^2 c)^A0, beta = d_bar / 2N.

    With rounds and p given, the exponent carries Eq. (2)'s density correction
    c = exp(-B T p); without them c = 1 (the uncorrected form, reported beside
    it in Table 5.1). Evaluated on a circuit's OWN measured graph this is a
    model prediction; the distance between it and a measured peeling rate is
    reported as a residual whose cause is NOT isolated (all three peeler
    variants resolve the same shots on [[72,12,6]], below even the single-pass
    bound exp(-beta lambda^2)).
    """
    corr = 1.0 if rounds is None or p is None else float(np.exp(-B * rounds * p))
    return float(np.exp(-mean_degree / (2 * faults) * lam ** 2 * corr) ** a0)


@app.function
def build_reference_circuit(code, rounds, p, extraction="z-only", gate_noise="depolarize2",
                            measurement_noise=True, boundary_noise=True):
    """
    Z-basis memory in the Pakhunov reference model: Z-check extraction only, no
    flags, DEPOLARIZE2 per CNOT, one measurement flip per ancilla per round and
    boundary faults from the data reset and final readout.

    extraction="both" reproduces section 4's gate count instead, and the
    gate_noise variants apply one X mechanism per CNOT, so the cause of a gap
    can be isolated one axis at a time. Detectors are Z-check only either way,
    matching build_memory_circuit(x_detectors=False, use_flags=False).
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
            else:
                for a, b in zip(controls, targets):
                    c.append("CORRELATED_ERROR", [stim.target_x(a), stim.target_x(b)], p)
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
        if xanc:
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

    d_idx = measure_into("M", data)
    for i in range(mz):
        c.append("DETECTOR", [rec(d_idx[q]) for q in hz_rows[i]] + [rec(prev_z[i])],
                 [i, rounds, 3])
    for j in range(lz.shape[0]):
        c.append("OBSERVABLE_INCLUDE", [rec(d_idx[q]) for q in np.flatnonzero(lz[j])], j)
    return c


@app.function
def reference_report(code, rounds, p, extraction="z-only", gate_noise="depolarize2"):
    """Measured DEM quantities beside what Theorems 1, 2, 3 and 5 predict."""
    circ = build_reference_circuit(code, rounds, p, extraction=extraction,
                                   gate_noise=gate_noise)
    H, _L, priors = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
    th = theory_faults(code, rounds)
    lam = float(priors.sum())
    Hb = (H > 0).astype(np.int8)
    G = (Hb.T @ Hb).tocsr()
    deg = float((np.diff(G.indptr) - 1).mean())
    a0 = A0_BY_CODE.get(getattr(code, "name", ""), A0_GROSS)
    return dict(detectors=int(H.shape[0]), faults=int(H.shape[1]),
                theory_faults=th["total"], fault_ratio=H.shape[1] / th["total"],
                lam=lam, theory_lambda=theory_lambda(code, rounds, p),
                lambda_ratio=lam / theory_lambda(code, rounds, p),
                mean_degree=deg, a0=a0,
                theory_peel=theory_peel(H.shape[1], lam, deg, a0, rounds, p),
                theory_peel_uncorrected=theory_peel(H.shape[1], lam, deg, a0),
                single_pass_bound=float(np.exp(-deg / (2 * H.shape[1]) * lam ** 2)),
                uniform_weight=th["uniform"], column_weight=th["column_weight"])


@app.cell
def _():
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
            target = TABLE_II[(key, T, 1e-3)]
            assert abs(r["theory_peel"] - target) < 0.02, \
                f"{key}: predicted {r['theory_peel']:.3f} against Table II {target:.3f}"
            assert abs(r["theory_peel_uncorrected"] - target) < 0.02

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

    def _density_correction_value():
        """Eq. (2)'s correction moves [[72,12,6]] T=12 from 0.935 to 0.937."""
        r = reference_report(_code(), 12, 1e-3)
        assert abs(r["theory_peel"] - 0.9372) < 5e-4, r["theory_peel"]
        assert r["theory_peel"] > r["theory_peel_uncorrected"]
        assert r["single_pass_bound"] < r["theory_peel"]

    def _bad_reference_options_rejected():
        c = _code()
        for kw in (dict(gate_noise="nope"), dict(extraction="nope")):
            try:
                build_reference_circuit(c, 3, 1e-3, **kw)
            except ValueError:
                continue
            raise AssertionError(f"{kw} was accepted")
        try:
            build_reference_circuit(c, 0, 1e-3)
        except ValueError:
            return
        raise AssertionError("rounds=0 was accepted")

    def _noise_axis_keeps_fault_count():
        c = _code()
        def nf(**kw):
            circ = build_reference_circuit(c, 12, 1e-3, **kw)
            return dem_to_matrices(circ.detector_error_model(decompose_errors=False))[0].shape[1]
        assert nf() == nf(gate_noise="one-fault-control")
        assert nf(extraction="both") / nf() > 1.35

    def _reference_matches_experiment_circuit_with_both():
        """extraction='both' reproduces section 4's circuit (no flags, Z detectors)."""
        c = _code()
        a = build_reference_circuit(c, 4, 1e-3, extraction="both")
        b = build_memory_circuit(c, 4, 1e-3, use_flags=False, x_detectors=False)
        ha = dem_to_matrices(a.detector_error_model(decompose_errors=False))
        hb = dem_to_matrices(b.detector_error_model(decompose_errors=False))
        assert ha[0].shape == hb[0].shape and abs(ha[2].sum() - hb[2].sum()) < 1e-9

    tests_reference = run_checks([
        ("reference circuit is well formed and silent at p = 0", _reference_circuit_is_well_formed),
        ("Theorem 1 fault count is reproduced exactly", _theorem_1_is_exact),
        ("Theorem 2 lambda is reproduced within 10%", _theorem_2_lambda),
        ("mean fault-graph degree matches the paper's 52.3", _mean_degree_matches_paper),
        ("analytic peeling prediction reproduces Table II", _analytic_peel_reproduces_table_ii),
        ("section 4's circuit deviates from Theorem 1, as designed",
         _experiment_circuit_deviates_on_purpose),
        ("column-weight uniformity is reported", _non_uniform_weight_is_flagged),
        ("Eq. (2) density correction gives 0.9372 on [[72,12,6]]", _density_correction_value),
        ("bad gate_noise, extraction and rounds are rejected", _bad_reference_options_rejected),
        ("noise channel keeps the fault count; extraction raises it >35%",
         _noise_axis_keeps_fault_count),
        ("extraction='both' reproduces section 4's circuit", _reference_matches_experiment_circuit_with_both),
    ])
    render_checks("5.1 reference model", tests_reference)
    return (tests_reference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. The surface-code problem

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

    **Detector convention.** Stim's generated memory circuit also compares the
    opposite-type stabilisers (48/144/288 detectors at $T = 6$ for $d = 3/5/7$),
    which can never flip the Z observable. As for the BB codes, those detectors
    are dropped by default, leaving $(d^2-1)/2\cdot(T+1)$ = 28/84/168.
    **Decoder strength.** On these circuits MWPM is *more* accurate than BP+OSD-0
    at low $p$, so Experiment 6 measures both on the same samples and never
    claims BP+OSD-0 is the stronger decoder.
    """)
    return


@app.function
def surface_code_problem(distance, rounds, p, basis="Z", x_detectors=False):
    """
    Rotated surface-code memory as a decoding problem, in Stim's own circuit.

    Returns the same (circuit, H, L, priors) shape the BB path produces, so a
    baseline run reuses the decoders and metrics unchanged.

    Detector convention (the same rule as every BB experiment): with
    x_detectors=False (the default) only the detectors of the memory basis are
    kept. Stim's generated memory circuit compares every stabiliser of both
    types from round 2 on; the opposite-type detectors report errors that can
    never flip the observable, so they are dropped. A detector is kept iff its
    (x, y) coordinate already carries a detector at t = 0, which holds exactly
    for the memory-basis stabilisers. That leaves (d^2 - 1)/2 * (T + 1)
    detectors; x_detectors=True keeps Stim's (d^2 - 1) * T.
    """
    if basis.upper() not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    if distance < 3 or distance % 2 == 0:
        raise ValueError("rotated surface code needs an odd distance >= 3")
    gen = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}",
        distance=distance, rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p)
    if x_detectors:
        circ = gen
    else:
        flat = gen.flattened()
        first = {tuple(inst.gate_args_copy()[:2]) for inst in flat
                 if inst.name == "DETECTOR" and len(inst.gate_args_copy()) >= 3
                 and inst.gate_args_copy()[2] == 0}
        circ = stim.Circuit()
        for inst in flat:
            if inst.name == "DETECTOR" and tuple(inst.gate_args_copy()[:2]) not in first:
                continue
            circ.append(inst)
    H, L, priors = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
    used = sorted({q for inst in circ.flattened() for t in inst.targets_copy()
                   if t.is_qubit_target for q in [t.value]})
    return dict(circuit=circ, H=H, L=L, priors=priors,
                distance=distance, rounds=rounds, p=p, basis=basis.upper(),
                x_detectors=bool(x_detectors),
                detector_convention="surface-all" if x_detectors else "surface-z-only",
                qubits_used=len(used), qubit_index_span=int(circ.num_qubits),
                name=f"surface d={distance}")


@app.function
def is_graphlike(circuit):
    """
    True when every fault mechanism has at most two symptoms, i.e. the DEM is a
    matching graph. Surface codes are; bivariate bicycle codes are not, which
    is why this thesis decodes BB codes with BP+OSD instead of MWPM.
    """
    try:
        circuit.detector_error_model(decompose_errors=True)
        return True
    except ValueError:
        return False


@app.function
def dem_mean_degree(H):
    """Mean number of other faults sharing a detector with a given fault."""
    Hb = (H > 0).astype(np.int8)
    G = (Hb.T @ Hb).tocsr()
    return float((np.diff(G.indptr) - 1).mean())


@app.function
def mwpm_decode(circuit, det, obs):
    """
    Minimum-weight perfect matching (pymatching) on a graphlike circuit.

    Returns per-shot `fail` and `time_us` (batch time spread evenly), so the
    surface-code baseline can put MWPM beside BP+OSD-0 on identical samples.
    """
    import pymatching
    dem = circuit.detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    det = np.asarray(det, dtype=np.uint8)
    t0 = time.perf_counter()
    pred = m.decode_batch(det)
    dt = (time.perf_counter() - t0) * 1e6
    fail = np.any(np.asarray(pred, dtype=np.uint8) != np.asarray(obs, dtype=np.uint8), axis=1)
    return dict(fail=fail, time_us=np.full(len(det), dt / max(1, len(det))))


@app.cell
def _():
    def _builds_at_every_distance():
        for d in (3, 5, 7):
            for T in (3, 6):
                pr = surface_code_problem(d, T, 1e-3)
                assert pr["circuit"].num_detectors == (d * d - 1) // 2 * (T + 1), \
                    f"d={d} T={T}: {pr['circuit'].num_detectors} detectors (Z-only)"
                full = surface_code_problem(d, T, 1e-3, x_detectors=True)
                assert full["circuit"].num_detectors == (d * d - 1) * T
                assert pr["detector_convention"] == "surface-z-only"
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
        for x in (False, True):
            pr = surface_code_problem(5, 3, 0.0, x_detectors=x)
            det, obs = pr["circuit"].compile_detector_sampler(seed=4).sample(
                32, separate_observables=True)
            assert not det.any() and not obs.any()
        assert surface_code_problem(3, 3, 1e-3)["qubits_used"] == 17

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
        assert pt["detectors"] == (3 * 3 - 1) // 2 * (3 + 1) == 16
        assert pt["time_p50_us"] > 0
        assert "mwpm_failures" in pt and ("sequential" in pt["backend"] or "parallel" in pt["backend"])
        assert pt["qubits_used"] == 17 and pt["detector_convention"] == "surface-z-only"

    def _threshold_estimate_finds_a_crossing():
        # synthetic: d=7 beats d=5 below 3e-3 and loses above it
        pts = []
        for p, l5, l7 in [(1e-3, 1e-2, 3e-3), (3e-3, 4e-2, 4e-2), (1e-2, 1e-1, 2e-1)]:
            pts.append(dict(distance=5, p=p, ler=l5))
            pts.append(dict(distance=7, p=p, ler=l7))
        t = threshold_crossing(pts, 5, 7)
        assert t is not None and 1e-3 <= t <= 1e-2, f"threshold {t}"
        # no crossing -> None, rather than a fabricated number
        flat = [dict(distance=d, p=p, ler=1e-2 if d == 5 else 1e-3)
                for d in (5, 7) for p in (1e-3, 3e-3)]
        assert threshold_crossing(flat, 5, 7) is None
        assert threshold_crossing([dict(distance=3, p=1e-3, ler=1e-2)], 5, 7) is None

    def _sweep_plot_renders():
        import io
        pts = [dict(distance=d, p=p, ler=1e-3 * d, ci_low=5e-4 * d, ci_high=2e-3 * d,
                    failures=10, shots=1000)
               for d in (3, 5, 7) for p in (1e-3, 3e-3)]
        pts[0]["failures"] = 0                      # a zero-failure point
        fig = plot_surface_sweep(pts)
        fig.savefig(io.BytesIO(), format="png")
        # the zero-failure point is a hollow marker at its bound, no bar
        ax = fig.axes[0]
        assert any(getattr(l, "get_marker", lambda: None)() == "v" for l in ax.get_lines())

    tests_surface = run_checks([
        ("builds at d = 3, 5, 7 with the Z-only detector count (x_detectors switch)", _builds_at_every_distance),
        ("even or too-small distances are rejected", _rejects_bad_parameters),
        ("noiseless circuit fires nothing", _noiseless_is_silent),
        ("decodes with the same BP+OSD decoder, corrections consistent",
         _decodes_with_the_same_decoder),
        ("surface DEM is graphlike, BB DEM is not", _surface_is_graphlike_and_bb_is_not),
        ("surface fault graph is far sparser than a BB one", _surface_graph_is_far_sparser),
        ("Experiment 6 sweep point runs end to end, MWPM beside BP+OSD-0", _sweep_point_runs_end_to_end),
        ("threshold estimate finds a crossing and refuses to invent one",
         _threshold_estimate_finds_a_crossing),
        ("surface sweep plot renders", _sweep_plot_renders),
    ])
    render_checks("6 surface-code problem", tests_surface)
    return (tests_surface,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Decoders

    Every notebook decoder exposes `decode(syndrome) -> DecodeResult`; the
    compiled decoders used by the experiments (`make_decoder`) return
    `(correction, converged, work)` so they can run in worker processes.

    | role | decoder | failure signal |
    |---|---|---|
    | weak | **greedy peeling** `peel` (queue-based, dominance rule; after Pakhunov, 2026) | residual syndrome not cleared |
    | weak | `peel-naive` (uniqueness rule) / `peel-single` (one removal pass) | residual syndrome not cleared |
    | weak | **BP** (min-sum `BP-ms-N`, product-sum `BP-ps-N`) | BP did not converge |
    | strong | **BP+OSD** (Roffe et al., 2020) | never fails to reproduce the syndrome |
    | strong | **BP+LSD** (Hillmann et al., 2025) | never fails to reproduce the syndrome |
    | strong | **Relay-BP** (Müller et al., 2025): built-in numba port; IBM's `relay-bp` package optional | no leg converged |
    | surface only | **MWPM** (pymatching) | — |

    **One specification.** Every BP-family decoder is built by one factory with
    the fixed settings in `LDPC_FIXED` (parallel schedule, min-sum scaling 0.625,
    `osd_cs` / `LSD_CS`) and the strong decoder always runs
    `STRONG_MAX_ITER = 30` BP iterations: `strong_spec(order)` is the only way an
    experiment names BP+OSD, so an arm labelled "BP+OSD-0" is the same decoder in
    every experiment.

    **Peeling rule.** A fault is peeled when its whole detector signature is
    active and every other fully-active fault touching those detectors has a
    signature *contained in* its own. One fault explaining the syndrome is far
    more likely than several, so a dominating fault wins. After each removal the
    detectors of the removed fault's neighbours are re-queued. `peel-naive`
    refuses whenever a rival is fully active; `peel-single` also skips the
    re-queueing. All three resolve the same shots on the reference circuit
    (Experiment 1), so the peeler's shortfall against the analytical model is not
    a single-pass effect.

    **Soft output.** BP-based decoders return their posterior log-likelihood
    ratios in `soft` (positive = "no fault").

    **Cost.** `work` counts decoder-native operations: detector visits for
    peeling, BP iterations for the BP family. OSD/LSD post-processing cost is
    *not* included in `work` -- compare those decoders by wall-clock time.
    """)
    return


@app.class_definition
@dataclass
class DecodeResult:
    correction: np.ndarray      # uint8, length num_faults
    converged: bool             # True iff H @ correction == syndrome (mod 2)
    work: int = 0               # decoder-specific operation count
    escalated: bool = False     # set by SwitchPolicy when the secondary ran
    soft: np.ndarray = None     # posterior LLRs (BP family); None for peeling


@app.class_definition
class PeelingDecoder:
    """
    Queue-based greedy peeling over the DEM, with the dominance rule -- the
    pure-Python reference implementation of `FastPeeling` (which the
    experiments run). A self-test checks the two agree shot by shot.

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
        self.fault_dets = [Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]]
                           for j in range(self.num_faults)]
        self.det_faults = [Hr.indices[Hr.indptr[i]:Hr.indptr[i + 1]]
                           for i in range(self.num_detectors)]
        self.fault_set = [frozenset(d.tolist()) for d in self.fault_dets]
        self._neighbourhood = {}

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
        full = [int(j) for j in self.det_faults[d] if self._fully_active(j, s)]
        if not full:
            return None
        best = max(full, key=lambda j: len(self.fault_dets[j]))
        best_set = self.fault_set[best]
        for d2 in self.fault_dets[best]:
            for k in self.det_faults[d2]:
                if k != best and self._fully_active(k, s) \
                        and not self.fault_set[k] <= best_set:
                    return None
        return best

    def decode(self, syndrome):
        s = np.asarray(syndrome, dtype=np.uint8).copy()
        e = np.zeros(self.num_faults, dtype=np.uint8)
        queue = deque(np.flatnonzero(s).tolist())
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
            e[j] ^= 1
            s[self.fault_dets[j]] ^= 1
            for d3 in self._nearby_detectors(j):
                if s[d3] and not queued[d3]:
                    queue.append(int(d3))
                    queued[d3] = True
        return DecodeResult(e, not s.any(), visits)


@app.function
def ldpc_kwargs(priors, max_iter, bp_method="ms"):
    """Settings shared by every BP-family decoder (LDPC_FIXED), so comparisons are fair."""
    kw = dict(error_channel=[float(x) for x in priors], max_iter=max_iter,
              bp_method=bp_method, schedule=LDPC_FIXED["schedule"])
    if bp_method == "ms":
        kw["ms_scaling_factor"] = LDPC_FIXED["ms_scaling_factor"]
    return kw


@app.class_definition
class BpDecoder:
    """
    Min-sum belief propagation (weak decoder).

    Contract: converged=True iff BP converged, which in ldpc means the
    correction reproduces the syndrome. work = BP iterations used.
    soft = posterior log-likelihood ratios.
    """

    def __init__(self, H, priors, max_iter=STRONG_MAX_ITER):
        self.dec = LdpcBpBase(sp.csr_matrix(H, dtype=np.uint8), **ldpc_kwargs(priors, max_iter))

    def decode(self, syndrome):
        e = self.dec.decode(np.asarray(syndrome, dtype=np.uint8))
        return DecodeResult(np.asarray(e, dtype=np.uint8), bool(self.dec.converge),
                            int(self.dec.iter), soft=np.array(self.dec.log_prob_ratios))


@app.class_definition
class BpOsdDecoder:
    """
    BP + ordered-statistics decoding (strong decoder), STRONG_MAX_ITER BP
    iterations by default -- the same decoder `strong_spec(order)` builds.

    ldpc runs OSD only when BP fails to converge; otherwise the BP solution is
    returned unchanged. Contract: the correction always reproduces the
    syndrome, so converged is always True. work = BP iterations used.
    """

    def __init__(self, H, priors, max_iter=STRONG_MAX_ITER, osd_order=0):
        self.dec = LdpcBpOsdBase(sp.csr_matrix(H, dtype=np.uint8),
                              osd_method=LDPC_FIXED["osd_method"], osd_order=osd_order,
                              **ldpc_kwargs(priors, max_iter))

    def decode(self, syndrome):
        e = self.dec.decode(np.asarray(syndrome, dtype=np.uint8))
        return DecodeResult(np.asarray(e, dtype=np.uint8), True,
                            int(self.dec.iter), soft=np.array(self.dec.log_prob_ratios))


@app.class_definition
class BpLsdDecoder:
    """
    BP + localized statistics decoding (strong decoder). Same contract as
    BpOsdDecoder: LSD runs only when BP fails, and the correction always
    reproduces the syndrome.
    """

    def __init__(self, H, priors, max_iter=STRONG_MAX_ITER, lsd_order=0):
        self.dec = LdpcBpLsdBase(sp.csr_matrix(H, dtype=np.uint8),
                              lsd_method=LDPC_FIXED["lsd_method"], lsd_order=lsd_order,
                              **ldpc_kwargs(priors, max_iter))

    def decode(self, syndrome):
        e = self.dec.decode(np.asarray(syndrome, dtype=np.uint8))
        return DecodeResult(np.asarray(e, dtype=np.uint8), True,
                            int(self.dec.iter), soft=np.array(self.dec.log_prob_ratios))


@app.function
def peel_kernel_py(syndrome, row_ptr, row_idx, col_ptr, col_idx, n_faults, dominance,
                   requeue):
    """
    Queue-based peeling. A fully-active fault is peeled when every other
    fully-active fault touching its detectors has a signature contained in
    its own (dominance=1), or only when it has no fully-active rival at all
    (dominance=0). requeue=1 revisits the neighbours of every removed fault;
    requeue=0 visits each initially active detector once (a single pass).
    Returns (correction, converged, detector_visits).
    """
    m = syndrome.shape[0]
    s = syndrome.copy()
    e = np.zeros(n_faults, dtype=np.uint8)
    queued = np.zeros(m, dtype=np.uint8)
    mark = np.zeros(m, dtype=np.uint8)
    queue = np.empty(m, dtype=np.int64)
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
            continue
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
        if requeue == 1:
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


@app.function
def compiled_peel_kernel():
    """The numba-compiled peel kernel, built once per process (pure Python if
    numba is missing). On-disk caching is used when numba can locate this file."""
    if "peel" not in COMPILED:
        if HAVE_NUMBA:
            try:
                COMPILED["peel"] = numba.njit(cache=True, nogil=True)(peel_kernel_py)
            except Exception:
                COMPILED["peel"] = numba.njit(cache=False, nogil=True)(peel_kernel_py)
        else:
            COMPILED["peel"] = peel_kernel_py
    return COMPILED["peel"]


@app.class_definition
class FastPeeling:
    """Compiled peeling decoder: decode(s) -> (correction, converged, work)."""

    def __init__(self, H, priors=None, dominance=True, requeue=True):
        self.dominance = np.uint8(1 if dominance else 0)
        self.requeue = np.uint8(1 if requeue else 0)
        Hr, Hc = sp.csr_matrix(H), sp.csc_matrix(H)
        self.row_ptr = Hr.indptr.astype(np.int64)
        self.row_idx = Hr.indices.astype(np.int64)
        self.col_ptr = Hc.indptr.astype(np.int64)
        self.col_idx = Hc.indices.astype(np.int64)
        self.n_faults = H.shape[1]
        self.kernel = compiled_peel_kernel()
        self.params = dict(kind="peel", dominance=bool(dominance), requeue=bool(requeue),
                           implementation="numba" if HAVE_NUMBA else "python")

    def decode(self, syndrome):
        return self.kernel(np.ascontiguousarray(syndrome, dtype=np.uint8), self.row_ptr,
                           self.row_idx, self.col_ptr, self.col_idx, self.n_faults,
                           self.dominance, self.requeue)


@app.function
def strong_spec(osd_order=0, kind="osd"):
    """The one strong-decoder specification: BP+OSD (or BP+LSD) of a given
    order with STRONG_MAX_ITER iterations and the fixed LDPC_FIXED settings."""
    if kind == "osd":
        return dict(kind="osd", osd_order=int(osd_order), max_iter=STRONG_MAX_ITER)
    if kind == "lsd":
        return dict(kind="lsd", lsd_order=int(osd_order), max_iter=STRONG_MAX_ITER)
    raise ValueError(f"kind must be 'osd' or 'lsd', got {kind!r}")


@app.class_definition
class LdpcDecoder:
    """ldpc BP / BP+OSD / BP+LSD with the fixed shared settings:
    decode(s) -> (e, converged, work). `params` records every setting passed."""

    def __init__(self, H, priors, kind, max_iter=STRONG_MAX_ITER, bp_method="ms",
                 osd_order=0, lsd_order=0):
        from ldpc.bp_decoder import BpDecoder
        from ldpc.bplsd_decoder import BpLsdDecoder
        from ldpc.bposd_decoder import BpOsdDecoder
        kw = dict(error_channel=[float(x) for x in priors], max_iter=int(max_iter),
                  bp_method=bp_method, schedule=LDPC_FIXED["schedule"])
        if bp_method == "ms":
            kw["ms_scaling_factor"] = LDPC_FIXED["ms_scaling_factor"]
        H = sp.csr_matrix(H, dtype=np.uint8)
        self.kind = kind
        self.params = {k: v for k, v in kw.items() if k != "error_channel"}
        self.params["kind"] = kind
        if kind == "bp":
            self.dec = BpDecoder(H, **kw)
        elif kind == "osd":
            self.dec = BpOsdDecoder(H, osd_method=LDPC_FIXED["osd_method"],
                                    osd_order=osd_order, **kw)
            self.params.update(method=LDPC_FIXED["osd_method"], order=int(osd_order))
        elif kind == "lsd":
            self.dec = BpLsdDecoder(H, lsd_method=LDPC_FIXED["lsd_method"],
                                    lsd_order=lsd_order, **kw)
            self.params.update(method=LDPC_FIXED["lsd_method"], order=int(lsd_order))
        else:
            raise ValueError(kind)

    def update_priors(self, priors):
        """Replace the channel probabilities in place (per-shot prior updating)."""
        self.dec.update_channel_probs([float(x) for x in priors])

    def decode(self, syndrome):
        e = np.asarray(self.dec.decode(np.asarray(syndrome, dtype=np.uint8)), dtype=np.uint8)
        converged = bool(self.dec.converge) if self.kind == "bp" else True
        return e, converged, int(self.dec.iter)


@app.class_definition
class SilentControl:
    """
    POSITIVE CONTROL decoder. Decodes with peeling, then on a chosen fraction of
    shots adds `vector` to the correction. The result still reproduces the
    syndrome and still reports converged=True, but it flips a logical
    observable -- a silent failure by construction. With flagged_only=True
    those failures sit exactly on flagged shots, so a working analysis must
    report non-zero headroom AND (with the placebo comparison) call a flag rule
    flag-informative. control-any-20% injects failures regardless of flags: it
    must be found viable but NOT flag-informative.
    """

    def __init__(self, H, priors, vector, rate=0.2, flagged_only=True, flag_mask=None,
                 seed=1234):
        self.base = FastPeeling(H, priors)
        self.vector = np.asarray(vector, dtype=np.uint8)
        self.rate = float(rate)
        self.flagged_only = bool(flagged_only)
        self.flag_mask = None if flag_mask is None else np.asarray(flag_mask, bool)
        self.seed = int(seed)
        self.params = dict(kind="control", rate=self.rate, flagged_only=self.flagged_only)

    def decode(self, syndrome):
        e, converged, work = self.base.decode(syndrome)
        if not converged:
            return e, converged, work
        if self.flagged_only:
            if self.flag_mask is None or not np.asarray(syndrome)[self.flag_mask].any():
                return e, converged, work
        nz = np.flatnonzero(syndrome)[:8]
        h = (int(np.asarray(syndrome, dtype=np.uint8).sum()) * 2654435761 + self.seed) & 0xFFFFFFFF
        if len(nz):
            h ^= int(np.dot(nz, [1, 3, 5, 7, 11, 13, 17, 19][:len(nz)]))
        if (h % 1000) / 1000.0 < self.rate:
            return (e ^ self.vector) % 2, True, work
        return e, converged, work


@app.function
def decoder_spec_full(spec):
    """
    Every parameter a decoder spec resolves to, JSON-safe (never arrays), for
    the result files: peel -> dominance/requeue/implementation; bp/osd/lsd ->
    bp_method, max_iter, schedule, ms_scaling_factor, method, order; control
    -> rate, flagged_only; relay -> its tuning parameters.
    """
    s = dict(CATALOGUE[spec]) if isinstance(spec, str) else dict(spec)
    kind = s.get("kind")
    if kind == "peel":
        return dict(kind="peel", dominance=bool(s.get("dominance", True)),
                    requeue=bool(s.get("requeue", True)),
                    implementation="numba" if HAVE_NUMBA else "python")
    if kind in ("bp", "osd", "lsd"):
        bp_method = s.get("bp_method", "ms")
        out = dict(kind=kind, bp_method=bp_method,
                   max_iter=int(s.get("max_iter", STRONG_MAX_ITER)),
                   schedule=LDPC_FIXED["schedule"])
        if bp_method == "ms":
            out["ms_scaling_factor"] = LDPC_FIXED["ms_scaling_factor"]
        if kind == "osd":
            out.update(method=LDPC_FIXED["osd_method"], order=int(s.get("osd_order", 0)))
        if kind == "lsd":
            out.update(method=LDPC_FIXED["lsd_method"], order=int(s.get("lsd_order", 0)))
        return out
    if kind == "control":
        return dict(kind="control", rate=float(s.get("rate", 0.2)),
                    flagged_only=bool(s.get("flagged_only", True)))
    if kind == "relay":
        d = dict(gamma0=0.65, pre_iter=80, num_sets=100, set_max_iter=60,
                 gamma_dist_interval=[-0.24, 0.66], stop_nconv=5, seed=0, backend="native")
        d.update({k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items() if k != "kind"})
        return dict(kind="relay", **d)
    raise ValueError(f"unknown decoder kind {kind!r}")


@app.function
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
    if kind == "relay":
        return RelayDecoder(H, priors, **spec)
    return LdpcDecoder(H, priors, kind, **spec)


@app.function
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


@app.function
def gf2_solve_bitpacked(rows, b, n_cols):
    """
    Solve M v = b over GF(2) for a wide M given as packed uint64 rows.
    Returns a particular solution v (uint8, length n_cols) or None.
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
        return None
    v = np.zeros(n_cols, dtype=np.uint8)
    for i, c in enumerate(pivots):
        v[c] = rhs[i]
    return v


@app.function
def logical_null_vector(H, L, observable=0):
    """
    A fault vector v with H v = 0 and L[observable] v = 1: invisible in the
    syndrome, but it flips a logical observable. Adding it to any correction
    turns that correction into a SILENT failure.
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


@app.function
def parallel_workers(shots, workers, min_shots=2000):
    """
    Each worker process builds its own decoders, which costs about a second.
    That only pays off for runs of at least `min_shots` shots.
    """
    return workers if shots >= min_shots else 1


@app.function
def chunk_for(shots, workers, checkpoint=False):
    """Chunk size: fixed when checkpointing (so a resumed run splits identically),
    otherwise a few chunks per worker so the load balances."""
    if checkpoint:
        return CHUNK_SIZE
    return max(25, -(-shots // max(1, workers * 3)))


@app.function
def relay_kernel_py(syn, row_ptr, row_cols, col_ptr, col_edges, lpr, gamma0, pre_iter,
                    num_sets, set_max_iter, glo, ghi, stop_nconv, rng, out):
    """
    Relay-BP (Müller et al., 2025), following the reference Rust implementation
    of the relay-bp package: min-sum BP with per-variable memory strengths
    gamma, run first with gamma = gamma0 ("pre" leg), then in up to `num_sets`
    further legs whose gammas are drawn uniformly from [glo, ghi). Each leg
    restarts the messages but keeps the posteriors of the previous leg. Every
    converged leg is scored by its quality sum(log-prior ratio of the flipped
    faults); the best is kept and the search stops after `stop_nconv`
    converged legs. Writes the correction into `out`; returns
    (converged, total iterations). `rng` is a one-element uint64 xorshift state.
    """
    m = len(row_ptr) - 1
    n = len(col_ptr) - 1
    ne = len(row_cols)
    v2c = np.empty(ne)
    c2v = np.zeros(ne)
    post = lpr.copy()
    mem = np.full(n, gamma0)
    dec = np.zeros(n, np.uint8)
    best = np.zeros(n, np.uint8)
    first = np.zeros(n, np.uint8)
    for e in range(ne):
        v2c[e] = lpr[row_cols[e]]
    best_q = np.inf
    n_conv = 0
    total = 0
    for leg in range(num_sets + 1):
        max_iter = pre_iter if leg == 0 else set_max_iter
        if leg > 0:
            for j in range(n):                              # xorshift64* uniform draws
                x = rng[0]
                x ^= x >> np.uint64(12)
                x ^= x << np.uint64(25)
                x ^= x >> np.uint64(27)
                rng[0] = x
                u = float((x * np.uint64(2685821657736338717)) >> np.uint64(11)) / 9007199254740992.0
                mem[j] = glo + (ghi - glo) * u
            for e in range(ne):
                c2v[e] = 0.0
                v2c[e] = lpr[row_cols[e]]
        ok = False
        it = 0
        while it < max_iter:
            it += 1
            for r in range(m):                              # check -> variable
                sign = syn[r] == 1
                min1 = np.inf
                min2 = np.inf
                arg = -1
                for e in range(row_ptr[r], row_ptr[r + 1]):
                    val = v2c[e]
                    if val < 0:
                        sign = not sign
                    a = abs(val)
                    if a <= min1:
                        min2 = min1
                        min1 = a
                        arg = e
                    elif a <= min2:
                        min2 = a
                for e in range(row_ptr[r], row_ptr[r + 1]):
                    s_e = sign != (v2c[e] < 0)
                    mag = min2 if e == arg else min1
                    c2v[e] = -mag if s_e else mag
            for j in range(n):                              # variable -> check, with memory
                tot = lpr[j] * (1.0 - mem[j]) + post[j] * mem[j]
                for k in range(col_ptr[j], col_ptr[j + 1]):
                    tot += c2v[col_edges[k]]
                post[j] = tot
                for k in range(col_ptr[j], col_ptr[j + 1]):
                    e = col_edges[k]
                    v2c[e] = tot - c2v[e]
                dec[j] = 1 if tot <= 0.0 else 0
            ok = True                                       # does dec reproduce the syndrome?
            for r in range(m):
                par = 0
                for e in range(row_ptr[r], row_ptr[r + 1]):
                    par ^= dec[row_cols[e]]
                if par != syn[r]:
                    ok = False
                    break
            if ok:
                break
        total += it
        if leg == 0:
            for j in range(n):
                first[j] = dec[j]
        if ok:
            n_conv += 1
            q = 0.0
            for j in range(n):
                if dec[j]:
                    q += lpr[j]
            if q < best_q:
                best_q = q
                for j in range(n):
                    best[j] = dec[j]
            if n_conv >= stop_nconv:
                break
    src = best if n_conv > 0 else first
    for j in range(n):
        out[j] = src[j]
    return n_conv > 0, total


@app.function
def compiled_relay_kernel():
    """The Relay-BP kernel, numba-compiled once per process (Python fallback)."""
    if "relay" not in COMPILED:
        COMPILED["relay"] = (numba.njit(cache=False)(relay_kernel_py) if HAVE_NUMBA
                             else relay_kernel_py)
    return COMPILED["relay"]


@app.class_definition
class RelayDecoder:
    """
    Relay-BP behind the (correction, converged, work) contract.

    backend "native" (default): this notebook's numba port of the reference
    implementation (relay_kernel_py) -- no extra package, and a prior update
    only recomputes the log-prior ratios. backend "package": IBM's relay-bp
    package (Rust; `pip install relay-bp`, which needs a Rust toolchain where
    no wheel exists, e.g. on Windows); it has no in-place prior update, so
    update_priors rebuilds the decoder. A self-test checks the two backends
    agree shot by shot on the deterministic first leg and statistically on the
    full relay. The random draws differ between backends, so individual
    corrections after the first leg can differ.
    """

    def __init__(self, H, priors, gamma0=0.65, pre_iter=80, num_sets=100, set_max_iter=60,
                 gamma_dist_interval=(-0.24, 0.66), stop_nconv=5, seed=0, backend="native"):
        if backend not in ("native", "package"):
            raise ValueError("backend must be 'native' or 'package'")
        self.backend = backend
        self.H = sp.csr_matrix(H, dtype=np.uint8)
        self.H.sort_indices()
        self.kw = dict(gamma0=float(gamma0), pre_iter=int(pre_iter), num_sets=int(num_sets),
                       set_max_iter=int(set_max_iter),
                       gamma_dist_interval=tuple(float(x) for x in gamma_dist_interval),
                       stop_nconv=int(stop_nconv), seed=int(seed))
        self.params = dict(kind="relay", backend=backend,
                           **{k: (list(v) if isinstance(v, tuple) else v) for k, v in self.kw.items()})
        if backend == "package":
            try:
                import relay_bp
            except ImportError as exc:
                raise ImportError("backend='package' needs `pip install relay-bp` (Rust toolchain "
                                  "required where no wheel exists); the default backend "
                                  "'native' needs nothing extra") from exc
            self._cls = relay_bp.RelayDecoderF64
        else:
            self.row_ptr = self.H.indptr.astype(np.int64)
            self.row_cols = self.H.indices.astype(np.int64)
            n = self.H.shape[1]
            order = np.argsort(self.row_cols, kind="stable")      # edges grouped by column
            self.col_edges = order.astype(np.int64)
            self.col_ptr = np.zeros(n + 1, np.int64)
            np.cumsum(np.bincount(self.row_cols, minlength=n), out=self.col_ptr[1:])
            s = (int(seed) * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019) & 0xFFFFFFFFFFFFFFFF
            self.rng = np.array([s or 1], dtype=np.uint64)
            self.out = np.zeros(n, np.uint8)
            self.kernel = compiled_relay_kernel()
        self.update_priors(priors)

    def update_priors(self, priors):
        p = np.clip(np.asarray(priors, dtype=np.float64), 1e-15, 1 - 1e-15)
        if self.backend == "package":
            self.dec = self._cls(self.H, error_priors=p, **self.kw)
        else:
            self.lpr = np.log((1.0 - p) / p)

    def decode(self, syndrome):
        s = np.asarray(syndrome, dtype=np.uint8)
        if self.backend == "package":
            r = self.dec.decode_detailed(s)
            return np.asarray(r.decoding, dtype=np.uint8), bool(r.success), int(r.iterations)
        k = self.kw
        ok, it = self.kernel(s, self.row_ptr, self.row_cols, self.col_ptr, self.col_edges, self.lpr,
                             k["gamma0"], k["pre_iter"], k["num_sets"], k["set_max_iter"],
                             k["gamma_dist_interval"][0], k["gamma_dist_interval"][1],
                             k["stop_nconv"], self.rng, self.out)
        return self.out.copy(), bool(ok), int(it)


@app.cell
def _():
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
                assert r.converged == _consistent(H, s, r), "converged flag disagrees with H·e = s"
        return check

    def _always_consistent(make):
        def check():
            H, pr, det = _setup()
            dec = make(H, pr)
            for s in det:
                r = dec.decode(s)
                assert r.converged and _consistent(H, s, r), "correction does not reproduce the syndrome"
        return check

    def _soft_output():
        H, pr, det = _setup()
        for make in (BpDecoder, BpOsdDecoder, BpLsdDecoder):
            r = make(H, pr).decode(det[0])
            assert r.soft is not None and r.soft.shape == (H.shape[1],), make.__name__
            assert np.isfinite(r.soft).all(), f"{make.__name__}: non-finite LLRs"
        assert PeelingDecoder(H, pr).decode(det[0]).soft is None

    def _strong_keeps_bp_solution():
        # OSD/LSD act only when BP fails: on BP-converged shots the strong decoder
        # returns BP's own answer. This is why "BP-ms primary, BP+OSD secondary"
        # repeats work rather than adding a decoder (Experiment 4, nested pairs).
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
        c = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        circ = build_memory_circuit(c, 3, 4e-3, use_flags=True)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        det = circ.compile_detector_sampler(seed=21).sample(150).astype(np.uint8)
        dec = PeelingDecoder(H, pr)
        for s in det:
            r = dec.decode(s)
            residual = (s ^ (H @ r.correction) % 2).astype(np.uint8)
            again = dec.decode(residual)
            assert not again.correction.any(), "peeling left a peelable fault behind"

    def _work_counts():
        H, pr, det = _setup()
        nz = next(s for s in det if s.any())
        assert PeelingDecoder(H, pr).decode(nz).work > 0
        r = BpDecoder(H, pr, max_iter=7).decode(nz)
        assert 1 <= r.work <= 7, f"BP work {r.work} outside [1, max_iter]"

    def _fast_peel_matches_reference():
        """The compiled kernel and the pure-Python PeelingDecoder agree shot by shot."""
        H, pr, det = _setup()
        fast, ref = FastPeeling(H, pr), PeelingDecoder(H, pr)
        for s in det:
            e1, c1, _ = fast.decode(s)
            r = ref.decode(s)
            assert c1 == r.converged and np.array_equal(e1, r.correction)

    def _peel_variants_are_wired():
        """peel = dominance + requeue; peel-naive = uniqueness; peel-single = no requeue."""
        H, pr, det = _setup(p=3e-3)
        d = {n: make_decoder(n, H, pr) for n in ("peel", "peel-naive", "peel-single")}
        assert d["peel"].params["dominance"] and d["peel"].params["requeue"]
        assert not d["peel-naive"].params["dominance"] and d["peel-naive"].params["requeue"]
        assert not d["peel-single"].params["requeue"]
        conv = {n: sum(dec.decode(s)[1] for s in det) for n, dec in d.items()}
        assert conv["peel"] >= conv["peel-naive"] >= conv["peel-single"], conv

    def _one_strong_spec():
        """The catalogue, strong_spec and the decoder agree on 30 iterations."""
        assert strong_spec(0) == CATALOGUE["OSD-0"]
        assert strong_spec(4) == CATALOGUE["OSD-CS4"]
        assert strong_spec(0, "lsd") == CATALOGUE["LSD-0"]
        H, pr, _ = _setup()
        assert BpOsdDecoder(H, pr).dec.max_iter == make_decoder("OSD-0", H, pr).dec.max_iter \
            == STRONG_MAX_ITER == 30
        d = make_decoder("OSD-0", H, pr)
        assert d.params["ms_scaling_factor"] == 0.625 and d.params["schedule"] == "parallel"
        assert d.params == {k: v for k, v in decoder_spec_full("OSD-0").items()}, \
            (d.params, decoder_spec_full("OSD-0"))

    def _spec_full_is_json_safe():
        for name in list(WEAK) + [n for n in STRONG if n != "relay"] + list(CONTROLS):
            json.dumps(decoder_spec_full(name))
        assert decoder_spec_full("peel")["dominance"] is True
        assert decoder_spec_full("peel-naive")["dominance"] is False
        assert decoder_spec_full("BP-ps-30")["bp_method"] == "ps"
        assert "ms_scaling_factor" not in decoder_spec_full("BP-ps-30")

    def _priors_update_in_place():
        """ldpc exposes channel_probs and update_channel_probs (per-shot priors need them)."""
        H, pr, det = _setup()
        d = make_decoder("OSD-0", H, pr)
        new = np.clip(pr * 2, 1e-12, 0.49)
        d.update_priors(new)
        assert np.allclose(np.asarray(d.dec.channel_probs), new)

    def _relay_native():
        """The built-in Relay-BP needs no extra package; converged corrections are valid."""
        H, pr, det = _setup()
        d = make_decoder(dict(kind="relay", num_sets=10, stop_nconv=2), H, pr)
        assert d.backend == "native"
        for s in det[:20]:
            e, ok, work = d.decode(s)
            assert isinstance(work, int) and work > 0
            if ok:
                assert np.array_equal((H @ e) % 2, s), "Relay-BP correction misses the syndrome"
        d.update_priors(np.clip(pr * 2, 0, 0.4))
        d.decode(det[0])

    def _relay_matches_the_package():
        """Against IBM's relay-bp (when installed): the deterministic first leg gives
        the same correction shot by shot; the full relay gives a statistically
        equal failure count."""
        try:
            import relay_bp  # noqa: F401
        except ImportError:
            return
        c = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        circ = build_memory_circuit(c, 3, 4e-3, use_flags=False, x_detectors=False)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        det, obs = circ.compile_detector_sampler(seed=5).sample(300, separate_observables=True)
        det, obs = det.astype(np.uint8), obs.astype(np.uint8)
        a = RelayDecoder(H, pr, num_sets=0, stop_nconv=1)
        b = RelayDecoder(H, pr, num_sets=0, stop_nconv=1, backend="package")
        same = sum(np.array_equal(a.decode(s)[0], b.decode(s)[0]) for s in det)
        assert same >= 0.97 * len(det), f"first leg identical on only {same}/{len(det)} shots"
        fails = {}
        for be in ("native", "package"):
            d = RelayDecoder(H, pr, num_sets=60, stop_nconv=3, backend=be)
            fails[be] = sum(bool(np.any((L @ d.decode(s)[0]) % 2 != obs[i])) for i, s in enumerate(det))
        k1, k2 = fails["native"], fails["package"]
        assert abs(k1 - k2) <= 3 * math.sqrt(k1 + k2 + 1) + 2, fails

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
        ("compiled peeling matches the reference implementation", _fast_peel_matches_reference),
        ("peel, peel-naive and peel-single are wired as documented", _peel_variants_are_wired),
        ("one strong spec: 30 iterations, fixed settings, recorded params", _one_strong_spec),
        ("decoder_spec_full is complete and JSON-safe", _spec_full_is_json_safe),
        ("ldpc priors can be replaced in place", _priors_update_in_place),
        ("Relay-BP (built-in): valid corrections, cheap prior updates", _relay_native),
        ("built-in Relay-BP agrees with the relay-bp package (skipped if not installed)",
         _relay_matches_the_package),
    ])
    render_checks("decoders", tests_decoders)
    return (tests_decoders,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8. Switch policies

    | trigger | escalate to the secondary when |
    |---|---|
    | `never` | never (primary only) |
    | `always` | always (secondary only) |
    | `primary_fail` | the primary does not converge — the trivial baseline |
    | `flag>=1` | a flag fired (**before** the primary runs), else on primary failure — "flag or failure" |
    | `flag-only` | a flag fired; **never** on primary failure — isolates the flag signal |

    Older result files carry two triggers, `flag` and `flag_or_fail`, that were
    defined identically ("flag, else primary failure"); they are read as
    `flag>=1` with the duplicate dropped. The decoder grid (Experiment 4) adds
    `flag>=k` (k = 2..4) and `rounds>=k`.

    **Beating `always` is possible.** A policy can fail less than `always` when
    the primary is right on shots where the secondary errs. Over 250,000 shots on
    [[72,12,6]] at $T = 12$, peeling converged and was correct where BP+OSD-0 was
    wrong on **66** shots, and converged-but-wrong where BP+OSD-0 was right on
    **0** — so `primary_fail` keeps 66 answers that `always` would have thrown
    away and loses none. That is the whole of its 0.00512 against 0.00538 margin.
    This is a by-construction consequence of peeling's zero silent failures, an
    exploratory observation, not a confirmatory test.
    """)
    return


@app.class_definition
class SwitchPolicy:
    """
    Contract:
      - unknown trigger -> ValueError; flag triggers without a flag mask -> ValueError
      - `always` never calls the primary; `never` never calls the secondary
      - for flag triggers, a fired flag escalates WITHOUT calling the primary
      - `flag-only` never escalates on primary failure
      - the returned DecodeResult has escalated=True iff the secondary ran

    work: primary work plus secondary work when both run. If the two decoders
    count different things (detector visits vs BP iterations) the sum mixes
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
        if self.trigger not in ("never", "flag-only") and not r.converged:
            return self._escalate(syndrome, r.work)
        return DecodeResult(r.correction, r.converged, r.work,
                            escalated=False, soft=r.soft)


@app.cell
def _():
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
        for trig in FLAG_TRIGGERS:
            r, pc, sc = _run(trig, True, _flagged)
            assert (pc, sc, r.escalated) == (0, 1, True), f"{trig}: primary ran on a flagged shot"

    def _flag_fallback():
        # flag>=1 falls back to primary_fail on unflagged shots
        r, pc, sc = _run("flag>=1", True, _quiet)
        assert (pc, sc, r.escalated) == (1, 0, False)
        r, pc, sc = _run("flag>=1", False, _quiet)
        assert (pc, sc, r.escalated) == (1, 1, True)

    def _flag_only_ignores_failure():
        """flag-only must NOT escalate an unflagged, unconverged shot."""
        r, pc, sc = _run("flag-only", False, _quiet)
        assert (pc, sc, r.escalated) == (1, 0, False), "flag-only escalated on primary failure"
        r, pc, sc = _run("flag-only", False, _flagged)
        assert (pc, sc, r.escalated) == (0, 1, True)

    def _triggers_are_distinct():
        """No two triggers are the same rule."""
        outcomes = {}
        for trig in TRIGGERS:
            outcomes[trig] = tuple(_run(trig, ok, s)[0].escalated
                                   for ok in (True, False) for s in (_quiet, _flagged))
        assert len(set(outcomes.values())) == len(TRIGGERS), outcomes

    def _validation():
        for bad in (dict(trigger="sometimes", flag_mask=_mask),
                    dict(trigger="flag", flag_mask=_mask),           # the retired name
                    dict(trigger="flag>=1", flag_mask=None),
                    dict(trigger="flag-only", flag_mask=np.zeros(4, bool))):
            try:
                SwitchPolicy(_Mock(True), _Mock(True), **bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted invalid config {bad}")

    def _beating_always_is_representable():
        """A policy CAN fail less than `always`: one shot, primary right and
        secondary wrong. primary_fail keeps the primary's answer."""
        n = 4
        weak = dict(conv=np.ones(n, bool), fail=np.zeros(n, bool),
                    work=np.ones(n, np.int64), time_us=np.ones(n))
        strong = dict(conv=np.ones(n, bool), fail=np.zeros(n, bool),
                      work=np.ones(n, np.int64), time_us=np.ones(n))
        strong["fail"][2] = True
        none = np.zeros(n, bool)
        pf = derive(weak, strong, none, True)
        always = derive(weak, strong, np.ones(n, bool), False)
        assert int(pf["fail"].sum()) == 0
        assert int(always["fail"].sum()) == 1
        assert pf["fail"].sum() < always["fail"].sum()

    def _escalating_a_correct_primary_can_harm():
        """A pre-decode trigger that fires on that shot converts a success into a
        failure (the `harm` column of Experiment 4)."""
        n = 4
        weak = dict(conv=np.ones(n, bool), fail=np.zeros(n, bool),
                    work=np.ones(n, np.int64), time_us=np.ones(n))
        strong = dict(conv=np.ones(n, bool), fail=np.zeros(n, bool),
                      work=np.ones(n, np.int64), time_us=np.ones(n))
        strong["fail"][2] = True
        pre = np.zeros(n, bool)
        pre[2] = True
        flagged = derive(weak, strong, pre, True)
        assert int(flagged["fail"].sum()) == 1
        assert flagged["cost"][2] == 1.0, "a pre-escalated shot must not pay the primary"

    tests_switch = run_checks([
        ("never: secondary is never called", _never),
        ("always: primary is never called", _always),
        ("primary_fail: escalates only on primary failure, ignores flags", _primary_fail),
        ("flag triggers: a fired flag skips the primary pass", _flag_skips_primary),
        ("flag>=1: unflagged shots fall back to primary_fail", _flag_fallback),
        ("flag-only: never escalates on primary failure", _flag_only_ignores_failure),
        ("every trigger is a distinct rule", _triggers_are_distinct),
        ("invalid trigger (incl. the retired `flag`) or missing mask raises", _validation),
        ("a policy beating `always` is representable, not a bug",
         _beating_always_is_representable),
        ("escalating a correct primary is recorded as harm, and skips its cost",
         _escalating_a_correct_primary_can_harm),
    ])
    render_checks("switch policies", tests_switch)
    return (tests_switch,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9. Statistics and benchmarking

    - **Logical error** = predicted observable flips $L\hat e$ differ from the
      true flips sampled by Stim, on *any* of the $k$ observables.
    - **Confidence intervals:** Wilson score interval, 95% two-sided. At zero
      failures its upper end, $z^2/(n+z^2)$, is a **97.5% one-sided** bound;
      `cp_upper` gives the exact Clopper–Pearson one-sided bound used by the
      silent-failure census.
    - **Paired comparison** (same circuit, same samples: policies; blind, sighted
      and updated arms): exact McNemar sign test on the discordant shots.
    - **Unpaired comparison** (different circuits, e.g. unflagged vs sighted):
      `rate_ratio_ci`, the exact conditional (Clopper-Pearson) interval for the
      ratio of two Poisson/binomial rates and its binomial test. Equal seeds do
      not pair different circuits; the result files record a hash of every
      sample set so pairing is proven, not assumed.
    - **Minimum detectable effect** (`mde_ratio`, `min_detectable_discordant`) is
      computed and stored before a run, so an inconclusive result is labelled as
      such rather than read as a null.
    - **Multiplicity**: confirmatory claims are a pre-registered Holm family
      (§17); exploratory comparisons are Bonferroni-corrected within their
      family (112 comparisons in the decoder grid).
    - **Trend tests**: weighted least squares (`wls_slope`) with inverse-variance
      weights and a one-sided $t$ test; per-code intercepts when several codes
      share one slope.
    - **Timing comparisons**: per-shot times of two arms decoded on the same
      shots are compared with the Wilcoxon signed-rank test; median-time ratios
      get a paired bootstrap interval. Cost savings of switching rules get a
      chunk-block standard error (250-shot blocks), so correlated timings within a
      chunk do not shrink it.
    - **Thresholds**: curve crossings interpolated in log–log coordinates, with a
      parametric binomial bootstrap (2,000 resamples) for the interval.

    **Which experiment uses what.**

    | experiment | estimate | comparison / test | multiplicity |
    |---|---|---|---|
    | 1 noise model, peeling | peeling success with Wilson CI; fault count, λ and $\bar d$ against Theorems 1–5 | decision rule: is a target inside the Wilson interval | — |
    | 2 syndrome augmentation | LER per arm with Wilson CI | sighted/blind vs unflagged: exact conditional rate ratio (unpaired); blind/sighted/updated: exact McNemar (paired); shards summed | Holm T4, T7–T9 |
    | 3 flag rate | fraction of shots with a flag, Wilson CI | law L6 prediction; Pearson χ² (descriptive) | — |
    | 4 decoder grid | LER, escalation, mean/median/p99 cost per policy | McNemar vs `primary_fail` and vs a placebo trigger; cost saving with a chunk-block interval; risk ratio and mutual information of flags vs failures | Bonferroni, m = 112 |
    | 5 census | silent-failure rate | one-sided 95% Clopper–Pearson bound | — |
    | 6 surface code | LER per (d, p), Wilson CI | MWPM vs BP+OSD-0: McNemar per point; threshold crossing with bootstrap CI | — (validation rule) |
    | 7 frontier | LER and median time per arm/decoder | Pareto dominance within one worker count | — |
    | 8 prior updating | LER, median/p99 time (update included), BP iterations | updated vs sighted: McNemar; updated vs sighted time: Wilcoxon signed-rank; gain vs $\ln p$: WLS | Holm T1–T3, T5, T6, T11, T14 |
    | 9 w-family | LER per arm per code | ln(blind/updated) vs w: WLS, one-sided t | Holm T10 |
    | 10 BP+LSD | as 8 | McNemar | Holm T12 |
    | 11 Relay-BP | as 8 | McNemar | Holm T13 |

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


@app.class_definition
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
    # paired comparisons on the same shots, {worse, better, p} (ablation runs)
    vs_primary_fail: dict = field(default=None, compare=False, repr=False)
    vs_always: dict = field(default=None, compare=False, repr=False)

    @property
    def escalated_failures(self):
        return self.failures - self.silent_failures - self.unconverged_failures


@app.function
def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials -> (p_hat, low, high).
    95% two-sided; at k = 0 the upper end is a 97.5% one-sided bound."""
    if n <= 0:
        return 0.0, 0.0, 1.0
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    p_hat = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / denom
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return p_hat, low, high


@app.function
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


@app.function
def paired_exact(fail_a, fail_b):
    """
    Exact two-sided sign test on discordant shots (McNemar), for two arms that
    decoded the SAME shots. x = shots only A fails, y = shots only B fails.
    """
    fail_a = np.asarray(fail_a, dtype=bool)
    fail_b = np.asarray(fail_b, dtype=bool)
    if fail_a.shape != fail_b.shape:
        raise ValueError("paired_exact needs arms decoded on the same shots")
    x = int((fail_a & ~fail_b).sum())
    y = int((~fail_a & fail_b).sum())
    n = x + y
    if n == 0:
        return x, y, 1.0
    tail = sum(comb(n, i) for i in range(min(x, y) + 1))
    return x, y, min(1.0, 2 * tail / 2 ** n)


@app.function
def min_detectable_discordant(alpha=0.05):
    """Fewest one-sided discordant shots the paired exact test can call
    significant: with x = 0, p = 2 / 2**y, so y must satisfy 2/2**y < alpha."""
    y = 1
    while 2 / 2 ** y >= alpha:
        y += 1
    return y


@app.function
def rate_ratio_ci(k1, n1, k0, n0, alpha=0.05):
    """
    UNPAIRED comparison of two failure rates from independent samples (e.g.
    sighted vs unflagged: different circuits). Exact conditional method:
    given m = k1 + k0, k1 ~ Binomial(m, pi) with pi = n1 r / (n1 r + n0), where
    r = (k1/n1)/(k0/n0). A Clopper-Pearson interval for pi maps to an interval
    for r; the binomial test of pi = n1/(n1+n0) gives the p-value.
    Handles k1 = 0 (ratio 0, finite upper end) and k0 = 0 (ratio inf).
    """
    k1, n1, k0, n0 = int(k1), int(n1), int(k0), int(n0)
    m = k1 + k0
    out = dict(k1=k1, n1=n1, k0=k0, n0=n0, paired=False, alpha=alpha,
               method="exact conditional (Clopper-Pearson on k1 | k1+k0)")
    if m == 0:
        out.update(ratio=float("nan"), lo=0.0, hi=float("inf"), p=1.0)
        return out
    lo_pi = 0.0 if k1 == 0 else float(st.beta.ppf(alpha / 2, k1, m - k1 + 1))
    hi_pi = 1.0 if k1 == m else float(st.beta.ppf(1 - alpha / 2, k1 + 1, m - k1))

    def to_r(pi):
        if pi >= 1.0:
            return float("inf")
        return pi * n0 / ((1 - pi) * n1)

    ratio = float("inf") if k0 == 0 else (k1 / n1) / (k0 / n0)
    p = float(st.binomtest(k1, m, n1 / (n1 + n0)).pvalue)
    out.update(ratio=ratio, lo=to_r(lo_pi), hi=to_r(hi_pi), p=p)
    return out


@app.function
def mde_ratio(mu0, alpha=0.05, power=0.8, iterations=50):
    """
    Smallest rate ratio R = mu0/mu1 an unpaired comparison can detect, given
    mu0 expected failures in the reference arm: R = exp((z_{1-a/2} + z_power)
    sqrt(1/mu0 + 1/mu1)) with mu1 = mu0/R, solved by fixed-point iteration.
    """
    if mu0 <= 0:
        return float("inf")
    z = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
    R = 1.0
    for _ in range(iterations):
        e = z * math.sqrt(1.0 / mu0 + R / mu0)
        if e > 50:                     # no finite ratio is detectable with so few failures
            return float("inf")
        R = math.exp(e)
    return float(R)


@app.function
def cp_upper(k, n, conf=0.95):
    """Exact one-sided Clopper-Pearson upper bound for a rate (k = 0 -> ~3/n)."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(st.beta.ppf(conf, k + 1, n - k))


@app.function
def holm(pvalues, alpha=0.05):
    """
    Holm step-down over a family. Returns a list of dicts (index, p, threshold,
    reject, p_adjusted) in the original order. Reject H(i) while
    p(i) <= alpha/(m - i + 1), stopping at the first failure.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    reject = np.zeros(m, bool)
    adj = np.zeros(m)
    running = 0.0
    stopped = False
    for rank, i in enumerate(order):
        thr = alpha / (m - rank)
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
        if not stopped and p[i] <= thr:
            reject[i] = True
        else:
            stopped = True
    return [dict(index=int(i), p=float(p[i]), threshold=alpha / (m - int(np.where(order == i)[0][0])),
                 reject=bool(reject[i]), p_adjusted=float(adj[i])) for i in range(m)]


@app.function
def wls_slope(x, y, var, alternative="less", groups=None):
    """
    Weighted least squares slope of y on x with weights 1/var, optionally with a
    fixed effect (intercept) per group. Returns slope, se, t, df and the
    one-sided p-value for H1: slope < 0 ("less") or > 0 ("greater").
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = 1.0 / np.asarray(var, float)
    if groups is None:
        X = np.column_stack([np.ones_like(x), x])
    else:
        labels = sorted(set(groups))
        X = np.column_stack([x] + [np.array([g == lab for g in groups], float) for lab in labels])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    beta = np.linalg.solve(XtWX, X.T @ W @ y)
    resid = y - X @ beta
    df = len(x) - X.shape[1]
    sigma2 = float(resid @ W @ resid / df) if df > 0 else float("nan")
    cov = np.linalg.inv(XtWX) * (max(sigma2, 1.0) if df > 0 else 1.0)
    k = 1 if groups is None else 0
    slope, se = float(beta[k]), float(np.sqrt(cov[k, k]))
    t = slope / se if se > 0 else float("nan")
    if alternative == "less":
        p = float(st.t.cdf(t, df)) if df > 0 else float("nan")
    else:
        p = float(st.t.sf(t, df)) if df > 0 else float("nan")
    return dict(slope=slope, se=se, t=t, df=int(df), p=p, alternative=alternative)


@app.cell
def _():
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
        for i, name in enumerate(["never", "always", "primary_fail", "flag>=1"]):
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

    def _rate_ratio_exact():
        """The exact conditional interval on realistic counts."""
        r = rate_ratio_ci(17, 300000, 154, 300000)
        assert 0.06 < r["lo"] < r["ratio"] < r["hi"] < 0.19 and r["p"] < 1e-20, r
        z = rate_ratio_ci(0, 20000, 4, 20000)
        assert z["ratio"] == 0 and 1.4 < z["hi"] < 1.6 and not z["paired"], z
        inv = rate_ratio_ci(154, 300000, 17, 300000)
        assert abs(inv["ratio"] - 154 / 17) < 1e-9 and 5.4 < inv["lo"] < 5.6 and 15.5 < inv["hi"] < 16.5

    def _rate_ratio_coverage():
        rng = np.random.default_rng(7)
        n, hits = 2000, 0
        for _ in range(n):
            k1 = rng.binomial(20000, 0.004)
            k0 = rng.binomial(20000, 0.002)
            r = rate_ratio_ci(k1, 20000, k0, 20000)
            hits += r["lo"] <= 2.0 <= r["hi"]
        assert hits / n >= 0.94, hits / n

    def _mde_and_discordant():
        assert mde_ratio(400) < mde_ratio(40)
        assert abs(mde_ratio(400) - 1.22) < 0.05, mde_ratio(400)
        assert min_detectable_discordant(0.05) == 6
        assert min_detectable_discordant(0.05 / 14) == 10       # Holm's strictest step
        assert min_detectable_discordant(0.05 / 112) == 13

    def _paired_exact_values():
        a = np.zeros(20, bool)
        b = a.copy()
        b[:9] = True
        assert paired_exact(a, b) == (0, 9, 2 * 0.5 ** 9)
        b[9] = True
        assert abs(paired_exact(a, b)[2] - 1.953125e-3) < 1e-12

    def _holm_known():
        res = holm([0.001, 0.04, 0.012, 0.3], alpha=0.05)
        assert [r["reject"] for r in res] == [True, False, True, False], res
        assert all(r["reject"] for r in holm([0.01] * 5))

    def _wls_recovers_slope():
        x = np.arange(4, 13, dtype=float)
        y = 2.0 - 0.15 * x
        r = wls_slope(x, y + np.random.default_rng(1).normal(0, 0.01, x.size), np.full(x.size, 1e-4))
        assert abs(r["slope"] + 0.15) < 0.01 and r["p"] < 0.01 and r["df"] == 7

    def _cp_upper_zero():
        assert abs(cp_upper(0, 250000) - 2.9957 / 250000) < 1e-8
        assert abs(wilson(0, 250000)[2] - 1.5366e-5) < 1e-8

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
        ("unpaired exact rate-ratio interval on known counts", _rate_ratio_exact),
        ("rate-ratio interval covers the true ratio >= 94% of the time", _rate_ratio_coverage),
        ("minimum detectable effects (unpaired ratio, paired discordants)", _mde_and_discordant),
        ("exact McNemar values: 9 one-way -> 3.9e-3, 10 -> 1.95e-3", _paired_exact_values),
        ("Holm step-down rejects the expected hypotheses", _holm_known),
        ("weighted least squares recovers a known slope", _wls_recovers_slope),
        ("Clopper-Pearson zero-count bound; Wilson's is 97.5% one-sided", _cp_upper_zero),
    ])
    render_checks("statistics", tests_stats)
    return (tests_stats,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 10. Parallel decoding and checkpoints

    Every long computation is a loop over **chunks** of 250 shots. `run_zoo`
    decodes every chunk with every requested decoder, in worker processes when
    the run is large; `run_resumable` does the same for loops that cannot use the
    pool (per-shot priors). Worker processes load *this file by its path* and
    call `decode_chunk`, so everything on that path is a top-level function that
    depends only on the setup cell; the file name and the current directory do
    not matter.

    **Decoder cache.** Each process keeps the decoders of the current run, keyed
    on a fingerprint of $(H, \text{priors}, \text{decoder specs})$. A key on the
    decoder name and matrix shape alone would let a process that decodes the same
    code at a second noise level silently reuse the first level's priors (a
    self-test below measures exactly that case). Every chunk also runs one
    **untimed warm-up decode** per decoder, so numba compilation and decoder
    construction never land in a timed shot (without it, chunk-start shots
    average thousands of µs against ~24 µs for the rest).

    **Checkpoints.** Finished chunks are written to `results/checkpoints/` at
    most `CKPT["minutes"]` apart (set in the panel below; 0 = every chunk),
    crash-safely (temporary file, fsync, atomic rename). Each checkpoint is
    tagged with a fingerprint of everything its results depend on — matrices,
    priors, samples, decoder specs, chunk size, and the version of the chunk
    decoder — so a resumed run can only ever load chunks of the identical
    computation. Folders written in any other checkpoint format are ignored. A
    checkpoint folder is never a result: it carries a `role` field and a
    README, and the result loaders refuse checkpoint paths.

    **Samples.** `sample_hash` fingerprints every sampled `(det, obs)` set and
    `stim_selfcheck` fingerprints the sampler itself, so a later run can prove
    two arms decoded the same shots — or that this machine's Stim no longer
    reproduces a seed, in which case `verify_result` re-decodes stored samples.
    """)
    return


@app.function
def hash_feed(h, obj):
    """Feed any mix of arrays, sparse matrices, containers and scalars into a
    hash, by value (arrays by dtype, shape and bytes)."""
    if sp.issparse(obj):
        m = sp.csr_matrix(obj)
        m.sort_indices()
        h.update(b"S" + repr((m.shape, str(m.dtype))).encode())
        for a in (m.indptr, m.indices, m.data):
            h.update(np.ascontiguousarray(a).tobytes())
    elif isinstance(obj, np.ndarray):
        h.update(b"A" + repr((obj.shape, str(obj.dtype))).encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, dict):
        h.update(b"D%d" % len(obj))
        for k in sorted(obj, key=repr):
            hash_feed(h, k)
            hash_feed(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        h.update(b"L%d" % len(obj))
        for x in obj:
            hash_feed(h, x)
    elif isinstance(obj, np.generic):
        hash_feed(h, obj.item())
    else:
        h.update(b"V" + repr(obj).encode())


@app.function
def fingerprint(*objs):
    """Content hash of everything that determines a computation's result."""
    h = hashlib.blake2b(digest_size=20)
    h.update(b"FlaggedDecoding-checkpoint-v%d" % CHECKPOINT_FORMAT)
    for o in objs:
        hash_feed(h, o)
    return h.hexdigest()


@app.function
def sample_hash(det, obs):
    """sha256 of a sampled shot set: shapes plus bit-packed detectors and
    observables. Independent of dtype (bool and uint8 hash equal)."""
    det = np.asarray(det).astype(np.uint8)
    obs = np.asarray(obs).astype(np.uint8)
    h = hashlib.sha256()
    h.update(repr(("det", det.shape, "obs", obs.shape)).encode())
    h.update(np.packbits(det, axis=-1).tobytes())
    h.update(np.packbits(obs, axis=-1).tobytes())
    return h.hexdigest()


@app.function
def stim_selfcheck():
    """A fingerprint of this machine's Stim sampler: a fixed repetition-code
    sample (d=5, r=5, p=0.05, seed 1, 4,096 shots). Equal digests mean seeded
    samples reproduce bit for bit between two machines."""
    circ = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5,
                                  after_clifford_depolarization=0.05)
    det, obs = circ.compile_detector_sampler(seed=1).sample(4096, separate_observables=True)
    return dict(digest=sample_hash(det, obs), stim=stim.__version__,
                machine=platform.machine(), processor=platform.processor())


@app.function
def circuit_identity(circuit, H, priors):
    """What a result file records about the circuit it sampled."""
    used = {t.value for inst in circuit.flattened() for t in inst.targets_copy()
            if t.is_qubit_target}
    Hc = sp.csr_matrix(H)
    hd = hashlib.sha256()
    for a in (Hc.indptr, Hc.indices, np.asarray(priors, dtype=np.float64)):
        hd.update(np.ascontiguousarray(a).tobytes())
    return dict(circuit_sha256=hashlib.sha256(str(circuit).encode()).hexdigest(),
                num_qubits=int(circuit.num_qubits), qubits_used=len(used),
                num_detectors=int(circuit.num_detectors),
                num_observables=int(circuit.num_observables),
                faults=int(H.shape[1]), lam=float(np.sum(priors)),
                dem_sha256=hd.hexdigest())


@app.function
def cached_decoder(run_key, name, spec, H, priors):
    """
    The decoder `name` of the run `run_key`, built once per process. The key
    is a fingerprint of (H, priors, decoder specs), so a process never reuses
    a decoder built for other priors. Returns (decoder, build_us).
    """
    key = (run_key, name)
    dec = DECODER_CACHE.get(key)
    if dec is not None:
        return dec, 0.0
    if any(k[0] != run_key for k in DECODER_CACHE) or not CACHE_ALL_DECODERS:
        DECODER_CACHE.clear()                    # a new run: free the old decoders
        gc_collect()
    t0 = time.perf_counter()
    dec = make_decoder(spec, H, priors)
    DECODER_CACHE[key] = dec
    return dec, (time.perf_counter() - t0) * 1e6


@app.function
def gc_collect():
    import gc
    gc.collect()


@app.function
def decode_chunk(task):
    """
    Worker: decode one chunk of shots with every requested decoder.

    task = (H, L, priors, det, obs, pairs, run_key). For each decoder one
    UNTIMED warm-up decode runs first (numba compilation, lazy allocation),
    then every shot is timed. Returns {name: {conv, fail, work, time_us,
    build_us}}; build_us is 0 except at index 0 when the decoder was built.
    """
    H, L, priors, det, obs, pairs, run_key = task
    L = sp.csr_matrix(L)
    out = {}
    for name, spec in pairs:
        dec, build_us = cached_decoder(run_key, name, spec, H, priors)
        n = len(det)
        conv = np.zeros(n, bool)
        fail = np.zeros(n, bool)
        work = np.zeros(n, np.int32)
        t_us = np.zeros(n, np.float32)
        b_us = np.zeros(n, np.float32)
        if n:
            dec.decode(det[0])                       # untimed warm-up
            b_us[0] = build_us
        for i in range(n):
            t0 = time.perf_counter()
            e, c, w = dec.decode(det[i])
            t_us[i] = (time.perf_counter() - t0) * 1e6
            conv[i], work[i] = c, w
            fail[i] = bool(np.any((L @ e) % 2 != obs[i]))
        out[name] = dict(conv=conv, fail=fail, work=work, time_us=t_us, build_us=b_us)
    return out


@app.function
def configure_checkpoints(root=None, minutes=None, min_shots=None):
    """Set where checkpoints go, how often they are written, and the auto threshold."""
    if root is not None:
        CKPT["root"] = os.path.abspath(str(root))
    if minutes is not None:
        if float(minutes) < 0:
            raise ValueError("minutes must be >= 0")
        CKPT["minutes"] = float(minutes)
    if min_shots is not None:
        CKPT["min_shots"] = int(min_shots)
    return dict(CKPT)


@app.function
def fsync_dir(path):
    if os.name == "nt":                  # Windows cannot open a directory for fsync
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


@app.function
def durable_write(path, write):
    """write(fileobj) into a temporary file, force it to disk, then rename into
    place. A file cut short by a power failure never replaces a good one."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(10):            # Windows: antivirus/indexer can hold the file
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)
    fsync_dir(os.path.dirname(path) or ".")


@app.function
def read_manifest(directory):
    try:
        with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@app.class_definition
class ChunkStore:
    """
    One run's finished chunks on disk. `fp` is the run's fingerprint; a chunk
    file written for any other fingerprint (or without one) is ignored and
    overwritten when that chunk is recomputed.
    """

    README = ("This folder is a CHECKPOINT: partial decoder outputs of one run, "
              "kept so the run can resume after an interruption. It is not a result "
              "file; delete it once the run's result file is written.\n")

    def __init__(self, directory, fp, n_chunks, chunk_size, shots, label="", names=(),
                 minutes=None, result_file=None):
        self.dir = os.path.abspath(directory)
        os.makedirs(self.dir, exist_ok=True)
        self.fp = fp
        self.minutes = CKPT["minutes"] if minutes is None else float(minutes)
        self.pending = {}
        self.last_flush = time.monotonic()
        self.written = 0
        self.rejected = 0
        for f in os.listdir(self.dir):          # half-written files from a power cut
            if f.endswith(".tmp"):
                try:
                    os.remove(os.path.join(self.dir, f))
                except OSError:
                    pass
        readme = os.path.join(self.dir, "README.txt")
        if not os.path.exists(readme):
            durable_write(readme, lambda f: f.write(self.README.encode()))
        self.manifest = dict(format=CHECKPOINT_FORMAT, fingerprint=fp, label=str(label),
                             role="checkpoint (partial decoder outputs; not a result file)",
                             result_file=result_file,
                             names=[str(n) for n in names], shots=int(shots),
                             chunk_size=int(chunk_size), n_chunks=int(n_chunks),
                             created=time.strftime("%Y-%m-%d %H:%M:%S"),
                             updated=None, complete=False)
        old = read_manifest(self.dir)
        if old and old.get("fingerprint") == fp:
            self.manifest["created"] = old.get("created", self.manifest["created"])
        self.write_manifest()

    def path(self, start):
        return os.path.join(self.dir, f"chunk_{int(start):09d}.npz")

    def load(self, start):
        """The stored part for the chunk starting at `start`, or None."""
        p = self.path(start)
        if not os.path.exists(p):
            return None
        try:
            with np.load(p, allow_pickle=False) as d:
                if "fingerprint" not in d.files or str(d["fingerprint"]) != self.fp:
                    return None
                part = {}
                for key in d.files:
                    if "__" in key:
                        n, k = key.split("__", 1)
                        part.setdefault(n, {})[k] = np.array(d[key])
                return part
        except Exception:                       # damaged file: recompute this chunk
            self.rejected += 1
            return None

    def put(self, start, part):
        self.pending[int(start)] = part
        if self.minutes <= 0 or time.monotonic() - self.last_flush >= 60.0 * self.minutes:
            self.flush()

    def flush(self):
        """Write every finished-but-unsaved chunk to disk now."""
        for start in sorted(self.pending):
            part = self.pending[start]
            arrays = {f"{n}__{k}": np.asarray(v) for n, r in part.items() for k, v in r.items()}
            durable_write(self.path(start), lambda f, a=arrays: np.savez(
                f, fingerprint=np.array(self.fp), names=np.array(self.manifest["names"]), **a))
            self.written += 1
        had = bool(self.pending)
        self.pending.clear()
        self.last_flush = time.monotonic()
        if had:
            self.write_manifest()

    def mark_complete(self):
        self.flush()
        self.manifest["complete"] = True
        self.write_manifest()

    def done(self):
        return len([f for f in os.listdir(self.dir)
                    if f.startswith("chunk_") and f.endswith(".npz")])

    def write_manifest(self):
        self.manifest["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.manifest["done"] = self.done()
        text = json.dumps(self.manifest, indent=1).encode()
        durable_write(os.path.join(self.dir, "manifest.json"), lambda f: f.write(text))


@app.function
def open_store(kind, n, chunk_size, key, checkpoint_dir, label, names, result_file=None):
    """
    Resolve where (and whether) a run is checkpointed. checkpoint_dir:
      None  -> automatic, under CKPT["root"], when n >= CKPT["min_shots"]
      False -> never
      path  -> that folder
    Returns (store or None, chunk_size). Automatic runs always use CHUNK_SIZE,
    so the chunking (and hence the fingerprint) does not depend on the worker
    count. The label is NOT part of the fingerprint, so renaming a label never
    orphans an unfinished checkpoint.
    """
    if checkpoint_dir is False or n == 0:
        return None, chunk_size
    if checkpoint_dir is None:
        if not CKPT["root"] or n < CKPT["min_shots"]:
            return None, chunk_size
        chunk_size = CHUNK_SIZE
    fp = fingerprint(kind, key, int(chunk_size), int(n), DECODE_CHUNK_VERSION)
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(CKPT["root"], f"auto_{fp[:16]}")
    n_chunks = -(-n // chunk_size)
    return ChunkStore(checkpoint_dir, fp, n_chunks, chunk_size, n, label=label or kind,
                      names=names, result_file=result_file), chunk_size


@app.function
def run_zoo(H, L, priors, det, obs, names, workers=1, chunk_size=CHUNK_SIZE, on_chunk=None,
            checkpoint_dir=None, label=None, result_file=None, return_info=False):
    """
    Decode all shots with every decoder in `names`.

    workers > 1 uses a process pool (spawn, so it behaves the same on Windows,
    macOS and Linux); each worker imports this file and calls decode_chunk. If
    the pool cannot start, falls back to fewer workers and then to sequential.

    Checkpointing: checkpoint_dir=None checkpoints a run of at least
    CKPT["min_shots"] shots automatically; pass a folder to choose it, or False
    to switch it off. Returns (results, backend), or (results, backend, info)
    with return_info=True: info = {dir, fingerprint, resumed_chunks,
    total_chunks, chunk_size, workers_requested, workers_effective}.
    results[name][field] -> per-shot array (conv, fail, work, time_us, build_us).
    """
    det = np.asarray(det, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    pairs = normalise_names(names)
    names = [n for n, _ in pairs]
    run_key = fingerprint("decoders", H, priors, pairs)
    store, chunk_size = open_store(
        "zoo", len(det), chunk_size, (H, L, priors, det, obs, pairs), checkpoint_dir,
        label or f"run_zoo: {', '.join(names)} on {len(det):,} shots, H {H.shape[0]}x{H.shape[1]}",
        names, result_file=result_file)
    starts = list(range(0, len(det), chunk_size))
    parts = [None] * len(starts)
    if store:
        for i, a in enumerate(starts):
            parts[i] = store.load(a)
    resumed = sum(p is not None for p in parts)
    for _ in range(resumed):
        if on_chunk:
            on_chunk()

    def task(i):
        a = starts[i]
        return (H, L, priors, det[a:a + chunk_size], obs[a:a + chunk_size], pairs, run_key)

    def finished(i, part):
        parts[i] = part
        if store:
            store.put(starts[i], part)
        if on_chunk:
            on_chunk()

    backend = "sequential"
    effective = 1
    note = ""
    tried = workers
    expr = WORKER_LOADER + ".decode_chunk(task)"
    try:
        while tried > 1 and [i for i, p in enumerate(parts) if p is None]:
            todo = [i for i, p in enumerate(parts) if p is None]
            try:
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=tried, mp_context=ctx) as pool:
                    futures = {pool.submit(eval, expr, {"task": task(i), "N": MODULE_NAME,
                                                          "P": NOTEBOOK_FILE}): i
                               for i in todo}
                    for f in as_completed(futures):
                        finished(futures[f], f.result())
                backend = f"parallel ({tried} processes){note}"
                effective = tried
                break
            except Exception as exc:                 # noqa: BLE001
                note = f"; retried after {type(exc).__name__} at {tried} workers"
                tried //= 2
                gc_collect()
                if tried <= 1:
                    backend = (f"sequential (process pool failed: {type(exc).__name__}: {exc}; "
                               "try fewer workers, or fewer decoders per run)")
        for i in [i for i, p in enumerate(parts) if p is None]:
            finished(i, decode_chunk(task(i)))
    finally:
        if store:
            store.flush()          # an interrupt or an error must not lose finished chunks
    if store:
        store.mark_complete()
        backend += f"; checkpoint {store.dir}"
    if resumed:
        backend += f"; resumed {resumed}/{len(starts)} chunks from checkpoint"
    results = {n: {k: np.concatenate([p[n][k] for p in parts]) for k in parts[0][n]}
               for n in names}
    if not return_info:
        return results, backend
    info = dict(dir=store.dir if store else None, fingerprint=store.fp if store else None,
                resumed_chunks=resumed, total_chunks=len(starts), chunk_size=chunk_size,
                workers_requested=int(workers), workers_effective=int(effective))
    return results, backend, info


@app.function
def run_resumable(n, compute, key, chunk_size=CHUNK_SIZE, checkpoint_dir=None, label=None,
                  on_chunk=None, result_file=None):
    """
    A checkpointed loop for work that cannot go through run_zoo -- decoding with
    priors that change every shot (the updated arm).

    compute(a, b) -> dict of 1-D arrays, each of length b - a, for shots a..b-1.
    key: everything the results depend on; it is hashed into the fingerprint.
    on_chunk(a, b, resumed) is called after every chunk, resumed or computed.
    Returns (arrays, info) with info = {resumed, chunks, checkpoint_dir,
    fingerprint}.
    """
    n = int(n)
    store, chunk_size = open_store("loop", n, chunk_size, key, checkpoint_dir,
                                   label or f"resumable loop over {n:,} shots", ["run"],
                                   result_file=result_file)
    starts = list(range(0, n, chunk_size))
    parts, resumed = [], 0
    try:
        for a in starts:
            b = min(a + chunk_size, n)
            part = store.load(a) if store else None
            was_resumed = part is not None
            if was_resumed:
                part = part["run"]
                resumed += 1
            else:
                part = {k: np.asarray(v) for k, v in compute(a, b).items()}
                if store:
                    store.put(a, {"run": part})
            for k, v in part.items():
                if len(v) != b - a:
                    raise ValueError(f"compute({a}, {b}) returned {k} of length {len(v)}")
            parts.append(part)
            if on_chunk is not None:
                on_chunk(a, b, was_resumed)
    finally:
        if store:
            store.flush()
    if store:
        store.mark_complete()
    out = ({k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {})
    return out, dict(resumed=resumed, chunks=len(starts),
                     checkpoint_dir=store.dir if store else None,
                     fingerprint=store.fp if store else None)


@app.function
def list_checkpoints(root=None):
    """Every checkpoint folder under `root` (default CKPT["root"]), newest first."""
    root = os.path.abspath(root or CKPT["root"])
    rows = []
    if not os.path.isdir(root):
        return rows
    for dirpath, _dirs, files in os.walk(root):
        chunks = [f for f in files if f.startswith("chunk_") and f.endswith(".npz")]
        if "manifest.json" not in files and not chunks:
            continue
        m = read_manifest(dirpath) or {}
        size = 0
        for f in files:
            try:
                size += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
        total = m.get("n_chunks")
        rows.append(dict(folder=os.path.relpath(dirpath, root), path=dirpath,
                         label=m.get("label", "(written before manifests existed)"),
                         format=m.get("format"), done=len(chunks), total=total,
                         percent=(100.0 * len(chunks) / total) if total else None,
                         complete=bool(m.get("complete", False)),
                         usable=m.get("format") == CHECKPOINT_FORMAT,
                         updated=m.get("updated") or time.strftime(
                             "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(dirpath))),
                         megabytes=size / 1e6))
    rows.sort(key=lambda r: r["updated"], reverse=True)
    return rows


@app.function
def clear_checkpoints(root=None, only_complete=True):
    """
    Delete checkpoint folders: by default only those whose run finished (their
    results live in the experiment's own result file). only_complete=False also
    discards unfinished runs. Returns the list of folders removed.
    """
    root = os.path.abspath(root or CKPT["root"])
    removed = []
    for r in list_checkpoints(root):
        if r["complete"] or not only_complete:
            shutil.rmtree(r["path"], ignore_errors=True)
            removed.append(r["folder"])
            parent = os.path.dirname(r["path"])
            while parent.startswith(root + os.sep) and os.path.isdir(parent) \
                    and not os.listdir(parent):
                os.rmdir(parent)
                parent = os.path.dirname(parent)
    return removed


@app.function
def is_checkpoint_path(path):
    """True for anything inside a checkpoint folder: never load it as a result."""
    p = os.path.abspath(str(path))
    parts = p.replace("\\", "/").split("/")
    base = os.path.basename(p)
    return ("checkpoints" in parts or base == "manifest.json"
            or (base.startswith("chunk_") and base.endswith(".npz")))


@app.function
def usable_checkpoints(root=None):
    """Checkpoint folders of the current format only (older folders are never resumed)."""
    return [r for r in list_checkpoints(root) if r["usable"]]


@app.function
def old_checkpoint_folders(root=None):
    """Folders written in another checkpoint format: they hold nothing resumable."""
    return [r for r in list_checkpoints(root) if not r["usable"]]


@app.cell
def _():
    ui_ckpt_refresh = mo.ui.run_button(label="Refresh")
    ui_ckpt_clear = mo.ui.run_button(label="Delete checkpoints of finished runs")
    ui_ckpt_clear_old = mo.ui.run_button(label="Delete old-format folders")
    mo.vstack([
        mo.md(f"""
    ### Checkpoints

    Long runs are saved to `{CKPT["root"]}` at most every
    **{CKPT["minutes"]:g} minutes** (`CHECKPOINT_MINUTES` in the setup cell).
    **To resume after a power cut:** reopen this notebook, wait for the tests,
    set the same configuration as the unfinished run below and press that
    experiment's Run button again. A changed setting (shots, seed, p, T, code,
    decoder) is a different run and starts from zero; the worker count can
    change freely.
    """),
        mo.hstack([ui_ckpt_refresh, ui_ckpt_clear, ui_ckpt_clear_old], justify="start")])
    return ui_ckpt_clear, ui_ckpt_clear_old, ui_ckpt_refresh


@app.cell
def _(ui_ckpt_clear, ui_ckpt_clear_old, ui_ckpt_refresh):
    _ = ui_ckpt_refresh.value
    _removed = clear_checkpoints() if ui_ckpt_clear.value else []
    if ui_ckpt_clear_old.value:
        for _r in old_checkpoint_folders():
            shutil.rmtree(_r["path"], ignore_errors=True)
            _removed.append(_r["folder"])
    ckpt_rows = usable_checkpoints()

    def _ckpt_label(r):
        prog = ("finished" if r["complete"] else
                f"{r['done']}/{r['total']} chunks" if r["total"] else f"{r['done']} chunks")
        return f"{r['updated']} · {prog} · {r['label'][:70]}"

    ui_ckpt_pick = mo.ui.dropdown({_ckpt_label(r): i for i, r in enumerate(ckpt_rows)},
                                  value=_ckpt_label(ckpt_rows[0]) if ckpt_rows else None,
                                  label=f"{len(ckpt_rows)} checkpoint(s)")
    _n_old = len(old_checkpoint_folders())
    _mb = sum(r["megabytes"] for r in ckpt_rows)
    mo.vstack([
        mo.md(f"*Deleted {len(_removed)} folder(s).*") if _removed else mo.md(""),
        (mo.vstack([ui_ckpt_pick,
                    mo.md(f"Total {_mb:,.1f} MB on disk"
                          + (f" · {_n_old} old-format folder(s) hidden (not resumable)" if _n_old else ""))])
         if ckpt_rows else mo.md("*No checkpoints on disk.*"
                                 + (f" {_n_old} old-format folder(s) hidden." if _n_old else "")))])
    return ckpt_rows, ui_ckpt_pick


@app.cell
def _(ckpt_rows, ui_ckpt_pick):
    mo.stop(not ckpt_rows or ui_ckpt_pick.value is None)
    _r = ckpt_rows[ui_ckpt_pick.value]
    _m = read_manifest(_r["path"]) or {}
    mo.md(f"""
    | | |
    |:--|:--|
    | run | {_r['label']} |
    | progress | {"finished" if _r['complete'] else f"{_r['done']}/{_r['total']} chunks ({(_r['percent'] or 0):.0f}%)"} |
    | last saved | {_r['updated']} |
    | size | {_r['megabytes']:.1f} MB |
    | result file | `{_m.get('result_file') or '—'}` |
    | folder | `{_r['path']}` |
    """)
    return


@app.function
def cpu_affinity():
    """The logical cores this process may run on (None if it cannot be read)."""
    try:
        if psutil is not None:
            return sorted(psutil.Process().cpu_affinity())
        if hasattr(os, "sched_getaffinity"):
            return sorted(os.sched_getaffinity(0))
    except Exception:
        pass
    return None


@app.function
def pin_to_cores(cores=None):
    """
    Restrict this process -- and every worker process it starts afterwards,
    which inherit the setting -- to the given logical cores; None or [] =
    every core again. Returns the cores now in effect. Needs psutil on Windows
    and macOS (`pip install psutil`); Linux works without it.
    """
    allc = list(range(os.cpu_count() or 1))
    want = sorted({int(c) for c in cores}) if cores else allc
    bad = [c for c in want if c not in allc]
    if bad:
        raise ValueError(f"no such logical core(s): {bad}; this machine has 0..{allc[-1]}")
    if psutil is not None:
        psutil.Process().cpu_affinity(want)
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, want)
    else:
        raise RuntimeError("pinning needs psutil here: pip install psutil")
    return cpu_affinity()


@app.function
def power_plan():
    """The operating system's power setting, as text (None if unknown)."""
    try:
        if os.name == "nt":
            r = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True,
                               timeout=5)
            m = re.search(r"\((.+)\)\s*$", r.stdout.strip())
            return m.group(1) if m else (r.stdout.strip() or None)
        gov = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        if os.path.exists(gov):
            return "governor " + open(gov).read().strip()
    except Exception:
        pass
    return None


@app.function
def timing_environment():
    """Where and how this process was running: cores it may use, power plan,
    clock speed, battery, and machine load. Stored in every result file."""
    aff = cpu_affinity()
    n = os.cpu_count() or 1
    out = dict(affinity=aff, pinned=bool(aff is not None and len(aff) < n), logical_cores=n,
               physical_cores=None, power_plan=power_plan(), cpu_mhz=None, cpu_max_mhz=None,
               on_battery=None, load_percent=None, process_priority=None)
    if psutil is not None:
        try:
            out["physical_cores"] = psutil.cpu_count(logical=False)
            f = psutil.cpu_freq()
            if f:
                out["cpu_mhz"], out["cpu_max_mhz"] = round(f.current), round(f.max) or None
            b = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
            out["on_battery"] = None if b is None else (not b.power_plugged)
            out["load_percent"] = psutil.cpu_percent(interval=0.2)
            out["process_priority"] = str(psutil.Process().nice())
        except Exception:
            pass
    return out


@app.cell
def _():
    import tempfile as _tempfile

    def _tmp():
        return _tempfile.mkdtemp()

    def _problem(shots=60, seed=5, p=2e-3):
        code = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        circ = build_memory_circuit(code, 2, p, use_flags=True)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        det, obs = circ.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
        return H, L, pr, det, obs

    def _interrupted_run_resumes():
        H, L, pr, det, obs = _problem()
        ck = os.path.join(_tmp(), "ck")
        names = ["peel", "OSD-0"]
        full, _ = run_zoo(H, L, pr, det, obs, names, chunk_size=20, checkpoint_dir=ck)
        os.remove(os.path.join(ck, "chunk_000000020.npz"))       # the run "died" here
        again, backend = run_zoo(H, L, pr, det, obs, names, chunk_size=20, checkpoint_dir=ck)
        assert "resumed 2/3" in backend, backend
        for n in names:
            for k in ("conv", "fail", "work"):
                assert np.array_equal(full[n][k], again[n][k]), f"{n}.{k} changed on resume"
        _, other = run_zoo(H, L, pr, det, obs, ["peel"], chunk_size=20, checkpoint_dir=ck)
        assert "resumed" not in other, "chunks from a different decoder list were reused"

    def _changed_samples_are_not_resumed():
        H, L, pr, det, obs = _problem(seed=5)
        det2, obs2 = _problem(seed=6)[3:]
        ck = os.path.join(_tmp(), "ck")
        run_zoo(H, L, pr, det, obs, ["peel"], chunk_size=20, checkpoint_dir=ck)
        got, backend = run_zoo(H, L, pr, det2, obs2, ["peel"], chunk_size=20, checkpoint_dir=ck)
        assert "resumed" not in backend
        fresh, _ = run_zoo(H, L, pr, det2, obs2, ["peel"], chunk_size=20, checkpoint_dir=False)
        assert np.array_equal(got["peel"]["fail"], fresh["peel"]["fail"])

    def _damaged_chunk_is_recomputed():
        H, L, pr, det, obs = _problem()
        ck = os.path.join(_tmp(), "ck")
        full, _ = run_zoo(H, L, pr, det, obs, ["peel"], chunk_size=20, checkpoint_dir=ck)
        with open(os.path.join(ck, "chunk_000000020.npz"), "wb") as f:
            f.write(b"cut off by a power failure")
        again, backend = run_zoo(H, L, pr, det, obs, ["peel"], chunk_size=20, checkpoint_dir=ck)
        assert "resumed 2/3" in backend, backend
        assert np.array_equal(full["peel"]["fail"], again["peel"]["fail"])

    def _saves_follow_the_interval():
        H, L, pr, det, obs = _problem()
        old = CKPT["minutes"]
        try:
            for minutes, expect_mid_run in ((0, True), (60, False)):
                configure_checkpoints(minutes=minutes)
                ck = os.path.join(_tmp(), "ck")
                seen = []
                count = lambda: len([f for f in os.listdir(ck) if f.startswith("chunk_")])
                run_zoo(H, L, pr, det, obs, ["peel"], chunk_size=20, checkpoint_dir=ck,
                        on_chunk=lambda: seen.append(count()))
                assert (seen[0] >= 1) == expect_mid_run, (minutes, seen)
                assert count() == 3, "every chunk must be on disk when the run ends"
        finally:
            configure_checkpoints(minutes=old)

    def _automatic_checkpoint_resumes():
        H, L, pr, det, obs = _problem(shots=600)
        old = dict(root=CKPT["root"], min_shots=CKPT["min_shots"])
        try:
            configure_checkpoints(root=_tmp(), min_shots=500)
            a, _ = run_zoo(H, L, pr, det, obs, ["peel"], workers=1, chunk_size=37)
            rows = list_checkpoints()
            assert len(rows) == 1 and rows[0]["complete"] and rows[0]["total"] == 3, rows
            b, backend = run_zoo(H, L, pr, det, obs, ["peel"], workers=1, chunk_size=99)
            assert "resumed 3/3" in backend, backend    # chunking does not follow chunk_size
            assert np.array_equal(a["peel"]["fail"], b["peel"]["fail"])
            assert clear_checkpoints() and not list_checkpoints()
            run_zoo(H, L, pr, det[:100], obs[:100], ["peel"])
            assert not list_checkpoints(), "runs below the threshold must not checkpoint"
        finally:
            configure_checkpoints(**old)

    def _resumable_loop_resumes():
        calls = []

        def compute(a, b):
            calls.append(a)
            return dict(x=np.arange(a, b) ** 2, y=np.arange(a, b) % 3 == 0)
        ck = os.path.join(_tmp(), "loop")
        full, info = run_resumable(95, compute, ("k", 1), chunk_size=20, checkpoint_dir=ck)
        assert info["resumed"] == 0 and len(calls) == 5
        os.remove(os.path.join(ck, "chunk_000000040.npz"))
        calls.clear()
        flags = []
        again, info = run_resumable(95, compute, ("k", 1), chunk_size=20, checkpoint_dir=ck,
                                    on_chunk=lambda a, b, r: flags.append(r))
        assert calls == [40] and info["resumed"] == 4, (calls, info)
        assert flags == [True, True, False, True, True], flags
        for k in full:
            assert np.array_equal(full[k], again[k]) and full[k].dtype == again[k].dtype
        _, info = run_resumable(95, compute, ("k", 2), chunk_size=20, checkpoint_dir=ck)
        assert info["resumed"] == 0, "a different key must not reuse chunks"

    def _stale_priors_are_not_reused():
        """Same structure, different priors -> different decoder."""
        pa = surface_code_problem(3, 3, 0.012)
        pb = surface_code_problem(3, 3, 5e-4)
        assert pa["H"].shape == pb["H"].shape
        det, obs = pa["circuit"].compile_detector_sampler(seed=3).sample(
            400, separate_observables=True)
        DECODER_CACHE.clear()
        a1, _ = run_zoo(pa["H"], pa["L"], pa["priors"], det, obs, [("s", "OSD-0")],
                        checkpoint_dir=False)
        run_zoo(pb["H"], pb["L"], pb["priors"], det, obs, [("s", "OSD-0")], checkpoint_dir=False)
        a2, _ = run_zoo(pa["H"], pa["L"], pa["priors"], det, obs, [("s", "OSD-0")],
                        checkpoint_dir=False)
        assert np.array_equal(a1["s"]["fail"], a2["s"]["fail"]), "stale priors were reused"
        (dec,) = [d for k, d in DECODER_CACHE.items() if k[1] == "s"]
        assert np.allclose(np.asarray(dec.dec.channel_probs), pa["priors"])

    def _warmup_removes_chunk_start_spikes():
        """With three decoders and four chunks, chunk-start shots are not slow."""
        H, L, pr, det, obs = _problem(shots=1000, p=1e-3)
        DECODER_CACHE.clear()
        r, _ = run_zoo(H, L, pr, det, obs, ["peel", "BP-ms-10", "OSD-0"], chunk_size=250,
                       checkpoint_dir=False)
        for name in ("peel", "OSD-0"):
            t = r[name]["time_us"].astype(float)
            starts = t[::250]
            rest = np.delete(t, np.arange(0, 1000, 250))
            assert np.median(starts) < 3 * np.median(rest) + 50, (name, starts, np.median(rest))
            assert t.max() < 100 * np.median(t) + 5000, (name, t.max(), np.median(t))
        assert r["peel"]["build_us"][0] > 0

    def _format_2_chunks_are_refused():
        """A chunk without this format's fingerprint is never loaded."""
        H, L, pr, det, obs = _problem()
        ck = os.path.join(_tmp(), "ck")
        run_zoo(H, L, pr, det, obs, ["peel"], chunk_size=20, checkpoint_dir=ck)
        p = os.path.join(ck, "chunk_000000000.npz")
        with np.load(p) as d:
            arrays = {k: d[k] for k in d.files if k != "fingerprint"}
        np.savez(p, **arrays)                              # an old-format chunk: no fingerprint
        _, backend = run_zoo(H, L, pr, det, obs, ["peel"], chunk_size=20, checkpoint_dir=ck)
        assert "resumed 2/3" in backend, backend

    def _checkpoint_folder_is_labelled():
        """Role, README and the info dict."""
        H, L, pr, det, obs = _problem()
        ck = os.path.join(_tmp(), "checkpoints", "x")
        _, _, info = run_zoo(H, L, pr, det, obs, ["peel"], chunk_size=20, checkpoint_dir=ck,
                             result_file="E08_x.json", return_info=True)
        m = read_manifest(ck)
        assert m["role"].startswith("checkpoint") and m["result_file"] == "E08_x.json"
        assert os.path.exists(os.path.join(ck, "README.txt"))
        assert info["fingerprint"] == m["fingerprint"] and info["total_chunks"] == 3
        assert info["workers_effective"] == 1
        assert is_checkpoint_path(os.path.join(ck, "manifest.json"))
        assert is_checkpoint_path(os.path.join(ck, "anything.json"))

    def _sample_hash_properties():
        H, L, pr, det, obs = _problem()
        assert sample_hash(det, obs) == sample_hash(det.astype(np.uint8), obs.astype(bool))
        d2 = det.copy()
        d2[0, 0] ^= True
        assert sample_hash(d2, obs) != sample_hash(det, obs)
        assert sample_hash(*_problem()[3:]) == sample_hash(det, obs)   # same seed, same machine
        a, b = stim_selfcheck(), stim_selfcheck()
        assert a == b and len(a["digest"]) == 64

    def _workers_load_this_file():
        """The worker path: a 2-process pool decodes by loading this file from its path."""
        H, L, pr, det, obs = _problem(shots=300)
        seq, _ = run_zoo(H, L, pr, det, obs, ["peel"], workers=1, chunk_size=100,
                         checkpoint_dir=False)
        par, backend = run_zoo(H, L, pr, det, obs, ["peel"], workers=2, chunk_size=100,
                               checkpoint_dir=False)
        assert backend.startswith("parallel (2 processes)"), backend
        assert np.array_equal(seq["peel"]["fail"], par["peel"]["fail"])

    def _loader_ignores_case_and_cwd():
        """A fresh interpreter, started elsewhere, loads the file under a lower-case
        module name (as a Windows path may spell it) through WORKER_LOADER."""
        code = ("import sys; N, P = sys.argv[1], sys.argv[2]; "
                "m = eval(sys.argv[3], {'N': N, 'P': P}); print(callable(m.decode_chunk))")
        r = subprocess.run([sys.executable, "-c", code, MODULE_NAME.lower(), NOTEBOOK_FILE,
                            WORKER_LOADER], cwd=_tmp(), capture_output=True, text=True, timeout=300)
        assert r.returncode == 0 and r.stdout.strip().endswith("True"), r.stderr[-500:]

    def _pinning_round_trip():
        """Pin to one core, see it in timing_environment, unpin again."""
        before = cpu_affinity()
        if before is None or (psutil is None and not hasattr(os, "sched_setaffinity")):
            return
        try:
            now = pin_to_cores([before[0]])
            assert now == [before[0]]
            env = timing_environment()
            assert env["affinity"] == [before[0]]
            assert env["pinned"] == ((os.cpu_count() or 1) > 1)
        finally:
            pin_to_cores(before)
        assert cpu_affinity() == before
        try:
            pin_to_cores([10 ** 6])
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a core that does not exist")

    tests_engine = run_checks([
        ("an interrupted run resumes with identical results", _interrupted_run_resumes),
        ("checkpoints from different samples are not reused", _changed_samples_are_not_resumed),
        ("a damaged checkpoint file is recomputed, not trusted", _damaged_chunk_is_recomputed),
        ("checkpoints are written on the CHECKPOINT_MINUTES schedule", _saves_follow_the_interval),
        ("large runs checkpoint automatically and resume", _automatic_checkpoint_resumes),
        ("a resumable loop resumes with identical results", _resumable_loop_resumes),
        ("the decoder cache never reuses stale priors", _stale_priors_are_not_reused),
        ("warm-up removes chunk-start timing spikes", _warmup_removes_chunk_start_spikes),
        ("format-2 chunks without a fingerprint are refused", _format_2_chunks_are_refused),
        ("checkpoint folders say they are not results", _checkpoint_folder_is_labelled),
        ("sample hashes are dtype-independent and bit-sensitive", _sample_hash_properties),
        ("worker processes decode by loading this file", _workers_load_this_file),
        ("the worker loader ignores the file-name case and the current folder", _loader_ignores_case_and_cwd),
        ("pinning to a core round-trips and is recorded", _pinning_round_trip),
    ])
    render_checks("parallel decoding and checkpoints", tests_engine)
    return (tests_engine,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 11. Flag structure: groups, exact posteriors and laws L1-L7

    **Condition 4.1 (disjoint single-parity flag groups).** Write $G_i$ for the
    support of the $i$-th flag row of $H$. The condition holds when the $G_i$
    are pairwise disjoint: every fault mechanism appears in at most one flag row.

    **Proposition 4.2 (Gallager's parity-check posterior).** Under Condition 4.1
    and the independent-fault model of the DEM, with soft values
    $\lambda_k = 1 - 2p_k = \tanh(L_k/2)$ ($L_k$ the prior log-likelihood ratio)
    and $s = (-1)^f$,
    $$P(e_j = 1 \mid f) = p_j \,\frac{1 - s\prod_{k \in G \setminus j} \lambda_k}{1 + s\prod_{k \in G}\lambda_k}.$$

    *Attribution.* The formula is not new. It is Gallager's (1962, Lemma 1;
    1963, Lemma 4.1 and Theorem 4.1): the probability that a set of independent
    bits has even parity is $(1 + \prod_k \lambda_k)/2$, and the posterior of one
    bit given the parity uses the leave-one-out product over the other bits —
    our denominator is twice his normaliser. The same identity is the
    sum-product check-node update (the "boxplus" of soft values), MacKay and
    Neal's $\delta r$ recursion, and the piling-up lemma of linear cryptanalysis
    (Matsui, 1993); in quantum decoding it appears with the $(-1)^{s}$ syndrome
    prefactor as Lemma 3.1 of Gong, Cammerer and Renes (2023). The brute-force
    self-test below checks it for completeness only.

    **What this section establishes instead** — the parts Gallager's identity
    presupposes:

    1. *The model mapping.* On the flagged circuit of §4 every flag detector is
       the XOR of an identifiable set of Stim DEM fault mechanisms, and that set
       has exactly $w$ members (law L1, measured on every code in the notebook).
       Gallager's lemma needs a parity relation; whether a flagged BB-code
       extraction circuit produces one is a property of the circuit.
    2. *Disjointness gives exactness for every flag at once.* When every fault
       lies in at most one flag row (Condition 4.1, flag-matrix column weight
       ≤ 1), different flags depend on disjoint sets of independent faults, so
       $P(e_j \mid \text{all flags}) = P(e_j \mid \text{its own flag})$ exactly,
       for every $j$ simultaneously. Gallager's own derivation is exact only on
       a tree (his first tier); MacKay and Neal call the full algorithm
       approximate *because of loops*, not because of the check-node formula.
       Column weight ≤ 1 is what makes every flag group loop-free at once.
    3. *The update is lossy, by construction.* Conditioning on $f$ leaves the
       faults of a group **correlated**: for $|G| = 2$ and $f = 0$, $e_1 = e_2$
       with probability 1. A prior *vector* records only the marginals, so the
       updated arm discards the within-group correlation — it is a strictly
       weaker model of the same bits than the sighted arm, which keeps the flag
       row as a check. A self-test below shows the loss on a two-fault group.

    The practical claim — no calibration, no Monte Carlo, no lookup table: one
    closed-form evaluation per circuit, then a table look-up of two precomputed
    values per fault per shot — is a claim about the alternatives (e.g. the
    sampled lookup tables of Takada and Fujii, 2024) and is unaffected by the
    attribution.

    **What a group contains.** On [[72,12,6]] every group has six members: five
    genuine ancilla faults (prior $8p/15 = 5.33\times10^{-4}$ at $p = 10^{-3}$,
    posterior **0.1025** when the flag fires) and one fault of the flag's own
    apparatus — readout or reset flip, or an X/Y on the flag at either flag CNOT
    (prior $2p + 8p/15 = 2.53\times10^{-3}$, posterior **0.4873**). The largest
    posterior is therefore the flag apparatus, not a hook, and it sits just below
    0.49.

    **Laws** (derived on the circuit of §4, verified below; $w$ = X-check weight,
    $m_X$ = number of X checks):

    | law | statement |
    |---|---|
    | L1 | $\lvert G\rvert = w$ (the window holds $w-2$ data layers) |
    | L2 | groups are pairwise disjoint (Condition 4.1) |
    | L3 | $p_{gen} = 8p/15$, $p_{app} = 2p + 8p/15$ |
    | L4 | $S = \sum_{G} p = (8w/15 + 2)\,p$ |
    | L5 | $\max_j P(e_j \mid f=1) = 38/(8w+30)$, independent of $p$ |
    | L6 | flag rate $= 1 - (1-S)^{m_X T}$ |
    | L7 | flags add exactly $3 m_X T$ fault mechanisms |

    **L7 and merged faults.** Stim merges faults with identical detector and
    observable signatures into one DEM column. On some codes two *different*
    faults of the unflagged circuit have the same signature, but fire different
    flags in the flagged circuit, so there they are two columns. The DEM
    difference flagged − unflagged is then $3 m_X T$ **plus** the number of such
    pairs; `flag_structure_report` counts them (`separated_by_flags`) so L7 is
    checked on the difference minus them. Example: GB24 w=11 has 144 such pairs
    at $T = 12$ (12 per round, each pair firing flags of two different X checks);
    the other w-family members and every BB and GB preset have none.
    """)
    return


@app.function
def flag_groups(H, mask):
    """
    The flag rows of H as groups of fault indices, padded to a rectangle.

    Returns (idx, disjoint, info). `idx` is (n_flag_rows, max_group) with -1
    in the padding. `disjoint` is False when some fault lights more than one
    flag (Condition 4.1 fails), in which case the closed form of
    `flag_posteriors` does not apply and the caller must not use it.
    """
    HF = (H[mask] > 0).tocsr()
    groups = [HF.indices[HF.indptr[i]:HF.indptr[i + 1]] for i in range(HF.shape[0])]
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


@app.function
def flag_posteriors(priors, idx, floor=1e-12, ceiling=1.0 - 1e-12):
    """
    Exact per-fault posterior for each flag outcome (Proposition 4.2 --
    Gallager's parity-check posterior), computed once per circuit.

    Returns (lo, hi, owner): lo[j] and hi[j] are fault j's probability when
    its flag reads 0 and 1; owner[j] is the flag row that owns fault j, or -1
    when no flag touches it (those faults keep their prior in both vectors).

    The leave-one-out product is formed from prefix and suffix cumulative
    products, so a prior of exactly 0.5 (lambda = 0) cannot divide by zero.
    `floor` and `ceiling` only keep the log-likelihood ratio finite; they are
    not a probability clamp. A posterior above 0.5 is meaningful and is kept.
    On [[72,12,6]] the largest posterior is 0.4873 (the flag's own apparatus);
    the five genuine ancilla faults of a group get 0.1025 each.
    """
    pad = idx < 0
    P = np.where(pad, 0.0, priors[np.clip(idx, 0, None)])
    lam = 1.0 - 2.0 * P
    pre = np.ones_like(lam)
    suf = np.ones_like(lam)
    if lam.shape[1] > 1:
        pre[:, 1:] = np.cumprod(lam[:, :-1], axis=1)
        suf[:, :-1] = np.cumprod(lam[:, :0:-1], axis=1)[:, ::-1]
    excl = pre * suf
    total = (excl * lam)[:, :1] if lam.shape[1] else np.ones((lam.shape[0], 1))

    def post(s):
        num = P * (1.0 - s * excl) / 2.0
        den = (1.0 + s * total) / 2.0
        return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), P)

    lo, hi = priors.copy(), priors.copy()
    owner = np.full(priors.shape[0], -1, dtype=np.int64)
    keep = ~pad
    rows = np.broadcast_to(np.arange(idx.shape[0])[:, None], idx.shape)
    lo[idx[keep]] = np.clip(post(1.0)[keep], floor, ceiling)
    hi[idx[keep]] = np.clip(post(-1.0)[keep], floor, ceiling)
    owner[idx[keep]] = rows[keep]
    return lo, hi, owner


@app.function
def shot_priors(lo, hi, owner, flag_bits):
    """Fault priors for one shot, given which flag detectors fired."""
    fired = np.zeros(lo.shape[0], dtype=bool)
    touched = owner >= 0
    fired[touched] = np.asarray(flag_bits, dtype=bool)[owner[touched]]
    return np.where(fired, hi, lo)


@app.function
def flag_law_predictions(w, p, mx, rounds):
    """Laws L1-L7 as numbers, for a code with X-check weight w and mx X checks."""
    S = (8 * w / 15 + 2) * p
    return dict(L1_group_size=int(w), L3_p_gen=8 * p / 15, L3_p_app=2 * p + 8 * p / 15,
                L4_S=S, L5_max_posterior=38 / (8 * w + 30),
                L6_flag_rate=1 - (1 - S) ** (mx * rounds),
                L7_added_mechanisms=3 * mx * rounds)


@app.function
def flag_structure_report(code, rounds, p):
    """
    Measure L1-L7 on the real circuit (flagged and unflagged, X detectors off)
    and put the law predictions beside them.
    """
    circ = build_memory_circuit(code, rounds, p, use_flags=True, x_detectors=False)
    H, _L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
    mask = flag_detector_mask(code, rounds, True, circ.num_detectors, x_detectors=False)
    idx, disjoint, info = flag_groups(H, mask)
    lo, hi, owner = flag_posteriors(pr, idx)
    sums = np.where(idx >= 0, pr[np.clip(idx, 0, None)], 0.0).sum(1)
    base = build_memory_circuit(code, rounds, p, use_flags=False, x_detectors=False)
    Hb, _Lb, _pb = dem_to_matrices(base.detector_error_model(decompose_errors=False))
    w = int(np.asarray(code.hx.sum(axis=1)).ravel().max())
    mx = code.hx.shape[0]
    pred = flag_law_predictions(w, p, mx, rounds)
    touched = owner >= 0
    # faults that look identical without the flag rows but fire different flags
    Hc, Lc = sp.csc_matrix(H), sp.csc_matrix(_L)
    keep = np.flatnonzero(~mask)
    pos = np.full(H.shape[0], -1, np.int64)
    pos[keep] = np.arange(len(keep))
    groups_by_sig = {}
    for j in range(H.shape[1]):
        rows = Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]]
        zr = tuple(sorted(int(pos[r]) for r in rows if pos[r] >= 0))
        lr = tuple(sorted(Lc.indices[Lc.indptr[j]:Lc.indptr[j + 1]].tolist()))
        fr = tuple(sorted(int(r) for r in rows if pos[r] < 0))
        groups_by_sig.setdefault((zr, lr), set()).add(fr)
    separated = sum(len([f for f in fs if f]) - 1 for (zr, lr), fs in groups_by_sig.items()
                    if (zr or lr) and len([f for f in fs if f]) > 1)
    return dict(code=code.name, rounds=int(rounds), p=float(p), w=w, mx=int(mx),
                disjoint=bool(disjoint), max_flag_weight=info["max_flag_weight"],
                flag_rows=info["flag_rows"], faults_covered=info["faults_covered"],
                faults_total=int(H.shape[1]), group_sizes=info["group_sizes"],
                S_mean=float(sums.mean()), S_max=float(sums.max()),
                max_posterior=float(hi[touched].max()) if touched.any() else float("nan"),
                added_mechanisms=int(H.shape[1] - Hb.shape[1]),
                separated_by_flags=int(separated),
                predicted_flag_rate=float(1 - np.prod(1 - sums)),
                **pred)


@app.cell
def _():
    def _brute_force_posterior(p, j, f):
        """Exact marginal of bit j given the group's parity, by enumeration."""
        import itertools
        num = den = 0.0
        for b in itertools.product([0, 1], repeat=len(p)):
            if sum(b) % 2 != f:
                continue
            wgt = np.prod([p[k] if b[k] else 1 - p[k] for k in range(len(p))])
            den += wgt
            if b[j]:
                num += wgt
        return num / den

    def _closed_form_matches_brute_force():
        rng = np.random.default_rng(3)
        for size in (1, 2, 4, 6, 7):
            pr = rng.uniform(1e-4, 0.3, size)
            lo, hi, _owner = flag_posteriors(pr, np.arange(size)[None, :])
            for j in range(size):
                # tolerance 2e-12: a lone fault's posterior is exactly 0 or 1, which
                # the LLR floor/ceiling holds 1e-12 away
                assert abs(lo[j] - _brute_force_posterior(pr, j, 0)) < 2e-12, (size, j)
                assert abs(hi[j] - _brute_force_posterior(pr, j, 1)) < 2e-12, (size, j)

    def _padding_does_not_change_a_group():
        pr = np.array([0.01, 0.02, 0.03, 0.04])
        a_lo, a_hi, _ = flag_posteriors(pr, np.array([[0, 1, 2, 3]]))
        b_lo, b_hi, _ = flag_posteriors(pr, np.array([[0, 1, 2, 3, -1, -1]]))
        assert np.allclose(a_lo, b_lo) and np.allclose(a_hi, b_hi)

    def _half_probability_does_not_divide_by_zero():
        lo, hi, _ = flag_posteriors(np.array([0.5, 0.5, 0.1]), np.array([[0, 1, 2]]))
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
        assert (hi > pr).all() and (lo < pr).all()

    def _untouched_faults_keep_their_prior():
        pr = np.array([1e-3, 2e-3, 3e-3, 4e-3])
        lo, hi, owner = flag_posteriors(pr, np.array([[0, 1]]))
        assert lo[2] == pr[2] and hi[3] == pr[3]
        assert owner[2] == -1 and owner[0] == 0

    def _shot_priors_selects_by_flag():
        pr = np.full(4, 1e-3)
        lo, hi, owner = flag_posteriors(pr, np.array([[0, 1], [2, 3]]))
        out = shot_priors(lo, hi, owner, np.array([True, False]))
        assert out[0] == hi[0] and out[1] == hi[1] and out[2] == lo[2] and out[3] == lo[3]

    def _non_disjoint_groups_are_detected():
        H = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
        _idx, ok, info = flag_groups(H, np.array([True, True]))
        assert not ok, info
        _idx, ok, _ = flag_groups(sp.csr_matrix(np.array([[1, 1, 0, 0], [0, 0, 1, 1]],
                                                         dtype=np.uint8)), np.array([True, True]))
        assert ok

    def _bounds_only_keep_the_llr_finite():
        pr = np.array([1.0 - 1e-18, 1e-18])
        lo, hi, _ = flag_posteriors(pr, np.array([[0, 1]]))
        assert np.all(np.isfinite(np.log(lo / (1 - lo))))
        assert np.all(np.isfinite(np.log(hi / (1 - hi))))
        _lo2, hi2, _ = flag_posteriors(np.array([0.4, 0.05, 0.05]), np.array([[0, 1, 2]]))
        assert hi2[0] > 0.5, hi2

    def _posterior_values_on_72():
        """five faults at 0.10255 and one at 0.48727 per group, max < 0.49."""
        c = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        circ = build_memory_circuit(c, 12, 1e-3, use_flags=True, x_detectors=False)
        H, _L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(c, 12, True, circ.num_detectors, x_detectors=False)
        idx, ok, info = flag_groups(H, mask)
        lo, hi, owner = flag_posteriors(pr, idx)
        v = hi[owner >= 0]
        genuine = np.abs(v - 0.10255) < 1e-4
        apparatus = np.abs(v - 0.48727) < 1e-4
        assert genuine.sum() == 2160 and apparatus.sum() == 432, (genuine.sum(), apparatus.sum())
        assert v.max() < 0.49

    def _condition_4_1_on_every_code():
        """disjoint groups, max flag weight 1, |G| = X-check weight (L1)."""
        for key, w in (("[[72, 12, 6]]", 6), ("[[144, 12, 12]]", 6), ("[[288, 12, 18]]", 6),
                       ("[[46, 2, 9]]", 10), ("[[48, 6, 8]]", 8)):
            c = code_from_key(key)
            r = flag_structure_report(c, 2, 1e-3)
            assert r["disjoint"] and r["max_flag_weight"] == 1, (key, r)
            assert r["group_sizes"] == [w] == [r["L1_group_size"]], (key, r["group_sizes"])

    def _laws_hold_on_72():
        """L4, L5, L6, L7 on [[72,12,6]] (L6 against the per-group product)."""
        r = flag_structure_report(bb_from_preset("[[72, 12, 6]]", BB_PRESETS), 12, 1e-3)
        assert abs(r["S_mean"] / r["L4_S"] - 1) < 2e-3, (r["S_mean"], r["L4_S"])
        assert abs(r["max_posterior"] - r["L5_max_posterior"]) < 5e-4
        assert abs(r["predicted_flag_rate"] - r["L6_flag_rate"]) < 2e-3
        assert r["added_mechanisms"] == r["L7_added_mechanisms"] == 3 * 36 * 12
        assert r["separated_by_flags"] == 0
        assert r["flag_rows"] == 432 and r["faults_covered"] == 2592

    def _laws_hold_on_the_w_family():
        """L1, L4, L5, L7 on four members of the Experiment 9 family (w = 4, 8, 11, 12);
        w = 11 is the member whose unflagged circuit merges fault pairs."""
        for key in (W_FAMILY_KEYS[0], W_FAMILY_KEYS[4], W_FAMILY_KEYS[7], W_FAMILY_KEYS[8]):
            r = flag_structure_report(code_from_key(key), 2, 1e-3)
            assert r["group_sizes"] == [r["w"]], (key, r["group_sizes"])
            assert abs(r["S_mean"] / r["L4_S"] - 1) < 2e-3, key
            assert abs(r["max_posterior"] - r["L5_max_posterior"]) < 5e-4, key
            assert r["added_mechanisms"] - r["separated_by_flags"] == r["L7_added_mechanisms"], (
                key, r["added_mechanisms"], r["separated_by_flags"])
            assert (r["separated_by_flags"] > 0) == (r["w"] == 11), (key, r["separated_by_flags"])

    def _update_discards_within_group_correlation():
        """|G| = 2, f = 0: the exact joint has e1 = e2 always; the product of the
        updated marginals (what a prior vector encodes) does not."""
        import itertools
        p = np.array([0.1, 0.2])
        lo, hi, owner = flag_posteriors(p.copy(), np.array([[0, 1]]))
        joint = {b: np.prod([p[k] if b[k] else 1 - p[k] for k in range(2)])
                 for b in itertools.product([0, 1], repeat=2) if sum(b) % 2 == 0}
        z = sum(joint.values())
        p_differ_exact = sum(v for b, v in joint.items() if b[0] != b[1]) / z
        p_differ_vector = lo[0] * (1 - lo[1]) + lo[1] * (1 - lo[0])
        assert p_differ_exact == 0.0 and p_differ_vector > 1e-3
        for j in range(2):                 # the marginals themselves are exact
            assert abs(lo[j] - _brute_force_posterior(p, j, 0)) < 1e-12

    tests_flags = run_checks([
        ("closed form equals brute-force enumeration (groups of 1-7)", _closed_form_matches_brute_force),
        ("marginals are exact but a prior vector drops within-group correlation", _update_discards_within_group_correlation),
        ("the bounds keep the LLR finite without clamping", _bounds_only_keep_the_llr_finite),
        ("padding a group leaves it unchanged", _padding_does_not_change_a_group),
        ("a prior of exactly 0.5 does not divide by zero", _half_probability_does_not_divide_by_zero),
        ("posteriors stay inside [0, 1]", _posteriors_are_probabilities),
        ("a fired flag raises its faults, a quiet one lowers them", _a_fired_flag_raises_and_a_quiet_one_lowers),
        ("faults no flag touches keep their prior", _untouched_faults_keep_their_prior),
        ("a shot's priors select by its own flag bits", _shot_priors_selects_by_flag),
        ("overlapping flag groups are detected, not used", _non_disjoint_groups_are_detected),
        ("[[72,12,6]] posteriors: 2,160 at 0.1025, 432 at 0.4873", _posterior_values_on_72),
        ("Condition 4.1 and L1 on five codes", _condition_4_1_on_every_code),
        ("laws L4-L7 hold on [[72,12,6]]", _laws_hold_on_72),
        ("laws L1, L4, L5, L7 hold on the w-family (merged pairs counted)", _laws_hold_on_the_w_family),
    ])
    render_checks("11 flag structure", tests_flags)
    return (tests_flags,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 12. Result files and provenance

    Every result is saved with everything needed to trust it later, and every
    thesis table and figure is regenerated from these files (§18).

    **Schema 2.** A result file is written **only by a run cell**, atomically
    (UTF-8, temporary file, fsync, rename) and validated before it is written. It
    records: kind, experiment number, run id, status (`partial` while a sweep is
    running, then `complete` or `aborted`), start / update / finish times,
    software versions (stim, ldpc, numpy, scipy, marimo, numba, matplotlib,
    pymatching, relay-bp), the machine, a sha256 of this file and the git commit,
    a Stim self-check digest, the full configuration (seed, detector convention,
    every decoder parameter), the timing regime (workers requested and actually
    used, backends, resumed checkpoint chunks) and a sha256 of every sample set.
    **Displaying or reloading a file never rewrites it**; re-exporting a loaded
    file is an explicit button, and the export records "re-export of a saved
    file" with the data file's hash.

    **File names** list their fields in a fixed order and always end with the
    detector-convention token: `E08_72-12-6_T12_p0.001_50000shots_seed…_osd0_detZ_arms-…`.
    A name without a token was written by an earlier version of this notebook.
    Writing a *different* run to an existing name moves the old file to
    `results/superseded/` first.

    **Older files** (schema 1) still load. A field they did not record is
    reported as *unknown*, never filled with a default — an ablation file without
    an `x_detectors` field must not be read as "X detectors off" when it may have
    run with them on. The detector convention is taken from the file, then from
    `LEGACY_PROVENANCE` (setup cell; each entry states its evidence), then
    inferred from stored detector or fault counts where those determine it.
    """)
    return


@app.class_definition
@dataclass(frozen=True)
class ExperimentConfig:
    code: str = "[[72, 12, 6]]"
    rounds: int = 12
    p: float = 1e-3
    shots: int = 1000
    seed: int = 20260921
    use_flags: bool = True
    osd_order: int = 0
    workers: int = 1          # CPU processes used for decoding; 1 = sequential
    # False drops the X-check detectors (the corrected model). None means
    # "unknown" -- an older file that did not record it. Runners refuse None.
    x_detectors: object = False

    @property
    def detector_convention(self):
        if self.x_detectors is None:
            return "unknown"
        return "x+z" if self.x_detectors else "z-only"


@app.function
def software_versions():
    """Versions of every package a result depends on."""
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in ("stim", "ldpc", "numpy", "scipy", "marimo", "numba", "matplotlib",
                "pymatching", "relay-bp", "psutil"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


@app.function
def environment():
    """The machine a run executed on."""
    return dict(python=platform.python_version(),
                implementation=platform.python_implementation(),
                platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), cpu_count=os.cpu_count() or 1,
                numba=NUMBA_STATUS, timing=timing_environment())


@app.function
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@app.function
def code_identity():
    """sha256 of this notebook file and the git commit it sits in (if any)."""
    out = dict(file=os.path.basename(NOTEBOOK_FILE), sha256=None, git_commit=None,
               git_dirty=None)
    try:
        out["sha256"] = file_sha256(NOTEBOOK_FILE)
    except OSError:
        pass
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=NOTEBOOK_DIR,
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            out["git_commit"] = r.stdout.strip()
            d = subprocess.run(["git", "status", "--porcelain", "--", os.path.basename(NOTEBOOK_FILE)],
                               cwd=NOTEBOOK_DIR, capture_output=True, text=True, timeout=5)
            out["git_dirty"] = bool(d.stdout.strip())
    except Exception:
        pass
    return out


@app.function
def start_run(kind, experiment):
    """The stamp every file of one run carries: one run id, one start time."""
    import uuid
    return dict(run_id=uuid.uuid4().hex, kind=str(kind), experiment=str(experiment),
                started=datetime.datetime.now().isoformat(timespec="seconds"),
                software=software_versions(), environment=environment(),
                code=code_identity(), stim_selfcheck=stim_selfcheck())


@app.function
def validate_result(doc):
    """
    Check a schema-2 result document before it is written. Raises ValueError
    naming the first missing or invalid field.
    """
    for key in ("schema", "kind", "experiment", "numbering", "status", "run_id", "started",
                "updated", "software", "environment", "code", "config", "timing", "samples"):
        if key not in doc:
            raise ValueError(f"result is missing '{key}'")
    if doc["status"] not in ("partial", "complete", "aborted"):
        raise ValueError(f"status must be partial/complete/aborted, got {doc['status']!r}")
    cfg = doc["config"]
    if not any(k in cfg for k in ("code", "codes", "distances")):
        raise ValueError("config needs 'code', 'codes' or 'distances'")
    if not any(k in cfg for k in ("rounds", "rounds_list")):
        raise ValueError("config needs 'rounds' or 'rounds_list'")
    if not any(k in cfg for k in ("p", "ps")):
        raise ValueError("config needs 'p' or 'ps'")
    if not any(k in cfg for k in ("shots", "shots_per_config")):
        raise ValueError("config needs 'shots'")
    if not any(cfg.get(k) is not None for k in ("seed", "master_seed", "seeds")):
        raise ValueError("config needs 'seed'")
    conv = cfg.get("detector_convention")
    if conv not in DETECTOR_CONVENTIONS:
        raise ValueError(f"detector_convention must be one of {DETECTOR_CONVENTIONS}, got {conv!r}")
    if "decoders" not in cfg:
        raise ValueError("config needs 'decoders' (decoder_spec_full of every decoder)")
    for key in ("workers_requested", "workers_effective", "backends", "resumed_chunks",
                "total_chunks"):
        if key not in doc["timing"]:
            raise ValueError(f"timing is missing '{key}'")
    return True


@app.function
def write_result_file(path, payload, stamp, status="complete"):
    """
    Write one result document (schema 2), validated, atomically, UTF-8.

    `stamp` comes from start_run(). Before the first write of a run, a file
    already at `path` from a DIFFERENT run (or a schema-1 file) is moved to
    results/superseded/<name>.<timestamp> together with its export folder,
    and the new file records what it superseded. Never call this from a cell
    that only displays a loaded result.
    """
    now = datetime.datetime.now().isoformat(timespec="seconds")
    doc = {"schema": SCHEMA_VERSION, "numbering": NUMBERING, **payload,
           "kind": stamp["kind"], "experiment": stamp["experiment"],
           "run_id": stamp["run_id"], "status": status, "started": stamp["started"],
           "updated": now, "finished": now if status != "partial" else None,
           "software": stamp["software"], "environment": stamp["environment"],
           "code": stamp["code"], "stim_selfcheck": stamp["stim_selfcheck"]}
    validate_result(doc)
    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                old = json.load(fh)
        except (OSError, ValueError):
            old = {}
        if old.get("run_id") != stamp["run_id"]:
            sup = os.path.join(folder, "superseded")
            os.makedirs(sup, exist_ok=True)
            tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            moved = os.path.join(sup, f"{os.path.basename(path)}.{tag}")
            shutil.move(path, moved)
            bundle = os.path.splitext(path)[0]
            if os.path.isdir(bundle):
                shutil.move(bundle, moved + ".export")
            doc["supersedes"] = dict(file=os.path.relpath(moved, folder),
                                     run_id=old.get("run_id"))
        elif old.get("supersedes"):
            doc["supersedes"] = old["supersedes"]
    text = json.dumps(doc, indent=1, ensure_ascii=False, default=float).encode("utf-8")
    durable_write(path, lambda f: f.write(text))
    return path


@app.function
def read_result_file(path, kind=None):
    """
    Load a result document (schema 1 or 2). Refuses checkpoint folders. With
    `kind`, raises ValueError unless the file's kind (or its older alias)
    matches.
    """
    if is_checkpoint_path(path):
        raise ValueError(f"{path} is inside a checkpoint folder, not a result file")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if kind is not None:
        kinds = (kind,) if isinstance(kind, str) else tuple(kind)
        actual = d.get("kind")
        if actual not in kinds and LEGACY_KIND_ALIASES.get(actual) not in kinds:
            raise ValueError(f"{path} holds a {actual!r} result, expected {kinds}")
    return d


@app.function
def normalise_rounds(r):
    """(rounds as int, rounds as list): accepts 12, [12] or [3, 6, 12]."""
    if isinstance(r, (list, tuple)):
        lst = [int(x) for x in r]
        return (lst[0] if len(lst) == 1 else lst[-1]), lst
    return int(r), [int(r)]


@app.function
def infer_convention(kind, d):
    """
    The detector convention of an older file, from what it stored. Returns
    (x_detectors True/False/None, how). EXP2 points store detector counts;
    EXP1 stores its fault count (checked against both conventions); EXP7 is
    the surface code with all of Stim's detectors.
    """
    if kind in ("exp2", "L2-depth-sweep"):
        pts = d.get("points") or []
        if pts and "detectors_unflagged" in pts[0]:
            pt = pts[0]
            code = code_from_key(d["config"]["code"])
            T = int(pt["rounds"])
            for x in (False, True):
                if (expected_num_detectors(code, T, False, x_detectors=x) == pt["detectors_unflagged"]
                        and expected_num_detectors(code, T, True, x_detectors=x)
                        == pt["detectors_flagged"]):
                    return x, f"inferred: detector counts match x_detectors={x}"
    checks = d.get("checks", d)
    if kind in ("validation", "L1-validation") and "v1" in checks:
        cfg = d.get("config", d)
        code = code_from_key(cfg["code"])
        T = int(cfg["rounds"][-1] if isinstance(cfg["rounds"], list) else cfg["rounds"])
        for x in (False, True):
            circ = build_memory_circuit(code, T, float(cfg["p"]), use_flags=True, x_detectors=x)
            n_faults = circ.detector_error_model(decompose_errors=False).num_errors
            if n_faults == checks["v1"]["our_faults"]:
                return x, f"inferred: fault count {n_faults} matches x_detectors={x}"
    if kind in ("exp7", "E06-surface"):
        return True, "inferred: surface code with every Stim detector ((d^2-1)T)"
    return None, "unknown"


@app.function
def config_from_file(d, path):
    """
    (config dict, provenance dict) for any result file. Missing fields stay
    None with status "unknown"; the detector convention comes from the file,
    then LEGACY_PROVENANCE, then inference from stored counts.
    """
    cfg = dict(d.get("config") or {})
    for key in ("code", "rounds", "p", "shots"):
        if key not in cfg and key in d:
            cfg[key] = d[key]
    prov = {}
    base = os.path.basename(str(path))
    reg = LEGACY_PROVENANCE.get(base, {})
    if "x_detectors" in cfg and cfg["x_detectors"] is not None:
        prov["x_detectors"] = "recorded"
    elif "x_detectors" in reg:
        cfg["x_detectors"] = reg["x_detectors"]
        prov["x_detectors"] = f"from LEGACY_PROVENANCE: {reg['evidence']}"
    else:
        x, how = infer_convention(d.get("kind"), d)
        cfg["x_detectors"] = x
        prov["x_detectors"] = how
    if d.get("kind") == "pakhunov":
        # the reference circuit has Z-check detectors only, by construction
        cfg["x_detectors"] = False
        cfg["detector_convention"] = "reference-z-only"
        prov["x_detectors"] = "inferred: the reference circuit extracts Z checks only"
    if "detector_convention" not in cfg:
        cfg["detector_convention"] = reg.get("detector_convention") or (
            "unknown" if cfg["x_detectors"] is None else
            ("x+z" if cfg["x_detectors"] else "z-only"))
    if cfg.get("seed") is None:
        if d.get("kind") == "pakhunov":
            cfg["seed"], prov["seed"] = 7, "inferred: the default seed of the earlier PK script"
        else:
            cfg["seed"], prov["seed"] = None, "unknown"
    else:
        prov["seed"] = "recorded"
    if d.get("schema", 1) < 2:
        prov["created"] = "last-written (older versions re-saved files when displaying them)"
    prov["status"] = reg.get("status", d.get("status", "older file"))
    if reg:
        prov["evidence"] = reg.get("evidence")
    return cfg, prov


@app.function
def result_stem(prefix, code=None, rounds=None, p=None, shots=None, seed=None, flags=None,
                osd=None, convention=None, extra=None, scaled=False):
    """
    The one file-name builder. Tokens in a fixed order: prefix, code tag,
    T{T} or T{min}-{max}x{n}, p{p} or p{min}-{max}x{n}, {shots}shots[-scaled],
    seed{seed}, flags/noflags, osd{o}, the detector-convention token (always),
    then extras. A name without a convention token comes from an older version.
    """
    def rng(vals, fmt):
        vals = list(vals)
        if len(vals) == 1:
            return fmt(vals[0])
        return f"{fmt(min(vals))}-{fmt(max(vals))}x{len(vals)}"

    tok = [prefix]
    if code is not None:
        codes = code if isinstance(code, (list, tuple)) else [code]
        tok.append("+".join(c.strip("[]").replace(", ", "-").replace(" ", "") for c in codes)
                   if len(codes) <= 3 else f"{len(codes)}codes")
    if rounds is not None:
        rs = rounds if isinstance(rounds, (list, tuple)) else [rounds]
        tok.append("T" + rng(rs, lambda v: f"{int(v)}"))
    if p is not None:
        ps = p if isinstance(p, (list, tuple)) else [p]
        tok.append("p" + rng(ps, lambda v: f"{float(v):g}"))
    if shots is not None:
        tok.append(f"{int(shots)}shots" + ("-scaled" if scaled else ""))
    if seed is not None:
        tok.append(f"seed{int(seed)}")
    if flags is not None:
        tok.append("flags" if flags else "noflags")
    if osd is not None:
        os_ = osd if isinstance(osd, (list, tuple)) else [osd]
        tok.append("osd" + rng(os_, lambda v: f"{int(v)}"))
    tok.append({"z-only": "detZ", "x+z": "detXZ", "surface-all": "detSurf",
                "surface-z-only": "detSurfZ", "reference-z-only": "detRef",
                "reference-both": "detRefXZ", "mixed": "detMixed",
                "not applicable": "detNA"}[convention or "z-only"])
    if extra:
        tok.extend(str(e) for e in (extra if isinstance(extra, (list, tuple)) else [extra]))
    return "_".join(tok)


@app.function
def result_file(prefix, ext=".json", **fields):
    return os.path.join(RESULTS_DIR, result_stem(prefix, **fields) + ext)


@app.function
def finalise_run(path, stamp, ckpt_dirs, expect=None):
    """
    After a run's complete write: re-read the file, check run id and the
    expected fields, then delete its checkpoint folders and record that.
    On any mismatch the checkpoints are KEPT and the error propagates.
    """
    d = read_result_file(path)
    if d.get("run_id") != stamp["run_id"] or d.get("status") != "complete":
        raise RuntimeError(f"{path}: written file does not belong to this complete run")
    for k, v in (expect or {}).items():
        if d.get(k) != v and d.get("config", {}).get(k) != v:
            raise RuntimeError(f"{path}: {k} = {d.get(k)!r}, expected {v!r}")
    cleared = []
    for c in ckpt_dirs or []:
        if c and os.path.isdir(c):
            shutil.rmtree(c)
            cleared.append(c)
    d["checkpoint_cleared"] = True
    text = json.dumps(d, indent=1, ensure_ascii=False, default=float).encode("utf-8")
    durable_write(path, lambda f: f.write(text))
    return cleared


@app.function
def save_samples(path, det, obs):
    """Sidecar <stem>_samples.npz with bit-packed samples, so a cited result can
    be re-decoded even where Stim does not reproduce the seed."""
    det = np.asarray(det).astype(np.uint8)
    obs = np.asarray(obs).astype(np.uint8)
    side = os.path.splitext(path)[0] + "_samples.npz"
    buf = io.BytesIO()
    np.savez_compressed(buf, det=np.packbits(det, axis=-1), obs=np.packbits(obs, axis=-1),
                        det_shape=np.array(det.shape), obs_shape=np.array(obs.shape))
    durable_write(side, lambda f: f.write(buf.getvalue()))
    return side


@app.function
def load_samples(path):
    """(det, obs) as uint8 arrays from a <stem>_samples.npz sidecar."""
    side = path if path.endswith("_samples.npz") else os.path.splitext(path)[0] + "_samples.npz"
    with np.load(side) as z:
        dshape = tuple(int(x) for x in z["det_shape"])
        oshape = tuple(int(x) for x in z["obs_shape"])
        det = np.unpackbits(z["det"], axis=-1)[..., :dshape[-1]].reshape(dshape)
        obs = np.unpackbits(z["obs"], axis=-1)[..., :oshape[-1]].reshape(oshape)
    return det, obs


@app.function
def verify_result(path):
    """
    Can this machine reproduce a result's samples? Compares the file's Stim
    self-check with this machine's; with a samples sidecar, checks its hash
    against the recorded one. Returns a short verdict and the details.
    """
    d = read_result_file(path)
    here = stim_selfcheck()
    rec = d.get("stim_selfcheck") or {}
    same = rec.get("digest") == here["digest"]
    out = dict(sampler=("bit-identical" if same else
                        "sampler not reproducible on this machine -- use stored samples"),
               recorded=rec, here=here)
    side = os.path.splitext(path)[0] + "_samples.npz"
    if os.path.exists(side):
        det, obs = load_samples(side)
        out["sidecar_hash"] = sample_hash(det, obs)
        hashes = json.dumps(d.get("samples", {}))
        out["sidecar_matches_record"] = out["sidecar_hash"] in hashes
    return out


@app.function
def metrics_to_json(m, include_samples):
    d = asdict(m)
    samples = d.pop("samples", None)
    if include_samples and samples is not None:
        d["samples"] = {k: np.asarray(v).tolist() for k, v in samples.items()}
    return d


@app.function
def metrics_from_json(d):
    d = dict(d)
    samples = d.pop("samples", None)
    if samples is not None:
        samples = {k: np.asarray(v) for k, v in samples.items()}
    known = set(Metrics.__dataclass_fields__)
    return Metrics(**{k: v for k, v in d.items() if k in known}, samples=samples)


@app.function
def config_payload(config, prov=None, decoders=None):
    """Schema-2 'config' for an ExperimentConfig-driven run."""
    if config.x_detectors is None:
        raise ValueError("refusing to save a run whose detector convention is unknown")
    cfg = asdict(config)
    cfg["detector_convention"] = config.detector_convention
    cfg["decoders"] = decoders if decoders is not None else (prov or {}).get("decoders", {})
    return cfg


@app.function
def save_results(path, config, results, stamp, prov=None, include_samples=False,
                 status="complete"):
    """Write an ablation-style result (dict policy -> Metrics) as schema 2."""
    prov = prov or {}
    return write_result_file(path, {
        "config": config_payload(config, prov),
        "timing": prov.get("timing", dict(workers_requested=config.workers,
                                          workers_effective=None, backends=[],
                                          resumed_chunks=0, total_chunks=0)),
        "samples": prov.get("samples", {}),
        "results": {n: metrics_to_json(m, include_samples) for n, m in results.items()},
    }, stamp, status)


@app.function
def load_results(path):
    """-> (config dict, dict name -> Metrics, meta). Legacy trigger names are
    mapped (`flag` -> `flag>=1`, the byte-identical `flag_or_fail` dropped)."""
    d = read_result_file(path, ("ablation", "L3-ablation"))
    cfg, prov = config_from_file(d, path)
    res, dup = {}, False
    for n, m in d["results"].items():
        if n in LEGACY_TRIGGER_MAP:
            new = LEGACY_TRIGGER_MAP[n]
            if new is None:
                dup = True
                continue
            n = new
        res[n] = metrics_from_json(m)
    meta = dict(provenance=prov, status=prov.get("status"), schema=d.get("schema", 1),
                legacy_duplicate_trigger=dup, run_id=d.get("run_id"))
    return cfg, res, meta


@app.function
def save_sweep(path, config, sweep, stamp, prov=None, include_samples=False, status="complete"):
    """Like save_results, for dict p -> dict policy -> Metrics. The config's p
    is not used by a sweep, so it is stored as None with the swept ps."""
    prov = prov or {}
    cfg = config_payload(config, prov)
    cfg["ps"] = [float(p) for p in sweep]
    cfg["p"] = None
    cfg["p_role"] = "replaced per point"
    return write_result_file(path, {
        "config": cfg,
        "timing": prov.get("timing", dict(workers_requested=config.workers,
                                          workers_effective=None, backends=[],
                                          resumed_chunks=0, total_chunks=0)),
        "samples": prov.get("samples", {}),
        "points": [{"p": float(p), "results": {n: metrics_to_json(m, include_samples)
                                               for n, m in res.items()}}
                   for p, res in sweep.items()],
    }, stamp, status)


@app.function
def load_sweep(path):
    """-> (config dict, dict p -> dict name -> Metrics, meta)."""
    d = read_result_file(path, ("sweep", "L4-sweep"))
    cfg, prov = config_from_file(d, path)
    out = {}
    dup = False
    for pt in d["points"]:
        res = {}
        for n, m in pt["results"].items():
            if n in LEGACY_TRIGGER_MAP:
                if LEGACY_TRIGGER_MAP[n] is None:
                    dup = True
                    continue
                n = LEGACY_TRIGGER_MAP[n]
            res[n] = metrics_from_json(m)
        out[pt["p"]] = res
    return cfg, out, dict(provenance=prov, status=prov.get("status"), schema=d.get("schema", 1),
                          legacy_duplicate_trigger=dup)


@app.function
def export_bundle(data_path, title, figures, tables, notes="", data_meta=None,
                  export_type="fresh run"):
    """
    Write figures (PNG + PDF), tables (CSV, UTF-8) and report.md into a folder
    named after `data_path`, plus export_meta.json. report.md states the data
    file's sha256, the run's start and status, when and by which code it was
    exported, and whether this is a fresh run or a re-export of a saved file.
    """
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, (tuple, list)):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        return str(v)

    folder = os.path.splitext(data_path)[0]
    os.makedirs(folder, exist_ok=True)
    meta = dict(data_meta or {})
    data_sha = file_sha256(data_path) if os.path.exists(data_path) else None
    code = code_identity()
    stamp = dict(data_file=os.path.basename(data_path), data_sha256=data_sha,
                 run_started=meta.get("started"), run_status=meta.get("status"),
                 run_id=meta.get("run_id"),
                 exported=datetime.datetime.now().isoformat(timespec="seconds"),
                 code_sha256=code["sha256"], export_type=export_type)
    lines = [f"# {title}", "",
             f"Data file: `{os.path.basename(data_path)}` (sha256 `{data_sha}`)  ",
             f"Run started: {stamp['run_started']} · status: {stamp['run_status']}  ",
             f"Exported: {stamp['exported']} by code {str(code['sha256'])[:12]}  ",
             f"Export type: {export_type}", ""]
    if notes:
        lines += [notes, ""]
    for name, rows in tables.items():
        path = os.path.join(folder, f"{name}.csv")
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="", encoding="utf-8") as fh:
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
        lines += [f"## Figure: {name}", "", f"![{name}]({name}.png)  ",
                  f"Vector version: [{name}.pdf]({name}.pdf)", ""]
    with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(folder, "export_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, indent=1)
    return folder


@app.function
def metrics_rows(results, **extra):
    """Metrics dict -> CSV rows (per-shot samples left out); `extra` carries the
    configuration columns (code, rounds, p, shots, seed, osd_order, use_flags,
    detector_convention)."""
    rows = []
    for name, m in results.items():
        rows.append(dict(extra, policy=name, shots=m.shots, failures=m.failures, ler=m.ler,
                         ci_low=m.ci_low, ci_high=m.ci_high, escalation_rate=m.escalation_rate,
                         silent_failures=m.silent_failures,
                         unconverged_failures=m.unconverged_failures,
                         escalated_failures=m.escalated_failures, work_mean=m.work_mean,
                         work_p99=m.work_p99, time_p50_us=m.time_p50_us,
                         time_p99_us=m.time_p99_us))
    return rows


@app.function
def read_csv_any(path):
    """Read a CSV as UTF-8, falling back to cp1252 for older files whose '×'
    was written in the Windows platform encoding."""
    try:
        with open(path, encoding="utf-8") as fh:
            return list(csv.DictReader(fh)), "utf-8"
    except UnicodeDecodeError:
        with open(path, encoding="cp1252") as fh:
            return list(csv.DictReader(fh)), "cp1252 (older file)"


@app.function
def saved_runs(prefix, legacy_prefixes=(), ext=".json", legacy_kinds=None):
    """Result files of one experiment: new files `<prefix>_*` plus older
    files whose old prefix maps here, never checkpoints or superseded files."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{prefix}_*{ext}")))
    for lp in legacy_prefixes:
        for f in sorted(glob.glob(os.path.join(RESULTS_DIR, f"{lp}_*{ext}"))):
            if legacy_kinds is not None and ext == ".json":
                try:
                    if read_result_file(f).get("kind") not in legacy_kinds:
                        continue
                except (ValueError, OSError):
                    continue
            files.append(f)
    return [f for f in files if not is_checkpoint_path(f) and "superseded" not in f
            and not f.endswith("_samples.npz")]


@app.function
def saved_run_picker(prefix, label, legacy_prefixes=(), ext=".json", legacy_kinds=None):
    """Dropdown over the saved runs of one experiment, plus the file list."""
    files = saved_runs(prefix, legacy_prefixes, ext, legacy_kinds)
    return mo.ui.dropdown({os.path.basename(f): f for f in files},
                          value=os.path.basename(files[-1]) if files else None,
                          label=label), files


@app.function
def provenance_banner(meta):
    """A one-line warning for loaded files whose provenance is not 'recorded'."""
    prov = (meta or {}).get("provenance", {})
    notes = [f"{k}: {v}" for k, v in prov.items()
             if k not in ("status",) and v not in ("recorded", None)]
    status = (meta or {}).get("status")
    head = f"**Loaded file status: {status}.** " if status and status != "complete" else ""
    if not notes and not head:
        return mo.md("")
    return mo.callout(mo.md(head + " · ".join(notes)), kind="warn")


@app.function
def maybe_export(data, title, figures, tables, notes="", reexport=False):
    """
    Export a FRESH run once; a LOADED file only when its Re-export button was
    pressed (and then labelled "re-export of a saved file"). Never touches the
    data file itself. `data` = dict(path, doc, fresh).
    """
    doc, path = data["doc"], data["path"]
    if data.get("fresh"):
        if doc.get("run_id") in EXPORTED_RUNS:
            return mo.md(f"Exported to `{os.path.splitext(path)[0]}`")
        export_type = "fresh run"
    elif reexport:
        export_type = "re-export of a saved file"
    else:
        return mo.md("*Loaded file: displayed only, nothing rewritten. Press **Re-export** "
                     "to rebuild its export folder.*")
    folder = export_bundle(path, title, figures, tables, notes,
                           data_meta=dict(started=doc.get("started"), status=doc.get("status"),
                                          run_id=doc.get("run_id")),
                           export_type=export_type)
    if data.get("fresh"):
        EXPORTED_RUNS.add(doc.get("run_id"))
    return mo.md(f"Exported ({export_type}) to `{folder}`")


@app.cell
def _():
    def _tmp():
        import tempfile
        return tempfile.mkdtemp()

    def _stamp(kind="test", exp="E00"):
        return start_run(kind, exp)

    def _payload(**over):
        p = dict(config=dict(code="[[72, 12, 6]]", rounds=2, p=1e-3, shots=10, seed=1,
                             detector_convention="z-only", decoders={}),
                 timing=dict(workers_requested=1, workers_effective=1, backends=["sequential"],
                             resumed_chunks=0, total_chunks=1),
                 samples={})
        p.update(over)
        return p

    def _results_dir_absolute():
        assert os.path.isabs(RESULTS_DIR)
        assert os.path.dirname(RESULTS_DIR) == NOTEBOOK_DIR, "results must live next to the notebook"

    def _bundle_written():
        data = os.path.join(_tmp(), "E1_demo.json")
        open(data, "w").write("{}")
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        m = Metrics(100, 3, 0.03, 0.01, 0.08, 0.4, 2.0, 5.0, 10.0, 50.0, 1, 0)
        folder = export_bundle(data, "demo", {"curve": fig}, {"metrics": metrics_rows({"always": m})},
                               data_meta=dict(started="t0", status="complete"))
        assert folder == os.path.splitext(data)[0]
        for f in ("curve.png", "curve.pdf", "metrics.csv", "report.md", "export_meta.json"):
            assert os.path.getsize(os.path.join(folder, f)) > 0, f"{f} missing or empty"
        rows = list(csv.DictReader(open(os.path.join(folder, "metrics.csv"), encoding="utf-8")))
        assert len(rows) == 1 and rows[0]["policy"] == "always" and rows[0]["failures"] == "3"
        report = open(os.path.join(folder, "report.md"), encoding="utf-8").read()
        assert "curve.png" in report and "Export type: fresh run" in report
        em = json.load(open(os.path.join(folder, "export_meta.json"), encoding="utf-8"))
        assert em["data_sha256"] == file_sha256(data) and em["data_sha256"] in report

    def _figures_still_display():
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        data = os.path.join(_tmp(), "x.json")
        open(data, "w").write("{}")
        export_bundle(data, "x", {"f": fig}, {})
        assert plt.fignum_exists(fig.number), "export closed the figure, so it will not display"
        fig.savefig(io.BytesIO(), format="png")

    def _utf8_everywhere():
        """'×' is written as UTF-8 and read back; legacy cp1252 still reads."""
        folder = _tmp()
        data = os.path.join(folder, "u.json")
        open(data, "w").write("{}")
        export_bundle(data, "u", {}, {"t": [{"gain": "1.4×"}]})
        raw = open(os.path.join(folder, "u", "t.csv"), "rb").read()
        assert "1.4×".encode("utf-8") in raw and b"\xd7" not in raw.replace("×".encode(), b"")
        legacy = os.path.join(folder, "legacy.csv")
        open(legacy, "wb").write(b"gain\n1.4\xd7\n")
        rows, enc = read_csv_any(legacy)
        assert rows[0]["gain"] == "1.4×" and enc.startswith("cp1252")

    def _schema2_written_and_validated():
        p = os.path.join(_tmp(), "E00_x.json")
        st_ = _stamp()
        write_result_file(p, _payload(), st_, status="partial")
        d1 = read_result_file(p)
        assert d1["status"] == "partial" and d1["finished"] is None and d1["schema"] == 2
        time.sleep(1.1)
        write_result_file(p, _payload(), st_, status="complete")
        d2 = read_result_file(p)
        assert d2["run_id"] == d1["run_id"] and d2["started"] == d1["started"]
        assert d2["updated"] > d1["updated"] and d2["finished"]
        for k in ("software", "environment", "code", "stim_selfcheck"):
            assert d2[k], k
        assert d2["software"]["numba"] is not None or not HAVE_NUMBA

    def _missing_seed_is_refused():
        p = os.path.join(_tmp(), "E00_bad.json")
        bad = _payload(config=dict(code="c", rounds=1, p=1e-3, shots=1, seed=None,
                                   detector_convention="z-only", decoders={}))
        try:
            write_result_file(p, bad, _stamp())
        except ValueError as exc:
            assert "seed" in str(exc) and not os.path.exists(p)
        else:
            raise AssertionError("a result without a seed was written")
        worse = _payload(config=dict(code="c", rounds=1, p=1e-3, shots=1, seed=1,
                                     detector_convention="maybe", decoders={}))
        try:
            write_result_file(p, worse, _stamp())
        except ValueError:
            return
        raise AssertionError("an unknown detector convention was accepted")

    def _other_run_is_superseded():
        """run B to A's file moves A to superseded/ and records it."""
        p = os.path.join(_tmp(), "E00_same.json")
        a, b = _stamp(), _stamp()
        write_result_file(p, _payload(), a)
        write_result_file(p, _payload(), b)
        d = read_result_file(p)
        assert d["run_id"] == b["run_id"] and d["supersedes"]["run_id"] == a["run_id"]
        sup = os.path.join(os.path.dirname(p), "superseded")
        moved = os.listdir(sup)
        assert len(moved) == 1
        assert json.load(open(os.path.join(sup, moved[0]), encoding="utf-8"))["run_id"] == a["run_id"]

    def _roundtrip():
        cfg = ExperimentConfig(p=2e-3, shots=10)
        res = {"always": Metrics(10, 1, 0.1, 0.02, 0.4, 1.0, 5.0, 9.0, 100.0, 200.0)}
        path = os.path.join(_tmp(), "r.json")
        save_results(path, cfg, res, _stamp("ablation", "L3"))
        cfg2, res2, meta = load_results(path)
        assert ExperimentConfig(**{k: cfg2[k] for k in asdict(cfg)}) == cfg and res2 == res
        assert meta["status"] == "complete"

    def _samples_roundtrip():
        cfg = ExperimentConfig(shots=3)
        smp = dict(time_us=np.array([1.5, 2.5, 3.5]), failed=np.array([True, False, False]))
        res = {"a": Metrics(3, 1, 1 / 3, 0.06, 0.79, 0.0, 1.0, 1.0, 2.5, 3.5, 1, 0, samples=smp)}
        p = os.path.join(_tmp(), "nested", "s.json")
        save_results(p, cfg, res, _stamp("ablation", "L3"))
        assert load_results(p)[1]["a"].samples is None, "samples should be dropped by default"
        st2 = _stamp("ablation", "L3")
        save_results(p, cfg, res, st2, include_samples=True)
        back = load_results(p)[1]["a"]
        assert np.array_equal(back.samples["time_us"], smp["time_us"])
        assert back.samples["failed"].dtype == bool and back == res["a"]

    def _sweep_roundtrip():
        cfg = ExperimentConfig()
        m = Metrics(10, 1, 0.1, 0.02, 0.4, 1.0, 5.0, 9.0, 100.0, 200.0)
        sweep = {5e-4: {"always": m}, 1.2345678901234567e-3: {"always": m, "never": m}}
        p = os.path.join(_tmp(), "sw.json")
        save_sweep(p, cfg, sweep, _stamp("sweep", "L4"))
        cfg2, sweep2, _meta = load_sweep(p)
        assert sweep2 == sweep and cfg2["p"] is None
        assert cfg2["ps"] == [5e-4, 1.2345678901234567e-3]

    def _kind_checked():
        p = os.path.join(_tmp(), "k.json")
        save_sweep(p, ExperimentConfig(), {1e-3: {}}, _stamp("sweep", "L4"))
        try:
            load_results(p)
        except ValueError:
            return
        raise AssertionError("load_results accepted a sweep file")

    def _legacy_missing_field_is_unknown():
        """A schema-1 ablation without x_detectors loads as unknown;
        LEGACY_PROVENANCE labels the files known to have run with X detectors."""
        folder = _tmp()
        legacy = dict(kind="ablation", schema=1, config=dict(code="[[72, 12, 6]]", rounds=2, p=1e-3,
                                                           shots=10, seed=3, use_flags=True,
                                                           osd_order=0, workers=1),
                      results={"flag": asdict(Metrics(10, 1, 0.1, 0.0, 0.4, 0.5, 1, 1, 1, 1)),
                               "flag_or_fail": asdict(Metrics(10, 1, 0.1, 0.0, 0.4, 0.5, 1, 1, 1, 1))})
        for k in legacy["results"].values():
            k.pop("samples", None)
        p = os.path.join(folder, "EXP4_unknown.json")
        json.dump(legacy, open(p, "w", encoding="utf-8"))
        cfg, res, meta = load_results(p)
        assert cfg["x_detectors"] is None and cfg["detector_convention"] == "unknown"
        assert set(res) == {"flag>=1"} and meta["legacy_duplicate_trigger"]
        reg = os.path.join(folder, "EXP4_72-12-6_T6_p0.001_25000shots_seed64260921_flags_osd0.json")
        json.dump(legacy, open(reg, "w", encoding="utf-8"))
        cfg, _res, meta = load_results(reg)
        assert cfg["x_detectors"] is True
        assert meta["provenance"]["x_detectors"].startswith("from LEGACY_PROVENANCE")
        try:
            save_results(os.path.join(folder, "x.json"), ExperimentConfig(x_detectors=None), {},
                         _stamp("ablation", "L3"))
        except ValueError:
            return
        raise AssertionError("a run with an unknown detector convention was saved")

    def _registry_entries_state_evidence():
        for name, entry in LEGACY_PROVENANCE.items():
            assert len(entry.get("evidence", "")) > 10 and entry.get("status"), name

    def _names_are_deterministic_and_distinct():
        """File names are deterministic, complete and distinct."""
        a = result_stem("E08", code="[[72, 12, 6]]", rounds=12, p=1e-3, shots=50000, seed=7,
                        osd=0, convention="z-only", extra="arms-blind-sighted-updated-unflagged")
        assert a == result_stem("E08", code="[[72, 12, 6]]", rounds=12, p=1e-3, shots=50000,
                                seed=7, osd=0, convention="z-only",
                                extra="arms-blind-sighted-updated-unflagged")
        assert a.endswith("_detZ_arms-blind-sighted-updated-unflagged") and "seed7" in a
        assert result_stem("E03", code="[[72, 12, 6]]", rounds=[3, 6, 12], p=1e-3, shots=200000,
                           seed=1) != result_stem("E03", code="[[72, 12, 6]]", rounds=[3, 6, 12],
                                                  p=1e-3, shots=200000, seed=2)
        sw = result_stem("L4", code="[[72, 12, 6]]", rounds=6, p=[5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 6e-3],
                         shots=500, seed=1)
        assert "_p0.0005-0.006x6_" in sw and "_p0.001_" not in sw
        assert result_stem("E06", rounds=6, convention="surface-z-only").endswith("detSurfZ")

    def _rounds_normalised():
        assert normalise_rounds([12]) == normalise_rounds(12) == (12, [12])
        assert normalise_rounds([3, 6, 12]) == (12, [3, 6, 12])

    def _checkpoint_paths_refused():
        p = os.path.join(_tmp(), "checkpoints", "a.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write("{}")
        try:
            read_result_file(p)
        except ValueError:
            return
        raise AssertionError("a file inside checkpoints/ was read as a result")

    def _finalise_clears_only_after_verifying():
        folder = _tmp()
        p = os.path.join(folder, "E00_fin.json")
        ck = os.path.join(folder, "checkpoints", "E00_fin")
        os.makedirs(ck)
        st_ = _stamp()
        write_result_file(p, _payload(), st_)
        assert finalise_run(p, st_, [ck]) == [ck] and not os.path.isdir(ck)
        assert read_result_file(p)["checkpoint_cleared"] is True
        ck2 = os.path.join(folder, "checkpoints", "keep")
        os.makedirs(ck2)
        try:
            finalise_run(p, _stamp(), [ck2])
        except RuntimeError:
            assert os.path.isdir(ck2)
            return
        raise AssertionError("checkpoints of another run were deleted")

    def _samples_sidecar_roundtrip():
        folder = _tmp()
        p = os.path.join(folder, "E00_s.json")
        det = np.random.default_rng(1).integers(0, 2, (37, 21)).astype(bool)
        obs = np.random.default_rng(2).integers(0, 2, (37, 3)).astype(bool)
        save_samples(p, det, obs)
        d2, o2 = load_samples(p)
        assert np.array_equal(d2, det.astype(np.uint8)) and np.array_equal(o2, obs.astype(np.uint8))
        assert sample_hash(d2, o2) == sample_hash(det, obs)

    tests_persistence = run_checks([
        ("results folder is absolute and next to the notebook", _results_dir_absolute),
        ("export writes PNG, PDF, CSV, report.md and export_meta.json", _bundle_written),
        ("figures still display after being exported", _figures_still_display),
        ("every text file is UTF-8; older cp1252 CSVs still read", _utf8_everywhere),
        ("schema-2 files: status, run id, times, versions, machine, code hash", _schema2_written_and_validated),
        ("a result without a seed or with an unknown convention is refused", _missing_seed_is_refused),
        ("writing another run to a file supersedes it, never overwrites", _other_run_is_superseded),
        ("save_results / load_results round-trip exactly", _roundtrip),
        ("per-shot samples are optional and round-trip exactly", _samples_roundtrip),
        ("save_sweep stores the swept ps and no unused p", _sweep_roundtrip),
        ("loading the wrong kind of file raises ValueError", _kind_checked),
        ("fields an older file did not record load as unknown", _legacy_missing_field_is_unknown),
        ("every LEGACY_PROVENANCE entry states its evidence", _registry_entries_state_evidence),
        ("file names are deterministic, complete and distinct", _names_are_deterministic_and_distinct),
        ("rounds are normalised for older and new files", _rounds_normalised),
        ("checkpoint paths are never read as results", _checkpoint_paths_refused),
        ("checkpoints are deleted only after the result verifies", _finalise_clears_only_after_verifying),
        ("packed sample sidecars round-trip bit for bit", _samples_sidecar_roundtrip),
    ])
    render_checks("12 result files and provenance", tests_persistence)
    return (tests_persistence,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 13. Policies, costs and verdicts

    Every policy number in the thesis comes from **one function**, `derive`, run
    on the per-shot results of a primary ("weak") and a secondary ("strong")
    decoder that decoded the *same* shots. A policy is a pair (pre, fallback):

    * `pre` — per shot, escalate **before** the primary runs (a flag rule);
    * `fallback` — escalate after the primary: `True` = when the primary does not
      converge, `False` = never, or a per-shot mask (the oracle uses the
      primary's actual failures, which no real decoder can know).

    **Cost convention.** A pre-escalated shot skips the primary, so it pays only
    the secondary; every other shot pays the primary, plus the secondary when it
    escalates. Every time column states this convention (`cost_convention`). The
    alternative that charges the primary on every shot is available as
    `policy_table(..., convention="charge-primary")` for comparison only.

    **Verdicts** for a (primary, secondary) pair, in this order:

    1. *inconclusive* — `primary_fail` fails on fewer than `min_failures` shots;
    2. *viable: accuracy* — some rule fails significantly less than
       `primary_fail` (exact paired test, Bonferroni over the
       m = primaries × secondaries × (rules − 1) comparisons);
    3. *not viable (impossible: no headroom)* — the primary has no silent failure
       that the secondary corrects, so no rule can gain accuracy;
    4. *not viable*.

    A pair whose secondary re-runs the primary's own BP (BP-ms-k inside BP+OSD
    with ≥ k iterations, same settings) is **impossible by construction** on
    accuracy, and its cost is **not attributable**: the code does not reuse the
    BP run, so the second BP is charged twice.

    *Viable: cost* is judged separately and needs all of: no accuracy loss
    against `primary_fail` (no harmed shot, no net loss); a mean saving of at
    least `min_saving` against `primary_fail` **and** against `always` (unless
    `always` is significantly less accurate than the rule); and the
    family-corrected upper bound of the per-shot cost difference, from a
    chunk-block standard error over 250-shot blocks, below zero for each
    reference.

    A flag rule is **flag-informative** only if it also beats a *placebo* — the
    same number of shots, chosen at random, escalated with the same fallback.
    Without this check a rule that escalates many shots looks useful for reasons
    that have nothing to do with the flags.
    """)
    return


@app.function
def flag_features(det, flag_mask, mx, rounds):
    """Per shot: number of flag bits fired, and number of rounds with any flag."""
    bits = np.asarray(det)[:, np.asarray(flag_mask, bool)].astype(bool)
    if bits.shape[1] == 0:
        z = np.zeros(len(bits), int)
        return z, z
    return bits.sum(1), bits.reshape(len(bits), rounds, -1).any(2).sum(1)


@app.function
def placebo_mask(n, k, seed):
    """k shots out of n, chosen at random (seeded), as a boolean mask."""
    m = np.zeros(n, bool)
    if k > 0:
        m[np.random.default_rng(seed).choice(n, size=int(k), replace=False)] = True
    return m


@app.function
def trigger_rules(n_flags, flag_rounds, placebo_seed=None):
    """
    name -> (pre, fallback). `pre` escalates before the primary runs;
    `fallback` True also escalates when the primary fails to converge.
    With `placebo_seed`, every flag rule gets a partner "placebo:<rule>" that
    escalates the same NUMBER of shots, chosen at random, with the same fallback.
    """
    n_flags = np.asarray(n_flags)
    flag_rounds = np.asarray(flag_rounds)
    none = np.zeros(len(n_flags), bool)
    rules = {"primary_fail": (none, True)}
    for k in (1, 2, 3, 4):
        rules[f"flag>={k}"] = (n_flags >= k, True)
    for k in (2, 3):
        rules[f"rounds>={k}"] = (flag_rounds >= k, True)
    rules["flag-only"] = (n_flags >= 1, False)
    if placebo_seed is not None:
        for i, (name, (pre, fb)) in enumerate(list(rules.items())):
            if name == "primary_fail":
                continue
            rules[f"placebo:{name}"] = (placebo_mask(len(pre), int(pre.sum()),
                                                     [int(placebo_seed), i]), fb)
    return rules


@app.function
def derive(weak, strong, pre, fallback, convention="derive"):
    """
    One policy's per-shot outcome from its primary (`weak`) and secondary
    (`strong`) results on the same shots.

    pre      : bool or per-shot bool array -- escalate before the primary runs.
    fallback : True (escalate when the primary does not converge), False, or a
               per-shot bool array (escalate exactly there; the oracle passes
               the primary's failures).
    convention "derive": a pre-escalated shot does not pay the primary.
    "charge-primary" (comparison only) charges the primary always.
    """
    n = len(weak["fail"])
    pre = np.broadcast_to(np.asarray(pre, bool), (n,))
    if isinstance(fallback, (bool, np.bool_)):
        post = ~weak["conv"].astype(bool) if fallback else np.zeros(n, bool)
    else:
        post = np.broadcast_to(np.asarray(fallback, bool), (n,))
    esc = pre | (~pre & post)
    fail = np.where(esc, strong["fail"], weak["fail"]).astype(bool)
    wt = weak["time_us"].astype(float)
    stt = strong["time_us"].astype(float)
    if convention == "derive":
        cost = np.where(pre, 0.0, wt) + np.where(esc, stt, 0.0)
        work = (np.where(pre, 0, weak["work"].astype(np.int64))
                + np.where(esc, strong["work"].astype(np.int64), 0))
    elif convention == "charge-primary":
        cost = wt + np.where(esc, stt, 0.0)
        work = weak["work"].astype(np.int64) + np.where(esc, strong["work"].astype(np.int64), 0)
    else:
        raise ValueError(f"convention must be 'derive' or 'charge-primary', got {convention!r}")
    conv = weak["conv"].astype(bool)
    return dict(esc=esc, fail=fail, cost=cost, work=work,
                silent=~esc & conv & weak["fail"].astype(bool),
                unconverged=~esc & ~conv & weak["fail"].astype(bool))


@app.function
def oracle_policy(weak, strong):
    """Escalate exactly the shots the primary gets wrong: unimplementable, a bound."""
    return derive(weak, strong, False, weak["fail"].astype(bool))


@app.function
def reference_policies(weak, strong, n_flags):
    """The six reference policies of Table 5.2, as name -> (pre, fallback)."""
    n = len(weak["fail"])
    flagged = np.asarray(n_flags) > 0
    zeros, ones = np.zeros(n, bool), np.ones(n, bool)
    return {"never": (zeros, False), "flag-only": (flagged, False), "flag>=1": (flagged, True),
            "primary_fail": (zeros, True), "always": (ones, False),
            "oracle": (zeros, weak["fail"].astype(bool))}


@app.function
def policy_table(results, weak, strong, n_flags, workers=None, convention="derive"):
    """
    One row per reference policy: escalation, failures, LER with Wilson
    interval, mean / median / p99 cost, worker count and cost convention.
    `workers` is an int, or a dict decoder -> worker count; results timed with
    different worker counts are refused (their times are not comparable).
    """
    if isinstance(workers, dict):
        ws = {workers.get(weak), workers.get(strong)}
        if len(ws) > 1:
            raise ValueError(f"{weak} and {strong} were timed with different worker counts "
                             f"{sorted(map(str, ws))}; their times cannot be combined")
        workers = ws.pop()
    W, S = results[weak], results[strong]
    n = len(W["fail"])
    rows = []
    for name, (pre, fb) in reference_policies(W, S, n_flags).items():
        P = derive(W, S, pre, fb, convention)
        k = int(P["fail"].sum())
        ler, lo, hi = wilson(k, n)
        rows.append(dict(policy=name, weak=weak, strong=strong, shots=n,
                         escalation=float(P["esc"].mean()), failures=k, ler=ler,
                         ci_low=lo, ci_high=hi, cost_mean_us=float(P["cost"].mean()),
                         cost_median_us=float(np.median(P["cost"])),
                         cost_p99_us=float(np.percentile(P["cost"], 99)),
                         workers=workers, cost_convention=COST_CONVENTIONS[convention]))
    return rows


@app.function
def breakeven_ratio(esc_rule, pre, esc_pf):
    """
    A fallback rule is cheaper than primary_fail iff  c_weak / c_strong > r*,
    with r* = (P(escalate | rule) - P(escalate | primary_fail)) / P(pre) and
    the cost ratio measured as cost_ratio_cond. Hardware-independent.
    """
    p_pre = float(np.mean(pre))
    if p_pre == 0:
        return float("inf")
    return (float(np.mean(esc_rule)) - float(np.mean(esc_pf))) / p_pre


@app.function
def cost_ratio_cond(weak, strong, pre, esc_rule, esc_pf):
    """
    The cost ratio that the break-even compares against: mean primary time on
    the pre-escalated shots over mean secondary time on the shots the rule
    escalates and primary_fail does not. For a fallback rule,
    cost_ratio_cond > breakeven  <=>  the rule is cheaper than primary_fail.
    """
    pre = np.asarray(pre, bool)
    extra = np.asarray(esc_rule, bool) & ~np.asarray(esc_pf, bool)
    if not pre.any():
        return float("nan")
    if not extra.any():
        return float("inf")
    return float(weak["time_us"][pre].astype(float).mean()
                 / max(strong["time_us"][extra].astype(float).mean(), 1e-12))


@app.function
def nested_bp(weak_spec, strong_spec_):
    """
    Does the secondary re-run the primary's BP? True when the primary is
    BP-k and the secondary is BP+OSD/LSD with the same BP method, schedule
    and scaling and at least k iterations: then on every shot the primary
    converges, the secondary returns the same correction.
    """
    try:
        w = decoder_spec_full(weak_spec)
        s = decoder_spec_full(strong_spec_)
    except (KeyError, ValueError):
        return False
    if w["kind"] != "bp" or s["kind"] not in ("osd", "lsd"):
        return False
    return (w["bp_method"] == s["bp_method"] and w["schedule"] == s["schedule"]
            and w.get("ms_scaling_factor") == s.get("ms_scaling_factor")
            and w["max_iter"] <= s["max_iter"])


@app.function
def nested_mismatch(weak, strong):
    """Shots where the primary converged but the secondary's outcome differs.
    Zero for a genuinely nested pair."""
    conv = weak["conv"].astype(bool)
    return int((conv & (weak["fail"].astype(bool) != strong["fail"].astype(bool))).sum())


@app.function
def chunk_block_bound(diff, chunk_size=CHUNK_SIZE, z=1.96):
    """
    (mean, se, upper) of a per-shot difference, with the standard error from
    the means of consecutive `chunk_size` blocks (so correlated timings inside
    a chunk do not shrink it). Fewer than two blocks gives se = inf.
    """
    diff = np.asarray(diff, float)
    n = len(diff)
    mean = float(diff.mean()) if n else float("nan")
    nb = n // chunk_size
    if nb < 2:
        return mean, float("inf"), float("inf")
    blocks = diff[:nb * chunk_size].reshape(nb, chunk_size).mean(1)
    se = float(blocks.std(ddof=1) / np.sqrt(nb))
    return mean, se, mean + z * se


@app.function
def chunk_start_mask(n, chunk_size=CHUNK_SIZE):
    """True at the first shot of every chunk (excluded from timing summaries of
    runs made before the untimed warm-up, whose chunk starts carry a decoder
    build)."""
    m = np.zeros(n, bool)
    m[::chunk_size] = True
    return m


@app.function
def trimmed_mean(x, cut=0.01):
    """Mean after dropping the top and bottom `cut` fraction."""
    return float(st.trim_mean(np.asarray(x, float), cut))


@app.function
def strong_ceiling(results, strong_names):
    """
    For each ordered pair of secondaries: how many shots A gets wrong that B
    gets right. Headroom left in the SECONDARY: if a much stronger decoder
    fixes almost none of A's failures, no reweighting of A's priors can help.
    """
    out = []
    for a in strong_names:
        for b in strong_names:
            if a == b:
                continue
            fa, fb = results[a]["fail"].astype(bool), results[b]["fail"].astype(bool)
            out.append(dict(decoder=a, compared_with=b, failures=int(fa.sum()),
                            fixed_by_other=int((fa & ~fb).sum()),
                            broken_by_other=int((~fa & fb).sum())))
    return out


@app.function
def flag_diagnostics(results, weak_names, strong_names, n_flags):
    """
    Mutual information (bits) between 'a flag fired' and 'this decoder was
    wrong', with the risk ratio and share of failures on flagged shots. Near
    zero means the flag carries no usable signal about this decoder's errors.
    """
    flagged = np.asarray(n_flags) > 0

    def mutual_information(x, y):
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
        wrong = results[name]["fail"].astype(bool)
        rate_f = float(wrong[flagged].mean()) if flagged.any() else float("nan")
        rate_u = float(wrong[~flagged].mean()) if (~flagged).any() else float("nan")
        out.append(dict(decoder=name, flagged_fraction=float(flagged.mean()),
                        error_rate_flagged=rate_f, error_rate_unflagged=rate_u,
                        risk_ratio=(rate_f / rate_u) if rate_u > 0 else float("inf"),
                        mutual_information_bits=mutual_information(flagged, wrong),
                        share_of_failures_on_flagged=(float((wrong & flagged).sum() / wrong.sum())
                                                      if wrong.any() else float("nan"))))
    return out


@app.function
def analyse(results, weak_names, strong_names, n_flags, flag_rounds, alpha=0.05,
            min_failures=20, min_saving=0.05, workers=None, chunk_size=CHUNK_SIZE,
            placebo_seed=20260921, specs=None):
    """
    Evaluate every (primary, secondary, rule). Returns dict with
      rows   : one per (primary, secondary, rule), incl. placebo comparisons
      pairs  : one verdict per (primary, secondary) -- see §13 for the order
      weak   : silent-failure summary per primary
      m      : the Bonferroni family size, primaries x secondaries x (rules - 1)
    `specs` maps names to decoder specs (default: the catalogue) for the
    nested-BP check; `workers` is recorded in every row and pair.
    """
    specs = dict(CATALOGUE, **(specs or {}))
    rules = trigger_rules(n_flags, flag_rounds, placebo_seed=placebo_seed)
    real = [r for r in rules if not r.startswith("placebo:")]
    shots = len(n_flags)
    m = max(1, len(weak_names) * len(strong_names) * (len(real) - 1))
    alpha_m = alpha / m
    z = float(st.norm.ppf(1 - alpha / (2 * m)))
    n_flags = np.asarray(n_flags)
    rows, pairs, weak_summary = [], [], {}
    for w in weak_names:
        W = results[w]
        conv, fail = W["conv"].astype(bool), W["fail"].astype(bool)
        n_conv, n_sil = int(conv.sum()), int((conv & fail).sum())
        weak_summary[w] = dict(converged=n_conv, silent=n_sil,
                               silent_rate=wilson(n_sil, n_conv),
                               silent_flagged=int((conv & fail & (n_flags > 0)).sum()),
                               time_us=float(np.median(W["time_us"])), workers=workers)
    ones = np.ones(shots, bool)
    zeros = np.zeros(shots, bool)
    for w in weak_names:
        W = results[w]
        wconv, wfail = W["conv"].astype(bool), W["fail"].astype(bool)
        for s in strong_names:
            S = results[s]
            sfail = S["fail"].astype(bool)
            nested = nested_bp(specs.get(w, w), specs.get(s, s))
            mismatch = nested_mismatch(W, S) if nested else None
            if nested and mismatch:
                raise ValueError(f"{w} inside {s} should be nested but {mismatch} converged "
                                 "shots differ: the BP settings are not shared")
            pf = derive(W, S, zeros, True)
            al = derive(W, S, ones, False)
            pf_cost, al_cost = float(pf["cost"].mean()), float(al["cost"].mean())
            headroom = int((wconv & wfail & ~sfail).sum())
            best_acc, best_cost = None, None
            derived = {r: derive(W, S, *rules[r]) for r in rules}
            for rname in real:
                pre, fb = rules[rname]
                P = derived[rname]
                x, y, pval = paired_exact(P["fail"], pf["fail"])
                k = int(P["fail"].sum())
                row = dict(weak=w, strong=s, rule=rname, failures=k, ler=wilson(k, shots),
                           escalation=float(P["esc"].mean()), pre_rate=float(np.mean(pre)),
                           silent=int(P["silent"].sum()), unconverged=int(P["unconverged"].sum()),
                           catch=int((pre & wconv & wfail & ~sfail).sum()),
                           harm=int((pre & wconv & ~wfail & sfail).sum()),
                           worse=x, better=y, p=pval, p_adj=min(1.0, pval * m),
                           cost_us=float(P["cost"].mean()), pf_cost_us=pf_cost,
                           always_cost_us=al_cost, saving=0.0, saving_vs_always=0.0,
                           saving_ci=(float("nan"), float("nan")),
                           breakeven=float("nan"), cost_ratio_cond=float("nan"),
                           flag_informative=None, workers=workers, nested=nested)
                rows.append(row)
                if rname == "primary_fail":
                    continue
                if fb is True:
                    row["breakeven"] = breakeven_ratio(P["esc"], pre, pf["esc"])
                    row["cost_ratio_cond"] = cost_ratio_cond(W, S, pre, P["esc"], pf["esc"])
                # placebo: same number of pre-escalated shots, at random
                Q = derived[f"placebo:{rname}"]
                px, py, pp = paired_exact(P["fail"], Q["fail"])
                row.update(placebo_failures=int(Q["fail"].sum()), placebo_worse=px,
                           placebo_better=py, placebo_p_adj=min(1.0, pp * m),
                           flag_informative=bool(py > px and pp * m < alpha))
                # accuracy against primary_fail
                if y > x and row["p_adj"] < alpha and not nested and \
                        (best_acc is None or y - x > best_acc["better"] - best_acc["worse"]):
                    best_acc = row
                # cost, judged separately
                row["saving"] = 1 - row["cost_us"] / pf_cost if pf_cost > 0 else 0.0
                row["saving_vs_always"] = 1 - row["cost_us"] / al_cost if al_cost > 0 else 0.0
                ax, ay, ap = paired_exact(al["fail"], P["fail"])
                always_worse = ax < ay and ap * m < alpha     # always fails significantly more
                _, _, up_pf = chunk_block_bound(P["cost"] - pf["cost"], chunk_size, z)
                _, se_pf, _ = chunk_block_bound(P["cost"] - pf["cost"], chunk_size, z)
                _, _, up_al = chunk_block_bound(P["cost"] - al["cost"], chunk_size, z)
                mean_d = row["cost_us"] - pf_cost
                row["saving_ci"] = ((-(mean_d + z * se_pf) / pf_cost,
                                     -(mean_d - z * se_pf) / pf_cost) if pf_cost > 0
                                    else (float("nan"), float("nan")))
                no_loss = x <= y and row["harm"] == 0
                ok_pf = row["saving"] >= min_saving and up_pf < 0
                ok_al = always_worse or (row["saving_vs_always"] >= min_saving and up_al < 0)
                row["cost_viable"] = bool(no_loss and ok_pf and ok_al and not nested)
                if row["cost_viable"] and (best_cost is None or row["cost_us"] < best_cost["cost_us"]):
                    best_cost = row
            pf_fail = int(pf["fail"].sum())
            if nested:
                verdict_accuracy = "impossible by construction: strong decoder re-runs the weak BP"
                verdict_cost = "not attributable"
            else:
                if pf_fail < min_failures:
                    verdict_accuracy = "inconclusive"
                elif best_acc:
                    verdict_accuracy = "viable: accuracy"
                elif headroom == 0:
                    verdict_accuracy = "not viable (impossible: no headroom)"
                else:
                    verdict_accuracy = "not viable"
                verdict_cost = "viable: cost" if best_cost else "not viable"
            verdict = ("viable: accuracy" if verdict_accuracy == "viable: accuracy" else
                       "viable: cost" if verdict_cost == "viable: cost" else verdict_accuracy)
            pairs.append(dict(weak=w, strong=s, verdict=verdict,
                              verdict_accuracy=verdict_accuracy, verdict_cost=verdict_cost,
                              nested=nested, nested_mismatch=mismatch, headroom=headroom,
                              m=m, alpha_family=alpha_m,
                              min_detectable_discordant=min_detectable_discordant(alpha_m),
                              pf_failures=pf_fail, pf_escalation=float(pf["esc"].mean()),
                              pf_cost_us=pf_cost, always_failures=int(sfail.sum()),
                              always_cost_us=al_cost,
                              best_accuracy_rule=best_acc["rule"] if best_acc else None,
                              best_cost_rule=best_cost["rule"] if best_cost else None,
                              best_saving=best_cost["saving"] if best_cost else 0.0,
                              best_saving_vs_always=(best_cost["saving_vs_always"]
                                                     if best_cost else 0.0),
                              median_time_ratio=float(np.median(W["time_us"])
                                                      / max(np.median(S["time_us"]), 1e-9)),
                              workers=workers))
    return dict(rows=rows, pairs=pairs, weak=weak_summary, shots=shots, m=m,
                strong_ceiling=strong_ceiling(results, strong_names),
                flag_diagnostics=flag_diagnostics(results, weak_names, strong_names, n_flags))


@app.function
def save_zoo(path, results, n_flags, flag_rounds, meta):
    """Per-shot decoder-grid results as a compressed npz; meta stored as JSON."""
    arrays = {f"{n}__{k}": v for n, r in results.items() for k, v in r.items()}
    buf = io.BytesIO()
    np.savez_compressed(buf, n_flags=n_flags, flag_rounds=flag_rounds,
                        meta=np.array(json.dumps(meta, default=str)), **arrays)
    durable_write(path, lambda f: f.write(buf.getvalue()))
    return path


@app.function
def load_zoo(path):
    """-> (results, n_flags, flag_rounds, meta dict). Older files stored the
    meta as a Python repr; both are read."""
    if is_checkpoint_path(path):
        raise ValueError(f"{path} is inside a checkpoint folder")
    with np.load(path, allow_pickle=False) as d:
        results = {}
        for key in d.files:
            if "__" in key:
                n, k = key.split("__", 1)
                results.setdefault(n, {})[k] = d[key]
        raw = str(d["meta"])
        n_flags, flag_rounds = d["n_flags"], d["flag_rounds"]
    try:
        meta = json.loads(raw)
    except ValueError:
        try:
            meta = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            meta = {"raw": raw}
    return results, n_flags, flag_rounds, meta


@app.cell
def _():
    _code = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
    _T = 3
    _circ = build_memory_circuit(_code, _T, 2e-3, use_flags=True, x_detectors=False)
    _H, _L, _pr = dem_to_matrices(_circ.detector_error_model(decompose_errors=False))
    _mask = flag_detector_mask(_code, _T, True, _circ.num_detectors, x_detectors=False)
    _det, _obs = _circ.compile_detector_sampler(seed=17).sample(300, separate_observables=True)
    _det, _obs = _det.astype(np.uint8), _obs.astype(np.uint8)
    _names = ["peel", "BP-ms-10", "OSD-0", "LSD-0"]
    _state = {}

    def _zoo():
        if "res" not in _state:
            _state["res"], _state["backend"] = run_zoo(
                _H, _L, _pr, _det[:90], _obs[:90], _names, workers=2, chunk_size=30,
                checkpoint_dir=False)
        return _state["res"]

    def _features(n=90):
        return flag_features(_det[:n], _mask, _code.hx.shape[0], _T)

    def _toy(n=8):
        W = dict(conv=np.ones(n, bool), fail=np.zeros(n, bool), work=np.ones(n, int),
                 time_us=np.full(n, 10.0))
        S = dict(conv=np.ones(n, bool), fail=np.zeros(n, bool), work=np.full(n, 5),
                 time_us=np.full(n, 100.0))
        return W, S

    def _zoo_decoders_valid():
        for name in ("BP-ms-10", "OSD-0", "LSD-CS2"):
            dec = make_decoder(name, _H, _pr)
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
        det = _det[:90]
        W, S = _Adapt(make_decoder("peel", _H, _pr)), _Adapt(make_decoder("OSD-0", _H, _pr))
        n_flags, _ = _features()
        flagged = n_flags > 0
        for trig, pre, fb in [("never", False, False), ("always", True, False),
                              ("primary_fail", False, True), ("flag>=1", flagged, True),
                              ("flag-only", flagged, False)]:
            pol = SwitchPolicy(W, S, trig, _mask)
            P = derive(res["peel"], res["OSD-0"], pre, fb)
            for i, s in enumerate(det):
                r = pol.decode(s)
                f = bool(np.any((_L @ r.correction) % 2 != _obs[i]))
                assert r.escalated == P["esc"][i] and f == P["fail"][i], f"{trig}: shot {i} differs"

    def _parallel_matches_sequential():
        par = _zoo()
        seq, _ = run_zoo(_H, _L, _pr, _det[:90], _obs[:90], _names, workers=1, chunk_size=30,
                         checkpoint_dir=False)
        for n in _names:
            for k in ("conv", "fail", "work"):
                assert np.array_equal(par[n][k], seq[n][k]), f"{n}.{k} differs"

    def _derive_cost_convention():
        """always costs exactly the secondary; a pre-escalated shot
        pays only the secondary; the oracle pays W + W.fail * S."""
        res = _zoo()
        W, S = res["peel"], res["OSD-0"]
        n = len(W["fail"])
        assert np.allclose(derive(W, S, np.ones(n, bool), False)["cost"], S["time_us"])
        flagged = _features()[0] > 0
        c = derive(W, S, flagged, True)["cost"]
        assert np.allclose(c[flagged], S["time_us"][flagged])
        o = oracle_policy(W, S)
        assert np.allclose(o["cost"], W["time_us"] + W["fail"] * S["time_us"].astype(float))
        cp = derive(W, S, np.ones(n, bool), False, convention="charge-primary")["cost"]
        assert np.allclose(cp, W["time_us"].astype(float) + S["time_us"])

    def _oracle_fixture():
        """W fails {0,1,2}, S fails {1,5} -> oracle fails once, escalates 0-2."""
        W, S = _toy()
        W["fail"][[0, 1, 2]] = True
        S["fail"][[1, 5]] = True
        o = oracle_policy(W, S)
        assert int(o["fail"].sum()) == 1 and list(np.flatnonzero(o["esc"])) == [0, 1, 2]

    def _reference_policy_table():
        res = _zoo()
        rows = policy_table(res, "peel", "OSD-0", _features()[0], workers=1)
        assert [r["policy"] for r in rows] == ["never", "flag-only", "flag>=1", "primary_fail",
                                               "always", "oracle"]
        al = rows[4]
        assert abs(al["cost_mean_us"] - float(res["OSD-0"]["time_us"].astype(float).mean())) < 1e-6
        assert all(r["cost_convention"] == COST_CONVENTIONS["derive"] for r in rows)
        try:
            policy_table(res, "peel", "OSD-0", _features()[0], workers={"peel": 12, "OSD-0": 1})
        except ValueError:
            return
        raise AssertionError("results timed with 12 and 1 workers were merged")

    def _breakeven_formula():
        pre = np.array([1, 1, 0, 0] * 25, bool)
        esc_pf = np.array([0, 0, 1, 0] * 25, bool)
        esc_rule = pre | esc_pf
        r = breakeven_ratio(esc_rule, pre, esc_pf)
        assert abs(r - (0.75 - 0.25) / 0.5) < 1e-12, r
        assert breakeven_ratio(esc_pf, np.zeros(100, bool), esc_pf) == float("inf")

    def _breakeven_agrees_with_cost():
        """(cost_ratio_cond > breakeven) == (rule cheaper than primary_fail)."""
        a = analyse(_zoo(), ["peel", "BP-ms-10"], ["OSD-0", "LSD-0"], *_features(), min_failures=0)
        checked = 0
        for r in a["rows"]:
            if np.isfinite(r["breakeven"]) and np.isfinite(r["cost_ratio_cond"]) \
                    and r["cost_us"] != r["pf_cost_us"]:
                assert (r["cost_ratio_cond"] > r["breakeven"]) == (r["cost_us"] < r["pf_cost_us"]), r
                checked += 1
        assert checked > 0

    def _headroom_bounds_gain():
        a = analyse(_zoo(), ["peel", "BP-ms-10"], ["OSD-0", "LSD-0"], *_features())
        head = {(p["weak"], p["strong"]): p["headroom"] for p in a["pairs"]}
        for r in a["rows"]:
            assert r["better"] <= head[(r["weak"], r["strong"])], "gain exceeds headroom"
            if r["rule"] not in ("flag-only", "primary_fail"):
                assert (r["better"], r["worse"]) == (r["catch"], r["harm"]), \
                    "gains/losses vs primary_fail must be exactly catches/harms"
        assert len(a["pairs"]) == 4 and all("verdict" in p for p in a["pairs"])
        assert a["m"] == 2 * 2 * 7

    def _verdict_order():
        """inconclusive first, then accuracy, then 'impossible: no headroom'."""
        nf, fr = _features()
        a = analyse(_zoo(), ["peel"], ["OSD-0"], nf, fr, min_failures=10**9)
        assert a["pairs"][0]["verdict_accuracy"] == "inconclusive"
        a = analyse(_zoo(), ["peel"], ["OSD-0"], nf, fr, min_failures=0)
        p = a["pairs"][0]
        if p["headroom"] == 0:
            assert p["verdict_accuracy"] == "not viable (impossible: no headroom)", p
        # synthetic: no headroom, pf with 9 vs 25 failures
        for k, want in ((9, "inconclusive"), (25, "not viable (impossible: no headroom)")):
            W, S = _toy(400)
            W["conv"][:k] = False
            W["fail"][:k] = True
            S["fail"][:k] = True
            nflag = np.zeros(400, int)
            nflag[100:140] = 1
            got = analyse({"w": W, "s": S}, ["w"], ["s"], nflag, nflag,
                          specs={"w": {"kind": "peel"}, "s": strong_spec(0)})["pairs"][0]
            assert got["verdict_accuracy"] == want, (k, got["verdict_accuracy"])

    def _cost_must_also_beat_always():
        """a rule at 0.6 x primary_fail but 0.95 x always is not cost-viable."""
        n = 2000
        W, S = _toy(n)
        W["time_us"] = np.full(n, 95.0)            # primary costs 0.95 of the secondary
        S["time_us"] = np.full(n, 100.0)
        W["conv"][:1000] = False                   # pf escalates half: cost 95 + 50 = 145
        nflag = np.zeros(n, int)
        nflag[:1000] = 1                           # flag>=1 = pre on the unconverged half
        a = analyse({"w": W, "s": S}, ["w"], ["s"], nflag, nflag, min_failures=0,
                    specs={"w": {"kind": "peel"}, "s": strong_spec(0)})
        row = next(r for r in a["rows"] if r["rule"] == "flag>=1")
        assert abs(row["saving"] - (1 - 97.5 / 145)) < 1e-9 and row["saving_vs_always"] < 0.05
        assert a["pairs"][0]["verdict_cost"] == "not viable"

    def _nested_pairs_labelled():
        """BP-ms-10 inside OSD-0 is nested; BP-ps-30 is not."""
        assert nested_bp("BP-ms-10", "OSD-0") and nested_bp("BP-ms-30", "LSD-0")
        assert not nested_bp("BP-ps-30", "OSD-0") and not nested_bp("BP-ms-100", "OSD-0")
        assert not nested_bp("peel", "OSD-0")
        res = _zoo()
        assert nested_mismatch(res["BP-ms-10"], res["OSD-0"]) == 0
        a = analyse(res, ["BP-ms-10"], ["OSD-0"], *_features(), min_failures=0)
        p = a["pairs"][0]
        assert p["nested"] and p["verdict_cost"] == "not attributable"
        assert p["verdict_accuracy"].startswith("impossible by construction")

    def _cost_interval_needs_a_real_saving():
        """an 8% saving buried in noise far above the primary_fail cost is not
        cost-viable; the same saving measured cleanly is."""
        n = 2000
        for sigma, want in ((10000.0, False), (0.1, True)):
            rng = np.random.default_rng(3)
            W, S = _toy(n)
            W["time_us"] = 100.0 + rng.normal(0, sigma, n)
            S["time_us"] = np.full(n, 1000.0)
            W["conv"][::2] = False                 # pf escalates half the shots: cost 600
            nflag = np.zeros(n, int)
            nflag[::2] = 1                         # the flag predicts them exactly: cost 550
            got = analyse({"w": W, "s": S}, ["w"], ["s"], nflag, nflag, min_failures=0,
                          specs={"w": {"kind": "peel"}, "s": strong_spec(0)})
            row = next(r for r in got["rows"] if r["rule"] == "flag>=1")
            lo, hi = row["saving_ci"]
            assert lo <= row["saving"] <= hi and row["saving_vs_always"] > 0.4
            assert row["cost_viable"] == want, (sigma, row["saving"], row["saving_ci"])
            assert (got["pairs"][0]["verdict_cost"] == "viable: cost") == want

    def _best_saving_matches_best_cost():
        a = analyse(_zoo(), ["peel", "BP-ms-10"], ["OSD-0", "LSD-0"], *_features(), min_failures=0)
        for p in a["pairs"]:
            if p["best_cost_rule"]:
                r = next(r for r in a["rows"] if (r["weak"], r["strong"], r["rule"])
                         == (p["weak"], p["strong"], p["best_cost_rule"]))
                assert p["best_saving"] == r["saving"]
            else:
                assert p["best_saving"] == 0.0

    def _positive_control_and_placebo():
        """
        The check that makes a null believable: inject silent failures on flagged
        shots; the analysis must find headroom, call a flag rule viable, and call
        it flag-informative against its placebo. Failures injected regardless of
        flags must NOT be flag-informative.
        """
        v = logical_null_vector(_H, _L)
        assert v is not None and not ((_H @ v) % 2).any() and ((_L @ v) % 2).any()
        nf, fr = flag_features(_det, _mask, _code.hx.shape[0], _T)
        out = {}
        for ctrl in ("control-flagged-50%", "control-any-20%"):
            spec = dict(CONTROLS[ctrl], vector=v, flag_mask=_mask)
            res, _ = run_zoo(_H, _L, _pr, _det, _obs, [("control", spec), "OSD-0"],
                             workers=1, chunk_size=100, checkpoint_dir=False)
            silent = res["control"]["conv"] & res["control"]["fail"]
            assert silent.sum() > 0, f"{ctrl} produced no silent failures"
            if CONTROLS[ctrl]["flagged_only"]:
                assert (nf[silent] > 0).all(), "control failures must sit on flagged shots"
            a = analyse(res, ["control"], ["OSD-0"], nf, fr, min_failures=1,
                        specs={"control": {"kind": "control"}})
            out[ctrl] = a
        p = out["control-flagged-50%"]["pairs"][0]
        assert p["headroom"] > 0 and p["verdict_accuracy"] == "viable: accuracy", p
        row = next(r for r in out["control-flagged-50%"]["rows"] if r["rule"] == "flag>=1")
        assert row["flag_informative"], row
        row = next(r for r in out["control-any-20%"]["rows"] if r["rule"] == "flag>=1")
        assert not row["flag_informative"], row

    def _placebo_matches_escalation():
        nf, fr = _features()
        rules = trigger_rules(nf, fr, placebo_seed=1)
        for name, (pre, fb) in rules.items():
            if name.startswith("placebo:"):
                orig = rules[name.split(":", 1)[1]]
                assert pre.sum() == orig[0].sum() and fb == orig[1]
        again = trigger_rules(nf, fr, placebo_seed=1)
        assert np.array_equal(again["placebo:flag>=1"][0], rules["placebo:flag>=1"][0])

    def _ceiling_and_diagnostics():
        res = _zoo()
        n_flags, _ = _features()
        ceil = {(c["decoder"], c["compared_with"]): c
                for c in strong_ceiling(res, ["OSD-0", "LSD-0"])}
        c = ceil[("OSD-0", "LSD-0")]
        assert c["failures"] == int(res["OSD-0"]["fail"].sum())
        assert c["fixed_by_other"] == int((res["OSD-0"]["fail"] & ~res["LSD-0"]["fail"]).sum())
        diag = {d["decoder"]: d for d in flag_diagnostics(res, ["peel"], ["OSD-0"], n_flags)}
        assert 0.0 <= diag["peel"]["mutual_information_bits"] <= 1.0
        rng = np.random.default_rng(0)
        fake = dict(res["peel"], fail=rng.random(90) < 0.3)
        mi = flag_diagnostics({"peel": fake}, ["peel"], [], n_flags)[0]["mutual_information_bits"]
        assert mi < 0.05, f"independent signal reported {mi:.3f} bits"

    def _blind_decoding_removes_flag_rows():
        keep = ~_mask
        H_blind, det_blind = _H[keep], _det[:, keep]
        assert H_blind.shape[0] == _H.shape[0] - int(_mask.sum())
        dec = make_decoder(strong_spec(0), H_blind, _pr)
        for s_blind in det_blind[:15]:
            e, _, _ = dec.decode(s_blind)
            assert np.array_equal((H_blind @ e) % 2, s_blind)

    def _zoo_file_roundtrip():
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "z.npz")
        res = _zoo()
        nf, fr = _features()
        save_zoo(p, res, nf, fr, dict(code="[[72, 12, 6]]", rounds=_T, seed=17))
        r2, nf2, fr2, meta = load_zoo(p)
        assert meta["seed"] == 17 and np.array_equal(nf2, nf)
        assert all(np.array_equal(r2[n][k], res[n][k]) for n in res for k in res[n])
        older = os.path.join(os.path.dirname(p), "older.npz")
        np.savez_compressed(older, n_flags=nf, flag_rounds=fr, meta=np.array(repr({"seed": 3})),
                            **{"peel__fail": res["peel"]["fail"]})
        assert load_zoo(older)[3] == {"seed": 3}

    def _graphs_render():
        a = analyse(_zoo(), ["peel", "BP-ms-10"], ["OSD-0", "LSD-0"], *_features())
        for fn in (plot_zoo_verdicts, plot_zoo_silent, plot_zoo_breakeven, plot_zoo_tradeoffs):
            fn(a).savefig(io.BytesIO(), format="png")
        fig = plot_reference_policies(policy_table(_zoo(), "peel", "OSD-0", _features()[0]))
        fig.savefig(io.BytesIO(), format="png")

    tests_analysis = run_checks([
        ("zoo decoders: honest convergence, strong ones always valid", _zoo_decoders_valid),
        ("derived policies reproduce SwitchPolicy shot by shot", _derivation_matches_switchpolicy),
        ("parallel decoding gives the same results as sequential", _parallel_matches_sequential),
        ("one cost convention: always = secondary; oracle = W + W.fail·S", _derive_cost_convention),
        ("oracle on a fixture fails once and escalates the primary's failures", _oracle_fixture),
        ("reference-policy table; mixed worker counts refused", _reference_policy_table),
        ("break-even cost ratio formula", _breakeven_formula),
        ("break-even and measured cost always agree", _breakeven_agrees_with_cost),
        ("no rule gains more than the headroom; gains = catches", _headroom_bounds_gain),
        ("verdict order: inconclusive, accuracy, no headroom", _verdict_order),
        ("cost-viable must also beat always", _cost_must_also_beat_always),
        ("nested BP pairs: impossible by construction, cost not attributable", _nested_pairs_labelled),
        ("cost saving needs a family-corrected interval", _cost_interval_needs_a_real_saving),
        ("best saving uses the same predicate as best cost", _best_saving_matches_best_cost),
        ("POSITIVE CONTROL and placebo: flag-informative only when it should be", _positive_control_and_placebo),
        ("placebos escalate as many shots, reproducibly", _placebo_matches_escalation),
        ("strong-decoder ceiling and flag diagnostics are correct", _ceiling_and_diagnostics),
        ("blind arm removes flag rows, not zeroes them", _blind_decoding_removes_flag_rows),
        ("decoder-grid files round-trip; older repr meta still loads", _zoo_file_roundtrip),
        ("all decoder-grid graphs render", _graphs_render),
    ])
    _backend = _state.get("backend", "not run")
    mo.vstack([render_checks("13 policies and verdicts", tests_analysis),
               mo.md(f"Parallel backend on this machine: **{_backend}** · "
                     f"compiled peeling: **{NUMBA_STATUS}**"
                     + ("" if HAVE_NUMBA else
                        "  \nWithout numba the peeling decoder is orders of magnitude slower, "
                        "which makes every cost comparison involving `peel` meaningless. "
                        "Install it with `uv add numba`, then reload this notebook."))])
    return (tests_analysis,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 14. Figures

    Every figure of the thesis is drawn by one of these functions, from a result
    file, by the regeneration commands of §18. Conventions shared by all of them:

    * error bars are 95% Wilson intervals; a point with **zero failures** has no
      logarithm, so it is drawn as a hollow ▽ at its **97.5% one-sided upper
      bound** (Wilson, z = 1.96) with no bar, and the caption says so;
    * one colour per policy or arm, assigned in a fixed order;
    * times are labelled with the worker count they were measured at; points
      measured at different worker counts are never ranked against each other.
    """)
    return


@app.function
def plot_colours(names):
    """One colour per name, by order of first appearance."""
    return {n: PLOT_PALETTE[i % len(PLOT_PALETTE)] for i, n in enumerate(names)}


@app.function
def plot_ler_marks(ax, x, metrics, colour, label=None, horizontal=False):
    """
    LER with Wilson error bars on a log axis. A zero-failure point is a hollow
    ▽ (◁ when horizontal) at its 97.5% one-sided upper bound, with no bar.
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
        label = None


@app.function
def plot_ablation(results):
    """(a) LER per policy with Wilson intervals; (b) fraction escalated."""
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
    a.set_yticks(y, [TRIGGER_DISPLAY.get(n, n) for n in names])
    a.set_xlabel("logical error rate (95% Wilson CI)")
    a.set_title("(a) accuracy")
    a.grid(True, which="both", axis="x", alpha=0.3)
    b.set_xlim(0, 1.15)
    b.set_xlabel("fraction of shots escalated")
    b.set_title("(b) escalation")
    b.grid(True, axis="x", alpha=0.3)
    shots = next(iter(results.values())).shots
    fig.suptitle(f"Policy ablation — {shots:,} paired shots   ({ZERO_LABEL.replace('▽', '◁')})",
                 fontsize=10)
    fig.tight_layout()
    return fig


@app.function
def plot_sweep(sweep):
    """(a) LER and (b) escalation rate against p; sweep = p -> policy -> Metrics."""
    ps = sorted(sweep)
    names = list(sweep[ps[0]])
    col = plot_colours(names)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, n in enumerate(names):
        ms = [sweep[p][n] for p in ps]
        dodge = 1 + 0.035 * (i - (len(names) - 1) / 2)
        xs = [p * dodge for p in ps]
        plot_ler_marks(a, xs, ms, col[n], label=TRIGGER_DISPLAY.get(n, n))
        a.plot(xs, [m.ler if m.failures else m.ci_high for m in ms],
               color=col[n], alpha=0.5, lw=1)
        b.plot(ps, [m.escalation_rate for m in ms], "s-", color=col[n],
               label=TRIGGER_DISPLAY.get(n, n))
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xlabel("physical error rate p (per noise channel)")
        ax.grid(True, which="both", alpha=0.3)
    a.set_yscale("log")
    a.set_ylabel("logical error rate per shot")
    a.set_title(f"(a) accuracy ({ZERO_LABEL})\npoints offset sideways so overlaps stay visible",
                fontsize=9)
    b.set_ylim(-0.03, 1.03)
    b.set_ylabel("fraction escalated")
    b.set_title("(b) escalation")
    a.legend(fontsize=8)
    fig.tight_layout()
    return fig


@app.function
def plot_outcomes(results):
    """Each policy's failures split into escalated / silent / unconverged."""
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
    for i, n in enumerate(names):
        m = results[n]
        ax.text(left[i] * 1.01 + 1e-4, y[i],
                f"{m.failures} fail ({m.silent_failures} silent)", va="center", fontsize=8)
    ax.set_yticks(y, [TRIGGER_DISPLAY.get(n, n) for n in names])
    ax.set_xlim(0, max(left.max() * 1.35, 1e-3))
    ax.set_xlabel("fraction of shots that failed, split by category")
    ax.set_title("Where do failures come from?")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def plot_latency(results, workers=None):
    """Empirical CDF of per-shot decode time (needs per-shot samples)."""
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
    reg = f", {workers} worker{'s' if workers != 1 else ''}" if workers else ""
    ax.set_xlabel(f"decode time per shot (µs{reg}; relative comparison only)")
    ax.set_ylabel("fraction of shots decoded")
    ax.set_title("Latency distribution (dots = p99)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    return fig


@app.function
def plot_tradeoff(results):
    """Accuracy against p99 time, one point per policy: lower-left is better."""
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


@app.function
def zoo_pair_rows(a, w, s):
    return [r for r in a["rows"] if r["weak"] == w and r["strong"] == s]


@app.function
def plot_zoo_verdicts(a):
    """Heatmap: best flag rule's net gain over primary_fail per 10k shots, with verdicts."""
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
    marks = {"viable: accuracy": "✓ accuracy", "viable: cost": "✓ cost",
             "inconclusive": "? too few failures", "not viable": "✗ no gain",
             "not viable (impossible: no headroom)": "✗ no headroom"}
    for i, w in enumerate(W):
        for j, s in enumerate(S):
            p = pairs[(w, s)]
            mark = "– nested BP" if p.get("nested") else marks.get(p["verdict"], "✗")
            ax.text(j, i, f"{mark}\nΔ {grid[i, j]:+.1f}\nheadroom {p['headroom']}",
                    ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(S)), S, rotation=20)
    ax.set_yticks(range(len(W)), W)
    ax.set_xlabel("secondary decoder")
    ax.set_ylabel("primary decoder")
    fig.colorbar(im, ax=ax, label="best flag rule: failures saved per 10k shots vs primary_fail")
    ax.set_title(f"Flag-triggered switching ({a['shots']:,} paired shots; "
                 f"Bonferroni m = {a.get('m', '?')})")
    fig.tight_layout()
    return fig


@app.function
def plot_zoo_silent(a):
    """Silent-failure rate of each primary: the only failures a flag can catch."""
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
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_ / 3, hi_ * 3)
    ax.set_xticks(range(len(names)), names)
    ax.set_xlabel("primary decoder")
    ax.set_ylabel("silent failures per converged shot")
    ax.set_title(f"How often is each primary confidently wrong?\n({ZERO_LABEL})")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def plot_zoo_breakeven(a):
    """
    For each pair, the CHEAPEST fallback rule with no accuracy loss: its
    conditional cost ratio (x) against its break-even ratio (y). Below the
    diagonal the rule is cheaper than primary_fail. Legend entries only for
    decoders actually plotted; `fig._selected` lists the plotted rows.
    """
    colours = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    selected = []
    for p in a["pairs"]:
        rows = [r for r in zoo_pair_rows(a, p["weak"], p["strong"])
                if r["rule"] != "primary_fail" and r["worse"] <= r["better"]
                and r["harm"] == 0 and np.isfinite(r["breakeven"])
                and np.isfinite(r.get("cost_ratio_cond", np.nan))]
        if not rows:
            continue
        selected.append(min(rows, key=lambda r: r["cost_us"]))
    weak = list(dict.fromkeys(r["weak"] for r in selected))
    strong = list(dict.fromkeys(r["strong"] for r in selected))
    for r in selected:
        x, y = max(r["cost_ratio_cond"], 1e-4), max(r["breakeven"], 1e-3)
        ax.plot(x, y, markers[strong.index(r["strong"]) % len(markers)],
                color=colours[weak.index(r["weak"]) % len(colours)], ms=8, ls="none")
        ax.annotate(r["rule"], (x, y), fontsize=7, textcoords="offset points", xytext=(5, 4))
    xs = np.logspace(-4, 1.5, 100)
    ax.plot(xs, xs, "k--", lw=1)
    ax.fill_between(xs, 1e-3, xs, color="#2ca02c", alpha=0.08)
    ax.text(0.97, 0.05, "rule cheaper than primary_fail\n(cost ratio > break-even)",
            transform=ax.transAxes, ha="right", fontsize=8, color="#2ca02c")
    for i, w in enumerate(weak):
        ax.plot([], [], "o", color=colours[i % len(colours)], label=f"primary: {w}")
    for j, s in enumerate(strong):
        ax.plot([], [], markers[j % len(markers)], color="grey", label=f"secondary: {s}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 30)
    ax.set_ylim(1e-3, 30)
    ax.set_xlabel("conditional cost ratio: mean primary time on pre-escalated shots / "
                  "mean secondary time on the extra escalations")
    ax.set_ylabel("break-even ratio r*")
    ax.set_title("Cheapest no-loss rule per pair")
    if weak:
        ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig._selected = selected
    return fig


@app.function
def plot_zoo_tradeoffs(a):
    """Small multiples: failures against MEAN COST for every rule of every pair."""
    W = list(dict.fromkeys(p["weak"] for p in a["pairs"]))
    S = list(dict.fromkeys(p["strong"] for p in a["pairs"]))
    fig, axes = plt.subplots(len(W), len(S), figsize=(3.1 * len(S), 2.5 * len(W)),
                             squeeze=False)
    pairs = {(p["weak"], p["strong"]): p for p in a["pairs"]}
    for i, w in enumerate(W):
        for j, s in enumerate(S):
            ax = axes[i][j]
            for r in zoo_pair_rows(a, w, s):
                pf = r["rule"] == "primary_fail"
                ax.plot(r["cost_us"], r["failures"], "*" if pf else "o",
                        color="#2ca02c" if pf else ("#d62728" if r["rule"] == "flag-only"
                                                    else "#9467bd"),
                        ms=11 if pf else 5)
            p = pairs[(w, s)]
            ax.plot(p["always_cost_us"], p["always_failures"], "D", color="k", ms=6)
            ax.set_title(f"{w} → {s}", fontsize=8)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if i == len(W) - 1:
                ax.set_xlabel("mean decode time per shot (µs)", fontsize=8)
            if j == 0:
                ax.set_ylabel("logical failures", fontsize=8)
    fig.suptitle("Failures against mean cost: ★ primary_fail, ◆ always, ● flag rules "
                 "(red = flag-only). A useful rule sits below-left of both ★ and ◆.",
                 fontsize=9)
    fig.tight_layout()
    return fig


@app.function
def plot_reference_policies(rows):
    """
    Figure 5.2: the six reference policies of one (primary, secondary) pair,
    from policy_table rows: (a) LER with Wilson CI, (b) mean cost with median
    and p99 markers. The cost convention is printed in the title.
    """
    names = [r["policy"] for r in rows]
    col = plot_colours(names)
    y = np.arange(len(rows))[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 0.6 * len(rows) + 2), sharey=True)
    for yi, r in zip(y, rows):
        m = Metrics(r["shots"], r["failures"], r["ler"], r["ci_low"], r["ci_high"],
                    r["escalation"], 0.0, 0.0, r["cost_median_us"], r["cost_p99_us"])
        plot_ler_marks(a, [yi], [m], col[r["policy"]], horizontal=True)
        b.barh(yi, r["cost_mean_us"], color=col[r["policy"]], alpha=0.85)
        b.plot(r["cost_median_us"], yi, "|", color="k", ms=14, mew=2)
        b.plot(r["cost_p99_us"], yi, "x", color="k", ms=7)
        b.text(r["cost_mean_us"] * 1.02, yi + 0.18, f"{r['cost_mean_us']:,.0f} µs",
               fontsize=8, va="center")
    a.set_xscale("log")
    a.set_yticks(y, [TRIGGER_DISPLAY.get(n, n) for n in names])
    a.set_xlabel("logical error rate (95% Wilson CI)")
    a.set_title("(a) accuracy")
    a.grid(True, which="both", axis="x", alpha=0.3)
    b.set_xscale("log")
    b.set_xlabel("decode time per shot (µs): bar = mean, | = median, × = p99")
    b.set_title("(b) cost")
    b.grid(True, which="both", axis="x", alpha=0.3)
    r0 = rows[0]
    wk = f"{r0['workers']} worker(s)" if r0.get("workers") else "workers not recorded"
    fig.suptitle(f"{r0['weak']} → {r0['strong']}, {r0['shots']:,} shots, {wk}; "
                 f"cost: {r0['cost_convention']}", fontsize=10)
    fig.tight_layout()
    return fig


@app.function
def last_down_crossing(ps, diffs):
    """
    The p where diff = ln LER(d_a) - ln LER(d_b) changes from > 0 (the larger
    code is better) to <= 0, interpolated in ln p; the LAST such change in p
    is taken, because low-p points carry few failures and can flip sign by
    noise. NaN entries (both curves without failures) are skipped. None when
    the curves do not cross in the range.
    """
    found, prev = None, None
    for p, d in zip(ps, diffs):
        if d is None or (isinstance(d, float) and math.isnan(d)):
            continue
        if prev is not None:
            pp, pd = prev
            if pd > 0 and d < 0:
                t = pd / (pd - d)
                found = math.exp(math.log(pp) + t * (math.log(p) - math.log(pp)))
            elif pd > 0 and d == 0:
                found = p
        prev = (p, d)
    return None if found is None else float(found)


@app.function
def threshold_crossing(points, da, db, key="ler", floor=None):
    """
    The threshold estimate between distances da < db: where the curves cross
    (see last_down_crossing), interpolated in log p and log LER. Points where
    both distances have zero failures are skipped; a single zero is floored
    at 0.5 / shots (or `floor`).
    """
    by_d = {}
    for pt in points:
        by_d.setdefault(pt["distance"], {})[float(pt["p"])] = pt
    if not {da, db} <= set(by_d):
        return None
    ps = sorted(set(by_d[da]) & set(by_d[db]))

    def val(pt):
        v = float(pt[key])
        if v > 0:
            return v, True
        return (floor if floor is not None else (0.5 / pt["shots"] if pt.get("shots") else 1e-12)), False

    diffs = []
    for p in ps:
        (va, ia), (vb, ib) = val(by_d[da][p]), val(by_d[db][p])
        diffs.append(math.log(va) - math.log(vb) if (ia or ib) else float("nan"))
    return last_down_crossing(ps, diffs)


@app.function
def threshold_estimate(points, B=2000, pairs=((5, 7), (3, 5), (3, 7)), failures_key="failures",
                       seed=20260929, conf=0.95):
    """
    Crossings for each distance pair with a parametric binomial bootstrap:
    every (d, p) point is resampled as Binomial(shots, k/shots), B times, and
    the crossing recomputed with the same rule. Returns {"d5-d7": dict(p_star,
    ci, frac_no_crossing, B), ...} and "headline" = the first pair. Use
    failures_key="mwpm_failures" for the MWPM arm.
    """
    rng = np.random.default_rng(seed)
    pts = [dict(distance=int(pt["distance"]), p=float(pt["p"]), shots=int(pt["shots"]),
                k=int(pt[failures_key])) for pt in points if failures_key in pt]
    out = {}
    for da, db in pairs:
        get = {(pt["distance"], pt["p"]): pt for pt in pts if pt["distance"] in (da, db)}
        ps = sorted({p for d, p in get if d == da} & {p for d, p in get if d == db})
        name = f"d{da}-d{db}"
        if len(ps) < 2:
            out[name] = dict(p_star=None, ci=None, frac_no_crossing=None, B=B)
            continue
        point = threshold_crossing([dict(distance=d, p=p, ler=get[(d, p)]["k"] / get[(d, p)]["shots"],
                                         shots=get[(d, p)]["shots"]) for d in (da, db) for p in ps],
                                   da, db)
        n_a = np.array([get[(da, p)]["shots"] for p in ps])
        n_b = np.array([get[(db, p)]["shots"] for p in ps])
        k_a = rng.binomial(n_a, np.array([get[(da, p)]["k"] for p in ps]) / n_a, size=(B, len(ps)))
        k_b = rng.binomial(n_b, np.array([get[(db, p)]["k"] for p in ps]) / n_b, size=(B, len(ps)))
        diff = np.log(np.maximum(k_a, 0.5) / n_a) - np.log(np.maximum(k_b, 0.5) / n_b)
        diff[(k_a == 0) & (k_b == 0)] = np.nan
        lp = np.log(np.array(ps))
        stars = np.full(B, np.nan)
        prev_d = np.full(B, np.nan)
        prev_lp = np.full(B, np.nan)
        for j in range(len(ps)):
            d = diff[:, j]
            ok = ~np.isnan(d) & ~np.isnan(prev_d)
            down = ok & (prev_d > 0) & (d < 0)
            t = np.where(down, prev_d / np.where(down, prev_d - d, 1.0), 0.0)
            stars[down] = np.exp(prev_lp[down] + t[down] * (lp[j] - prev_lp[down]))
            at = ok & (prev_d > 0) & (d == 0)
            stars[at] = ps[j]
            has = ~np.isnan(d)
            prev_d[has], prev_lp[has] = d[has], lp[j]
        good = stars[~np.isnan(stars)]
        q = (1 - conf) / 2
        out[name] = dict(p_star=point,
                         ci=(float(np.quantile(good, q)), float(np.quantile(good, 1 - q))) if len(good) else None,
                         frac_no_crossing=float(np.isnan(stars).mean()), B=int(B))
    out["headline"] = f"d{pairs[0][0]}-d{pairs[0][1]}"
    return out


@app.function
def plot_surface_sweep(points, threshold=None, mwpm_threshold=None):
    """
    Figure 5.1: LER per T rounds against p, one colour per distance; BP+OSD-0
    solid, MWPM dashed (when the points carry it). Zero-failure points are
    hollow ▽ at the 97.5% one-sided bound, without a bar. The d5/d7 crossing
    and its bootstrap interval are annotated when given.
    """
    by_d = {}
    for pt in points:
        by_d.setdefault(pt["distance"], []).append(pt)
    rounds = sorted({pt.get("rounds") for pt in points if pt.get("rounds") is not None})
    fig, ax = plt.subplots(figsize=(8, 5))
    col = plot_colours(sorted(by_d))
    for i, d in enumerate(sorted(by_d)):
        rows = sorted(by_d[d], key=lambda r: r["p"])
        mk = "os^Dv"[i % 5]
        meas = [r for r in rows if r["failures"] > 0]
        zero = [r for r in rows if r["failures"] == 0]
        ax.errorbar([r["p"] for r in meas], [r["ler"] for r in meas],
                    yerr=[[r["ler"] - r["ci_low"] for r in meas],
                          [r["ci_high"] - r["ler"] for r in meas]],
                    marker=mk, color=col[d], capsize=3, ls="-", label=f"d = {d}, BP+OSD-0")
        if zero:
            ax.plot([r["p"] for r in zero], [r["ci_high"] for r in zero], "v", ms=9,
                    mfc="none", mec=col[d], ls="none")
        if all("mwpm_failures" in r for r in rows):
            ys = [r["mwpm_ler"] if r["mwpm_failures"] else r["mwpm_ci_high"] for r in rows]
            ax.plot([r["p"] for r in rows], ys, "--", color=col[d], lw=1.2, alpha=0.8,
                    label=f"d = {d}, MWPM")
    for th, name in ((threshold, "BP+OSD-0"), (mwpm_threshold, "MWPM")):
        if th and th.get(th["headline"], {}).get("p_star"):
            h = th[th["headline"]]
            ci = h["ci"]
            ax.axvline(h["p_star"], color="k", ls=":" if name == "MWPM" else "-.", lw=1)
            txt = f"{name} p* ({th['headline']}) = {h['p_star']:.2g}"
            if ci:
                txt += f" [{ci[0]:.2g}, {ci[1]:.2g}]"
            ax.annotate(txt, (h["p_star"], 0.02 if name == "MWPM" else 0.05),
                        xycoords=("data", "axes fraction"), fontsize=8, rotation=90,
                        xytext=(3, 0), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate $p$")
    tag = f"{rounds[0]} rounds" if len(rounds) == 1 else "shot"
    ax.set_ylabel(f"logical error rate per {tag}")
    ax.set_title(f"Surface-code baseline (Z-basis memory)\n{ZERO_LABEL}", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


@app.function
def timing_regimes(points):
    """The distinct worker counts a set of points was measured at."""
    return sorted({int(r.get("workers_effective", r.get("workers", 1)) or 1) for r in points})


@app.function
def pareto_frontier(points, cost="time_p50_us", value="ler", strict=True):
    """
    The non-dominated points, cheapest first. A point is dominated when
    another is no worse on both axes and strictly better on one; ties keep
    the first point. `strict` refuses points measured at different worker
    counts, whose per-shot times are not comparable.
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


@app.function
def plot_pareto(points):
    """Figure 5.5: LER against median decode time per arm and decoder, with the frontier."""
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    front = pareto_frontier(points, strict=False)

    def _y(r):
        return r["ler"] if r["failures"] > 0 else r["ci_high"]

    if len(front) > 1:
        ax.plot([r["time_p50_us"] for r in front], [_y(r) for r in front], "-",
                color="#d62728", lw=1.6, alpha=0.8, zorder=1, label="Pareto frontier")
    elif len(front) == 1:
        ax.axvline(front[0]["time_p50_us"], color="#d62728", ls="--", lw=1.2,
                   alpha=0.6, zorder=1, label="frontier (a single point)")
    for arm, (c, mk, lab) in ARM_STYLE.items():
        rows = sorted([r for r in points if r["arm"] == arm], key=lambda r: r["time_p50_us"])
        if not rows:
            continue
        meas = [r for r in rows if r["failures"] > 0]
        zero = [r for r in rows if r["failures"] == 0]
        if meas:
            ax.errorbar([r["time_p50_us"] for r in meas], [r["ler"] for r in meas],
                        yerr=[[r["ler"] - r["ci_low"] for r in meas],
                              [r["ci_high"] - r["ler"] for r in meas]],
                        fmt=mk, color=c, ms=8, capsize=3, ls=":", lw=1, label=lab, zorder=3)
        if zero:
            ax.plot([r["time_p50_us"] for r in zero], [r["ci_high"] for r in zero], "v",
                    color=c, ms=9, mfc="none", mew=1.8, ls="none", zorder=3,
                    label=None if meas else lab)
        groups = {}
        for r in rows:
            key = (round(np.log10(max(r["time_p50_us"], 1e-9)), 1),
                   round(np.log10(max(_y(r), 1e-12)), 2))
            groups.setdefault(key, []).append(r)
        for members in groups.values():
            names = sorted({m.get("decoder", f"OSD-{m.get('osd_order', 0)}") for m in members})
            ax.annotate(",".join(names), (max(m["time_p50_us"] for m in members),
                                          max(_y(m) for m in members)),
                        fontsize=8, color=c, textcoords="offset points", xytext=(8, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    reg = timing_regimes(points)
    ax.set_xlabel(f"median decode time per shot (µs; workers {reg})")
    ax.set_ylabel("logical error rate per shot")
    ax.set_title(f"Down and to the left is better\n{ZERO_LABEL}", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


@app.function
def plot_prior_update(result):
    """Figure 5.4: LER and median time of the arms of one four-arm result."""
    order = [a for a in ("unflagged", "blind", "updated", "sighted") if a in result["arms"]]
    lab = {"unflagged": "unflagged\n(no flag qubits)", "blind": "blind\n(flags ignored)",
           "updated": "prior updating\n(no extra rows)", "sighted": "sighted\n(flag rows in H)"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    xs = np.arange(len(order))
    for i, a in enumerate(order):
        m = result["arms"][a]
        c = ARM_STYLE[a][0]
        if m["failures"] > 0:
            a1.errorbar(i, m["ler"], yerr=[[m["ler"] - m["ci_low"]], [m["ci_high"] - m["ler"]]],
                        fmt="o", ms=11, color=c, ecolor="k", capsize=5)
            a1.annotate(f"{m['ler']:.2e}", (i, m["ler"]), fontsize=9, color=c,
                        textcoords="offset points", xytext=(11, 0), va="center")
        else:
            a1.plot(i, m["ci_high"], "v", ms=11, mfc="none", mec=c)
        a2.plot(i, m["time_p50_us"], "o", ms=11, color=c)
        txt = f"{m['time_p50_us']:,.0f} µs"
        if m.get("time_update_p50_us"):
            txt += f"\n(update {m['time_update_p50_us']:,.0f})"
        a2.annotate(txt, (i, m["time_p50_us"]), fontsize=8.5, color=c,
                    textcoords="offset points", xytext=(11, 0), va="center")
    for ax, t, yl in ((a1, "Accuracy", "logical error rate per shot"),
                      (a2, "Cost (update inside the timer)", "median decode time per shot (µs)")):
        ax.set_xticks(xs, [lab[a] for a in order], fontsize=9)
        ax.set_yscale("log")
        ax.set_xlim(-0.5, len(order) - 0.15)
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.grid(axis="y", which="both", alpha=0.25)
    dec = result.get("decoder", "")
    fig.suptitle(f"Does the flag information survive without the flag rows? {dec}", fontsize=12)
    fig.tight_layout()
    return fig


@app.cell
def _():
    def _fake_zoo_analysis():
        rng = np.random.default_rng(0)
        n = 500
        res = {}
        for name, rate, t in (("peel", 0.05, 20.0), ("BP-ps-30", 0.04, 300.0),
                              ("OSD-0", 0.03, 3000.0)):
            res[name] = dict(conv=rng.random(n) > rate, fail=rng.random(n) < rate,
                             work=np.ones(n, int), time_us=rng.gamma(4, t / 4, n))
        nf = rng.poisson(0.3, n)
        return res, nf, analyse(res, ["peel", "BP-ps-30"], ["OSD-0"], nf, nf, min_failures=0)

    def _zoo_plots_render_and_select():
        res, nf, a = _fake_zoo_analysis()
        for fn in (plot_zoo_verdicts, plot_zoo_silent, plot_zoo_tradeoffs):
            fn(a).savefig(io.BytesIO(), format="png")
        fig = plot_zoo_breakeven(a)
        fig.savefig(io.BytesIO(), format="png")
        for r in fig._selected:
            cands = [x for x in zoo_pair_rows(a, r["weak"], r["strong"])
                     if x["rule"] != "primary_fail" and x["worse"] <= x["better"] and x["harm"] == 0
                     and np.isfinite(x["breakeven"]) and np.isfinite(x["cost_ratio_cond"])]
            assert r is min(cands, key=lambda x: x["cost_us"])
        assert fig.axes[0].get_xlabel().startswith("conditional cost ratio")
        tf = plot_zoo_tradeoffs(a)
        assert tf.axes[-1].get_xlabel().startswith("mean decode time")

    def _crossing_is_log_log():
        """the crossing is interpolated in log p / log LER."""
        pts = [dict(distance=5, p=1e-3, ler=1e-2), dict(distance=7, p=1e-3, ler=1e-3),
               dict(distance=5, p=1e-1, ler=1e-1), dict(distance=7, p=1e-1, ler=1.0)]
        assert abs(threshold_crossing(pts, 5, 7) - 1e-2) < 1e-12

    def _bootstrap_covers_a_known_crossing():
        """a synthetic crossing at 1e-2 is covered in >= 44 of 50 datasets."""
        rng = np.random.default_rng(5)
        ps = [4e-3, 6e-3, 8e-3, 1.2e-2, 1.5e-2]
        covered = 0
        for rep in range(50):
            pts = []
            for d in (5, 7):
                for p in ps:
                    lam = 0.1 * (p / 1e-2) ** ((d + 1) / 2)
                    k = int(rng.binomial(20000, lam))
                    pts.append(dict(distance=d, p=p, shots=20000, failures=k))
            est = threshold_estimate(pts, B=400, pairs=((5, 7),), seed=rep)["d5-d7"]
            if est["ci"] and est["ci"][0] <= 1e-2 <= est["ci"][1]:
                covered += 1
        assert covered >= 44, f"covered {covered}/50"

    def _no_crossing_gives_none():
        flat = [dict(distance=d, p=p, shots=1000, failures=10 if d == 5 else 1)
                for d in (5, 7) for p in (1e-3, 3e-3)]
        est = threshold_estimate(flat, B=200, pairs=((5, 7),))
        assert est["d5-d7"]["p_star"] is None and est["d5-d7"]["frac_no_crossing"] > 0.5

    def _zero_points_have_no_bar():
        """a zero-failure point is a ▽ at ci_high and no error bar reaches it."""
        pts = [dict(distance=d, p=p, rounds=6, ler=1e-3 * d, ci_low=5e-4 * d, ci_high=2e-3 * d,
                    failures=10, shots=1000, mwpm_failures=8, mwpm_ler=8e-4 * d,
                    mwpm_ci_high=2e-3 * d) for d in (3, 5, 7) for p in (1e-3, 3e-3)]
        pts[0].update(failures=0, ler=0.0, ci_low=0.0, ci_high=3.7e-3)
        fig = plot_surface_sweep(pts)
        ax = fig.axes[0]
        tri = [l for l in ax.get_lines() if l.get_marker() == "v"]
        assert tri and abs(tri[0].get_ydata()[0] - 3.7e-3) < 1e-12
        assert any(l.get_linestyle() == "--" for l in ax.get_lines()), "MWPM must be dashed"
        assert "per 6 rounds" in ax.get_ylabel()

    def _figures_render():
        m = Metrics(100, 3, 0.03, 0.01, 0.08, 0.4, 2.0, 5.0, 10.0, 50.0, 1, 0,
                    samples=dict(time_us=np.arange(100.0)))
        z = Metrics(100, 0, 0.0, 0.0, 0.037, 0.4, 2.0, 5.0, 10.0, 50.0, 0, 0,
                    samples=dict(time_us=np.arange(100.0)))
        res = {"never": m, "flag>=1": z}
        for fig in (plot_ablation(res), plot_sweep({1e-3: res, 2e-3: res}), plot_outcomes(res),
                    plot_latency(res, workers=1), plot_tradeoff(res)):
            fig.savefig(io.BytesIO(), format="png")
        pts = [dict(arm=a, decoder="OSD-0", time_p50_us=t, ler=l, ci_low=l / 2, ci_high=l * 2,
                    failures=f, workers_effective=1)
               for a, t, l, f in (("unflagged", 10, 1e-2, 5), ("blind", 20, 8e-3, 4),
                                  ("sighted", 30, 5e-3, 0), ("updated", 25, 4e-3, 2))]
        plot_pareto(pts).savefig(io.BytesIO(), format="png")
        four = dict(decoder="OSD-0", arms={p["arm"]: dict(p, time_update_p50_us=3.0) for p in pts})
        plot_prior_update(four).savefig(io.BytesIO(), format="png")
        try:
            pareto_frontier(pts + [dict(pts[0], workers_effective=12)])
        except ValueError:
            return
        raise AssertionError("points from two worker counts were ranked together")

    tests_plotting = run_checks([
        ("decoder-grid plots render; breakeven plots the cheapest no-loss rule", _zoo_plots_render_and_select),
        ("threshold crossings are interpolated log-log", _crossing_is_log_log),
        ("bootstrap interval covers a known crossing in >= 44/50 datasets", _bootstrap_covers_a_known_crossing),
        ("no crossing in range -> None, never a fabricated number", _no_crossing_gives_none),
        ("zero-failure points: hollow ▽ at the bound, no bar; MWPM dashed", _zero_points_have_no_bar),
        ("every figure renders; mixed worker counts are not ranked", _figures_render),
    ])
    render_checks("14 figures", tests_plotting)
    return (tests_plotting,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 15. The four-arm runner

    One function, `four_arm_run`, produces every number of Experiments 2, 7, 8,
    10 and 11. It decodes one configuration (code, T, p, shots) four ways:

    | arm | circuit | matrix decoded | priors |
    |---|---|---|---|
    | `unflagged` | no flag qubits — **its own circuit and its own seed** | full H | the DEM's |
    | `blind` | flagged | H with the flag rows **removed** | the DEM's |
    | `sighted` | flagged | full H, flag rows included | the DEM's |
    | `updated` | flagged | the blind matrix | the exact flag posterior of Proposition 4.2, per shot |

    The three flagged arms decode **the same shots** of one circuit, so their
    comparisons are **paired** (exact McNemar on the discordant shots). The
    unflagged arm is a different circuit, so sighted-vs-unflagged and
    blind-vs-unflagged are **unpaired** (exact conditional rate-ratio interval,
    `rate_ratio_ci`). The thesis says which is which every time.

    **One decoder.** `decoder=` is a catalogue name (`"OSD-0"`, `"OSD-CS2"`,
    `"LSD-0"`, `"relay"`, …) or an explicit spec; every arm of a run uses the
    same one, with `STRONG_MAX_ITER` = 30 iterations and the shared ldpc
    settings. The full resolved specification is stored in the result.

    **Timing — the prior update is part of the cost.** Every chunk starts with
    one *untimed* warm-up decode. For the updated arm the clock starts **before**
    the per-shot priors are formed: `t0` → `shot_priors` (choose each flagged
    fault's posterior from the flag bits) → `update_priors` (load them into the
    decoder) → `t1` → decode → `t2`. The stored per-shot time is
    `time_us = (t1 − t0) + (t2 − t1)`, update plus decode, and **every** time
    comparison involving the updated arm uses it: the median and p99 columns of
    Tables 5.8/5.9, the frontier of Experiment 7 and the Wilcoxon test T14. The
    two parts are also stored separately (`time_update_us`, `time_decode_us`) so
    the tables can show how much of the cost is the update. The other arms have
    no update; their `time_update_us` is zero. Note that the update cost depends
    on the decoder library: ldpc copies the whole prior vector on every call
    (≈ 70 ns per fault mechanism, a few hundred µs per shot at $T = 12$), while
    the built-in Relay-BP only recomputes the log-prior ratios.

    **Arms in turns.** At workers = 1 every arm decodes chunk $k$ (250 shots)
    before any arm decodes chunk $k+1$, in an order that rotates from chunk to
    chunk (blind–sighted–updated, then sighted–updated–blind, …). A slow spell of
    the machine therefore lands on every arm alike and cannot bias a paired time
    comparison such as T14; the outcomes are identical to decoding arm by arm (a
    self-test checks this shot by shot). With more workers the fixed-prior arms
    use the process pool (from 2,000 shots up) and run one after another; the
    updated arm changes its priors every shot, so it always runs in this
    process. Every arm records the worker count it was actually timed at, and
    the result records the order (`timing_order`); latency claims use workers = 1
    runs only.

    **Seeds.** Per-configuration seeds come from `numpy.random.SeedSequence`,
    seeded with the experiment's master seed and indexed by (code, p, T, shard)
    — never a hand-picked number. The unflagged arm's seed is derived from the
    flagged one.

    **Checkpoints.** Each arm checkpoints into
    `results/checkpoints/<result stem>/<config>/<arm>_<decoder>` (arms in turns
    share one folder, `interleaved_<decoder>`); a configuration
    interrupted by a power cut resumes chunk by chunk, and a resumed arm says
    "times partly from an earlier session".
    """)
    return


@app.function
def stable_int(obj):
    """A 64-bit integer from any JSON-able value (stable across runs and machines)."""
    h = hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).digest()
    return int.from_bytes(h[:8], "little")


@app.function
def derive_seed(master, *index):
    """
    A Stim seed for one configuration: SeedSequence(master) indexed by the
    configuration's fields, e.g. derive_seed(m, "[[72, 12, 6]]", 1e-3, 12, 0).
    Deterministic, distinct for distinct indices, < 2**63.
    """
    words = [int(master) & 0xFFFFFFFF, (int(master) >> 32) & 0xFFFFFFFF]
    for i in index:
        v = stable_int(i)
        words += [v & 0xFFFFFFFF, v >> 32]
    a, b = np.random.SeedSequence(words).generate_state(2, np.uint32)
    return int((int(a) << 32 | int(b)) & (2 ** 63 - 1))


@app.function
def mcnemar_counts(x, y):
    """Exact two-sided sign test from discordant counts (x only-A, y only-B)."""
    x, y = int(x), int(y)
    n = x + y
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * st.binom.cdf(min(x, y), n, 0.5)))


@app.function
def four_arm_run(code, rounds, p, shots, seed, decoder="OSD-0", arms=FOUR_ARMS, workers=1,
                 shard=0, x_detectors=False, unflagged_seed=None, checkpoint_dir=None,
                 stem=None, decoder_spec=None, on_progress=None, keep_per_shot=True,
                 interleave=None, basis="Z"):
    """
    Decode one configuration with up to four arms: three PAIRED arms on one
    flagged circuit (blind, sighted, updated) and one UNPAIRED arm on the
    unflagged circuit. See §15. Returns a JSON-safe dict, plus "per_shot"
    (arm -> fail / time_us / time_update_us / time_decode_us / iters) when
    keep_per_shot. checkpoint_dir: None = automatic folder under
    CKPT["root"]/<stem> for runs of at least CKPT["min_shots"] shots, False =
    off, or a folder. interleave: None = automatic (arms decoded in turns, chunk
    by chunk, whenever the run is single-process and has more than one arm),
    True / False to force it. basis: "Z" (default) or "X" -- the X-basis
    memory with flags on the Z checks, run as the Z-basis memory of
    x_basis_mirror(code) (§4).
    """
    arms = tuple(a for a in FOUR_ARMS if a in arms)
    unknown = set(arms) - set(FOUR_ARMS)
    if unknown or not arms:
        raise ValueError(f"arms must be a non-empty subset of {FOUR_ARMS}")
    if x_detectors is None:
        raise ValueError("x_detectors must be True or False")
    code_obj = code_from_key(code) if isinstance(code, str) else code
    if basis not in ("Z", "X"):
        raise ValueError(f"basis must be 'Z' or 'X', got {basis!r}")
    if basis == "X":
        code_obj = x_basis_mirror(code_obj)
    key = code_obj.name
    spec = dict(decoder_spec) if decoder_spec is not None else dict(CATALOGUE[decoder])
    spec_full = decoder_spec_full(spec)
    shots, rounds, p, seed = int(shots), int(rounds), float(p), int(seed)
    if unflagged_seed is None:
        unflagged_seed = derive_seed(seed, "unflagged")
    stem = stem or result_stem("RUN", code=key, rounds=rounds, p=p, shots=shots, seed=seed,
                               convention="x+z" if x_detectors else "z-only",
                               extra=[decoder, f"s{shard}"])
    if checkpoint_dir is False:
        base = None
    elif checkpoint_dir is None:
        base = os.path.join(CKPT["root"], stem) if shots >= CKPT["min_shots"] else None
    else:
        base = checkpoint_dir

    def ck(arm):
        return False if base is None else os.path.join(base, f"{arm}_{decoder}")

    def bump(arm, a, b):
        if on_progress is not None:
            on_progress(arm, b - a)

    out = dict(code=key, rounds=rounds, p=p, shots=shots, shard=int(shard), seed=seed,
               unflagged_seed=int(unflagged_seed), decoder=decoder, decoder_spec=spec_full,
               x_detectors=bool(x_detectors),
               detector_convention="x+z" if x_detectors else "z-only",
               arms={}, paired={}, unpaired={}, per_shot={}, basis=basis)
    per = {}

    def summarise(arm, fail, t, t_up, t_dec, iters, Hrows, faults, ident, backend, info, shash,
                  arm_seed):
        k = int(np.sum(fail))
        ler, lo, hi = wilson(k, shots)
        # one float32 time base, so time = update + decode holds exactly per shot
        t_up = np.asarray(t_up, np.float32)
        t_dec = np.asarray(t_dec, np.float32)
        t = t_up + t_dec
        resumed = int(info.get("resumed_chunks", info.get("resumed", 0)))
        total = int(info.get("total_chunks", info.get("chunks", 0)))
        out["arms"][arm] = dict(
            failures=k, ler=ler, ci_low=lo, ci_high=hi,
            time_p50_us=float(np.median(t)), time_p99_us=float(np.percentile(t, 99)),
            time_mean_us=float(np.mean(t)),
            time_update_p50_us=float(np.median(t_up)),
            time_decode_p50_us=float(np.median(t_dec)),
            bp_iters_mean=float(np.mean(iters)), detectors=int(Hrows), faults=int(faults),
            qubits_used=int(ident["qubits_used"]), qubit_index_span=int(ident["num_qubits"]),
            dem_sha256=ident["dem_sha256"], circuit_sha256=ident["circuit_sha256"],
            workers_effective=int(info.get("workers_effective", 1)), backend=backend,
            resumed_chunks=resumed, total_chunks=total,
            checkpoint_dir=info.get("dir", info.get("checkpoint_dir")),
            timing_note=("times partly from an earlier session" if resumed else ""),
            sample_sha256=shash, seed=int(arm_seed))
        per[arm] = dict(fail=np.asarray(fail, bool), time_us=t, time_update_us=t_up,
                        time_decode_us=t_dec,
                        iters=np.asarray(iters, np.int32))

    flagged_arms = [a for a in arms if a != "unflagged"]
    w_pool = parallel_workers(shots, workers)
    if interleave is None:
        interleave = w_pool == 1 and len(arms) > 1
    prob = {}                                 # arm -> (H, L, priors, det, obs, ident, hash, seed)
    if flagged_arms:
        circ = build_memory_circuit(code_obj, rounds, p, use_flags=True, x_detectors=x_detectors)
        H, L, priors = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(code_obj, rounds, True, circ.num_detectors,
                                  x_detectors=x_detectors)
        det, obs = circ.compile_detector_sampler(seed=seed).sample(shots,
                                                                   separate_observables=True)
        det, obs = det.astype(np.uint8), obs.astype(np.uint8)
        shash = sample_hash(det, obs)
        ident = circuit_identity(circ, H, priors)
        flags = det[:, mask]
        HS, syn = H[~mask], det[:, ~mask]
        idx, disjoint, sinfo = flag_groups(H, mask)
        out["structure"] = dict(sinfo, disjoint=bool(disjoint))
        out["overhead"] = dict(detectors_flagged=int(H.shape[0]), detectors_blind=int(HS.shape[0]),
                               flag_detectors=int(mask.sum()), faults_flagged=int(H.shape[1]),
                               qubits_flagged=int(ident["qubits_used"]),
                               mean_flags_per_shot=float(flags.sum(1).mean()),
                               flagged_fraction=float((flags.sum(1) > 0).mean()))
        out["sample_sha256"] = shash
        prob["sighted"] = (H, L, priors, det, obs, ident, shash, seed)
        prob["blind"] = (HS, L, priors, syn, obs, ident, shash, seed)
        if "updated" in arms:
            if not disjoint:
                raise ValueError(f"flag groups overlap on {key} (Condition 4.1 fails): {sinfo}")
            lo_p, hi_p, owner = flag_posteriors(priors, idx)
            prob["updated"] = (HS, L, priors, syn, obs, ident, shash, seed)
    if "unflagged" in arms:
        circ_u = build_memory_circuit(code_obj, rounds, p, use_flags=False,
                                      x_detectors=x_detectors)
        Hu, Lu, pru = dem_to_matrices(circ_u.detector_error_model(decompose_errors=False))
        du, ou = circ_u.compile_detector_sampler(seed=int(unflagged_seed)).sample(
            shots, separate_observables=True)
        du, ou = du.astype(np.uint8), ou.astype(np.uint8)
        ident_u = circuit_identity(circ_u, Hu, pru)
        prob["unflagged"] = (Hu, Lu, pru, du, ou, ident_u, sample_hash(du, ou), int(unflagged_seed))
        out.setdefault("overhead", {}).update(detectors_unflagged=int(Hu.shape[0]),
                                              faults_unflagged=int(Hu.shape[1]),
                                              qubits_unflagged=int(ident_u["qubits_used"]))

    def updated_shots(dec, Ls, a, b, warm=True):
        """The updated arm on shots a..b-1: per-shot priors, update inside the timer."""
        n = b - a
        fail, it = np.zeros(n, bool), np.zeros(n, np.int32)
        t_up, t_de = np.zeros(n, np.float64), np.zeros(n, np.float64)
        if warm:
            dec.update_priors(shot_priors(lo_p, hi_p, owner, flags[a]))
            dec.decode(syn[a])                                     # untimed warm-up
        for j, i in enumerate(range(a, b)):
            t0 = time.perf_counter()
            dec.update_priors(shot_priors(lo_p, hi_p, owner, flags[i]))
            t1 = time.perf_counter()
            e, _c, w_ = dec.decode(syn[i])
            t2 = time.perf_counter()
            t_up[j], t_de[j] = (t1 - t0) * 1e6, (t2 - t1) * 1e6
            it[j] = w_
            fail[j] = bool(np.any((Ls @ e) % 2 != obs[i]))
        return dict(fail=fail, time_update_us=t_up, time_decode_us=t_de, iters=it)

    def fixed_shots(dec, Ls, d, o, a, b):
        """A fixed-prior arm on shots a..b-1, timed exactly as decode_chunk does."""
        n = b - a
        fail, it = np.zeros(n, bool), np.zeros(n, np.int32)
        t_de = np.zeros(n, np.float64)
        dec.decode(d[a])                                           # untimed warm-up
        for j, i in enumerate(range(a, b)):
            t0 = time.perf_counter()
            e, _c, w_ = dec.decode(d[i])
            t_de[j] = (time.perf_counter() - t0) * 1e6
            it[j] = w_
            fail[j] = bool(np.any((Ls @ e) % 2 != o[i]))
        return dict(fail=fail, time_update_us=np.zeros(n), time_decode_us=t_de, iters=it)

    order = [a for a in FOUR_ARMS if a in arms]
    if interleave:
        # Every arm decodes chunk k before any arm decodes chunk k + 1, in a
        # rotating order, so a slow spell of the machine (other programs,
        # temperature, clock changes) lands on all arms alike.
        decs = {a: make_decoder(dict(spec), prob[a][0], prob[a][2]) for a in order}
        Lsp = {a: sp.csr_matrix(prob[a][1]) for a in order}

        def _chunk(a, b):
            k = a // CHUNK_SIZE
            rot = order[k % len(order):] + order[:k % len(order)]
            part = {}
            for arm in rot:
                if arm == "updated":
                    r = updated_shots(decs[arm], Lsp[arm], a, b)
                else:
                    _H, _L, _p, d, o = prob[arm][:5]
                    r = fixed_shots(decs[arm], Lsp[arm], d, o, a, b)
                part.update({f"{arm}.{f}": v for f, v in r.items()})
            return part

        arrs, info = run_resumable(
            shots, _chunk, ("four-arm-interleaved", tuple(order),
                            *[(prob[a][0], prob[a][2], prob[a][3], prob[a][4]) for a in order],
                            flags if "updated" in order else None, spec_full),
            chunk_size=CHUNK_SIZE, checkpoint_dir=ck("interleaved"),
            label=f"{stem}: {'/'.join(order)} in turns, {decoder}, {key} T={rounds} p={p:g} "
                  f"{shots:,} shots",
            on_chunk=(None if on_progress is None else
                      (lambda a, b, _r: [bump(arm, a, b) for arm in order])))
        backend = ("single-process loop, arms in turns per chunk (rotating order)"
                   + (f"; checkpoint {info['checkpoint_dir']}" if info["checkpoint_dir"] else ""))
        for arm in order:
            Ha, _L, _p, _d, _o, idt, hsh, sd = prob[arm]
            t_up, t_de = arrs[f"{arm}.time_update_us"], arrs[f"{arm}.time_decode_us"]
            summarise(arm, arrs[f"{arm}.fail"], t_up + t_de, t_up, t_de, arrs[f"{arm}.iters"],
                      Ha.shape[0], Ha.shape[1], idt, backend,
                      dict(info, workers_effective=1, total_chunks=info["chunks"]), hsh, sd)
        out["timing_order"] = "interleaved: arms in turns per chunk"
    else:
        for arm in order:
            Ha, La, pa, da, oa, idt, hsh, sd = prob[arm]
            if arm == "updated":
                dec = make_decoder(dict(spec), HS, priors)
                Ls = sp.csr_matrix(L)
                arrs, info = run_resumable(
                    shots, lambda a, b: updated_shots(dec, Ls, a, b),
                    ("four-arm-updated", HS, L, priors, syn, obs, flags, spec_full),
                    chunk_size=CHUNK_SIZE, checkpoint_dir=ck("updated"),
                    label=f"{stem}: updated, {decoder}, {key} T={rounds} p={p:g} {shots:,} shots",
                    on_chunk=None if on_progress is None else (lambda a, b, _r: bump("updated", a, b)))
                t_up, t_de = arrs["time_update_us"], arrs["time_decode_us"]
                summarise("updated", arrs["fail"], t_up + t_de, t_up, t_de, arrs["iters"],
                          Ha.shape[0], Ha.shape[1], idt,
                          "single-process resumable loop" + (
                              f"; checkpoint {info['checkpoint_dir']}" if info["checkpoint_dir"] else ""),
                          dict(info, workers_effective=1, total_chunks=info["chunks"]), hsh, sd)
                continue
            cs = CHUNK_SIZE if ck(arm) else chunk_for(shots, w_pool)
            res, backend, info = run_zoo(
                Ha, La, pa, da, oa, [(decoder, spec)], workers=w_pool, chunk_size=cs,
                checkpoint_dir=ck(arm), return_info=True,
                on_chunk=(None if on_progress is None else (lambda _a=arm: on_progress(_a, cs))),
                label=f"{stem}: {arm}, {decoder}, {key} T={rounds} p={p:g} {shots:,} shots")
            r = res[decoder]
            z = np.zeros(shots, np.float32)
            summarise(arm, r["fail"], r["time_us"], z, r["time_us"], r["work"], Ha.shape[0],
                      Ha.shape[1], idt, backend, info, hsh, sd)
        out["timing_order"] = "arm by arm" if len(order) > 1 else "single arm"
    # ---- comparisons --------------------------------------------------------
    # pairing is COMPUTED from the sample hashes, never assumed
    hashes = {a: m["sample_sha256"] for a, m in out["arms"].items()}
    out["pairing"] = {f"{a}|{b}": hashes[a] == hashes[b]
                      for a, b in PAIRED_COMPARISONS + UNPAIRED_COMPARISONS
                      if a in hashes and b in hashes}
    for a, b in PAIRED_COMPARISONS:
        if a in per and b in per:
            x, y, pv = paired_exact(per[a]["fail"], per[b]["fail"])
            ka, kb = out["arms"][a]["failures"], out["arms"][b]["failures"]
            out["paired"][f"{a}|{b}"] = dict(a=a, b=b, a_only=x, b_only=y, p=pv,
                                             ratio=(ka / kb) if kb else float("inf"),
                                             paired=True, method="exact McNemar (two-sided)")
    for a, b in UNPAIRED_COMPARISONS:
        if a in per and b in per:
            rr = rate_ratio_ci(out["arms"][a]["failures"], shots,
                               out["arms"][b]["failures"], shots)
            out["unpaired"][f"{a}|{b}"] = dict(rr, a=a, b=b)
    if "unflagged" in per:
        out["mde_unpaired_ratio_at_observed_baseline"] = mde_ratio(
            out["arms"]["unflagged"]["failures"])
    out["min_discordant_holm"] = min_detectable_discordant(0.05 / 14)
    if keep_per_shot:
        out["per_shot"] = per
    else:
        out.pop("per_shot")
    return out


@app.function
def pareto_point(code, rounds, p, shots, seed, arm, osd_order, workers=1, x_detectors=False,
                 on_chunk=None, checkpoint_dir=None):
    """
    Experiment 7 wrapper: one (arm, OSD order) point of four_arm_run, as a
    flat row (arm, decoder, osd_order, failures, ler, CI, times, bp_iters,
    workers). `seed` is the seed of the circuit the arm decodes.
    """
    name = {0: "OSD-0", 2: "OSD-CS2", 4: "OSD-CS4", 7: "OSD-CS7"}.get(int(osd_order),
                                                                     f"OSD-CS{osd_order}")
    r = four_arm_run(code, rounds, p, shots, seed, decoder=name, arms=(arm,), workers=workers,
                     x_detectors=x_detectors, unflagged_seed=seed,
                     decoder_spec=strong_spec(osd_order), checkpoint_dir=checkpoint_dir,
                     on_progress=None if on_chunk is None else (lambda _a, n: on_chunk(n)))
    m = r["arms"][arm]
    return dict(m, arm=arm, decoder=name, osd_order=int(osd_order), code=r["code"],
                rounds=int(rounds), p=float(p), shots=int(shots), seed=int(seed),
                workers=m["workers_effective"], bp_iters=m["bp_iters_mean"],
                qubits=m["qubits_used"], per_shot=r["per_shot"][arm])


@app.function
def prior_update_run(code, rounds, p, shots, seed, osd_order=0, x_detectors=False,
                     arms=("blind", "updated", "sighted"), on_progress=None, checkpoint_dir=None):
    """Experiment 8 wrapper: four_arm_run with BP+OSD of the given order, the
    paired flagged arms by default."""
    name = {0: "OSD-0", 2: "OSD-CS2", 4: "OSD-CS4"}.get(int(osd_order), f"OSD-CS{osd_order}")
    return four_arm_run(code, rounds, p, shots, seed, decoder=name, arms=arms,
                        x_detectors=x_detectors, decoder_spec=strong_spec(osd_order),
                        checkpoint_dir=checkpoint_dir, on_progress=on_progress)


@app.function
def combine_shards(results):
    """
    Sum shards of the same (code, T, p, decoder): failures per arm add, and
    the paired discordant counts add before the exact test is recomputed
    (valid because the shards are independent). Returns one result per
    configuration with "shards" listing the shard ids.
    """
    groups = {}
    for r in results:
        groups.setdefault((r["code"], r["rounds"], r["p"], r["decoder"]), []).append(r)
    out = []
    for (code, T, p, dec), rs in groups.items():
        n = sum(r["shots"] for r in rs)
        c = dict(code=code, rounds=T, p=p, decoder=dec, shots=n,
                 shards=[r["shard"] for r in rs], seeds=[r["seed"] for r in rs],
                 arms={}, paired={}, unpaired={})
        arms = set.intersection(*[set(r["arms"]) for r in rs])
        for a in [x for x in FOUR_ARMS if x in arms]:
            k = sum(r["arms"][a]["failures"] for r in rs)
            ler, lo, hi = wilson(k, n)
            c["arms"][a] = dict(failures=k, ler=ler, ci_low=lo, ci_high=hi,
                                workers_effective=sorted({r["arms"][a]["workers_effective"]
                                                          for r in rs}))
        for key_ in set.intersection(*[set(r["paired"]) for r in rs]):
            x = sum(r["paired"][key_]["a_only"] for r in rs)
            y = sum(r["paired"][key_]["b_only"] for r in rs)
            a, b = key_.split("|")
            kb = c["arms"][b]["failures"]
            c["paired"][key_] = dict(a=a, b=b, a_only=x, b_only=y, p=mcnemar_counts(x, y),
                                     ratio=c["arms"][a]["failures"] / kb if kb else float("inf"),
                                     paired=True, method="exact McNemar on summed shards")
        for a, b in UNPAIRED_COMPARISONS:
            if a in c["arms"] and b in c["arms"]:
                c["unpaired"][f"{a}|{b}"] = dict(
                    rate_ratio_ci(c["arms"][a]["failures"], n, c["arms"][b]["failures"], n),
                    a=a, b=b)
        out.append(c)
    return out


@app.function
def grid_config_id(cfg):
    """Short id of one grid configuration, e.g. 72-12-6_T12_p0.001_s0_OSD-0."""
    tag = cfg["code"].strip("[]").replace(", ", "-").replace(" ", "")
    return f"{tag}_T{int(cfg['rounds'])}_p{float(cfg['p']):g}_s{int(cfg.get('shard', 0))}_{cfg.get('decoder', 'OSD-0')}" + (
        f"_{cfg['tag']}" if cfg.get("tag") else "")


@app.function
def grid_stem(experiment, configs, master_seed, scaled=False):
    """Deterministic result-file stem of a grid run."""
    codes = sorted({c["code"] for c in configs})
    ps = sorted({float(c["p"]) for c in configs})
    Ts = sorted({int(c["rounds"]) for c in configs})
    decs = sorted({c.get("decoder", "OSD-0") for c in configs})
    arms = [a for a in FOUR_ARMS if any(a in c.get("arms", FOUR_ARMS) for c in configs)]
    shots = sum(int(c["shots"]) for c in configs)
    h = hashlib.sha256(json.dumps(configs, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return result_stem(experiment, code=codes, rounds=Ts, p=ps, shots=shots, seed=master_seed,
                       convention="z-only", scaled=scaled,
                       extra=["arms-" + "-".join(arms), "dec-" + "+".join(decs), f"cfg{h}"])


@app.function
def resume_stamp(path, stamp, config):
    """
    Continue an interrupted grid run: when `path` holds a PARTIAL file of the
    same configuration, reuse its run id and start time (so the file is
    updated, not superseded). Otherwise return `stamp` unchanged.
    """
    try:
        old = read_result_file(path)
    except (OSError, ValueError):
        return stamp
    if old.get("status") == "partial" and old.get("config") == json.loads(
            json.dumps(config, default=float)):
        return dict(stamp, run_id=old["run_id"], started=old["started"])
    return stamp


@app.function
def save_per_shot(path, results):
    """Per-shot arrays of every configuration and arm, bit-packed where boolean."""
    arrays = {}
    for r in results:
        cid = r["config_id"]
        for arm, d in (r.get("per_shot") or {}).items():
            for k, v in d.items():
                v = np.asarray(v)
                if v.dtype == bool:
                    arrays[f"{cid}__{arm}__{k}__packed"] = np.packbits(v)
                    arrays[f"{cid}__{arm}__{k}__n"] = np.array(len(v))
                else:
                    arrays[f"{cid}__{arm}__{k}"] = v
    side = os.path.splitext(path)[0] + "_pershot.npz"
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    durable_write(side, lambda f: f.write(buf.getvalue()))
    return side


@app.function
def load_per_shot(path):
    """config_id -> arm -> field -> array, from a <stem>_pershot.npz sidecar."""
    side = path if path.endswith("_pershot.npz") else os.path.splitext(path)[0] + "_pershot.npz"
    out = {}
    with np.load(side) as z:
        for key in z.files:
            parts = key.split("__")
            if parts[-1] == "n":
                continue
            cid, arm, fld = parts[0], parts[1], parts[2]
            if parts[-1] == "packed":
                n = int(z[f"{cid}__{arm}__{fld}__n"])
                v = np.unpackbits(z[key])[:n].astype(bool)
            else:
                v = z[key]
            out.setdefault(cid, {}).setdefault(arm, {})[fld] = v
    return out


@app.function
def run_grid(experiment, kind, configs, master_seed, workers=1, path=None, scaled=False,
             on_progress=None, extra_config=None):
    """
    Run a list of four-arm configurations as ONE result file. Each config is
    dict(code, rounds, p, shots, shard=0, decoder="OSD-0", arms=FOUR_ARMS,
    seed=None, decoder_spec=None, tag=None); a missing seed is derived from
    master_seed and (code, p, T, shard). After every configuration the file is
    rewritten with status "partial" (plus the per-shot sidecar), so a power cut
    loses at most the configuration in progress -- and that one resumes from
    its checkpoints. Returns (path, document, results-with-per-shot).
    """
    configs = [dict(c) for c in configs]
    for c in configs:
        c.setdefault("shard", 0)
        c.setdefault("decoder", "OSD-0")
        c["arms"] = list(c.get("arms", FOUR_ARMS))
        if c.get("seed") is None:
            c["seed"] = derive_seed(master_seed, c["code"], float(c["p"]), int(c["rounds"]),
                                    int(c["shard"]))
        c["config_id"] = grid_config_id(c)
    seen = {}
    for c in configs:                     # two configs of one cell (e.g. 50k and 10k) need
        seen.setdefault(c["config_id"], []).append(c)          # distinct ids, folders, keys
    for cid, group in seen.items():
        if len(group) > 1:
            for c in group:
                c["config_id"] = f"{cid}_n{int(c['shots'])}" + (
                    "_" + "-".join(c["arms"]) if sum(g["shots"] == c["shots"] for g in group) > 1
                    else "")
    stem = grid_stem(experiment, configs, master_seed, scaled)
    path = path or os.path.join(RESULTS_DIR, stem + ".json")
    decoders = {}
    for c in configs:
        decoders[c["decoder"]] = decoder_spec_full(c.get("decoder_spec") or c["decoder"])
    config = dict(codes=sorted({c["code"] for c in configs}),
                  rounds_list=sorted({int(c["rounds"]) for c in configs}),
                  ps=sorted({float(c["p"]) for c in configs}),
                  shots_per_config=[int(c["shots"]) for c in configs],
                  master_seed=int(master_seed), detector_convention="z-only",
                  x_detectors=False, idle_noise=False, decoders=decoders,
                  workers=int(workers), scaled=bool(scaled),
                  configs=[{k: v for k, v in c.items() if k != "decoder_spec"} | (
                      {"decoder_spec": decoder_spec_full(c["decoder_spec"])}
                      if c.get("decoder_spec") else {}) for c in configs],
                  **(extra_config or {}))
    config = json.loads(json.dumps(config, default=float))
    stamp = resume_stamp(path, start_run(kind, experiment), config)
    results, ckpts = [], []
    for c in configs:
        ck = os.path.join(CKPT["root"], stem, c["config_id"])
        ckpts.append(ck)
        r = four_arm_run(c["code"], c["rounds"], c["p"], c["shots"], c["seed"],
                         decoder=c["decoder"], arms=c["arms"], workers=workers,
                         shard=c["shard"], decoder_spec=c.get("decoder_spec"),
                         checkpoint_dir=ck if int(c["shots"]) >= CKPT["min_shots"] else False,
                         stem=stem, on_progress=on_progress, basis=c.get("basis", "Z"))
        r["config_id"] = c["config_id"]
        r["tag"] = c.get("tag")
        results.append(r)
        write_grid_file(path, stamp, config, results, "partial" if c is not configs[-1] else "complete")
    finalise_run(path, stamp, ckpts)
    try:
        os.rmdir(os.path.join(CKPT["root"], stem))      # empty once every config is cleared
    except OSError:
        pass
    return path, read_result_file(path), results


@app.function
def write_grid_file(path, stamp, config, results, status):
    timing = dict(workers_requested=config["workers"],
                  workers_effective=sorted({a["workers_effective"] for r in results
                                            for a in r["arms"].values()}),
                  backends=sorted({a["backend"].split(";")[0] for r in results
                                   for a in r["arms"].values()}),
                  resumed_chunks=sum(a["resumed_chunks"] for r in results for a in r["arms"].values()),
                  total_chunks=sum(a["total_chunks"] for r in results for a in r["arms"].values()))
    samples = {r["config_id"]: dict(flagged=r.get("sample_sha256"),
                                    unflagged=r["arms"].get("unflagged", {}).get("sample_sha256"))
               for r in results}
    write_result_file(path, dict(config=config, timing=timing, samples=samples,
                                 results=[{k: v for k, v in r.items() if k != "per_shot"}
                                          for r in results]), stamp, status)
    save_per_shot(path, results)


@app.function
def load_grid(path, kinds=None):
    """-> (doc, results, per_shot or None). Never writes."""
    d = read_result_file(path, kinds)
    side = os.path.splitext(path)[0] + "_pershot.npz"
    per = load_per_shot(side) if os.path.exists(side) else None
    return d, d["results"], per


@app.function
def grid_identity(r):
    """What makes two grid results the same configuration."""
    return (r["code"], int(r["rounds"]), round(float(r["p"]), 12), r["decoder"],
            int(r.get("shard", 0)), r.get("tag"))


@app.function
def merge_grids(loaded, paths=None):
    """
    Several result files of one experiment read as one -> (doc, results,
    per_shot). Used when an experiment is run in parts (e.g. Experiment 8's
    A1 and A3 single-threaded, the rest in parallel). Refuses files of
    different kinds and any configuration (code, T, p, decoder, shard, tag)
    that appears in more than one file, so a rerun can never be counted twice.
    """
    loaded = list(loaded)
    paths = list(paths) if paths is not None else [f"file {i + 1}" for i in range(len(loaded))]
    if not loaded:
        raise ValueError("no files to merge")
    if len(loaded) == 1:
        return loaded[0]
    kinds = {d.get("kind") for d, _r, _p in loaded}
    if len(kinds) > 1:
        raise ValueError(f"cannot merge files of different kinds: {sorted(map(str, kinds))}")
    owner, results, per, samples = {}, [], {}, {}
    for (d, res, ps), path in zip(loaded, paths):
        here = set()
        for r in res:
            k = grid_identity(r)
            if k in owner and owner[k] != path:
                raise ValueError(f"{k[0]} T={k[1]} p={k[2]:g} {k[3]} shard {k[4]}"
                                 f"{' (' + k[5] + ')' if k[5] else ''} is in both "
                                 f"{os.path.basename(owner[k])} and {os.path.basename(path)}; "
                                 "choose one of them")
            here.add(k)
            results.append(r)
        owner.update({k: path for k in here})
        for cid, arms in (ps or {}).items():
            per.setdefault(cid, {}).update(arms)
        samples.update(d.get("samples") or {})
    doc = dict(loaded[0][0], results=results, samples=samples,
               merged_from=[os.path.basename(p) for p in paths],
               master_seeds=sorted({d.get("config", {}).get("master_seed") for d, _r, _p in loaded},
                                   key=str))
    return doc, results, (per or None)


@app.function
def grid_rows(results):
    """Flat rows (one per configuration and arm) for tables and CSV."""
    rows = []
    for r in results:
        for arm, m in r["arms"].items():
            rows.append(dict(code=r["code"], rounds=r["rounds"], p=r["p"], shots=r["shots"],
                             shard=r.get("shard", 0), decoder=r["decoder"], arm=arm,
                             failures=m["failures"], ler=m["ler"], ci_low=m["ci_low"],
                             ci_high=m["ci_high"], time_p50_us=m.get("time_p50_us"),
                             time_p99_us=m.get("time_p99_us"), time_mean_us=m.get("time_mean_us"),
                             time_update_p50_us=m.get("time_update_p50_us"),
                             bp_iters_mean=m.get("bp_iters_mean"),
                             detectors=m.get("detectors"), qubits_used=m.get("qubits_used"),
                             workers=m.get("workers_effective"),
                             resumed_chunks=m.get("resumed_chunks"), seed=m.get("seed")))
    return rows


@app.function
def comparison_rows(results):
    """One row per configuration and comparison, paired and unpaired."""
    rows = []
    for r in results:
        for key_, c in r["paired"].items():
            rows.append(dict(code=r["code"], rounds=r["rounds"], p=r["p"], shots=r["shots"],
                             decoder=r["decoder"], comparison=key_, paired=True,
                             a_only=c["a_only"], b_only=c["b_only"], ratio=c["ratio"],
                             lo=None, hi=None, p_value=c["p"], method=c["method"]))
        for key_, c in r["unpaired"].items():
            rows.append(dict(code=r["code"], rounds=r["rounds"], p=r["p"], shots=r["shots"],
                             decoder=r["decoder"], comparison=key_, paired=False,
                             a_only=None, b_only=None, ratio=c["ratio"], lo=c["lo"], hi=c["hi"],
                             p_value=c["p"], method=c["method"]))
    return rows


@app.function
def grid_table_md(results):
    """Markdown table: LER with CI and median time per arm, per configuration."""
    head = ("| code | T | p | shots | decoder | arm | failures | LER [95% CI] | "
            "median µs incl. prior update (update part) | p99 µs | BP iters | workers |")
    lines = [head, "|:--|--:|--:|--:|:--|:--|--:|:--|--:|--:|--:|--:|"]
    for row in grid_rows(results):
        upd = f" ({row['time_update_p50_us']:,.0f})" if row["time_update_p50_us"] else ""
        lines.append(
            f"| {row['code']} | {row['rounds']} | {row['p']:g} | {row['shots']:,} | "
            f"{row['decoder']} | {row['arm']} | {row['failures']} | {row['ler']:.2e} "
            f"[{row['ci_low']:.2e}, {row['ci_high']:.2e}] | {row['time_p50_us']:,.0f}{upd} | "
            f"{row['time_p99_us']:,.0f} | {row['bp_iters_mean']:.1f} | {row['workers']} |")
    return "\n".join(lines)


@app.function
def comparison_table_md(results):
    lines = ["| code | T | p | decoder | comparison | test | split / ratio [95% CI] | p |",
             "|:--|--:|--:|:--|:--|:--|:--|--:|"]
    for row in comparison_rows(results):
        if row["paired"]:
            eff = f"{row['a_only']} : {row['b_only']} discordant"
            test = "paired McNemar"
        else:
            eff = f"{row['ratio']:.3g} [{row['lo']:.3g}, {row['hi']:.3g}]"
            test = "unpaired rate ratio"
        lines.append(f"| {row['code']} | {row['rounds']} | {row['p']:g} | {row['decoder']} | "
                     f"{row['comparison'].replace('|', ' vs ')} | {test} | {eff} | "
                     f"{row['p_value']:.2e} |")
    return "\n".join(lines)


@app.function
def grid_picker(experiment, legacy_prefixes=(), legacy_kinds=None):
    """Dropdown over the saved runs of a grid experiment."""
    return saved_run_picker(experiment, f"Saved Experiment {experiment[1:].lstrip('0')} run",
                            legacy_prefixes=legacy_prefixes, legacy_kinds=legacy_kinds)


@app.function
def run_status_banner(doc, path):
    """Status line for a run or a loaded file (partial runs are flagged)."""
    st_ = doc.get("status", "older file")
    t = doc.get("timing", {})
    msg = (f"`{os.path.basename(path)}` · status **{st_}** · started {doc.get('started')} · "
           f"workers {t.get('workers_effective')} · resumed chunks {t.get('resumed_chunks', 0)}"
           f"/{t.get('total_chunks', '?')}")
    return mo.callout(mo.md(msg), kind="success" if st_ == "complete" else "warn")


@app.function
def grid_design_ui(design, master_seed, default=None, label="configurations"):
    """Controls for a grid experiment: which registered configurations to run,
    a shot scale (anything but 1 is marked -scaled and never cited), the
    master seed and the worker count."""
    return mo.ui.dictionary(dict(
        configs=mo.ui.multiselect(list(design), value=list(default or design), label=label),
        scale=mo.ui.dropdown(list(SHOT_SCALES), value="1 (registered design)", label="shots"),
        master_seed=mo.ui.number(value=int(master_seed), label="master seed"),
        workers=mo.ui.slider(1, max(2, os.cpu_count() or 1), value=1, show_value=True,
                             label="workers (1 for any run that feeds a latency claim)"),
    ))


@app.function
def grid_setup(value, design):
    """(configs, scaled, master_seed, workers) from grid_design_ui's value."""
    scale = SHOT_SCALES[value["scale"]]
    cfgs = []
    for cid in value["configs"]:
        c = dict(design[cid], design_id=cid)
        c["shots"] = max(20, int(round(c["shots"] * scale)))
        cfgs.append(c)
    return cfgs, scale != 1.0, int(value["master_seed"]), int(value["workers"])


@app.function
def design_table_md(design, us_per_shot=None):
    """The registered design as a table."""
    lines = ["| id | code | T | p | shots | shard | decoder | arms | purpose |",
             "|:--|:--|--:|--:|--:|--:|:--|:--|:--|"]
    for cid, c in design.items():
        lines.append(f"| {cid} | {c['code']}{' — X basis' if c.get('basis') == 'X' else ''} | "
                     f"{c['rounds']} | {c['p']:g} | {c['shots']:,} | "
                     f"{c.get('shard', 0)} | {c.get('decoder', 'OSD-0')} | "
                     f"{', '.join(c.get('arms', FOUR_ARMS))} | {c.get('purpose', '')} |")
    return "\n".join(lines)


@app.function
def grid_run_cell(experiment, kind, design_value, design, tests_ok, run_clicked, label,
                  extra_config=None):
    """
    The body of every grid experiment's Run cell. Returns (data or None,
    output). Writes a result file only when the button was pressed.
    """
    if not tests_ok:
        return None, mo.md(f"*{label} locked until every test passes.*")
    if not run_clicked:
        return None, mo.md(f"*Press **Run {label}**, or load a saved run below.*")
    cfgs, scaled, master, workers = grid_setup(design_value, design)
    if not cfgs:
        return None, mo.md("*Select at least one configuration.*")
    total = sum(c["shots"] * len(c.get("arms", FOUR_ARMS)) for c in cfgs)
    with mo.status.progress_bar(total=total, title=label, show_eta=True) as bar:
        path, doc, results = run_grid(experiment, kind, cfgs, master, workers=workers,
                                      scaled=scaled, extra_config=extra_config,
                                      on_progress=lambda _arm, n: bar.update(increment=n))
    return dict(path=path, doc=doc, results=doc["results"], per_shot=None, fresh=True), \
        mo.md(f"Saved `{path}`" + (" — **scaled run: rehearsal only, never cited**" if scaled else ""))


@app.cell
def _():
    def _c72():
        return bb_from_preset("[[72, 12, 6]]", BB_PRESETS)

    def _tmpd():
        import tempfile
        return tempfile.mkdtemp()

    def _update_inside_timer():
        """the update is timed, and time = update + decode exactly."""
        r = four_arm_run(_c72(), 3, 2e-3, 60, 11, arms=("updated",), checkpoint_dir=False)
        m = r["arms"]["updated"]
        ps_ = r["per_shot"]["updated"]
        assert m["time_update_p50_us"] > 0
        assert abs(m["time_p50_us"] - float(np.median(ps_["time_update_us"] + ps_["time_decode_us"]))) < 1e-6
        assert "bp_iters_mean" in m and m["workers_effective"] == 1

    def _pairing_and_seeds():
        """The flagged arms share one sample set; unflagged has its own derived seed."""
        r = four_arm_run(_c72(), 3, 2e-3, 40, 5, checkpoint_dir=False)
        hs = {a: r["arms"][a]["sample_sha256"] for a in r["arms"]}
        assert hs["blind"] == hs["sighted"] == hs["updated"] == r["sample_sha256"]
        assert hs["unflagged"] != hs["blind"]
        assert r["arms"]["unflagged"]["seed"] == derive_seed(5, "unflagged") != 5
        assert set(r["paired"]) == {"blind|sighted", "sighted|updated", "blind|updated"}
        assert set(r["unpaired"]) == {"sighted|unflagged", "blind|unflagged", "updated|unflagged"}
        assert r["pairing"]["blind|sighted"] is True and r["pairing"]["sighted|unflagged"] is False
        assert all(not c["paired"] for c in r["unpaired"].values())
        assert r["decoder_spec"] == decoder_spec_full("OSD-0") and r["decoder_spec"]["max_iter"] == 30
        assert r["arms"]["blind"]["detectors"] == r["arms"]["sighted"]["detectors"] - r["overhead"]["flag_detectors"]

    def _wrapper_matches_runner():
        """the Experiment 7 wrapper decodes the same samples."""
        a = pareto_point(_c72(), 3, 2e-3, 40, 9, "blind", 0, checkpoint_dir=False)
        b = four_arm_run(_c72(), 3, 2e-3, 40, 9, arms=("blind",), checkpoint_dir=False)
        assert a["sample_sha256"] == b["arms"]["blind"]["sample_sha256"]
        assert a["failures"] == b["arms"]["blind"]["failures"]
        c = prior_update_run(_c72(), 3, 2e-3, 40, 9, checkpoint_dir=False)
        assert c["arms"]["blind"]["failures"] == a["failures"]

    def _resume_reproduces():
        """The second call on the same checkpoint folder resumes every chunk."""
        d = _tmpd()
        a = four_arm_run(_c72(), 3, 2e-3, 300, 4, arms=("blind", "updated"), checkpoint_dir=d)
        b = four_arm_run(_c72(), 3, 2e-3, 300, 4, arms=("blind", "updated"), checkpoint_dir=d)
        for arm in ("blind", "updated"):
            assert b["arms"][arm]["resumed_chunks"] == b["arms"][arm]["total_chunks"] == 2
            assert b["arms"][arm]["failures"] == a["arms"][arm]["failures"]
            assert np.array_equal(b["per_shot"][arm]["fail"], a["per_shot"][arm]["fail"])
            assert b["arms"][arm]["timing_note"] == "times partly from an earlier session"

    def _interleaved_matches_arm_by_arm():
        """Arms in turns per chunk give the same per-shot outcomes as arm by arm,
        rotate their order chunk by chunk, and record the order used."""
        kw = dict(checkpoint_dir=False)
        a = four_arm_run(_c72(), 3, 2e-3, 600, 8, interleave=True, **kw)
        b = four_arm_run(_c72(), 3, 2e-3, 600, 8, interleave=False, **kw)
        assert a["timing_order"].startswith("interleaved") and b["timing_order"] == "arm by arm"
        for arm in FOUR_ARMS:
            assert np.array_equal(a["per_shot"][arm]["fail"], b["per_shot"][arm]["fail"]), arm
            assert a["arms"][arm]["sample_sha256"] == b["arms"][arm]["sample_sha256"]
            assert a["arms"][arm]["workers_effective"] == 1
        assert a["paired"] == b["paired"]
        c = four_arm_run(_c72(), 3, 2e-3, 300, 8, arms=("blind", "updated"), **kw)
        assert c["timing_order"].startswith("interleaved")          # automatic at workers = 1
        d = four_arm_run(_c72(), 3, 2e-3, 300, 8, arms=("blind",), **kw)
        assert d["timing_order"] == "single arm"

    def _merge_grids_refuses_duplicates():
        base = dict(code="c", rounds=12, p=1e-3, decoder="OSD-0", shard=0, arms={"blind": {}},
                    config_id="A1")
        d1 = (dict(kind="E08-prior-update", config={}), [base], {"A1": {"blind": {"x": 1}}})
        d2 = (dict(kind="E08-prior-update", config={}), [dict(base, p=2e-3, config_id="A2")],
              {"A2": {"blind": {"x": 2}}})
        doc, res, per = merge_grids([d1, d2], ["a.json", "b.json"])
        assert len(res) == 2 and set(per) == {"A1", "A2"} and doc["merged_from"] == ["a.json", "b.json"]
        for bad in ([d1, d1], [d1, (dict(kind="E02-augmentation", config={}), [], None)]):
            try:
                merge_grids(bad, ["a.json", "c.json"])
            except ValueError:
                continue
            raise AssertionError("merged a duplicate configuration or mixed kinds")

    def _x_basis_runs_and_is_labelled():
        r = four_arm_run(_c72(), 2, 2e-3, 30, 3, arms=("blind", "updated"), basis="X",
                         checkpoint_dir=False)
        assert r["basis"] == "X" and r["code"].endswith("(X basis)")
        assert r["arms"]["updated"]["failures"] <= 30

    def _small_runs_are_single_process():
        """below 2,000 shots the pool is not used."""
        r = four_arm_run(_c72(), 3, 2e-3, 50, 3, arms=("sighted",), workers=4, checkpoint_dir=False)
        assert r["arms"]["sighted"]["workers_effective"] == 1

    def _updated_priors_change_per_shot():
        c = _c72()
        circ = build_memory_circuit(c, 3, 2e-3, use_flags=True, x_detectors=False)
        H, _L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(c, 3, True, circ.num_detectors, x_detectors=False)
        idx, ok, _ = flag_groups(H, mask)
        lo, hi, owner = flag_posteriors(pr, idx)
        f0 = np.zeros(int(mask.sum()), bool)
        f1 = f0.copy()
        f1[0] = True
        a, b = shot_priors(lo, hi, owner, f0), shot_priors(lo, hi, owner, f1)
        assert ok and not np.allclose(a, b) and np.allclose(a[owner < 0], pr[owner < 0])

    def _seeds_are_derived():
        s = {derive_seed(7, c, p, T, sh) for c in ("[[72, 12, 6]]", "[[144, 12, 12]]")
             for p in (1e-3, 3e-3) for T in (12,) for sh in (0, 1)}
        assert len(s) == 8 and all(0 <= x < 2 ** 63 for x in s)
        assert derive_seed(7, "[[72, 12, 6]]", 1e-3, 12, 0) == derive_seed(7, "[[72, 12, 6]]", 1e-3, 12, 0)

    def _shards_combine():
        r1 = dict(code="c", rounds=1, p=1e-3, decoder="OSD-0", shots=100, shard=0, seed=1,
                  arms={"sighted": dict(failures=5, workers_effective=1),
                        "updated": dict(failures=2, workers_effective=1)},
                  paired={"sighted|updated": dict(a_only=4, b_only=1)}, unpaired={})
        r2 = dict(r1, shard=1, seed=2, paired={"sighted|updated": dict(a_only=6, b_only=0)})
        c = combine_shards([r1, r2])[0]
        assert c["shots"] == 200 and c["arms"]["sighted"]["failures"] == 10
        pr = c["paired"]["sighted|updated"]
        assert (pr["a_only"], pr["b_only"]) == (10, 1)
        assert abs(pr["p"] - 2 * sum(comb(11, i) for i in range(2)) / 2 ** 11) < 1e-12
        assert mcnemar_counts(0, 10) == 2 / 2 ** 10

    def _grid_writes_partial_then_complete():
        d = _tmpd()
        old_root, old_min = CKPT["root"], CKPT["min_shots"]
        configure_checkpoints(root=os.path.join(d, "checkpoints"), min_shots=100)
        try:
            cfgs = [dict(code="[[72, 12, 6]]", rounds=2, p=2e-3, shots=120, arms=("blind", "sighted")),
                    dict(code="[[72, 12, 6]]", rounds=2, p=3e-3, shots=60, arms=("unflagged", "updated"))]
            path = os.path.join(d, "E00_grid.json")
            p_, doc, res = run_grid("E00", "E00-test", cfgs, master_seed=99, path=path)
            assert doc["status"] == "complete" and doc["checkpoint_cleared"]
            assert len(doc["results"]) == 2 and doc["config"]["master_seed"] == 99
            assert not os.listdir(os.path.join(d, "checkpoints")), "checkpoints left behind"
            per = load_per_shot(path)
            cid = res[0]["config_id"]
            assert np.array_equal(per[cid]["blind"]["fail"], res[0]["per_shot"]["blind"]["fail"])
            before = (os.path.getmtime(path), file_sha256(path))
            _d, _r, _p = load_grid(path)
            grid_table_md(_r), comparison_table_md(_r), grid_rows(_r)
            assert (os.path.getmtime(path), file_sha256(path)) == before, "viewing rewrote the file"
            assert res[0]["seed"] == derive_seed(99, "[[72, 12, 6]]", 2e-3, 2, 0)
        finally:
            configure_checkpoints(root=old_root, min_shots=old_min)

    def _stems_differ_by_arms_and_orders():
        """Stems differ when only the arms or the decoder differ."""
        base = [dict(code="[[72, 12, 6]]", rounds=12, p=1e-3, shots=10)]
        a = grid_stem("E07", [dict(base[0], arms=["blind"], decoder="OSD-0")], 1)
        b = grid_stem("E07", [dict(base[0], arms=["updated"], decoder="OSD-0")], 1)
        c = grid_stem("E07", [dict(base[0], arms=["blind"], decoder="OSD-CS2")], 1)
        assert len({a, b, c}) == 3 and all(s.endswith(s[-11:]) for s in (a, b, c))

    def _relay_arm_runs():
        r = four_arm_run(_c72(), 2, 2e-3, 12, 3, decoder="relay", arms=("blind", "updated"),
                         checkpoint_dir=False,
                         decoder_spec=dict(kind="relay", num_sets=4, set_max_iter=20, pre_iter=20))
        assert r["arms"]["updated"]["time_update_p50_us"] > 0

    tests_runner = run_checks([
        ("updated arm: prior update inside the timer, time = update + decode", _update_inside_timer),
        ("flagged arms share samples; unflagged has its own derived seed; one decoder spec", _pairing_and_seeds),
        ("Experiment 7/8 wrappers decode the same samples as the runner", _wrapper_matches_runner),
        ("a resumed configuration reproduces failures and says so", _resume_reproduces),
        ("runs below 2,000 shots stay single-process", _small_runs_are_single_process),
        ("X-basis runs through the same runner and is labelled", _x_basis_runs_and_is_labelled),
        ("single-process runs decode the arms in turns, with identical outcomes", _interleaved_matches_arm_by_arm),
        ("several files of one experiment merge; duplicate configurations are refused", _merge_grids_refuses_duplicates),
        ("the updated arm's priors depend on the flags", _updated_priors_change_per_shot),
        ("configuration seeds are derived, distinct and reproducible", _seeds_are_derived),
        ("shards combine by summing discordant counts", _shards_combine),
        ("grid runs write partial then complete; viewing never rewrites", _grid_writes_partial_then_complete),
        ("result stems differ by arms and decoder", _stems_differ_by_arms_and_orders),
        ("Relay-BP arm runs through the same runner (built-in backend)", _relay_arm_runs),
    ])
    render_checks("15 four-arm runner", tests_runner)
    return (tests_runner,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 16. Planning, scope and implementation status

    ### Graph preview

    A quick end-to-end run on [[72, 12, 6]] to check the pipeline and the graphs.
    **Not a thesis result** — 300 shots, T = 3, the corrected (Z-only) detector
    convention. It unlocks once every test in §2–§15 passes.
    """)
    return


@app.cell
def _():
    run_preview = mo.ui.run_button(label="Run graph preview (~1–2 min)")
    run_preview
    return (run_preview,)


@app.cell
def _():
    preview_code = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
    preview_rounds = 3

    def make_preview_policies(p, names=None):
        """The preview policies on one circuit at noise p (flags on, Z-only detectors)."""
        circ = build_memory_circuit(preview_code, preview_rounds, p, use_flags=True,
                                    x_detectors=False)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(preview_code, preview_rounds, True, circ.num_detectors,
                                  x_detectors=False)
        peel, bp, osd = PeelingDecoder(H, pr), BpDecoder(H, pr), BpOsdDecoder(H, pr)
        pol = {
            "peeling only": SwitchPolicy(peel, osd, "never"),
            "always BP+OSD": SwitchPolicy(peel, osd, "always"),
            "peel → OSD on fail": SwitchPolicy(peel, osd, "primary_fail"),
            "peel → OSD on flag or fail": SwitchPolicy(peel, osd, "flag>=1", mask),
            "peel → OSD on flag only": SwitchPolicy(peel, osd, "flag-only", mask),
            "BP → OSD on flag or fail": SwitchPolicy(bp, osd, "flag>=1", mask),
        }
        if names is not None:
            pol = {n: pol[n] for n in names}
        return circ, L, pol

    return make_preview_policies, preview_code, preview_rounds


@app.cell
def _(
    core_ready,
    make_preview_policies,
    preview_code,
    preview_rounds,
    run_preview,
):
    mo.stop(not core_ready, mo.md("*Preview locked: every test in §2–§15 must pass.*"))
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
        mo.md(f"#### Preview — {preview_code.name}, T = {preview_rounds}, p = {_p}, 300 shots "
              "(not a thesis result)"),
        mo.md("\n".join(_rows)),
        plot_ablation(preview_results), plot_outcomes(preview_results),
        plot_latency(preview_results, workers=1), plot_tradeoff(preview_results),
    ])
    return


@app.cell
def _(
    core_ready,
    make_preview_policies,
    preview_code,
    preview_rounds,
    run_preview,
):
    mo.stop(not core_ready or not run_preview.value)
    _names = ["always BP+OSD", "peel → OSD on fail", "peel → OSD on flag or fail"]
    preview_sweep = {}
    for _p in (1e-3, 2e-3, 4e-3):
        _circ, _L, _pol = make_preview_policies(_p, _names)
        preview_sweep[_p] = run_benchmark(_circ, _pol, _L, shots=150, seed=11)
    mo.vstack([mo.md(f"#### Preview sweep — {preview_code.name}, T = {preview_rounds}, "
                     "150 shots/point"), plot_sweep(preview_sweep)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Scope: what is simulated and what is not

    **Real, and the results stand on it.** The decoders are production
    implementations (Roffe's `ldpc` for BP, BP+OSD and BP+LSD; `pymatching` for
    MWPM; a built-in port of IBM's `relay-bp` reference for Relay-BP), and every correction a strong decoder returns
    reproduces its syndrome. The noise is Stim circuit-level noise, gate by gate.
    Failures are scored as $L\hat e \neq$ obs against Stim's sampled observables.

    **Idealised, and claims are bounded accordingly.**

    - **Timing is CPU wall-clock, not latency.** `time.perf_counter()` around a
      Python call. Times are compared only between decoders measured in the same
      run at the same worker count; latency claims use workers = 1 runs on an
      otherwise idle machine. The break-even criterion of Experiment 4 is a
      ratio for exactly this reason.
    - **The pipeline is offline, not streaming.** A whole $T$-round block is
      sampled, then decoded; the backlog problem is motivation, not a
      measurement.
    - **The flag is a detector bit, not a hardware signal.** Routing it to the
      controller is a latency this work does not model.
    - **Policies are derived, not executed.** Experiment 4 decodes each shot
      once per decoder and then selects. A test asserts shot by shot that this
      equals running `SwitchPolicy`; the reported cost is a reconstruction under
      one stated convention (§13).
    - **No idle noise.** Every circuit has gate, measurement and reset noise
      only — the same model as the 250,000-shot decoder grid of Experiment 4, so the
      new runs stay comparable with it.

    Not modelled: pipelining, leakage, crosstalk, drift, readout latency.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Timing runs: one core, a quiet machine

    Failure counts do not depend on what else the computer is doing; **decode
    times do**. Other programs compete for the core, its cache and memory
    bandwidth, and the clock speed follows temperature and load. Medians are
    fairly robust to light use; p99 values are not. Three things keep the timed
    runs (Experiment 4's re-timing, Experiment 7, Experiment 8's A1 and A3) clean:

    1. **Arms in turns.** A single-process run decodes its arms chunk by chunk in
       a rotating order (§15), so a slow spell hits every arm alike and cannot
       bias a paired time comparison.
    2. **Pin to a core** (below). With hyperthreading, logical cores come in
       pairs on one physical core (0–1, 2–3, … on most machines): pin to one of
       a pair and keep the other free. On Intel CPUs with performance and
       efficiency cores, pin to a performance core. Worker processes started
       afterwards inherit the setting, so for a parallel run select at least as
       many cores as workers — or unpin.
    3. **Keep other programs off that core and the clock steady** — outside this
       notebook: on Windows, Process Lasso's *CPU Sets* reserve cores for one
       program; plug a laptop in and use the High / Best performance power plan
       (setting *Maximum processor state* to 99% turns off turbo boost: slower
       but steadier times); pause Windows Update and exclude `results/` from
       antivirus scanning.

    Every result file records the cores, power plan, clock, battery state and
    machine load at the start of the run (`environment.timing`).
    """)
    return


@app.cell
def _():
    _aff = cpu_affinity() or list(range(os.cpu_count() or 1))
    ui_pin_cores = mo.ui.multiselect([str(c) for c in range(os.cpu_count() or 1)],
                                     value=[str(c) for c in _aff], label="logical cores")
    ui_pin_btn = mo.ui.run_button(label="Apply")
    ui_unpin_btn = mo.ui.run_button(label="Use every core")
    mo.hstack([ui_pin_cores, ui_pin_btn, ui_unpin_btn], justify="start")
    return ui_pin_btn, ui_pin_cores, ui_unpin_btn


@app.cell
def _(ui_pin_btn, ui_pin_cores, ui_unpin_btn):
    _msg = ""
    try:
        if ui_unpin_btn.value:
            pin_to_cores(None)
            _msg = "Unpinned."
        elif ui_pin_btn.value:
            pin_to_cores([int(c) for c in ui_pin_cores.value])
            _msg = "Pinned."
    except Exception as _e:
        _msg = f"**Could not change the cores:** {type(_e).__name__}: {_e}"
    _t = timing_environment()
    mo.md((_msg + "\n\n" if _msg else "") + f"""
    | | |
    |:--|:--|
    | cores this notebook may use | {", ".join(map(str, _t["affinity"])) if _t["affinity"] is not None else "unknown"} ({"pinned" if _t["pinned"] else "not pinned"}) |
    | logical / physical cores | {_t["logical_cores"]} / {_t["physical_cores"] or "?"} |
    | power plan | {_t["power_plan"] or "unknown"} |
    | clock now / max | {_t["cpu_mhz"] or "?"} / {_t["cpu_max_mhz"] or "?"} MHz |
    | on battery | {"unknown" if _t["on_battery"] is None else ("**yes — plug in for timed runs**" if _t["on_battery"] else "no")} |
    | machine load | {"?" if _t["load_percent"] is None else f"{_t['load_percent']:.0f}%"} |
    """ + ("" if psutil is not None else "\n*psutil is not installed (`pip install psutil`): "
            "pinning works only on Linux without it, and clock, battery and load stay unknown.*"))
    return


@app.function
def shots_for_failures(ler, target=50, floor=2000, cap=None):
    """
    Shots needed to observe about `target` failures at a given logical error
    rate. Precision is set by the FAILURE count: ~10 failures place a point,
    ~30 make it plottable, ~100 make it quotable.
    """
    if not 0 < ler <= 1:
        raise ValueError(f"ler must be in (0, 1], got {ler}")
    n = int(max(floor, round(target / ler)))
    return int(min(n, cap)) if cap else n


@app.function
def wilson_halfwidth(ler, shots, z=1.96):
    """Half-width of the Wilson interval, for sizing a run before paying for it."""
    d = 1 + z * z / shots
    return float(z * np.sqrt(ler * (1 - ler) / shots + z * z / (4 * shots ** 2)) / d)


@app.function
def core_hours(shots, us_per_shot):
    """Cost of a run in core-hours: shots x per-shot µs / 3.6e9."""
    return float(shots) * float(us_per_shot) / 3.6e9


@app.cell
def _():
    def _planner_arithmetic():
        assert shots_for_failures(1e-2, target=50, floor=0) == 5000
        assert shots_for_failures(1e-3, target=50, floor=0) == 50000
        assert shots_for_failures(1e-1, target=50, floor=2000) == 2000
        assert shots_for_failures(1e-3, target=50, cap=10000) == 10000
        for bad in (0.0, -1, 2.0):
            try:
                shots_for_failures(bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted ler={bad}")
        assert abs(core_hours(50000, 13790) - 0.19) < 0.005

    def _halfwidth_shrinks_with_shots():
        hw = [wilson_halfwidth(3e-3, n) for n in (1000, 10000, 100000)]
        assert hw == sorted(hw, reverse=True), hw
        assert wilson_halfwidth(3e-3, shots_for_failures(3e-3, target=50)) < 3e-3

    def _gross_code_runs_end_to_end():
        """[[144,12,12]] at T = 12 decodes end to end before a long run."""
        c = bb_from_preset("[[144, 12, 12]]", BB_PRESETS)
        circ = build_memory_circuit(c, 12, 1e-3, use_flags=True, x_detectors=False)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        ratio = H.shape[1] / theory_faults(c, 12)["total"]
        assert ratio < 3, f"Gross DEM is {ratio:.1f}x the analytic fault count"
        det = circ.compile_detector_sampler(seed=8).sample(12).astype(np.uint8)
        osd = make_decoder("OSD-0", H, pr)
        for sdr in det:
            e, _c, _w = osd.decode(sdr)
            assert np.array_equal((H @ e) % 2, sdr)

    tests_planning = run_checks([
        ("shot planner arithmetic, floor, cap and the core-hour formula", _planner_arithmetic),
        ("Wilson half-width shrinks, and 50 failures resolve the rate", _halfwidth_shrinks_with_shots),
        ("[[144, 12, 12]] decodes end to end before a long run", _gross_code_runs_end_to_end),
    ])
    render_checks("16 planning", tests_planning)
    return (tests_planning,)


@app.cell
def _(
    tests_analysis,
    tests_circuit,
    tests_codes,
    tests_decoders,
    tests_dem,
    tests_engine,
    tests_flags,
    tests_gb,
    tests_gf2,
    tests_persistence,
    tests_planning,
    tests_plotting,
    tests_reference,
    tests_runner,
    tests_stats,
    tests_surface,
    tests_switch,
):
    _sections = [
        ("2 GF(2)", tests_gf2), ("3 BB codes", tests_codes), ("3 GB codes", tests_gb),
        ("4 Circuits", tests_circuit), ("5 DEM", tests_dem), ("5 Reference model", tests_reference),
        ("6 Surface baseline", tests_surface), ("7 Decoders", tests_decoders),
        ("8 Switch policies", tests_switch), ("9 Statistics", tests_stats),
        ("10 Engine and checkpoints", tests_engine), ("11 Flag structure", tests_flags),
        ("12 Result files", tests_persistence), ("13 Policies and verdicts", tests_analysis),
        ("14 Figures", tests_plotting), ("15 Four-arm runner", tests_runner),
        ("16 Planning", tests_planning),
    ]
    _rows = ["| section | " + " | ".join(STATUS_ICON.values()) + " |", "|:--|--:|--:|--:|"]
    for _name, _res in _sections:
        _c = [sum(r["status"] == s for r in _res) for s in STATUS_ICON]
        _rows.append(f"| {_name} | " + " | ".join(map(str, _c)) + " |")
    core_ready = all_pass(*[r for _, r in _sections])
    mo.vstack([
        mo.md("### Implementation status"),
        mo.md("\n".join(_rows)),
        mo.md("**All tests pass — experiments unlocked.**" if core_ready else
              "Experiments stay locked until every test above passes."),
    ])
    return (core_ready,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Experiments

    Numbered in thesis order (§1 has the map). Every experiment has the same
    shape: a design paragraph with its decision rule, its own self-tests, the
    controls, a **Run** button (the only cell that writes a result file), a
    picker that **loads** a saved run (displayed, never rewritten), the view,
    and an export that runs once for a fresh run or on an explicit
    **Re-export** for a loaded one.

    ## Experiment 1 — noise-model validation and peeling baseline (Table 5.1)

    **Why this experiment.** Every later experiment samples from detector error
    models built by the same circuit and noise code, whichever decoder it then
    runs. The first question is therefore whether that model has the size and
    density the theory predicts. Pakhunov (2026) gives closed forms for a
    bivariate-bicycle memory under circuit-level noise: the number of fault
    mechanisms (Theorem 1), the expected number firing per shot $\lambda$
    (Theorem 2), and — from the fault-graph degree $\bar d$ — the probability
    that greedy peeling clears a syndrome on its own (Theorems 3 and 5, Table II).

    The experiment has two parts that matter to different chapters:

    * **Part A (all experiments).** Fault count, $\lambda$ and $\bar d$ of the
      reference circuit against Theorems 1 and 2 — decoder-independent, so it
      validates the noise model that Experiments 2–11 all share. It needs no
      sampling.
    * **Part B (Experiments 4 and 5 only).** How often our peeler succeeds,
      against the analytic prediction and Table II. Peeling is only the fast
      primary of the switching experiments, so Part B qualifies those results and
      nothing else. The decoder pipeline used for the decoder-input experiments
      (BP+OSD, BP+LSD, Relay-BP) is validated by Experiment 6 instead.

    **Design.** [[72,12,6]], [[144,12,12]] and [[288,12,18]] on the reference
    circuit (Z-check extraction only, no flags, DEPOLARIZE2 per CNOT) at
    $p = 10^{-3}$, $T = 12$, 100,000 shots each with a recorded seed, decoded by
    three peeler variants on the same shots: `peel` (queue + dominance rule; the
    one used everywhere else), `peel-naive` (uniqueness rule) and `peel-single`
    (one removal pass). The prediction is given **with and without** Eq. (2)'s
    density correction $e^{-BTp}$, beside the single-pass bound
    $e^{-\beta\lambda^2}$. The circuit variants (one fault per CNOT, correlated
    XX, both check families) are exploratory.

    **Statistics.** Each success rate is a binomial proportion with a 95% Wilson
    interval. **Decision rule:** a peeler variant is named as matching a target
    only if its interval contains the single-pass bound or the Table II value;
    otherwise the gap is reported as unexplained.
    """)
    return


@app.cell
def _():
    E01_CODES = ["[[72, 12, 6]]", "[[144, 12, 12]]", "[[288, 12, 18]]"]
    E01_PEELERS = ["peel", "peel-naive", "peel-single"]
    E01_MASTER_SEED = 2026110101
    return E01_CODES, E01_MASTER_SEED, E01_PEELERS


@app.function
def e01_measure(code_key, rounds, p, shots, seed, peelers=("peel",), extraction="z-only",
                gate_noise="depolarize2", workers=1, checkpoint_dir=None):
    """
    One circuit variant decoded by every peeler in `peelers` on the SAME
    shots. Returns one row per peeler: raw resolved count k, Wilson interval,
    and every theorem's prediction (with and without e^{-BTp}).
    """
    code = code_from_key(code_key)
    rep = reference_report(code, rounds, p, extraction=extraction, gate_noise=gate_noise)
    circ = build_reference_circuit(code, rounds, p, extraction=extraction, gate_noise=gate_noise)
    H, L, priors = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
    det, obs = circ.compile_detector_sampler(seed=int(seed)).sample(int(shots),
                                                                     separate_observables=True)
    det, obs = det.astype(np.uint8), obs.astype(np.uint8)
    w = parallel_workers(shots, workers)
    res, backend, info = run_zoo(H, L, priors, det, obs, list(peelers), workers=w,
                                 chunk_size=CHUNK_SIZE if checkpoint_dir else chunk_for(shots, w),
                                 checkpoint_dir=checkpoint_dir if checkpoint_dir else False,
                                 return_info=True,
                                 label=f"E01 {code_key} T={rounds} p={p:g} {extraction}/{gate_noise}")
    target = TABLE_II.get((code_key, int(rounds), float(p)))
    rows = []
    for name in peelers:
        k = int(res[name]["conv"].sum())
        rate, lo, hi = wilson(k, shots)
        rows.append(dict(code=code_key, rounds=int(rounds), p=float(p), shots=int(shots),
                         seed=int(seed), extraction=extraction, gate_noise=gate_noise,
                         peeler=name, peeler_spec=decoder_spec_full(name), resolved=k,
                         peel=rate, peel_lo=lo, peel_hi=hi,
                         failures=int(res[name]["fail"].sum()),
                         time_p50_us=float(np.median(res[name]["time_us"])),
                         table_ii=target, sample_sha256=sample_hash(det, obs),
                         workers_effective=info["workers_effective"], backend=backend,
                         resumed_chunks=info["resumed_chunks"], total_chunks=info["total_chunks"],
                         **rep))
    return rows


@app.function
def e01_decision(rows):
    """
    Per (code, circuit variant): which peeler's Wilson interval contains the
    single-pass bound or the Table II value. Returns one dict per group with
    `named` (list of peelers) and a verdict string.
    """
    groups = {}
    for r in rows:
        groups.setdefault((r["code"], r["rounds"], r["p"], r["extraction"], r["gate_noise"]),
                          []).append(r)
    out = []
    for (code, T, p, ex, gn), rs in groups.items():
        sp_ = rs[0]["single_pass_bound"]
        tgt = rs[0].get("table_ii")
        named = []
        for r in rs:
            hits = []
            if r["peel_lo"] <= sp_ <= r["peel_hi"]:
                hits.append("single-pass bound")
            if tgt is not None and r["peel_lo"] <= tgt <= r["peel_hi"]:
                hits.append("Table II")
            if hits:
                named.append(dict(peeler=r["peeler"], matches=hits))
        verdict = ("; ".join(f"{n['peeler']} matches the {' and '.join(n['matches'])}"
                             for n in named) if named else
                   "unexplained: no peeler variant's interval contains the single-pass bound "
                   "or the Table II value")
        identical = len({r["resolved"] for r in rs}) == 1 and len(rs) > 1
        out.append(dict(code=code, rounds=T, p=p, extraction=ex, gate_noise=gn,
                        single_pass_bound=sp_, table_ii=tgt, named=named, verdict=verdict,
                        variants_identical=identical,
                        theory_peel=rs[0]["theory_peel"],
                        theory_peel_uncorrected=rs[0]["theory_peel_uncorrected"]))
    return out


@app.function
def run_e01(setup, on_progress=None):
    """The Experiment 1 run: writes E01_*.json (partial after each code, then complete)."""
    stamp = start_run("E01-peeling", "E01")
    codes, variants = setup["codes"], setup["variants"]
    stem = result_stem("E01", code=codes, rounds=setup["rounds"], p=setup["p"],
                       shots=setup["shots"], seed=setup["master_seed"],
                       convention="reference-z-only",
                       extra=["peel-" + "-".join(v.replace("peel-", "") if v != "peel" else "q"
                                                for v in setup["peelers"]),
                              f"v{len(variants)}"])
    path = os.path.join(RESULTS_DIR, stem + ".json")
    config = dict(codes=codes, rounds=int(setup["rounds"]), p=float(setup["p"]),
                  shots=int(setup["shots"]), master_seed=int(setup["master_seed"]),
                  detector_convention="reference-z-only", x_detectors=False,
                  peelers=list(setup["peelers"]), variants=variants,
                  decoders={n: decoder_spec_full(n) for n in setup["peelers"]},
                  seeds={c: derive_seed(setup["master_seed"], c, setup["p"], setup["rounds"], 0)
                         for c in codes})
    config = json.loads(json.dumps(config, default=float))
    stamp = resume_stamp(path, stamp, config)
    rows, ckpts = [], []
    jobs = [(c, v) for c in codes for v in variants]
    for i, (c, vname) in enumerate(jobs):
        kw = dict(REFERENCE_VARIANTS)[vname]
        ck = os.path.join(CKPT["root"], stem, f"{c.strip('[]').replace(', ', '-')}_{i}")
        ckpts.append(ck)
        new = e01_measure(c, setup["rounds"], setup["p"], setup["shots"], config["seeds"][c],
                          peelers=setup["peelers"], workers=setup["workers"],
                          checkpoint_dir=ck if setup["shots"] >= CKPT["min_shots"] else None,
                          **kw)
        for r in new:
            r["variant"] = vname.strip(" .")
        rows += new
        if on_progress:
            on_progress()
        write_result_file(path, dict(
            config=config,
            timing=dict(workers_requested=int(setup["workers"]),
                        workers_effective=sorted({r["workers_effective"] for r in rows}),
                        backends=sorted({r["backend"].split(";")[0] for r in rows}),
                        resumed_chunks=sum(r["resumed_chunks"] for r in rows[::len(setup["peelers"])]),
                        total_chunks=sum(r["total_chunks"] for r in rows[::len(setup["peelers"])])),
            samples={f"{r['code']}|{r['variant']}": r["sample_sha256"] for r in rows},
            rows=rows, decision=e01_decision(rows),
            source="Pakhunov (2026), Table II, 'Actual' column"),
            stamp, "complete" if i == len(jobs) - 1 else "partial")
    finalise_run(path, stamp, ckpts)
    return dict(path=path, doc=read_result_file(path), fresh=True)


@app.function
def e01_load(path):
    """A saved Experiment 1 file, or an older PK_* file (no raw counts and no
    recorded seed; shown with its provenance)."""
    d = read_result_file(path, ("E01-peeling", "pakhunov"))
    if d.get("kind") == "pakhunov":
        cfg, prov = config_from_file(d, path)
        rows, reps = [], {}
        for r in d["rows"]:
            kw = r.get("kwargs", {})
            key = (r["code"], r["rounds"], r["p"], kw.get("extraction", "z-only"),
                   kw.get("gate_noise", "depolarize2"))
            if key not in reps:              # deterministic: no sampling, just the DEM
                rep = reference_report(code_from_key(r["code"]), r["rounds"], r["p"],
                                       extraction=key[3], gate_noise=key[4])
                reps[key] = {k: rep[k] for k in ("theory_peel", "theory_peel_uncorrected",
                                                 "single_pass_bound")}
            # the older files stored the UNcorrected prediction; recompute both
            r = dict(r, stored_theory_peel=r.get("theory_peel"), **reps[key])
            rows.append(dict(r, peeler="peel", extraction=r.get("kwargs", {}).get("extraction"),
                             gate_noise=r.get("kwargs", {}).get("gate_noise"),
                             variant=r["variant"].strip(" ."), resolved=None,
                             table_ii=TABLE_II.get((r["code"], r["rounds"], r["p"]))))
        d = dict(d, rows=rows, provenance=prov,
                 config=dict(d["config"], x_detectors=cfg["x_detectors"],
                             detector_convention=cfg["detector_convention"], seed=cfg["seed"]))
    return dict(path=path, doc=d, fresh=False)


@app.function
def e01_figure(rows):
    """Measured peeling rate per peeler beside the model, per code (reference circuit)."""
    ref = [r for r in rows if r.get("extraction") == "z-only" and r.get("gate_noise") == "depolarize2"]
    codes = list(dict.fromkeys(r["code"] for r in ref))
    peelers = list(dict.fromkeys(r["peeler"] for r in ref))
    fig, ax = plt.subplots(figsize=(1.0 + 2.6 * max(1, len(codes)), 4.4))
    col = plot_colours(peelers)
    width = 0.8 / (len(peelers) + 3)
    for i, c in enumerate(codes):
        rs = {r["peeler"]: r for r in ref if r["code"] == c}
        r0 = next(iter(rs.values()))
        bars = [("model e^{-BTp}", r0["theory_peel"], "#9467bd"),
                ("model, uncorrected", r0["theory_peel_uncorrected"], "#c5b0d5"),
                ("single-pass bound", r0["single_pass_bound"], "#7f7f7f")]
        for j, (lab, v, colr) in enumerate(bars):
            ax.bar(i + (j - 1.5) * width, v, width, color=colr, label=lab if i == 0 else None)
        for j, pn in enumerate(peelers):
            r = rs.get(pn)
            if r is None:
                continue
            x = i + (j + 1.5) * width
            ax.bar(x, r["peel"], width, color=col[pn], label=f"measured: {pn}" if i == 0 else None)
            ax.errorbar(x, r["peel"], yerr=[[r["peel"] - r["peel_lo"]], [r["peel_hi"] - r["peel"]]],
                        fmt="none", color="k", capsize=3)
        if r0.get("table_ii"):
            ax.hlines(r0["table_ii"], i - 0.45, i + 0.45, colors="#d62728", linestyles="--",
                      label="Table II" if i == 0 else None)
    ax.set_xticks(range(len(codes)), codes)
    lo_ = min([r["peel_lo"] for r in ref] + [0.5])
    ax.set_ylim(max(0, lo_ - 0.1), 1.0)
    ax.set_ylabel("fraction of shots peeling resolves")
    ax.set_title("Peeling on the reference circuit: model, bound and three peelers (95% Wilson)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(E01_PEELERS):
    def _e01_tiny():
        return e01_measure("[[72, 12, 6]]", 3, 1e-3, 120, 5, peelers=E01_PEELERS)

    def _e01_rows_have_raw_counts():
        rows = _e01_tiny()
        assert [r["peeler"] for r in rows] == E01_PEELERS
        for r in rows:
            assert isinstance(r["resolved"], int) and r["peel"] == r["resolved"] / 120
            assert r["peel_lo"] <= r["peel"] <= r["peel_hi"] and r["seed"] == 5
            assert r["faults"] == r["theory_faults"]                  # Theorem 1 exact
            assert r["theory_peel"] >= r["theory_peel_uncorrected"]   # e^{-BTp} < 1 helps
        assert len({r["sample_sha256"] for r in rows}) == 1, "peelers must see the same shots"

    def _e01_decision_rule():
        base = dict(code="c", rounds=12, p=1e-3, extraction="z-only", gate_noise="depolarize2",
                    single_pass_bound=0.92, table_ii=0.935, theory_peel=0.937,
                    theory_peel_uncorrected=0.935)
        rows = [dict(base, peeler="peel", resolved=893, peel=0.893, peel_lo=0.88, peel_hi=0.905),
                dict(base, peeler="peel-naive", resolved=893, peel=0.893, peel_lo=0.88, peel_hi=0.905)]
        d = e01_decision(rows)[0]
        assert d["named"] == [] and d["verdict"].startswith("unexplained") and d["variants_identical"]
        rows[1].update(peel=0.925, peel_lo=0.915, peel_hi=0.935, resolved=925)
        d = e01_decision(rows)[0]
        assert d["named"] == [dict(peeler="peel-naive", matches=["single-pass bound", "Table II"])]

    def _e01_targets_are_keyed_by_p():
        assert TABLE_II.get(("[[72, 12, 6]]", 12, 0.003)) is None
        assert TABLE_II[("[[72, 12, 6]]", 12, 0.001)] == 0.935

    def _e01_legacy_pk_loads():
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, "PK_72-12-6_T12_p0.001_2000shots.json")
        json.dump(dict(kind="pakhunov", config=dict(code="[[72, 12, 6]]", p=1e-3, rounds=[12],
                                                    shots=2000),
                       rows=[dict(variant="Pakhunov reference", rounds=12, p=1e-3,
                                  code="[[72, 12, 6]]", kwargs=dict(extraction="z-only",
                                                                    gate_noise="depolarize2"),
                                  peel=0.89, peel_lo=0.876, peel_hi=0.903, theory_peel=0.935,
                                  shots=2000)]), open(p, "w", encoding="utf-8"))
        before = file_sha256(p)
        data = e01_load(p)
        assert data["doc"]["provenance"]["seed"].startswith("inferred")
        assert data["doc"]["rows"][0]["resolved"] is None and file_sha256(p) == before

    tests_e01 = run_checks([
        ("one sample set, three peelers, raw counts and both formula columns", _e01_rows_have_raw_counts),
        ("decision rule names a variant only when its interval contains a target", _e01_decision_rule),
        ("Table II targets are keyed by (code, T, p)", _e01_targets_are_keyed_by_p),
        ("older PK files load (seed inferred, no counts) and are never rewritten", _e01_legacy_pk_loads),
    ])
    render_checks("Experiment 1", tests_e01)
    return (tests_e01,)


@app.cell
def _(E01_CODES, E01_MASTER_SEED, E01_PEELERS):
    e01_ui = mo.ui.dictionary(dict(
        codes=mo.ui.multiselect(list(BB_PRESETS), value=E01_CODES, label="codes"),
        rounds=mo.ui.number(start=1, stop=48, value=12, label="rounds T"),
        p=mo.ui.dropdown(["0.0005", "0.001", "0.002", "0.003"], value="0.001", label="p"),
        shots=mo.ui.number(start=100, stop=1_000_000, step=100, value=100_000, label="shots"),
        master_seed=mo.ui.number(value=E01_MASTER_SEED, label="master seed"),
        peelers=mo.ui.multiselect(E01_PEELERS, value=E01_PEELERS, label="peeler variants"),
        variants=mo.ui.multiselect([n for n, _ in REFERENCE_VARIANTS],
                                   value=[REFERENCE_VARIANTS[0][0]],
                                   label="circuit variants (the first is the design; others exploratory)"),
        workers=mo.ui.slider(1, max(2, os.cpu_count() or 1), value=1, label="workers",
                             show_value=True),
    ))
    e01_run_btn = mo.ui.run_button(label="Run Experiment 1")
    mo.vstack([mo.md("### Experiment 1 — controls"), e01_ui, e01_run_btn])
    return e01_run_btn, e01_ui


@app.cell
def _(E01_CODES, core_ready, e01_run_btn, e01_ui, tests_e01):
    e01_run = None
    _s = e01_ui.value
    if not (core_ready and all_pass(tests_e01)):
        _out = mo.md("*Experiment 1 locked until every test passes.*")
    elif not e01_run_btn.value:
        _out = mo.md("*Press **Run Experiment 1**, or load a saved run below.*")
    else:
        _setup = dict(codes=list(_s["codes"]) or E01_CODES, rounds=int(_s["rounds"]),
                      p=float(_s["p"]), shots=int(_s["shots"]), master_seed=int(_s["master_seed"]),
                      peelers=list(_s["peelers"]) or ["peel"],
                      variants=list(_s["variants"]) or [REFERENCE_VARIANTS[0][0]],
                      workers=int(_s["workers"]))
        with mo.status.progress_bar(total=len(_setup["codes"]) * len(_setup["variants"]),
                                    title="Experiment 1") as _bar:
            e01_run = run_e01(_setup, on_progress=_bar.update)
        _out = mo.md(f"Saved `{e01_run['path']}`")
    _out
    return (e01_run,)


@app.cell
def _(e01_run):
    _ = e01_run
    e01_file, _files = saved_run_picker("E01", "Saved Experiment 1 run", legacy_prefixes=("PK",))
    e01_load_btn = mo.ui.run_button(label="Load")
    e01_reexport_btn = mo.ui.run_button(label="Re-export loaded run")
    mo.hstack([e01_file, e01_load_btn, e01_reexport_btn]) if _files else mo.md("*No saved runs yet.*")
    return e01_file, e01_load_btn, e01_reexport_btn


@app.cell
def _(e01_file, e01_load_btn):
    e01_loaded = e01_load(e01_file.value) if (e01_load_btn.value and e01_file.value) else None
    return (e01_loaded,)


@app.cell
def _(e01_loaded, e01_reexport_btn, e01_run):
    e01_data = e01_run if e01_run is not None else e01_loaded
    mo.stop(e01_data is None, mo.md("*Run Experiment 1 or load a saved run.*"))
    _d = e01_data["doc"]
    _rows = _d["rows"]
    _dec = _d.get("decision") or (e01_decision(_rows) if all(r.get("resolved") is not None
                                                             for r in _rows) else [])
    _tblA = ["| code | circuit | faults | Theorem 1 | ratio | λ | Theorem 2 | ratio | d̄ |",
             "|:--|:--|--:|--:|--:|--:|--:|--:|--:|"]
    _seen = set()
    for _r in _rows:
        _key = (_r["code"], _r.get("variant", ""))
        if _key in _seen:
            continue
        _seen.add(_key)
        _tblA.append(f"| {_r['code']} | {_r.get('variant', '')} | {_r['faults']:,} | "
                     f"{_r['theory_faults']:,} | {_r['fault_ratio']:.3f} | {_r['lam']:.2f} | "
                     f"{_r['theory_lambda']:.2f} | {_r['lambda_ratio']:.3f} | {_r['mean_degree']:.1f} |")
    _tbl = ["| code | circuit | peeler | resolved | rate [95% CI] | model e^{-BTp} | uncorrected "
            "| single-pass bound | Table II |",
            "|:--|:--|:--|--:|:--|--:|--:|--:|--:|"]
    for _r in _rows:
        _k = "—" if _r.get("resolved") is None else f"{_r['resolved']:,}/{_r['shots']:,}"
        _tbl.append(
            f"| {_r['code']} | {_r.get('variant', '')} | {_r['peeler']} | {_k} | "
            f"{_r['peel']:.4f} [{_r['peel_lo']:.4f}, {_r['peel_hi']:.4f}] | "
            f"{_r['theory_peel']:.4f} | {_r.get('theory_peel_uncorrected', float('nan')):.4f} | "
            f"{_r.get('single_pass_bound', float('nan')):.4f} | "
            f"{_r['table_ii'] if _r.get('table_ii') else '—'} |")
    _verdicts = "\n".join(f"- **{v['code']}** ({v['extraction']}, {v['gate_noise']}): {v['verdict']}"
                          + (" — all peelers resolve the same count." if v["variants_identical"] else "")
                          for v in _dec)
    _fig = e01_figure(_rows)
    _exp = maybe_export(e01_data, "Experiment 1 — noise-model validation and peeling (Table 5.1)", {"peeling": _fig},
                        {"rows": [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in _rows],
                         "decision": [{k: v for k, v in x.items() if k != "named"} for x in _dec]},
                        notes=_verdicts, reexport=e01_reexport_btn.value)
    mo.vstack([run_status_banner(_d, e01_data["path"]),
               provenance_banner(dict(provenance=_d.get("provenance", {}), status=_d.get("status"))),
               mo.md("#### Part A — the noise model against Theorems 1 and 2\n\n" + "\n".join(_tblA)),
               mo.md("#### Part B — peeling success\n\n" + "\n".join(_tbl)),
               mo.md("#### Decision (Part B)\n\n" + (_verdicts or
                                                                   "*Older file: no raw counts, so no decision.*")),
               _fig, _exp])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiment 2 — syndrome augmentation: do the flag rows help BP+OSD? (Table 5.7, Fig. 5.3)

    **Question.** With the same decoder (BP+OSD-0, 30 iterations), does a flagged
    circuit decoded **with** its flag rows (`sighted`) fail less often than the
    same code **without** flag qubits (`unflagged`)? And what does building the
    flags cost when the decoder ignores them (`blind`)?

    **Design.** Five codes at $p = 3\times10^{-3}$, $T = 12$, all four arms of
    §15 on every configuration: [[72,12,6]] (A4, 100k shots), [[144,12,12]]
    (A5, 100k), [[288,12,18]] (A6, four independent 25k shards), [[46,2,9]]
    (A7b, 100k) and [[48,6,8]] (A8b, 100k).

    **Statistics.**
    * Each arm's LER is a binomial proportion with a 95% Wilson interval.
    * sighted vs unflagged and blind vs unflagged compare **different circuits**,
      so they are **unpaired**: the exact conditional rate-ratio test
      (`rate_ratio_ci`: $k_1 \mid k_1 + k_0 \sim \text{Binomial}$, Clopper–Pearson
      interval mapped to the ratio). These are Holm tests **T7–T9** (A4, A5, A6;
      §17). The pre-registration named Boschloo's unconditional test, which is
      computationally infeasible at $10^5$ shots per arm; the switch to the exact
      conditional test was recorded before sampling.
    * blind, sighted and updated share their shots, so they are **paired**:
      exact McNemar on the discordant shots (T4 on the [[288,12,18]] shards).
    * The [[288,12,18]] shards are independent; failures and discordant counts
      are summed before testing.
    * The minimum detectable rate ratio at the observed unflagged count is stored
      with each configuration.
    """)
    return


@app.cell
def _():
    E02_MASTER_SEED = 2026110201
    E02_DESIGN = {
        "A4": dict(code="[[72, 12, 6]]", rounds=12, p=3e-3, shots=100_000, purpose="Table 5.7; Holm T7"),
        "A5": dict(code="[[144, 12, 12]]", rounds=12, p=3e-3, shots=100_000, purpose="Table 5.7; Holm T8"),
        **{f"A6-s{i}": dict(code="[[288, 12, 18]]", rounds=12, p=3e-3, shots=25_000, shard=i,
                            purpose="Table 5.7; Holm T9 (+T4 via shards)") for i in range(4)},
        "A7b": dict(code="[[46, 2, 9]]", rounds=12, p=3e-3, shots=100_000, purpose="Table 5.7"),
        "A8b": dict(code="[[48, 6, 8]]", rounds=12, p=3e-3, shots=100_000, purpose="Table 5.7"),
    }
    return E02_DESIGN, E02_MASTER_SEED


@app.function
def plot_arm_grid(results, title="LER by arm"):
    """Figure 5.3: LER of every arm for every configuration (shards combined)."""
    comb_ = combine_shards(results)
    labels = [f"{c['code']}\np={c['p']:g}" for c in comb_]
    arms = [a for a in FOUR_ARMS if any(a in c["arms"] for c in comb_)]
    fig, ax = plt.subplots(figsize=(2.2 * max(2, len(comb_)) + 1.5, 4.6))
    width = 0.8 / len(arms)
    for j, a in enumerate(arms):
        c_ = ARM_STYLE[a][0]
        for i, c in enumerate(comb_):
            m = c["arms"].get(a)
            if m is None:
                continue
            x = i + (j - (len(arms) - 1) / 2) * width
            if m["failures"] > 0:
                ax.bar(x, m["ler"], width, color=c_, label=ARM_STYLE[a][2] if i == 0 else None)
                ax.errorbar(x, m["ler"], yerr=[[m["ler"] - m["ci_low"]], [m["ci_high"] - m["ler"]]],
                            fmt="none", color="k", capsize=3)
            else:
                ax.plot(x, m["ci_high"], "v", ms=9, mfc="none", mec=c_,
                        label=ARM_STYLE[a][2] if i == 0 else None)
    ax.set_xticks(range(len(comb_)), labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("logical error rate per shot (T = 12)")
    ax.set_title(f"{title}\n{ZERO_LABEL}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def overhead_rows(results):
    """Table 4.1 rows: qubits and detectors with and without flags, mean flags."""
    rows = {}
    for r in results:
        o = r.get("overhead", {})
        rows[(r["code"], r["rounds"])] = dict(
            code=r["code"], rounds=r["rounds"], qubits_unflagged=o.get("qubits_unflagged"),
            qubits_flagged=o.get("qubits_flagged"), detectors_unflagged=o.get("detectors_unflagged"),
            detectors_flagged=o.get("detectors_flagged"), flag_detectors=o.get("flag_detectors"),
            faults_unflagged=o.get("faults_unflagged"), faults_flagged=o.get("faults_flagged"),
            mean_flags_per_shot=o.get("mean_flags_per_shot"),
            flagged_fraction=o.get("flagged_fraction"))
    return list(rows.values())


@app.cell
def _(E02_DESIGN):
    def _e02_tiny_grid():
        import tempfile
        d = tempfile.mkdtemp()
        cfgs = [dict(code="[[72, 12, 6]]", rounds=2, p=3e-3, shots=40, shard=s) for s in (0, 1)]
        _p, doc, res = run_grid("E02", "E02-augmentation", cfgs, 1, path=os.path.join(d, "E02_t.json"))
        return doc, res

    def _e02_overhead_and_unpaired():
        doc, res = _e02_tiny_grid()
        o = overhead_rows(doc["results"])[0]
        c = code_from_key("[[72, 12, 6]]")
        assert o["detectors_flagged"] - o["detectors_unflagged"] == o["flag_detectors"] == \
            c.hx.shape[0] * 2
        assert o["faults_flagged"] - o["faults_unflagged"] == 3 * c.hx.shape[0] * 2      # law L7
        comb_ = combine_shards(doc["results"])[0]
        assert comb_["shots"] == 80 and comb_["unpaired"]["sighted|unflagged"]["paired"] is False
        plot_arm_grid(doc["results"]).savefig(io.BytesIO(), format="png")

    def _e02_design_is_registered():
        assert sum(c["shots"] for k, c in E02_DESIGN.items() if k.startswith("A6")) == 100_000
        assert all(c["p"] == 3e-3 and c["rounds"] == 12 for c in E02_DESIGN.values())
        cfgs, scaled, _m, _w = grid_setup(dict(configs=["A4"], scale="0.01 (smoke test)",
                                               master_seed=1, workers=1), E02_DESIGN)
        assert scaled and cfgs[0]["shots"] == 1000
        assert "-scaled" in grid_stem("E02", cfgs, 1, scaled=True)

    tests_e02 = run_checks([
        ("overhead rows obey law L7; unflagged comparisons are unpaired; shards combine",
         _e02_overhead_and_unpaired),
        ("registered design and scaled rehearsals are labelled", _e02_design_is_registered),
    ])
    render_checks("Experiment 2", tests_e02)
    return (tests_e02,)


@app.cell
def _(E02_DESIGN, E02_MASTER_SEED):
    e02_ui = grid_design_ui(E02_DESIGN, E02_MASTER_SEED)
    e02_run_btn = mo.ui.run_button(label="Run Experiment 2")
    mo.vstack([mo.md("### Experiment 2 — controls\n\n" + design_table_md(E02_DESIGN)), e02_ui,
               e02_run_btn])
    return e02_run_btn, e02_ui


@app.cell
def _(E02_DESIGN, core_ready, e02_run_btn, e02_ui, tests_e02):
    e02_run, _out = grid_run_cell("E02", "E02-augmentation", e02_ui.value, E02_DESIGN,
                                  core_ready and all_pass(tests_e02), e02_run_btn.value,
                                  "Experiment 2")
    _out
    return (e02_run,)


@app.cell
def _(e02_run):
    _ = e02_run
    e02_file, _files = grid_picker("E02")
    e02_load_btn = mo.ui.run_button(label="Load")
    e02_reexport_btn = mo.ui.run_button(label="Re-export loaded run")
    mo.hstack([e02_file, e02_load_btn, e02_reexport_btn]) if _files else mo.md("*No saved runs yet.*")
    return e02_file, e02_load_btn, e02_reexport_btn


@app.cell
def _(e02_file, e02_load_btn):
    e02_loaded = None
    if e02_load_btn.value and e02_file.value:
        _d, _r, _p = load_grid(e02_file.value, "E02-augmentation")
        e02_loaded = dict(path=e02_file.value, doc=_d, results=_r, per_shot=_p, fresh=False)
    return (e02_loaded,)


@app.cell
def _(e02_loaded, e02_reexport_btn, e02_run):
    e02_data = e02_run if e02_run is not None else e02_loaded
    mo.stop(e02_data is None, mo.md("*Run Experiment 2 or load a saved run.*"))
    _res = e02_data["results"]
    _comb = combine_shards(_res)
    _t57 = ["| code | p | shots | unflagged | blind | sighted | updated | sighted / unflagged "
            "(unpaired, 95% CI) | blind / unflagged |", "|:--|--:|--:|--:|--:|--:|--:|:--|:--|"]
    for _c in _comb:
        _a = _c["arms"]
        _f = lambda a: f"{_a[a]['ler']:.3e}" if a in _a else "—"
        _u = _c["unpaired"]
        _rr = lambda k: (f"{_u[k]['ratio']:.3f} [{_u[k]['lo']:.3f}, {_u[k]['hi']:.3f}]"
                         if k in _u else "—")
        _t57.append(f"| {_c['code']} | {_c['p']:g} | {_c['shots']:,} | {_f('unflagged')} | "
                    f"{_f('blind')} | {_f('sighted')} | {_f('updated')} | "
                    f"{_rr('sighted|unflagged')} | {_rr('blind|unflagged')} |")
    _ov = overhead_rows(_res)
    _t41 = ["| code | T | qubits (unflagged → flagged) | detectors | flag detectors | faults | "
            "mean flags / shot |", "|:--|--:|:--|:--|--:|:--|--:|"]
    for _o in _ov:
        _t41.append(f"| {_o['code']} | {_o['rounds']} | {_o['qubits_unflagged']} → {_o['qubits_flagged']}"
                    f" | {_o['detectors_unflagged']} → {_o['detectors_flagged']} | {_o['flag_detectors']}"
                    f" | {_o['faults_unflagged']} → {_o['faults_flagged']} | "
                    f"{(_o['mean_flags_per_shot'] or 0):.2f} |")
    _fig = plot_arm_grid(_res, "Experiment 2: LER by arm at p = 3e-3")
    _exp = maybe_export(e02_data, "Experiment 2 — syndrome augmentation (Table 5.7)",
                        {"arms": _fig}, {"arms": grid_rows(_res), "comparisons": comparison_rows(_res),
                                         "table_5_7": [dict(code=c["code"], p=c["p"], shots=c["shots"],
                                                            **{f"{a}_ler": m["ler"] for a, m in c["arms"].items()})
                                                       for c in _comb],
                                         "overhead": _ov},
                        reexport=e02_reexport_btn.value)
    mo.vstack([run_status_banner(e02_data["doc"], e02_data["path"]),
               mo.md("#### Table 5.7 (shards combined)\n\n" + "\n".join(_t57)),
               mo.md("#### Overhead (Table 4.1, BB rows)\n\n" + "\n".join(_t41)),
               _fig,
               mo.accordion({"every configuration and arm": mo.md(grid_table_md(_res)),
                             "every comparison (per shard)": mo.md(comparison_table_md(_res))}),
               _exp])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiment 3 — how often does a flag fire? (law L6, Table 5.6)

    **Question.** A block-level trigger fires when **any** of the $m_X T$ flag
    detectors fires. Law L6 predicts the fraction of shots with at least one flag:
    $1 - (1-S)^{m_X T}$, with $S$ the probability that one flag detector fires.
    A trigger that fires on nearly every shot escalates nearly every shot, so
    this fraction bounds how selective any flag rule can be.

    **Design.** Five codes at $p = 10^{-3}$, $T \in \{3, 6, 12, 18, 24\}$,
    200,000 samples per point, each $T$ with its **own** seed
    (`SeedSequence([master, code, T])`). Nothing is decoded (about 0.01
    core-hours). Each row stores the integer count $k$ of shots with a flag, the
    flag bits fired, the seed and a sample hash.

    **Statistics.** $k/n$ with a 95% Wilson interval. L6's prediction uses $S$
    computed from the same circuit's DEM (the probability that an odd number of a
    flag's faults fire), and a per-flag product as a check of L6's
    common-$S$ approximation. A Pearson $\chi^2$ of observed against predicted
    counts is reported **descriptively** (not a hypothesis test of the thesis).
    """)
    return


@app.cell
def _():
    E03_MASTER_SEED = 2026110301
    E03_CODES = ["[[72, 12, 6]]", "[[144, 12, 12]]", "[[288, 12, 18]]", "[[46, 2, 9]]", "[[48, 6, 8]]"]
    E03_ROUNDS = [3, 6, 12, 18, 24]
    return E03_CODES, E03_MASTER_SEED, E03_ROUNDS


@app.function
def flag_fire_probability(H, priors, mask):
    """Per flag detector: P(odd number of its faults fire) = (1 - prod(1 - 2 p_j)) / 2."""
    HF = (H[mask] > 0).tocsr()
    out = np.zeros(HF.shape[0])
    for i in range(HF.shape[0]):
        cols = HF.indices[HF.indptr[i]:HF.indptr[i + 1]]
        out[i] = (1 - np.prod(1 - 2 * priors[cols])) / 2
    return out


@app.function
def e03_measure(code_key, rounds, p, shots, seed):
    """One (code, T) point: count shots with >= 1 flag, against law L6."""
    code = code_from_key(code_key)
    circ = build_memory_circuit(code, rounds, p, use_flags=True, x_detectors=False)
    H, _L, priors = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
    mask = flag_detector_mask(code, rounds, True, circ.num_detectors, x_detectors=False)
    det, obs = circ.compile_detector_sampler(seed=int(seed)).sample(int(shots),
                                                                     separate_observables=True)
    bits = det[:, mask]
    k = int((bits.sum(1) > 0).sum())
    frac, lo, hi = wilson(k, shots)
    S_each = flag_fire_probability(H, priors, mask)
    S = float(S_each.mean())
    mx = int(code.hx.shape[0])
    l6 = 1 - (1 - S) ** (mx * rounds)
    exact = 1 - float(np.prod(1 - S_each))       # independence across flags, per-flag S
    return dict(code=code_key, rounds=int(rounds), p=float(p), shots=int(shots), seed=int(seed),
                flag_detectors=int(mask.sum()), m_x=mx, shots_with_flag=k,
                fraction_any_flag=frac, ci_low=lo, ci_high=hi,
                flag_bits_total=int(bits.sum()), mean_flags_per_shot=float(bits.sum(1).mean()),
                S=S, l6_prediction=float(l6), l6_per_flag_product=float(exact),
                sample_sha256=sample_hash(det, obs))


@app.function
def e03_chi2(rows):
    """Descriptive Pearson chi-square of observed k against L6 over all rows."""
    stat, dof = 0.0, 0
    for r in rows:
        n, e = r["shots"], r["l6_prediction"]
        if 0 < e < 1:
            k = r["shots_with_flag"]
            stat += (k - n * e) ** 2 / (n * e) + ((n - k) - n * (1 - e)) ** 2 / (n * (1 - e))
            dof += 1
    return dict(chi2=stat, dof=dof, p=float(st.chi2.sf(stat, dof)) if dof else float("nan"),
                note="descriptive: L6 assumes independent flags with one common S")


@app.function
def run_e03(setup, on_progress=None):
    stamp = start_run("E03-flag-rate", "E03")
    seeds = {f"{c}|{T}": derive_seed(setup["master_seed"], c, T)
             for c in setup["codes"] for T in setup["rounds_list"]}
    config = dict(codes=setup["codes"], rounds_list=setup["rounds_list"], p=setup["p"],
                  shots=setup["shots"], master_seed=setup["master_seed"], use_flags=True,
                  x_detectors=False, detector_convention="z-only", seeds=seeds,
                  decoders={}, kind_note="no decoding")
    path = result_file("E03", code=setup["codes"], rounds=setup["rounds_list"], p=setup["p"],
                       shots=setup["shots"], seed=setup["master_seed"], flags=True,
                       convention="z-only")
    stamp = resume_stamp(path, stamp, json.loads(json.dumps(config)))
    rows = []
    for c in setup["codes"]:
        for T in setup["rounds_list"]:
            rows.append(e03_measure(c, T, setup["p"], setup["shots"], seeds[f"{c}|{T}"]))
            if on_progress:
                on_progress()
        write_result_file(path, dict(config=config, rows=rows, chi2=e03_chi2(rows),
                                     timing=dict(workers_requested=1, workers_effective=1,
                                                 backends=["sampling only"], resumed_chunks=0,
                                                 total_chunks=0),
                                     samples={f"{r['code']}|{r['rounds']}": r["sample_sha256"]
                                              for r in rows}),
                          stamp, "complete" if c == setup["codes"][-1] else "partial")
    finalise_run(path, stamp, [])
    return dict(path=path, doc=read_result_file(path), fresh=True)


@app.function
def e03_load(path):
    """New E03 files, or older EXP3 files (kind 'validation', rows in checks.v5)."""
    d = read_result_file(path, ("E03-flag-rate", "validation"))
    if d.get("kind") == "validation":
        rows = []
        for r in d["checks"]["v5"]:
            n = d["config"]["shots"]
            k = int(round(r["fraction_any_flag"] * n))
            frac, lo, hi = wilson(k, n)
            rows.append(dict(r, code=d["config"]["code"], p=d["config"]["p"], shots=n, seed=None,
                             shots_with_flag=k, ci_low=lo, ci_high=hi, l6_prediction=None))
        cfg, prov = config_from_file(d, path)
        d = dict(d, rows=rows, provenance=prov,
                 legacy_note="older 2,000-shot file: no seed, k reconstructed from the rate")
    return dict(path=path, doc=d, fresh=False)


@app.function
def plot_e03(rows):
    codes = list(dict.fromkeys(r["code"] for r in rows))
    col = plot_colours(codes)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    for c in codes:
        rs = sorted([r for r in rows if r["code"] == c], key=lambda r: r["rounds"])
        Ts = [r["rounds"] for r in rs]
        a.errorbar(Ts, [r["fraction_any_flag"] for r in rs],
                   yerr=[[r["fraction_any_flag"] - r["ci_low"] for r in rs],
                         [r["ci_high"] - r["fraction_any_flag"] for r in rs]],
                   fmt="o", color=col[c], capsize=3, label=f"{c} measured")
        if all(r.get("l6_prediction") is not None for r in rs):
            a.plot(Ts, [r["l6_prediction"] for r in rs], "--", color=col[c], lw=1)
        b.plot(Ts, [r["mean_flags_per_shot"] for r in rs], "s-", color=col[c], label=c)
    a.axhline(1, color="k", lw=0.6, ls=":")
    a.set_ylim(0, 1.05)
    a.set_xlabel("rounds T")
    a.set_ylabel("fraction of shots with ≥ 1 flag")
    a.set_title("Measured (95% Wilson) and law L6 (dashed)")
    b.set_xlabel("rounds T")
    b.set_ylabel("mean flag bits fired per shot")
    b.set_title("Flags fire in proportion to rounds")
    for ax in (a, b):
        ax.grid(True, alpha=0.3)
    a.legend(fontsize=7)
    fig.tight_layout()
    return fig


@app.cell
def _():
    def _e03_row_fields():
        a = e03_measure("[[72, 12, 6]]", 2, 1e-3, 50, derive_seed(1, "[[72, 12, 6]]", 2))
        b = e03_measure("[[72, 12, 6]]", 3, 1e-3, 50, derive_seed(1, "[[72, 12, 6]]", 3))
        assert a["seed"] != b["seed"] and isinstance(a["shots_with_flag"], int)
        assert a["ci_low"] <= a["fraction_any_flag"] <= a["ci_high"]

    def _e03_matches_l6():
        """At T = 3, p = 1e-3 the measured fraction lies within 4 SE of 1 - prod(1 - S_i)."""
        r = e03_measure("[[72, 12, 6]]", 3, 1e-3, 20000, 12345)
        se = math.sqrt(r["l6_per_flag_product"] * (1 - r["l6_per_flag_product"]) / r["shots"])
        assert abs(r["fraction_any_flag"] - r["l6_per_flag_product"]) < 4 * se, r
        assert abs(r["l6_prediction"] - r["l6_per_flag_product"]) < 0.01

    def _e03_chi2_descriptive():
        rows = [dict(shots=1000, shots_with_flag=500, l6_prediction=0.5)]
        c = e03_chi2(rows)
        assert c["chi2"] == 0 and c["dof"] == 1

    tests_e03 = run_checks([
        ("per-T seeds, integer counts, Wilson intervals", _e03_row_fields),
        ("measured flag rate agrees with law L6 within 4 SE", _e03_matches_l6),
        ("χ² against L6 is computed (descriptively)", _e03_chi2_descriptive),
    ])
    render_checks("Experiment 3", tests_e03)
    return (tests_e03,)


@app.cell
def _(E03_CODES, E03_MASTER_SEED, E03_ROUNDS):
    e03_ui = mo.ui.dictionary(dict(
        codes=mo.ui.multiselect(list(BB_PRESETS) + list(GB_PRESETS), value=E03_CODES, label="codes"),
        rounds=mo.ui.multiselect([str(t) for t in (2, 3, 6, 12, 18, 24, 36)],
                                 value=[str(t) for t in E03_ROUNDS], label="rounds T"),
        p=mo.ui.dropdown(["0.0005", "0.001", "0.002", "0.003"], value="0.001", label="p"),
        shots=mo.ui.number(start=100, stop=2_000_000, step=100, value=200_000, label="shots per point"),
        master_seed=mo.ui.number(value=E03_MASTER_SEED, label="master seed"),
    ))
    e03_run_btn = mo.ui.run_button(label="Run Experiment 3 (sampling only)")
    mo.vstack([mo.md("### Experiment 3 — controls"), e03_ui, e03_run_btn])
    return e03_run_btn, e03_ui


@app.cell
def _(core_ready, e03_run_btn, e03_ui, tests_e03):
    e03_run = None
    _s = e03_ui.value
    if not (core_ready and all_pass(tests_e03)):
        _out = mo.md("*Experiment 3 locked until every test passes.*")
    elif not e03_run_btn.value:
        _out = mo.md("*Press **Run Experiment 3**, or load a saved run below.*")
    else:
        _setup = dict(codes=list(_s["codes"]), rounds_list=sorted(int(t) for t in _s["rounds"]),
                      p=float(_s["p"]), shots=int(_s["shots"]), master_seed=int(_s["master_seed"]))
        with mo.status.progress_bar(total=len(_setup["codes"]) * len(_setup["rounds_list"]),
                                    title="Experiment 3") as _bar:
            e03_run = run_e03(_setup, on_progress=_bar.update)
        _out = mo.md(f"Saved `{e03_run['path']}`")
    _out
    return (e03_run,)


@app.cell
def _(e03_run):
    _ = e03_run
    e03_file, _files = saved_run_picker("E03", "Saved Experiment 3 run", legacy_prefixes=("EXP3",))
    e03_load_btn = mo.ui.run_button(label="Load")
    e03_reexport_btn = mo.ui.run_button(label="Re-export loaded run")
    mo.hstack([e03_file, e03_load_btn, e03_reexport_btn]) if _files else mo.md("*No saved runs yet.*")
    return e03_file, e03_load_btn, e03_reexport_btn


@app.cell
def _(e03_file, e03_load_btn):
    e03_loaded = e03_load(e03_file.value) if (e03_load_btn.value and e03_file.value) else None
    return (e03_loaded,)


@app.cell
def _(e03_loaded, e03_reexport_btn, e03_run):
    e03_data = e03_run if e03_run is not None else e03_loaded
    mo.stop(e03_data is None, mo.md("*Run Experiment 3 or load a saved run.*"))
    _rows = e03_data["doc"]["rows"]
    _tbl = ["| code | T | flag detectors | shots with a flag | fraction [95% CI] | L6 | "
            "mean flags / shot | seed |", "|:--|--:|--:|--:|:--|--:|--:|--:|"]
    for _r in _rows:
        _l6 = f"{_r['l6_prediction']:.4f}" if _r.get("l6_prediction") is not None else "—"
        _tbl.append(f"| {_r['code']} | {_r['rounds']} | {_r['flag_detectors']} | "
                    f"{_r['shots_with_flag']:,}/{_r['shots']:,} | {_r['fraction_any_flag']:.4f} "
                    f"[{_r['ci_low']:.4f}, {_r['ci_high']:.4f}] | {_l6} | "
                    f"{_r['mean_flags_per_shot']:.2f} | {_r.get('seed')} |")
    _chi = e03_data["doc"].get("chi2")
    _fig = plot_e03(_rows)
    _exp = maybe_export(e03_data, "Experiment 3 — flag rate against law L6 (Table 5.6)",
                        {"flag_rate": _fig}, {"flag_rate": _rows}, reexport=e03_reexport_btn.value)
    mo.vstack([run_status_banner(e03_data["doc"], e03_data["path"]),
               provenance_banner(dict(provenance=e03_data["doc"].get("provenance", {}),
                                      status=e03_data["doc"].get("status"))),
               mo.md("\n".join(_tbl)),
               mo.md(f"χ² against L6 (descriptive): {_chi['chi2']:.1f} on {_chi['dof']} points, "
                     f"p = {_chi['p']:.3g}" if _chi else ""),
               _fig, _exp])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiment 4 — flag-triggered switching: the decoder grid (Tables 5.2–5.5, Fig. 5.2)

    **Question.** Is "escalate to a stronger decoder when a flag fires" worth
    anything — in accuracy, or in time — for any pairing of a fast primary and a
    strong secondary? And how much do the flags actually tell us about a
    decoder's errors?

    **Source.** A 250,000-shot grid on [[72,12,6]], $T = 12$, $p = 10^{-3}$:
    four primaries (peel, BP-ms-10, BP-ms-30, BP-ps-30) × four secondaries
    (OSD-0, OSD-CS4, LSD-0, LSD-CS4), every decoder on the same shots, saved by an
    earlier version of this notebook as `results/EXP6_…_250000shots_….npz`. Its
    per-shot outcomes are re-analysed here under §13's single cost convention
    (a new full rerun is possible from the controls but adds no accuracy
    evidence):

    * **Table 5.2 / Fig. 5.2** — the six reference policies of peel → OSD-0;
    * **Table 5.3** — every trigger rule of that pair;
    * **Table 5.4** — what the flags tell us about each decoder's errors;
    * **Table 5.5** — verdicts for all 16 pairs.

    **Timing.** That grid was timed in a 12-process pool, before the untimed
    warm-up existed, so its chunk-start shots carry decoder builds. Its times are
    re-analysed with chunk starts excluded (medians and 1%-trimmed means), and the
    registered run **re-times** peel, BP-ms-10, BP-ms-30, BP-ps-30 and OSD-0
    single-threaded on 20,000 fresh shots. Secondaries not re-timed (OSD-CS4,
    LSD-0, LSD-CS4) are labelled "12-process pool, relative only".

    **Statistics.**
    * Accuracy of each rule against `primary_fail`: exact McNemar on the paired
      shots, Bonferroni-corrected over m = 4 × 4 × 7 = 112 comparisons; the
      verdict order is fixed in §13.
    * **Placebo control**: every flag rule is also compared (McNemar) with a rule
      that escalates the same number of randomly chosen shots; a rule is
      "flag-informative" only if it beats its placebo.
    * **Cost**: per-shot cost differences against `primary_fail` and `always`
      with a chunk-block standard error (250-shot blocks) and a family-corrected
      bound. A saving is claimed only if that interval excludes zero for a rule
      with no accuracy loss.
    * **Flag information**: risk ratio of failure on flagged vs unflagged shots
      and the mutual information (bits) between "a flag fired" and "the decoder
      failed".
    * The "hybrid beats the strong decoder" observation (shots failing only
      under `primary_fail` vs only under `always`) is exploratory and holds by
      construction when peeling has no silent failures.
    """)
    return


@app.cell
def _():
    E04_MASTER_SEED = 2026110401
    E04_RETIME_DECODERS = ["peel", "BP-ms-10", "BP-ms-30", "BP-ps-30", "OSD-0"]

    def e04_problem(code_key, rounds, p):
        """Circuit, DEM matrices and flag mask of a decoder-grid configuration."""
        code = code_from_key(code_key)
        circ = build_memory_circuit(code, rounds, p, use_flags=True, x_detectors=False)
        H, L, pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
        mask = flag_detector_mask(code, rounds, True, circ.num_detectors, x_detectors=False)
        return code, circ, H, L, pr, mask

    return E04_MASTER_SEED, E04_RETIME_DECODERS, e04_problem


@app.cell
def _(e04_problem):
    def run_e04(setup, kind="E04-grid", on_chunk=None, results_dir=None):
        """
        Decode one configuration with every primary and secondary on the same
        shots; save the per-shot grid as <stem>.npz beside a schema-2 <stem>.json.
        kind "E04-retime" is the registered single-thread re-timing (workers forced to 1).
        """
        if kind == "E04-retime":
            setup = dict(setup, workers=1)
        code, circ, H, L, pr, mask = e04_problem(setup["code"], setup["rounds"], setup["p"])
        det, obs = circ.compile_detector_sampler(seed=int(setup["seed"])).sample(
            int(setup["shots"]), separate_observables=True)
        det, obs = det.astype(np.uint8), obs.astype(np.uint8)
        n_flags, flag_rounds = flag_features(det, mask, code.hx.shape[0], setup["rounds"])
        names, vector = [], None
        for n in setup["weak"] + setup["strong"]:
            spec = dict(CATALOGUE[n])
            if spec.get("kind") == "control":
                if vector is None:
                    vector = logical_null_vector(H, L)
                spec.update(vector=vector, flag_mask=mask)
            names.append((n, spec))
        dec_hash = hashlib.sha256(",".join(setup["weak"] + setup["strong"]).encode()).hexdigest()[:8]
        prefix = "E04" if kind == "E04-grid" else "E04R"
        stem = result_stem(prefix, code=setup["code"], rounds=setup["rounds"], p=setup["p"],
                           shots=setup["shots"], seed=setup["seed"], flags=True, convention="z-only",
                           extra=[f"dec{dec_hash}", f"w{setup['workers']}"])
        folder = results_dir or RESULTS_DIR
        path = os.path.join(folder, stem + ".json")
        stamp = start_run(kind, "E04")
        ck = os.path.join(CKPT["root"], stem)
        res, backend, info = run_zoo(H, L, pr, det, obs, names, workers=setup["workers"],
                                     chunk_size=CHUNK_SIZE, checkpoint_dir=ck, return_info=True,
                                     on_chunk=on_chunk, label=f"{stem}: {len(names)} decoders")
        npz = os.path.join(folder, stem + ".npz")
        meta = dict(setup, backend=backend, workers_effective=info["workers_effective"],
                    chunk_size=info["chunk_size"], numba=HAVE_NUMBA, warmup="untimed per chunk")
        save_zoo(npz, res, n_flags, flag_rounds, meta)
        write_result_file(path, dict(
            config=dict(code=setup["code"], rounds=int(setup["rounds"]), p=float(setup["p"]),
                        shots=int(setup["shots"]), seed=int(setup["seed"]), use_flags=True,
                        x_detectors=False, detector_convention="z-only",
                        weak=setup["weak"], strong=setup["strong"],
                        decoders={n: decoder_spec_full({k: v for k, v in s.items()
                                                         if k not in ("vector", "flag_mask")})
                                  for n, s in names}),
            timing=dict(workers_requested=int(setup["workers"]),
                        workers_effective=info["workers_effective"], backends=[backend.split(";")[0]],
                        resumed_chunks=info["resumed_chunks"], total_chunks=info["total_chunks"],
                        warmup="one untimed decode per chunk and decoder"),
            samples={"grid": sample_hash(det, obs)}, data_file=os.path.basename(npz),
            data_sha256=file_sha256(npz)), stamp, "complete")
        finalise_run(path, stamp, [ck])
        return dict(path=path, doc=read_result_file(path), results=res, n_flags=n_flags,
                    flag_rounds=flag_rounds, meta=meta, fresh=True)

    return (run_e04,)


@app.function
def e04_load(path):
    """A new E04 JSON (+ npz), or the 250,000-shot EXP6_*.npz grid (meta repr; 12-worker pool)."""
    if path.endswith(".json"):
        d = read_result_file(path, ("E04-grid", "E04-retime"))
        npz = os.path.join(os.path.dirname(path), d["data_file"])
        if file_sha256(npz) != d["data_sha256"]:
            raise ValueError(f"{npz} does not match the sha256 recorded in {path}")
        res, nf, fr, meta = load_zoo(npz)
        return dict(path=path, doc=d, results=res, n_flags=nf, flag_rounds=fr, meta=meta,
                    fresh=False)
    res, nf, fr, meta = load_zoo(path)
    meta = dict(meta)
    meta.setdefault("workers_effective", meta.get("workers"))
    doc = dict(kind="EXP6 grid (npz)", status="older file (re-analysed)", started=meta.get("created"),
               config=dict(code=meta.get("code"), rounds=meta.get("rounds"), p=meta.get("p"),
                           shots=len(nf), seed=meta.get("seed"),
                           x_detectors=meta.get("x_detectors"),
                           detector_convention=({False: "z-only", True: "x+z"}.get(
                               meta.get("x_detectors"), "unknown"))),
               timing=dict(workers_effective=meta.get("workers"), resumed_chunks=0,
                           total_chunks=None),
               provenance=dict(timing=f"{meta.get('workers')}-process pool; decoder builds "
                                      "inside chunk-start shots: relative only"))
    return dict(path=path, doc=doc, results=res, n_flags=nf, flag_rounds=fr, meta=meta, fresh=False)


@app.function
def e04_names(results):
    names = list(results)
    weak = [n for n in names if n in WEAK or n in CONTROLS or n == "control"]
    strong = [n for n in names if n in STRONG]
    return weak, strong


@app.function
def e04_cost_reanalysis(results, n_flags, weak, strong, chunk_size=CHUNK_SIZE,
                        exclude_chunk_starts=True, trim=0.01):
    """
    Cost re-analysis: every reference policy of one pair under the 'derive' convention,
    with mean, median and 1%-trimmed mean; chunk-start shots (which carried
    decoder builds in the 250k grid) excluded unless asked otherwise.
    Savings are against both always and primary_fail. The shared-BP column
    (exploratory) charges the secondary only its time beyond the primary's
    BP, as if the BP run were reused -- which the code does not do.
    """
    W, S = results[weak], results[strong]
    n = len(W["fail"])
    keep = ~chunk_start_mask(n, chunk_size) if exclude_chunk_starts else np.ones(n, bool)
    nested = nested_bp(weak, strong)
    rows = []
    pols = reference_policies(W, S, n_flags)
    costs = {name: derive(W, S, pre, fb)["cost"] for name, (pre, fb) in pols.items()}
    for name, (pre, fb) in pols.items():
        P = derive(W, S, pre, fb)
        c = costs[name][keep]
        row = dict(weak=weak, strong=strong, policy=name, shots_used=int(keep.sum()),
                   chunk_starts_excluded=bool(exclude_chunk_starts),
                   failures=int(P["fail"].sum()), escalation=float(P["esc"].mean()),
                   mean_us=float(c.mean()), median_us=float(np.median(c)),
                   trimmed_mean_us=trimmed_mean(c, trim),
                   saving_vs_always=1 - float(c.mean()) / float(costs["always"][keep].mean()),
                   saving_vs_primary_fail=1 - float(c.mean()) / float(costs["primary_fail"][keep].mean()),
                   cost_convention=COST_CONVENTIONS["derive"])
        if nested:
            extra = np.maximum(S["time_us"].astype(float) - W["time_us"].astype(float), 0.0)
            shared = (np.where(np.asarray(pre, bool), S["time_us"].astype(float),
                               W["time_us"].astype(float) + np.where(P["esc"], extra, 0.0)))
            row["shared_bp_mean_us_exploratory"] = float(shared[keep].mean())
        rows.append(row)
    return rows


@app.function
def e04_hybrid(results, weak="peel", strong="OSD-0"):
    """Shots failing only under primary_fail vs only under always (exploratory)."""
    W, S = results[weak], results[strong]
    pf = derive(W, S, False, True)
    x, y, pv = paired_exact(pf["fail"], S["fail"].astype(bool))
    return dict(weak=weak, strong=strong, only_primary_fail=x, only_always=y, p=pv,
                primary_fail_failures=int(pf["fail"].sum()), always_failures=int(S["fail"].sum()),
                peel_silent=int((W["conv"] & W["fail"]).sum()),
                note="exploratory, by construction: a converged peel correction is kept")


@app.function
def e04_timing_label(meta, decoder):
    """Workers note for a time column."""
    w = meta.get("workers_effective") or meta.get("workers")
    if w and int(w) > 1:
        return f"{w}-process pool, relative only"
    return "single-threaded"


@app.cell
def _(e04_problem, run_e04):
    def _e04_tiny():
        code, circ, H, L, pr, mask = e04_problem("[[72, 12, 6]]", 3, 2e-3)
        det, obs = circ.compile_detector_sampler(seed=5).sample(500, separate_observables=True)
        res, _b = run_zoo(H, L, pr, det, obs, ["peel", "BP-ms-10", "OSD-0"], workers=1,
                          chunk_size=CHUNK_SIZE, checkpoint_dir=False)
        nf, fr = flag_features(det, mask, code.hx.shape[0], 3)
        return res, nf, fr

    def _e04_reanalysis_reproduces_derive():
        """With chunk starts included, the cost re-analysis reproduces derive's means exactly."""
        res, nf, _fr = _e04_tiny()
        rows = e04_cost_reanalysis(res, nf, "peel", "OSD-0", exclude_chunk_starts=False)
        pols = reference_policies(res["peel"], res["OSD-0"], nf)
        for r in rows:
            P = derive(res["peel"], res["OSD-0"], *pols[r["policy"]])
            assert abs(r["mean_us"] - float(P["cost"].mean())) < 1e-6, r["policy"]
        ex = e04_cost_reanalysis(res, nf, "peel", "OSD-0")
        assert ex[0]["shots_used"] == 500 - 2
        nest = e04_cost_reanalysis(res, nf, "BP-ms-10", "OSD-0")
        assert "shared_bp_mean_us_exploratory" in nest[0]

    def _e04_hybrid_counts():
        res, _nf, _fr = _e04_tiny()
        h = e04_hybrid(res)
        pf = derive(res["peel"], res["OSD-0"], False, True)
        assert h["primary_fail_failures"] - h["always_failures"] == h["only_primary_fail"] - h["only_always"]
        assert h["only_primary_fail"] <= h["peel_silent"]
        assert int(pf["fail"].sum()) == h["primary_fail_failures"]

    def _e04_run_and_load_roundtrip():
        import tempfile
        old_root = CKPT["root"]
        d = tempfile.mkdtemp()
        configure_checkpoints(root=os.path.join(d, "ck"))
        try:
            setup = dict(code="[[72, 12, 6]]", rounds=2, p=2e-3, shots=60, seed=5,
                         weak=["peel"], strong=["OSD-0"], workers=3)
            r = run_e04(setup, kind="E04-retime", results_dir=d)
            assert r["meta"]["workers"] == 1 and r["doc"]["timing"]["workers_effective"] == 1
            before = file_sha256(r["path"])
            back = e04_load(r["path"])
            assert np.array_equal(back["results"]["peel"]["fail"], r["results"]["peel"]["fail"])
            assert file_sha256(r["path"]) == before
        finally:
            configure_checkpoints(root=old_root)

    def _e04_timing_labels():
        assert e04_timing_label({"workers": 12}, "OSD-CS4") == "12-process pool, relative only"
        assert e04_timing_label({"workers_effective": 1}, "peel") == "single-threaded"

    tests_e04 = run_checks([
        ("the cost re-analysis with chunk starts included reproduces derive's means", _e04_reanalysis_reproduces_derive),
        ("hybrid split: only-pf vs only-always counts are consistent", _e04_hybrid_counts),
        ("re-timing runs single-threaded; npz + JSON round-trip, load never writes", _e04_run_and_load_roundtrip),
        ("time columns name their worker regime", _e04_timing_labels),
    ])
    render_checks("Experiment 4", tests_e04)
    return (tests_e04,)


@app.cell
def _(E04_MASTER_SEED, E04_RETIME_DECODERS):
    e04_ui = mo.ui.dictionary(dict(
        code=mo.ui.dropdown(list(BB_PRESETS) + list(GB_PRESETS), value="[[72, 12, 6]]", label="code"),
        rounds=mo.ui.number(start=1, stop=48, value=12, label="rounds T"),
        p=mo.ui.dropdown(["0.0005", "0.001", "0.002", "0.003"], value="0.001", label="p"),
        shots=mo.ui.number(start=100, stop=1_000_000, step=100, value=20_000, label="shots"),
        seed=mo.ui.number(value=E04_MASTER_SEED + 1, label="seed (fresh shots for re-timing)"),
        workers=mo.ui.slider(1, max(2, os.cpu_count() or 1), value=1, show_value=True,
                             label="workers (re-timing always uses 1)"),
        weak=mo.ui.multiselect(list(WEAK) + list(CONTROLS), value=E04_RETIME_DECODERS[:4],
                               label="primaries"),
        strong=mo.ui.multiselect(list(STRONG), value=["OSD-0"], label="secondaries"),
        mode=mo.ui.radio({"single-thread re-timing (registered)": "E04-retime",
                          "full decoder grid (exploratory rerun)": "E04-grid"},
                         value="single-thread re-timing (registered)", label="run"),
    ))
    e04_run_btn = mo.ui.run_button(label="Run Experiment 4")
    mo.vstack([mo.md("### Experiment 4 — controls\n\nThe registered run re-times the five "
                     "decoders below, single-threaded, on 20,000 fresh shots. A full grid rerun is "
                     "possible but adds no accuracy evidence."), e04_ui, e04_run_btn])
    return e04_run_btn, e04_ui


@app.cell
def _(core_ready, e04_run_btn, e04_ui, run_e04, tests_e04):
    e04_run = None
    _s = e04_ui.value
    if not (core_ready and all_pass(tests_e04)):
        _out = mo.md("*Experiment 4 locked until every test passes.*")
    elif not e04_run_btn.value:
        _out = mo.md("*Press **Run Experiment 4**, or load the 250k grid or a saved run below.*")
    else:
        _setup = dict(code=_s["code"], rounds=int(_s["rounds"]), p=float(_s["p"]),
                      shots=int(_s["shots"]), seed=int(_s["seed"]), workers=int(_s["workers"]),
                      weak=list(_s["weak"]), strong=list(_s["strong"]))
        with mo.status.progress_bar(total=-(-_setup["shots"] // CHUNK_SIZE),
                                    title="Experiment 4", show_eta=True) as _bar:
            e04_run = run_e04(_setup, kind=_s["mode"], on_chunk=_bar.update)
        _out = mo.md(f"Saved `{e04_run['path']}`")
    _out
    return (e04_run,)


@app.cell
def _(e04_run):
    _ = e04_run
    _new = saved_runs("E04") + saved_runs("E04R")
    _old = sorted(glob.glob(os.path.join(RESULTS_DIR, "EXP6_*.npz")))
    _files = _new + [f for f in _old if not is_checkpoint_path(f)]
    e04_file = mo.ui.dropdown({os.path.basename(f): f for f in _files},
                              value=os.path.basename(_files[-1]) if _files else None,
                              label="Saved Experiment 4 run (EXP6_*.npz = the 250k grid)")
    e04_load_btn = mo.ui.run_button(label="Load")
    e04_reexport_btn = mo.ui.run_button(label="Re-export loaded run")
    e04_pair = mo.ui.text(value="peel → OSD-0", label="pair for Tables 5.2/5.3 (primary → secondary)")
    mo.vstack([mo.hstack([e04_file, e04_load_btn, e04_reexport_btn]), e04_pair]) if _files \
        else mo.md("*No saved runs yet. Put the 250k-grid `EXP6_*.npz` in `results/` to re-analyse it.*")
    return e04_file, e04_load_btn, e04_pair, e04_reexport_btn


@app.cell
def _(e04_file, e04_load_btn):
    e04_loaded = e04_load(e04_file.value) if (e04_load_btn.value and e04_file.value) else None
    return (e04_loaded,)


@app.cell
def _(e04_loaded, e04_pair, e04_reexport_btn, e04_run):
    e04_data = e04_run if e04_run is not None else e04_loaded
    mo.stop(e04_data is None, mo.md("*Run Experiment 4 or load a saved run.*"))
    _res, _nf, _fr, _meta = (e04_data["results"], e04_data["n_flags"], e04_data["flag_rounds"],
                             e04_data["meta"])
    _weak, _strong = e04_names(_res)
    _w0, _s0 = [x.strip() for x in e04_pair.value.split("→")] if "→" in e04_pair.value else ("peel", "OSD-0")
    _w0 = _w0 if _w0 in _weak else _weak[0]
    _s0 = _s0 if _s0 in _strong else _strong[0]
    _workers = _meta.get("workers_effective") or _meta.get("workers")
    _pt = policy_table(_res, _w0, _s0, _nf, workers=_workers)
    e04_analysis = analyse(_res, _weak, _strong, _nf, _fr, workers=_workers,
                           chunk_size=int(_meta.get("chunk_size", CHUNK_SIZE)))
    _cost = e04_cost_reanalysis(_res, _nf, _w0, _s0, chunk_size=int(_meta.get("chunk_size", CHUNK_SIZE)))
    _hyb = e04_hybrid(_res, _w0, _s0) if _w0 == "peel" else None
    _lab = e04_timing_label(_meta, _s0)

    _t52 = [f"| policy | escalated | failures | LER [95% CI] | mean µs | median µs | p99 µs |",
            "|:--|--:|--:|:--|--:|--:|--:|"]
    for _r in _pt:
        _t52.append(f"| {TRIGGER_DISPLAY.get(_r['policy'], _r['policy'])} | {_r['escalation']:.1%} | "
                    f"{_r['failures']} | {_r['ler']:.2e} [{_r['ci_low']:.2e}, {_r['ci_high']:.2e}] | "
                    f"{_r['cost_mean_us']:,.1f} | {_r['cost_median_us']:,.1f} | {_r['cost_p99_us']:,.0f} |")
    _t53 = ["| rule | escalated | failures | better : worse vs pf | p (Bonf.) | saving vs pf "
            "[family CI] | vs always | flag-informative (vs placebo) | cost-viable |",
            "|:--|--:|--:|:--|--:|:--|--:|:--|:--|"]
    for _r in zoo_pair_rows(e04_analysis, _w0, _s0):
        if _r["rule"] == "primary_fail":
            continue
        _lo, _hi = _r["saving_ci"]
        _t53.append(f"| {_r['rule']} | {_r['escalation']:.1%} | {_r['failures']} | "
                    f"{_r['better']} : {_r['worse']} | {_r['p_adj']:.2g} | {_r['saving']:+.1%} "
                    f"[{_lo:+.1%}, {_hi:+.1%}] | {_r['saving_vs_always']:+.1%} | "
                    f"{'yes' if _r['flag_informative'] else 'no'} | "
                    f"{'yes' if _r.get('cost_viable') else 'no'} |")
    _t54 = ["| decoder | error rate, flagged | unflagged | risk ratio | MI (bits) | "
            "share of failures on flagged |", "|:--|--:|--:|--:|--:|--:|"]
    for _f in e04_analysis["flag_diagnostics"]:
        _t54.append(f"| {_f['decoder']} | {_f['error_rate_flagged']:.2e} | "
                    f"{_f['error_rate_unflagged']:.2e} | {_f['risk_ratio']:.2f} | "
                    f"{_f['mutual_information_bits']:.5f} | {_f['share_of_failures_on_flagged']:.1%} |")
    _t55 = [f"| primary → secondary | accuracy | cost | headroom | pf fails | always fails | "
            f"best rule | best saving | timing |", "|:--|:--|:--|--:|--:|--:|:--|--:|:--|"]
    for _p in e04_analysis["pairs"]:
        _t55.append(f"| {_p['weak']} → {_p['strong']} | {_p['verdict_accuracy']} | "
                    f"{_p['verdict_cost']} | {_p['headroom']} | {_p['pf_failures']} | "
                    f"{_p['always_failures']} | {_p['best_accuracy_rule'] or _p['best_cost_rule'] or '—'} | "
                    f"{_p['best_saving']:+.1%} | {e04_timing_label(_meta, _p['strong'])} |")
    _tn5 = ["| policy | mean µs | median µs | 1%-trimmed mean | vs always | vs primary_fail |"
            + (" shared-BP mean (exploratory) |" if "shared_bp_mean_us_exploratory" in _cost[0] else ""),
            "|:--|--:|--:|--:|--:|--:|" + ("--:|" if "shared_bp_mean_us_exploratory" in _cost[0] else "")]
    for _r in _cost:
        _tn5.append(f"| {_r['policy']} | {_r['mean_us']:,.1f} | {_r['median_us']:,.1f} | "
                    f"{_r['trimmed_mean_us']:,.1f} | {_r['saving_vs_always']:+.1%} | "
                    f"{_r['saving_vs_primary_fail']:+.1%} |"
                    + (f" {_r['shared_bp_mean_us_exploratory']:,.1f} |"
                       if "shared_bp_mean_us_exploratory" in _r else ""))
    _figs = {"reference_policies": plot_reference_policies(_pt),
             "verdicts": plot_zoo_verdicts(e04_analysis), "silent": plot_zoo_silent(e04_analysis),
             "breakeven": plot_zoo_breakeven(e04_analysis), "tradeoffs": plot_zoo_tradeoffs(e04_analysis)}
    _viable_cost = [p for p in e04_analysis["pairs"] if p["verdict_cost"] == "viable: cost"]
    _n5 = ("**Cost decision:** a cost saving survives the family-corrected interval for "
           + ", ".join(f"{p['weak']} → {p['strong']} ({p['best_cost_rule']}, {p['best_saving']:+.1%})"
                       for p in _viable_cost) + "." if _viable_cost else
           "**Cost decision:** no rule with zero accuracy loss has a family-corrected saving "
           "interval that excludes zero against both `primary_fail` and `always`; the "
           "\"viable on cost\" claim is not supported by this file.")
    _hyb_md = (f"**Hybrid (exploratory, by construction):** {_hyb['only_primary_fail']} shots fail "
               f"only under primary_fail, {_hyb['only_always']} only under always "
               f"(p = {_hyb['p']:.2g}); peel had {_hyb['peel_silent']} silent failures."
               if _hyb else "")
    _exp = maybe_export(e04_data,
                        f"Experiment 4 — decoder grid ({len(_nf):,} shots)", _figs,
                        {"table_5_2": _pt, "table_5_3": [r for r in zoo_pair_rows(e04_analysis, _w0, _s0)],
                         "table_5_4": e04_analysis["flag_diagnostics"], "table_5_5": e04_analysis["pairs"],
                         "cost_reanalysis": _cost, "all_rules": e04_analysis["rows"],
                         "strong_ceiling": e04_analysis["strong_ceiling"]},
                        notes=_n5 + "\n\n" + _hyb_md, reexport=e04_reexport_btn.value) \
        if e04_data["path"].endswith(".json") else mo.md(
            "*EXP6 npz: displayed only. Use §18's regeneration commands to write tables.*")
    mo.vstack([
        run_status_banner(e04_data["doc"], e04_data["path"]),
        provenance_banner(dict(provenance=e04_data["doc"].get("provenance", {}),
                               status=e04_data["doc"].get("status"))),
        mo.md(f"**{len(_nf):,} shots · {len(_weak)} primaries × {len(_strong)} secondaries · "
              f"Bonferroni m = {e04_analysis['m']} · timing: {_lab}**"),
        mo.md(f"#### Table 5.2 — reference policies, {_w0} → {_s0}\n\n" + "\n".join(_t52)),
        _figs["reference_policies"],
        mo.md(f"#### Table 5.3 — trigger rules, {_w0} → {_s0}\n\n" + "\n".join(_t53)),
        mo.md("#### Table 5.4 — what do the flags tell us?\n\n" + "\n".join(_t54)),
        mo.md("#### Table 5.5 — verdicts for every pair\n\n" + "\n".join(_t55)),
        mo.md(f"#### Cost re-analysis, {_w0} → {_s0} (chunk starts excluded)\n\n"
              + "\n".join(_tn5)),
        mo.md(_n5), mo.md(_hyb_md),
        _figs["verdicts"], _figs["silent"], _figs["breakeven"], _figs["tradeoffs"], _exp,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiment 5 — silent-failure census of the peeler

    **Question.** A flag rule can only gain accuracy on shots where the primary
    is *confidently wrong* (converged but failed) and the secondary is right.
    How many such shots does peeling produce, across codes and noise levels?

    **Design.** Peeling only, on the flagged circuit, 5 codes ×
    $p \in \{10^{-3}, 2\times10^{-3}, 3\times10^{-3}\}$ × $T = 12$, 250,000 shots
    each. Every shot stores converged, failed and the number of flags. If any
    silent failure occurs, those shots and peeling's other failures are decoded
    with BP+OSD-0 to obtain the headroom
    $|\{\text{conv} \wedge \text{fail} \wedge \neg\text{OSD-0 fail}\}|$.

    **Statistics.** The silent count $k$ per converged shot, with a one-sided 95%
    Clopper–Pearson upper bound ($k = 0$ gives $\approx 3/n = 1.2\times10^{-5}$ at
    250,000). The `primary_fail` escalation rate against $p$ is descriptive.
    """)
    return


@app.cell
def _(e04_problem):
    E05_MASTER_SEED = 2026110501
    E05_CODES = ["[[72, 12, 6]]", "[[144, 12, 12]]", "[[288, 12, 18]]", "[[46, 2, 9]]", "[[48, 6, 8]]"]
    E05_PS = [1e-3, 2e-3, 3e-3]

    def e05_point(code_key, rounds, p, shots, seed, workers=1, checkpoint_dir=None):
        """Peel every shot; on silent or failed shots, decode with OSD-0 for headroom."""
        code, circ, H, L, pr, mask = e04_problem(code_key, rounds, p)
        det, obs = circ.compile_detector_sampler(seed=int(seed)).sample(int(shots),
                                                                         separate_observables=True)
        det, obs = det.astype(np.uint8), obs.astype(np.uint8)
        w = parallel_workers(shots, workers)
        res, backend, info = run_zoo(H, L, pr, det, obs, ["peel"], workers=w,
                                     chunk_size=CHUNK_SIZE if checkpoint_dir else chunk_for(shots, w),
                                     checkpoint_dir=checkpoint_dir or False, return_info=True,
                                     label=f"E05 {code_key} T={rounds} p={p:g}")
        W = res["peel"]
        conv, fail = W["conv"].astype(bool), W["fail"].astype(bool)
        silent = conv & fail
        k = int(silent.sum())
        n_flags = det[:, mask].sum(1)
        out = dict(code=code_key, rounds=int(rounds), p=float(p), shots=int(shots), seed=int(seed),
                   converged=int(conv.sum()), failures=int(fail.sum()), silent=k,
                   silent_rate=k / max(1, int(conv.sum())),
                   silent_upper_95=cp_upper(k, max(1, int(conv.sum())), 0.95),
                   silent_on_flagged=int((silent & (n_flags > 0)).sum()),
                   escalation_primary_fail=float((~conv).mean()),
                   flagged_fraction=float((n_flags > 0).mean()), sample_sha256=sample_hash(det, obs),
                   workers_effective=info["workers_effective"], resumed_chunks=info["resumed_chunks"],
                   total_chunks=info["total_chunks"], headroom=None, osd_on_peel_failures=None)
        idx = np.flatnonzero(fail)
        if k > 0 and len(idx):
            r2, _b = run_zoo(H, L, pr, det[idx], obs[idx], ["OSD-0"], workers=1,
                             chunk_size=max(1, len(idx)), checkpoint_dir=False)
            sfail = r2["OSD-0"]["fail"].astype(bool)
            out["headroom"] = int((silent[idx] & ~sfail).sum())
            out["osd_on_peel_failures"] = dict(decoded=int(len(idx)), osd_failures=int(sfail.sum()))
        elif k == 0:
            out["headroom"] = 0
        per = dict(conv=conv, fail=fail, n_flags=n_flags.astype(np.int16))
        return out, per

    return E05_CODES, E05_MASTER_SEED, e05_point


@app.cell
def _(e05_point):
    def run_e05(setup, on_progress=None):
        stamp = start_run("E05-census", "E05")
        path = result_file("E05", code=setup["codes"], rounds=setup["rounds"], p=setup["ps"],
                           shots=setup["shots"], seed=setup["master_seed"], flags=True,
                           convention="z-only", extra="peel")
        seeds = {f"{c}|{p:g}": derive_seed(setup["master_seed"], c, float(p), setup["rounds"], 0)
                 for c in setup["codes"] for p in setup["ps"]}
        config = dict(codes=setup["codes"], rounds=setup["rounds"], ps=setup["ps"],
                      shots=setup["shots"], master_seed=setup["master_seed"], seeds=seeds,
                      x_detectors=False, detector_convention="z-only",
                      decoders={"peel": decoder_spec_full("peel"), "OSD-0": decoder_spec_full("OSD-0")})
        config = json.loads(json.dumps(config))
        stamp = resume_stamp(path, stamp, config)
        rows, per, ckpts = [], {}, []
        jobs = [(c, p) for c in setup["codes"] for p in setup["ps"]]
        stem = os.path.splitext(os.path.basename(path))[0]
        for i, (c, p) in enumerate(jobs):
            ck = os.path.join(CKPT["root"], stem, f"{i}")
            ckpts.append(ck)
            row, arrs = e05_point(c, setup["rounds"], p, setup["shots"], seeds[f"{c}|{p:g}"],
                                  workers=setup["workers"],
                                  checkpoint_dir=ck if setup["shots"] >= CKPT["min_shots"] else None)
            rows.append(row)
            per[f"{i}"] = arrs
            if on_progress:
                on_progress()
            write_result_file(path, dict(
                config=config, rows=rows,
                timing=dict(workers_requested=setup["workers"],
                            workers_effective=sorted({r["workers_effective"] for r in rows}),
                            backends=["run_zoo"], resumed_chunks=sum(r["resumed_chunks"] for r in rows),
                            total_chunks=sum(r["total_chunks"] for r in rows)),
                samples={f"{r['code']}|{r['p']:g}": r["sample_sha256"] for r in rows}),
                stamp, "complete" if i == len(jobs) - 1 else "partial")
            buf = io.BytesIO()
            np.savez_compressed(buf, **{f"{j}__{k}": v for j, d in per.items() for k, v in d.items()})
            durable_write(os.path.splitext(path)[0] + "_pershot.npz", lambda f: f.write(buf.getvalue()))
        finalise_run(path, stamp, ckpts)
        return dict(path=path, doc=read_result_file(path), fresh=True)

    return (run_e05,)


@app.function
def plot_e05(rows):
    codes = list(dict.fromkeys(r["code"] for r in rows))
    col = plot_colours(codes)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    for c in codes:
        rs = sorted([r for r in rows if r["code"] == c], key=lambda r: r["p"])
        ps = [r["p"] for r in rs]
        for r in rs:
            if r["silent"]:
                a.plot(r["p"], r["silent_rate"], "o", color=col[c])
            a.plot(r["p"], r["silent_upper_95"], "v", ms=8, mfc="none", mec=col[c])
        a.plot(ps, [r["silent_upper_95"] for r in rs], ":", color=col[c], label=c)
        b.plot(ps, [r["escalation_primary_fail"] for r in rs], "s-", color=col[c], label=c)
    a.set_yscale("log")
    a.set_xlabel("p")
    a.set_ylabel("silent failures per converged shot")
    a.set_title("Peeling: silent-failure rate (● measured, ▽ one-sided 95% CP bound)")
    b.set_xlabel("p")
    b.set_ylabel("fraction of shots peeling does not converge")
    b.set_title("primary_fail escalation rate")
    for ax in (a, b):
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


@app.cell
def _(e05_point):
    def _e05_point_fields():
        row, per = e05_point("[[72, 12, 6]]", 2, 3e-3, 200, 4)
        assert row["converged"] + int((~per["conv"]).sum()) == 200
        assert row["silent"] == int((per["conv"] & per["fail"]).sum())
        assert row["silent_upper_95"] >= row["silent_rate"]
        if row["silent"] == 0:
            assert row["headroom"] == 0 and abs(row["silent_upper_95"] - cp_upper(0, row["converged"])) < 1e-12

    def _e05_zero_bound():
        assert abs(cp_upper(0, 250_000) - 1.198e-5) < 1e-7      # ~3/n

    tests_e05 = run_checks([
        ("census rows: silent = converged ∧ failed, with a CP upper bound", _e05_point_fields),
        ("k = 0 bound is ≈ 3/n (1.2e-5 at 250k)", _e05_zero_bound),
    ])
    render_checks("Experiment 5", tests_e05)
    return (tests_e05,)


@app.cell
def _(E05_CODES, E05_MASTER_SEED):
    e05_ui = mo.ui.dictionary(dict(
        codes=mo.ui.multiselect(list(BB_PRESETS) + list(GB_PRESETS), value=E05_CODES, label="codes"),
        ps=mo.ui.multiselect(["0.0005", "0.001", "0.002", "0.003", "0.004"],
                             value=["0.001", "0.002", "0.003"], label="p"),
        rounds=mo.ui.number(start=1, stop=48, value=12, label="rounds T"),
        shots=mo.ui.number(start=100, stop=2_000_000, step=100, value=250_000, label="shots per point"),
        master_seed=mo.ui.number(value=E05_MASTER_SEED, label="master seed"),
        workers=mo.ui.slider(1, max(2, os.cpu_count() or 1), value=max(1, (os.cpu_count() or 2) - 2),
                             show_value=True, label="workers (no timing claim here)"),
    ))
    e05_run_btn = mo.ui.run_button(label="Run Experiment 5")
    mo.vstack([mo.md("### Experiment 5 — controls"), e05_ui, e05_run_btn])
    return e05_run_btn, e05_ui


@app.cell
def _(core_ready, e05_run_btn, e05_ui, run_e05, tests_e05):
    e05_run = None
    _s = e05_ui.value
    if not (core_ready and all_pass(tests_e05)):
        _out = mo.md("*Experiment 5 locked until every test passes.*")
    elif not e05_run_btn.value:
        _out = mo.md("*Press **Run Experiment 5**, or load a saved run below.*")
    else:
        _setup = dict(codes=list(_s["codes"]), ps=sorted(float(p) for p in _s["ps"]),
                      rounds=int(_s["rounds"]), shots=int(_s["shots"]),
                      master_seed=int(_s["master_seed"]), workers=int(_s["workers"]))
        with mo.status.progress_bar(total=len(_setup["codes"]) * len(_setup["ps"]),
                                    title="Experiment 5", show_eta=True) as _bar:
            e05_run = run_e05(_setup, on_progress=_bar.update)
        _out = mo.md(f"Saved `{e05_run['path']}`")
    _out
    return (e05_run,)


@app.cell
def _(e05_run):
    _ = e05_run
    e05_file, _files = saved_run_picker("E05", "Saved Experiment 5 run")
    e05_load_btn = mo.ui.run_button(label="Load")
    e05_reexport_btn = mo.ui.run_button(label="Re-export loaded run")
    mo.hstack([e05_file, e05_load_btn, e05_reexport_btn]) if _files else mo.md("*No saved runs yet.*")
    return e05_file, e05_load_btn, e05_reexport_btn


@app.cell
def _(e05_file, e05_load_btn):
    e05_loaded = (dict(path=e05_file.value, doc=read_result_file(e05_file.value, "E05-census"),
                       fresh=False) if (e05_load_btn.value and e05_file.value) else None)
    return (e05_loaded,)


@app.cell
def _(e05_loaded, e05_reexport_btn, e05_run):
    e05_data = e05_run if e05_run is not None else e05_loaded
    mo.stop(e05_data is None, mo.md("*Run Experiment 5 or load a saved run.*"))
    _rows = e05_data["doc"]["rows"]
    _tbl = ["| code | p | shots | converged | silent k | rate | one-sided 95% bound | on flagged | "
            "headroom | pf escalation |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for _r in _rows:
        _tbl.append(f"| {_r['code']} | {_r['p']:g} | {_r['shots']:,} | {_r['converged']:,} | "
                    f"{_r['silent']} | {_r['silent_rate']:.2e} | {_r['silent_upper_95']:.2e} | "
                    f"{_r['silent_on_flagged']} | {_r['headroom']} | {_r['escalation_primary_fail']:.2%} |")
    _tot = sum(r["silent"] for r in _rows)
    _head = sum(r["headroom"] or 0 for r in _rows)
    _fig = plot_e05(_rows)
    _exp = maybe_export(e05_data, "Experiment 5 — silent-failure census", {"census": _fig},
                        {"census": [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in _rows]},
                        reexport=e05_reexport_btn.value)
    mo.vstack([run_status_banner(e05_data["doc"], e05_data["path"]), mo.md("\n".join(_tbl)),
               mo.md(f"**{_tot} silent failures in total; headroom {_head}** — the most accuracy any "
                     "flag rule on a peeling primary could gain across this grid."),
               _fig, _exp])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiment 6 — surface-code baseline: BP+OSD-0 and MWPM (Fig. 5.1)

    **Question.** Does the pipeline — circuit, DEM, decoder, scoring — reproduce
    the well-known behaviour of the rotated surface code, a family with no stake
    in the thesis? Only circuit construction differs from the BB runs. This is
    the validation the decoder-input experiments rest on.

    **Design.** $d \in \{3, 5, 7\}$, $p \in \{5\times10^{-4}, 10^{-3},
    2\times10^{-3}, 3\times10^{-3}, 5\times10^{-3}, 8\times10^{-3},
    1.2\times10^{-2}\}$, $T = 6$, 20,000 shots per point; **Z-only detectors**
    (the BB convention). BP+OSD-0 and pymatching MWPM decode **identical
    samples**. Extension: $d = 5, 7$ at $p \in \{5\times10^{-4}, 10^{-3}\}$ with
    200,000 shots.

    **Statistics.**
    * LER per (d, p) with 95% Wilson intervals (zero failures: 97.5% one-sided
      bound).
    * MWPM vs BP+OSD-0 at each point: exact McNemar on the paired shots
      (descriptive).
    * **Threshold** $p^*$: the $d_5/d_7$ crossing interpolated in log–log
      coordinates, with a 2,000-resample parametric binomial bootstrap 95%
      interval, quoted to two significant figures; the $d_3/d_5$ and $d_3/d_7$
      crossings are also reported, for both decoders.
    * **Validation rule** (not in the Holm family): pass only if (1) for each
      decoder the $d_5/d_7$ curves cross inside the range with a 95% interval
      inside [0.5%, 2%], and (2) at the two lowest $p$ the 200k counts order
      $d_7 \le d_5$ with non-overlapping Wilson intervals for at least one $p$.

    Older surface-code files (every Stim detector kept, 20 BP iterations, no
    MWPM) still load, labelled `surface-all`.
    """)
    return


@app.cell
def _():
    E06_MASTER_SEED = 2026110601
    E06_PS = [5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 1.2e-2]
    E06_EXTENSION = dict(distances=[5, 7], ps=[5e-4, 1e-3], shots=200_000)
    return E06_EXTENSION, E06_MASTER_SEED, E06_PS


@app.function
def surface_sweep_point(distance, rounds, p, shots, seed, osd_order=0, workers=1,
                        x_detectors=False, on_chunk=None, checkpoint_dir=False):
    """One (d, p) point: BP+OSD-0 (the shared spec) and MWPM on identical samples."""
    pr = surface_code_problem(distance, rounds, p, x_detectors=x_detectors)
    det, obs = pr["circuit"].compile_detector_sampler(seed=int(seed)).sample(
        int(shots), separate_observables=True)
    det, obs = det.astype(np.uint8), obs.astype(np.uint8)
    w = parallel_workers(shots, workers)
    spec = strong_spec(osd_order)
    res, backend, info = run_zoo(
        pr["H"], pr["L"], pr["priors"], det, obs, [("strong", spec)], workers=w,
        chunk_size=CHUNK_SIZE if checkpoint_dir else chunk_for(shots, w), on_chunk=on_chunk,
        checkpoint_dir=checkpoint_dir, return_info=True,
        label=f"E06 surface d={distance}: T={rounds}, p={p:g}, {shots:,} shots, seed {seed}")
    f_osd = res["strong"]["fail"].astype(bool)
    mw = mwpm_decode(pr["circuit"], det, obs)
    f_mw = np.asarray(mw["fail"], bool)
    k, km = int(f_osd.sum()), int(f_mw.sum())
    ler, lo, hi = wilson(k, shots)
    lm, lom, him = wilson(km, shots)
    x, y, pv = paired_exact(f_osd, f_mw)
    return dict(distance=int(distance), rounds=int(rounds), p=float(p), shots=int(shots),
                seed=int(seed), osd_order=int(osd_order), decoder_spec=decoder_spec_full(spec),
                failures=k, ler=ler, ci_low=lo, ci_high=hi,
                mwpm_failures=km, mwpm_ler=lm, mwpm_ci_low=lom, mwpm_ci_high=him,
                osd_only=x, mwpm_only=y, mcnemar_p=pv,
                detectors=int(pr["H"].shape[0]), faults=int(pr["H"].shape[1]),
                qubits_used=pr["qubits_used"], qubit_index_span=pr["qubit_index_span"],
                detector_convention=pr["detector_convention"], x_detectors=bool(x_detectors),
                backend=backend, workers_effective=info["workers_effective"],
                resumed_chunks=info["resumed_chunks"], total_chunks=info["total_chunks"],
                time_p50_us=float(np.median(res["strong"]["time_us"])),
                mwpm_time_us_per_shot=float(mw["time_us"][0]) if len(det) else 0.0,
                sample_sha256=sample_hash(det, obs))


@app.function
def e06_thresholds(points, B=2000):
    """Bootstrap crossings for BP+OSD-0 and for MWPM."""
    osd = threshold_estimate(points, B=B, failures_key="failures")
    mw = threshold_estimate(points, B=B, failures_key="mwpm_failures") \
        if all("mwpm_failures" in p for p in points) else None
    return dict(osd=osd, mwpm=mw)


@app.function
def e06_decision(points, thresholds, extension=None):
    """The pre-registered validation rule (see above)."""
    ok1 = {}
    for name, th in thresholds.items():
        if th is None:
            continue
        h = th.get("d5-d7", {})
        ok1[name] = bool(h.get("p_star") and h.get("ci") and 5e-3 <= h["ci"][0] and h["ci"][1] <= 2e-2)
    ok2, detail = None, []
    if extension:
        by = {(pt["distance"], pt["p"]): pt for pt in extension}
        ok2 = True
        any_sep = False
        for p in sorted({pt["p"] for pt in extension})[:2]:
            a, b = by.get((5, p)), by.get((7, p))
            if a is None or b is None:
                ok2 = False
                continue
            order = b["failures"] <= a["failures"]
            sep = b["ci_high"] < a["ci_low"]
            any_sep |= sep
            ok2 &= order
            detail.append(dict(p=p, d5=a["failures"], d7=b["failures"], ordered=order, separated=sep))
        ok2 = bool(ok2 and any_sep)
    passed = all(ok1.values()) and bool(ok1) and (ok2 is not False)
    return dict(crossing_ok=ok1, extension_ok=ok2, extension=detail,
                passed=passed if ok2 is not None else None,
                note="condition (2) needs the 200k extension" if ok2 is None else "")


@app.function
def run_e06(setup, mode="sweep", on_progress=None):
    """mode 'sweep' (main grid) or 'extension' (200k at the two lowest p)."""
    if mode == "extension":
        ds, ps, shots = setup["ext_distances"], setup["ext_ps"], setup["ext_shots"]
    else:
        ds, ps, shots = setup["distances"], setup["ps"], setup["shots"]
    stamp = start_run("E06-surface", "E06")
    path = os.path.join(RESULTS_DIR, result_stem(
        "E06", rounds=setup["rounds"], p=ps, shots=shots, seed=setup["master_seed"], osd=0,
        convention="surface-all" if setup["x_detectors"] else "surface-z-only",
        extra=[f"d{'-'.join(map(str, ds))}", mode, "mwpm"]) + ".json")
    seeds = {f"{d}|{p:g}": derive_seed(setup["master_seed"], d, float(p), setup["rounds"], mode)
             for d in ds for p in ps}
    config = dict(code="rotated surface code (Z memory)", distances=ds, rounds=setup["rounds"],
                  ps=ps, shots=shots, master_seed=setup["master_seed"], seeds=seeds, mode=mode,
                  x_detectors=bool(setup["x_detectors"]),
                  detector_convention="surface-all" if setup["x_detectors"] else "surface-z-only",
                  decoders={"BP+OSD-0": decoder_spec_full(strong_spec(0)),
                            "MWPM": dict(kind="mwpm", package="pymatching",
                                         dem="decompose_errors=True")})
    config = json.loads(json.dumps(config))
    stamp = resume_stamp(path, stamp, config)
    pts, ckpts = [], []
    stem = os.path.splitext(os.path.basename(path))[0]
    jobs = [(d, p) for d in ds for p in ps]
    for i, (d, p) in enumerate(jobs):
        ck = os.path.join(CKPT["root"], stem, f"d{d}_p{p:g}") if shots >= CKPT["min_shots"] else False
        if ck:
            ckpts.append(ck)
        pts.append(surface_sweep_point(d, setup["rounds"], p, shots, seeds[f"{d}|{p:g}"],
                                       workers=setup["workers"], x_detectors=setup["x_detectors"],
                                       checkpoint_dir=ck))
        if on_progress:
            on_progress()
        th = e06_thresholds(pts, B=setup.get("B", 2000)) if mode == "sweep" else None
        write_result_file(path, dict(
            config=config, points=pts, thresholds=th,
            timing=dict(workers_requested=setup["workers"],
                        workers_effective=sorted({x["workers_effective"] for x in pts}),
                        backends=sorted({x["backend"].split(";")[0] for x in pts}),
                        resumed_chunks=sum(x["resumed_chunks"] for x in pts),
                        total_chunks=sum(x["total_chunks"] for x in pts)),
            samples={f"{x['distance']}|{x['p']:g}": x["sample_sha256"] for x in pts}),
            stamp, "complete" if i == len(jobs) - 1 else "partial")
    finalise_run(path, stamp, ckpts)
    return dict(path=path, doc=read_result_file(path), fresh=True)


@app.function
def e06_load(path):
    d = read_result_file(path, ("E06-surface", "exp7"))
    if d.get("kind") == "exp7":
        cfg, prov = config_from_file(d, path)
        d = dict(d, config=dict(d["config"], **{k: cfg[k] for k in ("x_detectors", "detector_convention")}),
                 provenance=prov, thresholds=dict(osd=threshold_estimate(d["points"], B=2000), mwpm=None))
    return dict(path=path, doc=d, fresh=False)


@app.cell
def _(E06_PS):
    def _e06_mwpm_beside_osd():
        pt = surface_sweep_point(3, 3, 3e-3, 2000, 7)
        assert pt["mwpm_failures"] <= pt["failures"] + 3 * math.sqrt(pt["failures"] + 1)
        assert pt["detectors"] == 16 and pt["qubits_used"] == 17
        assert pt["decoder_spec"]["max_iter"] == STRONG_MAX_ITER
        assert pt["osd_only"] - pt["mwpm_only"] == pt["failures"] - pt["mwpm_failures"]

    def _e06_default_grid():
        """the default grid reaches above 1% so the curves can cross."""
        assert max(E06_PS) > 0.01 and len(E06_PS) == 7

    def _e06_decision_rule():
        th = dict(osd={"d5-d7": dict(p_star=1.0e-2, ci=(8e-3, 1.3e-2))},
                  mwpm={"d5-d7": dict(p_star=1.0e-2, ci=(4e-3, 1.3e-2))})
        d = e06_decision([], th)
        assert d["crossing_ok"] == {"osd": True, "mwpm": False} and d["passed"] is None
        ext = [dict(distance=5, p=5e-4, failures=9, ci_low=5e-5, ci_high=8e-5),
               dict(distance=7, p=5e-4, failures=1, ci_low=1e-7, ci_high=2e-5),
               dict(distance=5, p=1e-3, failures=80, ci_low=3e-4, ci_high=5e-4),
               dict(distance=7, p=1e-3, failures=11, ci_low=3e-5, ci_high=9e-5)]
        th["mwpm"]["d5-d7"]["ci"] = (8e-3, 1.3e-2)
        assert e06_decision([], th, ext)["passed"] is True

    tests_e06 = run_checks([
        ("MWPM and BP+OSD-0 on identical samples; Z-only counts; shared spec", _e06_mwpm_beside_osd),
        ("the default p grid includes a value above 1%", _e06_default_grid),
        ("the registered validation rule", _e06_decision_rule),
    ])
    render_checks("Experiment 6", tests_e06)
    return (tests_e06,)


@app.cell
def _(E06_MASTER_SEED, E06_PS):
    e06_ui = mo.ui.dictionary(dict(
        distances=mo.ui.multiselect(["3", "5", "7", "9"], value=["3", "5", "7"], label="distances"),
        ps=mo.ui.multiselect([f"{p:g}" for p in E06_PS], value=[f"{p:g}" for p in E06_PS], label="p"),
        rounds=mo.ui.number(start=1, stop=24, value=6, label="rounds T"),
        shots=mo.ui.number(start=100, stop=1_000_000, step=100, value=20_000, label="shots per point"),
        ext_shots=mo.ui.number(start=100, stop=2_000_000, step=100, value=200_000,
                               label="extension shots"),
        master_seed=mo.ui.number(value=E06_MASTER_SEED, label="master seed"),
        x_detectors=mo.ui.switch(value=False, label="keep Stim's X-type detectors (older convention)"),
        workers=mo.ui.slider(1, max(2, os.cpu_count() or 1), value=1, show_value=True, label="workers"),
        mode=mo.ui.radio({"main sweep": "sweep", "200k extension (d = 5, 7; two lowest p)": "extension"},
                         value="main sweep", label="run"),
    ))
    e06_run_btn = mo.ui.run_button(label="Run Experiment 6")
    mo.vstack([mo.md("### Experiment 6 — controls"), e06_ui, e06_run_btn])
    return e06_run_btn, e06_ui


@app.cell
def _(E06_EXTENSION, core_ready, e06_run_btn, e06_ui, tests_e06):
    e06_run = None
    _s = e06_ui.value
    if not (core_ready and all_pass(tests_e06)):
        _out = mo.md("*Experiment 6 locked until every test passes.*")
    elif not e06_run_btn.value:
        _out = mo.md("*Press **Run Experiment 6**, or load saved runs below.*")
    else:
        _setup = dict(distances=sorted(int(d) for d in _s["distances"]),
                      ps=sorted(float(p) for p in _s["ps"]), rounds=int(_s["rounds"]),
                      shots=int(_s["shots"]), master_seed=int(_s["master_seed"]),
                      x_detectors=bool(_s["x_detectors"]), workers=int(_s["workers"]),
                      ext_distances=E06_EXTENSION["distances"], ext_ps=E06_EXTENSION["ps"],
                      ext_shots=int(_s["ext_shots"]))
        _n = (len(_setup["distances"]) * len(_setup["ps"]) if _s["mode"] == "sweep"
              else len(_setup["ext_distances"]) * len(_setup["ext_ps"]))
        with mo.status.progress_bar(total=_n, title="Experiment 6", show_eta=True) as _bar:
            e06_run = run_e06(_setup, mode=_s["mode"], on_progress=_bar.update)
        _out = mo.md(f"Saved `{e06_run['path']}`")
    _out
    return (e06_run,)


@app.cell
def _(e06_run):
    _ = e06_run
    _files = saved_runs("E06") + saved_runs("EXP7")
    e06_file = mo.ui.dropdown({os.path.basename(f): f for f in _files},
                              value=next((os.path.basename(f) for f in reversed(_files)
                                          if "_sweep_" in f), None) if _files else None,
                              label="main sweep")
    e06_ext_file = mo.ui.dropdown({os.path.basename(f): f for f in _files if "_extension_" in f},
                                  label="200k extension (optional)")
    e06_load_btn = mo.ui.run_button(label="Load")
    e06_reexport_btn = mo.ui.run_button(label="Re-export loaded run")
    mo.hstack([e06_file, e06_ext_file, e06_load_btn, e06_reexport_btn]) if _files else \
        mo.md("*No saved runs yet.*")
    return e06_ext_file, e06_file, e06_load_btn, e06_reexport_btn


@app.cell
def _(e06_ext_file, e06_file, e06_load_btn):
    e06_loaded = e06_load(e06_file.value) if (e06_load_btn.value and e06_file.value) else None
    e06_ext = e06_load(e06_ext_file.value) if (e06_load_btn.value and e06_ext_file.value) else None
    return e06_ext, e06_loaded


@app.cell
def _(e06_ext, e06_loaded, e06_reexport_btn, e06_run):
    e06_data = e06_run if (e06_run is not None and e06_run["doc"]["config"]["mode"] == "sweep") else e06_loaded
    mo.stop(e06_data is None, mo.md("*Run the Experiment 6 sweep or load a saved one.*"))
    _pts = e06_data["doc"]["points"]
    _th = e06_data["doc"].get("thresholds") or e06_thresholds(_pts)
    _ext_doc = (e06_run if (e06_run is not None and e06_run["doc"]["config"]["mode"] == "extension")
                else e06_ext)
    _ext = _ext_doc["doc"]["points"] if _ext_doc else None
    _dec = e06_decision(_pts, _th, _ext)
    _rows = ["| d | p | shots | BP+OSD-0 failures | LER [95% CI] | MWPM failures | MWPM LER | "
             "OSD-only : MWPM-only (p) | detectors | qubits |",
             "|--:|--:|--:|--:|:--|--:|--:|:--|--:|--:|"]
    for _pt in sorted(_pts + (_ext or []), key=lambda r: (r["distance"], r["p"], r["shots"])):
        _mw = (f"{_pt['mwpm_failures']} | {_pt['mwpm_ler']:.2e} | {_pt['osd_only']} : "
               f"{_pt['mwpm_only']} ({_pt['mcnemar_p']:.2g})" if "mwpm_failures" in _pt else "— | — | —")
        _rows.append(f"| {_pt['distance']} | {_pt['p']:g} | {_pt['shots']:,} | {_pt['failures']} | "
                     f"{_pt['ler']:.2e} [{_pt['ci_low']:.1e}, {_pt['ci_high']:.1e}] | {_mw} | "
                     f"{_pt['detectors']} | {_pt.get('qubits_used', _pt.get('qubits'))} |")
    _thr_lines = []
    for _name, _t in _th.items():
        if not _t:
            continue
        for _pair in ("d5-d7", "d3-d5", "d3-d7"):
            _h = _t.get(_pair)
            if _h and _h.get("p_star"):
                _ci = f" [{_h['ci'][0]:.2g}, {_h['ci'][1]:.2g}]" if _h.get("ci") else ""
                _thr_lines.append(f"- {_name.upper()} {_pair}: p* = {_h['p_star']:.2g}{_ci} "
                                  f"(no crossing in {_h['frac_no_crossing']:.0%} of resamples)")
            elif _h:
                _thr_lines.append(f"- {_name.upper()} {_pair}: no crossing in range")
    _fig = plot_surface_sweep(_pts, threshold=_th.get("osd"), mwpm_threshold=_th.get("mwpm"))
    _verdict = ("**Validation passed.**" if _dec["passed"] else
                "**Validation not yet decided** — " + _dec["note"] if _dec["passed"] is None else
                "**Validation failed** — see the crossing intervals and extension ordering.")
    _exp = maybe_export(e06_data, "Experiment 6 — surface-code baseline (Fig. 5.1)",
                        {"surface_sweep": _fig},
                        {"points": [{k: v for k, v in p.items() if not isinstance(v, dict)} for p in _pts],
                         "extension": [{k: v for k, v in p.items() if not isinstance(v, dict)}
                                       for p in (_ext or [])]},
                        notes="\n".join(_thr_lines), reexport=e06_reexport_btn.value)
    mo.vstack([run_status_banner(e06_data["doc"], e06_data["path"]),
               provenance_banner(dict(provenance=e06_data["doc"].get("provenance", {}),
                                      status=e06_data["doc"].get("status"))),
               mo.md("#### Threshold crossings (log–log, parametric bootstrap 95%)\n\n"
                     + "\n".join(_thr_lines)),
               mo.md(_verdict), _fig, mo.md("\n".join(_rows)), _exp])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiment 7 — the accuracy–latency frontier (Table 5.9, Fig. 5.5)

    **Question.** Counting decode time as well as accuracy, which way of using
    the flags — none, blind, sighted, updated — and which OSD order sit on the
    Pareto frontier?

    **Design.** The OSD-0 points are Experiment 8's A1 ([[72,12,6]],
    $p = 10^{-3}$) and A3 ([[144,12,12]]), decoded single-threaded. This
    experiment adds OSD-CS2 points **on the same samples** — its seeds come from
    Experiment 8's master seed. A point runs at full size only if its 2,000-shot
    pilot median is below **3× updated-OSD-0's median** (pre-registered pruning
    rule):

    * [[72,12,6]]: unflagged-OSD-CS2 and updated-OSD-CS2 at 50,000 shots;
      blind-OSD-CS2 and sighted-OSD-CS2 at 10,000 shots (for the p99 values);
    * [[144,12,12]]: unflagged-OSD-CS2 at 20,000 shots; blind-OSD-CS2 is pruned
      (its median is hundreds of milliseconds, so it cannot dominate
      updated-OSD-0).

    **Statistics.** Each point is an LER with a 95% Wilson interval and a median
    and p99 decode time; **the updated arm's time includes its prior update**
    (§15). A point is on the frontier if no other point is at least as fast and
    at least as accurate and strictly better in one. The frontier is drawn only
    from points timed at the same worker count (workers = 1); points from
    different worker counts are refused.
    """)
    return


@app.cell
def _(E08_MASTER_SEED):
    E07_DESIGN = {
        "F1": dict(code="[[72, 12, 6]]", rounds=12, p=1e-3, shots=50_000, decoder="OSD-CS2",
                   arms=["unflagged", "updated"], purpose="frontier, [[72,12,6]]"),
        "F2": dict(code="[[72, 12, 6]]", rounds=12, p=1e-3, shots=10_000, decoder="OSD-CS2",
                   arms=["blind", "sighted"], purpose="frontier p99, [[72,12,6]]"),
        "F3": dict(code="[[144, 12, 12]]", rounds=12, p=1e-3, shots=20_000, decoder="OSD-CS2",
                   arms=["unflagged"], purpose="frontier, [[144,12,12]]"),
    }
    E07_PRUNE_FACTOR = 3.0

    def e07_pilot(code_key, p, candidates, shots=2000, seed=None, rounds=12):
        """
        The pruning rule: time each candidate (arm, decoder) on a 2,000-shot pilot
        (single-threaded) and keep it iff its median is below 3x updated-OSD-0's.
        Returns rows with the medians and the keep decision.
        """
        seed = seed if seed is not None else derive_seed(E08_MASTER_SEED, "pilot", code_key, p)
        ref = four_arm_run(code_key, rounds, p, shots, seed, decoder="OSD-0", arms=("updated",),
                           checkpoint_dir=False, keep_per_shot=False)["arms"]["updated"]["time_p50_us"]
        rows = []
        for arm, dec in candidates:
            m = four_arm_run(code_key, rounds, p, shots, seed, decoder=dec, arms=(arm,),
                             checkpoint_dir=False, keep_per_shot=False)["arms"][arm]["time_p50_us"]
            rows.append(dict(code=code_key, p=p, arm=arm, decoder=dec, pilot_median_us=m,
                             updated_osd0_median_us=ref, keep=bool(m < E07_PRUNE_FACTOR * ref)))
        return rows

    return E07_DESIGN, E07_PRUNE_FACTOR, e07_pilot


@app.function
def frontier_points(*result_lists):
    """Flat frontier points (arm x decoder) from one or more grid result lists."""
    pts = []
    for results in result_lists:
        for r in results:
            for arm, m in r["arms"].items():
                pts.append(dict(m, arm=arm, decoder=r["decoder"], code=r["code"], p=r["p"],
                                shots=r["shots"], osd_order=r["decoder_spec"].get("order", 0),
                                config_id=r.get("config_id")))
    return pts


@app.function
def frontier_verdict(points):
    """Whether flags earn their decode time, and the cheapest way to each LER."""
    regimes = timing_regimes(points)
    if len(regimes) > 1:
        return dict(frontier=[], flagged_on_frontier=[], flags_pay=False, best_ler_arm=None,
                    regimes=regimes, comparable=False,
                    verdict=f"not comparable: points span worker counts {regimes}")
    front = pareto_frontier(points)
    flagged = [r for r in front if r["arm"] != "unflagged"]
    best = min(points, key=lambda r: r["ler"]) if points else None
    names = [f"{r['arm']}/{r['decoder']}" for r in flagged]
    if any(r["arm"] == "updated" for r in front) and len(front) == 1:
        verdict = "prior updating dominates every other point"
    elif flagged:
        verdict = "flags are on the frontier"
    else:
        verdict = "the frontier is entirely unflagged"
    return dict(frontier=front, flagged_on_frontier=names, flags_pay=bool(flagged),
                regimes=regimes, comparable=True, best_ler_arm=best["arm"] if best else None,
                verdict=verdict)


@app.cell
def _(E07_DESIGN, E08_MASTER_SEED):
    def _e07_same_samples_as_e08():
        """Frontier configurations derive A1's seed, so the flagged arms are paired with A1."""
        s_f1 = derive_seed(E08_MASTER_SEED, "[[72, 12, 6]]", 1e-3, 12, 0)
        cfgs, _sc, _m, _w = grid_setup(dict(configs=["F1"], scale="1 (registered design)",
                                            master_seed=E08_MASTER_SEED, workers=1), E07_DESIGN)
        assert cfgs[0].get("seed") is None        # derived by run_grid from the master and indices
        assert derive_seed(E08_MASTER_SEED, cfgs[0]["code"], cfgs[0]["p"], cfgs[0]["rounds"], 0) == s_f1

    def _e07_verdicts():
        pts = [dict(arm="unflagged", decoder="OSD-0", time_p50_us=10, ler=1e-2, workers_effective=1),
               dict(arm="updated", decoder="OSD-0", time_p50_us=12, ler=1e-3, workers_effective=1),
               dict(arm="sighted", decoder="OSD-0", time_p50_us=30, ler=2e-3, workers_effective=1)]
        v = frontier_verdict(pts)
        assert v["flagged_on_frontier"] == ["updated/OSD-0"] and v["flags_pay"]
        v2 = frontier_verdict(pts + [dict(pts[0], workers_effective=12)])
        assert not v2["comparable"]

    tests_e07 = run_checks([
        ("frontier points decode Experiment 8's A1/A3 samples", _e07_same_samples_as_e08),
        ("frontier verdicts; mixed worker counts are not comparable", _e07_verdicts),
    ])
    render_checks("Experiment 7", tests_e07)
    return (tests_e07,)


@app.cell
def _(E07_DESIGN, E08_MASTER_SEED):
    e07_ui = grid_design_ui(E07_DESIGN, E08_MASTER_SEED)
    e07_pilot_btn = mo.ui.run_button(label="Run the 2,000-shot pruning pilot")
    e07_run_btn = mo.ui.run_button(label="Run Experiment 7")
    mo.vstack([mo.md("### Experiment 7 — controls\n\nThe master seed defaults to **Experiment 8's** "
                     "so the frontier points share A1/A3's samples. Run at workers = 1.\n\n"
                     + design_table_md(E07_DESIGN)), e07_ui, mo.hstack([e07_pilot_btn, e07_run_btn])])
    return e07_pilot_btn, e07_run_btn, e07_ui


@app.cell
def _(E07_PRUNE_FACTOR, e07_pilot, e07_pilot_btn):
    mo.stop(not e07_pilot_btn.value)
    _rows = []
    for _code, _cands in (("[[72, 12, 6]]", [("unflagged", "OSD-CS2"), ("updated", "OSD-CS2"),
                                             ("blind", "OSD-CS2"), ("sighted", "OSD-CS2")]),
                          ("[[144, 12, 12]]", [("unflagged", "OSD-CS2"), ("blind", "OSD-CS2")])):
        _rows += e07_pilot(_code, 1e-3, _cands)
    mo.md("| code | arm | decoder | pilot median µs | 3 × updated-OSD-0 | run at full size? |\n"
          "|:--|:--|:--|--:|--:|:--|\n" + "\n".join(
              f"| {r['code']} | {r['arm']} | {r['decoder']} | {r['pilot_median_us']:,.0f} | "
              f"{E07_PRUNE_FACTOR * r['updated_osd0_median_us']:,.0f} | {'yes' if r['keep'] else 'pruned'} |"
              for r in _rows))
    return


@app.cell
def _(E07_DESIGN, core_ready, e07_run_btn, e07_ui, tests_e07):
    e07_run, _out = grid_run_cell("E07", "E07-frontier", e07_ui.value, E07_DESIGN,
                                  core_ready and all_pass(tests_e07), e07_run_btn.value,
                                  "Experiment 7")
    _out
    return (e07_run,)


@app.cell
def _(e07_run):
    _ = e07_run
    e07_file, _files = grid_picker("E07")
    e07_e08_file, _f8 = grid_picker("E08")
    e07_load_btn = mo.ui.run_button(label="Load")
    e07_reexport_btn = mo.ui.run_button(label="Re-export loaded run")
    mo.vstack([mo.hstack([e07_file, e07_load_btn, e07_reexport_btn]),
               mo.md("OSD-0 points from Experiment 8:"), e07_e08_file]) if _files else \
        mo.md("*No saved runs yet.*")
    return e07_e08_file, e07_file, e07_load_btn, e07_reexport_btn


@app.cell
def _(e07_e08_file, e07_file, e07_load_btn):
    e07_loaded = None
    if e07_load_btn.value and e07_file.value:
        _d, _r, _p = load_grid(e07_file.value, "E07-frontier")
        e07_loaded = dict(path=e07_file.value, doc=_d, results=_r, per_shot=_p, fresh=False)
    e07_osd0 = (load_grid(e07_e08_file.value, "E08-prior-update")[1]
                if (e07_load_btn.value and e07_e08_file.value) else [])
    return e07_loaded, e07_osd0


@app.cell
def _(e07_loaded, e07_osd0, e07_reexport_btn, e07_run):
    e07_data = e07_run if e07_run is not None else e07_loaded
    mo.stop(e07_data is None, mo.md("*Run Experiment 7 or load a saved run.*"))
    _base = [r for r in e07_osd0 if r["config_id"].split("_s")[0] in
             {c["config_id"].split("_s")[0] for c in e07_data["results"]} and not r.get("tag")]
    _out_md, _figs, _tables = [], {}, {}
    for _code in dict.fromkeys(r["code"] for r in e07_data["results"]):
        _pts = frontier_points([r for r in e07_data["results"] if r["code"] == _code],
                               [r for r in _base if r["code"] == _code])
        _v = frontier_verdict(_pts)
        _front = {(r["arm"], r["decoder"]) for r in pareto_frontier(_pts, strict=False)}
        _t = ["| arm | decoder | shots | failures | LER [95% CI] | median µs | p99 µs | workers | "
              "on frontier |", "|:--|:--|--:|--:|:--|--:|--:|--:|:--|"]
        for _r in sorted(_pts, key=lambda r: r["time_p50_us"]):
            _t.append(f"| {_r['arm']} | {_r['decoder']} | {_r['shots']:,} | {_r['failures']} | "
                      f"{_r['ler']:.2e} [{_r['ci_low']:.2e}, {_r['ci_high']:.2e}] | "
                      f"{_r['time_p50_us']:,.0f} | {_r['time_p99_us']:,.0f} | {_r['workers_effective']} | "
                      f"{'★' if (_r['arm'], _r['decoder']) in _front else ''} |")
        _out_md.append(mo.md(f"#### Table 5.9 — {_code}: {_v['verdict']}\n\n" + "\n".join(_t)))
        _figs[f"frontier_{_code.strip('[]').replace(', ', '-')}"] = plot_pareto(_pts)
        _tables[f"table_5_9_{_code.strip('[]').replace(', ', '-')}"] = [
            {k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in _pts]
    _exp = maybe_export(e07_data, "Experiment 7 — accuracy–latency frontier (Table 5.9)", _figs,
                        _tables, reexport=e07_reexport_btn.value)
    mo.vstack([run_status_banner(e07_data["doc"], e07_data["path"]),
               mo.md("" if e07_osd0 else "*Load the Experiment 8 file too, to put the OSD-0 points "
                                         "on the same frontier.*"),
               *_out_md, *_figs.values(), _exp])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiment 8 — exact flag-conditioned prior updating (Table 5.8, Fig. 5.4)

    **Question.** Proposition 4.2 (Gallager's parity-check posterior, §11) gives,
    for every fault a flag touches, its exact posterior probability given whether
    that flag fired. Feeding those
    posteriors to BP+OSD as per-shot priors (`updated`) keeps the decoder on the
    **blind** matrix — no extra rows — yet uses the flag information. Does it
    beat adding the flag rows (`sighted`)? Is it cheaper, **once the per-shot
    prior update is paid for**? (Both arms see the same bits; if the updated arm
    wins, the gap is one of inference, and Experiment 12 asks where it comes
    from.)

    **Design.** The four-arm runner of §15 with BP+OSD-0 (30 iterations),
    $T = 12$:

    * A1 [[72,12,6]], $p = 10^{-3}$, 50,000 shots — Holm **T1**, and the
      single-thread timing run for **T14**;
    * A2 [[72,12,6]], $2\times10^{-3}$, 20,000 — **T2**;
    * A3 [[144,12,12]], $10^{-3}$, 150,000 — **T3**, single-thread timing.
      150,000 rather than 100,000 because Holm's strictest threshold
      (0.05/14) needs at least 10 one-way discordant shots;
    * A7 [[46,2,9]] and A8 [[48,6,8]] at $10^{-3}$, 50,000 — **T5**, **T6**
      (these codes fail on more than half of the shots at $3\times10^{-3}$);
    * A9 [[144,12,12]] at $2\times10^{-3}$, 50,000 — the third point of the
      [[144,12,12]] $p$-sweep for **T11** (optional);
    * A7x — [[46,2,9]] unflagged with OSD-CS4, 20,000 shots, a cross-check of
      the [[46,2,9]] OSD-0 error rate;
    * replication — A1, A2 and A3 (at 100,000) again under an **independent**
      master seed, reported beside the main runs and never pooled;
    * A1-X — A1's configuration as an **X-basis** memory with the flags on the
      Z checks (§4), 50,000 shots, own seed (optional, exploratory, not in the
      Holm family): a check that the results do not depend on the memory basis.

    **T4** ([[288,12,18]]) uses the A6 shards of Experiment 2, summed.

    **Statistics.**
    * updated vs sighted, updated vs blind, blind vs sighted: exact McNemar on
      the discordant shots (paired: same samples), reported with the split and a
      failure ratio.
    * **Time (T14)**: per-shot times of updated and sighted on the same shots,
      Wilcoxon signed-rank test (one-sided), with the median-time ratio and a
      paired bootstrap interval. The updated arm's per-shot time **includes** the
      prior update; the table shows the update share separately.
    * **T11**: weighted least-squares slope of $\ln(k_\text{blind}/k_\text{updated})$
      on $\ln p$ with one intercept per code, one-sided $t$.
    """)
    return


@app.cell
def _():
    E08_MASTER_SEED = 2026110801
    E08_MASTER_SEED_2 = 2026110802          # independent replication
    E08_DESIGN = {
        "A1": dict(code="[[72, 12, 6]]", rounds=12, p=1e-3, shots=50_000,
                   purpose="Table 5.8; Holm T1, T14 (timing, workers 1)"),
        "A2": dict(code="[[72, 12, 6]]", rounds=12, p=2e-3, shots=20_000, purpose="Holm T2"),
        "A3": dict(code="[[144, 12, 12]]", rounds=12, p=1e-3, shots=150_000,
                   purpose="Holm T3 (timing, workers 1)"),
        "A7": dict(code="[[46, 2, 9]]", rounds=12, p=1e-3, shots=50_000, purpose="Holm T5"),
        "A8": dict(code="[[48, 6, 8]]", rounds=12, p=1e-3, shots=50_000, purpose="Holm T6"),
        "A9": dict(code="[[144, 12, 12]]", rounds=12, p=2e-3, shots=50_000, purpose="Holm T11 (optional)"),
        "A7x": dict(code="[[46, 2, 9]]", rounds=12, p=1e-3, shots=20_000, arms=["unflagged"],
                    decoder="OSD-CS4", purpose="cross-check of [[46,2,9]] OSD-0"),
        "A1-X": dict(code="[[72, 12, 6]]", rounds=12, p=1e-3, shots=50_000, basis="X", tag="xbasis",
                     seed=derive_seed(E08_MASTER_SEED, "X basis", "[[72, 12, 6]]", 1e-3, 12, 0),
                     purpose="X-basis mirror of A1 (exploratory, not in Holm)"),
        **{f"rep-{k}": dict(code=c, rounds=12, p=p, shots=n, tag="seed2",
                            seed=derive_seed(E08_MASTER_SEED_2, c, p, 12, 0),
                            purpose="replication, independent seed (not pooled)")
           for k, c, p, n in (("A1", "[[72, 12, 6]]", 1e-3, 50_000), ("A2", "[[72, 12, 6]]", 2e-3, 20_000),
                              ("A3", "[[144, 12, 12]]", 1e-3, 100_000))},
    }
    E08_DEFAULT = ["A1", "A2", "A3", "A7", "A8", "A7x"]
    return E08_DEFAULT, E08_DESIGN, E08_MASTER_SEED


@app.function
def basis_mirror_rows(results):
    """Each X-basis configuration beside its Z-basis twin (same code, T, p):
    failures per arm and the unpaired exact rate ratio X / Z (different circuits)."""
    z = {(r["code"], r["rounds"], r["p"]): r for r in results
         if r.get("basis", "Z") == "Z" and not r.get("tag")}
    rows = []
    for rx in results:
        if rx.get("basis") != "X":
            continue
        rz = z.get((rx["code"].replace(" (X basis)", ""), rx["rounds"], rx["p"]))
        if rz is None:
            continue
        for arm in [a for a in FOUR_ARMS if a in rx["arms"] and a in rz["arms"]]:
            kx, kz = rx["arms"][arm]["failures"], rz["arms"][arm]["failures"]
            rr = rate_ratio_ci(kx, rx["shots"], kz, rz["shots"])
            rows.append(dict(code=rz["code"], p=rz["p"], arm=arm, shots_x=rx["shots"],
                             shots_z=rz["shots"], failures_x=kx, failures_z=kz,
                             ratio=rr["ratio"], lo=rr["lo"], hi=rr["hi"], p_value=rr["p"]))
    return rows


@app.function
def table_5_8_rows(results):
    """Table 5.8: per configuration, the paired arms with times, iterations and tests."""
    rows = []
    for r in combine_shards(results) if any(x.get("shard") for x in results) else results:
        a = r["arms"]
        row = dict(code=r["code"], p=r["p"], shots=r["shots"], decoder=r["decoder"],
                   tag=r.get("tag"))
        for arm in FOUR_ARMS:
            if arm in a:
                row[f"{arm}_failures"] = a[arm]["failures"]
                row[f"{arm}_ler"] = a[arm]["ler"]
                row[f"{arm}_p50_us"] = a[arm].get("time_p50_us")
                row[f"{arm}_p99_us"] = a[arm].get("time_p99_us")
                row[f"{arm}_iters"] = a[arm].get("bp_iters_mean")
        for key_, c in r["paired"].items():
            row[f"{key_}_split"] = f"{c['a_only']}:{c['b_only']}"
            row[f"{key_}_p"] = c["p"]
        rows.append(row)
    return rows


@app.function
def e08_headline(res, alpha=0.05):
    """
    One sentence for a configuration: does prior updating EXCEED, MATCH or
    NOT REACH the sighted arm (paired McNemar), and by how much vs blind.
    """
    a = res["arms"]
    c = res["paired"].get("sighted|updated")
    if c is None or "updated" not in a or "sighted" not in a:
        return "updated vs sighted: not measured"
    ku, ks = a["updated"]["failures"], a["sighted"]["failures"]
    s_only, u_only = c["a_only"], c["b_only"]
    if c["p"] < alpha and u_only < s_only:
        verb = "exceeds"
    elif c["p"] < alpha:
        verb = "does not reach"
    else:
        verb = "matches"
    txt = (f"prior updating {verb} the sighted arm on {res['code']} at p = {res['p']:g}: "
           f"{ku} vs {ks} failures (updated-only {u_only} : sighted-only {s_only}, p = {c['p']:.2g})")
    if "blind" in a and a["updated"]["failures"]:
        txt += f"; {a['blind']['failures'] / ku:.1f}× fewer failures than blind"
    return txt


@app.cell
def _(E08_DESIGN, E08_MASTER_SEED):
    def _e08_headline_branches():
        """updated 5, sighted 255, blind 902, discordant 252:2 -> "exceeds"."""
        res = dict(code="[[72, 12, 6]]", p=1e-3, arms=dict(updated=dict(failures=5),
                                                           sighted=dict(failures=255),
                                                           blind=dict(failures=902)),
                   paired={"sighted|updated": dict(a_only=252, b_only=2, p=mcnemar_counts(252, 2))})
        h = e08_headline(res)
        assert "exceeds" in h and "within 5%" not in h
        res["paired"]["sighted|updated"] = dict(a_only=3, b_only=4, p=1.0)
        assert "matches" in e08_headline(res)
        res["paired"]["sighted|updated"] = dict(a_only=2, b_only=40, p=mcnemar_counts(2, 40))
        assert "does not reach" in e08_headline(res)

    def _e08_mirror_rows():
        z = dict(code="[[72, 12, 6]]", rounds=12, p=1e-3, shots=1000, basis="Z", tag=None,
                 arms={"updated": dict(failures=4), "sighted": dict(failures=40)})
        x = dict(z, code="[[72, 12, 6]] (X basis)", basis="X", tag="xbasis",
                 arms={"updated": dict(failures=5), "sighted": dict(failures=38)})
        rows = basis_mirror_rows([z, x])
        assert {r["arm"] for r in rows} == {"updated", "sighted"}
        assert all(r["lo"] <= 1 <= r["hi"] for r in rows)

    def _e08_design():
        assert E08_DESIGN["A1-X"]["basis"] == "X" and E08_DESIGN["A1-X"]["tag"] == "xbasis"
        assert E08_DESIGN["A3"]["shots"] == 150_000 and E08_DESIGN["A7x"]["decoder"] == "OSD-CS4"
        s1 = E08_DESIGN["rep-A1"]["seed"]
        assert s1 != derive_seed(E08_MASTER_SEED, "[[72, 12, 6]]", 1e-3, 12, 0)
        ids = {grid_config_id(dict(c, shard=c.get("shard", 0))) for c in E08_DESIGN.values()}
        assert len(ids) == len(E08_DESIGN), "two configurations share a config id"

    def _e08_table_rows():
        r = four_arm_run(code_from_key("[[72, 12, 6]]"), 2, 2e-3, 30, 3, arms=("blind", "sighted", "updated"),
                         checkpoint_dir=False, keep_per_shot=False)
        rows = table_5_8_rows([dict(r, config_id="x")])
        assert rows[0]["updated_failures"] == r["arms"]["updated"]["failures"]
        assert "sighted|updated_split" in rows[0] and rows[0]["updated_iters"] >= 0

    tests_e08 = run_checks([
        ("registered design: A3 150k, A7x OSD-CS4, replication seeds independent, ids unique", _e08_design),
        ("headline says exceeds / matches / does not reach from the paired test", _e08_headline_branches),
        ("X-basis mirror rows: unpaired rate ratios against the Z-basis twin", _e08_mirror_rows),
        ("Table 5.8 rows carry every arm, time, iteration count and test", _e08_table_rows),
    ])
    render_checks("Experiment 8", tests_e08)
    return (tests_e08,)


@app.cell
def _(E08_MASTER_SEED):
    def smoke_cross_check(older_path, shots=2000, seed=None):
        """
        Before the long runs: the runner with the OLDER file's decoder spec
        (BP+OSD-0, 20 iterations) must reproduce that depth-series file's T = 12
        blind and sighted error rates within their Wilson intervals (overlap).
        """
        doc = read_result_file(older_path, ("exp2", "L2-depth-sweep"))
        pt = next(q for q in doc["points"] if q["rounds"] == 12)
        spec = dict(kind="osd", osd_order=0, max_iter=20)
        seed = seed if seed is not None else derive_seed(E08_MASTER_SEED, "smoke")
        r = four_arm_run(doc["config"]["code"], 12, doc["config"]["p"], shots, seed,
                         decoder="OSD-0", decoder_spec=spec, arms=("blind", "sighted"),
                         checkpoint_dir=False, keep_per_shot=False)
        rows = []
        for arm, old_arm in (("blind", "flagged, blind"), ("sighted", "flagged, sighted")):
            a, b = r["arms"][arm], pt["arms"][old_arm]
            rows.append(dict(arm=arm, new=a["ler"], new_ci=(a["ci_low"], a["ci_high"]), older=b["ler"],
                             older_ci=(b["ci_low"], b["ci_high"]),
                             overlap=bool(a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"])))
        return rows

    return (smoke_cross_check,)


@app.cell
def _():
    e08_smoke_btn = mo.ui.run_button(label="Smoke cross-check against an older depth-series file (≈1 min)")
    e08_smoke_btn
    return (e08_smoke_btn,)


@app.cell
def _(e08_smoke_btn, smoke_cross_check):
    mo.stop(not e08_smoke_btn.value)
    _cands = [f for f in saved_runs("EXP2") if "_72-12-6_p0.001_" in f and "_osd0" in f]
    mo.stop(not _cands, mo.md("*No `EXP2_72-12-6_p0.001_*_osd0*.json` in results/.*"))
    _rows = smoke_cross_check(max(_cands, key=os.path.getsize))
    mo.md("| arm | this runner (20 it.) | older file | intervals overlap |\n|:--|:--|:--|:--|\n" + "\n".join(
        f"| {r['arm']} | {r['new']:.2e} [{r['new_ci'][0]:.2e}, {r['new_ci'][1]:.2e}] | "
        f"{r['older']:.2e} [{r['older_ci'][0]:.2e}, {r['older_ci'][1]:.2e}] | "
        f"{'yes' if r['overlap'] else '**no**'} |" for r in _rows))
    return


@app.cell
def _(E08_DEFAULT, E08_DESIGN, E08_MASTER_SEED):
    e08_ui = grid_design_ui(E08_DESIGN, E08_MASTER_SEED, default=E08_DEFAULT)
    e08_run_btn = mo.ui.run_button(label="Run Experiment 8")
    mo.vstack([mo.md("### Experiment 8 — controls\n\nRun A1 and A3 at **workers = 1** on an "
                     "otherwise idle machine: they feed the latency claims (T14). A9 and the "
                     "replication runs are optional.\n\n" + design_table_md(E08_DESIGN)), e08_ui, e08_run_btn])
    return e08_run_btn, e08_ui


@app.cell
def _(E08_DESIGN, core_ready, e08_run_btn, e08_ui, tests_e08):
    e08_run, _out = grid_run_cell("E08", "E08-prior-update", e08_ui.value, E08_DESIGN,
                                  core_ready and all_pass(tests_e08), e08_run_btn.value,
                                  "Experiment 8")
    _out
    return (e08_run,)


@app.cell
def _(e08_run):
    _ = e08_run
    e08_file, _files = grid_picker("E08")
    e08_load_btn = mo.ui.run_button(label="Load")
    e08_reexport_btn = mo.ui.run_button(label="Re-export loaded run")
    mo.hstack([e08_file, e08_load_btn, e08_reexport_btn]) if _files else mo.md("*No saved runs yet.*")
    return e08_file, e08_load_btn, e08_reexport_btn


@app.cell
def _(e08_file, e08_load_btn):
    e08_loaded = None
    if e08_load_btn.value and e08_file.value:
        _d, _r, _p = load_grid(e08_file.value, "E08-prior-update")
        e08_loaded = dict(path=e08_file.value, doc=_d, results=_r, per_shot=_p, fresh=False)
    return (e08_loaded,)


@app.cell
def _(e08_loaded, e08_reexport_btn, e08_run):
    e08_data = e08_run if e08_run is not None else e08_loaded
    mo.stop(e08_data is None, mo.md("*Run Experiment 8 or load a saved run.*"))
    _res = e08_data["results"]
    _t = ["| config | code | p | shots | blind | sighted | updated | unflagged | "
          "sighted : updated only (p) | blind : updated only (p) | median µs b / s / u, u incl. prior update (update part) | "
          "BP iters b / s / u |", "|:--|:--|--:|--:|--:|--:|--:|--:|:--|:--|:--|:--|"]
    for _r in _res:
        _a = _r["arms"]
        _f = lambda a: f"{_a[a]['failures']}" if a in _a else "—"
        _pp = lambda k: (f"{_r['paired'][k]['a_only']} : {_r['paired'][k]['b_only']} "
                         f"({_r['paired'][k]['p']:.2g})" if k in _r["paired"] else "—")
        _tm = " / ".join(f"{_a[a]['time_p50_us']:,.0f}" if a in _a else "—"
                         for a in ("blind", "sighted", "updated"))
        if "updated" in _a:
            _tm += f" ({_a['updated']['time_update_p50_us']:,.0f})"
        _it = " / ".join(f"{_a[a]['bp_iters_mean']:.1f}" if a in _a else "—"
                         for a in ("blind", "sighted", "updated"))
        _t.append(f"| {_r['config_id']} | {_r['code']} | {_r['p']:g} | {_r['shots']:,} | "
                  f"{_f('blind')} | {_f('sighted')} | {_f('updated')} | {_f('unflagged')} | "
                  f"{_pp('sighted|updated')} | {_pp('blind|updated')} | {_tm} | {_it} |")
    _figs = {}
    for _r in _res:
        if len(_r["arms"]) > 1:
            _figs[f"arms_{_r['config_id']}"] = plot_prior_update(_r)
    _mir = basis_mirror_rows(_res)
    _exp = maybe_export(e08_data, "Experiment 8 — prior updating (Table 5.8)", _figs,
                        {"table_5_8": table_5_8_rows(_res), "arms": grid_rows(_res),
                         "comparisons": comparison_rows(_res), "x_basis_mirror": _mir},
                        reexport=e08_reexport_btn.value)
    mo.vstack([run_status_banner(e08_data["doc"], e08_data["path"]),
               mo.md("\n".join(f"- {e08_headline(r)}" for r in _res if "updated" in r["arms"])),
               mo.md("#### Table 5.8 — failures per arm (paired arms share samples)\n\n" + "\n".join(_t)),
               mo.md("#### X-basis mirror (exploratory)\n\n| code | p | arm | X basis | Z basis | "
                     "rate ratio X / Z [95% CI] | p |\n|:--|--:|:--|--:|--:|:--|--:|\n" + "\n".join(
                         f"| {m['code']} | {m['p']:g} | {m['arm']} | {m['failures_x']} / {m['shots_x']:,} | "
                         f"{m['failures_z']} / {m['shots_z']:,} | {m['ratio']:.2f} [{m['lo']:.2f}, "
                         f"{m['hi']:.2f}] | {m['p_value']:.2g} |" for m in _mir)
                     + "\n\nDifferent circuits, so unpaired (exact conditional rate ratio). A ratio "
                       "consistent with 1 in every arm means the results do not depend on the "
                       "memory basis.") if _mir else mo.md(""),
               *list(_figs.values())[:4],
               mo.accordion({"every arm": mo.md(grid_table_md(_res)),
                             "every comparison": mo.md(comparison_table_md(_res))}), _exp])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiment 9 — the check-weight family ($w = 4 \ldots 12$)

    **Question.** Laws L1–L7 say how the flag information scales with the X-check
    weight $w$: a flag covers $w$ faults (L1), fires with probability
    $S = (8w/15 + 2)p$ (L4), and its largest posterior is $38/(8w+30)$ (L5) — so
    heavier checks give **less** specific flags. Does the benefit of prior
    updating over a blind decoder shrink with $w$, as the laws predict?

    **Codes.** Nine generalised-bicycle codes on $\ell = 24$ ($n = 48$), one per
    $w = 4, \ldots, 12$. Their distances are not known in closed form; a
    randomised information-set search gives an **upper bound** before any
    sampling.

    **Design.** Four arms, $T = 12$.
    * *Pilot*: 2,000 shots per code at
      $p \in \{5\times10^{-4}, 7\times10^{-4}, 10^{-3}, 2\times10^{-3}, 3\times10^{-3}\}$
      (blind and updated arms).
    * *Freeze one p for all nine codes*: the **largest** $p$ at which every
      code's pilot blind LER is ≤ 0.35. The cap keeps every code away from
      saturation: as the blind LER approaches its ceiling, $k_\text{blind}$
      stops growing and $\ln(k_\text{blind}/k_\text{updated})$ is compressed —
      and the weak, heavy-check members would then fake the predicted negative
      slope.
    * *Main*: all four arms at the frozen $p$, with **per-code shot counts**
      chosen from the pilot so that every code projects ≥ 30 updated-arm
      failures: $N_c = \max(20{,}000,\ \lceil 1.2 \times 30 / \hat r_c \rceil)$,
      rounded up to 10,000 and capped at 200,000 ($\hat r_c$ = pilot updated
      failure rate; a code with no pilot failure gets the cap).

    **Deviation from the registered rule, recorded before main sampling.** The
    registration fixed $N = 20{,}000$ for every code and the *smallest* $p$ in
    $\{10^{-3}, 2\times10^{-3}, 3\times10^{-3}\}$ meeting both criteria. On these
    codes no such $p$ exists: at $10^{-3}$ the strongest members (w = 5, 8)
    project only ≈ 20 updated failures in 20,000 shots, while the weakest
    (w = 11) already has a blind LER of ≈ 0.5; at higher $p$ more members
    saturate. The two criteria are kept; only the $p$ grid (extended downward)
    and the shot counts (per code) change. Under the registered rule as written,
    T10 would leave the family ($m = 13$); §17 lets you choose.

    **Statistics.** Holm **T10**: the weighted least-squares slope of
    $\ln(k_\text{blind}/k_\text{updated})$ on $w$, with variance
    $(k_{10} + k_{01})/(k_\text{blind}\,k_\text{updated})$ from the paired
    discordant counts; one-sided $t$ test, 7 degrees of freedom; H1: slope < 0.
    The unflagged comparisons (exact conditional rate ratios, unpaired) are
    exploratory.
    """)
    return


@app.cell
def _():
    E09_MASTER_SEED = 2026110901
    E09_PILOT_SEED = 2026110902
    E09_PS = [5e-4, 7e-4, 1e-3, 2e-3, 3e-3]
    E09_REGISTERED_PS = [1e-3, 2e-3, 3e-3]
    return E09_MASTER_SEED, E09_PILOT_SEED, E09_PS, E09_REGISTERED_PS


@app.function
def distance_upper_bound(code, trials=300, seed=0, pairs=True):
    """
    Randomised information-set search for low-weight logicals: permute the
    columns of a basis of ker(H_X) (resp. ker(H_Z)), row-reduce, and keep the
    lightest row (and, with `pairs`, sum of two rows) that is a NONTRIVIAL
    logical (anticommutes with some logical of the other type). Returns
    min(d_X, d_Z) found -- an upper bound on the distance.
    """
    rng = np.random.default_rng(seed)
    best = code.n
    for H, other in ((code.hx, code.lx), (code.hz, code.lz)):
        G = gf2_nullspace(H)
        O = np.asarray(other, np.uint8)
        for _ in range(trials):
            perm = rng.permutation(code.n)
            R, piv = gf2_rref(G[:, perm])
            R = R[:len(piv)]
            rows = np.zeros_like(R)
            rows[:, perm] = R
            cands = [rows]
            if pairs and len(rows) > 1:
                i, j = np.triu_indices(len(rows), 1)
                cands.append(rows[i] ^ rows[j])
            for C in cands:
                nontriv = ((C.astype(np.int64) @ O.T.astype(np.int64)) % 2).any(1)
                if nontriv.any():
                    best = min(best, int(C[nontriv].sum(1).min()))
    return best


@app.function
def w_family_row(key, rounds=12, p=1e-3, trials=200):
    """One member of the w-family: n, k, a distance upper bound, and every flag
    law measured on its circuit beside the law's prediction."""
    c = code_from_key(key)
    r = flag_structure_report(c, rounds, p)
    if r["w"] != W_FAMILY_PRESETS[key]["w"]:
        raise ValueError(f"{key}: measured check weight {r['w']} is not the preset's "
                         f"{W_FAMILY_PRESETS[key]['w']}")
    return dict(r, code=key, n=int(c.n), k=int(c.k),
                d_upper=int(distance_upper_bound(c, trials=trials, seed=1)))


@app.function
def w_family_table(rounds=12, p=1e-3, trials=200, on_progress=None):
    """Zero-compute: one w_family_row per member, in order of w."""
    rows = []
    for key in W_FAMILY_KEYS:
        rows.append(w_family_row(key, rounds, p, trials))
        if on_progress is not None:
            on_progress()
    return rows


@app.function
def w_family_table_md(rows):
    """Measured value / law prediction for L1, L4, L5, L6 and L7, per member."""
    T, p = rows[0]["rounds"], rows[0]["p"]
    head = ("| code | w | n | k | d ≤ | group size (L1) | S × 10³ (L4) | max posterior (L5) | "
            f"flag rate, T = {T} (L6) | faults added (L7) | told apart by flags |",
            "|:--|--:|--:|--:|--:|:--|:--|:--|:--|:--|--:|")
    lines = list(head)
    for r in rows:
        gs = ", ".join(map(str, r["group_sizes"]))
        l7 = r["added_mechanisms"] - r["separated_by_flags"]
        lines.append(
            f"| {r['code']} | {r['w']} | {r['n']} | {r['k']} | {r['d_upper']} | "
            f"{gs} / {r['L1_group_size']} | "
            f"{r['S_mean'] * 1e3:.4f} / {r['L4_S'] * 1e3:.4f} | "
            f"{r['max_posterior']:.4f} / {r['L5_max_posterior']:.4f} | "
            f"{r['predicted_flag_rate']:.3f} / {r['L6_flag_rate']:.3f} | "
            f"{r['added_mechanisms']}"
            + (f" − {r['separated_by_flags']} = {l7}" if r["separated_by_flags"] else "")
            + f" / {r['L7_added_mechanisms']} | {r['separated_by_flags']} |")
    return ("\n".join(lines) + f"\n\nEach cell is **measured / law** at $p$ = {p:g}. "
            "Measured values come from the member's own DEM: group size = faults per flag, "
            "S = summed prior of one flag's group, max posterior = the largest P(fault | flag "
            "fired), flag rate = 1 − Π(1 − S) over every flag. *d ≤* is an upper bound from "
            f"random information sets. *Told apart by flags*: pairs of faults that the "
            "unflagged circuit cannot distinguish (Stim merges them into one column) but that "
            "fire different flags; they are subtracted before comparing with L7 (§11).")


@app.cell
def _(E09_PILOT_SEED):
    def e09_pilot(codes, ps, shots=1000, master=E09_PILOT_SEED, on_progress=None):
        """Pilot rows: blind and updated failures per (code, p) at `shots`."""
        rows = []
        for c in codes:
            for p in ps:
                seed = derive_seed(master, c, float(p), 12, 0)
                r = four_arm_run(c, 12, p, shots, seed, arms=("blind", "updated"), checkpoint_dir=False,
                                 keep_per_shot=False)
                rows.append(dict(code=c, p=float(p), shots=shots, seed=seed,
                                 blind_failures=r["arms"]["blind"]["failures"],
                                 updated_failures=r["arms"]["updated"]["failures"],
                                 blind_ler=r["arms"]["blind"]["ler"],
                                 projected_updated_20k=r["arms"]["updated"]["failures"] * 20000 / shots,
                                 time_p50_us=r["arms"]["updated"]["time_p50_us"]))
                if on_progress:
                    on_progress()
        return rows

    return (e09_pilot,)


@app.function
def e09_freeze(pilot_rows, target=30, main_shots=20000, max_blind_ler=0.35):
    """The registered freeze rule: smallest p where every code qualifies, else None."""
    codes = sorted({r["code"] for r in pilot_rows})
    for p in sorted({r["p"] for r in pilot_rows}):
        rs = [r for r in pilot_rows if r["p"] == p]
        if len(rs) != len(codes):
            continue
        if all(r["updated_failures"] * main_shots / r["shots"] >= target
               and r["blind_ler"] <= max_blind_ler for r in rs):
            return p
    return None


@app.function
def e09_freeze_revised(pilot_rows, max_blind_ler=0.35):
    """The revised rule: the LARGEST pilot p at which every code's blind LER is
    at most the cap, else None."""
    codes = sorted({r["code"] for r in pilot_rows})
    ok = [p for p in sorted({r["p"] for r in pilot_rows})
          if len([r for r in pilot_rows if r["p"] == p]) == len(codes)
          and all(r["blind_ler"] <= max_blind_ler for r in pilot_rows if r["p"] == p)]
    return max(ok) if ok else None


@app.function
def e09_main_shots(pilot_rows, p, target=30, floor=20_000, cap=200_000, step=10_000, margin=1.2):
    """Per-code main shots at the frozen p: enough for `target` updated failures."""
    out = {}
    for r in pilot_rows:
        if abs(r["p"] - p) > 1e-15:
            continue
        rate = r["updated_failures"] / r["shots"]
        need = cap if rate == 0 else math.ceil(margin * target / rate)
        out[r["code"]] = int(min(cap, max(floor, -(-need // step) * step)))
    return out


@app.function
def e09_pilot_md(pilot_rows, frozen=None, shots=None, target=30, max_blind_ler=0.35):
    """Every pilot cell: blind LER (✗ above the cap) and the shots the main run
    would need for `target` updated failures; the frozen column is marked."""
    ps = sorted({r["p"] for r in pilot_rows})
    codes = list(dict.fromkeys(r["code"] for r in pilot_rows))
    cell = {(r["code"], r["p"]): r for r in pilot_rows}
    head = ("| code | " + " | ".join(f"p = {p:g}" + (" (frozen)" if frozen is not None and
                                                       abs(p - frozen) < 1e-15 else "")
                                     for p in ps) + (" | main shots |" if shots else " |"))
    lines = [head, "|:--|" + ":--|" * len(ps) + ("--:|" if shots else "")]
    for c in codes:
        parts = []
        for p in ps:
            r = cell.get((c, p))
            if r is None:
                parts.append("—")
                continue
            k = r["updated_failures"]
            need = f"{math.ceil(target * r['shots'] / k):,}" if k else "> pilot"
            bad = r["blind_ler"] > max_blind_ler
            parts.append(f"{r['blind_ler']:.3f}{' ✗' if bad else ''} · {k} → {need}")
        lines.append(f"| {c} | " + " | ".join(parts)
                     + (f" | {shots.get(c, 0):,} |" if shots else " |"))
    return ("\n".join(lines) + f"\n\nEach cell: pilot blind LER (✗ above {max_blind_ler}) · "
            f"pilot updated failures → shots needed for {target} updated failures.")


@app.function
def e09_slope(results):
    """Holm T10: WLS slope of ln(k_blind / k_updated) on w (one-sided, H1 < 0)."""
    xs, ys, vs, used = [], [], [], []
    for r in results:
        a = r["arms"]
        if "blind" not in a or "updated" not in a:
            continue
        kb, ku = a["blind"]["failures"], a["updated"]["failures"]
        pr = r["paired"].get("blind|updated")
        if kb == 0 or ku == 0 or pr is None:
            continue
        w = W_FAMILY_PRESETS.get(r["code"], {}).get("w")
        if w is None:
            continue
        xs.append(w)
        ys.append(math.log(kb / ku))
        vs.append(max(pr["a_only"] + pr["b_only"], 1) / (kb * ku))
        used.append(r["code"])
    if len(xs) < 3:
        return dict(slope=None, note="fewer than three usable codes", codes=used)
    out = wls_slope(xs, ys, vs, alternative="less")
    return dict(out, codes=used, x=xs, y=ys, var=vs)


@app.function
def plot_e09(results, slope=None):
    rows = sorted([r for r in results if r["code"] in W_FAMILY_PRESETS],
                  key=lambda r: W_FAMILY_PRESETS[r["code"]]["w"])
    ws = [W_FAMILY_PRESETS[r["code"]]["w"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    for arm in FOUR_ARMS:
        ys = [r["arms"][arm]["ler"] if arm in r["arms"] else np.nan for r in rows]
        a.plot(ws, ys, ARM_STYLE[arm][1] + "-", color=ARM_STYLE[arm][0], label=ARM_STYLE[arm][2])
    a.set_yscale("log")
    a.set_xlabel("X-check weight w")
    a.set_ylabel("logical error rate per shot")
    a.set_title("LER by arm across the w-family")
    a.legend(fontsize=8)
    if slope and slope.get("slope") is not None:
        b.errorbar(slope["x"], slope["y"], yerr=np.sqrt(slope["var"]), fmt="o", color="#1f6fb4",
                   capsize=3)
        xx = np.array([min(slope["x"]), max(slope["x"])])
        yb = np.average(slope["y"], weights=1 / np.asarray(slope["var"]))
        xb = np.average(slope["x"], weights=1 / np.asarray(slope["var"]))
        b.plot(xx, yb + slope["slope"] * (xx - xb), "--", color="k",
               label=f"WLS slope {slope['slope']:.3f} ± {slope['se']:.3f} (p = {slope['p']:.2g})")
        b.legend(fontsize=8)
    b.axhline(0, color="grey", lw=0.6)
    b.set_xlabel("X-check weight w")
    b.set_ylabel("ln(blind failures / updated failures)")
    b.set_title("Holm T10: does the updating gain shrink with w?")
    for ax in (a, b):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _():
    def _e09_distance_bound():
        c = bb_from_preset("[[72, 12, 6]]", BB_PRESETS)
        d = distance_upper_bound(c, trials=150, seed=2)
        assert d == 6, f"information-set search found {d}, the known distance is 6"
        assert distance_upper_bound(gb_from_preset("[[48, 6, 8]]", GB_PRESETS), trials=150, seed=2) >= 8

    def _e09_freeze_rule():
        rows = [dict(code=c, p=p, shots=1000, updated_failures=u, blind_ler=b)
                for c, p, u, b in (("a", 1e-3, 1, 0.01), ("b", 1e-3, 2, 0.02),
                                   ("a", 2e-3, 2, 0.05), ("b", 2e-3, 3, 0.06),
                                   ("a", 3e-3, 9, 0.2), ("b", 3e-3, 9, 0.5))]
        assert e09_freeze(rows) == 2e-3
        assert e09_freeze_revised(rows) == 2e-3          # 3e-3 has a blind LER above 0.35
        sh = e09_main_shots(rows, 2e-3)
        assert sh == {"a": 20_000, "b": 20_000}
        rows[3]["updated_failures"] = 1
        assert e09_freeze(rows) is None
        assert e09_main_shots(rows, 2e-3)["b"] == 40_000  # 1.2 x 30 / 0.001 = 36,000 -> 40,000
        rows[3]["updated_failures"] = 0
        assert e09_main_shots(rows, 2e-3)["b"] == 200_000
        assert e09_freeze_revised([dict(r, blind_ler=0.9) for r in rows]) is None

    def _e09_slope_math():
        res = []
        for w, kb, ku in ((4, 400, 100), (6, 300, 120), (8, 250, 150), (10, 200, 160), (12, 180, 170)):
            key = next(k for k in W_FAMILY_KEYS if W_FAMILY_PRESETS[k]["w"] == w)
            res.append(dict(code=key, arms=dict(blind=dict(failures=kb), updated=dict(failures=ku)),
                            paired={"blind|updated": dict(a_only=kb - ku + 10, b_only=10)}))
        s = e09_slope(res)
        assert s["slope"] < 0 and s["p"] < 0.05 and s["df"] == 3

    def _e09_table_rows():
        """A table row carries both the code's k and the measured w, and renders."""
        r = w_family_row(W_FAMILY_KEYS[7], rounds=2, trials=5)
        assert r["w"] == 11 and r["k"] == 4 and r["n"] == 48 and r["code"] == W_FAMILY_KEYS[7]
        md = w_family_table_md([w_family_row(W_FAMILY_KEYS[0], rounds=2, trials=5), r])
        assert md.count("\n|") >= 3 and " = " in md      # w = 11 shows its subtraction

    tests_e09 = run_checks([
        ("w-family table rows build and render (w = 4, 11)", _e09_table_rows),
        ("information-set search finds d = 6 on [[72,12,6]]", _e09_distance_bound),
        ("freeze rules (registered and revised) and per-code main shots", _e09_freeze_rule),
        ("T10 slope: WLS on ln(blind/updated), one-sided", _e09_slope_math),
    ])
    render_checks("Experiment 9", tests_e09)
    return (tests_e09,)


@app.cell
def _():
    e09_table_btn = mo.ui.run_button(label="Compute the w-family table (no sampling, ~1 min)")
    e09_table_btn
    return (e09_table_btn,)


@app.cell
def _(e09_table_btn):
    mo.stop(not e09_table_btn.value)
    with mo.status.progress_bar(total=len(W_FAMILY_KEYS), title="w-family table") as _bar:
        e09_table = w_family_table(on_progress=_bar.update)
    mo.md("#### The w-family: flag laws measured on every member\n\n" + w_family_table_md(e09_table))
    return


@app.cell
def _(E09_PILOT_SEED):
    e09_pilot_ui = mo.ui.dictionary(dict(
        shots=mo.ui.number(start=100, stop=20_000, step=100, value=2000, label="pilot shots"),
        master_seed=mo.ui.number(value=E09_PILOT_SEED, label="pilot master seed"),
    ))
    e09_pilot_btn = mo.ui.run_button(label="Run the pilot and freeze p")
    mo.vstack([mo.md("### Experiment 9 — pilot"), e09_pilot_ui, e09_pilot_btn])
    return e09_pilot_btn, e09_pilot_ui


@app.cell
def _(
    E09_PS,
    E09_REGISTERED_PS,
    core_ready,
    e09_pilot,
    e09_pilot_btn,
    e09_pilot_ui,
    tests_e09,
):
    e09_pilot_run = None
    if e09_pilot_btn.value and core_ready and all_pass(tests_e09):
        _sh, _m = int(e09_pilot_ui.value["shots"]), int(e09_pilot_ui.value["master_seed"])
        with mo.status.progress_bar(total=len(W_FAMILY_KEYS) * len(E09_PS), title="E09 pilot") as _bar:
            _rows = e09_pilot(W_FAMILY_KEYS, E09_PS, shots=_sh, master=_m, on_progress=_bar.update)
        _reg = e09_freeze([r for r in _rows if r["p"] in E09_REGISTERED_PS])
        _frozen = e09_freeze_revised(_rows)
        _shots = e09_main_shots(_rows, _frozen) if _frozen else {}
        _stamp = start_run("E09-pilot", "E09")
        _path = result_file("E09P", code=W_FAMILY_KEYS, rounds=12, p=E09_PS, shots=_sh, seed=_m,
                            convention="z-only", extra="pilot")
        write_result_file(_path, dict(
            config=dict(codes=W_FAMILY_KEYS, rounds=12, ps=E09_PS, shots=_sh, master_seed=_m,
                        detector_convention="z-only", decoders={"OSD-0": decoder_spec_full("OSD-0")},
                        rule=("largest p with every blind LER <= 0.35; per-code main shots for >= 30 "
                              "updated failures (x1.2, 20k..200k in 10k steps)"),
                        registered_rule=("smallest p in {1e-3, 2e-3, 3e-3} with >= 30 projected "
                                         "updated failures at 20k and blind LER <= 0.35")),
            timing=dict(workers_requested=1, workers_effective=1, backends=["sequential"],
                        resumed_chunks=0, total_chunks=0),
            samples={}, rows=_rows, frozen_p=_frozen, registered_frozen_p=_reg, main_shots=_shots),
            _stamp, "complete")
        e09_pilot_run = dict(path=_path, rows=_rows, frozen_p=_frozen, registered_frozen_p=_reg,
                             main_shots=_shots)
    _pf = sorted(glob.glob(os.path.join(RESULTS_DIR, "E09P_*.json")))
    if e09_pilot_run is not None:
        _pil = e09_pilot_run
    elif _pf:
        _d = read_result_file(_pf[-1])
        _rows = _d.get("rows", [])
        _fr = e09_freeze_revised(_rows)
        _pil = dict(path=_pf[-1], rows=_rows, frozen_p=_fr,
                    registered_frozen_p=e09_freeze([r for r in _rows if r["p"] in E09_REGISTERED_PS]),
                    main_shots=(_d.get("main_shots") or (e09_main_shots(_rows, _fr) if _fr else {})))
    else:
        _pil = None
    e09_frozen = _pil["frozen_p"] if _pil else None
    e09_shots_by_code = _pil["main_shots"] if _pil else {}
    if _pil is None:
        _msg = "*No pilot yet: run the pilot.*"
    else:
        _grid = sorted({r["p"] for r in _pil["rows"]})
        _msg = (f"**Frozen p = {e09_frozen:g}** (revised rule, from `{os.path.basename(_pil['path'])}`)."
                if e09_frozen else
                "**No p qualifies even under the revised rule** — some code's blind LER is above 0.35 "
                f"at every pilot p ({', '.join(f'{p:g}' for p in _grid)}). "
                + ("This pilot predates the extended grid: run the pilot again." if min(_grid) > 5e-4
                   else "Extend `E09_PS` downward."))
        _msg += (f" The registered rule would freeze p = {_pil['registered_frozen_p']:g}."
                 if _pil["registered_frozen_p"] else
                 " The registered rule finds no p (T10 would leave the family; see §17).")
        _msg += "\n\n" + e09_pilot_md(_pil["rows"], e09_frozen, e09_shots_by_code or None)
    mo.md(_msg)
    return e09_frozen, e09_shots_by_code


@app.cell
def _(E09_MASTER_SEED, e09_frozen, e09_shots_by_code):
    E09_DESIGN = {f"w{W_FAMILY_PRESETS[k]['w']}": dict(code=k, rounds=12, p=e09_frozen or 1e-3,
                                                       shots=int(e09_shots_by_code.get(k, 20_000)),
                                                       purpose="Holm T10")
                  for k in W_FAMILY_KEYS}
    e09_ui = grid_design_ui(E09_DESIGN, E09_MASTER_SEED)
    e09_run_btn = mo.ui.run_button(label="Run Experiment 9 (main)")
    mo.vstack([mo.md("### Experiment 9 — main run at the frozen p\n\n" + design_table_md(E09_DESIGN)),
               e09_ui, e09_run_btn])
    return E09_DESIGN, e09_run_btn, e09_ui


@app.cell
def _(
    E09_DESIGN,
    core_ready,
    e09_frozen,
    e09_run_btn,
    e09_shots_by_code,
    e09_ui,
    tests_e09,
):
    e09_run, _out = grid_run_cell("E09", "E09-w-family", e09_ui.value, E09_DESIGN,
                                  core_ready and all_pass(tests_e09) and e09_frozen is not None,
                                  e09_run_btn.value, "Experiment 9",
                                  extra_config=dict(frozen_p=e09_frozen, main_shots=e09_shots_by_code))
    _out
    return (e09_run,)


@app.cell
def _(e09_run):
    _ = e09_run
    e09_file, _files = grid_picker("E09")
    e09_load_btn = mo.ui.run_button(label="Load")
    e09_reexport_btn = mo.ui.run_button(label="Re-export loaded run")
    mo.hstack([e09_file, e09_load_btn, e09_reexport_btn]) if _files else mo.md("*No saved runs yet.*")
    return e09_file, e09_load_btn, e09_reexport_btn


@app.cell
def _(e09_file, e09_load_btn):
    e09_loaded = None
    if e09_load_btn.value and e09_file.value:
        _d, _r, _p = load_grid(e09_file.value, "E09-w-family")
        e09_loaded = dict(path=e09_file.value, doc=_d, results=_r, per_shot=_p, fresh=False)
    return (e09_loaded,)


@app.cell
def _(e09_loaded, e09_reexport_btn, e09_run):
    e09_data = e09_run if e09_run is not None else e09_loaded
    mo.stop(e09_data is None, mo.md("*Run Experiment 9 or load a saved run.*"))
    _res = e09_data["results"]
    _sl = e09_slope(_res)
    _fig = plot_e09(_res, _sl)
    _p34 = ["| code | w | sighted / unflagged | blind / unflagged | updated / unflagged |",
            "|:--|--:|:--|:--|:--|"]
    for _r in sorted(_res, key=lambda r: W_FAMILY_PRESETS.get(r["code"], {}).get("w", 0)):
        _u = _r["unpaired"]
        _f = lambda k: (f"{_u[k]['ratio']:.3g} [{_u[k]['lo']:.3g}, {_u[k]['hi']:.3g}]" if k in _u else "—")
        _p34.append(f"| {_r['code']} | {W_FAMILY_PRESETS.get(_r['code'], {}).get('w')} | "
                    f"{_f('sighted|unflagged')} | {_f('blind|unflagged')} | {_f('updated|unflagged')} |")
    _exp = maybe_export(e09_data, "Experiment 9 — check-weight family", {"w_family": _fig},
                        {"arms": grid_rows(_res), "comparisons": comparison_rows(_res),
                         "t10": [{k: v for k, v in _sl.items() if not isinstance(v, list)}]},
                        reexport=e09_reexport_btn.value)
    mo.vstack([run_status_banner(e09_data["doc"], e09_data["path"]),
               mo.md(f"**Holm T10:** slope {_sl['slope']:.4f} ± {_sl['se']:.4f} per unit w, "
                     f"t = {_sl['t']:.2f} on {_sl['df']} df, one-sided p = {_sl['p']:.3g}"
                     if _sl.get("slope") is not None else f"**T10 not computable:** {_sl['note']}"),
               _fig, mo.md("#### P3/P4 (exploratory, unpaired)\n\n" + "\n".join(_p34)),
               mo.accordion({"every arm": mo.md(grid_table_md(_res)),
                             "every comparison": mo.md(comparison_table_md(_res))}), _exp])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiment 10 — prior updating with BP+LSD

    **Question.** Is the prior-updating gain a property of the flag posterior or
    of OSD? Repeat the paired arms with BP+LSD-0 (localised statistics decoding,
    `LSD_CS`, 30 BP iterations) in place of BP+OSD-0.

    **Design.** blind, sighted and updated with `LSD-0` on Experiment 8's A1
    configuration ([[72,12,6]], $p = 10^{-3}$, 50,000 shots) and on
    [[144,12,12]] at $10^{-3}$ with 50,000 shots (descriptive). The seeds come
    from Experiment 8's master seed, so these arms decode **the same shots** as
    A1/A3 with OSD-0 — the decoder is the only difference.

    **Statistics.** Holm **T12**: updated vs sighted, exact McNemar on the
    discordant shots. Times as in Experiment 8 (update included).
    """)
    return


@app.cell
def _():
    E10_DESIGN = {
        "B1": dict(code="[[72, 12, 6]]", rounds=12, p=1e-3, shots=50_000, decoder="LSD-0",
                   arms=["blind", "sighted", "updated"], purpose="Holm T12"),
        "B2": dict(code="[[144, 12, 12]]", rounds=12, p=1e-3, shots=50_000, decoder="LSD-0",
                   arms=["blind", "sighted", "updated"], purpose="descriptive"),
    }
    return (E10_DESIGN,)


@app.cell
def _():
    def _e10_lsd_spec():
        r = four_arm_run(code_from_key("[[72, 12, 6]]"), 2, 2e-3, 30, 4, decoder="LSD-0",
                         arms=("blind", "updated"), checkpoint_dir=False, keep_per_shot=False)
        assert r["decoder_spec"]["kind"] == "lsd" and r["decoder_spec"]["method"] == "LSD_CS"
        assert r["decoder_spec"]["max_iter"] == STRONG_MAX_ITER
        assert r["arms"]["updated"]["time_update_p50_us"] > 0

    tests_e10 = run_checks([("the LSD-0 arms run through the shared runner and spec", _e10_lsd_spec)])
    render_checks("Experiment 10", tests_e10)
    return (tests_e10,)


@app.cell
def _(E08_MASTER_SEED, E10_DESIGN):
    e10_ui = grid_design_ui(E10_DESIGN, E08_MASTER_SEED)
    e10_run_btn = mo.ui.run_button(label="Run Experiment 10")
    mo.vstack([mo.md("### Experiment 10 — controls (master seed = Experiment 8's)\n\n"
                     + design_table_md(E10_DESIGN)), e10_ui, e10_run_btn])
    return e10_run_btn, e10_ui


@app.cell
def _(E10_DESIGN, core_ready, e10_run_btn, e10_ui, tests_e10):
    e10_run, _out = grid_run_cell("E10", "E10-lsd", e10_ui.value, E10_DESIGN,
                                  core_ready and all_pass(tests_e10), e10_run_btn.value, "Experiment 10")
    _out
    return (e10_run,)


@app.cell
def _(e10_run):
    _ = e10_run
    e10_file, _files = grid_picker("E10")
    e10_load_btn = mo.ui.run_button(label="Load")
    e10_reexport_btn = mo.ui.run_button(label="Re-export loaded run")
    mo.hstack([e10_file, e10_load_btn, e10_reexport_btn]) if _files else mo.md("*No saved runs yet.*")
    return e10_file, e10_load_btn, e10_reexport_btn


@app.cell
def _(e10_file, e10_load_btn):
    e10_loaded = None
    if e10_load_btn.value and e10_file.value:
        _d, _r, _p = load_grid(e10_file.value, "E10-lsd")
        e10_loaded = dict(path=e10_file.value, doc=_d, results=_r, per_shot=_p, fresh=False)
    return (e10_loaded,)


@app.cell
def _(e10_loaded, e10_reexport_btn, e10_run):
    e10_data = e10_run if e10_run is not None else e10_loaded
    mo.stop(e10_data is None, mo.md("*Run Experiment 10 or load a saved run.*"))
    _res = e10_data["results"]
    _figs = {f"arms_{r['config_id']}": plot_prior_update(r) for r in _res}
    _exp = maybe_export(e10_data, "Experiment 10 — prior updating with BP+LSD-0", _figs,
                        {"arms": grid_rows(_res), "comparisons": comparison_rows(_res)},
                        reexport=e10_reexport_btn.value)
    mo.vstack([run_status_banner(e10_data["doc"], e10_data["path"]), mo.md(grid_table_md(_res)),
               mo.md(comparison_table_md(_res)), *_figs.values(), _exp])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiment 11 — prior updating with tuned Relay-BP

    **Question.** Relay-BP (Müller et al., 2025) is a BP variant with memory
    terms that reaches OSD-like accuracy without OSD. Does the flag posterior help
    it as well — and is it then competitive in time?

    **Decoder.** This notebook carries its own Relay-BP (§7): a numba port of
    the reference implementation in IBM's `relay-bp` package, so nothing extra
    needs installing. A self-test checks the port against the package when the
    package is present (identical corrections on the deterministic first leg;
    statistically equal failure counts on the full relay). Because a prior
    update only recomputes the log-prior ratios, the updated arm's timing
    reflects the method rather than a library's rebuild cost.

    **Design.**
    1. **Tuning** on a separate seed: [[72,12,6]], $p = 2\times10^{-3}$, 4,000
       shots per grid point (blind arm), over $\gamma_0$, the number of relay
       sets and the stop rule. The best point (fewest failures; ties → lower
       median time) is **frozen** in a result file before confirmatory sampling.
    2. **Timing pilot**: 1,000 shots per planned configuration; the projected
       cost is recorded with a **stop rule** — above 20 core-hours the
       [[144,12,12]] descriptive run is cut first.
    3. **Confirmatory**: A2's configuration ([[72,12,6]], $2\times10^{-3}$,
       20,000 shots). **Descriptive**: A1's configuration (50,000) and
       [[144,12,12]] at $10^{-3}$ (50,000). Seeds come from Experiment 8's master
       seed (same shots as the OSD-0 arms).

    **Statistics.** Holm **T13**: updated vs blind, exact McNemar. Tuning is
    selection on a separate sample (fewest failures), so it cannot bias the
    confirmatory test.
    """)
    return


@app.cell
def _():
    E11_TUNE_SEED = 2026111101
    E11_TUNING_GRID = [dict(gamma0=g, num_sets=n, stop_nconv=s, pre_iter=80, set_max_iter=60,
                            gamma_dist_interval=(-0.24, 0.66))
                       for g in (0.35, 0.65) for n in (20, 100) for s in (1, 5)]
    E11_STOP_CORE_HOURS = 20.0
    return E11_STOP_CORE_HOURS, E11_TUNE_SEED, E11_TUNING_GRID


@app.cell
def _(E11_TUNE_SEED):
    def e11_tune(grid, shots=4000, seed=E11_TUNE_SEED, code="[[72, 12, 6]]", p=2e-3, on_progress=None):
        """Blind-arm failures and median time for every tuning point; best = fewest failures, then time."""
        rows = []
        for i, params in enumerate(grid):
            spec = dict(kind="relay", **params)
            r = four_arm_run(code, 12, p, shots, seed, decoder="relay", decoder_spec=spec,
                             arms=("blind",), checkpoint_dir=False, keep_per_shot=False)
            m = r["arms"]["blind"]
            rows.append(dict(point=i, **{k: (list(v) if isinstance(v, tuple) else v)
                                         for k, v in params.items()},
                             failures=m["failures"], ler=m["ler"], time_p50_us=m["time_p50_us"]))
            if on_progress:
                on_progress()
        best = min(rows, key=lambda r: (r["failures"], r["time_p50_us"]))
        return rows, best

    return (e11_tune,)


@app.function
def e11_design(frozen):
    """The confirmatory and descriptive configurations with the frozen Relay-BP spec."""
    spec = dict(kind="relay", **{k: frozen[k] for k in ("gamma0", "num_sets", "stop_nconv",
                                                        "pre_iter", "set_max_iter",
                                                        "gamma_dist_interval")})
    spec["gamma_dist_interval"] = tuple(spec["gamma_dist_interval"])
    base = dict(rounds=12, decoder="relay", decoder_spec=spec, arms=["blind", "sighted", "updated"])
    return {"R1": dict(base, code="[[72, 12, 6]]", p=2e-3, shots=20_000, purpose="Holm T13"),
            "R2": dict(base, code="[[72, 12, 6]]", p=1e-3, shots=50_000, purpose="descriptive (A1)"),
            "R3": dict(base, code="[[144, 12, 12]]", p=1e-3, shots=50_000,
                       purpose="descriptive; cut first by the stop rule")}


@app.cell
def _(E08_MASTER_SEED, E11_STOP_CORE_HOURS):
    def e11_pilot(design, shots=1000, master=None):
        """Projected core-hours per configuration from a 1,000-shot timing pilot."""
        master = master if master is not None else E08_MASTER_SEED
        rows = []
        for cid, c in design.items():
            seed = derive_seed(master, c["code"], c["p"], c["rounds"], 0)
            r = four_arm_run(c["code"], c["rounds"], c["p"], shots, seed, decoder="relay",
                             decoder_spec=c["decoder_spec"], arms=c["arms"], checkpoint_dir=False,
                             keep_per_shot=False)
            us = sum(m["time_mean_us"] for m in r["arms"].values())
            rows.append(dict(config=cid, us_per_shot_all_arms=us,
                             projected_core_hours=core_hours(c["shots"], us)))
        total = sum(r["projected_core_hours"] for r in rows)
        cut = []
        if total > E11_STOP_CORE_HOURS and "R3" in design:
            cut.append("R3")
        return rows, total, cut

    return (e11_pilot,)


@app.cell
def _(E11_TUNING_GRID, e11_tune):
    def _e11_tuning_grid_is_registered():
        assert len(E11_TUNING_GRID) == 8
        d = e11_design(dict(E11_TUNING_GRID[0], gamma_dist_interval=[-0.24, 0.66]))
        assert d["R1"]["shots"] == 20_000 and d["R1"]["decoder_spec"]["kind"] == "relay"
        assert isinstance(d["R1"]["decoder_spec"]["gamma_dist_interval"], tuple)

    def _e11_tiny_tune():
        small = [dict(g, num_sets=4, set_max_iter=20, pre_iter=20) for g in E11_TUNING_GRID[:2]]
        rows, best = e11_tune(small, shots=8, p=2e-3)
        assert best in rows and all("failures" in r for r in rows)

    tests_e11 = run_checks([
        ("tuning grid, frozen design and stop rule are registered", _e11_tuning_grid_is_registered),
        ("tuning runs end to end with the built-in Relay-BP", _e11_tiny_tune),
    ])
    render_checks("Experiment 11", tests_e11)
    return (tests_e11,)


@app.cell
def _(E11_TUNE_SEED):
    e11_tune_ui = mo.ui.dictionary(dict(
        shots=mo.ui.number(start=100, stop=20_000, step=100, value=4000, label="shots per grid point"),
        seed=mo.ui.number(value=E11_TUNE_SEED, label="tuning seed (separate from confirmatory)"),
    ))
    e11_tune_btn = mo.ui.run_button(label="Tune and freeze Relay-BP")
    mo.vstack([mo.md("### Experiment 11 — tuning"), e11_tune_ui, e11_tune_btn])
    return e11_tune_btn, e11_tune_ui


@app.cell
def _(
    E11_TUNING_GRID,
    core_ready,
    e11_tune,
    e11_tune_btn,
    e11_tune_ui,
    tests_e11,
):
    e11_tuned = None
    if e11_tune_btn.value and core_ready and all_pass(tests_e11):
        _sh, _sd = int(e11_tune_ui.value["shots"]), int(e11_tune_ui.value["seed"])
        with mo.status.progress_bar(total=len(E11_TUNING_GRID), title="Relay-BP tuning") as _bar:
            _rows, _best = e11_tune(E11_TUNING_GRID, shots=_sh, seed=_sd, on_progress=_bar.update)
        _stamp = start_run("E11-tuning", "E11")
        _path = result_file("E11T", code="[[72, 12, 6]]", rounds=12, p=2e-3, shots=_sh, seed=_sd,
                            convention="z-only", extra="tuning")
        write_result_file(_path, dict(
            config=dict(code="[[72, 12, 6]]", rounds=12, p=2e-3, shots=_sh, seed=_sd,
                        detector_convention="z-only", grid=E11_TUNING_GRID,
                        decoders={"relay": dict(kind="relay", **{
                            k: (list(v) if isinstance(v, tuple) else v) for k, v in _best.items()
                            if k in ("gamma0", "num_sets", "stop_nconv", "pre_iter", "set_max_iter",
                                     "gamma_dist_interval")})}),
            timing=dict(workers_requested=1, workers_effective=1, backends=["sequential"],
                        resumed_chunks=0, total_chunks=0),
            samples={}, rows=_rows, frozen=_best), _stamp, "complete")
        e11_tuned = dict(path=_path, rows=_rows, frozen=_best)
    _tf = sorted(glob.glob(os.path.join(RESULTS_DIR, "E11T_*.json")))
    e11_frozen = (e11_tuned["frozen"] if e11_tuned else
                  (read_result_file(_tf[-1]).get("frozen") if _tf else None))
    mo.md(("**Frozen Relay-BP parameters:** " + ", ".join(f"{k} = {v}" for k, v in e11_frozen.items()
                                                          if k not in ("point", "failures", "ler",
                                                                       "time_p50_us")))
          if e11_frozen else "*Not tuned yet: run the tuning step first.*")
    return (e11_frozen,)


@app.cell
def _(E08_MASTER_SEED, e11_frozen):
    E11_DESIGN = e11_design(e11_frozen) if e11_frozen else {}
    e11_ui = grid_design_ui(E11_DESIGN, E08_MASTER_SEED) if E11_DESIGN else None
    e11_pilot_btn = mo.ui.run_button(label="Timing pilot (stop rule)")
    e11_run_btn = mo.ui.run_button(label="Run Experiment 11")
    mo.vstack([mo.md("### Experiment 11 — confirmatory and descriptive runs\n\n"
                     + design_table_md(E11_DESIGN)), e11_ui, mo.hstack([e11_pilot_btn, e11_run_btn])]) \
        if E11_DESIGN else mo.md("")
    return E11_DESIGN, e11_pilot_btn, e11_run_btn, e11_ui


@app.cell
def _(E11_DESIGN, E11_STOP_CORE_HOURS, e11_pilot, e11_pilot_btn):
    mo.stop(not (E11_DESIGN and e11_pilot_btn.value))
    _rows, _total, _cut = e11_pilot(E11_DESIGN)
    mo.md("| config | µs per shot (all arms) | projected core-hours |\n|:--|--:|--:|\n"
          + "\n".join(f"| {r['config']} | {r['us_per_shot_all_arms']:,.0f} | {r['projected_core_hours']:.2f} |"
                      for r in _rows)
          + f"\n\n**Total {_total:.1f} core-hours.** "
          + (f"Stop rule: above {E11_STOP_CORE_HOURS:g} — deselect {', '.join(_cut)} before running."
             if _cut else "Within the stop rule."))
    return


@app.cell
def _(E11_DESIGN, core_ready, e11_frozen, e11_run_btn, e11_ui, tests_e11):
    e11_run, _out = (grid_run_cell("E11", "E11-relay", e11_ui.value, E11_DESIGN,
                                   core_ready and all_pass(tests_e11),
                                   e11_run_btn.value, "Experiment 11",
                                   extra_config=dict(frozen=e11_frozen))
                     if E11_DESIGN else (None, mo.md("")))
    _out
    return (e11_run,)


@app.cell
def _(e11_run):
    _ = e11_run
    e11_file, _files = grid_picker("E11")
    e11_load_btn = mo.ui.run_button(label="Load")
    e11_reexport_btn = mo.ui.run_button(label="Re-export loaded run")
    mo.hstack([e11_file, e11_load_btn, e11_reexport_btn]) if _files else mo.md("*No saved runs yet.*")
    return e11_file, e11_load_btn, e11_reexport_btn


@app.cell
def _(e11_file, e11_load_btn):
    e11_loaded = None
    if e11_load_btn.value and e11_file.value:
        _d, _r, _p = load_grid(e11_file.value, "E11-relay")
        e11_loaded = dict(path=e11_file.value, doc=_d, results=_r, per_shot=_p, fresh=False)
    return (e11_loaded,)


@app.cell
def _(e11_loaded, e11_reexport_btn, e11_run):
    e11_data = e11_run if e11_run is not None else e11_loaded
    mo.stop(e11_data is None, mo.md("*Run Experiment 11 or load a saved run.*"))
    _res = e11_data["results"]
    _figs = {f"arms_{r['config_id']}": plot_prior_update(r) for r in _res}
    _exp = maybe_export(e11_data, "Experiment 11 — prior updating with Relay-BP", _figs,
                        {"arms": grid_rows(_res), "comparisons": comparison_rows(_res)},
                        reexport=e11_reexport_btn.value)
    mo.vstack([run_status_banner(e11_data["doc"], e11_data["path"]), mo.md(grid_table_md(_res)),
               mo.md(comparison_table_md(_res)), *_figs.values(), _exp])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiment 12 — the inference gap: Tanner-graph cycles, and min-sum vs sum-product

    **Question.** The sighted arm gives BP the flag rows as checks; the updated
    arm folds the same bits into the priors with Gallager's formula and removes
    the rows (§11). The two arms see identical information, and the sighted arm
    even keeps the within-group correlation that a prior vector drops — yet in
    earlier exploratory runs it failed far more often. If Experiment 8 confirms
    that, the gap is one of **inference**, and it has two candidate causes that
    this experiment separates:

    1. **Cycles.** A flag row is not a leaf of the Tanner graph: each of its $w$
       faults also sits in several syndrome detectors, so the row closes new
       short cycles through which flag information feeds back into the checks it
       came from. The prior update folds each flag in once and leaves the graph
       exactly as the DEM made it.
    2. **Min-sum.** The decoder of every other experiment is min-sum BP (scaling
       0.625), which replaces the exact tanh-product of the check-node update by
       a minimum over magnitudes. Our closed form is exact arithmetic; min-sum
       handed the same flag row computes an approximation of it even where the
       graph has no loops.

    **Design.**

    * *Part A — cycle census (no sampling).* For five codes at $T = 12$: the
      number of 4- and 6-cycles in the Tanner graph of the blind matrix (the
      graph the updated arm decodes) and of the sighted matrix (blind + flag
      rows), and how many of the sighted graph's cycles pass through a flag row.
    * *Part B — sum-product arms.* blind, sighted and updated with
      **BP+OSD-0 using exact sum-product BP** (`OSD-0-ps`; everything else as
      `OSD-0`) on Experiment 8's A1 configuration ([[72,12,6]], $p = 10^{-3}$,
      50,000 shots) and on [[144,12,12]] at $10^{-3}$ (50,000 shots). Seeds come
      from Experiment 8's master seed, so S1 decodes **the same shots** as A1;
      the sample hashes are checked before any cross-run comparison.

    **Reading the outcome.** If sum-product closes most of the sighted–updated
    gap, the min-sum approximation is the main cause; if the gap persists under
    sum-product, the cycles are (Part A says how many there are). Both can
    contribute.

    **Statistics.** Part A is exact counting (no uncertainty). Part B: Wilson
    intervals per arm; exact McNemar within the run (sighted vs updated, as in
    Experiment 8) and **across runs** on the same shots — for each arm, min-sum
    (Experiment 8's A1) against sum-product (S1), using the discordant shots.
    All of it is descriptive: Experiment 12 is not part of the registered Holm
    family (§17).
    """)
    return


@app.cell
def _():
    E12_DESIGN = {
        "S1": dict(code="[[72, 12, 6]]", rounds=12, p=1e-3, shots=50_000, decoder="OSD-0-ps",
                   arms=["blind", "sighted", "updated"],
                   purpose="sum-product on A1's shots (descriptive)"),
        "S2": dict(code="[[144, 12, 12]]", rounds=12, p=1e-3, shots=50_000, decoder="OSD-0-ps",
                   arms=["blind", "sighted", "updated"], purpose="descriptive"),
    }
    E12_CYCLE_CODES = ["[[72, 12, 6]]", "[[144, 12, 12]]", "[[288, 12, 18]]", "[[46, 2, 9]]",
                       "[[48, 6, 8]]"]
    return E12_CYCLE_CODES, E12_DESIGN


@app.function
def tanner_cycles(H):
    """
    Exact numbers of 4- and 6-cycles in the Tanner graph of H (rows = checks,
    columns = faults), from the row-overlap matrix M = H H^T:
      4-cycles = sum over row pairs of C(M_rs, 2);
      6-cycles = tr(M0^3)/6 - sum_c (deg_c - 2) sum_{r<s in c} M_rs + 2 sum_c C(deg_c, 3),
    where M0 is M without its diagonal (the column terms remove triples of
    rows whose "cycle" reuses a column shared by all three).
    """
    H = sp.csr_matrix(H, dtype=np.int64)
    H.data[:] = 1
    H.eliminate_zeros()
    M = (H @ H.T).tocsr()
    M0 = M.copy()
    M0.setdiag(0)
    M0.eliminate_zeros()
    c4 = int((M0.data * (M0.data - 1) // 2).sum() // 2)
    tr3 = int((M0 @ M0).multiply(M0).sum())
    Hc = H.tocsc()
    corr, trip = 0, 0
    for j in range(Hc.shape[1]):
        rows = Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]]
        d = len(rows)
        if d < 3:
            continue
        sub = M0[rows][:, rows].toarray()
        corr += (d - 2) * int(np.triu(sub, 1).sum())
        trip += comb(d, 3)
    if tr3 % 6:
        raise ArithmeticError("tr(M0^3) is not a multiple of 6")
    c6 = tr3 // 6 - corr + 2 * trip
    girth = 4 if c4 else (6 if c6 else None)
    return dict(rows=int(H.shape[0]), cols=int(H.shape[1]), cycles4=c4, cycles6=int(c6),
                girth=girth)


@app.function
def cycle_census(code_key, rounds=12, p=1e-3):
    """Cycles of the blind matrix (what the updated arm decodes) and of the
    sighted matrix (blind + flag rows); the difference passes through a flag row."""
    c = code_from_key(code_key)
    circ = build_memory_circuit(c, rounds, p, use_flags=True, x_detectors=False)
    H, _L, _pr = dem_to_matrices(circ.detector_error_model(decompose_errors=False))
    mask = flag_detector_mask(c, rounds, True, circ.num_detectors, x_detectors=False)
    blind, sighted = tanner_cycles(H[~mask]), tanner_cycles(H)
    return dict(code=code_key, rounds=int(rounds), p=float(p), flag_rows=int(mask.sum()),
                blind=blind, sighted=sighted,
                flag_cycles4=sighted["cycles4"] - blind["cycles4"],
                flag_cycles6=sighted["cycles6"] - blind["cycles6"])


@app.function
def cycle_census_md(rows):
    lines = ["| code | T | rows blind → sighted | 4-cycles blind | 4-cycles sighted (via flag rows) "
             "| 4-cycles per row: detector / flag | 6-cycles blind | 6-cycles sighted (via flag rows) "
             "| girth blind / sighted |",
             "|:--|--:|:--|--:|:--|:--|--:|:--|:--|"]
    g = lambda x: "≥ 8" if x is None else str(x)
    for r in rows:
        b, s = r["blind"], r["sighted"]
        lines.append(f"| {r['code']} | {r['rounds']} | {b['rows']:,} → {s['rows']:,} | "
                     f"{b['cycles4']:,} | {s['cycles4']:,} ({r['flag_cycles4']:,}) | "
                     f"{(2 * b['cycles4'] + r['flag_cycles4']) / b['rows']:,.0f} / "
                     f"{r['flag_cycles4'] / max(r['flag_rows'], 1):,.0f} | "
                     f"{b['cycles6']:,} | {s['cycles6']:,} ({r['flag_cycles6']:,}) | "
                     f"{g(b['girth'])} / {g(s['girth'])} |")
    return "\n".join(lines)


@app.function
def decoder_effect_rows(res_a, per_a, res_b, per_b, label_a="min-sum", label_b="sum-product"):
    """
    Same configuration, same shots, two decoders: for every shared arm the
    failures under each, the discordant split and the exact McNemar p-value.
    Configurations are matched on (code, T, p, shard, shots); configurations
    with different shot counts decoded different samples and are skipped. A
    matched pair whose sample hashes differ is refused.
    """
    def key(r):
        return (r["code"], int(r["rounds"]), round(float(r["p"]), 12), int(r.get("shard", 0)))
    idx_b = {key(r): r for r in res_b if not r.get("tag")}
    rows = []
    for ra in res_a:
        if ra.get("tag") or key(ra) not in idx_b:
            continue
        rb = idx_b[key(ra)]
        if int(ra["shots"]) != int(rb["shots"]):
            continue                      # different shot counts: different samples, not a pair
        pa, pb = (per_a or {}).get(ra.get("config_id")), (per_b or {}).get(rb.get("config_id"))
        if pa is None or pb is None:
            continue
        for arm in [a for a in FOUR_ARMS if a in ra["arms"] and a in rb["arms"]]:
            if ra["arms"][arm]["sample_sha256"] != rb["arms"][arm]["sample_sha256"]:
                raise ValueError(f"{ra['code']} p={ra['p']:g} {arm}: the two runs decoded "
                                 "different samples")
            fa, fb = np.asarray(pa[arm]["fail"], bool), np.asarray(pb[arm]["fail"], bool)
            x, y, pv = paired_exact(fa, fb)
            rows.append(dict(code=ra["code"], p=ra["p"], arm=arm, shots=int(len(fa)),
                             decoder_a=ra["decoder"], decoder_b=rb["decoder"],
                             failures_a=int(fa.sum()), failures_b=int(fb.sum()),
                             only_a=int(x), only_b=int(y), p_mcnemar=float(pv),
                             label_a=label_a, label_b=label_b))
    return rows


@app.function
def gap_rows(results):
    """sighted / updated failure ratio and its paired split, per configuration."""
    out = []
    for r in results:
        if not {"sighted", "updated"} <= set(r["arms"]):
            continue
        ks, ku = r["arms"]["sighted"]["failures"], r["arms"]["updated"]["failures"]
        c = r["paired"].get("sighted|updated", {})
        out.append(dict(code=r["code"], p=r["p"], decoder=r["decoder"], sighted=ks, updated=ku,
                        ratio=(ks / ku) if ku else (float("inf") if ks else None),
                        sighted_only=c.get("a_only"), updated_only=c.get("b_only"),
                        p_mcnemar=c.get("p")))
    return out


@app.cell
def _():
    def _brute_cycles(H):
        import itertools
        H = np.asarray(H, bool)
        R, C = H.shape
        c4 = sum(1 for r1, r2 in itertools.permutations(range(R), 2)
                 for c1, c2 in itertools.permutations(range(C), 2)
                 if H[r1, c1] and H[r2, c1] and H[r2, c2] and H[r1, c2]) // 4
        c6 = sum(1 for r1, r2, r3 in itertools.permutations(range(R), 3)
                 for c1, c2, c3 in itertools.permutations(range(C), 3)
                 if H[r1, c1] and H[r2, c1] and H[r2, c2] and H[r3, c2] and H[r3, c3] and H[r1, c3]) // 6
        return c4, c6

    def _e12_cycle_counts_match_brute_force():
        rng = np.random.default_rng(12)
        for trial in range(6):
            H = (rng.random((6, 7)) < (0.35 + 0.1 * trial)).astype(np.uint8)
            t = tanner_cycles(H)
            assert (t["cycles4"], t["cycles6"]) == _brute_cycles(H), (trial, t, _brute_cycles(H))
        # a 6-cycle with no 4-cycle, and a tree
        ring = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], np.uint8)
        assert tanner_cycles(ring) == dict(rows=3, cols=3, cycles4=0, cycles6=1, girth=6)
        assert tanner_cycles(np.array([[1, 1, 0], [0, 1, 1]], np.uint8))["girth"] is None

    def _e12_census_on_72():
        r = cycle_census("[[72, 12, 6]]", rounds=2)
        assert r["sighted"]["rows"] - r["blind"]["rows"] == r["flag_rows"] == 72
        assert r["flag_cycles4"] >= 0 and r["flag_cycles6"] >= 0

    def _e12_sum_product_arms_and_cross_run():
        a = four_arm_run(code_from_key("[[72, 12, 6]]"), 2, 2e-3, 60, 4, decoder="OSD-0",
                         arms=("blind", "sighted", "updated"), checkpoint_dir=False)
        b = four_arm_run(code_from_key("[[72, 12, 6]]"), 2, 2e-3, 60, 4, decoder="OSD-0-ps",
                         arms=("blind", "sighted", "updated"), checkpoint_dir=False)
        assert b["decoder_spec"]["bp_method"] == "ps" and "ms_scaling_factor" not in b["decoder_spec"]
        assert b["decoder_spec"]["max_iter"] == STRONG_MAX_ITER
        rows = decoder_effect_rows([dict(a, config_id="x")], {"x": a["per_shot"]},
                                   [dict(b, config_id="y")], {"y": b["per_shot"]})
        assert {r["arm"] for r in rows} == {"blind", "sighted", "updated"}
        assert all(r["only_a"] + r["failures_b"] == r["only_b"] + r["failures_a"] for r in rows)
        c = four_arm_run(code_from_key("[[72, 12, 6]]"), 2, 2e-3, 60, 5, decoder="OSD-0-ps",
                         arms=("blind",), checkpoint_dir=False)
        try:
            decoder_effect_rows([dict(a, config_id="x")], {"x": a["per_shot"]},
                                [dict(c, config_id="z")], {"z": c["per_shot"]})
        except ValueError:
            pass
        else:
            raise AssertionError("compared runs on different samples")

    tests_e12 = run_checks([
        ("4- and 6-cycle counts equal brute-force enumeration", _e12_cycle_counts_match_brute_force),
        ("cycle census: flag rows add exactly m_X T rows", _e12_census_on_72),
        ("sum-product arms run; cross-run comparison needs identical samples",
         _e12_sum_product_arms_and_cross_run),
    ])
    render_checks("Experiment 12", tests_e12)
    return (tests_e12,)


@app.cell
def _():
    e12_census_btn = mo.ui.run_button(label="Part A: count Tanner-graph cycles (no sampling, ~1 min)")
    e12_census_btn
    return (e12_census_btn,)


@app.cell
def _(E12_CYCLE_CODES, e12_census_btn):
    mo.stop(not e12_census_btn.value)
    with mo.status.progress_bar(total=len(E12_CYCLE_CODES), title="cycle census") as _bar:
        _rows = []
        for _c in E12_CYCLE_CODES:
            _rows.append(cycle_census(_c))
            _bar.update()
    _stamp = start_run("E12-cycles", "E12")
    _path = result_file("E12C", code=E12_CYCLE_CODES, rounds=12, p=1e-3, shots=0, seed=0,
                        convention="z-only", extra="cycles")
    write_result_file(_path, dict(
        config=dict(codes=E12_CYCLE_CODES, rounds=12, p=1e-3, shots=0, seed=0,
                    detector_convention="z-only", decoders={}),
        timing=dict(workers_requested=1, workers_effective=1, backends=["none"], resumed_chunks=0,
                    total_chunks=0),
        samples={}, rows=_rows), _stamp, "complete")
    e12_census = _rows
    mo.md("#### Part A — short cycles with and without the flag rows (T = 12)\n\n"
          + cycle_census_md(_rows)
          + f"\n\nSaved `{_path}`. Numbers in brackets are the sighted graph's cycles that pass "
            "through at least one flag row; a 4-cycle can contain at most one flag row, because "
            "flag groups are disjoint (Condition 4.1), so *per row* compares how many 4-cycles "
            "a detector row and a flag row each sit on. The updated arm decodes the blind graph.")
    return


@app.cell
def _(E08_MASTER_SEED, E12_DESIGN):
    e12_ui = grid_design_ui(E12_DESIGN, E08_MASTER_SEED)
    e12_run_btn = mo.ui.run_button(label="Run Experiment 12, Part B")
    mo.vstack([mo.md("### Experiment 12, Part B — controls (master seed = Experiment 8's, so S1 "
                     "decodes A1's shots)\n\n" + design_table_md(E12_DESIGN)), e12_ui, e12_run_btn])
    return e12_run_btn, e12_ui


@app.cell
def _(E12_DESIGN, core_ready, e12_run_btn, e12_ui, tests_e12):
    e12_run, _out = grid_run_cell("E12", "E12-sum-product", e12_ui.value, E12_DESIGN,
                                  core_ready and all_pass(tests_e12), e12_run_btn.value,
                                  "Experiment 12")
    _out
    return (e12_run,)


@app.cell
def _(e12_run):
    _ = e12_run
    e12_file, _files = grid_picker("E12")
    e12_e08_file, _f8 = grid_picker("E08")
    e12_load_btn = mo.ui.run_button(label="Load")
    e12_reexport_btn = mo.ui.run_button(label="Re-export loaded run")
    mo.vstack([mo.hstack([e12_file, e12_load_btn, e12_reexport_btn]),
               mo.md("Experiment 8 file with the min-sum arms on the same shots (A1):"), e12_e08_file]) \
        if _files else mo.md("*No saved Part B runs yet.*")
    return e12_e08_file, e12_file, e12_load_btn, e12_reexport_btn


@app.cell
def _(e12_e08_file, e12_file, e12_load_btn):
    e12_loaded = None
    if e12_load_btn.value and e12_file.value:
        _d, _r, _p = load_grid(e12_file.value, "E12-sum-product")
        e12_loaded = dict(path=e12_file.value, doc=_d, results=_r, per_shot=_p, fresh=False)
    e12_ms = (load_grid(e12_e08_file.value, "E08-prior-update")
              if (e12_load_btn.value and e12_e08_file.value) else None)
    return e12_loaded, e12_ms


@app.cell
def _(e12_loaded, e12_ms, e12_reexport_btn, e12_run):
    e12_data = e12_run if e12_run is not None else e12_loaded
    mo.stop(e12_data is None, mo.md("*Run Experiment 12 Part B or load a saved run.*"))
    _res = e12_data["results"]
    _per = e12_data.get("per_shot")
    if _per is None:
        _side = os.path.splitext(e12_data["path"])[0] + "_pershot.npz"
        _per = load_per_shot(_side) if os.path.exists(_side) else None
    _gap = gap_rows(_res) + (gap_rows(e12_ms[1]) if e12_ms else [])
    _tg = ["| code | p | BP | sighted failures | updated failures | sighted / updated | "
           "sighted-only : updated-only (McNemar p) |", "|:--|--:|:--|--:|--:|--:|:--|"]
    for _g in sorted(_gap, key=lambda g: (g["code"], g["p"], g["decoder"])):
        _bp = "sum-product" if _g["decoder"].endswith("-ps") else "min-sum"
        _split = ("—" if _g["p_mcnemar"] is None else
                  f"{_g['sighted_only']} : {_g['updated_only']} ({_g['p_mcnemar']:.2g})")
        _tg.append(f"| {_g['code']} | {_g['p']:g} | {_bp} ({_g['decoder']}) | {_g['sighted']} | "
                   f"{_g['updated']} | "
                   + ("—" if _g["ratio"] is None else ("∞" if _g["ratio"] == float("inf")
                                                       else f"{_g['ratio']:.2f}"))
                   + f" | {_split} |")
    _eff, _eff_md = [], "*Load the Experiment 8 file to compare min-sum and sum-product on the same shots.*"
    if e12_ms is not None:
        try:
            _eff = decoder_effect_rows(e12_ms[1], e12_ms[2], _res, _per)
            _eff_md = "\n".join(
                ["| code | arm | shots | min-sum failures | sum-product failures | "
                 "min-sum-only : sum-product-only | McNemar p |", "|:--|:--|--:|--:|--:|:--|--:|"]
                + [f"| {r['code']} | {r['arm']} | {r['shots']:,} | {r['failures_a']} | {r['failures_b']} "
                   f"| {r['only_a']} : {r['only_b']} | {r['p_mcnemar']:.2g} |" for r in _eff]) \
                if _eff else "*No configuration of the two files decodes the same shots.*"
        except ValueError as _e:
            _eff_md = f"**Refused:** {_e}"
    _figs = {f"arms_{r['config_id']}": plot_prior_update(r) for r in _res}
    _exp = maybe_export(e12_data, "Experiment 12 — sum-product arms", _figs,
                        {"arms": grid_rows(_res), "comparisons": comparison_rows(_res),
                         "gap": _gap, "min_sum_vs_sum_product": _eff},
                        reexport=e12_reexport_btn.value)
    mo.vstack([run_status_banner(e12_data["doc"], e12_data["path"]),
               mo.md("#### The sighted–updated gap under each BP rule\n\n" + "\n".join(_tg)),
               mo.md("#### Same shots, min-sum against sum-product, arm by arm\n\n" + _eff_md),
               mo.md(grid_table_md(_res)), *_figs.values(), _exp])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 17. The pre-registered confirmatory family (Holm, m = 14)

    Frozen before any Experiment 2, 8, 9, 10 or 11 sampling (registration
    record below).

    **What the Holm procedure does.** The thesis makes 14 confirmatory claims,
    each tested at the usual 5% level. Testing 14 hypotheses separately would
    give roughly a 1 − 0.95¹⁴ ≈ 51% chance of at least one false positive even
    if none of the effects were real. Holm's step-down procedure (Holm, 1979)
    keeps that *family-wise* error rate at 5% while being less strict than the
    plain Bonferroni correction (every test at 0.05/14): sort the 14 p-values
    from smallest to largest; the smallest must be below 0.05/14, the next below
    0.05/13, and so on up to 0.05/1 for the largest; stop at the first p-value
    that misses its threshold — that test and every later one are not rejected.
    "Pre-registered" means the 14 tests, their data and their thresholds were
    fixed and written down before the data were sampled, so the family cannot be
    chosen after seeing the results. Every other comparison in the notebook is
    exploratory and is reported without a confirmatory claim.

    | test | H1 | data | test |
    |---|---|---|---|
    | T1 | updated < sighted | E08 A1 [[72,12,6]] 1e-3, 50k | exact McNemar, two-sided |
    | T2 | updated < sighted | E08 A2 [[72,12,6]] 2e-3, 20k | exact McNemar |
    | T3 | updated < sighted | E08 A3 [[144,12,12]] 1e-3, 150k | exact McNemar |
    | T4 | updated < sighted | E02 A6 [[288,12,18]] 3e-3, 4 × 25k | McNemar on summed discordants |
    | T5 | updated < sighted | E08 A7 [[46,2,9]] 1e-3, 50k | exact McNemar |
    | T6 | updated < sighted | E08 A8 [[48,6,8]] 1e-3, 50k | exact McNemar |
    | T7 | sighted < unflagged | E02 A4 [[72,12,6]] 3e-3, 100k | exact conditional binomial (unpaired) |
    | T8 | sighted < unflagged | E02 A5 [[144,12,12]] 3e-3, 100k | exact conditional binomial |
    | T9 | sighted < unflagged | E02 A6 [[288,12,18]] 3e-3, 100k | exact conditional binomial |
    | T10 | slope of ln(blind/updated) on w < 0 | E09, nine codes | WLS, one-sided t (7 df) |
    | T11 | slope of ln(blind/updated) on ln p < 0 | E08 A1, A2 + E02 A4; E08 A3, A9 + E02 A5 | WLS, code fixed effect, one-sided t |
    | T12 | updated < sighted, BP+LSD-0 | E10 B1 | exact McNemar |
    | T13 | updated < blind, tuned Relay-BP | E11 R1 | exact McNemar |
    | T14 | per-shot time: updated-OSD-0 < sighted-OSD-0 | E08 A1, single-thread | Wilcoxon signed-rank, one-sided |

    **Decision rule.** Each test has a null hypothesis H0 (no effect) and the
    claim H1 above. Order the p-values; reject H0 of test $(i)$ — that is,
    **confirm** its claim — while $p_{(i)} \le 0.05/(14 - i + 1)$; stop at the
    first failure. The tables report the outcome as *confirmed* / *not
    confirmed* rather than "reject", since rejecting H0 is what supports the
    claim. A test whose data
    are missing enters with $p = 1$ (it cannot be rejected and is reported as
    missing). If A9 is not run, T11 leaves the family **at registration** and
    $m = 13$ (and so on for every test that leaves, e.g. T10 below).

    **Deviation recorded before sampling.** T7–T9 were registered as Boschloo's
    exact unconditional test. At $n = 10^5$ per arm that test's nuisance-parameter
    maximisation is computationally infeasible, so the exact conditional binomial
    test (`rate_ratio_ci`, the same exact family) is used, with its
    Clopper–Pearson rate-ratio interval as the effect size. T7–T9 are unpaired by
    construction.

    **Deviation recorded before Experiment 9's main run (2026-10-30).** T10's
    registered freeze rule (one $p$ from $\{10^{-3}, 2\times10^{-3},
    3\times10^{-3}\}$, 20,000 shots per code, ≥ 30 projected updated failures and
    blind LER ≤ 0.35 for every code) cannot be met by the nine w-family codes: at
    $10^{-3}$ the strongest members project ≈ 20 updated failures while the
    weakest already has a blind LER ≈ 0.5, and higher $p$ only saturates more
    members. Two readings are possible, and the switch below chooses between
    them: (a) **as registered** — no $p$ qualifies, so T10 leaves the family and
    $m = 13$ (Experiment 9 is then exploratory); (b) **revised** — keep both
    criteria but extend the pilot grid down to $5\times10^{-4}$ and give each
    code enough shots for 30 updated failures (Experiment 9, *Design*); T10 stays
    and $m = 14$. Both the registered and the revised outcome are stored in the
    pilot file.

    **Reporting.** Every test is reported with its effect size: the failure
    ratio with a 95% interval, the discordant split (paired tests), and, for T14,
    the median time ratio with a paired bootstrap interval.
    """)
    return


@app.cell
def _():
    HOLM_REGISTRATION = dict(
        registered="2026-10-28", alpha=0.05, m=14,
        tests=["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14"],
        a3_shots=150_000, frontier_pruning="pilot median < 3 x updated-OSD-0 median",
        w_family_freeze="smallest p with >= 30 projected updated failures at 20k and blind LER <= 0.35",
        relay_stop_rule_core_hours=20.0,
        master_seeds=dict(E02=2026110201, E08=2026110801, E08_replication=2026110802, E09=2026110901,
                          E09_pilot=2026110902, E11_tuning=2026111101),
        t11_requires="E08 A9 ([[144,12,12]], 2e-3); without it T11 leaves the family and m = 13",
        deviation_t7_t9=("Boschloo exact unconditional test registered; replaced before sampling by "
                         "the exact conditional binomial test (computational infeasibility at n = 1e5)"),
        deviation_t10=("2026-10-30, before E09 main sampling: registered freeze rule infeasible on the "
                       "w-family; revised to the largest p in {5e-4, 7e-4, 1e-3, 2e-3, 3e-3} with "
                       "every blind LER <= 0.35 and per-code shots for >= 30 updated failures "
                       "(20k..200k); alternatively T10 leaves the family (m = 13)"),
    )
    return (HOLM_REGISTRATION,)


@app.function
def median_ratio_ci(a, b, B=2000, seed=0, conf=0.95):
    """Median of a / median of b with a paired bootstrap interval over shots."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    r = float(np.median(a) / np.median(b))
    idx = rng.integers(0, len(a), size=(B, len(a)))
    bs = np.median(a[idx], axis=1) / np.median(b[idx], axis=1)
    q = (1 - conf) / 2
    return dict(ratio=r, lo=float(np.quantile(bs, q)), hi=float(np.quantile(bs, 1 - q)), B=B)


@app.function
def find_config(results, code, p, decoder="OSD-0", tag=None, rounds=12):
    """The (shard-combined) result of one configuration, or None."""
    rs = [r for r in results if r["code"] == code and abs(r["p"] - p) < 1e-12
          and r["decoder"] == decoder and r.get("tag") == tag and r["rounds"] == rounds]
    if not rs:
        return None
    return combine_shards(rs)[0] if len(rs) > 1 else rs[0]


@app.function
def paired_test(res, a, b, tid, h1):
    """A paired McNemar row: H1 "a < b" is tested two-sided on the a|b discordants."""
    if res is None:
        return dict(test=tid, h1=h1, p=1.0, missing=True)
    key_ = f"{b}|{a}" if f"{b}|{a}" in res["paired"] else f"{a}|{b}"
    c = res["paired"].get(key_)
    if c is None:
        return dict(test=tid, h1=h1, p=1.0, missing=True)
    if res.get("pairing", {}).get(key_) is False:
        raise ValueError(f"{tid}: {key_} is labelled paired but the arms decoded different samples")
    ka, kb = res["arms"][a]["failures"], res["arms"][b]["failures"]
    a_only = c["a_only"] if c["a"] == a else c["b_only"]
    b_only = c["b_only"] if c["a"] == a else c["a_only"]
    rr = rate_ratio_ci(ka, res["shots"], kb, res["shots"])
    return dict(test=tid, h1=h1, p=float(c["p"]), missing=False, code=res["code"], p_phys=res["p"],
                shots=res["shots"], effect=f"{a}-only {a_only} : {b}-only {b_only}",
                ratio=rr["ratio"], ratio_lo=rr["lo"], ratio_hi=rr["hi"],
                ratio_note="interval treats the arms as unpaired (conservative)",
                direction_ok=bool(a_only < b_only))


@app.function
def unpaired_test(res, a, b, tid, h1):
    if res is None or a not in res["arms"] or b not in res["arms"]:
        return dict(test=tid, h1=h1, p=1.0, missing=True)
    rr = rate_ratio_ci(res["arms"][a]["failures"], res["shots"], res["arms"][b]["failures"],
                       res["shots"])
    return dict(test=tid, h1=h1, p=float(rr["p"]), missing=False, code=res["code"], p_phys=res["p"],
                shots=res["shots"], effect=f"{res['arms'][a]['failures']} vs {res['arms'][b]['failures']}",
                ratio=rr["ratio"], ratio_lo=rr["lo"], ratio_hi=rr["hi"], ratio_note="unpaired",
                direction_ok=bool(rr["ratio"] < 1))


@app.function
def t11_slope(results_e08, results_e02):
    """T11: ln(blind/updated) on ln p with a fixed effect per code."""
    pts = []
    for code, sources in (("[[72, 12, 6]]", ((results_e08, 1e-3), (results_e08, 2e-3), (results_e02, 3e-3))),
                          ("[[144, 12, 12]]", ((results_e08, 1e-3), (results_e08, 2e-3), (results_e02, 3e-3)))):
        for results, p in sources:
            r = find_config(results, code, p)
            if r is None or "blind|updated" not in r["paired"]:
                continue
            kb, ku = r["arms"]["blind"]["failures"], r["arms"]["updated"]["failures"]
            if kb and ku:
                c = r["paired"]["blind|updated"]
                pts.append(dict(code=code, p=p, y=math.log(kb / ku),
                                var=max(c["a_only"] + c["b_only"], 1) / (kb * ku)))
    if len(pts) < 4 or len({x["code"] for x in pts}) < 1:
        return None, pts
    out = wls_slope([math.log(x["p"]) for x in pts], [x["y"] for x in pts], [x["var"] for x in pts],
                    alternative="less", groups=[x["code"] for x in pts])
    return out, pts


@app.function
def holm_analysis(e02=None, e08=None, e09=None, e10=None, e11=None, per_shot_e08=None,
                  include_t11=True, include_t10=True, alpha=0.05):
    """
    The confirmatory family from the result lists of each experiment.
    Returns (rows, holm rows, m). Missing data enter with p = 1.
    """
    e02, e08, e09, e10, e11 = (x or [] for x in (e02, e08, e09, e10, e11))
    rows = [
        paired_test(find_config(e08, "[[72, 12, 6]]", 1e-3), "updated", "sighted", "T1", "updated < sighted"),
        paired_test(find_config(e08, "[[72, 12, 6]]", 2e-3), "updated", "sighted", "T2", "updated < sighted"),
        paired_test(find_config(e08, "[[144, 12, 12]]", 1e-3), "updated", "sighted", "T3", "updated < sighted"),
        paired_test(find_config(e02, "[[288, 12, 18]]", 3e-3), "updated", "sighted", "T4", "updated < sighted"),
        paired_test(find_config(e08, "[[46, 2, 9]]", 1e-3), "updated", "sighted", "T5", "updated < sighted"),
        paired_test(find_config(e08, "[[48, 6, 8]]", 1e-3), "updated", "sighted", "T6", "updated < sighted"),
        unpaired_test(find_config(e02, "[[72, 12, 6]]", 3e-3), "sighted", "unflagged", "T7", "sighted < unflagged"),
        unpaired_test(find_config(e02, "[[144, 12, 12]]", 3e-3), "sighted", "unflagged", "T8", "sighted < unflagged"),
        unpaired_test(find_config(e02, "[[288, 12, 18]]", 3e-3), "sighted", "unflagged", "T9", "sighted < unflagged"),
    ]
    if include_t10:
        s10 = e09_slope(e09) if e09 else dict(slope=None)
        rows.append(dict(test="T10", h1="slope on w < 0",
                         p=s10["p"] if s10.get("slope") is not None else 1.0,
                         missing=s10.get("slope") is None,
                         effect=(f"slope {s10['slope']:.4f} ± {s10['se']:.4f} (df {s10['df']})"
                                 if s10.get("slope") is not None else None)))
    if include_t11:
        s11, _pts = t11_slope(e08, e02)
        rows.append(dict(test="T11", h1="slope on ln p < 0", p=s11["p"] if s11 else 1.0,
                         missing=s11 is None,
                         effect=(f"slope {s11['slope']:.3f} ± {s11['se']:.3f} (df {s11['df']})"
                                 if s11 else None)))
    rows.append(paired_test(find_config(e10, "[[72, 12, 6]]", 1e-3, decoder="LSD-0"), "updated",
                            "sighted", "T12", "updated < sighted (BP+LSD-0)"))
    rows.append(paired_test(find_config(e11, "[[72, 12, 6]]", 2e-3, decoder="relay"), "updated",
                            "blind", "T13", "updated < blind (Relay-BP)"))
    t14 = dict(test="T14", h1="time updated < sighted (A1, workers 1)", p=1.0, missing=True)
    a1 = find_config(e08, "[[72, 12, 6]]", 1e-3)
    if a1 is not None and per_shot_e08 is not None:
        ps_ = per_shot_e08.get(a1.get("config_id"), {})
        if "updated" in ps_ and "sighted" in ps_:
            wk = {a1["arms"][a]["workers_effective"] for a in ("updated", "sighted")}
            tu, ts = ps_["updated"]["time_us"].astype(float), ps_["sighted"]["time_us"].astype(float)
            w = st.wilcoxon(tu - ts, alternative="less")
            mr = median_ratio_ci(tu, ts)
            t14 = dict(test="T14", h1=t14["h1"], p=float(w.pvalue), missing=False,
                       effect=f"median ratio {mr['ratio']:.3f} [{mr['lo']:.3f}, {mr['hi']:.3f}]",
                       workers=sorted(wk), valid_regime=wk == {1})
    rows.append(t14)
    m = len(rows)
    h = holm([r["p"] for r in rows], alpha)
    for r, x in zip(rows, h):
        r.update(threshold=x["threshold"], reject=x["reject"], p_holm=x["p_adjusted"])
        if r.get("direction_ok") is False:
            r["reject_note"] = "significant in the OPPOSITE direction to H1" if x["reject"] else ""
        r["outcome"] = holm_outcome(r)
    return rows, h, m


@app.function
def holm_outcome(row):
    """Plain-language verdict of one family member. Rejecting the null
    hypothesis is what CONFIRMS the claim H1, so 'reject' is reported as
    'confirmed'; a significant result in the opposite direction confirms nothing."""
    if row.get("missing"):
        return "not tested (data missing)"
    if not row.get("reject"):
        return "not confirmed"
    if row.get("direction_ok") is False:
        return "not confirmed (significant in the opposite direction)"
    return "confirmed"


@app.cell
def _():
    def _holm_known_vector():
        p = [0.001, 0.004, 0.03, 0.04]
        h = holm(p)
        assert [x["reject"] for x in h] == [True, True, False, False]
        assert abs(2 * 0.5 ** 9 - 3.90625e-3) < 1e-12 and abs(2 * 0.5 ** 10 - 1.953125e-3) < 1e-12

    def _missing_tests_enter_with_p_one():
        rows, _h, m = holm_analysis()
        assert m == 14 and all(r["missing"] and r["p"] == 1.0 for r in rows)
        rows, _h, m = holm_analysis(include_t11=False)
        assert m == 13
        rows, _h, m = holm_analysis(include_t11=False, include_t10=False)
        assert m == 12 and "T10" not in {r["test"] for r in rows}

    def _outcome_wording():
        assert holm_outcome(dict(reject=True)) == "confirmed"
        assert holm_outcome(dict(reject=False)) == "not confirmed"
        assert holm_outcome(dict(reject=True, direction_ok=False)).startswith("not confirmed")
        assert holm_outcome(dict(missing=True, reject=False)).startswith("not tested")
        rows, _h, _m = holm_analysis()
        assert all(r["outcome"].startswith("not tested") for r in rows)

    def _paired_direction():
        res = dict(code="c", p=1e-3, shots=1000, decoder="OSD-0", rounds=12,
                   arms=dict(updated=dict(failures=2), sighted=dict(failures=14)),
                   paired={"sighted|updated": dict(a="sighted", b="updated", a_only=12, b_only=0,
                                                   p=2 * 0.5 ** 12)})
        r = paired_test(res, "updated", "sighted", "T1", "updated < sighted")
        assert r["effect"] == "updated-only 0 : sighted-only 12" and r["direction_ok"]
        assert abs(r["p"] - 2 * 0.5 ** 12) < 1e-15

    def _median_ratio_ci():
        rng = np.random.default_rng(0)
        a = rng.gamma(4, 100, 4000)
        m = median_ratio_ci(a * 0.8, a, B=500)
        assert abs(m["ratio"] - 0.8) < 1e-9 and m["lo"] <= 0.8 <= m["hi"]

    tests_holm = run_checks([
        ("Holm step-down on a known vector; all-one-way k = 9, 10 p-values", _holm_known_vector),
        ("missing tests enter with p = 1; without A9 m = 13", _missing_tests_enter_with_p_one),
        ("paired tests report the split in H1's direction", _paired_direction),
        ("outcomes read 'confirmed' / 'not confirmed', never a bare 'reject'", _outcome_wording),
        ("paired bootstrap median-ratio interval", _median_ratio_ci),
    ])
    render_checks("17 confirmatory family", tests_holm)
    return


@app.cell
def _():
    def _multi(k):
        fs = saved_runs(k)
        return mo.ui.multiselect({os.path.basename(f): f for f in fs},
                                 value=[os.path.basename(fs[-1])] if fs else [],
                                 label=f"{k} file(s)")

    holm_files = mo.ui.dictionary({k: _multi(k) for k in ("E02", "E08", "E09", "E10", "E11")})
    holm_t11 = mo.ui.switch(value=True, label="T11 in the family (A9 was run)")
    holm_t10 = mo.ui.switch(value=True, label="T10 in the family (revised Experiment 9 freeze rule)")
    holm_btn = mo.ui.run_button(label="Evaluate the family")
    holm_reg_btn = mo.ui.run_button(label="Write the registration record (once, before sampling)")
    mo.vstack([holm_files, holm_t11, holm_t10, mo.hstack([holm_btn, holm_reg_btn])])
    return holm_btn, holm_files, holm_reg_btn, holm_t10, holm_t11


@app.cell
def _(HOLM_REGISTRATION, holm_reg_btn):
    mo.stop(not holm_reg_btn.value)
    _path = os.path.join(RESULTS_DIR, "registration_holm_family.json")
    if os.path.exists(_path):
        _msg = f"A registration record already exists (`{_path}`); it is never overwritten."
    else:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        _txt = json.dumps(dict(HOLM_REGISTRATION, written=datetime.datetime.now().isoformat(
            timespec="seconds"), code=code_identity()), indent=1).encode("utf-8")
        durable_write(_path, lambda f: f.write(_txt))
        _msg = f"Registration written to `{_path}`."
    mo.md(_msg)
    return


@app.cell
def _(holm_btn, holm_files, holm_t10, holm_t11):
    mo.stop(not holm_btn.value, mo.md("*Choose the result files and press **Evaluate the family**. "
                                       "An experiment run in parts (e.g. Experiment 8's A1 and A3 "
                                       "single-threaded, the rest in parallel) takes all its files; "
                                       "a configuration found in two files is refused.*"))
    _v = holm_files.value
    _ld = {k: (merge_grids([load_grid(f) for f in fs], fs) if fs else None) for k, fs in _v.items()}
    holm_rows, _h, _m = holm_analysis(
        e02=_ld["E02"][1] if _ld["E02"] else None, e08=_ld["E08"][1] if _ld["E08"] else None,
        e09=_ld["E09"][1] if _ld["E09"] else None, e10=_ld["E10"][1] if _ld["E10"] else None,
        e11=_ld["E11"][1] if _ld["E11"] else None, per_shot_e08=_ld["E08"][2] if _ld["E08"] else None,
        include_t11=holm_t11.value, include_t10=holm_t10.value)
    _reg = os.path.join(RESULTS_DIR, "registration_holm_family.json")
    _dated = (json.load(open(_reg, encoding="utf-8")).get("written") if os.path.exists(_reg) else None)
    _t = ["| test | claim (H1) | p | Holm threshold | claim | effect | ratio [95% CI] |",
          "|:--|:--|--:|--:|:--|:--|:--|"]
    for _r in holm_rows:
        _rat = (f"{_r['ratio']:.3g} [{_r['ratio_lo']:.3g}, {_r['ratio_hi']:.3g}]"
                if _r.get("ratio") is not None else "")
        _pv = ("missing" if _r["missing"] else
               ("< 1e-300" if _r["p"] == 0 else f"{_r['p']:.3g}"))
        _oc = _r["outcome"]
        _t.append(f"| {_r['test']} | {_r['h1']} | {_pv} | {_r['threshold']:.4g} | "
                  f"{'**' + _oc + '**' if _oc == 'confirmed' else _oc} | "
                  f"{_r.get('effect') or ''} | {_rat} |")
    mo.vstack([mo.md(f"**m = {_m}** · registration record: "
                     + (f"written {_dated}" if _dated else "**not found** — the family is not registered")),
               mo.md("\n".join(_t)),
               mo.md("*claim*: **confirmed** means the test's null hypothesis (no effect) is "
                     "rejected at its Holm threshold, i.e. the evidence supports the claim H1. "
                     "*not confirmed* means the data do not reach that threshold — it is not "
                     "evidence for the opposite. A $p$ of 0 in double precision is shown as "
                     "< 1e-300.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 18. Regenerating every thesis table and figure

    Every number in Chapter 5 is re-derivable from a result file by one command:

    ```
    python FlaggedDecoding.py tables     [--manifest thesis_sources.json] [--out thesis_out] [--allow-missing]
    python FlaggedDecoding.py figures    [...]
    python FlaggedDecoding.py regenerate [...]      # tables and figures
    python FlaggedDecoding.py selftest
    ```

    The commands run this notebook headless (its self-tests included, a couple
    of minutes) and then work from **`thesis_sources.json`**, a manifest that
    names, for each thesis item, the result file(s) it comes from, their sha256,
    and what the caption claims (`expect`: code, $T$, $p$, shots …). For every
    item the regenerator

    1. resolves the file and **refuses** anything inside `checkpoints/` or
       `superseded/` → `MissingSource` / `SourceMismatch`;
    2. compares the file's sha256 with the manifest → `SourceMismatch`;
    3. loads it through the provenance-aware loaders of §12 and **refuses** an
       unknown detector convention → `ProvenanceUnknown`;
    4. checks every `expect` field against the file's configuration, and that
       items declared to share samples really do (Table 5.9 with Table 5.8)
       → `CaptionMismatch`;
    5. computes the item and writes CSV + LaTeX (tables) or PNG + PDF (figures)
       with a JSON sidecar of the source hashes, and logs the outcome.

    Any failure exits with status 1 (`--allow-missing` only skips missing files,
    for drafts). Every time column carries its cost convention. Older CSVs
    written in the platform encoding are read with a cp1252 fallback.
    """)
    return


@app.class_definition
class RegenerationError(Exception):
    """Base class: a thesis item could not be regenerated faithfully."""


@app.class_definition
class MissingSource(RegenerationError):
    pass


@app.class_definition
class SourceMismatch(RegenerationError):
    pass


@app.class_definition
class CaptionMismatch(RegenerationError):
    pass


@app.class_definition
class ProvenanceUnknown(RegenerationError):
    pass


@app.cell
def _():
    THESIS_ITEMS = {
        "table_4_1": dict(kind="table", sources=["E02"], optional=["E06"],
                          caption="Table 4.1 — qubits, detectors and faults with and without flags"),
        "table_5_1": dict(kind="table", sources=["E01"], caption="Table 5.1 — peeling baseline"),
        "table_5_2": dict(kind="table", sources=["E04"], caption="Table 5.2 — reference policies"),
        "table_5_3": dict(kind="table", sources=["E04"], caption="Table 5.3 — trigger rules"),
        "table_5_4": dict(kind="table", sources=["E04"], caption="Table 5.4 — flag diagnostics"),
        "table_5_5": dict(kind="table", sources=["E04"], caption="Table 5.5 — verdicts per pair"),
        "table_5_6": dict(kind="table", sources=["E03"], caption="Table 5.6 — flag rate vs law L6"),
        "table_5_7": dict(kind="table", sources=["E02"], caption="Table 5.7 — syndrome augmentation"),
        "table_5_8": dict(kind="table", sources=["E08"], caption="Table 5.8 — prior updating"),
        "table_5_9": dict(kind="table", sources=["E07", "E08"], same_samples=("E07", "E08"),
                          caption="Table 5.9 — accuracy-latency frontier"),
        "table_holm": dict(kind="table", sources=[], optional=["E02", "E08", "E09", "E10", "E11"],
                           caption="Confirmatory family (Holm)"),
        "fig_5_1": dict(kind="figure", sources=["E06"], optional=["E06X"],
                        caption="Figure 5.1 — surface-code baseline"),
        "fig_5_2": dict(kind="figure", sources=["E04"], caption="Figure 5.2 — reference policies"),
        "fig_5_3": dict(kind="figure", sources=["E02"], caption="Figure 5.3 — LER by arm"),
        "fig_5_4": dict(kind="figure", sources=["E08"], caption="Figure 5.4 — prior updating (A1)"),
        "fig_5_5": dict(kind="figure", sources=["E07", "E08"], same_samples=("E07", "E08"),
                        caption="Figure 5.5 — frontier"),
    }
    return (THESIS_ITEMS,)


@app.function
def load_any(path):
    """
    Load any result file through its experiment's loader. Returns dict(doc,
    config, data) where data is what the builders consume.
    """
    base = os.path.basename(path)
    if path.endswith(".npz"):
        d = e04_load(path)
        return dict(doc=d["doc"], config=d["doc"]["config"], data=d)
    doc = read_result_file(path)
    kind = doc.get("kind")
    if kind in ("E04-grid", "E04-retime"):
        d = e04_load(path)
        return dict(doc=d["doc"], config=d["doc"]["config"], data=d)
    if kind in ("E01-peeling", "pakhunov"):
        d = e01_load(path)
    elif kind in ("E03-flag-rate", "validation") and "checks" in doc and "v5" in doc.get("checks", {}):
        d = e03_load(path)
    elif kind == "E03-flag-rate":
        d = e03_load(path)
    elif kind in ("E06-surface", "exp7"):
        d = e06_load(path)
    elif doc.get("results") is not None and kind and kind[:3] in (
            "E02", "E07", "E08", "E09", "E10", "E11"):
        dd, res, per = load_grid(path)
        d = dict(path=path, doc=dd, results=res, per_shot=per, fresh=False)
    else:
        cfg, prov = config_from_file(doc, path)
        d = dict(path=path, doc=dict(doc, config=cfg, provenance=prov), fresh=False)
    cfg = dict(d["doc"].get("config") or {})
    if d["doc"].get("provenance"):
        cfg.setdefault("x_detectors", None)
    return dict(doc=d["doc"], config=cfg, data=d)


@app.function
def check_expect(config, expect, item):
    """CaptionMismatch unless every expected field matches the file's configuration."""
    alias = {"rounds": ("rounds", "rounds_list"), "p": ("p", "ps"), "code": ("code", "codes"),
             "shots": ("shots", "shots_per_config"), "seed": ("seed", "master_seed")}
    for k, want in (expect or {}).items():
        have = None
        for a in alias.get(k, (k,)):
            if a in config and config[a] is not None:
                have = config[a]
                break
        ok = (want in have) if isinstance(have, list) and not isinstance(want, list) else (
            have == want or (isinstance(have, float) and isinstance(want, (int, float))
                             and abs(have - want) < 1e-12))
        if not ok:
            raise CaptionMismatch(f"{item}: caption says {k} = {want!r}, the file has {have!r}")


@app.function
def check_provenance(config, item):
    conv = config.get("detector_convention")
    if conv in (None, "unknown") or ("x_detectors" in config and config["x_detectors"] is None
                                     and conv not in DETECTOR_CONVENTIONS):
        raise ProvenanceUnknown(f"{item}: the source's detector convention is unknown")


@app.function
def resolve_source(path, sha, item, root):
    full = path if os.path.isabs(path) else os.path.join(root, path)
    parts = full.replace("\\", "/").split("/")
    if "checkpoints" in parts or "superseded" in parts:
        raise SourceMismatch(f"{item}: {path} is a checkpoint or superseded file, not a result")
    if not os.path.exists(full):
        raise MissingSource(f"{item}: {path} does not exist")
    if sha and file_sha256(full) != sha:
        raise SourceMismatch(f"{item}: {path} has changed since the manifest was written "
                             "(sha256 differs)")
    return full


@app.function
def shared_sample_check(res_a, res_b):
    """
    Which configurations of two grid files decode the same shots. Pairs are
    matched on (code, T, p, shard, shots); a smaller or larger run of the same
    configuration (e.g. Experiment 7's 10,000-shot p99 points) is a different
    sample set and is not compared. Within a matched pair every arm present in
    both must carry the same sample hash. Returns (number of matched arm pairs,
    list of mismatches).
    """
    def key(r):
        return (r["code"], int(r["rounds"]), round(float(r["p"]), 12), int(r.get("shard", 0)),
                int(r["shots"]))
    by_b = {}
    for r in res_b:
        if not r.get("tag"):
            by_b.setdefault(key(r), []).append(r)
    matched, bad = 0, []
    for ra in res_a:
        if ra.get("tag"):
            continue
        for rb in by_b.get(key(ra), []):
            for arm in set(ra["arms"]) & set(rb["arms"]):
                ha = ra["arms"][arm].get("sample_sha256")
                hb = rb["arms"][arm].get("sample_sha256")
                if ha is None or hb is None:
                    continue
                matched += 1
                if ha != hb:
                    bad.append(f"{ra['code']} p={ra['p']:g} {ra['shots']:,} shots, {arm}")
    return matched, bad


@app.function
def tex_table(rows, caption):
    """A plain LaTeX tabular of the rows (numbers to 4 significant figures)."""
    if not rows:
        return f"% {caption}: no rows\n"
    cols = list(dict.fromkeys(k for r in rows for k in r))

    def cell(v):
        if isinstance(v, float):
            v = f"{v:.4g}"
        return str(v).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")

    lines = [f"% {caption}", r"\begin{tabular}{" + "l" * len(cols) + "}", r"\hline",
             " & ".join(cell(c) for c in cols) + r" \\", r"\hline"]
    lines += [" & ".join(cell(r.get(c, "")) for c in cols) + r" \\" for r in rows]
    lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


@app.function
def flat_rows(rows):
    """Rows with nested values dropped or flattened, for CSV/TeX."""
    out = []
    for r in rows:
        o = {}
        for k, v in r.items():
            if isinstance(v, (list, tuple)) and len(v) == 3 and k in ("ler",):
                o["ler"], o["ler_low"], o["ler_high"] = v
            elif isinstance(v, (list, tuple)) and len(v) == 2 and k in ("saving_ci",):
                o["saving_lo"], o["saving_hi"] = v
            elif isinstance(v, (dict, list, tuple, np.ndarray)):
                continue
            else:
                o[k] = v
        out.append(o)
    return out


@app.function
def build_item(item, src, options=None):
    """Compute one thesis item from its loaded sources -> (rows or None, figure or None)."""
    o = options or {}
    if item in ("table_5_2", "table_5_3", "table_5_4", "table_5_5", "fig_5_2"):
        d = src["E04"]["data"]
        res, nf, fr, meta = d["results"], d["n_flags"], d["flag_rounds"], d["meta"]
        w0, s0 = o.get("weak", "peel"), o.get("strong", "OSD-0")
        workers = meta.get("workers_effective") or meta.get("workers")
        if item in ("table_5_2", "fig_5_2"):
            rows = policy_table(res, w0, s0, nf, workers=workers,
                                convention=o.get("convention", "derive"))
            return (rows, None) if item == "table_5_2" else (None, plot_reference_policies(rows))
        weak, strong = e04_names(res)
        a = analyse(res, weak, strong, nf, fr, workers=workers,
                    chunk_size=int(meta.get("chunk_size", CHUNK_SIZE)))
        if item == "table_5_3":
            return zoo_pair_rows(a, w0, s0), None
        if item == "table_5_4":
            return a["flag_diagnostics"], None
        return [dict(p, timing=e04_timing_label(meta, p["strong"])) for p in a["pairs"]], None
    if item == "table_5_1":
        return [r for r in src["E01"]["doc"]["rows"]], None
    if item == "table_5_6":
        return src["E03"]["doc"]["rows"], None
    if item == "table_5_7":
        comb_ = combine_shards(src["E02"]["data"]["results"])
        return [dict(code=c["code"], p=c["p"], shots=c["shots"],
                     **{f"{a}_failures": m["failures"] for a, m in c["arms"].items()},
                     **{f"{a}_ler": m["ler"] for a, m in c["arms"].items()},
                     **{f"{k}_ratio": u["ratio"] for k, u in c["unpaired"].items()},
                     **{f"{k}_lo": u["lo"] for k, u in c["unpaired"].items()},
                     **{f"{k}_hi": u["hi"] for k, u in c["unpaired"].items()}) for c in comb_], None
    if item == "fig_5_3":
        return None, plot_arm_grid(src["E02"]["data"]["results"])
    if item == "table_4_1":
        rows = overhead_rows(src["E02"]["data"]["results"])
        if "E06" in src:
            for pt in src["E06"]["doc"]["points"]:
                if pt["p"] == min(x["p"] for x in src["E06"]["doc"]["points"]):
                    rows.append(dict(code=f"surface d={pt['distance']}", rounds=pt["rounds"],
                                     qubits_unflagged=pt.get("qubits_used"),
                                     detectors_unflagged=pt["detectors"],
                                     faults_unflagged=pt["faults"]))
        return rows, None
    if item in ("table_5_8", "fig_5_4"):
        res = src["E08"]["data"]["results"]
        if item == "table_5_8":
            return table_5_8_rows(res), None
        a1 = find_config(res, o.get("code", "[[72, 12, 6]]"), o.get("p", 1e-3),
                         rounds=o.get("rounds", 12))
        if a1 is None:
            raise CaptionMismatch(f"{item}: no {o.get('code', '[[72, 12, 6]]')} configuration at "
                                  f"p = {o.get('p', 1e-3)}, T = {o.get('rounds', 12)} in the source")
        return None, plot_prior_update(a1)
    if item in ("table_5_9", "fig_5_5"):
        e07, e08 = src["E07"]["data"]["results"], src["E08"]["data"]["results"]
        code = o.get("code", "[[72, 12, 6]]")
        T = o.get("rounds", 12)
        pts = frontier_points([r for r in e07 if r["code"] == code and r["rounds"] == T],
                              [r for r in e08 if r["code"] == code and not r.get("tag")
                               and r["rounds"] == T and abs(r["p"] - o.get("p", 1e-3)) < 1e-12])
        if not pts:
            raise CaptionMismatch(f"{item}: no frontier points for {code} at T = {T}")
        if item == "table_5_9":
            front = {(r["arm"], r["decoder"]) for r in pareto_frontier(pts)}
            return [dict({k: v for k, v in r.items() if not isinstance(v, (dict, list))},
                         on_frontier=(r["arm"], r["decoder"]) in front) for r in pts], None
        return None, plot_pareto(pts)
    if item == "fig_5_1":
        doc = src["E06"]["doc"]
        th = doc.get("thresholds") or e06_thresholds(doc["points"])
        return None, plot_surface_sweep(doc["points"], threshold=th.get("osd"),
                                        mwpm_threshold=th.get("mwpm"))
    if item == "table_holm":
        g = {k: src[k]["data"] for k in ("E02", "E08", "E09", "E10", "E11") if k in src}
        rows, _h, _m = holm_analysis(
            e02=g.get("E02", {}).get("results"), e08=g.get("E08", {}).get("results"),
            e09=g.get("E09", {}).get("results"), e10=g.get("E10", {}).get("results"),
            e11=g.get("E11", {}).get("results"),
            per_shot_e08=g.get("E08", {}).get("per_shot"), include_t11=o.get("include_t11", True),
            include_t10=o.get("include_t10", True))
        rows = [dict({k: v for k, v in r.items() if k not in ("reject", "reject_note", "outcome")},
                     claim=r["outcome"]) for r in rows]
        return rows, None
    raise KeyError(f"unknown thesis item {item!r}")


@app.cell
def _(THESIS_ITEMS):
    def regenerate(manifest_path, out_dir, what=("table", "figure"), allow_missing=False, only=None):
        """
        Regenerate every manifest item of the requested kinds. Returns (log, exit
        code). Never writes to the results folder.
        """
        with open(manifest_path, encoding="utf-8") as fh:
            man = json.load(fh)
        root = os.path.dirname(os.path.abspath(manifest_path))
        os.makedirs(out_dir, exist_ok=True)
        log, failed = [], False
        for item, entry in man["items"].items():
            spec = THESIS_ITEMS.get(item)
            if spec is None or spec["kind"] not in what or (only and item not in only):
                continue
            rec = dict(item=item, caption=spec["caption"], sources={}, status="ok")
            try:
                src = {}
                for exp, paths in entry.get("files", {}).items():
                    single = isinstance(paths, str)
                    paths = [paths] if single else list(paths)
                    shas = entry.get("sha256", {}).get(exp)
                    shas = [shas] if single or shas is None else list(shas)
                    shas += [None] * (len(paths) - len(shas))
                    try:
                        fulls = [resolve_source(pth, sha, item, root) for pth, sha in zip(paths, shas)]
                    except MissingSource:
                        if exp in spec.get("optional", []):
                            continue
                        raise
                    lds = [load_any(f) for f in fulls]
                    for ld in lds:
                        check_provenance(ld["config"], item)
                        check_expect(ld["config"], entry.get("expect", {}).get(exp), item)
                    if len(lds) == 1:
                        src[exp] = lds[0]
                    else:
                        if any("results" not in ld["data"] for ld in lds):
                            raise SourceMismatch(f"{item}: only grid experiments can take several "
                                                 f"files ({exp})")
                        try:
                            doc, res, per = merge_grids(
                                [(ld["doc"], ld["data"]["results"], ld["data"].get("per_shot"))
                                 for ld in lds], fulls)
                        except ValueError as exc:
                            raise SourceMismatch(f"{item}: {exc}") from None
                        src[exp] = dict(doc=doc, config=lds[0]["config"],
                                        data=dict(path=fulls, doc=doc, results=res, per_shot=per,
                                                  fresh=False))
                    rec["sources"][exp] = [dict(path=pth, sha256=file_sha256(f))
                                           for pth, f in zip(paths, fulls)]
                for exp in spec["sources"]:
                    if exp not in src:
                        raise MissingSource(f"{item}: no source for {exp} in the manifest")
                if spec.get("same_samples"):
                    a, b = spec["same_samples"]
                    matched, bad = shared_sample_check(src[a]["data"].get("results") or [],
                                                       src[b]["data"].get("results") or [])
                    if bad:
                        raise CaptionMismatch(f"{item}: {a} and {b} do not decode the same samples "
                                              f"for {', '.join(bad)}")
                    if not matched:
                        raise CaptionMismatch(f"{item}: no configuration of {a} has a configuration of "
                                              f"{b} with the same code, T, p and shot count")
                rows, fig = build_item(item, src, entry.get("options"))
                base = os.path.join(out_dir, item)
                if rows is not None:
                    fr = flat_rows(rows)
                    cols = list(dict.fromkeys(k for r in fr for k in r))
                    with open(base + ".csv", "w", newline="", encoding="utf-8") as fh:
                        w = csv.DictWriter(fh, fieldnames=cols)
                        w.writeheader()
                        for r in fr:
                            w.writerow({k: r.get(k, "") for k in cols})
                    with open(base + ".tex", "w", encoding="utf-8") as fh:
                        fh.write(tex_table(fr, spec["caption"]))
                if fig is not None:
                    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
                    fig.savefig(base + ".pdf", bbox_inches="tight")
                    plt.close(fig)
                with open(base + ".sources.json", "w", encoding="utf-8") as fh:
                    json.dump(dict(rec, code=code_identity()), fh, indent=1)
            except MissingSource as exc:
                rec.update(status="missing", error=str(exc))
                failed |= not allow_missing
            except RegenerationError as exc:
                rec.update(status=type(exc).__name__, error=str(exc))
                failed = True
            except Exception as exc:                       # noqa: BLE001 - never a silent partial run
                rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
                failed = True
            log.append(rec)
        with open(os.path.join(out_dir, "regeneration_log.json"), "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=1, default=str)
        return log, (1 if failed else 0)

    return (regenerate,)


@app.cell
def _(THESIS_ITEMS):
    def write_manifest(path, files, expect=None, options=None, items=None):
        """
        Write thesis_sources.json: `files` maps experiment -> result path, or a
        list of paths for a grid experiment run in parts (relative to the
        manifest's folder is best). Every requested item gets the files of its
        experiments, their sha256 and the expect/options given.
        """
        def sha(f):
            return file_sha256(f if os.path.isabs(f) else os.path.join(root, f))

        root = os.path.dirname(os.path.abspath(path))
        out = {"written": datetime.datetime.now().isoformat(timespec="seconds"), "items": {}}
        for item, spec in THESIS_ITEMS.items():
            if items and item not in items:
                continue
            exps = spec["sources"] + spec.get("optional", [])
            fs = {e: files[e] for e in exps if files.get(e)}
            if not all(e in fs for e in spec["sources"]):
                continue
            out["items"][item] = dict(
                files=fs, sha256={e: (sha(f) if isinstance(f, str) else [sha(x) for x in f])
                                  for e, f in fs.items()},
                expect={e: (expect or {}).get(e, {}) for e in fs}, options=(options or {}).get(item, {}))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        return out

    return (write_manifest,)


@app.cell
def _(regenerate, write_manifest):
    def regeneration_selftest():
        """the required failure modes and multi-file sources, on tiny fixtures. Returns (ok, lines)."""
        import tempfile
        lines, ok = [], True
        d = tempfile.mkdtemp()
        rng = np.random.default_rng(3)
        n = 500
        W = dict(conv=rng.random(n) > 0.05, fail=rng.random(n) < 0.04, work=np.ones(n, int),
                 time_us=rng.gamma(4, 5, n))
        S = dict(conv=np.ones(n, bool), fail=rng.random(n) < 0.03, work=np.ones(n, int),
                 time_us=rng.gamma(4, 500, n))
        nf = rng.poisson(0.5, n)
        npz = os.path.join(d, "EXP6_fixture.npz")
        save_zoo(npz, {"peel": W, "OSD-0": S}, nf, nf,
                 dict(code="[[72, 12, 6]]", rounds=12, p=1e-3, shots=n, seed=1, workers=1,
                      x_detectors=False))
        man = os.path.join(d, "thesis_sources.json")

        def run(expect=None, items=("table_5_2",)):
            write_manifest(man, {"E04": "EXP6_fixture.npz"}, expect=expect, items=items)
            return regenerate(man, os.path.join(d, "out"), what=("table",))

        def check(name, cond):
            nonlocal ok
            ok &= bool(cond)
            lines.append(("PASS " if cond else "FAIL ") + name)

        log, code = run({"E04": {"rounds": 12, "p": 1e-3}})
        rows = list(csv.DictReader(open(os.path.join(d, "out", "table_5_2.csv"), encoding="utf-8")))
        pt = {r["policy"]: r for r in rows}
        P = derive(W, S, False, True)
        check("(1) Table 5.2 failures equal derive's; oracle fails on W∧S, escalates W.fail",
              code == 0 and int(pt["primary_fail"]["failures"]) == int(P["fail"].sum())
              and int(pt["oracle"]["failures"]) == int((W["fail"] & S["fail"]).sum())
              and abs(float(pt["oracle"]["escalation"]) - float(W["fail"].mean())) < 1e-12)
        write_manifest(man, {"E04": "EXP6_fixture.npz"}, items=("table_5_2",))
        with open(npz, "ab") as fh:
            fh.write(b"\0")
        log, code = regenerate(man, os.path.join(d, "out"), what=("table",))
        check("(2) a changed byte -> SourceMismatch, exit 1",
              code == 1 and log[0]["status"] == "SourceMismatch")
        os.remove(npz)
        log, code = regenerate(man, os.path.join(d, "out"), what=("table",))
        check("(3) a deleted file -> MissingSource", code == 1 and log[0]["status"] == "missing")
        save_zoo(npz, {"peel": W, "OSD-0": S}, nf, nf,
                 dict(code="[[72, 12, 6]]", rounds=12, p=1e-3, shots=n, seed=1, workers=1,
                      x_detectors=False))
        log, code = run({"E04": {"rounds": 99}})
        check("(4) expect.rounds = 99 -> CaptionMismatch", code == 1 and log[0]["status"] == "CaptionMismatch")
        save_zoo(npz, {"peel": W, "OSD-0": S}, nf, nf,
                 dict(code="[[72, 12, 6]]", rounds=12, p=1e-3, shots=n, seed=1, workers=1))
        log, code = run()
        check("(5) unknown convention -> ProvenanceUnknown",
              code == 1 and log[0]["status"] == "ProvenanceUnknown")
        stamp = start_run("E08-prior-update", "E08")
        base = dict(config=dict(codes=["[[72, 12, 6]]"], rounds_list=[12], ps=[1e-3], shots_per_config=[10],
                                master_seed=1, detector_convention="z-only", decoders={}),
                    timing=dict(workers_requested=1, workers_effective=[1], backends=[], resumed_chunks=0,
                                total_chunks=0), results=[])
        def _res(h, shots=10, dec="OSD-0"):
            return [dict(code="[[72, 12, 6]]", rounds=12, p=1e-3, shard=0, shots=shots, decoder=dec,
                         config_id="x", arms={"updated": dict(failures=0, sample_sha256=h)},
                         paired={}, unpaired={})]
        write_result_file(os.path.join(d, "E08_f.json"),
                          dict(base, results=_res("aaa"),
                               samples={"72-12-6_T12_p0.001_s0_OSD-0": dict(flagged="aaa")}), stamp)
        stamp7 = start_run("E07-frontier", "E07")
        write_result_file(os.path.join(d, "E07_f.json"),
                          dict(base, results=_res("bbb", dec="OSD-CS2"),
                               samples={"72-12-6_T12_p0.001_s0_OSD-CS2": dict(flagged="bbb")}), stamp7)
        write_manifest(man, {"E07": "E07_f.json", "E08": "E08_f.json"}, items=("table_5_9",))
        log, code = regenerate(man, os.path.join(d, "out"), what=("table",))
        check("(6) Table 5.9 and 5.8 with different samples -> CaptionMismatch",
              code == 1 and log[0]["status"] == "CaptionMismatch")
        parts = {}
        for name, pp in (("E08_a.json", 2e-3), ("E08_b.json", 3e-3), ("E08_c.json", 2e-3)):
            run_grid("E08", "E08-prior-update",
                     [dict(code="[[72, 12, 6]]", rounds=2, p=pp, shots=30,
                           arms=["blind", "sighted", "updated"])], 1, path=os.path.join(d, name))
            parts[name] = name
        write_manifest(man, {"E08": ["E08_a.json", "E08_b.json"]}, items=("table_5_8",))
        log, code = regenerate(man, os.path.join(d, "out"), what=("table",))
        rows = list(csv.DictReader(open(os.path.join(d, "out", "table_5_8.csv"), encoding="utf-8"))) \
            if code == 0 else []
        check("(7) an experiment run in two parts regenerates from both files",
              code == 0 and len(rows) == 2 and len(log[0]["sources"]["E08"]) == 2)
        write_manifest(man, {"E08": ["E08_a.json", "E08_c.json"]}, items=("table_5_8",))
        log, code = regenerate(man, os.path.join(d, "out"), what=("table",))
        check("(8) the same configuration in two files -> SourceMismatch",
              code == 1 and log[0]["status"] == "SourceMismatch")
        return ok, lines

    return (regeneration_selftest,)


@app.function
def regenerate_cli(argv, app=None):
    """
    Command line: tables | figures | regenerate | selftest, with --manifest,
    --out, --allow-missing and --only item[,item]. Runs the notebook headless
    to obtain its definitions, then regenerates. Returns the exit code.
    """
    cmd = argv[0]
    opts = dict(manifest=os.path.join(NOTEBOOK_DIR, "thesis_sources.json"),
                out=os.path.join(NOTEBOOK_DIR, "thesis_out"), allow_missing=False, only=None)
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ("--manifest", "--out", "--only"):
            opts[a[2:]] = argv[i + 1]
            i += 2
        elif a == "--allow-missing":
            opts["allow_missing"] = True
            i += 1
        else:
            print(f"unknown option {a}")
            return 2
    if app is None:
        print("regenerate_cli needs the marimo app")
        return 2
    print("running the notebook headless (self-tests included) ...", flush=True)
    _outs, defs = app.run()
    if cmd == "selftest":
        ok, lines = defs["regeneration_selftest"]()
        print("\n".join(lines))
        return 0 if ok else 1
    what = {"tables": ("table",), "figures": ("figure",), "regenerate": ("table", "figure")}[cmd]
    only = set(opts["only"].split(",")) if opts["only"] else None
    log, code = defs["regenerate"](opts["manifest"], opts["out"], what=what,
                                   allow_missing=opts["allow_missing"], only=only)
    for rec in log:
        print(f"{rec['status']:>18}  {rec['item']}" + (f"  — {rec['error']}" if rec.get("error") else ""))
    print(f"exit {code}; outputs in {opts['out']}")
    return code


@app.cell
def _(regeneration_selftest):
    def _from_line(line):
        def _check():
            assert line.startswith("PASS"), line
        return _check

    def _shared_samples_by_shot_count():
        """Experiment 7's 10k p99 point shares a config id with its 50k point; only
        equal shot counts are compared, so the 10k point is skipped, not a mismatch."""
        mk = lambda shots, h, arm: dict(code="c", rounds=12, p=1e-3, shard=0, shots=shots,
                                        arms={arm: dict(sample_sha256=h)})
        e07 = [mk(50000, "A", "updated"), mk(10000, "Z", "sighted")]
        e08 = [dict(mk(50000, "A", "updated"), arms={"updated": dict(sample_sha256="A"),
                                                     "sighted": dict(sample_sha256="A")})]
        assert shared_sample_check(e07, e08) == (1, [])
        assert shared_sample_check([mk(50000, "B", "updated")], e08)[1]
        assert shared_sample_check([mk(20000, "A", "updated")], e08) == (0, [])

    _ok, _lines = regeneration_selftest()
    tests_regeneration = run_checks([(line[5:], _from_line(line)) for line in _lines]
                                    + [("same-sample check compares equal shot counts only",
                                        _shared_samples_by_shot_count)])
    render_checks("18 regeneration self-test", tests_regeneration)
    return


@app.cell
def _():
    _pick = {k: saved_run_picker(k, k)[0] for k in ("E01", "E03")}
    for _k in ("E02", "E07", "E08", "E09", "E10", "E11"):    # grid experiments: one or more files
        _fs = saved_runs(_k)
        _pick[_k] = mo.ui.multiselect({os.path.basename(f): f for f in _fs},
                                      value=[os.path.basename(_fs[-1])] if _fs else [], label=_k)
    _e04 = saved_runs("E04") + sorted(f for f in glob.glob(os.path.join(RESULTS_DIR, "EXP6_*.npz"))
                                      if not is_checkpoint_path(f))
    _pick["E04"] = mo.ui.dropdown({os.path.basename(f): f for f in _e04},
                                  value=os.path.basename(_e04[-1]) if _e04 else None, label="E04")
    _e06 = [f for f in saved_runs("E06") if "_sweep_" in f]
    _pick["E06"] = mo.ui.dropdown({os.path.basename(f): f for f in _e06},
                                  value=os.path.basename(_e06[-1]) if _e06 else None, label="E06")
    regen_files = mo.ui.dictionary(_pick)
    regen_btn = mo.ui.run_button(label="Write thesis_sources.json and regenerate")
    mo.vstack([mo.md("### Regenerate from the notebook\n\nPick the result file of each experiment "
                     "(Experiment 4: the 250k-grid `EXP6_*.npz` or a new grid), then press the button. "
                     "The manifest records each file's sha256; `expect` fields can be added by hand."),
               regen_files, regen_btn])
    return regen_btn, regen_files


@app.cell
def _(regen_btn, regen_files, regenerate, write_manifest):
    mo.stop(not regen_btn.value)
    _files = {k: (os.path.relpath(v, NOTEBOOK_DIR) if isinstance(v, str) else
                  ([os.path.relpath(x, NOTEBOOK_DIR) for x in v] if len(v) > 1 else
                   os.path.relpath(v[0], NOTEBOOK_DIR)))
              for k, v in regen_files.value.items() if v}
    _man = os.path.join(NOTEBOOK_DIR, "thesis_sources.json")
    write_manifest(_man, _files)
    _log, _code = regenerate(_man, os.path.join(NOTEBOOK_DIR, "thesis_out"), allow_missing=True)
    mo.md("| item | status | detail |\n|:--|:--|:--|\n" + "\n".join(
        f"| {r['item']} | {r['status']} | {r.get('error', '')} |" for r in _log)
          + f"\n\nExit code {_code}; outputs in `thesis_out/`.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 19. Results

    Every number below comes from a registered run of this notebook (result
    files of 28–30 October 2026; all complete, none resumed, none scaled, every
    timed run single-threaded with arms in turns) and is regenerated by §18.

    | thesis item | source | status |
    |---|---|---|
    | Table 5.1 noise model and peeling | Experiment 1 | final |
    | Tables 5.2–5.5, Fig. 5.2 | Experiment 4: 250,000-shot grid (counts) + single-thread re-timing (times) | final |
    | Table 5.6 flag rate | Experiment 3 | final |
    | Table 5.7, Fig. 5.3 | Experiment 2 | final |
    | Table 5.8, Fig. 5.4 | Experiment 8 | final |
    | Table 5.9, Fig. 5.5 | Experiments 7 and 8 | final |
    | Fig. 5.1, $p^*$ | Experiment 6 | final for BP+OSD-0; MWPM crossing beyond the grid |
    | w-family | Experiment 9 | final (revised freeze rule, §17) |
    | inference-gap ablations | Experiments 10–12 | final, descriptive |
    | X-basis mirror of A1 | Experiment 8, A1-X | pending (optional, exploratory) |
    | Holm family T1–T14 | §17 | evaluated: 12 of 14 rejected |

    ### The confirmatory family

    | test | claim | data | result |
    |---|---|---|---|
    | T1 | updated < sighted | [[72,12,6]], $10^{-3}$, 50k | 7 vs 254 failures (2 : 249), $p = 2\times10^{-71}$ ✔ |
    | T2 | updated < sighted | [[72,12,6]], $2\times10^{-3}$, 20k | 55 vs 748 (15 : 708), $p = 2\times10^{-187}$ ✔ |
    | T3 | updated < sighted | [[144,12,12]], $10^{-3}$, 150k | 0 vs 27 (0 : 27), $p = 1.5\times10^{-8}$ ✔ |
    | T4 | updated < sighted | [[288,12,18]], $3\times10^{-3}$, 100k | 1 vs 69 (1 : 69), $p = 10^{-19}$ ✔ |
    | T5 | updated < sighted | [[46,2,9]], $10^{-3}$, 50k | 685 vs 5,792, $p < 10^{-300}$ ✔ |
    | T6 | updated < sighted | [[48,6,8]], $10^{-3}$, 50k | 273 vs 3,925, $p < 10^{-300}$ ✔ |
    | T7 | sighted < unflagged | [[72,12,6]], $3\times10^{-3}$, 100k | rate ratio 0.735 [0.717, 0.752] ✔ |
    | T8 | sighted < unflagged | [[144,12,12]], $3\times10^{-3}$, 100k | 0.392 [0.372, 0.413] ✔ |
    | T9 | sighted < unflagged | [[288,12,18]], $3\times10^{-3}$, 100k | 0.437 [0.324, 0.583] ✔ |
    | T10 | gain falls with $w$ | w-family, $7\times10^{-4}$ | slope +0.16 ± 0.11, $p = 0.90$ ✘ |
    | T11 | gain falls with $p$ | [[72,12,6]], [[144,12,12]] | slope −1.90 ± 0.24, df 1, $p = 0.040 > 0.025$ ✘ |
    | T12 | updated < sighted, BP+LSD-0 | [[72,12,6]], $10^{-3}$, 50k | 2 : 249, $p = 2\times10^{-71}$ ✔ (see R7) |
    | T13 | updated < blind, Relay-BP | [[72,12,6]], $2\times10^{-3}$, 20k | 38 vs 117 (16 : 95), ratio 0.33 [0.22, 0.47] ✔ |
    | T14 | updated faster than sighted | [[72,12,6]], $10^{-3}$, A1 | median time ratio 0.437 [0.355, 0.493] ✔ |

    $p$-values shown as $< 10^{-300}$ underflow to 0 in double precision. The
    Holm procedure stops at T11 (0.040 against its threshold 0.05/2), so T11 and
    T10 are not rejected.

    **R1 — flags help any decoder that reads them** *(Experiment 2, T7–T9).*
    At $p = 3\times10^{-3}$ the flag rows read as checks cut the failure rate
    against the circuit without flags to 0.74×, 0.39× and 0.44× on [[72,12,6]],
    [[144,12,12]] and [[288,12,18]]. Building the flags and ignoring them costs
    accuracy: the blind arm fails 1.34×, 1.46× and 3.2× as often as the unflagged
    circuit. Prior updating turns the same bits into an 11×, 119× and 158×
    improvement over the unflagged circuit (1,456 vs 16,091; 43 vs 5,099; 1 vs
    158). On [[46,2,9]] and [[48,6,8]] every arm fails on more than 14% of shots
    at this noise level.

    **R2 — no flag-derived trigger beats "escalate when the primary fails"**
    *(Experiment 4).* On 250,000 paired shots ([[72,12,6]], $T = 12$,
    $p = 10^{-3}$, peeling → BP+OSD-0), `primary_fail` fails on 1,279 shots and no
    flag rule fails on fewer; the oracle that escalates exactly peeling's failures
    fails on 1,216 and escalates a strict subset of `primary_fail`'s shots
    (0 : 63). The single-thread re-timing on 20,000 fresh shots, with 19 decoders,
    reproduces this: `primary_fail` 100 failures at 44.0% escalation, mean cost
    3,238 µs; `always` 105 failures at 5,629 µs; `flag>=1` 105 failures, 91.4%
    escalated, 5,564 µs — 72% *more* expensive than `primary_fail`. Across all
    pairs (Bonferroni m = 616) no flag rule beats `primary_fail` in accuracy. Two
    BP-ms → Relay-BP pairs show a family-corrected saving of 8–18% with
    `flag>=2` or `rounds>=2`, but a rule escalating the same number of *random*
    shots does as well (not flag-informative): the saving comes from skipping
    the primary BP, not from the flags.

    **R3 — why: no headroom, and the trigger is not selective** *(Experiments 3,
    4, 5).* Peeling produced **no silent failure** in any of 35 configurations
    (seven codes × $p \in \{5\times10^{-4}, \ldots, 4\times10^{-3}\}$, 250,000
    shots each; tightest one-sided 95% bound $1.4\times10^{-5}$ per converged
    shot), so no pre-decode trigger can gain accuracy over `primary_fail`. In the
    re-timed grid no non-control pair has headroom above 2. Flags fire on most
    shots: 89.2% at $T = 12$ on [[72,12,6]] (law L6: 89.4%; L6 lies inside the
    Wilson interval in 22 of 25 cells, never more than 0.2 percentage points
    off). "A flag fired" carries 0.018 bits about peeling's failure (risk ratio
    2.44).

    **R4 — switching works without flags** *(Experiment 4, exploratory).*
    Single-threaded, `primary_fail` matches `always` (100 vs 105 failures) at a
    43% lower mean cost and a median of 24.5 µs against 3,655 µs; the accuracy
    edge is by construction (0 : 5), since a converged peeling correction is never
    wrong here.

    **R5 — the noise model reproduces the published theory; our peeler does not
    reach it** *(Experiment 1).* The fault count equals Theorem 1 exactly and
    $\lambda$ is 1.6% above Theorem 2. Measured peeling resolves 0.895, 0.803 and
    0.647 of shots ([[72,12,6]], [[144,12,12]], [[288,12,18]], 100,000 each)
    against Theorems 3/5's 0.937, 0.878 and 0.772 — below even the single-pass
    bound (0.926, 0.857, 0.734). All three peeler variants resolve exactly the same
    shots; the cause is **not isolated**.

    **R6 — prior updating beats the flag rows as checks, and is faster**
    *(Experiments 7, 8; T1–T6, T14).* With BP+OSD-0 the updated arm fails 8.5× to 69×
    less often than the sighted arm on every code and noise level tested, and on
    [[144,12,12]] it had **no failure** in 150,000 shots at $10^{-3}$ or 50,000 at
    $2\times10^{-3}$ (sighted: 27 and 141). It is also faster, with the update
    inside the clock: on [[72,12,6]] a median of 1,280 µs (386 µs of it the
    update) against 2,932 µs; on [[144,12,12]] 2,998 µs against 13,401 µs. On the
    accuracy–latency frontier (Table 5.9) the updated arm costs 16% more median
    time than the unflagged circuit's decode on [[72,12,6]] for 80× fewer
    failures, and on [[144,12,12]] it is both faster and better than the
    unflagged decode (2,998 vs 5,696 µs; 0 vs 166 failures). OSD-CS2 adds nothing
    to the updated arm (7 failures either way) and raises its p99 eighteen-fold.

    **R7 — the gap is one of inference, not information** *(Experiments 10–12,
    descriptive).* Both arms see the same bits. On the same 50,000 shots of
    [[72,12,6]] at $10^{-3}$, the sighted arm's failures fall from 254 with
    min-sum BP+OSD-0 to 75 with sum-product BP+OSD-0 and to **6 with Relay-BP**,
    while the updated arm stays at 7, 8 and 7. Under Relay-BP the two arms are
    indistinguishable (6 vs 7; 44 vs 38 at $2\times10^{-3}$, $p = 0.45$), and the
    updated arm keeps only its time advantage (5.1 vs 8.9 ms median). BP+LSD-0
    fails on exactly the same shots as BP+OSD-0 at $10^{-3}$, so T12 repeats T1
    rather than testing OSD-independence. Short cycles do not explain the gap: a
    flag row lies on 33 four-cycles, a detector row on about 460 (BB codes).

    This answers the comparison Takada and Fujii anticipated. Their expectation
    that flags as extra parity checks reach lower error rates than
    flag-conditioned weights is borne out *once the decoder is good enough*
    (Relay-BP); with min-sum BP+OSD, the decoder class most often used for BB
    codes, it fails by up to 36×. The closed-form update makes a weak decoder
    perform like a strong one on these bits, at lower cost, with any decoder that
    accepts per-fault probabilities.

    **R8 — the check-weight trend is code-dependent, not monotone** *(Experiment 9,
    T10).* At $p = 7\times10^{-4}$ the gain blind/updated is 1.2, 7.6, 59, 17, 18,
    12, 19.5, 14 and 6.5 for $w = 4 \ldots 12$; the registered slope is positive
    (+0.16 ± 0.11). The $w = 4$ code ($d \le 3$), which flags barely help, carries
    the largest weight; without it (exploratory) the slope is −0.13 ± 0.09
    ($p = 0.09$). The $w = 6$ gain is inflated by an unusually poor blind arm
    (blind LER 0.26 against 0.0096 for its unflagged circuit). Laws L1–L7 hold on
    every member (§11); the decoding gain depends at least as much on the
    individual code as on $w$.

    **R9 — the noise-dependence of the gain** *(T11, not confirmed).* On
    [[72,12,6]] the gain rises from 15× at $3\times10^{-3}$ to 32× at
    $2\times10^{-3}$ and 125× at $10^{-3}$ (slope −1.9 in $\ln p$). The registered
    test drops points with zero updated failures, and [[144,12,12]] had none at
    $10^{-3}$ and $2\times10^{-3}$, leaving one point for that code and one degree
    of freedom; the trend is therefore reported as exploratory.

    **R10 — pipeline validation** *(Experiment 6).* On the rotated surface code
    BP+OSD-0 crosses at $p^* = 1.13\%$ [1.05%, 1.19%] ($d_5/d_7$; 1.08% for
    $d_3/d_5$, 1.11% for $d_3/d_7$), and at $10^{-3}$ $d = 7$ beats $d = 5$ with
    non-overlapping intervals (5 vs 28 failures in 200,000 shots). MWPM's
    $d_5/d_7$ curves do not cross below 1.2% (no crossing in 78% of bootstrap
    resamples), so its threshold lies above the measured range. The two decoders
    are indistinguishable (exact McNemar) up to $3\times10^{-3}$; above that MWPM
    is better (e.g. 175 vs 217 failures at $d = 7$, $5\times10^{-3}$).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion and limitations

    **Why the trigger fails.** A useful pre-decode trigger needs a primary that
    fails *silently* — failures it reports itself are already caught by
    `primary_fail` — and a selective signal. Peeling produced no silent failure in
    8.75 million shots across seven codes and five noise levels, and "any flag
    fired" is true on most shots and nearly uninformative. The only cost savings
    found come from skipping the primary decoder on a fraction of shots, which a
    random rule achieves as well.

    **Why prior updating wins, and what that means.** The flag outcomes are
    informative in aggregate — every decoder that reads them improves (R1) — but
    how much a decoder extracts depends on its inference, not on the bits (R7).
    Min-sum BP+OSD handed the flag rows as checks recovers only part of the
    information; exact sum-product recovers more; Relay-BP recovers all of it. The
    closed-form update needs no such capability: it folds each flag in once,
    exactly, before decoding, so min-sum BP+OSD-0 with the update matches
    Relay-BP with flags at $10^{-3}$ (7 against 6–7 failures) and comes close at
    $2\times10^{-3}$ (55 against 38–44), at about a quarter of Relay-BP's decode
    time. Takada and
    Fujii's expectation that extra parity checks should do better is therefore
    right for a strong decoder and wrong for the decoders in common use; the
    practical value of the update is that it makes the result independent of the
    decoder's inference quality.

    **Limitations.**

    1. The experiment circuit has 1.42× the fault mechanisms of Theorem 1's
       count because it extracts both check families (the flags sit on the
       X-check ancillas). Absolute rates are not comparable with published BB
       figures; the reference circuit (§5.1) is used whenever they must be.
    2. Comparisons across circuits (flagged vs unflagged) are unpaired by
       construction and use exact rate-ratio intervals; only arms on one
       circuit are compared shot by shot.
    3. Timing is Python-level CPU time on one machine. Latency claims use
       single-threaded runs only. In Experiment 4's grid the decoders run in a
       fixed order within each chunk; there, OSD-0 and OSD-CS2 differ by about 30%
       in median time on shots where they do identical work, so grid timings
       carry an uncertainty of that order. The four-arm runs (Experiments 2,
       7–12) rotate the arm order and are not affected.
    4. T11 as registered discards points with zero updated failures. The updated
       arm was so strong on [[144,12,12]] (0 failures at $10^{-3}$ and
       $2\times10^{-3}$) that the test was left with one degree of freedom; the
       noise-dependence of the gain is reported as exploratory.
    5. Experiment 9's registered freeze rule was infeasible on the w-family and
       was revised before the main run (§17); the w-family codes also differ in
       distance and rate, which confounds any trend in $w$ (R8).
    6. BP+OSD-0 is a weak decoder for [[46,2,9]] (its unflagged failure rate
       drops from 23.4% to 6.8% with OSD-CS4), and on [[48,6,8]] and GB24 w=4
       the sighted arm fails more often than the blind arm. Gains on those codes
       are relative to a weak baseline.
    7. The surface-code grid stops at $p = 1.2\%$, below MWPM's threshold.
    8. Stim's sampler is not guaranteed bit-identical across versions or
       machines; every result stores its sample hash and a Stim self-check
       digest, and cited runs keep a packed sample sidecar.
    9. Every confirmatory run is a Z-basis memory with flags on the X checks
       only. By the BB codes' X↔Z symmetry the X-basis memory with flags on the
       Z checks has the same flag structure (checked in §4); Experiment 8's A1-X
       tests it on A1's configuration. A memory protecting both bases needs flags
       on both families, which doubles the flag overhead and adds faults that the
       other basis's decoder cannot use; that cost is not measured here. No idle
       noise is modelled.
    10. The prior update is lossy by construction: conditioning on a flag leaves
        its group's faults correlated, and a prior vector keeps only their
        marginals (§11). It is exact per fault, not as a joint model.
    11. The posterior is exact for the DEM's noise model, not for a real device.
        Per-mechanism probabilities estimated from device data could be
        conditioned on the flags in the same closed form; this is not tested here.
    12. Pre-registration evidence: the confirmatory family was fixed in this
        notebook and in a dated project document on 28 October 2026, before
        sampling. The registration record file in `results/` was written later
        (30 October) and is not itself evidence of the timing.

    ## Conclusion and future work

    Flag information is useful to a decoder and not useful as a routing signal.
    As a pre-decode trigger it cannot beat the primary decoder's own convergence
    flag: there is nothing silent to catch, and the trigger is not selective. As
    decoder input it pays on every code tested — read as extra checks it lowers
    the failure rate by up to 2.6×, and turned into exact per-shot priors by
    Gallager's posterior it lowers it by one to two orders of magnitude while
    making the decode faster. The advantage of the prior update over the flag
    rows is an inference gap of min-sum BP+OSD, which Relay-BP closes; the update
    delivers that accuracy with a cheap decoder. Twelve of the fourteen
    pre-registered tests are confirmed; the predicted decline of the gain with
    check weight is not, and its decline with noise is seen but not confirmed.

    **Future work.** Priors estimated from device data, conditioned on the flags
    in closed form; flag-matrix column weight > 1 (overlapping groups, as in
    Takada and Fujii's circuits), where the closed form no longer applies per
    flag; the noise-dependence of the gain with enough shots for nonzero
    updated failures on [[144,12,12]]; a w-family with matched distance; the
    surface-code sweep extended above 1.2% for MWPM; isolate the peeler's residual
    against the single-pass bound; graded flag signals (counts, clustering) as
    decoder input; comparison with other hook-error mitigations at equal $p$.
    """)
    return


@app.cell(hide_code=True)
def _():
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
    9. R. Toshio et al., "Decoder switching: breaking the speed-accuracy tradeoff
       in real-time quantum error correction," arXiv:2510.25222 (2025).
    10. T. Hillmann et al., "Localized statistics decoding: a parallel decoding
        algorithm for quantum LDPC codes," *Nat. Commun.* **16**, 8214 (2025).
    11. T. Müller et al., "Improved belief propagation is sufficient for real-time
        decoding of quantum memory" (Relay-BP), arXiv:2506.01779 (2025).
    12. O. Higgott and C. Gidney, "Sparse Blossom: correcting a million errors per
        core second with minimum-weight matching," *Quantum* **9** (2025),
        arXiv:2303.15933 (pymatching).
    13. S. Vittal, A. Javadi-Abhari, A. W. Cross, L. S. Bishop and M. Qureshi,
        "Flag proxy networks: tackling the architectural, scheduling, and decoding
        obstacles of quantum LDPC codes," arXiv:2409.14283 (2024).
    14. R. G. Gallager, "Low-density parity-check codes," *IRE Trans. Inf. Theory*
        **8**, 21 (1962); *Low-Density Parity-Check Codes* (MIT Press, 1963).
    15. D. J. C. MacKay and R. M. Neal, "Near Shannon limit performance of low
        density parity check codes," *Electron. Lett.* **32**, 1645 (1996).
    16. M. Matsui, "Linear cryptanalysis method for DES cipher," *EUROCRYPT '93*,
        LNCS **765** (Springer, 1994) — the piling-up lemma.
    17. A. Gong, S. Cammerer and J. M. Renes, "Graph neural networks for enhanced
        decoding of quantum LDPC codes," *Proc. IEEE ISIT* (2024), arXiv:2310.17758.
    18. J. Chen, Z. Yi, Z. Liang and X. Wang, "Improved belief propagation
        decoding algorithms for surface codes," *IEEE Trans. Quantum Eng.* (2025),
        arXiv:2407.11523.
    19. Y. Takada and K. Fujii, "Improving threshold for fault-tolerant color code
        quantum computing by flagged weight optimization," *PRX Quantum* **5**,
        030352 (2024), arXiv:2402.13958.
    20. B. Anker and M. Marvian, "Flag gadgets based on classical codes,"
        arXiv:2212.10738 (2022).
    21. S. Holm, "A simple sequentially rejective multiple test procedure,"
        *Scand. J. Stat.* **6**, 65 (1979).
    """)
    return


if __name__ == "__main__":
    import sys as _sys

    # `python FlaggedDecoding.py tables|figures|regenerate|selftest [options]`
    # regenerates the thesis tables and figures from saved result files
    # without opening the notebook; with no arguments the app runs as usual.
    if len(_sys.argv) > 1 and _sys.argv[1] in REGENERATION_COMMANDS:
        _sys.exit(regenerate_cli(_sys.argv[1:], app))
    app.run()
