"""Reproduce every number in the paper. Writes results/*.json.

  python benchmarks/run_experiments.py counterexamples
  python benchmarks/run_experiments.py synthetic   [--seeds 10]
  python benchmarks/run_experiments.py ann         [--seeds 5]
  python benchmarks/run_experiments.py digits      [--seeds 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from emeans_plus import reference_1d as r1  # noqa: E402
from emeans_plus.core import (  # noqa: E402
    assign_nearest, compute_loss, emeans, kmeanspp_init, lloyd, parameters, sq_dists)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", name)


# --------------------------------------------------------------------------
# Counterexamples (1-D reference recipe)
# --------------------------------------------------------------------------
VARIANTS = {  # name: (tau, revival, safety)   C = check-and-fall-back, S = polish + check
    "e-means": (1e-10, "mass", False),
    "e-means + C": (1e-10, "mass", "check"),
    "e-means + S": (1e-10, "mass", "polish"),
    "tau": (1.0, "mass", False),
    "tau + C (e-means+)": (1.0, "mass", "check"),
    "tau + S": (1.0, "mass", "polish"),
    "guard": (1e-10, "guarded", False),
    "tau + guard + S": (1.0, "guarded", "polish"),
}


def run1d(x, seeds, T, B, a, var, seed=0, prefix=()):
    tau, rev, safe = VARIANTS[var]
    c, _, ev = r1.emeans(x, seeds, T, B, a, revival=rev, tau=tau, seed=seed, prefix=prefix)
    if safe:
        c = r1.safe(x, seeds, c, polish=safe == "polish")
    return c, ev


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def counterexamples(trials=20000):
    res = {}
    # ---- Counterexample 1: revival (deterministic: B = n) ----
    ce1 = []
    for m in (10, 20, 50):
        D = m + 4
        x = [1] * m + [3] * m + [D]
        T = 10 * len(x)
        B, a = r1.parameters(len(x), T)
        probs = r1.kpp_probabilities(x)
        row = {"m": m, "D": D, "B": B, "alpha0": a, "lloyd_seed": r1.loss(x, r1.lloyd(x, (1, D)))}
        exp_l = sum(p * Fraction(round(r1.loss(x, r1.lloyd(x, s)))) for s, p in probs.items())
        row["lloyd_kpp"] = float(exp_l)
        for v in VARIANTS:
            c, _ = run1d(x, (1, D), T, B, a, v)
            row[v + "|seed"] = r1.loss(x, c)
            e = Fraction(0)
            for s, p in probs.items():
                cs, _ = run1d(x, s, T, B, a, v)
                e += p * Fraction(round(r1.loss(x, cs), 9)).limit_denominator(10**6)
            row[v + "|kpp"] = float(e)
        ce1.append(row)
    res["ce1"] = ce1

    # ---- Counterexample 2: forgetting ----
    X = [0, 0, 15, 29, 29, 29, 29, 29]
    T = 25
    B, a = r1.parameters(len(X), T)
    tau_star = 2 * a / (14 * (1 - a))
    res["ce2_params"] = {"B": B, "alpha0": a, "tau_star": tau_star,
                         "lloyd": r1.loss(X, r1.lloyd(X, (15, 29))),
                         "seed_loss": r1.loss(X, (15, 29))}
    prefix = [[15, 29, 29, 29, 0, 0, 29, 29], [0, 0, 29, 29, 15, 29, 29, 29]]
    res["ce2_prescribed"] = {}
    for v in VARIANTS:
        c, _ = run1d(X, (15, 29), T, B, a, v, prefix=prefix)
        res["ce2_prescribed"][v] = r1.loss(X, c)

    def fail_rate(var, tau=None):
        t0, rev, safe = VARIANTS[var]
        tau = t0 if tau is None else tau
        fails, losses = 0, []
        for s in range(trials):
            c, _, _ = r1.emeans(X, (15, 29), T, B, a, revival=rev, tau=tau, seed=s)
            if safe:
                c = r1.safe(X, (15, 29), c, polish=safe == "polish")
            L = r1.loss(X, c)
            losses.append(L)
            fails += L > 150 + 1e-9
        lo, hi = wilson(fails, trials)
        return {"fail": fails / trials, "ci": [lo, hi], "mean_loss": float(np.mean(losses)),
                "trials": trials}

    res["ce2_variants"] = {v: fail_rate(v) for v in VARIANTS}
    taus = [1e-10, 0.05, 0.1, 0.15, 0.18, 0.19, 0.191, 0.192, 0.1925, 0.193, 0.194, 0.195, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0]
    res["ce2_tau_sweep"] = [{"tau": t, **fail_rate("e-means", tau=t)} for t in taus]

    # ---- Counterexample 2 under k-means++ seeding (exact seed probabilities) ----
    probs = r1.kpp_probabilities(X)
    kpp = {"lloyd": float(sum(p * Fraction(r1.loss(X, r1.lloyd(X, s))).limit_denominator(10**6)
                              for s, p in probs.items()))}
    per = max(1, trials // 10)
    for v in VARIANTS:
        e = 0.0
        for s, p in probs.items():
            L = [r1.loss(X, run1d(X, s, T, B, a, v, seed=q)[0]) for q in range(per)]
            e += float(p) * float(np.mean(L))
        kpp[v] = e
    kpp["shuffles_per_seed_pair"] = per
    res["ce2_kpp"] = kpp
    dump("counterexamples.json", res)


# --------------------------------------------------------------------------
# Multi-dimensional benchmarks
# --------------------------------------------------------------------------
METHODS = {
    # S = safety net (Lloyd polish to convergence + fall back to Lloyd(C0))
    "e-means": dict(tau=1e-10, revival="mass", safety=False),
    "e-means + C": dict(tau=1e-10, revival="mass", safety="check"),
    "e-means + S": dict(tau=1e-10, revival="mass", safety="polish"),
    "tau + C (e-means+)": dict(tau=1.0, revival="mass", safety="check"),
    "tau + S": dict(tau=1.0, revival="mass", safety="polish"),
    "guard + S": dict(tau=1e-10, revival="guarded", safety="polish"),
    "tau + no revival + S": dict(tau=1.0, revival="off", safety="polish"),
    "tau + guard + S": dict(tau=1.0, revival="guarded", safety="polish"),
}
ANN_METHODS = [m for m in METHODS if m != "tau + no revival + S"]


def evaluate_recall(xb, xq, gt, C, nprobes=(1, 4, 16), top_k=10):
    k = len(C)
    lab, _ = assign_nearest(xb, C)
    inv = [np.where(lab == j)[0] for j in range(k)]
    order = np.argsort(sq_dists(xq, C), axis=1)
    out = {}
    for p in nprobes:
        hits = 0
        for i in range(len(xq)):
            ids = np.concatenate([inv[c] for c in order[i, :p]])
            if len(ids) == 0:
                continue
            dd = np.sum((xb[ids] - xq[i]) ** 2, axis=1)
            kk = min(top_k, len(ids))
            best = ids[np.argpartition(dd, kk - 1)[:kk]]
            hits += len(np.intersect1d(best, gt[i, :top_k]))
        out[str(p)] = hits / (len(xq) * top_k)
    return out


def exact_knn(xb, xq, k=10, chunk=256):
    gt = np.empty((len(xq), k), dtype=np.int64)
    for s in range(0, len(xq), chunk):
        d = sq_dists(xq[s:s + chunk], xb)
        p = np.argpartition(d, k - 1, axis=1)[:, :k]
        o = np.argsort(np.take_along_axis(d, p, 1), axis=1)
        gt[s:s + chunk] = np.take_along_axis(p, o, 1)
    return gt


def run_suite(name, X, k, seeds, T=2, B=None, alpha0=None, recall=None, audit=True,
              methods=None, lloyd_baselines=True, verbose=False):
    methods = methods or list(METHODS)
    rows = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        C0 = kmeanspp_init(X, k, rng)
        phi0 = compute_loss(X, C0)
        rec = {"seed": s, "seed_loss": phi0}

        def record(label, C, t, passes, extra=None):
            lab, mins = assign_nearest(X, C)
            d = {"loss": float(mins.sum()), "time": t, "passes": passes,
                 "dead": int(np.sum(np.bincount(lab, minlength=k) == 0))}
            if recall is not None:
                d["recall"] = evaluate_recall(recall[0], recall[1], recall[2], C)
            if extra:
                d.update(extra)
            rec[label] = d

        if lloyd_baselines:
            t = time.perf_counter()
            C, it, conv = lloyd(X, C0, 25)
            record("Lloyd-25", C, time.perf_counter() - t, it + 1, {"converged": conv})
            if verbose:
                print(f"  seed {s} Lloyd-25: {time.perf_counter() - t:.0f}s", flush=True)
            t = time.perf_counter()
            C, it, conv = lloyd(X, C0, 300)
            record("Lloyd (to convergence)", C, time.perf_counter() - t, it + 1, {"converged": conv})
            t = time.perf_counter()
            best, bl, tot = None, np.inf, 0
            for q in range(3):
                Cq = C0 if q == 0 else kmeanspp_init(X, k, rng)
                Cq, it, _ = lloyd(X, Cq, 25)
                tot += it + 1
                L = compute_loss(X, Cq)
                if L < bl:
                    best, bl = Cq, L
            record("Lloyd-25, best of 3 seeds", best, time.perf_counter() - t, tot)
        for mname in methods:
            kw = METHODS[mname]
            t = time.perf_counter()
            C, info = emeans(X, C0, T=T, B=B, alpha0=alpha0, seed=s,
                             audit=audit and kw["revival"] == "guarded", **kw)
            extra = {"converged": info.converged, "fell_back": info.fell_back,
                     "polish_iters": info.polish_iters,
                     "swaps_accepted": info.swaps_accepted, "swaps_rejected": info.swaps_rejected,
                     "swap_full_deltas": info.swap_full_deltas}
            record(mname, C, time.perf_counter() - t, info.passes, extra)
            if verbose:
                print(f"  seed {s} {mname}: loss {rec[mname]['loss']:.1f} "
                      f"({time.perf_counter() - t:.0f}s)", flush=True)
        rows.append(rec)
        if verbose:   # save after every seed so an interrupted run keeps its data
            Bv, Av = parameters(len(X), T, k)
            dump(f"{name}.json", {"meta": {"n": len(X), "d": X.shape[1], "k": k, "T": T,
                                           "B": B or Bv, "alpha0": alpha0 or Av,
                                           "seeds": len(rows), "partial": s + 1 < seeds},
                                  "runs": rows})
        print(name, s, {m: round(v["loss"], 1) for m, v in rec.items() if isinstance(v, dict)},
              flush=True)
    Bv, Av = parameters(len(X), T, k)
    meta = {"n": len(X), "d": X.shape[1], "k": k, "T": T,
            "B": B or Bv, "alpha0": alpha0 or Av, "seeds": seeds}
    dump(f"{name}.json", {"meta": meta, "runs": rows})


SIFT_URLS = ["ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"]


def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32).astype(np.float64)


def read_ivecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def load_sift(sift_dir):
    """SIFT1M (Jegou et al., TEXMEX corpus). Downloads sift.tar.gz (~160 MB) if needed."""
    import tarfile
    import urllib.request
    files = ["sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs"]

    def find():
        for root in (sift_dir, os.path.join(sift_dir, "sift")):
            if all(os.path.exists(os.path.join(root, f)) for f in files):
                return root
        return None

    root = find()
    if root is None:
        os.makedirs(sift_dir, exist_ok=True)
        tgz = os.path.join(sift_dir, "sift.tar.gz")
        if not os.path.exists(tgz):
            for url in SIFT_URLS:
                print(f"downloading {url} (about 160 MB) ...", flush=True)
                try:
                    urllib.request.urlretrieve(url, tgz)
                    break
                except Exception as e:  # noqa: BLE001
                    print("  failed:", e)
            else:
                sys.exit("Download failed. Fetch sift.tar.gz from http://corpus-texmex.irisa.fr/ "
                         f"by hand, put it in {sift_dir}, and run again.")
        with tarfile.open(tgz) as tf:
            tf.extractall(sift_dir)
        root = find()
    xb = read_fvecs(os.path.join(root, files[0]))
    xq = read_fvecs(os.path.join(root, files[1]))
    gt = read_ivecs(os.path.join(root, files[2]))
    return xb, xq, gt


SIFT_METHODS = ["e-means", "e-means + C", "tau + C (e-means+)", "tau + S", "tau + guard + S"]


def make_rare_tail(seed=7):
    r = np.random.default_rng(seed)
    dense = r.normal(0, 0.2, (4950, 8)) + r.uniform(-5, 5, (45, 8))[r.integers(0, 45, 4950)]
    tail = np.repeat(np.eye(5, 8) * 40 + 20, 10, axis=0) + r.normal(0, 0.05, (50, 8))
    return np.vstack([dense, tail])


def make_heavy_tailed(n=50_000, nq=1000, d=128, modes=256, seed=42):
    r = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, modes + 1) ** 1.15
    w /= w.sum()
    cen = r.normal(0, 1.5, (modes, d))
    cen[int(0.9 * modes):] *= 3.5
    xb = cen[r.choice(modes, n, p=w)] + r.normal(0, 0.35, (n, d))
    qm = np.concatenate([r.choice(modes, nq // 2, p=w), r.integers(int(0.9 * modes), modes, nq - nq // 2)])
    xq = cen[qm] + r.normal(0, 0.35, (nq, d))
    return xb, xq


def tail_mechanism(seeds=10):
    """Rare tail: do seeds cover the tail, and does mass revival uncover it?"""
    X = make_rare_tail()
    tail = X[-50:]
    rows = []
    for s in range(seeds):
        C0 = kmeanspp_init(X, 50, np.random.default_rng(1000 + s))
        base = compute_loss(X, lloyd(X, C0, 25)[0])
        row = {"seed": s, "seed_tail_max": float(np.sqrt(assign_nearest(tail, C0)[1].max()))}
        for rev in ("mass", "off"):
            C, _ = emeans(X, C0, T=2, revival=rev, seed=s)
            row[rev] = {"ratio": compute_loss(X, C) / base,
                        "tail_max": float(np.sqrt(assign_nearest(tail, C)[1].max()))}
        rows.append(row)
        print(row, flush=True)
    dump("rare_tail_mechanism.json", rows)


def port_agreement(trials=2000):
    """NumPy recipe vs 1-D reference port on Counterexample 2 (failure rates)."""
    X = [0, 0, 15, 29, 29, 29, 29, 29]
    Xn = np.array(X, float)[:, None]
    C0 = np.array([[15.0], [29.0]])
    B, a = r1.parameters(8, 25)
    out = {}
    for tau in (1e-10, 1.0):
        f1 = sum(r1.loss(X, r1.emeans(X, (15, 29), 25, B, a, tau=tau, seed=s)[0]) > 150 + 1e-9
                 for s in range(trials))
        fn = sum(compute_loss(Xn, emeans(Xn, C0, T=25, tau=tau, seed=s)[0]) > 150 + 1e-9
                 for s in range(trials))
        out[str(tau)] = {"reference": f1 / trials, "numpy": fn / trials,
                         "ci_reference": wilson(f1, trials), "ci_numpy": wilson(fn, trials)}
    out["trials"] = trials
    dump("port_agreement.json", out)


def tau_sensitivity(seeds=10, taus=(1e-10, 0.05, 0.25, 1.0, 4.0)):
    from sklearn.datasets import load_digits
    dig = load_digits().data.astype(np.float64)
    sets = {"rare_tail": (make_rare_tail(), 50), "digits_k10": (dig, 10), "digits_k64": (dig, 64)}
    out = {}
    for name, (X, k) in sets.items():
        rows = []
        for tau in taus:
            L, fb = [], 0
            for s in range(seeds):
                C0 = kmeanspp_init(X, k, np.random.default_rng(1000 + s))
                C, info = emeans(X, C0, T=2, tau=tau, revival="mass", safety="check", seed=s)
                L.append(compute_loss(X, C))
                fb += info.fell_back
            rows.append({"tau": tau, "mean": float(np.mean(L)), "sd": float(np.std(L, ddof=1)),
                         "fell_back": fb})
            print(name, tau, rows[-1], flush=True)
        out[name] = rows
    dump("tau_sensitivity.json", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["counterexamples", "synthetic", "ann", "digits", "tau", "port", "tail",
                                    "sift"])
    ap.add_argument("--sift-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "sift"))
    ap.add_argument("--train-n", type=int, default=1_000_000, help="SIFT: training vectors")
    ap.add_argument("--k", type=int, default=1024, help="SIFT: clusters")
    ap.add_argument("--queries", type=int, default=1000, help="SIFT: queries for recall")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--trials", type=int, default=20000)
    a = ap.parse_args()
    if a.what == "counterexamples":
        counterexamples(a.trials)
    elif a.what == "synthetic":
        X = make_rare_tail()
        run_suite("rare_tail", X, 50, a.seeds or 10)
        B32 = 32
        a32 = min(1.0, 0.824 * 50**-0.46 * B32**0.9 * len(X)**-0.45 * 2**-0.11)
        run_suite("rare_tail_B32", X, 50, a.seeds or 10, B=B32, alpha0=a32)
    elif a.what == "ann":
        xb, xq = make_heavy_tailed()
        gt = exact_knn(xb, xq)
        run_suite("heavy_tailed_ann", xb, 256, a.seeds or 5, recall=(xb, xq, gt),
                  methods=ANN_METHODS)
    elif a.what == "sift":
        xb, xq, gt = load_sift(a.sift_dir)
        X = xb[:a.train_n]
        print(f"SIFT: base {xb.shape}, training on {len(X)}, k={a.k}, "
              f"{a.queries} queries, {a.seeds or 3} seeds", flush=True)
        run_suite("sift1m", X, a.k, a.seeds or 3, recall=(xb, xq[:a.queries], gt[:a.queries, :10]),
                  methods=SIFT_METHODS, audit=False, verbose=True)
    elif a.what == "tail":
        tail_mechanism()
    elif a.what == "port":
        port_agreement()
    elif a.what == "tau":
        tau_sensitivity(a.seeds or 10)
    elif a.what == "digits":
        from sklearn.datasets import load_digits
        X = load_digits().data.astype(np.float64)
        run_suite("digits_k10", X, 10, a.seeds or 10)
        run_suite("digits_k64", X, 64, a.seeds or 10)
