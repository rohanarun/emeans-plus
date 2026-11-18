"""NumPy implementation of e-means (reference recipe) and e-means+.

e-means baseline: EMA counts/sums with cosine-annealed step, mass-based
revival every 5 steps (share < 0.1/k, donor split by (1 +/- 1/1024)), and
last-epoch means as the output, following Bessa et al. (2026) as reproduced
in Assylbekov's reference code. Batch size and peak step follow the printed
laws (see ``parameters``). This is a NumPy re-implementation for study, not
the Faiss ``contrib/e_means`` code.

e-means+ adds three independent switches:
  tau       initial pseudo-count N_j(0) (baseline: 1e-10)
  revival   "mass" | "guarded" | "off"
  safety    "check": if the output is worse than the seeds C0, return
            Lloyd(C0) instead (two loss evaluations, no polish);
            True/"polish": Lloyd polish to convergence, then the same check.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import log2
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")


def parameters(n: int, T: int, k: int) -> tuple[int, float]:
    """Printed batch-size and peak-step laws of the e-means recipe."""
    raw = 0.035 * k**0.13 * n**0.88 * T**0.81
    B = min(n, 2 ** max(0, round(log2(raw))))
    alpha0 = min(1.0, 0.824 * k**-0.46 * B**0.90 * n**-0.45 * T**-0.11)
    return int(B), float(alpha0)


def sq_dists(A: np.ndarray, B: np.ndarray, a_norm: Optional[np.ndarray] = None) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if a_norm is None:
            a_norm = np.sum(A * A, axis=1, keepdims=True)
        d = a_norm + np.sum(B * B, axis=1) - 2.0 * (A @ B.T)
    np.maximum(d, 0.0, out=d)
    return d


def assign_nearest(X: np.ndarray, C: np.ndarray, chunk: int = 16384):
    n = len(X)
    labels = np.empty(n, dtype=np.int64)
    mins = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        d = sq_dists(X[s:s + chunk], C)
        lb = np.argmin(d, axis=1)
        labels[s:s + chunk] = lb
        mins[s:s + chunk] = d[np.arange(len(lb)), lb]
    return labels, mins


def compute_loss(X: np.ndarray, C: np.ndarray) -> float:
    return float(np.sum(assign_nearest(X, C)[1]))


def kmeanspp_init(X: np.ndarray, k: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n, d = X.shape
    C = np.empty((k, d), dtype=np.float64)
    C[0] = X[int(rng.integers(0, n))]
    closest = np.sum((X - C[0]) ** 2, axis=1, dtype=np.float64)
    for c in range(1, k):
        tot = float(closest.sum())
        idx = int(rng.integers(0, n)) if tot <= 0 else int(rng.choice(n, p=closest / tot))
        C[c] = X[idx]
        closest = np.minimum(closest, np.sum((X - C[c]) ** 2, axis=1, dtype=np.float64))
    return C


def lloyd(X: np.ndarray, C_init: np.ndarray, max_iter: int = 25):
    """Full-batch Lloyd. Returns (C, iterations, converged).

    converged=True means the last assignment equals the previous one, so C is
    the mean of every non-empty cell of its own Voronoi partition.
    """
    C = np.array(C_init, dtype=np.float64, copy=True)
    k = len(C)
    prev = None
    for it in range(1, max_iter + 1):
        labels, _ = assign_nearest(X, C)
        if prev is not None and np.array_equal(labels, prev):
            return C, it - 1, True
        prev = labels
        cnt = np.bincount(labels, minlength=k)
        sums = np.zeros_like(C)
        np.add.at(sums, labels, X)
        nz = cnt > 0
        C[nz] = sums[nz] / cnt[nz, None]
    labels, _ = assign_nearest(X, C)
    return C, max_iter, bool(np.array_equal(labels, prev))


@dataclass
class RunInfo:
    passes: float = 0.0               # full-data passes (epochs + Lloyd iterations + loss checks)
    polish_iters: int = 0
    converged: bool = False
    fell_back: bool = False
    swaps_accepted: int = 0
    swaps_rejected: int = 0
    swap_full_deltas: list = field(default_factory=list)  # Phi(after)-Phi(before), audit only


def _guarded_revival(X_ref, x_ref_norm, C, N, S, cooldown_until, step, cooldown_steps,
                     rng, info: RunInfo, eps: float, audit_X=None):
    """Cost-guarded revival: a low-mass centre j is replaced by a D^2-sampled
    reservoir point x' only if the k-means loss on the reservoir drops by more
    than eps * Phi(reservoir)."""
    k = len(C)
    tot = float(N.sum())
    if tot <= 0 or k < 2 or len(X_ref) < 2:
        return
    cand = np.where((N / tot < 0.1 / k) & (step >= cooldown_until))[0]
    if len(cand) == 0:
        return
    d = sq_dists(X_ref, C, a_norm=x_ref_norm)
    rows = np.arange(len(X_ref))

    def top2(d):
        t2 = np.argpartition(d, kth=1, axis=1)[:, :2]
        a, b = d[rows, t2[:, 0]], d[rows, t2[:, 1]]
        sw = a > b
        return (np.where(sw, t2[:, 1], t2[:, 0]), np.where(sw, b, a), np.where(sw, a, b))

    c1, d1, d2 = top2(d)
    for j in cand:
        phi = float(d1.sum())
        if phi <= 1e-12:
            break
        in_j = c1 == j
        cost = float(np.sum(d2[in_j] - d1[in_j]))
        new = X_ref[int(rng.choice(len(X_ref), p=d1 / phi))]
        nd = x_ref_norm[:, 0] + float(new @ new) - 2.0 * (X_ref @ new)
        np.maximum(nd, 0.0, out=nd)
        base = np.where(in_j, d2, d1)
        gain = float(np.sum(np.maximum(0.0, base - nd)))
        cooldown_until[j] = step + cooldown_steps
        if gain - cost > eps * phi:
            if audit_X is not None:
                before = compute_loss(audit_X, C)
            donor = int(np.argmax(N))
            C[j] = new
            N[donor] *= 0.5
            N[j] = N[donor]
            S[donor] = N[donor] * C[donor]
            S[j] = N[j] * C[j]
            if audit_X is not None:
                info.swap_full_deltas.append(compute_loss(audit_X, C) - before)
            info.swaps_accepted += 1
            d[:, j] = nd
            c1, d1, d2 = top2(d)
        else:
            info.swaps_rejected += 1


def emeans(X: np.ndarray, C_init: np.ndarray, T: int = 2, B: Optional[int] = None,
           alpha0: Optional[float] = None, tau: float = 1e-10, revival: str = "mass",
           safety: bool = False, seed: int = 0, eps: float = 1e-3,
           reservoir_size: Optional[int] = None, max_polish: int = 300,
           audit: bool = False):
    """e-means (defaults) or e-means+ (tau=1, revival='guarded', safety=True).

    Returns (C, RunInfo).
    """
    X = np.asarray(X, dtype=np.float64)
    n, dim = X.shape
    k = len(C_init)
    lawB, lawA = parameters(n, T, k)
    B = lawB if B is None else B
    alpha0 = lawA if alpha0 is None else alpha0
    rng = np.random.default_rng(seed)
    info = RunInfo()

    C = np.array(C_init, dtype=np.float64, copy=True)
    N = np.full(k, float(tau))
    S = N[:, None] * C
    M = reservoir_size or min(n, max(256, 16 * k))
    res = np.empty((M, dim))
    res_norm = np.empty((M, 1))
    filled = seen = 0
    steps_per_epoch = n // B
    total = T * steps_per_epoch
    cooldown = np.zeros(k, dtype=np.int64)
    step = 0
    fc = np.zeros(k)
    fs = np.zeros((k, dim))
    for epoch in range(T):
        perm = rng.permutation(n)
        fc[:] = 0
        fs[:] = 0
        for b in range(steps_per_epoch):
            Xb = X[perm[b * B:(b + 1) * B]]
            if revival == "guarded" and epoch == 0:   # Vitter's Algorithm R, vectorised
                take = min(len(Xb), M - filled)
                if take > 0:
                    res[filled:filled + take] = Xb[:take]
                    res_norm[filled:filled + take, 0] = np.sum(Xb[:take] ** 2, axis=1)
                    filled += take
                    seen += take
                rem = Xb[take:]
                if len(rem):
                    tgt = rng.integers(0, seen + np.arange(len(rem)) + 1)
                    hit = tgt < M
                    res[tgt[hit]] = rem[hit]
                    res_norm[tgt[hit], 0] = np.sum(rem[hit] ** 2, axis=1)
                    seen += len(rem)
            a = alpha0 * (0.55 + 0.45 * np.cos(np.pi * step / total))
            lb = np.argmin(sq_dists(Xb, C), axis=1)
            bc = np.bincount(lb, minlength=k).astype(np.float64)
            bs = np.zeros((k, dim))
            np.add.at(bs, lb, Xb)
            N = (1 - a) * N + a * bc
            S = (1 - a) * S + a * bs
            fc += bc
            fs += bs
            if (step + 1) % 5 == 0:
                if revival == "mass":
                    for j in range(k):
                        if N[j] < 0.1 / k * N.sum():
                            donor = int(np.argmax(N))
                            cen = S[donor] / N[donor]
                            half = N[donor] / 2
                            N[donor] = N[j] = half
                            S[donor] = half * cen * (1 - 1 / 1024)
                            S[j] = half * cen * (1 + 1 / 1024)
                elif revival == "guarded":
                    C = S / N[:, None]
                    _guarded_revival(res[:filled], res_norm[:filled], C, N, S, cooldown, step,
                                     steps_per_epoch, rng, info, eps, X if audit else None)
            C = S / N[:, None]
            step += 1
    info.passes = float(T)
    nz = fc > 0
    C[nz] = fs[nz] / fc[nz, None]      # last-epoch means (recipe output)

    if safety:
        phi0 = compute_loss(X, C_init)          # free in practice: k-means++ computes it
        if safety == "check":                   # check-and-fall-back only
            info.passes += 2
            if compute_loss(X, C) > phi0:
                C, it, conv = lloyd(X, C_init, max_iter=max_polish)
                info.fell_back, info.converged = True, conv
                info.passes += it + 1
        else:                                   # polish to convergence, then check
            C, it, conv = lloyd(X, C, max_iter=max_polish)
            info.polish_iters, info.converged = it, conv
            info.passes += 2 + it + 1
            if compute_loss(X, C) > phi0:
                C, it, conv = lloyd(X, C_init, max_iter=max_polish)
                info.fell_back, info.converged = True, conv
                info.passes += it + 1
    return C, info


def emeans_plus(X, C_init, T: int = 2, seed: int = 0, **kw):
    """e-means+: tau = 1, the recipe's mass revival, check-and-fall-back.

    Pass safety=True for the polished variant (Lloyd fixed point output) and
    revival="guarded" for cost-guarded revival.
    """
    kw.setdefault("tau", 1.0)
    kw.setdefault("revival", "mass")
    kw.setdefault("safety", "check")
    return emeans(X, C_init, T=T, seed=seed, **kw)
