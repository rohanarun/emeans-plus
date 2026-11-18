"""One-dimensional, two-centre e-means recipe used for the counterexamples.

The baseline (``revival="mass"``, ``tau=1e-10``) is a line-for-line port of
``emeans`` in Assylbekov's reference code
(https://github.com/zh3nis/e-means-counterexamples, counterexamples.py),
which implements the recipe printed in Bessa et al. (2026) for k = 2.
Two optional changes are added, each off by default:

* ``tau``: the initial pseudo-count N_j(0) (the recipe uses 1e-10);
* ``revival="guarded"``: a low-mass centre is only replaced when the swap
  lowers the k-means loss on the data (cost-guarded revival).
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import cos, log2, pi
from random import Random
from typing import Optional, Sequence


def parameters(n: int, T: int, k: int = 2) -> tuple[int, float]:
    """Printed batch-size and peak-step laws (the reference uses k = 2)."""
    raw = 0.035 * k**0.13 * n**0.88 * T**0.81
    B = min(n, 2 ** max(0, round(log2(raw))))
    alpha0 = min(1.0, 0.824 * k**-0.46 * B**0.90 * n**-0.45 * T**-0.11)
    return B, alpha0


def assign(v: float, centers: Sequence[float]) -> int:
    return 0 if (v - centers[0]) ** 2 <= (v - centers[1]) ** 2 else 1


def stats(x, centers):
    counts, sums = [0, 0], [0.0, 0.0]
    for v in x:
        j = assign(v, centers)
        counts[j] += 1
        sums[j] += v
    return counts, sums


def loss(x, centers) -> float:
    return sum(min((v - c) ** 2 for c in centers) for v in x)


def lloyd(x, centers, max_iter: int = 1000):
    centers = list(centers)
    for _ in range(max_iter):
        counts, sums = stats(x, centers)
        updated = [sums[j] / counts[j] if counts[j] else centers[j] for j in range(2)]
        if updated == centers:
            return updated
        centers = updated
    raise RuntimeError("Lloyd did not converge")


def _guarded_swap(x, N, S, dead: int, rng: Random, eps: float) -> bool:
    """Replace centre `dead` by a D^2-sampled data point only if the loss on x drops."""
    centers = [S[j] / N[j] for j in range(2)]
    keep = 1 - dead
    d_keep = [(v - centers[keep]) ** 2 for v in x]
    d_now = [min((v - centers[0]) ** 2, (v - centers[1]) ** 2) for v in x]
    total = sum(d_now)
    if total <= 0:
        return False
    r, acc, new = rng.random() * total, 0.0, x[-1]
    for v, w in zip(x, d_now):
        acc += w
        if acc >= r:
            new = v
            break
    loss_before = total
    loss_after = sum(min(dk, (v - new) ** 2) for v, dk in zip(x, d_keep))
    if loss_before - loss_after > eps * loss_before:
        half = N[keep] / 2
        N[keep] = N[dead] = half
        S[keep] = half * centers[keep]
        S[dead] = half * new
        return True
    return False


def emeans(
    x,
    centers,
    T: int,
    B: int,
    alpha0: float,
    revival: str = "mass",
    tau: float = 1e-10,
    prefix: Sequence[Sequence[float]] = (),
    seed: int = 0,
    eps: float = 1e-3,
):
    """Returns (last-epoch means, centre history, revival events)."""
    assert len(x) % B == 0
    N = [tau, tau]
    S = [tau * c for c in centers]
    M = T * (len(x) // B)
    rng = Random(seed)
    guard_rng = Random(seed + 1)
    history, revivals = [], []
    final_counts, final_sums = [0, 0], [0.0, 0.0]
    for epoch in range(T):
        order = list(prefix[epoch]) if epoch < len(prefix) else list(x)
        if epoch >= len(prefix):
            rng.shuffle(order)
        final_counts, final_sums = [0, 0], [0.0, 0.0]
        for start in range(0, len(x), B):
            cur = [S[j] / N[j] for j in range(2)]
            counts, sums = stats(order[start:start + B], cur)
            t = len(history)
            a = alpha0 * (0.55 + 0.45 * cos(pi * t / M))
            for j in range(2):
                N[j] = (1 - a) * N[j] + a * counts[j]
                S[j] = (1 - a) * S[j] + a * sums[j]
                final_counts[j] += counts[j]
                final_sums[j] += sums[j]
            if revival != "off" and (t + 1) % 5 == 0 and min(N) < 0.05 * sum(N):
                dead = 0 if N[0] < N[1] else 1
                donor = 1 - dead
                if revival == "mass":
                    center, half = S[donor] / N[donor], N[donor] / 2
                    delta = 1 / 1024
                    N[donor] = N[dead] = half
                    S[donor] = half * center * (1 - delta)
                    S[dead] = half * center * (1 + delta)
                    revivals.append(t + 1)
                elif revival == "guarded":
                    if _guarded_swap(list(x), N, S, dead, guard_rng, eps):
                        revivals.append(t + 1)
            history.append([S[j] / N[j] for j in range(2)])
    final = [
        final_sums[j] / final_counts[j] if final_counts[j] else S[j] / N[j]
        for j in range(2)
    ]
    return final, history, revivals


def safe(x, c0, c_out, polish: bool = True):
    """Safety net: (optional) Lloyd polish to convergence, then fall back to Lloyd(c0)."""
    c = lloyd(x, c_out) if polish else list(c_out)
    if loss(x, c) > loss(x, c0):
        c = lloyd(x, c0)
    return c


def kpp_probabilities(x):
    """Exact probabilities of unordered seed-value pairs under k-means++ (k = 2)."""
    mass = Counter(x)
    probabilities = Counter()
    for u, nu in mass.items():
        denominator = sum(nv * (v - u) ** 2 for v, nv in mass.items())
        for v, nv in mass.items():
            if u != v:
                pair = tuple(sorted((u, v)))
                probabilities[pair] += Fraction(nu, len(x)) * Fraction(
                    nv * (v - u) ** 2, denominator
                )
    assert sum(probabilities.values()) == 1
    return probabilities
