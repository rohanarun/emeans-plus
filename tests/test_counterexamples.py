import numpy as np
import pytest

from emeans_plus import compute_loss, emeans, emeans_plus, kmeanspp_init, lloyd, parameters
from emeans_plus import reference_1d as r1

X2 = [0, 0, 15, 29, 29, 29, 29, 29]
PREFIX = [[15, 29, 29, 29, 0, 0, 29, 29], [0, 0, 29, 29, 15, 29, 29, 29]]


@pytest.mark.parametrize("m", [10, 20, 50])
def test_ce1_baseline_matches_reference_and_guard_matches_lloyd(m):
    D = m + 4
    x = [1] * m + [3] * m + [D]
    T = 10 * len(x)
    B, a = r1.parameters(len(x), T)
    assert (B, a) == (len(x), 1)
    for seeds in r1.kpp_probabilities(x):
        c, _, _ = r1.emeans(x, seeds, T, B, a)
        assert c == [1, 4]                              # Assylbekov's assertion
        lloyd = r1.lloyd(x, seeds)
        cp, _, _ = r1.emeans(x, seeds, T, B, a, tau=1.0)              # e-means+ (tau + C)
        assert r1.safe(x, seeds, cp, polish=False) == lloyd
        cg, _, _ = r1.emeans(x, seeds, T, B, a, revival="guarded")   # guard alone
        assert cg == lloyd


def test_ce2_prescribed_path():
    B, a = r1.parameters(8, 25)
    c, hist, ev = r1.emeans(X2, (15, 29), 25, B, a, prefix=PREFIX)
    assert c == [0, 80 / 3] and not ev
    assert hist[1][0] > 1 > hist[2][0]                  # crossing happens at batch 3
    c1, _, _ = r1.emeans(X2, (15, 29), 25, B, a, prefix=PREFIX, tau=1.0)
    assert r1.loss(X2, c1) == 150


def test_ce2_safety_net_alone_does_not_help():
    # (0, 80/3) is a Lloyd fixed point with loss 490/3 < 450 = seed loss
    c = [0, 80 / 3]
    assert r1.lloyd(X2, c) == c
    assert r1.safe(X2, (15, 29), c) == c


def test_ce2_tau_one_never_fails():
    B, a = r1.parameters(8, 25)
    for s in range(500):
        c, _, _ = r1.emeans(X2, (15, 29), 25, B, a, tau=1.0, seed=s)
        assert r1.loss(X2, c) == pytest.approx(150)


def test_one_step_threshold():
    B, a = r1.parameters(8, 25)
    tau_star = 2 * a / (14 * (1 - a))
    for tau, crosses in [(0.99 * tau_star, True), (1.01 * tau_star, False)]:
        N = (1 - a) * tau + a * 2            # first batch {0,0,29,29}
        cL = (1 - a) * tau * 15 / N
        assert ((cL + 29) / 2 < 15) == crosses


def test_numpy_port_agrees_with_reference_on_ce1():
    x = [1.0] * 10 + [3.0] * 10 + [14.0]
    X = np.array(x)[:, None]
    for seeds in [(1, 14), (3, 14), (1, 3)]:
        C0 = np.array(seeds, float)[:, None]
        c, _ = emeans(X, C0, T=210)
        assert np.allclose(c.ravel(), [1, 4])
        cp, _ = emeans_plus(X, C0, T=210)
        assert np.allclose(cp.ravel(), r1.lloyd(x, seeds))


def test_parameters_match_reference():
    for n, T in [(21, 210), (8, 25), (101, 1010)]:
        assert parameters(n, T, 2) == r1.parameters(n, T)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("revival", ["mass", "guarded"])
def test_safety_net_guarantees(seed, revival):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(600, 6))
    C0 = kmeanspp_init(X, 12, rng)
    phi0 = compute_loss(X, C0)
    C, info = emeans_plus(X, C0, T=2, seed=seed, revival=revival)          # check only
    assert compute_loss(X, C) <= phi0 + 1e-9
    C, info = emeans_plus(X, C0, T=2, seed=seed, revival=revival, safety=True)  # polish
    assert compute_loss(X, C) <= phi0 + 1e-9
    assert info.converged
    C2, it, conv = lloyd(X, C, 5)
    assert conv and it <= 1 and np.allclose(C2, C, atol=1e-12)  # Lloyd fixed point


def test_check_triggers_on_ce1():
    X = np.array([1.0] * 10 + [3.0] * 10 + [14.0])[:, None]
    C0 = np.array([[1.0], [14.0]])
    C, info = emeans_plus(X, C0, T=210)
    assert info.fell_back and compute_loss(X, C) == 20.0
