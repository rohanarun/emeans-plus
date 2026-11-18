"""Assert every qualitative claim made in the paper against results/*.json.

Run after run_experiments.py and make_tables.py. Exits non-zero on the first false claim.
"""
from __future__ import annotations

import json
import os

import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
J = lambda n: json.load(open(os.path.join(RES, n + ".json")))  # noqa: E731
ok = 0


def claim(cond, text):
    global ok
    if not cond:
        raise AssertionError("FALSE CLAIM: " + text)
    ok += 1
    print("  ok:", text)


# ------------------------------------------------------------- Counterexample 1
ce = J("counterexamples")
print("Counterexample 1")
for r in ce["ce1"]:
    m = r["m"]
    claim(r["e-means|seed"] == m * (m + 1) and r["e-means|kpp"] == m * (m + 1),
          f"m={m}: recipe loss m(m+1) from seeds and in k-means++ expectation")
    claim(r["tau|kpp"] == m * (m + 1), f"m={m}: tau alone does not fix it")
    claim(r["lloyd_kpp"] < 6 * m, f"m={m}: E[Lloyd] < 6m")
    claim(m * (m + 1) > 4 * m, f"m={m}: recipe loss exceeds seed loss 4m, so C falls back")
    for v in ["guard", "e-means + C", "tau + C (e-means+)", "tau + S", "tau + guard + S"]:
        claim(abs(r[v + "|seed"] - r["lloyd_seed"]) < 1e-9 and abs(r[v + "|kpp"] - r["lloyd_kpp"]) < 1e-6,
              f"m={m}: {v} equals Lloyd from the seeds and in expectation")

# ------------------------------------------------------------- Counterexample 2
print("Counterexample 2")
v = ce["ce2_variants"]
p = ce["ce2_params"]
claim(p["seed_loss"] == 450 and p["lloyd"] == 150, "seed loss 450, Lloyd 150")
for name in ["e-means", "guard", "e-means + C", "e-means + S"]:
    lo, hi = v[name]["ci"]
    claim(lo <= 24 / 49 <= hi, f"{name}: failure CI contains 24/49")
    claim(abs(ce["ce2_prescribed"][name] - 490 / 3) < 1e-9, f"{name}: prescribed path ends at 490/3")
for name in ["tau", "tau + C (e-means+)", "tau + S", "tau + guard + S"]:
    claim(v[name]["fail"] == 0, f"{name}: no failures")
    claim(ce["ce2_prescribed"][name] == 150, f"{name}: prescribed path ends at 150")
    claim(abs(ce["ce2_kpp"][name] - ce["ce2_kpp"]["lloyd"]) < 1e-9, f"{name}: k-means++ expectation equals Lloyd")
sw = {r["tau"]: r["fail"] for r in ce["ce2_tau_sweep"]}
last_fail = max(t for t, f in sw.items() if f > 0)
first_ok = min(t for t, f in sw.items() if f == 0)
claim(all(f > 0 for t, f in sw.items() if t < first_ok) and all(f == 0 for t, f in sw.items() if t >= first_ok),
      "sweep: single transition")
claim(last_fail < p["tau_star"] < first_ok, f"tau*={p['tau_star']:.5f} in ({last_fail}, {first_ok})")
claim(first_ok - last_fail <= 0.001, "transition resolved to 0.001")
# one-step algebra: first batch with z zeros (and no 15) crosses iff tau < z*a/(14(1-a))
a = p["alpha0"]
for z in (1, 2):
    thr = z * a / (14 * (1 - a))
    for tau, crosses in [(0.999 * thr, True), (1.001 * thr, False)]:
        cL = (1 - a) * tau * 15 / ((1 - a) * tau + a * z)
        claim(((cL + 29) / 2 < 15) == crosses, f"one-step crossing iff tau < {thr:.4f} (z={z})")
claim(sw[0.05] > 0.4 and sw[0.1] < 0.2 and 0.05 < p["tau_star"] / 2 < 0.1, "first drop brackets tau*/2")
from math import comb
pl = max((r for r in ce["ce2_tau_sweep"] if r["fail"] > 0), key=lambda r: r["tau"])
claim(comb(5, 2) / comb(8, 4) == 1 / 7, "P(first batch = both zeros, no 15) = 1/7")
claim(pl["ci"][0] <= 1 / 7 <= pl["ci"][1], f"plateau just below tau* ({pl['fail']:.4f}) has CI containing 1/7")
claim((comb(7, 4) - comb(5, 4)) / comb(8, 4) == 3 / 7 and sw[1e-10] > 3 / 7, "first-batch route alone (3/7) is below the recipe rate")
pa = J("port_agreement")
lo1, hi1 = pa["1e-10"]["ci_reference"]
lo2, hi2 = pa["1e-10"]["ci_numpy"]
claim(max(lo1, lo2) <= min(hi1, hi2), "NumPy and reference failure-rate CIs overlap")

# ------------------------------------------------------------- benchmarks
print("Benchmarks")
S = J("summary")
for key in [k for k in S if k != "sift1m"]:
    runs = J(key)["runs"]
    for m in runs[0]:
        if isinstance(runs[0][m], dict) and "converged" in runs[0][m] and m != "Lloyd-25":
            if m in ("e-means", "tau + C (e-means+)", "e-means + C") :
                # converged is only meaningful when Lloyd ran (fallback)
                claim(all(r[m]["converged"] for r in runs if r[m]["fell_back"]), f"{key}/{m}: every fallback Lloyd converged")
            else:
                claim(all(r[m]["converged"] for r in runs), f"{key}/{m}: every Lloyd run converged")

tm = J("rare_tail_mechanism")
claim(all(r["seed_tail_max"] < 1 for r in tm), "rare tail: seeds cover every tail cluster")
claim(all(r["mass"]["tail_max"] > 30 for r in tm), "rare tail: after mass revival a tail point is >30 from every centre, every seed")
claim(all(r["off"]["ratio"] < 1.01 for r in tm), "rare tail: recipe without revival is within 1% of Lloyd, every seed")
rt = S["rare_tail"]
claim(rt["e-means"]["ratio"] > 10 and S["rare_tail_B32"]["e-means"]["ratio"] > 10, "rare tail: recipe > 10x Lloyd")
n = rt["e-means"]["n"]
claim(rt["tau + C (e-means+)"]["fell_back"] == n, "rare tail: check falls back in every seed")
runs = J("rare_tail")["runs"]
claim(all(abs(r["tau + C (e-means+)"]["loss"] - r["Lloyd (to convergence)"]["loss"]) < 1e-6 for r in runs),
      "rare tail: e-means+ equals Lloyd")
claim(rt["tau + C (e-means+)"]["passes"] > rt["Lloyd-25"]["passes"], "rare tail: e-means+ uses more passes than Lloyd")
claim(rt["tau + guard + S"]["ratio"] < 1, "rare tail: tau+G+S slightly better than Lloyd on average")
claim(abs(rt["tau + guard + S"]["ratio"] - 1) < rt["tau + guard + S"]["ratio_sd"], "...but within one s.d.")

d10, d64 = S["digits_k10"], S["digits_k64"]
claim(d10["e-means"]["ratio"] > 1, "digits k=10: recipe worse than Lloyd-25")
claim(d10["tau + C (e-means+)"]["ratio"] < d10["e-means"]["ratio"], "digits k=10: tau helps")
r64 = J("digits_k64")["runs"]
claim(all(r["e-means"]["loss"] < r["Lloyd-25"]["loss"] for r in r64), "digits k=64: recipe below Lloyd-25 on every seed")
claim(d64["tau + C (e-means+)"]["ratio"] > 0.999 and
      d64["tau + C (e-means+)"]["ratio"] > d64["e-means"]["ratio"] + 0.005, "digits k=64: tau=1 removes the advantage")
claim(d64["guard + S"]["ratio"] > d64["e-means + S"]["ratio"], "digits k=64: G+S worse than recipe+S")
claim(abs(d10["guard + S"]["ratio"] - d10["e-means + S"]["ratio"]) < d10["e-means + S"]["ratio_sd"],
      "digits k=10: G+S and recipe+S within noise")
claim(d64["tau + no revival + S"]["ratio"] > d64["tau + S"]["ratio"], "digits k=64: removing revival hurts")
ts = J("tau_sensitivity")
means = [r["mean"] for r in ts["digits_k64"]]
claim(all(x < y for x, y in zip(means[1:], means[2:])) and means[-1] > means[0], "tau cost grows with tau on k=64 (tau >= 0.05)")
claim(sum(r["fell_back"] for k in ("digits_k10", "digits_k64") for r in ts[k]) == 0, "tau sensitivity: no fallbacks")

small = ["rare_tail", "rare_tail_B32", "digits_k10", "digits_k64"]
G = ("guard + S", "tau + guard + S")
tot = sum(S[k][m]["swaps"] for k in small for m in G)
down = sum(S[k][m]["swaps_down"] for k in small for m in G)
claim(tot > 0 and down < tot / 2, f"swap audit, small sets: only {down}/{tot} accepted swaps lowered the full loss")

if os.path.exists(os.path.join(RES, "heavy_tailed_ann.json")):
    print("Heavy-tailed ANN")
    an = S["heavy_tailed_ann"]
    claim(an["e-means"]["ratio"] > 1.5, "ANN: recipe far worse than Lloyd")
    runs = J("heavy_tailed_ann")["runs"]
    for m in ("tau + S", "e-means + S"):
        kept = [r[m]["loss"] / r["Lloyd-25"]["loss"] for r in runs if not r[m]["fell_back"]]
        claim(an[m]["fell_back"] == 3 and len(kept) == 2 and min(kept) > 1.5,
              f"ANN: {m} falls back in 3/5 seeds and is >1.5x Lloyd in the other 2")
    claim(an["e-means + C"]["fell_back"] == 4, "ANN: recipe + C falls back in 4/5 seeds")
    claim(all(abs(r["tau + C (e-means+)"]["loss"] - r["Lloyd (to convergence)"]["loss"]) < 1e-6 for r in runs),
          "ANN: e-means+ equals Lloyd to convergence on every seed")
    claim(an["tau + C (e-means+)"]["passes"] > 3 * an["Lloyd-25"]["passes"], "ANN: e-means+ costs >3x Lloyd-25 passes")
    claim(abs(an["tau + guard + S"]["passes"] - an["Lloyd-25, best of 3 seeds"]["passes"]) < 5,
          "ANN: tau+G+S uses about the passes of best-of-3")
    claim(an["tau + guard + S"]["wins"] == 5 and an["guard + S"]["wins"] == 5, "ANN: G variants beat Lloyd-25 on every seed")
    td = sum(an[m]["swaps_down"] for m in G); tt = sum(an[m]["swaps"] for m in G)
    claim(td > tt / 2, f"ANN swap audit: {td}/{tt} accepted swaps lowered the full loss")
    claim(an["guard + S"]["ratio"] < 1 and an["tau + guard + S"]["ratio"] < 1, "ANN: G variants beat Lloyd-25")
    claim(an["tau + guard + S"]["ratio"] < an["Lloyd-25, best of 3 seeds"]["ratio"], "ANN: tau+G+S beats best-of-3 Lloyd")
    claim(an["tau + guard + S"]["r1"] > an["Lloyd-25"]["r1"], "ANN: tau+G+S higher 1-probe recall than Lloyd")
    claim(an["tau + C (e-means+)"]["fell_back"] == an["tau + C (e-means+)"]["n"], "ANN: e-means+ falls back in every seed")

if os.path.exists(os.path.join(RES, "sift1m.json")):
    print("SIFT1M")
    sf = S["sift1m"]
    runs = J("sift1m")["runs"]
    claim(J("sift1m")["meta"]["n"] == 1_000_000 and len(runs) == 3, "SIFT: full 1M training set, 3 seeds")
    claim(all(abs(sf[m]["ratio"] - 1) < 0.001 for m in ("e-means", "e-means + C", "tau + C (e-means+)")),
          "SIFT: recipe, recipe+C and e-means+ within 0.1% of Lloyd-25")
    claim(sf["tau + C (e-means+)"]["fell_back"] == 0 and sf["e-means + C"]["fell_back"] == 0, "SIFT: check never falls back")
    claim(sf["tau + C (e-means+)"]["passes"] == 4 and sf["Lloyd-25"]["passes"] == 26, "SIFT: 4 passes vs 26")
    claim(sf["Lloyd-25"]["time"] / sf["tau + C (e-means+)"]["time"] > 5, "SIFT: e-means+ >5x faster than Lloyd-25 (same code, same machine)")
    claim(all(sf[m]["ratio"] < 0.998 for m in ("tau + S", "tau + guard + S", "Lloyd (to convergence)")),
          "SIFT: 300-iteration Lloyd/polish ~0.4% lower than Lloyd-25")
    claim(sf["tau + guard + S"]["ratio"] > sf["tau + S"]["ratio"], "SIFT: guard gives no gain over polish alone")
    capped = [r[m] for r in runs for m in ("tau + S", "tau + guard + S", "Lloyd (to convergence)")]
    claim(sum(not c["converged"] for c in capped) >= 8, "SIFT: (almost) all 300-iteration runs hit the cap unconverged")
    rec = [sf[m]["r1"] for m in ("Lloyd-25", "e-means", "tau + C (e-means+)", "tau + S", "tau + guard + S")]
    claim(max(rec) - min(rec) < 0.015, "SIFT: 1-probe recall within 1.5 points across variants")
    claim(abs(sf["tau + C (e-means+)"]["ratio"] - sf["e-means"]["ratio"]) < 0.0001, "SIFT: tau changes loss by <0.01%")

print(f"\nAll {ok} claims hold.")
