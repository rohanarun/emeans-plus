"""Generate every table and every number quoted in the paper from results/*.json.

Writes paper/generated/*.tex. The paper only \\input{}s these files, so the text
cannot drift from the measured results.
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
GEN = os.path.join(ROOT, "paper", "generated")
os.makedirs(GEN, exist_ok=True)
macros: list[str] = []


def load(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def mac(name, value):
    macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def fmt(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", "{,}")


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}\\%"


def write(name, body):
    with open(os.path.join(GEN, name), "w") as f:
        f.write(body)


# ---------------------------------------------------------------- counterexamples
ce = load("counterexamples")
ROW1D = [("Lloyd", None), ("e-means (recipe)", "e-means"), ("$\\tau$ only", "tau"),
         ("Guard only", "guard"), ("e-means + C", "e-means + C"),
         ("\\textbf{e-means+} ($\\tau$ + C)", "tau + C (e-means+)"),
         ("$\\tau$ + S", "tau + S"), ("$\\tau$ + guard + S", "tau + guard + S")]
lines = []
for label, key in ROW1D:
    cells = []
    for r in ce["ce1"]:
        if key is None:
            cells += [fmt(r["lloyd_seed"], 0), fmt(r["lloyd_kpp"], 2)]
        else:
            cells += [fmt(r[key + "|seed"], 0), fmt(r[key + "|kpp"], 2)]
    lines.append(label + " & " + " & ".join(cells) + " \\\\")
write("tab_ce1.tex", "\n".join(lines) + "\n")

p2 = ce["ce2_params"]
mac("ceTwoAlpha", f"{p2['alpha0']:.4f}")
mac("tauStar", f"{p2['tau_star']:.4f}")
mac("ceTwoTrials", f"{ce['ce2_variants']['e-means']['trials']:,}".replace(",", "{,}"))
v = ce["ce2_variants"]
mac("ceTwoFailRecipe", pct(v["e-means"]["fail"], 2))
mac("ceTwoFailRecipeLo", pct(v["e-means"]["ci"][0], 2))
mac("ceTwoFailRecipeHi", pct(v["e-means"]["ci"][1], 2))
mac("ceTwoZeroUpper", pct(v["tau"]["ci"][1], 3))
lines = []
for label, key in ROW1D[1:]:
    r = v[key]
    lines.append(f"{label} & {fmt(ce['ce2_prescribed'][key], 2)} & {pct(r['fail'], 2)} "
                 f"& [{pct(r['ci'][0], 2)}, {pct(r['ci'][1], 2)}] & {fmt(r['mean_loss'], 2)} "
                 f"& {fmt(ce['ce2_kpp'][key], 2)} \\\\")
write("tab_ce2.tex", "\n".join(lines) + "\n")
mac("ceTwoKppLloyd", fmt(ce["ce2_kpp"]["lloyd"], 2))
mac("ceTwoKppShuffles", f"{ce['ce2_kpp']['shuffles_per_seed_pair']:,}".replace(",", "{,}"))

sweep = ce["ce2_tau_sweep"]
lines = []
for r in sweep:
    t = "$10^{-10}$" if r["tau"] < 1e-6 else f"{r['tau']:g}"
    lines.append(f"{t} & {pct(r['fail'], 2)} & {pct(r['ci'][1], 2)} & {fmt(r['mean_loss'], 2)} \\\\")
write("tab_tau.tex", "\n".join(lines) + "\n")
fails = [r for r in sweep if r["fail"] > 0]
oks = [r for r in sweep if r["fail"] == 0]
last_fail = max(r["tau"] for r in fails)
first_ok = min(r["tau"] for r in oks)
assert all(r["tau"] < first_ok for r in fails), "sweep is not monotone"
mac("tauLastFail", f"{last_fail:g}")
mac("tauFirstOk", f"{first_ok:g}")
assert last_fail < p2["tau_star"] < first_ok, "tau* does not sit inside the observed transition"
mac("tauStarHalf", f"{p2['tau_star'] / 2:.4f}")
fs = {r["tau"]: r["fail"] for r in sweep}
assert fs[0.05] > 0.4 and fs[0.1] < 0.2, "first drop not between 0.05 and 0.1"
assert 0.05 < p2["tau_star"] / 2 < 0.1
mac("failAtFive", pct(fs[0.05], 1))
plateau = max((r for r in sweep if r["fail"] > 0), key=lambda r: r["tau"])
mac("plateauFail", pct(plateau["fail"], 2))
mac("plateauLo", pct(plateau["ci"][0], 2))
mac("plateauHi", pct(plateau["ci"][1], 2))
mac("failAtTen", pct(fs[0.1], 1))
pa = load("port_agreement")
mac("portTrials", f"{pa['trials']:,}".replace(",", "{,}"))
mac("portRef", pct(pa["1e-10"]["reference"], 2))
mac("portNumpy", pct(pa["1e-10"]["numpy"], 2))

# ---------------------------------------------------------------- benchmarks
BENCH = [("rare_tail", "Rare tail"), ("rare_tail_B32", "Rare tail, $B{=}32$"),
         ("digits_k10", "Digits, $k{=}10$"), ("digits_k64", "Digits, $k{=}64$"),
         ("heavy_tailed_ann", "Heavy-tailed ANN"), ("sift1m", "SIFT1M")]
METH = [("Lloyd-25", "Lloyd-25"), ("Lloyd-25, best of 3 seeds", "Lloyd-25, best of 3"),
        ("e-means", "e-means (recipe)"), ("e-means + C", "e-means + C"),
        ("tau + C (e-means+)", "\\textbf{e-means+} ($\\tau$ + C)"),
        ("e-means + S", "e-means + S"), ("tau + S", "$\\tau$ + S"),
        ("guard + S", "Guard + S"), ("tau + guard + S", "$\\tau$ + guard + S")]
data = {key: load(key) for key, _ in BENCH}
data = {k: d for k, d in data.items() if d is not None}
summary = {}
for key, d in data.items():
    runs = d["runs"]
    base = np.array([r["Lloyd-25"]["loss"] for r in runs])
    summary[key] = {}
    for m in [mm for mm, _ in METH] + ["tau + no revival + S", "Lloyd (to convergence)"]:
        if m not in runs[0]:
            continue
        L = np.array([r[m]["loss"] for r in runs])
        ratio = L / base
        deltas = [x for r in runs for x in r[m].get("swap_full_deltas", [])]
        summary[key][m] = {
            "ratio": float(ratio.mean()), "ratio_sd": float(ratio.std(ddof=1)),
            "wins": int(np.sum(ratio < 0.999)), "losses": int(np.sum(ratio > 1.001)),
            "passes": float(np.mean([r[m]["passes"] for r in runs])),
            "fell_back": int(sum(r[m].get("fell_back", False) for r in runs)),
            "swaps": len(deltas), "swaps_down": int(sum(x < 0 for x in deltas)),
            "n": len(runs),
            "time": float(np.mean([r[m]["time"] for r in runs])),
            "converged_all": all(r[m].get("converged", True) for r in runs),
            "converged_count": int(sum(bool(r[m].get("converged", False)) for r in runs)),
        }
        if "recall" in runs[0][m]:
            for p in ("1", "4", "16"):
                rc = np.array([r[m]["recall"][p] for r in runs])
                summary[key][m][f"r{p}"] = float(rc.mean())
                summary[key][m][f"r{p}sd"] = float(rc.std(ddof=1))

hdr = " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for key, lab in BENCH if key in data)
ncol = sum(key in data for key, _ in BENCH)
lines = ["\\begin{tabular}{l" + "rr" * ncol + "}", "\\toprule", "Method & " + hdr + " \\\\",
         " & " + " & ".join("loss ratio & passes" for key, _ in BENCH if key in data) + " \\\\",
         "\\midrule"]
for m, lab in METH:
    cells = []
    for key, _ in BENCH:
        if key not in data:
            continue
        s = summary[key].get(m)
        if s is None:
            cells += ["--", "--"]
        else:
            cells += [f"{s['ratio']:.3f} {{\\scriptsize $\\pm${s['ratio_sd']:.3f}}}",
                      f"{s['passes']:.1f}"]
    lines.append(lab + " & " + " & ".join(cells) + " \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
write("tab_bench.tex", "\n".join(lines) + "\n")

lines = ["\\begin{tabular}{l" + "c" * ncol + "}", "\\toprule",
         "Variant & " + " & ".join(lab for key, lab in BENCH if key in data) + " \\\\",
         "\\midrule"]
for m, lab in METH[2:]:
    cells = []
    for key, _ in BENCH:
        if key not in data:
            continue
        s = summary[key].get(m)
        cells.append("--" if s is None else f"{s['wins']}/{s['losses']}/{s['fell_back']}")
    lines.append(lab + " & " + " & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_wins.tex", "\n".join(lines) + "\n")

for key, short in [("rare_tail", "RT"), ("rare_tail_B32", "RTB"), ("digits_k10", "DTen"),
                   ("digits_k64", "DSixtyFour"), ("heavy_tailed_ann", "ANN"), ("sift1m", "SIFT")]:
    if key not in data:
        continue
    meta = data[key]["meta"]
    mac(f"seeds{short}", meta["seeds"])
    mac(f"B{short}", meta["B"])
    mac(f"alpha{short}", f"{meta['alpha0']:.3f}")
    for m, tag in [("e-means", "Recipe"), ("tau + C (e-means+)", "Plus"), ("tau + S", "PlusS"),
                   ("e-means + C", "RecipeC"), ("Lloyd-25", "Lloyd"),
                   ("Lloyd-25, best of 3 seeds", "LloydThree"), ("tau + guard + S", "Guard"),
                   ("guard + S", "GuardOnly"), ("e-means + S", "RecipeS"),
                   ("tau + no revival + S", "NoRev"), ("Lloyd (to convergence)", "LloydConv")]:
        s = summary[key].get(m)
        if s is None:
            continue
        mac(f"ratio{short}{tag}", f"{s['ratio']:.4f}" if short == "SIFT" else f"{s['ratio']:.3f}")
        mac(f"pass{short}{tag}", f"{s['passes']:.1f}")
        mac(f"wins{short}{tag}", s["wins"])
        mac(f"fb{short}{tag}", s["fell_back"])
        mac(f"time{short}{tag}", f"{s['time']:.0f}")
        mac(f"conv{short}{tag}", s["converged_count"])
        if s["swaps"]:
            mac(f"swaps{short}{tag}", s["swaps"])
            mac(f"swapsDown{short}{tag}", s["swaps_down"])
        if "r1" in s:
            for p in ("1", "4", "16"):
                name = {"1": "One", "4": "Four", "16": "Sixteen"}[p]
                mac(f"rec{name}{short}{tag}", f"{s['r' + p]:.3f}")

# swap audit, guarded runs pooled per group
for tag, keys in [("Small", ["rare_tail", "rare_tail_B32", "digits_k10", "digits_k64"]),
                  ("ANN", ["heavy_tailed_ann"])]:
    tot = down = 0
    for key in keys:
        for m in ("guard + S", "tau + guard + S"):
            s = summary.get(key, {}).get(m)
            if s:
                tot += s["swaps"]
                down += s["swaps_down"]
    if tot:
        mac(f"swaps{tag}Total", tot)
        mac(f"swaps{tag}Down", down)
        mac(f"swaps{tag}DownPct", pct(down / tot, 0))

if "heavy_tailed_ann" in data:
    lines = []
    for m, lab in METH:
        s = summary["heavy_tailed_ann"].get(m)
        if s is None:
            continue
        lines.append(f"{lab} & {s['ratio']:.3f} & {s['r1']:.3f} {{\\scriptsize $\\pm${s['r1sd']:.3f}}} & "
                     f"{s['r4']:.3f} & {s['r16']:.3f} & {s['passes']:.1f} \\\\")
    write("tab_ann.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- tau sensitivity
ts = load("tau_sensitivity")
if ts:
    taus = [r["tau"] for r in ts["digits_k64"]]
    lines = ["\\begin{tabular}{l" + "r" * len(taus) + "}", "\\toprule",
             "$\\tau$ & " + " & ".join("$10^{-10}$" if t < 1e-6 else f"{t:g}" for t in taus) + " \\\\",
             "\\midrule"]
    for key, lab in [("digits_k10", "Digits, $k{=}10$"), ("digits_k64", "Digits, $k{=}64$")]:
        b = ts[key][0]["mean"]
        cells = [f"{(r['mean'] / b - 1) * 100:+.2f}\\%" for r in ts[key]]
        lines.append(lab + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_tausens.tex", "\n".join(lines) + "\n")
    mac("tauSensFallbacks", sum(r["fell_back"] for k2 in ("digits_k10", "digits_k64") for r in ts[k2]))
    b = ts["digits_k64"][0]["mean"]
    for r in ts["digits_k64"]:
        if r["tau"] in (0.25, 1.0, 4.0):
            nm = {0.25: "Quarter", 1.0: "One", 4.0: "Four"}[r["tau"]]
            mac(f"tauCost{nm}", f"{(r['mean'] / b - 1) * 100:.2f}\\%")

tm = load("rare_tail_mechanism")
if tm:
    mac("tailSeedMax", f"{max(r['seed_tail_max'] for r in tm):.1f}")
    mac("tailMassMin", f"{min(r['mass']['tail_max'] for r in tm):.1f}")
    mac("tailOffRatio", f"{np.mean([r['off']['ratio'] for r in tm]):.3f}")
    mac("reservoirRT", f"{min(5000, max(256, 16 * 50)) * 10 / 5000:.1f}")

# ---------------------------------------------------------------- SIFT1M (optional)
sift = data.get("sift1m")
if sift:
    m = sift["meta"]
    S = summary["sift1m"]
    rows = []
    for key, lab in METH:
        if key not in S:
            continue
        x = S[key]
        rows.append(f"{lab} & {x['ratio']:.4f} {{\\scriptsize $\\pm${x['ratio_sd']:.4f}}} & {x['r1']:.3f} & "
                    f"{x['r4']:.3f} & {x['r16']:.3f} & {x['passes']:.1f} & {x['fell_back']} \\\\")
    nq = len(sift["runs"][0]["Lloyd-25"]) and None
    partial = " (partial run)" if m.get("partial") else ""
    body = f"""\\paragraph{{SIFT1M.}} We trained on {m['n']:,} SIFT descriptors ($d={m['d']}$) with $k={m['k']}$,
$T={m['T']}$, $B={m['B']}$ and $\\alpha_0={m['alpha0']:.3f}$ from the recipe's laws, over {m['seeds']} seeds{partial},
and evaluated IVF-Flat Recall@10 on the full base with the data set's ground truth
(Table~\\ref{{tab:sift}}). G was run without the per-swap audit.

\\begin{{table}}[t]
\\centering
\\small
\\caption{{SIFT1M: loss relative to Lloyd-25 (mean $\\pm$ s.d.\\ of the per-seed ratio), Recall@10 at 1, 4
and 16 probes, data passes, and seeds in which the seed-loss check fell back.}}\\label{{tab:sift}}
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
Variant & loss ratio & R@10, 1 & 4 & 16 & passes & fell back\\\\
\\midrule
""".replace(",", "{,}", 1) + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    write("sift_section.tex", body)
    write("scope.tex", "We ran one standard real benchmark, SIFT1M, on CPU with NumPy. We make no claim about "
          "speed at the scales e-means targets: we did not run Faiss, GPUs, or the larger benchmarks of~\\cite{bessa2026}.")
    write("limit_data.tex", "(i) Our only large real benchmark is SIFT1M, run with NumPy on CPU and few seeds; we did not "
          "run Faiss, GPUs, or the larger data sets of~\\cite{bessa2026} (e.g.\\ DEEP).")
else:
    for f in ("sift_section.tex",):
        if os.path.exists(os.path.join(GEN, f)):
            os.remove(os.path.join(GEN, f))
    write("scope.tex", "We make no claim about speed or quality at the scales e-means targets: we did not run Faiss, "
          "GPUs, or the large benchmarks of~\\cite{bessa2026}.")
    write("limit_data.tex", "(i) We did not run Faiss, GPUs, or any of the large data sets of~\\cite{bessa2026} "
          "(e.g.\\ SIFT1M, DEEP); their download hosts were not reachable from our environment.")

write("macros.tex", "\n".join(macros) + "\n")
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
print(f"wrote {len(macros)} macros; datasets: {list(data)}")
