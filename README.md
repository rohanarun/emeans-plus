# emeans-plus

Code and paper for **"Stabilizing Exponential Moving Average k-Means: Repairing Two Counterexamples"**
(Rohan Arun, Super Powers AI, 2026).

[e-means](https://arxiv.org/abs/2611.09501) (Bessa et al., 2026) is an exponential-moving-average
k-means. [Assylbekov (2026)](https://arxiv.org/abs/2611.22065) gave two counterexamples to its
published recipe ([reference code](https://github.com/zh3nis/e-means-counterexamples)). This
repository tests which minimal change fixes which counterexample:

| Change | Counterexample 1 (revival) | Counterexample 2 (forgetting) | Guarantee |
|---|---|---|---|
| **C** seed-loss check: if the output is worse than the seeds, return Lloyd from the seeds | fixed | not fixed | output loss ≤ seed loss on every input, so the k-means++ `8(ln k + 2)` bound holds and the expected-loss ratio to Lloyd is bounded |
| **τ** pseudo-count floor (initial counts τ = 1 instead of 1e-10) | not fixed | fixed | one-step bound; τ\* = 0.1927 for Counterexample 2, matching the observed transition |
| **G** cost-guarded revival (swap only if the loss on a reservoir sample drops) | fixed | not fixed | none on the full data (many accepted swaps raise it) |
| **S** Lloyd polish to convergence | — | — | output is a Lloyd fixed point |

**e-means+** = τ + C, with the recipe's mass revival kept. The paper also reports what these
changes cost on non-adversarial data. Some of the costs are real: τ = 1 raises the loss on the
digits data set with k = 64, and G helps on a heavy-tailed synthetic set but not on digits.

This is a NumPy study implementation, not the Faiss `contrib/e_means` code, and it was not run
on the large benchmarks of the e-means paper.

## Install and test

```bash
pip install -e ".[dev]"        # numpy, scikit-learn, pytest
pytest -q
```

## Reproduce the paper

```bash
python benchmarks/run_experiments.py counterexamples   # 1-D reference recipe (a few minutes)
python benchmarks/run_experiments.py port               # NumPy vs reference agreement
python benchmarks/run_experiments.py synthetic          # rare-tail data, 10 seeds
python benchmarks/run_experiments.py digits             # scikit-learn digits, 10 seeds
python benchmarks/run_experiments.py tau                # tau sensitivity
python benchmarks/run_experiments.py ann                # heavy-tailed ANN, 5 seeds (slow)
python benchmarks/make_tables.py                        # writes paper/generated/*.tex
cd paper && latexmk -pdf main.tex
```

### SIFT1M (run on your own machine)

```bash
pip3 install numpy
caffeinate -i python3 benchmarks/run_experiments.py sift      # macOS: caffeinate keeps the Mac awake
python3 benchmarks/make_tables.py && python3 benchmarks/check_claims.py
cd paper && latexmk -pdf main.tex
```

This downloads `sift.tar.gz` (about 160 MB) from the TEXMEX FTP server into `data/sift/`, trains
on all 1M base vectors with k = 1024 over 3 seeds, and evaluates Recall@10 with the official ground
truth. Results are saved after every seed to `results/sift1m.json`, and `make_tables.py` adds a
SIFT1M section to the paper. For a quick timing test first:
`python3 benchmarks/run_experiments.py sift --train-n 100000 --seeds 1`. If the FTP download
fails, download `sift.tar.gz` from http://corpus-texmex.irisa.fr/ by hand, put it in `data/sift/`
and run again.

Every table and every number in the paper is generated from `results/*.json` by
`make_tables.py`, which also asserts the claims it relies on (for example, that τ\* lies inside
the observed transition).

## Use

```python
import numpy as np
from emeans_plus import emeans_plus, kmeanspp_init, compute_loss

X = np.random.default_rng(0).normal(size=(10_000, 32))
C0 = kmeanspp_init(X, k=64, rng=np.random.default_rng(1))
C, info = emeans_plus(X, C0, T=2)                          # tau + C
C, info = emeans_plus(X, C0, T=2, safety=True)             # + polish (Lloyd fixed point)
C, info = emeans_plus(X, C0, T=2, revival="guarded")       # + cost-guarded revival
print(compute_loss(X, C), info.fell_back, info.passes)
```

## Layout

- `emeans_plus/reference_1d.py`: line-for-line port of the reference 1-D recipe, plus τ and G
- `emeans_plus/core.py`: multi-dimensional recipe, τ, G, C, S
- `benchmarks/run_experiments.py`: all experiments, writes `results/*.json`
- `benchmarks/make_tables.py`: tables and numbers for the paper
- `tests/`: checks of the reference behaviour and of the guarantees
- `paper/`: LaTeX source and PDF

## License

MIT.
