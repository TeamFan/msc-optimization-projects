# Derivative-Free Optimization — CMA-ES and Constraint Handling

## Contents

| Folder | Topic |
|---|---|
| [`cmaes-exercises/`](cmaes-exercises) | Class exercises with `pycma`: ill-conditioned and rotated quadratics, CMA-ES vs. (1+1)-ES, population size on Rastrigin |
| [`al-cmaes-project/`](al-cmaes-project) | Course project: Augmented Lagrangian constraint handling in CMA-ES (report + experiments) |

## CMA-ES exercises

`solve_cmaes_exercises.py` runs every question and writes `results/`:

* **Q1–Q6** — CMA-ES on `f_elli`, `f_tablet` and the randomly rotated
  `f_ellirot` in dimension 10 (rotation invariance: `f_elli` and `f_ellirot`
  need comparable budgets, about 5.0k and 4.5k evaluations to reach 1e-12).
* **Q7** — evaluations to reach 1e-6 on `f_elli`: CMA-ES (~2.4k) vs.
  (1+1)-ES with the one-fifth success rule (~1.6e5), i.e. covariance
  learning saves almost two orders of magnitude.
* **Q9** — Rastrigin: success probability as a function of the population
  size multiplier (best around 16× the default λ).

```bash
cd cmaes-exercises && python solve_cmaes_exercises.py
```

## AL-CMA-ES project

| File | Description |
|---|---|
| `lagrangian_comparison.py` | Deterministic augmented Lagrangian vs. standard Lagrangian/penalty on a convex QP and a non-convex double well (landscape shaping, multiplier sweeps) |
| `alcmaes_comparison.py` | CMA-ES constraint handling benchmark: vanilla, death penalty, static penalty, adaptive penalty, AL-CMA-ES and a surrogate-assisted AL on (1) sphere + linear constraint, (2) ill-conditioned ellipsoid, (3) non-smooth (kinked) constraint; 5 seeded runs each |
| `report_figures.py` | Illustrations of the penalty-update Conditions 1 and 2 and the AL-CMA-ES vs. static penalty convergence plots used in Section 3 of the report |
| `report/al_cmaes_report.{tex,pdf}` | Report |

```bash
cd al-cmaes-project
python lagrangian_comparison.py   # -> results/lagrangian_comparison/
python alcmaes_comparison.py      # -> results/alcmaes_comparison/
python report_figures.py          # -> results/report_figures/
```

All runs are seeded and deterministic for a given environment; the tables and
figures of the report were produced with `pycma` 4.5.0, NumPy 2.4 and Python 3.11.
Evaluation counts can shift slightly with other `pycma` versions or
linear-algebra backends, but the conclusions — AL-CMA-ES is the most
sample-efficient feasible method on smooth constraints, static penalties stall
at `1/(1+β)`, and the kinked constraint makes the multiplier oscillate unless a
surrogate of `g` is used — do not depend on them.

The LaTeX source compiles from its own folder (`graphicspath` points to
the project root, so figures are read from `results/`).
