# Cooperative Optimization and Learning

Two projects on distributed / federated / private optimization for Nyström
kernel ridge regression.

| Folder | Topic |
|---|---|
| [`coding-project/`](coding-project) | Decentralized, federated and differentially private kernel regression |
| [`paper-review/`](paper-review) | Review and reproduction of *Homomorphically encrypted gradient descent algorithms for quadratic programming* |

## Coding project — distributed kernel regression

`distributed_kernel_regression.py` solves

$$\min_\alpha\ \tfrac{\sigma^2}{2}\alpha^\top K_{mm}\alpha + \tfrac12\|y-K_{nm}\alpha\|^2 + \tfrac{\nu}{2}\|\alpha\|^2$$

split over agents/clients, and compares:

* **Part I — decentralized optimization** on line, small-world and complete
  graphs: DGD, gradient tracking, dual decomposition, ADMM (β sweep), a
  directed/lossy/asynchronous variant with push-sum, and an `n`-scaling study
  up to `n = 30 000`.
* **Part II — federated learning**: FedAvg (batch size, participation, local
  epochs, constant vs. diminishing rates) and SCAFFOLD.
* **Part III — differential privacy**: DP-DGD with theorem-aligned step,
  consensus and noise schedules for ε ∈ {0.1, 1, 10}, plus a larger
  `(n, m, a) = (1000, 33, 100)` setting.

```bash
cd coding-project
python distributed_kernel_regression.py --mode full      # ~1 min; --mode quick for a smoke test
python report/make_tables.py                              # regenerate report/tables/*.tex
```

Figures and `metrics_summary.csv` are written to `results/`; the report
(`report/coding_project_report.tex`) reads its figures from there.
`make_tables.py` recomputes the step-size checks of Table 2 directly from the
data (L-smoothness / strong convexity constants, spectral radius of the
gradient-tracking iteration restricted to its invariant subspace, the
largest stable GT step by bisection, the dual-ascent bound and the federated
smoothness constant).

Datasets: `data/first_database.pkl` (1-D noisy regression samples) and
`data/second_database.pkl` (five client datasets).

**Reproducibility.** Every random component is seeded from `--seed`
(default 42). Per-experiment seed offsets are derived with CRC32 of the
experiment tag, so results are identical from one run to the next (Python's
built-in `hash()` of strings is salted per process and must not be used for
seeding).

## Paper review — homomorphic encryption

`he_qp.py` reproduces the depth-limited GD/AGD trade-off of the reviewed
paper with a simulated CKKS backend (multiplicative depth and ciphertext
operation accounting), and extends it to the kernel regression QP of the
coding project.

```bash
cd paper-review
python he_qp.py all --mode full --outdir results   # paper/, kernel/ and validate/ sub-folders
```

`results/validate/validation_summary.json` checks the depth/cost formulas
and the QP construction. The review itself is `report/paper_review.{tex,pdf}`.
