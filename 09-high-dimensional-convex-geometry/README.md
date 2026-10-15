# PGMO Course — Isoperimetry and Convex Geometry in High Dimensions

Report on the PGMO / M2 Optimization mini-course on high-dimensional convex
geometry: isotropic position, the thin-shell problem, Bourgain's slicing
problem and the KLS conjecture; concentration on the sphere, log-concave
measures, the Bochner / H⁻¹ framework, and the stochastic-localization and
transport ideas behind the recent bound `Var(|X|²) ≤ C n` for isotropic
log-concave vectors.

* [`report/pgmo_report.pdf`](report/pgmo_report.pdf) — the report (LaTeX source alongside).
* `thin_shell_demo.py` — numerical illustration of the main statements.

```bash
python thin_shell_demo.py    # ~1 s, writes results/thin_shell.png
```

| `Var(|X|²)/n` | n = 10 | 100 | 1000 |
|---|---|---|---|
| isotropic cube | 0.79 | 0.80 | 0.80 |
| isotropic ball | 0.29 | 0.039 | 0.004 |
| isotropic simplex | 2.7 | 3.8 | 3.9 |
| two-Gaussian mixture (isotropic, not log-concave) | 8.6 | 60 | 563 |

For the three convex bodies `Var(|X|²)/n` stays bounded (cube: `E X₁⁴ − 1 =
0.8`; simplex: → 4), so `|X|` lives in a shell of width O(1) around `√n` and
random one-dimensional marginals are close to N(0, 1). The isotropic mixture of
two Gaussians with different radii violates thin shell (`Var(|X|²) ~ n²`) and
its marginals are visibly non-Gaussian — the example used in the report to
show that isotropy alone is not enough.
