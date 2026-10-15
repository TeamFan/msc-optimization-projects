# Algebra and Geometry in Optimization — Saddle Points of Polynomials

Paper study for the *Algebra and Geometry in Optimization* course: report on *The Saddle Point Problem of Polynomials* (J. Nie, Z. Yang and
G. Zhou), plus a small implementation of the method.

Given a polynomial `F(x, y)` and semialgebraic sets `X`, `Y`, find
`(x*, y*)` with `F(x*, y) ≤ F(x*, y*) ≤ F(x, y*)` for all `x ∈ X`, `y ∈ Y`, or
certify that no saddle point exists.

## Implementation

| File | Content |
|---|---|
| `polyopt.py` | Lasserre moment relaxations (moment / localizing matrices as sparse linear maps, solved with Clarabel), flat-truncation test and Henrion–Lasserre extraction of minimizers |
| `saddle_point.py` | The saddle-point algorithm: KKT system with polynomial multiplier maps (simplex and interval constraints), lower-level verification, exclusion cuts, and three examples |

The algorithm alternates

1. **upper level** — minimise `F` (plus a small generic linear tie-breaker)
   over the KKT points of both one-sided problems and the current cuts;
   infeasibility of the relaxation certifies that no saddle point exists;
2. **lower level** — compute `θ₁(y*) = min_x F(x, y*)` and
   `θ₂(x*) = max_y F(x*, y)`; the candidate is a saddle point iff
   `θ₁ = F(x*, y*) = θ₂` (saddle gap `θ₂ − θ₁ = 0`);
3. **exclusion** — a minimiser `u` (maximiser `v`) of a failed lower-level
   problem yields the cut `F(u, y) − F(x, y) ≥ 0` (`F(x, y) − F(x, v) ≥ 0`),
   which every saddle point satisfies.

```bash
python saddle_point.py     # ~15 s; output in results/saddle_point_run.txt
```

| Example | Result |
|---|---|
| 6.1(i): `X = Y = Δ₃`, `F = x₁x₂ + x₂x₃ + x₃y₁ + x₁y₃ + y₁y₂ + y₂y₃` | saddle point `x* = (0, 1, 0)`, `y* = (½, ½, 0)`, value ¼, found in one round (every `y` with `y₂ = ½` is a saddle partner of `x*`, including `(¼, ½, ¼)`) |
| `F = xy + x² − y²` on `[−1, 1]²` | saddle point `(0, 0)` |
| `F = (x − y)²` on `[−1, 1]²` | no saddle point (`min max = 1 > 0 = max min`): two exclusion cuts make the upper level infeasible |

Multiplier maps used: on the simplex `λ₀ = xᵀ∇ₓF`, `λᵢ = ∂F/∂xᵢ − xᵀ∇ₓF`;
on `[−1, 1] = {1 − x² ≥ 0}`, `λ = −x F'(x)/2`.

The report is in `report/` (LaTeX source and PDF).
