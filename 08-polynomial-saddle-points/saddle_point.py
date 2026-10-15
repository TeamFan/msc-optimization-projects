"""Saddle points of polynomials over semialgebraic sets (Nie-Yang-Zhou type algorithm).

Find (x*, y*) in X x Y with F(x*, y) <= F(x*, y*) <= F(x, y*), or certify that
none exists:

1. **Upper level** - minimise F over X x Y subject to the KKT relations of both
   one-sided problems, where the Lagrange multipliers are replaced by
   polynomial maps lambda(x, y), mu(x, y), plus the exclusion inequalities
   collected so far. Infeasibility certifies that no saddle point exists.
2. **Lower level** - for each candidate compute
   theta_1(y*) = min_{x in X} F(x, y*) and theta_2(x*) = max_{y in Y} F(x*, y);
   the candidate is a saddle point iff theta_1 = F(x*, y*) = theta_2.
3. **Exclusion** - a minimiser u of the failed x-problem gives the valid cut
   F(u, y) - F(x, y) >= 0, a maximiser v of the failed y-problem gives
   F(x, y) - F(x, v) >= 0; both keep every saddle point and remove the candidate.

All polynomial subproblems are solved with Lasserre moment relaxations
(``polyopt.MomentRelaxation``) and flat truncation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import warnings

import numpy as np
import sympy as sp

from polyopt import MomentRelaxation

TOL = 1e-4
TIE_BREAK = 1e-2


@dataclass
class SaddleProblem:
    name: str
    x: list
    y: list
    F: sp.Expr
    x_eqs: list
    x_ineqs: list
    y_eqs: list
    y_ineqs: list
    kkt: list            # polynomial equalities from multiplier maps (stationarity + complementarity)
    kkt_ineqs: list      # sign conditions on the multiplier maps
    cuts: list = field(default_factory=list)


def simplex_kkt(F, v, minimise: bool):
    """KKT relations on the simplex {v >= 0, sum v = 1} with the multiplier maps
    lambda_0 = v^T grad F,  lambda_i = dF/dv_i - v^T grad F."""
    grad = [sp.diff(F, vi) for vi in v]
    lam0 = sum(vi * gi for vi, gi in zip(v, grad))
    lam = [gi - lam0 for gi in grad]
    eqs = [sp.expand(li * vi) for li, vi in zip(lam, v)]            # complementarity lambda_i v_i = 0
    signs = [sp.expand(li if minimise else -li) for li in lam]       # lambda_i >= 0 (min) / <= 0 (max)
    return eqs, signs


def interval_kkt(F, v, minimise: bool):
    """KKT relations on [-1, 1] = {1 - v^2 >= 0} with the multiplier map lambda = -v F'(v) / 2."""
    (vi,) = v
    lam = -vi * sp.diff(F, vi) / 2
    eqs = [sp.expand(sp.diff(F, vi) + 2 * lam * vi), sp.expand(lam * (1 - vi**2))]
    signs = [sp.expand(lam if minimise else -lam)]
    return eqs, signs


def lower_level(problem: SaddleProblem, x_star, y_star):
    """theta_1(y*) = min_x F(x, y*), theta_2(x*) = max_y F(x*, y), with optimisers."""
    Fy = problem.F.subs(dict(zip(problem.y, y_star)))
    Fx = problem.F.subs(dict(zip(problem.x, x_star)))
    r1 = MomentRelaxation(problem.x, Fy, eqs=problem.x_eqs, ineqs=problem.x_ineqs).solve()
    r2 = MomentRelaxation(problem.y, -Fx, eqs=problem.y_eqs, ineqs=problem.y_ineqs).solve()
    return r1.value, r1.minimizers, -r2.value, r2.minimizers


def find_saddle_points(problem: SaddleProblem, max_rounds: int = 20, verbose: bool = True, seed: int = 0):
    z = problem.x + problem.y
    # Generic linear tie-breaker: the KKT set usually contains a continuum of points with the
    # same value of F; a small generic perturbation selects a unique candidate (flat, rank one).
    tie = np.random.default_rng(seed).standard_normal(len(z))
    for rnd in range(1, max_rounds + 1):
        upper = MomentRelaxation(
            z, problem.F + TIE_BREAK * sum(float(c) * zi for c, zi in zip(tie, z)),
            eqs=problem.x_eqs + problem.y_eqs + problem.kkt,
            ineqs=problem.x_ineqs + problem.y_ineqs + problem.kkt_ineqs + problem.cuts,
        ).solve(k_max=4)
        if upper.status == "infeasible":
            if verbose:
                print(f"  round {rnd}: upper-level relaxation infeasible -> no saddle point exists")
            return {"status": "no saddle point", "rounds": rnd, "cuts": len(problem.cuts)}
        if not upper.flat:
            raise RuntimeError(f"upper level not flat up to order 4 (status {upper.status})")
        found = []
        for cand in upper.minimizers:
            xs, ys = cand[:len(problem.x)], cand[len(problem.x):]
            fval = float(problem.F.subs(dict(zip(z, cand))))
            feasible = all(abs(float(e.subs(dict(zip(z, cand))))) < 1e-5 for e in problem.x_eqs + problem.y_eqs) and \
                all(float(g.subs(dict(zip(z, cand)))) > -1e-5 for g in problem.x_ineqs + problem.y_ineqs)
            if not feasible:
                raise RuntimeError(f"extracted candidate {cand} is not feasible")
            th1, mins, th2, maxs = lower_level(problem, xs, ys)
            gap = th2 - th1
            if verbose:
                print(f"  round {rnd}: candidate x={np.round(xs, 4) + 0.0}, y={np.round(ys, 4) + 0.0}, F={round(fval, 6) + 0.0:.6f}, "
                      f"theta1={th1:.6f}, theta2={th2:.6f}, saddle gap={gap:.2e}")
            if fval - th1 < TOL and th2 - fval < TOL:
                found.append((xs, ys, fval))
                continue
            if fval - th1 >= TOL:
                for u in mins:
                    problem.cuts.append(sp.expand(problem.F.subs(dict(zip(problem.x, u))) - problem.F))
            if th2 - fval >= TOL:
                for v in maxs:
                    problem.cuts.append(sp.expand(problem.F - problem.F.subs(dict(zip(problem.y, v)))))
        if found:
            return {"status": "saddle point found", "points": found, "rounds": rnd, "cuts": len(problem.cuts)}
    return {"status": "max rounds reached", "rounds": max_rounds, "cuts": len(problem.cuts)}


def example_6_1_i() -> SaddleProblem:
    """X = Y = simplex in R^3, F = x1x2 + x2x3 + x3y1 + x1y3 + y1y2 + y2y3."""
    x = list(sp.symbols("x1:4"))
    y = list(sp.symbols("y1:4"))
    F = x[0] * x[1] + x[1] * x[2] + x[2] * y[0] + x[0] * y[2] + y[0] * y[1] + y[1] * y[2]
    kx, sx = simplex_kkt(F, x, minimise=True)
    ky, sy = simplex_kkt(F, y, minimise=False)
    return SaddleProblem("Example 6.1(i): simplex x simplex", x, y, F,
                         [sum(x) - 1], list(x), [sum(y) - 1], list(y), kx + ky, sx + sy)


def example_no_saddle() -> SaddleProblem:
    """F = (x - y)^2 on [-1, 1]^2: min-max = 1 > 0 = max-min, so no saddle point exists."""
    x, y = sp.symbols("x y")
    F = (x - y) ** 2
    kx, sx = interval_kkt(F, [x], minimise=True)
    ky, sy = interval_kkt(F, [y], minimise=False)
    return SaddleProblem("(x - y)^2 on [-1, 1]^2", [x], [y], F, [], [1 - x**2], [], [1 - y**2], kx + ky, sx + sy)


def example_bilinear_box() -> SaddleProblem:
    """F = x y + x^2 - y^2 on [-1, 1]^2: convex-concave, unique saddle point (0, 0)."""
    x, y = sp.symbols("x y")
    F = x * y + x**2 - y**2
    kx, sx = interval_kkt(F, [x], minimise=True)
    ky, sy = interval_kkt(F, [y], minimise=False)
    return SaddleProblem("x y + x^2 - y^2 on [-1, 1]^2", [x], [y], F, [], [1 - x**2], [], [1 - y**2], kx + ky, sx + sy)


def main() -> None:
    warnings.filterwarnings("ignore", message="Solution may be inaccurate")
    for build in (example_6_1_i, example_bilinear_box, example_no_saddle):
        problem = build()
        print(f"\n=== {problem.name} ===")
        res = find_saddle_points(problem)
        print(f"  -> {res['status']} after {res['rounds']} round(s), {res['cuts']} exclusion cut(s)")
        for xs, ys, f in res.get("points", []):
            print(f"     x* = {np.round(xs, 4) + 0.0}, y* = {np.round(ys, 4) + 0.0}, F* = {round(f, 6) + 0.0:.6f}")


if __name__ == "__main__":
    main()
