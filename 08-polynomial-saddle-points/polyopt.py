"""Lasserre moment relaxations for polynomial optimization.

    min p(z)  s.t.  phi_i(z) = 0,  psi_j(z) >= 0,        z in R^n

The order-k relaxation optimises over truncated moment vectors w = (w_alpha),
|alpha| <= 2k, with w_0 = 1:

    min <p, w>  s.t.  M_k(w) >= 0,  L_{psi_j}^{(k)}(w) >= 0,  L_{phi_i}^{(k)}(w) = 0.

``solve`` increases k until flat truncation, rank M_t(w) = rank M_{t-d}(w),
holds; the minimizers are then extracted with the Henrion-Lasserre algorithm.
Polynomials are SymPy expressions in the given variables.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import sympy as sp
from scipy import sparse
from scipy.linalg import qr, schur


def monomials(n: int, d: int) -> list[tuple[int, ...]]:
    """Exponents of all monomials of degree <= d in n variables, graded order."""
    out = []
    for deg in range(d + 1):
        for c in itertools.combinations_with_replacement(range(n), deg):
            e = [0] * n
            for i in c:
                e[i] += 1
            out.append(tuple(e))
    return out


def to_dict(expr, variables) -> dict[tuple[int, ...], float]:
    poly = sp.Poly(sp.expand(expr), *variables)
    return {tuple(m): float(c) for m, c in poly.terms()}


def degree(expr, variables) -> int:
    return sp.Poly(sp.expand(expr), *variables).total_degree()


def independent_rows(E: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Indices of a maximal set of linearly independent rows (pivoted QR of E^T)."""
    if E.shape[0] == 0:
        return np.arange(0)
    _, R, piv = qr(E.T, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > tol * max(1.0, diag[0]))) if diag.size else 0
    return np.sort(piv[:rank])


def add(a, b):
    return tuple(i + j for i, j in zip(a, b))


@dataclass
class Relaxation:
    status: str
    order: int
    value: float = math.nan
    flat: bool = False
    rank: int = 0
    minimizers: list[np.ndarray] = field(default_factory=list)


class MomentRelaxation:
    def __init__(self, variables, objective, eqs=(), ineqs=()):
        self.vars = list(variables)
        self.n = len(self.vars)
        self.obj = to_dict(objective, self.vars)
        self.eqs = [(to_dict(e, self.vars), degree(e, self.vars)) for e in eqs]
        self.ineqs = [(to_dict(g, self.vars), degree(g, self.vars)) for g in ineqs]
        self.max_deg = max([degree(objective, self.vars)] + [d for _, d in self.eqs] + [d for _, d in self.ineqs] + [1])
        self.d = max(1, math.ceil(max([d for _, d in self.eqs + self.ineqs] + [1]) / 2))

    def _build(self, k: int):
        basis2k = monomials(self.n, 2 * k)
        index = {m: i for i, m in enumerate(basis2k)}
        N = len(basis2k)
        w = cp.Variable(N)
        cons = [w[0] == 1]

        def loc_matrix(poly: dict, order: int):
            """Localizing matrix L_poly as a sparse linear map of w, reshaped to (s, s)."""
            b = monomials(self.n, order)
            s = len(b)
            rows, cols, data = [], [], []
            for i, a in enumerate(b):
                for j, c in enumerate(b):
                    ac = add(a, c)
                    for mono, coef in poly.items():
                        rows.append(i * s + j)
                        cols.append(index[add(ac, mono)])
                        data.append(coef)
            A = sparse.csr_matrix((data, (rows, cols)), shape=(s * s, N))
            return cp.reshape(A @ w, (s, s), order="C")

        cons.append(loc_matrix({(0,) * self.n: 1.0}, k) >> 0)
        for poly, dg in self.ineqs:
            order = k - math.ceil(dg / 2)
            if order >= 0:
                L = loc_matrix(poly, order)
                cons.append(0.5 * (L + L.T) >> 0)
        rows, cols, data, r = [], [], [], 0
        for poly, dg in self.eqs:
            for a in monomials(self.n, 2 * k - dg):
                for m, c in poly.items():
                    rows.append(r)
                    cols.append(index[add(a, m)])
                    data.append(c)
                r += 1
        if r:
            E = sparse.csr_matrix((data, (rows, cols)), shape=(r, N)).toarray()
            E = E[independent_rows(E)]          # interior-point solvers need full row rank
            cons.append(sparse.csr_matrix(E) @ w == 0)
        c_obj = np.zeros(N)
        for m, c in self.obj.items():
            c_obj[index[m]] += c
        return cp.Problem(cp.Minimize(c_obj @ w), cons), w, index

    def solve_order(self, k: int, rank_tol: float = 1e-4) -> Relaxation:
        prob, w, index = self._build(k)
        try:
            # Moment relaxations rarely satisfy Slater's condition; stronger static
            # regularisation keeps the interior-point KKT systems factorizable.
            prob.solve(solver=cp.CLARABEL, static_regularization_constant=1e-6, equilibrate_enable=False)
        except cp.error.SolverError:
            prob.solve(solver=cp.SCS, eps=1e-10, max_iters=500_000)
        if prob.status in ("infeasible", "infeasible_inaccurate"):
            return Relaxation("infeasible", k)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return Relaxation(prob.status, k)
        wv = np.asarray(w.value)
        res = Relaxation("optimal", k, value=float(prob.value))
        # Flat truncation: rank M_t = rank M_{t-d} for some d <= t <= k.
        for t in range(self.d, k + 1):
            r_t = self._rank(wv, index, t, rank_tol)
            r_low = self._rank(wv, index, t - self.d, rank_tol)
            if r_t == r_low:
                res.flat, res.rank = True, r_t
                res.minimizers = self._extract(wv, index, t, r_t, rank_tol)
                break
        return res

    def solve(self, k_min: int | None = None, k_max: int = 4) -> Relaxation:
        k0 = max(math.ceil(self.max_deg / 2), self.d) if k_min is None else k_min
        res = Relaxation("not solved", k0)
        for k in range(k0, k_max + 1):
            res = self.solve_order(k)
            if res.status == "infeasible" or res.flat:
                return res
        return res

    # -- helpers ------------------------------------------------------------
    def _moment_matrix(self, wv, index, t):
        b = monomials(self.n, t)
        return np.array([[wv[index[add(a, c)]] for c in b] for a in b]), b

    def _rank(self, wv, index, t, tol):
        M, _ = self._moment_matrix(wv, index, t)
        s = np.linalg.svd(M, compute_uv=False)
        return int(np.sum(s > tol * max(1.0, s[0])))

    def _extract(self, wv, index, t, r, tol):
        """Henrion-Lasserre extraction of r atoms from the flat moment matrix M_t."""
        M, basis = self._moment_matrix(wv, index, t)
        vals, vecs = np.linalg.eigh(M)
        order = np.argsort(vals)[::-1][:r]
        V = vecs[:, order] * np.sqrt(np.maximum(vals[order], 0.0))
        if r == 1:
            return [V[1:1 + self.n, 0] / V[0, 0]]
        # Column echelon form of V (Gaussian elimination with pivoting on rows).
        U = V.T.copy()
        pivots = []
        row = 0
        for col in range(U.shape[1]):
            if row >= r:
                break
            p = row + int(np.argmax(np.abs(U[row:, col])))
            if abs(U[p, col]) < 1e-6:
                continue
            U[[row, p]] = U[[p, row]]
            U[row] /= U[row, col]
            for i in range(r):
                if i != row:
                    U[i] -= U[i, col] * U[row]
            pivots.append(col)
            row += 1
        U = U.T                              # rows indexed by monomials in basis, U[pivots] = I
        base = [basis[i] for i in pivots]
        pos = {m: i for i, m in enumerate(basis)}
        rng = np.random.default_rng(0)
        coeffs = rng.random(self.n)
        coeffs /= coeffs.sum()
        N = []
        for i in range(self.n):
            e = tuple(1 if j == i else 0 for j in range(self.n))
            N.append(np.array([U[pos[add(m, e)]] for m in base]))
        Ncomb = sum(c * Ni for c, Ni in zip(coeffs, N))
        _, Qs = schur(Ncomb, output="real")
        return [np.array([float(Qs[:, j] @ Ni @ Qs[:, j]) for Ni in N]) for j in range(r)]
