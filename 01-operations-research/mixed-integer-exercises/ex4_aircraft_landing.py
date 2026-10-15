"""Aircraft landing problem (single runway) as a mixed-integer linear program.

Data: OR-Library instance ``airland1`` (10 planes, Beasley et al., 2000).
For each plane p we know the earliest / target / latest landing times
(EAT_p, TAT_p, LAT_p), the per-minute penalties for landing early (EP_p) or
late (LP_p), and the minimum separation l_pq required when p lands before q.

Model
-----
    min   sum_p EP_p e_p + LP_p l_p
    s.t.  EAT_p <= t_p <= LAT_p
          t_p - TAT_p = l_p - e_p,           e_p, l_p >= 0
          t_q - t_p >= l_pq - M_pq (1 - z_pq)   (p before q)
          t_p - t_q >= l_qp - M_qp z_pq         (q before p)
          z_pq in {0, 1}                         for p < q

with the big-M constants M_pq = LAT_p + l_pq - EAT_q, which make the
inactive disjunct redundant over the time windows. The known optimal value
for airland1 is 700.
"""

from ortools.linear_solver import pywraplp


def solve_aircraft_landing(EAT, TAT, LAT, EP, LP, sep, verbose=True):
    """Solve the single-runway landing problem.

    EAT, TAT, LAT, EP, LP: lists of length n.
    sep: n x n matrix, sep[p][q] = minimum separation if p lands before q.
    All times are in minutes.
    """
    n = len(EAT)
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver not available in this environment.")

    # --- Variables ---
    t = [solver.NumVar(EAT[p], LAT[p], f"t[{p}]") for p in range(n)]       # landing times
    e = [solver.NumVar(0.0, solver.infinity(), f"e[{p}]") for p in range(n)]  # earliness
    l = [solver.NumVar(0.0, solver.infinity(), f"l[{p}]") for p in range(n)]  # lateness
    z = {(p, q): solver.BoolVar(f"z[{p},{q}]") for p in range(n) for q in range(p + 1, n)}

    # --- Objective: total earliness and lateness penalties ---
    solver.Minimize(sum(EP[p] * e[p] + LP[p] * l[p] for p in range(n)))

    # --- Deviation from target time: t - TAT = l - e ---
    for p in range(n):
        solver.Add(t[p] - TAT[p] == l[p] - e[p])

    # --- Disjunctive separation constraints (big-M linearisation) ---
    for p in range(n):
        for q in range(p + 1, n):
            M_pq = LAT[p] + sep[p][q] - EAT[q]
            M_qp = LAT[q] + sep[q][p] - EAT[p]
            # z = 1 (p before q): t_q >= t_p + l_pq
            solver.Add(t[q] - t[p] >= sep[p][q] - M_pq * (1 - z[(p, q)]))
            # z = 0 (q before p): t_p >= t_q + l_qp
            solver.Add(t[p] - t[q] >= sep[q][p] - M_qp * z[(p, q)])

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        print("No feasible solution found.")
        return None

    landings = sorted(
        ((p + 1, t[p].solution_value(), e[p].solution_value(), l[p].solution_value()) for p in range(n)),
        key=lambda row: row[1],
    )
    total_penalty = solver.Objective().Value()

    if verbose:
        print(f"Status: {'OPTIMAL' if status == pywraplp.Solver.OPTIMAL else 'FEASIBLE'}")
        print(f"Total penalty = {total_penalty:.2f}\n")
        print("Landing order and times:")
        for idx, time_p, early, late in landings:
            print(f"  Plane {idx:2d}: t={time_p:7.2f}  early={early:6.2f}  late={late:6.2f}")

    return {"objective": total_penalty, "schedule": landings}


# ------------------- OR-Library airland1 data -------------------
EAT = [129, 195, 89, 96, 110, 120, 124, 126, 135, 160]
TAT = [155, 258, 98, 106, 123, 135, 138, 140, 150, 180]
LAT = [559, 744, 510, 521, 555, 576, 577, 573, 591, 657]

# Penalties per minute: planes 1-2 cost 10, the others 30.
EP = [10, 10] + [30] * 8
LP = [10, 10] + [30] * 8

# Separation matrix sep[p][q] (airland1): 3 min between planes 1 and 2,
# 15 min whenever plane 1 or 2 is involved otherwise, 8 min between the others.
SEP = [
    [0, 3, 15, 15, 15, 15, 15, 15, 15, 15],
    [3, 0, 15, 15, 15, 15, 15, 15, 15, 15],
    [15, 15, 0, 8, 8, 8, 8, 8, 8, 8],
    [15, 15, 8, 0, 8, 8, 8, 8, 8, 8],
    [15, 15, 8, 8, 0, 8, 8, 8, 8, 8],
    [15, 15, 8, 8, 8, 0, 8, 8, 8, 8],
    [15, 15, 8, 8, 8, 8, 0, 8, 8, 8],
    [15, 15, 8, 8, 8, 8, 8, 0, 8, 8],
    [15, 15, 8, 8, 8, 8, 8, 8, 0, 8],
    [15, 15, 8, 8, 8, 8, 8, 8, 8, 0],
]


if __name__ == "__main__":
    solve_aircraft_landing(EAT, TAT, LAT, EP, LP, SEP)
