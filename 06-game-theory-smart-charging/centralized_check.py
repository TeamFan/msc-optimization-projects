"""Compare BRD (water-filling best responses) with the centralized LP optimum.

The centralized problem min_{l in L} sum_t f_CO2(L(t)) is a linear program because
f_CO2 is convex and piecewise linear. It is solved with HiGHS (scipy.optimize.linprog)
for the four scenarios of the report, and the BRD profiles are scored with the same
function. Output: results/centralized_check.csv.

    python centralized_check.py    # about 10 seconds
"""
import csv, os, sys, io, contextlib
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)
import smart_charging as sc  # noqa: E402

T = sc.TIME_SLOTS_PER_DAY
H = sc.SLOT_HOURS


def segments(ctx):
    """f(L) = max_k (a_k L + b_k) for the convex piecewise-linear f_co2 (g/h)."""
    p_hat, rates = ctx["p_hat"], ctx["cumulative_rates"]
    slopes, inter = [], []
    f = ctx["f_co2"]
    for i, r in enumerate(rates):
        a = 1000.0 * r
        x0 = p_hat[i]
        slopes.append(a)
        inter.append(f(x0) - a * x0)
    return np.array(slopes), np.array(inter)


def central_opt(evs, nf, scale, ctx):
    a, b = segments(ctx)
    var = {}
    for j, r in enumerate(evs):
        for t in sc._available_slots(r["time_slot_arr"], r["time_slot_dep"]):
            var[(j, t - 1)] = len(var)
    nx = len(var)
    nv = nx + T  # x vars (kW) + epigraph s_t (g/h)
    c = np.zeros(nv); c[nx:] = H
    k2mw = scale / 1000.0
    # epigraph: a_k (nf_t + k2mw*sum_j x_jt) + b_k <= s_t
    A = lil_matrix((len(a) * T, nv)); bub = np.zeros(len(a) * T)
    byt = [[] for _ in range(T)]
    for (j, t), i in var.items():
        byt[t].append(i)
    row = 0
    for t in range(T):
        for k in range(len(a)):
            for i in byt[t]:
                A[row, i] = a[k] * k2mw
            A[row, nx + t] = -1.0
            bub[row] = -(a[k] * nf[t] + b[k])
            row += 1
    Aeq = lil_matrix((len(evs), nv)); beq = np.zeros(len(evs))
    for (j, t), i in var.items():
        Aeq[j, i] = H
    for j, r in enumerate(evs):
        beq[j] = float(r["energy_need (kWh)"])
    bounds = [(0, sc.EV_MAX_POWER_KW)] * nx + [(None, None)] * T
    res = linprog(c, A_ub=csr_matrix(A), b_ub=bub, A_eq=csr_matrix(Aeq), b_eq=beq,
                  bounds=bounds, method="highs")
    assert res.status == 0, res.message
    return res.fun


rows = []
for year in (2019, 2020):
    for season in ("summer", "winter"):
        evs, nf = sc.load_data(year=year, season=season, ev_limit=sc.DEFAULT_SIM_FLEET_SIZE)
        evs = evs[: sc.DEFAULT_SIM_FLEET_SIZE]
        scale = sc.FLEET_SIZE / len(evs)
        ctx = sc.get_model_context()
        with contextlib.redirect_stdout(io.StringIO()):
            brd, hist = sc.brd_coordination(evs, nf, scale, ctx=ctx, max_k=10)
        brd_co2 = sc.calculate_co2_scores(nf, brd, scale, ctx)
        pc = [sc.policy_plug_and_charge(r) for r in evs]
        pc_co2 = sc.calculate_co2_scores(nf, pc, scale, ctx)
        opt = central_opt(evs, nf, scale, ctx)
        rows.append((year, season, len(hist), pc_co2 / 1e6, brd_co2 / 1e6, opt / 1e6,
                     (brd_co2 - opt) / opt * 100, (pc_co2 - opt) / pc_co2 * 100))

print(f"{'scenario':<14}{'sweeps':>7}{'PC (t)':>12}{'BRD (t)':>12}{'LP opt (t)':>12}{'BRD gap %':>11}{'LP vs PC %':>11}")
for y, s, k, pc, br, op, g, red in rows:
    print(f"{y} {s:<9}{k:>7}{pc:>12.0f}{br:>12.0f}{op:>12.0f}{g:>11.2e}{red:>11.2f}")

os.makedirs("results", exist_ok=True)
with open("results/centralized_check.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["year", "season", "brd_sweeps", "pc_co2_t", "brd_co2_t", "lp_opt_co2_t",
                "brd_relative_gap_percent", "lp_reduction_vs_pc_percent"])
    for r in rows:
        w.writerow([r[0], r[1], r[2]] + [f"{v:.6g}" for v in r[3:]])
print("Saved results/centralized_check.csv")
