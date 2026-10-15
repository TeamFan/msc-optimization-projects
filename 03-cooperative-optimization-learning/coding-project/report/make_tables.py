"""Generate the LaTeX tables of the report from the experiment outputs.

Reads ``results/metrics_summary.csv`` (written by
``distributed_kernel_regression.py``) and recomputes the step-size checks
quoted in Table 2 (smoothness/strong-convexity constants, gradient-tracking
spectral radius, dual-ascent bound, federated smoothness constant). Writes
``tables/table_setup.tex``, ``tables/table_theory.tex`` and
``tables/table_results.tex``.

Usage (from the ``coding-project`` folder)::

    python report/make_tables.py [--results results/metrics_summary.csv]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import distributed_kernel_regression as dkr  # noqa: E402

TABLES_DIR = Path(__file__).resolve().parent / "tables"

SIGMA, NU, SEED = 0.5, 1.0, 42
N_AGENTS, N_BASE, M_BASE = 5, 100, 10


def fmt(value: float, digits: int = 3) -> str:
    """Scientific notation with ``digits`` significant digits (siunitx friendly)."""
    value = float(value)
    if not np.isfinite(value):
        return "--"
    if abs(value) < 1e-15:
        return "0"
    return f"{value:.{max(digits - 1, 0)}e}"


# ---------------------------------------------------------------------------
# Step-size verification
# ---------------------------------------------------------------------------

def load_pickle(name: str):
    with open(dkr.project_file_path(name), "rb") as handle:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return pickle.load(handle)


def part1_problem():
    """Rebuild the Part I local quadratics exactly as in the experiment script."""
    rng = np.random.default_rng(SEED)
    x_big, y_big = (np.asarray(v, dtype=float) for v in load_pickle("first_database.pkl"))
    x_n, y_n = x_big[:N_BASE], y_big[:N_BASE]
    x_m = x_n[rng.choice(N_BASE, size=M_BASE, replace=False)]
    kmm = dkr.kernel(x_m, x_m)
    block = N_BASE // N_AGENTS
    h_list, b_list = [], []
    for i in range(N_AGENTS):
        h_i, b_i = dkr.local_quadratic_terms(
            x_n[i * block:(i + 1) * block], y_n[i * block:(i + 1) * block], x_m, kmm, SIGMA, NU, N_AGENTS
        )
        h_list.append(h_i)
        b_list.append(b_i)
    return h_list, b_list


def gradient_tracking_operator(h_list, w: np.ndarray, eta: float) -> np.ndarray:
    """Linear map (x, g) -> (x+, g+) of gradient tracking on quadratic local costs.

    x+ = W x - eta g,   g+ = W g + H x+ - H x   with H = blkdiag(H_i).
    """
    a, m = len(h_list), h_list[0].shape[0]
    wk = np.kron(w, np.eye(m))
    hb = np.zeros((a * m, a * m))
    for i, h in enumerate(h_list):
        hb[i * m:(i + 1) * m, i * m:(i + 1) * m] = h
    eye = np.eye(a * m)
    return np.block([[wk, -eta * eye], [hb @ wk - hb, wk - eta * hb]])


def reduced_spectral_radius(h_list, w: np.ndarray, eta: float) -> float:
    """Spectral radius of the GT map restricted to its invariant subspace.

    The quantity sum_i (g_i - H_i x_i) is conserved, which gives m unit
    eigenvalues; convergence is governed by the operator on the complement
    {sum_i (g_i - H_i x_i) = 0}, which is invariant.
    """
    a, m = len(h_list), h_list[0].shape[0]
    t = gradient_tracking_operator(h_list, w, eta)
    ones = np.kron(np.ones((1, a)), np.eye(m))
    hb_rows = np.hstack(list(h_list))  # sum_i H_i x_i
    left = np.hstack([-hb_rows, ones])  # conserved linear functionals (m x 2am)
    _, _, vt = np.linalg.svd(left)
    q = vt[m:].T  # orthonormal basis of the null space
    return float(np.max(np.abs(np.linalg.eigvals(q.T @ t @ q))))


def largest_stable_step(h_list, w: np.ndarray, lmax: float, grid_size: int = 200) -> float:
    """Scan eta on a grid in (0, 2/Lmax], then refine the stability threshold by bisection."""
    grid = np.linspace(0.01, 2.0, grid_size) / lmax
    stable = np.array([reduced_spectral_radius(h_list, w, eta) < 1.0 for eta in grid])
    if not stable[0]:
        return float("nan")
    if stable.all():
        return float(grid[-1])
    k = int(np.argmin(stable))  # first unstable grid point
    lo, hi = grid[k - 1], grid[k]
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if reduced_spectral_radius(h_list, w, mid) < 1.0 else (lo, mid)
    return float(lo)


def verification_constants() -> dict:
    h_list, _ = part1_problem()
    lmax = max(float(np.linalg.eigvalsh(h).max()) for h in h_list)
    mmin = min(float(np.linalg.eigvalsh(h).min()) for h in h_list)

    line_edges = dkr.build_line_graph(N_AGENTS)
    w_line = dkr.lazy_metropolis_weights(dkr.adjacency_from_edges(N_AGENTS, line_edges))
    sigma_max_a = float(np.linalg.svd(dkr.incidence_matrix(N_AGENTS, line_edges), compute_uv=False).max())

    eta_gt = 0.6 / lmax
    x_clients, y_clients = load_pickle("second_database.pkl")
    x_m_fed = np.linspace(-1.0, 1.0, 10)
    kmm_fed = dkr.kernel(x_m_fed, x_m_fed)
    h_fed = [
        dkr.local_quadratic_terms(np.asarray(x, float), np.asarray(y, float), x_m_fed, kmm_fed, SIGMA, NU, len(x_clients))[0]
        for x, y in zip(x_clients, y_clients)
    ]
    l_part2 = max(float(np.linalg.eigvalsh(h).max()) for h in h_fed)

    return {
        "lmax": lmax,
        "mmin": mmin,
        "sigma_max_a": sigma_max_a,
        "eta_dgd": 0.2 / lmax,
        "bound_dgd": 2.0 / lmax,
        "eta_gt": eta_gt,
        "rho_gt": reduced_spectral_radius(h_list, w_line, eta_gt),
        "alpha_gt_max": largest_stable_step(h_list, w_line, lmax),
        "eta_dual": 0.9 * 2.0 * mmin / sigma_max_a**2,
        "bound_dual": 2.0 * mmin / sigma_max_a**2,
        "l_part2": l_part2,
        "eta_fed": 0.002,
        "bound_fed": 2.0 / l_part2,
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def metric_row(df: pd.DataFrame, part: str, algorithm: str, require_runtime: bool = False, **filters) -> pd.Series:
    rows = df[(df.part == part) & (df.algorithm == algorithm)]
    for col, val in filters.items():
        rows = rows[rows[col].astype(str) == str(val)]
    if require_runtime:
        rows = rows[rows.runtime_s.notna()]
    if rows.empty:
        raise KeyError(f"metric row not found: part={part} algorithm={algorithm} {filters}")
    return rows.iloc[0]


TABLE_SETUP = r"""\begin{table}[!t]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.08}
\setlength{\tabcolsep}{4pt}
\caption{Notation and experimental settings.}
\label{tab:setup}
\begin{tabularx}{\linewidth}{@{}L{0.12\linewidth}YY@{}}
\toprule
Block & Data and model & Settings and metrics \\
\midrule
Global & RBF kernel $k(x,z)=\exp(-(x-z)^2)$, Nystrom rank $m$, and coefficient vector $\alpha\in\mathbb{R}^m$. & Shared constants are $\sigma=\num{0.5}$, $\nu=\num{1}$, and seed $\num{42}$. The centralized reference solves $(\sigma^2K_{mm}+K_{nm}^\top K_{nm}+\nu I)\alpha^\star=K_{nm}^\top y$. \\
Part I & First dataset; the first $n=\num{100}$ samples are split sequentially across $a=\num{5}$ agents, with $20$ points per agent and $m=\num{10}$ landmarks. & Line, small-world, and complete graphs are tested. The main metric is $\frac{1}{a}\sum_i \lVert \alpha_i^t-\alpha^\star\rVert_2$. The large-$n$ study keeps $m=\lceil \sqrt{n}\rceil$ and a fixed $200$-iteration DGD budget up to $n=\num{30000}$. \\
Part II & Second dataset; five client datasets $(X_i,Y_i)$ of size $20$, with a uniform landmark grid on $[-1,1]$ of size $10$. & FedAvg uses $B\in\{20,15\}$, $C\in\{5,3\}$, and $E\in\{1,5,50\}$ under constant or diminishing rates. The metric is $F(\alpha_t)-F(\alpha^\star)$; SCAFFOLD reuses the same client splits. \\
Part III & First dataset; baseline $(n,m,a)=(100,10,5)$ and larger-scale setting $(1000,33,100)$. & DP-DGD is evaluated for $\epsilon\in\{0.1,1,10\}$ using the theorem-aligned schedules from the course notes and the DP paper. The reported metric is the mean optimality gap, together with finite-horizon privacy audits. \\
\bottomrule
\end{tabularx}
\end{table}"""


def table_theory(c: dict, df: pd.DataFrame) -> str:
    ref = metric_row(df, "part3", "DGD_non_private", graph="line")
    dp1 = metric_row(df, "part3", "DP-DGD", graph="line", epsilon="1.0")
    rows = [
        rf"DGD & Smooth bound $O(1/L)$. & $\eta_{{\mathrm{{DGD}}}}=0.2/L_{{\max}}$. & $L_{{\max}}=\num{{{fmt(c['lmax'], 6)}}}$ and $\eta_{{\mathrm{{DGD}}}}=\num{{{fmt(c['eta_dgd'], 6)}}}<2/L_{{\max}}=\num{{{fmt(c['bound_dgd'], 6)}}}$; the baseline run is stable on all three undirected graphs. \\",
        rf"GT & Spectral-radius test. & $\eta_{{\mathrm{{GT}}}}=0.6/L_{{\max}}$. & $\rho(T(\eta_{{\mathrm{{GT}}}}))=\num{{{fmt(c['rho_gt'], 6)}}}<1$, with a scanned threshold near $\num{{{fmt(c['alpha_gt_max'], 6)}}}$; the method remains accurate across graph topologies. \\",
        rf"Dual & Dual-ascent bound. & $\eta_{{\mathrm{{dual}}}}=0.9\cdot 2\mu_{{\min}}/\sigma_{{\max}}(A)^2$. & $\mu_{{\min}}=\num{{{fmt(c['mmin'], 6)}}}$, $\sigma_{{\max}}(A)=\num{{{fmt(c['sigma_max_a'], 6)}}}$, and $\eta_{{\mathrm{{dual}}}}=\num{{{fmt(c['eta_dual'], 6)}}}<\num{{{fmt(c['bound_dual'], 6)}}}$; the line-graph gap reaches the \num{{1e-6}} range. \\",
        r"ADMM & Positive-penalty condition. & $\beta\in\{0.1,1,10,100\}$. & All tested penalties are admissible; $\beta=1$ and $\beta=10$ are the fastest practical choices. \\",
        rf"FedAvg & Smooth SGD bound. & $\eta_0=\num{{{fmt(c['eta_fed'], 4)}}}$ and $\eta_t=0.002/(1+0.01t)$. & $L_{{\max}}^{{\mathrm{{FL}}}}=\num{{{fmt(c['l_part2'], 6)}}}$ and $\eta_0<2/L_{{\max}}^{{\mathrm{{FL}}}}=\num{{{fmt(c['bound_fed'], 6)}}}$; $E=50$ gives the best final error in both learning-rate schedules. \\",
        r"SCAFFOLD & Control-variate correction. & Constant local rate $\eta=\num{2e-3}$, matched to FedAvg. & With the same local rate, the corrected runs outperform FedAvg at $E=50$ for both $C=5$ and $C=3$. \\",
        rf"DP-DGD & Summability conditions. & $\alpha_k=\frac{{0.002}}{{1+0.001k}}$, $\gamma_k=\frac{{1}}{{1+0.001k^{{0.9}}}}$, $\nu_k=\frac{{0.01}}{{\epsilon}}\bigl(1+0.001k^{{0.1}}\bigr)$. & On $3000$ steps, $\sum \alpha_k^2/\gamma_k=\num{{{fmt(ref.sum_alpha2_over_gamma, 6)}}}$ and $\sum \alpha_k/\nu_k|_{{\epsilon=1}}=\num{{{fmt(dp1.sum_alpha_over_nu, 6)}}}$; smaller $\epsilon$ yields larger final gaps. \\",
    ]
    head = [
        r"\begin{table}[!htb]",
        r"\centering",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.02}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{Parameter choices and checks; lecture-note bounds are used except for the DP schedule, which follows Wang--Nedic Theorem~2(3).}",
        r"\label{tab:theory}",
        r"\begin{tabularx}{\linewidth}{@{}L{0.10\linewidth}L{0.18\linewidth}L{0.23\linewidth}Y@{}}",
        r"\toprule",
        r"Method & Principle & Choice & Check \\",
        r"\midrule",
    ]
    tail = [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(head + rows + tail)


def table_results(df: pd.DataFrame) -> str:
    fed_dim = df[(df.part == "part2") & (df.algorithm == "FedAvg") & (df.B.astype(str) == "20.0")
                 & (df.Csel.astype(str) == "5.0") & (df.E.astype(str) == "50.0") & (df.eta == "0.002/(1+0.01t)")]
    if fed_dim.empty:
        raise KeyError("diminishing-rate FedAvg row not found")
    entries = [
        ("Part I", "DGD, line", metric_row(df, "part1", "DGD", True, graph="line"), "Gap"),
        ("Part I", "GT, line", metric_row(df, "part1", "GT", True, graph="line"), "Gap"),
        ("Part I", "Dual dec., line", metric_row(df, "part1", "DualDecomposition", graph="line"), "Gap"),
        ("Part I", r"ADMM, $\beta=1$", metric_row(df, "part1", "ADMM_beta1", graph="line", beta="1.0"), "Gap"),
        ("Part II", r"FedAvg $(20,5,50)$ const.", metric_row(df, "part2", "FedAvg", B="20.0", Csel="5.0", E="50.0", graph="cloud"), "Obj."),
        ("Part II", r"FedAvg $(20,5,50)$ dim.", fed_dim.iloc[0], "Obj."),
        ("Part II", r"FedAvg $(15,5,50)$", metric_row(df, "part2", "FedAvg", B="15.0", Csel="5.0", E="50.0", graph="cloud"), "Obj."),
        ("Part II", r"FedAvg $(15,3,50)$", metric_row(df, "part2", "FedAvg", B="15.0", Csel="3.0", E="50.0", graph="cloud"), "Obj."),
        ("Part II", r"SCAFFOLD $(20,5,50)$", metric_row(df, "part2", "SCAFFOLD", B="20.0", Csel="5.0", E="50.0", graph="cloud"), "Obj."),
        ("Part II", r"SCAFFOLD $(20,3,50)$", metric_row(df, "part2", "SCAFFOLD", B="20.0", Csel="3.0", E="50.0", graph="cloud"), "Obj."),
        ("Part III", "Non-private DGD", metric_row(df, "part3", "DGD_non_private", graph="line"), "Gap"),
        ("Part III", r"DP-DGD, $\epsilon=10$", metric_row(df, "part3", "DP-DGD", graph="line", epsilon="10.0"), "Gap"),
        ("Part III", r"DP-DGD, $\epsilon=1$", metric_row(df, "part3", "DP-DGD", graph="line", epsilon="1.0"), "Gap"),
        ("Part III", r"DP-DGD, $\epsilon=0.1$", metric_row(df, "part3", "DP-DGD", graph="line", epsilon="0.1"), "Gap"),
        ("Part III ext.", "Non-private DGD", metric_row(df, "part3_optional", "DGD_non_private", graph="line"), "Gap"),
        ("Part III ext.", r"DP-DGD, $\epsilon=10$", metric_row(df, "part3_optional", "DP-DGD", graph="line", epsilon="10.0"), "Gap"),
    ]
    lines = [
        r"\begin{table}[!htb]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.03}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Selected final metrics and runtimes.}",
        r"\label{tab:results}",
        r"\begin{tabular}{@{}l c r r@{}}",
        r"\toprule",
        r"Configuration & Metric & Final value & Runtime (s) \\",
        r"\midrule",
    ]
    block = ""
    for name, config, row, metric in entries:
        if name != block:
            if block:
                lines.append(r"\addlinespace[2pt]")
            lines.append(r"\multicolumn{4}{@{}l}{\textit{" + name + r"}} \\")
            block = name
        lines.append(rf"{config} & {metric} & \num{{{fmt(row.final_gap, 4)}}} & \num{{{fmt(row.runtime_s, 4)}}} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(PROJECT_DIR / "results" / "metrics_summary.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.results, keep_default_na=True)
    for col in ["B", "Csel", "E", "beta", "epsilon"]:
        df[col] = df[col].map(lambda v: "" if pd.isna(v) else (f"{float(v):.1f}" if str(v).replace(".", "", 1).isdigit() else str(v)))
    df["graph"] = df["graph"].fillna("")

    constants = verification_constants()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "table_setup.tex").write_text(TABLE_SETUP + "\n")
    (TABLES_DIR / "table_theory.tex").write_text(table_theory(constants, df) + "\n")
    (TABLES_DIR / "table_results.tex").write_text(table_results(df) + "\n")
    for key, value in constants.items():
        print(f"{key:>14s} = {value:.6e}")
    print(f"Tables written to {TABLES_DIR}")


if __name__ == "__main__":
    main()
