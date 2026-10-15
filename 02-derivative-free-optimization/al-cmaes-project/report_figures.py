"""Figures for the AL-CMA-ES section of the report and slides.

Generates, in ``results/report_figures/``:

* ``condition1.pdf``  - illustration of Condition 1 (primal change dominates)
  on f(x) = x^2 with g(x) = 2 - x <= 0;
* ``condition2.pdf``  - mean trajectory of AL-CMA-ES on f(x) = x1^2 + x2^2
  with g(x) = 2 - x1 <= 0, highlighting iterations where Condition 2 holds;
* ``al_vs_static_distance.png`` - ||m_t - x*|| versus evaluations;
* ``al_vs_static_sigma.png``    - step-size sigma_t versus evaluations,

the last two on the 10-D sphere with the linear constraint x1 >= 1
(AL-CMA-ES versus a static quadratic penalty).
"""

from __future__ import annotations

import os
import warnings

import cma
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join("results", "report_figures")
SEED = 42

# Default AL-CMA-ES constants (see the hyperparameter table of the report).
D_GAMMA = 5.0
D_OMEGA = 5.0
K1 = 3.0
K2 = 5.0


def phi(g: float, gamma: float, omega: float) -> float:
    """Augmented Lagrangian penalty term for one inequality constraint g <= 0."""
    if gamma + omega * g >= 0:
        return gamma * g + 0.5 * omega * g**2
    return -gamma**2 / (2.0 * omega)


def merit(f_val: float, g_val: float, gamma: float, omega: float) -> float:
    return f_val + phi(g_val, gamma, omega)


# ---------------------------------------------------------------------------
# Condition 1: one AL update in dimension 1
# ---------------------------------------------------------------------------

def plot_condition1(path: str) -> dict:
    n = 1
    f = lambda x: x**2
    g = lambda x: 2.0 - x

    gamma_t, omega_t = 1.0, 1.0
    m_t, m_next = 1.5, 1.7  # infeasible means approaching the boundary x = 2

    h = lambda x, gam, om: merit(f(x), g(x), gam, om)
    delta_primal = abs(h(m_next, gamma_t, omega_t) - h(m_t, gamma_t, omega_t))
    delta_dual = omega_t * g(m_next) ** 2
    cond1 = delta_dual < K1 * delta_primal / n

    chi = 2.0 ** (1.0 / n)
    gamma_next = max(0.0, gamma_t + omega_t / D_GAMMA * g(m_next))
    omega_next = omega_t * chi ** (1.0 / (4.0 * D_OMEGA)) if cond1 else omega_t * chi ** (-1.0 / D_OMEGA)

    xs = np.linspace(0.0, 3.0, 400)
    h_t = [h(x, gamma_t, omega_t) for x in xs]
    h_next = [h(x, gamma_next, omega_next) for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, h_t, color="tab:blue", lw=2, label=r"$h(x,\gamma_t,\omega_t)$")
    ax.plot(xs, h_next, color="tab:orange", lw=2, ls="--", label=r"$h(x,\gamma_{t+1},\omega_{t+1})$")
    ax.axvline(2.0, color="gray", ls=":", lw=1.5, label=r"constraint boundary $x=2$")
    ax.axvspan(0.0, 2.0, color="red", alpha=0.06)
    ax.scatter([m_t], [h(m_t, gamma_t, omega_t)], color="tab:blue", zorder=5, s=50, label=r"$h(m_t,\gamma_t,\omega_t)$")
    ax.scatter([m_next], [h(m_next, gamma_t, omega_t)], color="tab:green", zorder=5, s=50,
               label=r"$h(m_{t+1},\gamma_t,\omega_t)$")
    ax.annotate(
        "", xy=(m_next, h(m_next, gamma_t, omega_t)), xytext=(m_t, h(m_t, gamma_t, omega_t)),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("Augmented Lagrangian value")
    ax.set_title(
        rf"Condition 1 {'holds' if cond1 else 'fails'}: "
        rf"$\omega_t g(m_{{t+1}})^2={delta_dual:.3f} < k_1\Delta h_x/n={K1 * delta_primal / n:.3f}$",
        fontsize=10,
    )
    ax.text(0.1, 0.4, "infeasible region", color="red", fontsize=9)
    ax.set_ylim(0, 9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "delta_primal": delta_primal,
        "delta_dual": delta_dual,
        "condition1": cond1,
        "gamma_next": gamma_next,
        "omega_next": omega_next,
    }


# ---------------------------------------------------------------------------
# Condition 2: stagnation detection on a 2-D problem
# ---------------------------------------------------------------------------

def run_al_tracking(objective, constraint, x0, sigma0, max_fevals, seed):
    """Run AL-CMA-ES (pycma implementation) and record mean, sigma and multiplier."""
    cfun = cma.ConstrainedFitnessAL(objective, lambda x: [constraint(x)], logging=False)
    es = cma.CMAEvolutionStrategy(
        x0, sigma0, {"maxfevals": max_fevals, "seed": seed, "verbose": -9, "verb_log": 0, "tolstagnation": 0}
    )
    means, sigmas, evals, gammas = [np.array(es.mean, dtype=float)], [es.sigma], [0], [0.0]
    while not es.stop():
        X = es.ask()
        es.tell(X, [cfun(x) for x in X])
        cfun.update(es)
        means.append(np.array(es.mean, dtype=float))
        sigmas.append(es.sigma)
        evals.append(es.countevals)
        gammas.append(float(cfun.al.lam[0]) if getattr(cfun.al, "lam", None) is not None else 0.0)
    return np.array(means), np.array(sigmas), np.array(evals), np.array(gammas)


def plot_condition2(path: str) -> dict:
    f = lambda x: float(np.sum(np.asarray(x) ** 2))
    g = lambda x: 2.0 - float(x[0])
    means, _, _, _ = run_al_tracking(f, g, np.array([-1.0, 2.5]), 0.5, 1200, SEED)

    gvals = np.array([g(m) for m in means])
    cond2 = K2 * np.abs(np.diff(gvals)) < np.abs(gvals[1:])
    cond2 &= gvals[1:] > 0  # stagnation in the infeasible region

    x1, x2 = np.meshgrid(np.linspace(-1.5, 3.0, 300), np.linspace(-1.0, 3.0, 300))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    cs = ax.contour(x1, x2, x1**2 + x2**2, levels=12, cmap="Greys", linewidths=0.7)
    ax.clabel(cs, fontsize=6, fmt="%.0f")
    ax.axvspan(-1.5, 2.0, color="red", alpha=0.06)
    ax.axvline(2.0, color="gray", ls=":", lw=1.5, label=r"$g(x)=2-x_1=0$")
    ax.plot(means[:, 0], means[:, 1], "-", color="tab:blue", lw=1, alpha=0.7, label=r"mean $m_t$")
    ax.scatter(means[1:][cond2, 0], means[1:][cond2, 1], color="tab:orange", s=22, zorder=5,
               label="Condition 2 holds")
    ax.scatter([2.0], [0.0], marker="*", s=160, color="tab:green", zorder=6, label=r"$x^\star=(2,0)$")
    ax.set_xlim(-1.5, 3.0)
    ax.set_ylim(-1.0, 3.0)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(r"Condition 2: $k_2|g(m_{t+1})-g(m_t)|<|g(m_{t+1})|$", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"iterations": int(len(means) - 1), "condition2_triggers": int(cond2.sum()),
            "final_mean": means[-1].tolist()}


# ---------------------------------------------------------------------------
# AL-CMA-ES versus static penalty on the sphere with x1 >= 1
# ---------------------------------------------------------------------------

def run_static_tracking(objective, constraint, beta, x0, sigma0, max_fevals, seed):
    penalised = lambda x: objective(x) + beta * max(0.0, constraint(x)) ** 2
    es = cma.CMAEvolutionStrategy(x0, sigma0, {"maxfevals": max_fevals, "seed": seed, "verbose": -9, "verb_log": 0})
    means, sigmas, evals = [np.array(es.mean, dtype=float)], [es.sigma], [0]
    while not es.stop():
        X = es.ask()
        es.tell(X, [penalised(x) for x in X])
        means.append(np.array(es.mean, dtype=float))
        sigmas.append(es.sigma)
        evals.append(es.countevals)
    return np.array(means), np.array(sigmas), np.array(evals)


def plot_al_vs_static(dist_path: str, sigma_path: str, beta: float = 1e5) -> dict:
    n = 10
    f = lambda x: float(np.sum(np.asarray(x) ** 2))
    g = lambda x: 1.0 - float(x[0])
    x_opt = np.zeros(n)
    x_opt[0] = 1.0
    x0 = np.zeros(n)

    al_m, al_s, al_e, _ = run_al_tracking(f, g, x0, 1.0, 10000, SEED)
    st_m, st_s, st_e = run_static_tracking(f, g, beta, x0, 1.0, 10000, SEED)
    al_d = np.linalg.norm(al_m - x_opt, axis=1)
    st_d = np.linalg.norm(st_m - x_opt, axis=1)

    for path, (y_al, y_st), ylabel in [
        (dist_path, (al_d, st_d), r"$\|m_t-x^\star\|$"),
        (sigma_path, (al_s, st_s), r"step size $\sigma_t$"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.semilogy(al_e, y_al, lw=2, label="AL-CMA-ES")
        ax.semilogy(st_e, y_st, lw=2, ls="--", label=rf"static penalty ($\beta=10^{{{int(np.log10(beta))}}}$)")
        ax.set_xlabel("function evaluations")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return {"al_final_distance": float(al_d[-1]), "static_final_distance": float(st_d[-1]),
            "al_evals": int(al_e[-1]), "static_evals": int(st_e[-1])}


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    c1 = plot_condition1(os.path.join(OUTPUT_DIR, "condition1.pdf"))
    c2 = plot_condition2(os.path.join(OUTPUT_DIR, "condition2.pdf"))
    cmp_ = plot_al_vs_static(
        os.path.join(OUTPUT_DIR, "al_vs_static_distance.png"),
        os.path.join(OUTPUT_DIR, "al_vs_static_sigma.png"),
    )
    print("Condition 1:", c1)
    print("Condition 2:", c2)
    print("AL vs static:", cmp_)
    print(f"Figures written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
