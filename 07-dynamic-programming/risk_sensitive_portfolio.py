"""Risk-sensitive portfolio optimization in a finite-state Markovian market.

Implements Sections 1-3 of the report. The market regime X_t is a Markov
chain on E = {bull, neutral, bear} with transition matrix Q. Wealth evolves as

    V_{t+1} = V_t * G(a_t, Z_{t+1}),   G(a, z) = e^r + a (z - e^r),

where a in A (compact grid) is the fraction invested in the risky asset and
the gross return Z_{t+1} depends on the regime transition (X_t, X_{t+1}).

For a risk-sensitivity parameter gamma != 0 we maximise the long-run growth

    J_gamma = liminf_T  -(2 / (gamma T)) log E[ V_T^{-gamma/2} ].

With theta = -gamma/2 the problem is a multiplicative (risk-sensitive)
average-cost MDP whose ergodic Bellman equation, in growth units, reads

    rho + h(x) = max_a (1/theta) log sum_y Q(x, y) M_theta(x, a, y) exp(theta h(y)),
    M_theta(x, a, y) = E[ G(a, Z)^theta | x, y ],

(gamma > 0: risk-averse, theta < 0; gamma < 0: risk-seeking, theta > 0).
As gamma -> 0 it reduces to the risk-neutral average-reward equation with
reward r(x, a) = sum_y Q(x, y) E[log G(a, Z) | x, y] (Section 3, Eq. 3.2).

The script solves the equation with relative value iteration, checks every
optimal stationary policy independently through the Perron-Frobenius
eigenvalue of its twisted kernel K(x, y) = Q(x, y) M_theta(x, a(x), y), checks
the risk-neutral value by Monte Carlo simulation, and sweeps gamma.
"""

from __future__ import annotations

import csv
import os

import numpy as np

STATES = ["bull", "neutral", "bear"]

# Regime transition matrix (the three-state market of Section 4.1).
Q = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]])

# Monthly risk-free log-rate and log-normal risky returns indexed by the next regime y.
R_FREE = 0.002
MU = np.array([0.020, 0.008, -0.015])      # E[log Z | y]
SIGMA = np.array([0.035, 0.045, 0.075])    # Std[log Z | y]

ACTIONS = np.round(np.linspace(0.0, 1.5, 151), 4)   # allocation grid, leverage up to 1.5
N_QUAD = 9                                          # Gauss-Hermite nodes


def return_distribution():
    """Discretise log Z | y ~ N(MU[y], SIGMA[y]^2) with Gauss-Hermite quadrature."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(N_QUAD)
    weights = weights / weights.sum()
    z = np.exp(MU[:, None] + SIGMA[:, None] * nodes[None, :])   # (n_states, N_QUAD)
    return z, weights


Z, W = return_distribution()
GROSS = np.exp(R_FREE) + ACTIONS[:, None, None] * (Z[None, :, :] - np.exp(R_FREE))  # (n_actions, n_states, N_QUAD)
assert np.all(GROSS > 0), "allocation grid allows non-positive wealth"


def twisted_moments(theta: float) -> np.ndarray:
    """M_theta(a, y) = E[G(a, Z)^theta | y] (returns depend on the next regime only)."""
    return np.einsum("ayq,q->ay", GROSS ** theta, W)


def expected_log_growth() -> np.ndarray:
    """r(x, a) = sum_y Q(x, y) E[log G(a, Z) | y], shape (n_states, n_actions)."""
    elog = np.einsum("ayq,q->ay", np.log(GROSS), W)
    return Q @ elog.T


def bellman(h: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Risk-sensitive Bellman operator in growth units; returns (T h, values per (x, a))."""
    if theta == 0.0:
        vals = expected_log_growth() + (Q @ h)[:, None]
    else:
        M = twisted_moments(theta)                          # (a, y)
        inner = (Q * np.exp(theta * h)[None, :]) @ M.T      # (x, a): sum_y Q(x,y) e^{theta h(y)} M(a,y)
        vals = np.log(inner) / theta
    return vals.max(axis=1), vals


def solve_rvi(theta: float, ref_state: int = 0, tol: float = 1e-12, max_iter: int = 100_000) -> dict:
    h = np.zeros(len(STATES))
    for k in range(1, max_iter + 1):
        Th, _ = bellman(h, theta)
        h_next = Th - Th[ref_state]
        if np.max(np.abs(h_next - h)) < tol:
            h = h_next
            break
        h = h_next
    Th, vals = bellman(h, theta)
    policy = vals.argmax(axis=1)
    return {"theta": theta, "rho": float(Th[ref_state] - h[ref_state]), "h": h, "policy": policy,
            "allocation": ACTIONS[policy], "iters": k}


def policy_growth_rate(policy: np.ndarray, theta: float) -> float:
    """Exact risk-sensitive growth of a stationary policy via the Perron-Frobenius eigenvalue."""
    if theta == 0.0:
        r = expected_log_growth()[np.arange(len(STATES)), policy]
        w, v = np.linalg.eig(Q.T)
        pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
        pi /= pi.sum()
        return float(pi @ r)
    M = twisted_moments(theta)
    K = Q * M[policy, :]                     # K(x, y) = Q(x, y) M_theta(a(x), y)
    lam = np.max(np.real(np.linalg.eigvals(K)))
    return float(np.log(lam) / theta)


def monte_carlo_log_growth(policy: np.ndarray, T: int = 200_000, seed: int = 0) -> tuple[float, float]:
    """Average log-growth of a stationary policy along one long simulated path (risk-neutral check)."""
    rng = np.random.default_rng(seed)
    x = 0
    logs = np.empty(T)
    cumQ = np.cumsum(Q, axis=1)
    a_of = ACTIONS[policy]
    for t in range(T):
        y = int(np.searchsorted(cumQ[x], rng.random()))
        z = np.exp(MU[y] + SIGMA[y] * rng.standard_normal())
        logs[t] = np.log(np.exp(R_FREE) + a_of[x] * (z - np.exp(R_FREE)))
        x = y
    # batch-means standard error
    batches = logs[: (T // 100) * 100].reshape(100, -1).mean(axis=1)
    return float(logs.mean()), float(batches.std(ddof=1) / np.sqrt(len(batches)))


def main() -> None:
    os.makedirs("results", exist_ok=True)
    gammas = [-4.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    rows = []
    print(f"{'gamma':>7} {'rho (growth/month)':>19} {'PF check':>12}  allocation (bull, neutral, bear)  iters")
    for gamma in gammas:
        theta = -gamma / 2.0
        sol = solve_rvi(theta)
        check = policy_growth_rate(sol["policy"], theta)
        assert abs(check - sol["rho"]) < 1e-8, (gamma, check, sol["rho"])
        rows.append({"gamma": gamma, "rho": sol["rho"], "pf_check": check,
                     **{f"a_{s}": a for s, a in zip(STATES, sol["allocation"])},
                     **{f"h_{s}": v for s, v in zip(STATES, sol["h"])}, "iters": sol["iters"]})
        alloc = ", ".join(f"{a:.2f}" for a in sol["allocation"])
        print(f"{gamma:7.2f} {sol['rho']:19.6f} {check:12.6f}  ({alloc}){'':>17} {sol['iters']}")

    neutral = solve_rvi(0.0)
    mc_mean, mc_se = monte_carlo_log_growth(neutral["policy"])
    print(f"\nRisk-neutral optimum: rho = {neutral['rho']:.6f}; Monte Carlo average log-growth = "
          f"{mc_mean:.6f} +/- {1.96 * mc_se:.6f} (95% CI)")
    small = [r for r in rows if abs(r["gamma"]) == 0.1]
    print("gamma -> 0 continuity: " + ", ".join(f"rho(gamma={r['gamma']:+.1f}) = {r['rho']:.6f}" for r in small)
          + f", rho(0) = {neutral['rho']:.6f}")

    with open(os.path.join("results", "risk_sensitive_sweep.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = np.arange(len(rows))
    labels = [f"{r['gamma']:g}" for r in rows]
    zero = labels.index("0")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for s, marker in zip(STATES, "o^v"):
        axes[0].plot(g, [r[f"a_{s}"] for r in rows], marker=marker, label=s)
    for ax in axes:
        ax.axvline(zero, color="gray", ls=":")
        ax.set_xticks(g, labels)
    axes[0].set_xlabel(r"risk sensitivity $\gamma$  ($<0$ risk-seeking, $>0$ risk-averse)")
    axes[0].set_ylabel("optimal risky allocation $a^*(x)$")
    axes[0].legend(title="regime $x$")
    axes[0].grid(alpha=0.3)
    axes[1].plot(g, 1200 * np.array([r["rho"] for r in rows]), marker="o", color="tab:purple")
    axes[1].set_xlabel(r"risk sensitivity $\gamma$")
    axes[1].set_ylabel(r"optimal risk-adjusted growth $\rho_\gamma$ (% / year)")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Risk-sensitive portfolio in a three-regime Markov market")
    fig.tight_layout()
    fig.savefig(os.path.join("results", "risk_sensitive_sweep.png"), dpi=150)
    plt.close(fig)
    print("Saved results/risk_sensitive_sweep.csv and results/risk_sensitive_sweep.png")


if __name__ == "__main__":
    main()
