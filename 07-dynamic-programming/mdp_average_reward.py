"""Average-reward MDPs: Relative Value Iteration and Policy Iteration.

Implements Sections 4-5 of the report:

* ``relative_value_iteration`` - RVI, h_{k+1} = T h_k - (T h_k)(s0);
* ``policy_iteration`` - evaluation of (rho, h) with h(s_ref) = 0 and greedy improvement;

and runs the three examples of the report:

1. the 3-state / 2-action toy MDP of the appendix (cycle vs. stay);
2. the three-regime market with a single aggressive action (Section 4.1);
3. the two-state investment MDP with aggressive / conservative actions (Section 5.1).

Conventions: ``R[s, a]`` is the one-period reward and ``P[s, a, s']`` the
transition probability.
"""

from __future__ import annotations

import os

import numpy as np


def bellman_operator(R: np.ndarray, P: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (T h, Q) with Q(s, a) = r(s, a) + sum_s' P(s'|s, a) h(s')."""
    Q = R + P @ h
    return Q.max(axis=1), Q


def relative_value_iteration(R, P, eps=1e-10, max_iter=100_000, ref_state=0, aperiodicity=0.0):
    """Relative Value Iteration for unichain average-reward MDPs.

    ``aperiodicity`` = tau in (0, 1) applies the aperiodicity transform
    P <- tau I + (1 - tau) P, R <- (1 - tau) R, which keeps the optimal
    policies and h but makes RVI converge on periodic chains (rho is
    multiplied by 1 - tau and is rescaled back at the end).
    """
    R = np.asarray(R, dtype=float)
    P = np.asarray(P, dtype=float)
    if aperiodicity > 0.0:
        S, A = R.shape
        eye = np.repeat(np.eye(S)[:, None, :], A, axis=1)
        P = aperiodicity * eye + (1.0 - aperiodicity) * P
        R = (1.0 - aperiodicity) * R
    h = np.zeros(R.shape[0])
    span_history = []
    for k in range(1, max_iter + 1):
        Th, _ = bellman_operator(R, P, h)
        h_next = Th - Th[ref_state]
        diff = h_next - h
        span_history.append(float(diff.max() - diff.min()))
        h = h_next
        if np.max(np.abs(diff)) < eps:
            break
    Th, Q = bellman_operator(R, P, h)
    rho = float(Th[ref_state] - h[ref_state])
    if aperiodicity > 0.0:
        rho /= 1.0 - aperiodicity
    return {"rho": rho, "h": h, "policy": Q.argmax(axis=1), "iters": k, "span_history": span_history}


def evaluate_policy(R, P, policy, ref_state=0):
    """Solve rho + h(s) = r(s, pi(s)) + sum_s' P(s'|s, pi(s)) h(s'), with h(ref_state) = 0."""
    S = R.shape[0]
    r_pi = R[np.arange(S), policy]
    P_pi = P[np.arange(S), policy]
    # Unknowns: rho, h(0..S-1); equations: S Poisson equations + normalisation.
    A = np.zeros((S + 1, S + 1))
    b = np.zeros(S + 1)
    A[:S, 0] = 1.0
    A[:S, 1:] = np.eye(S) - P_pi
    b[:S] = r_pi
    A[S, 1 + ref_state] = 1.0
    sol = np.linalg.solve(A, b)
    return float(sol[0]), sol[1:]


def policy_iteration(R, P, policy0=None, ref_state=0, max_iter=100):
    S = R.shape[0]
    policy = np.zeros(S, dtype=int) if policy0 is None else np.asarray(policy0, dtype=int)
    history = []
    for it in range(1, max_iter + 1):
        rho, h = evaluate_policy(R, P, policy, ref_state)
        history.append({"iteration": it, "policy": policy.copy(), "rho": rho, "h": h.copy()})
        Q = R + P @ h
        # Keep the current action when it is (numerically) among the maximisers.
        best = Q.argmax(axis=1)
        keep = Q[np.arange(S), policy] >= Q[np.arange(S), best] - 1e-12
        new_policy = np.where(keep, policy, best)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return {"rho": rho, "h": h, "policy": policy, "history": history}


# ---------------------------------------------------------------------------
# Examples of the report
# ---------------------------------------------------------------------------

def make_small_mdp():
    """Appendix toy MDP: action 0 moves along the cycle 0 -> 1 -> 2 -> 0, action 1 stays."""
    R = np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 0.0]])
    P = np.zeros((3, 2, 3))
    P[0, 0, 1] = P[1, 0, 2] = P[2, 0, 0] = 1.0
    for s in range(3):
        P[s, 1, s] = 1.0
    return R, P


def make_three_state_market():
    """Section 4.1: bull / neutral / bear regimes with a single aggressive action."""
    R = np.array([[5.0], [2.0], [0.0]])
    P = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]])[:, None, :]
    return R, P


def make_two_state_investment():
    """Section 5.1: states G (good) / B (bad), actions A (aggressive) / C (conservative).

    Rewards and the transitions under A are those of the report. The
    transitions under C are taken from the transition diagram of the report's
    appendix (G->G 0.9, G->B 0.1, B->G 0.3, B->B 0.7): conservative investing
    makes the good regime more persistent.
    """
    R = np.array([[6.0, 3.0], [-2.0, 1.0]])
    P = np.zeros((2, 2, 2))
    P[:, 0, :] = [[0.7, 0.3], [0.4, 0.6]]   # action A
    P[:, 1, :] = [[0.9, 0.1], [0.3, 0.7]]   # action C
    return R, P


def plot_two_state_mdp(P, R, policy, path):
    """One panel per action: transition probabilities and rewards; optimal actions are highlighted."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    pos = {0: np.array([0.0, 0.0]), 1: np.array([3.0, 0.0])}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, (ax, name, color) in enumerate(zip(axes, ["aggressive (A)", "conservative (C)"], ["tab:red", "tab:blue"])):
        for s, xy in pos.items():
            optimal = policy[s] == a
            ax.add_patch(plt.Circle(xy, 0.42, color="#fde2e2" if optimal and a == 0 else "#dbe9f6" if optimal else "white",
                                    ec="black", lw=1.5, zorder=3))
            ax.text(*xy, "GB"[s], ha="center", va="center", fontsize=14, weight="bold", zorder=4)
            ax.text(xy[0], -0.75, f"r = {R[s, a]:g}", ha="center", fontsize=10)
            ax.text(xy[0], 0.95, f"stay {P[s, a, s]:.1f}", ha="center", fontsize=9, color=color)
            loop = FancyArrowPatch(xy + [-0.2, 0.38], xy + [0.2, 0.38], connectionstyle="arc3,rad=-1.6",
                                   arrowstyle="-|>", mutation_scale=12, color=color, lw=1.4)
            ax.add_patch(loop)
        for s, t, rad, dy in ((0, 1, -0.3, 0.62), (1, 0, -0.3, -0.62)):
            arrow = FancyArrowPatch(pos[s], pos[t], connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                    mutation_scale=15, color=color, lw=1.6, shrinkA=22, shrinkB=22)
            ax.add_patch(arrow)
            ax.text(1.5, dy, f"{P[s, a, t]:.1f}", ha="center", va="center", fontsize=10, color=color)
        ax.set_title(f"Action {name}", fontsize=11, color=color)
        ax.set_xlim(-1.0, 4.0)
        ax.set_ylim(-1.1, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
    opt = ", ".join(f"{'GB'[s]} -> {'AC'[policy[s]]}" for s in (0, 1))
    fig.suptitle(f"Two-state investment MDP (G: good, B: bad). Optimal policy (shaded): {opt}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs("results", exist_ok=True)
    np.set_printoptions(precision=6, suppress=True)

    print("=== Appendix: 3-state / 2-action toy MDP ===")
    R, P = make_small_mdp()
    res = relative_value_iteration(R, P, eps=1e-12, max_iter=2000)
    print(f"plain RVI: stopped after {res['iters']} iterations, last span(h_k+1 - h_k) = "
          f"{res['span_history'][-1]:.3f} -> no convergence (optimal chain is periodic)")
    res = relative_value_iteration(R, P, eps=1e-12, aperiodicity=0.5)
    print(f"RVI (aperiodicity transform, tau=0.5): rho = {res['rho']:.6f}, h = {res['h']}, "
          f"policy = {res['policy']}, iterations = {res['iters']}")
    pi = policy_iteration(R, P)
    print(f"PI : rho = {pi['rho']:.6f}, h = {pi['h']}, policy = {pi['policy']}")

    print("\n=== Section 4.1: three-regime market, single aggressive action ===")
    R, P = make_three_state_market()
    res = relative_value_iteration(R, P, eps=1e-12)
    P1 = P[:, 0, :]
    w, v = np.linalg.eig(P1.T)
    stat = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    stat /= stat.sum()
    print(f"RVI: rho = {res['rho']:.6f}, h = {res['h']} (S0 > S1 > S2), iterations = {res['iters']}")
    print(f"check: stationary distribution {stat}, pi . r = {stat @ R[:, 0]:.6f}")

    print("\n=== Section 5.1: two-state investment MDP ===")
    R, P = make_two_state_investment()
    pi = policy_iteration(R, P, policy0=[1, 1])
    for step in pi["history"]:
        pol = "".join("AC"[a] for a in step["policy"])
        print(f"PI iteration {step['iteration']}: policy (G,B) = {pol}, rho = {step['rho']:.6f}, h = {step['h']}")
    res = relative_value_iteration(R, P, eps=1e-12, ref_state=1)
    print(f"RVI check: rho = {res['rho']:.6f}, policy = {''.join('AC'[a] for a in res['policy'])}")
    for policy in ([0, 0], [0, 1], [1, 0], [1, 1]):
        rho, _ = evaluate_policy(R, P, np.array(policy))
        print(f"  rho(pi = {''.join('AC'[a] for a in policy)}) = {rho:.6f}")
    plot_two_state_mdp(P, R, pi["policy"], os.path.join("results", "two_state_mdp.png"))
    print("Saved results/two_state_mdp.png")


if __name__ == "__main__":
    main()
