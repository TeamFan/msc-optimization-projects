# Dynamic Programming — Risk-Sensitive Portfolio Optimization in a Markovian Market

Project 9 of the *Markov Decision Processes and Dynamic Programming* course.
The report ([`report/risk_sensitive_portfolio_report.pdf`](report/risk_sensitive_portfolio_report.pdf)) studies the long-run risk-sensitive growth criterion

$$J_\gamma(x,\pi)=\liminf_{T\to\infty}-\frac{2}{\gamma T}\log\mathbb{E}^\pi_x\big[V_T^{-\gamma/2}\big]$$

for a portfolio whose risky returns are driven by a finite-state Markov
regime, and the average-reward algorithms used to solve the resulting
ergodic Bellman equations.

## Report

`report/risk_sensitive_portfolio_report.{tex,pdf}`: model and risk-averse /
risk-seeking formulations (Sections 1–3), relative value iteration and policy
iteration with worked examples (Sections 4–5), and the numerical study of the
risk-sensitive portfolio (Section 6). All numbers in the report are produced by
the two scripts below.

## Code

| Script | Content |
|---|---|
| `mdp_average_reward.py` | Relative Value Iteration and Policy Iteration for average-reward MDPs, applied to the three examples of the report |
| `risk_sensitive_portfolio.py` | Risk-sensitive ergodic Bellman equation for the regime-switching portfolio, solved by RVI for risk-seeking (γ < 0), risk-neutral (γ = 0) and risk-averse (γ > 0) investors |

```bash
python mdp_average_reward.py
python risk_sensitive_portfolio.py    # writes results/risk_sensitive_sweep.{csv,png}
```

### Average-reward examples

* **Toy MDP** of Section 4 (cycle 0→1→2→0 vs. stay): the optimal policy is the
  period-3 cycle (ρ = 0.5). Plain RVI does not converge on this periodic
  chain — the span of `h_{k+1} − h_k` stays at 1 — so the script applies the
  aperiodicity transform `P ← τI + (1−τ)P`, which leaves the optimal policy and
  `h` unchanged; RVI and Policy Iteration then agree (`h = (0, −0.5, −0.5)`).
* **Three-regime market** (Section 4.1): ρ = 2.125 = π·r with stationary
  distribution π = (0.275, 0.375, 0.35), and `h(S0) > h(S1) > h(S2)`.
* **Two-state investment MDP** (Section 5.1): Policy Iteration from
  (C, C) converges in two steps to "aggressive in G, conservative in B"
  with ρ = 3.5 (`results/two_state_mdp.png`); the transitions under the
  conservative action are G→G 0.9, B→B 0.7 (Section 5.1 of the report).

### Risk-sensitive portfolio

* Regimes {bull, neutral, bear} with the transition matrix of Section 4.1,
  monthly risk-free log-rate 0.2 %, log-normal risky returns depending on the
  next regime (discretised with 9-point Gauss–Hermite quadrature), allocation
  grid `a ∈ [0, 1.5]`.
* With θ = −γ/2 the multiplicative criterion becomes the ergodic equation
  `ρ + h(x) = max_a θ⁻¹ log Σ_y Q(x,y) E[G(a,Z)^θ | y] e^{θ h(y)}`, which
  reduces to the risk-neutral equation (expected log-growth) when γ → 0.
* Every optimal policy is verified independently: its growth rate is
  recomputed from the Perron–Frobenius eigenvalue of the twisted kernel
  `K(x,y) = Q(x,y) E[G(a(x),Z)^θ | y]`, and the risk-neutral rate is checked by
  a 200 000-step Monte Carlo simulation.

| γ | −4 | −1 | 0 | 1 | 4 | 16 |
|---|---|---|---|---|---|---|
| allocation in *neutral* regime | 1.50 | 1.50 | 0.98 | 0.64 | 0.31 | 0.10 |
| ρ<sub>γ</sub> (% / year) | 13.1 | 9.3 | 8.2 | 7.5 | 6.0 | 3.6 |

Risk aversion first cuts exposure in the ambiguous neutral regime, then in
the bull regime; the bear regime is never invested.
