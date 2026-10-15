# Master's in Optimization — Course Projects

Projects, reports and code from my master's in optimization at Université
Paris-Saclay. Each folder contains the report and the code of the experiments, with a
README describing the problem, the method, how to run the code and the main
results.

| # | Folder | Course | Topic |
|---|---|---|---|
| 01 | [`01-operations-research`](01-operations-research) | Operations Research (IORA) | MILP exercises, aircraft landing, home-care nurse routing and scheduling (CBC / CP-SAT) |
| 02 | [`02-derivative-free-optimization`](02-derivative-free-optimization) | Derivative-Free Optimization | CMA-ES experiments; augmented Lagrangian constraint handling in CMA-ES |
| 03 | [`03-cooperative-optimization-learning`](03-cooperative-optimization-learning) | Cooperative Optimization and Learning | Decentralized (DGD, gradient tracking, ADMM), federated (FedAvg, SCAFFOLD) and differentially private kernel regression; review of homomorphically encrypted gradient methods |
| 04 | [`04-stochastic-optimization`](04-stochastic-optimization) | Stochastic Optimization | SGD, Adagrad and Adam from scratch for multinomial logistic regression on MNIST |
| 05 | [`05-online-learning`](05-online-learning) | Online Learning | Generalized weighted averaging for nonsmooth convex optimization: proofs and experiments |
| 06 | [`06-game-theory-smart-charging`](06-game-theory-smart-charging) | Applied Game Theory | Decentralized EV smart charging with best-response dynamics and water-filling |
| 07 | [`07-dynamic-programming`](07-dynamic-programming) | MDPs and Dynamic Programming | Risk-sensitive portfolio optimization in a Markovian market; relative value and policy iteration |
| 08 | [`08-polynomial-saddle-points`](08-polynomial-saddle-points) | Algebra and Geometry in Optimization | Saddle points of polynomials via Lasserre moment relaxations |
| 09 | [`09-high-dimensional-convex-geometry`](09-high-dimensional-convex-geometry) | PGMO course | Thin shell, slicing and KLS in high-dimensional convex geometry |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Scripts are meant to be run from their own project folder (see each README).
All experiments are seeded. LaTeX sources compile with `pdflatex`/`latexmk`
or `tectonic`.
