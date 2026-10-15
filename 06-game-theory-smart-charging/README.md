# Applied Game Theory — Decentralized EV Smart Charging for Decarbonization

Practical project of the *Applied Game Theory* course: schedule the night
charging of a 15-million-vehicle EV fleet so as to minimise the CO₂ content of
French electricity, using a decentralized best-response mechanism.

## Model

* **CO₂ curve `f_CO2`** — merit-order approximation: production assets
  (2022 French capacities, wind/solar capacity factors 0.3/0.2) are sorted by
  variable cost, and each load segment receives the cumulative combustion
  emission rate (piecewise-linear, convex).
* **EV problem `P_j`** — each EV minimises the emissions of the total load
  `ℓ_NF + Σ ℓ_EV` over its plug-in window, subject to a 7 kW power limit and
  its energy need; solved with the **water-filling** heuristic.
* **Best-Response Dynamics** — EVs update sequentially until the squared move
  norm drops below η; the game is a potential game, so BRD converges (3–7
  sweeps here). 500 simulated EVs are scaled up to the national fleet
  ("meta-EVs").
* **Reference policies** — Plug-and-Charge (PC), Off-Peak with seeded random
  start in 22:00–06:00 (OP), and a synchronized Off-Peak where the whole fleet
  starts at 22:00 (OP-22, herding stress test).

Data (`data/`): RTE eCO2mix day-ahead load forecasts (2019/2020, summer and
winter days) and the EV arrival/departure/energy scenarios of the practical.

```bash
python smart_charging.py    # ~1 min, writes results/
```

Outputs: load and EV-only profiles per scenario, water-filling views of the
BRD equilibrium, BRD convergence, CO₂ comparison, nuclear-availability
sensitivity, `summary.csv`, `intermediate_metrics.csv` (emissions, EV carbon
intensity, peak/mean/std of the load) and `threshold_metrics.csv` (hours and
MW·h above the nuclear and gas merit-order thresholds).

## Results (daily CO₂, tonnes)

| Scenario | PC | OP | BRD | BRD vs. PC |
|---|---|---|---|---|
| 2019 summer | 143,790 | 141,684 | 141,684 | −1.5 % |
| 2019 winter | 395,700 | 351,850 | 332,257 | −16.0 % |
| 2020 summer | 93,699 | 88,842 | 85,640 | −8.6 % |
| 2020 winter | 470,941 | 419,486 | 399,541 | −15.2 % |

BRD is never worse than the reference policies, keeps the peak load within
2 % of the non-flexible peak (PC: +40–57 %, OP-22: +143–234 %) and never
crosses the gas threshold. The gain over PC shrinks as nuclear availability
increases (21.7 % at 0.3 → ~0 % at 0.9).

The report is [`report/smart_charging_report.pdf`](report/smart_charging_report.pdf).
