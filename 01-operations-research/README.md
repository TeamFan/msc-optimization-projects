# Operations Research — Mixed-Integer Programming

Course: *Introduction to Operations Research and its Applications* (IORA).

## Contents

| Folder | Topic |
|---|---|
| [`mixed-integer-exercises/`](mixed-integer-exercises) | Small integer programs solved with OR-Tools / CBC, plus the aircraft landing scheduling MILP |
| [`nurse-scheduling/`](nurse-scheduling) | Course project: home-care nurse routing and scheduling with time windows (single and multi-nurse MILPs) |

## Mixed-integer exercises

| Script | Model | Optimal value |
|---|---|---|
| `ex1_1_integer_program.py` | General integer covering program (5 variables) | 50 |
| `ex1_2_binary_knapsack.py` | Binary knapsack with logical constraints | 14 |
| `ex2_1_staff_covering.py` | Shift/staff covering problem | 413 |
| `ex4_aircraft_landing.py` | Single-runway aircraft landing with disjunctive separation constraints (OR-Library `airland1`) | 700 |

The aircraft landing model uses the standard big-M linearisation of the
"p before q or q before p" disjunction, with
`M_pq = LAT_p + l_pq - EAT_q`. The value 700 matches the known optimum of the
OR-Library instance `airland1`.

## Nurse scheduling project

* `p1.py` — single nurse visiting every patient once within their time
  windows; minimises the working time (departure to return). The MILP uses
  binary arc variables, continuous start times and big-M precedence
  constraints, which also eliminate subtours.
* `p2.py` — team of nurses: every patient is served by exactly one nurse,
  idle nurses use the depot→sink arc; minimises total team working time.
  Supports CBC and CP-SAT.
* `p2_improve.py` — the large multi-nurse instance with symmetry breaking and
  optional strengthenings (working-time lower bounds, pruning of arcs that
  are infeasible with respect to time windows, arc-specific big-M values).

```bash
python nurse-scheduling/p1.py --instance small        # 186 min, optimal
python nurse-scheduling/p1.py --instance large        # 337 min, optimal
python nurse-scheduling/p2.py --instance small --solver cp_sat   # 354 min, optimal
python nurse-scheduling/p2.py --instance large --solver cp_sat   # 300 s time limit
python nurse-scheduling/p2_improve.py --solver cp_sat --strengthen 1,2,3
```

Without arguments the scripts ask for the instance and solver interactively.
Each run writes a Gantt chart and a route map (coordinates are reconstructed
from the travel-time matrix with classical MDS) to the working directory;
the figures used in the report are in `nurse-scheduling/figures/`.

The report is in [`nurse-scheduling/report/nurse_scheduling_report.pdf`](nurse-scheduling/report/nurse_scheduling_report.pdf).
