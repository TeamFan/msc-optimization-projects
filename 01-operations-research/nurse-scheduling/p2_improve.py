from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

from p2 import (
	Instance,
	TIME_LIMIT_SECONDS,
	_slugify,
	build_extended_travel,
	build_solution,
	get_large_instance,
	plot_gantt,
	plot_route_map,
	print_solution,
)


@dataclass(frozen=True)
class ImprovementOptions:
	lower_bound: bool = False
	eliminate_impossible: bool = False
	tighten_big_m: bool = False

	def summary(self) -> str:
		active = []
		if self.lower_bound:
			active.append("working-time lower bounds")
		if self.eliminate_impossible:
			active.append("prune impossible arcs")
		if self.tighten_big_m:
			active.append("tight big-M")
		return ", ".join(active) if active else "baseline symmetry only"


def _default_big_m(instance: Instance, T_ext: List[List[int]]) -> int:
	max_travel = max(max(row) for row in instance.T)
	return instance.A - instance.L + max_travel + sum(instance.d)


def _arc_is_time_feasible(instance: Instance, T_ext: List[List[int]], i: int, j: int) -> bool:
	if not (1 <= j <= instance.P):
		return True
	latest_start_j = instance.F[j - 1] - instance.d[j - 1]
	if i == 0:
		earliest_arrival = instance.L + T_ext[0][j]
	elif 1 <= i <= instance.P:
		earliest_arrival = instance.S[i - 1] + instance.d[i - 1] + T_ext[i][j]
	else:
		return True
	return earliest_arrival <= latest_start_j


def _build_arcs(
	instance: Instance,
	T_ext: List[List[int]],
	eliminate_impossible: bool,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
	arcs: List[Tuple[int, int]] = []
	removed: List[Tuple[int, int]] = []
	for i in range(0, instance.P + 1):
		for j in range(1, instance.P + 2):
			if i == j:
				continue
			if eliminate_impossible and not _arc_is_time_feasible(instance, T_ext, i, j):
				removed.append((i, j))
				continue
			arcs.append((i, j))
	return arcs, removed


def _compute_big_m_map(instance: Instance, T_ext: List[List[int]]) -> Dict[Tuple[int, int], int]:
	P = instance.P
	result: Dict[Tuple[int, int], int] = {}
	for i in range(0, P + 1):
		for j in range(1, P + 2):
			if i == j:
				continue
			if j == P + 1:
				if i == 0:
					value = instance.A - instance.L + T_ext[i][j]
				else:
					finish_max = instance.F[i - 1]
					value = finish_max + T_ext[i][j] - instance.L
			elif i == 0:
				value = instance.A + T_ext[0][j] - instance.S[j - 1]
			else:
				finish_max = instance.F[i - 1]
				value = finish_max + T_ext[i][j] - instance.S[j - 1]
			result[(i, j)] = max(0, value)
	return result


def _add_symmetry_breaking_cbc(
	solver: pywraplp.Solver,
	x: Dict[Tuple[int, int, int], pywraplp.Variable],
	dep: List[pywraplp.Variable],
	instance: Instance,
	arcs: List[Tuple[int, int]],
) -> None:
	"""Add symmetry-breaking constraints for the CBC formulation."""

	SRC = 0
	first_visit_vars: List[pywraplp.Variable] = []
	for nurse in range(instance.N):
		fv = solver.IntVar(0, instance.P + 1, f"first_visit_{nurse}")
		terms = [
			j * x[(nurse, SRC, j)]
			for j in range(1, instance.P + 2)
			if (SRC, j) in arcs
		]
		expr = solver.Sum(terms) if terms else 0
		solver.Add(fv == expr)
		first_visit_vars.append(fv)

	solver.Add(first_visit_vars[0] <= first_visit_vars[1])
	solver.Add(dep[0] <= dep[1])


def _add_symmetry_breaking_cp_sat(
	model: cp_model.CpModel,
	x: Dict[Tuple[int, int, int], cp_model.IntVar],
	dep: List[cp_model.IntVar],
	instance: Instance,
	arcs: List[Tuple[int, int]],
) -> None:
	"""Add symmetry-breaking constraints for the CP-SAT formulation."""

	SRC = 0
	first_visit_vars: List[cp_model.IntVar] = []
	for nurse in range(instance.N):
		fv = model.NewIntVar(0, instance.P + 1, f"first_visit_{nurse}")
		terms = [
			j * x[(nurse, SRC, j)]
			for j in range(1, instance.P + 2)
			if (SRC, j) in arcs
		]
		expr = sum(terms) if terms else 0
		model.Add(fv == expr)
		first_visit_vars.append(fv)

	model.Add(first_visit_vars[0] <= first_visit_vars[1])
	model.Add(dep[0] <= dep[1])


def solve_large_instance_cbc(instance: Instance, options: ImprovementOptions) -> Dict:
	"""Solve the large instance with optional cuts using CBC."""

	P = instance.P
	K = instance.N
	SRC = 0
	SNK = P + 1

	d_ext = [0] + instance.d + [0]
	T_ext = build_extended_travel(instance)
	arcs, removed_arcs = _build_arcs(instance, T_ext, options.eliminate_impossible)
	arc_set = set(arcs)

	solver = pywraplp.Solver.CreateSolver("CBC")
	if solver is None:
		raise RuntimeError("Could not create CBC solver. Ensure ortools is installed.")

	default_M = _default_big_m(instance, T_ext)
	big_m_map = _compute_big_m_map(instance, T_ext) if options.tighten_big_m else None

	x: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
	for k in range(K):
		for (i, j) in arcs:
			x[(k, i, j)] = solver.BoolVar(f"x_{k}_{i}_{j}")

	s = [None]
	for i in range(1, P + 1):
		lb = instance.S[i - 1]
		ub = instance.F[i - 1] - instance.d[i - 1]
		s.append(solver.NumVar(lb, ub, f"s_{i}"))

	dep = [solver.NumVar(instance.L, instance.A, f"dep_{k}") for k in range(K)]
	arr = [solver.NumVar(instance.L, instance.A, f"arr_{k}") for k in range(K)]

	for k in range(K):
		outbound = [x[(k, SRC, j)] for j in range(1, P + 2) if (SRC, j) in arc_set]
		inbound = [x[(k, i, SNK)] for i in range(0, P + 1) if (i, SNK) in arc_set]
		solver.Add(solver.Sum(outbound) == 1)
		solver.Add(solver.Sum(inbound) == 1)
		solver.Add(arr[k] >= dep[k])

	for k in range(K):
		for v in range(1, P + 1):
			incoming = solver.Sum(x[(k, i, v)] for i in range(0, P + 1) if (i, v) in arc_set)
			outgoing = solver.Sum(x[(k, v, j)] for j in range(1, P + 2) if (v, j) in arc_set)
			solver.Add(incoming == outgoing)

	for v in range(1, P + 1):
		solver.Add(
			solver.Sum(x[(k, i, v)] for k in range(K) for i in range(0, P + 1) if (i, v) in arc_set)
			== 1
		)
		solver.Add(
			solver.Sum(x[(k, v, j)] for k in range(K) for j in range(1, P + 2) if (v, j) in arc_set)
			== 1
		)

	for k in range(K):
		for (i, j) in arcs:
			travel = T_ext[i][j]
			var = x[(k, i, j)]
			M = big_m_map[(i, j)] if big_m_map is not None else default_M
			if i == SRC and j == SNK:
				solver.Add(arr[k] >= dep[k] + travel - M * (1 - var))
			elif i == SRC:
				solver.Add(s[j] >= dep[k] + travel - M * (1 - var))
			elif j == SNK:
				solver.Add(arr[k] >= s[i] + d_ext[i] + travel - M * (1 - var))
			else:
				solver.Add(s[j] >= s[i] + d_ext[i] + travel - M * (1 - var))

	if options.lower_bound:
		for k in range(K):
			travel_expr = sum(T_ext[i][j] * x[(k, i, j)] for (i, j) in arcs)
			service_expr = sum(
				instance.d[j - 1]
				* sum(x[(k, i, j)] for i in range(0, P + 1) if (i, j) in arc_set)
				for j in range(1, P + 1)
			)
			solver.Add(arr[k] - dep[k] >= travel_expr + service_expr)

	_add_symmetry_breaking_cbc(solver, x, dep, instance, arcs)

	objective = solver.Objective()
	for k in range(K):
		objective.SetCoefficient(arr[k], 1.0)
		objective.SetCoefficient(dep[k], -1.0)
	objective.SetMinimization()

	solver.SetTimeLimit(int(TIME_LIMIT_SECONDS * 1000))

	status = solver.Solve()
	solve_time_sec = solver.WallTime() / 1000.0

	if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
		raise RuntimeError(
			"No feasible solution found by CBC with symmetry breaking (status {} after {:.1f}s).".format(
				status, solve_time_sec
			)
		)

	s_vals = [None] + [s[i].solution_value() for i in range(1, P + 1)]
	dep_vals = [dep[k].solution_value() for k in range(K)]
	arr_vals = [arr[k].solution_value() for k in range(K)]

	solution = build_solution(
		instance,
		arcs,
		lambda nurse, i, j: x[(nurse, i, j)].solution_value() > 0.5,
		s_vals,
		dep_vals,
		arr_vals,
		T_ext,
		d_ext,
	)
	solution["status_code"] = status
	solution["status_str"] = {
		pywraplp.Solver.OPTIMAL: "OPTIMAL",
		pywraplp.Solver.FEASIBLE: "FEASIBLE (time limit)",
	}.get(status, f"STATUS_{status}")
	solution["solve_time_sec"] = solve_time_sec
	solution["engine"] = "CBC (symmetry broken)"
	solution["time_limit_sec"] = TIME_LIMIT_SECONDS
	solution["removed_arc_count"] = len(removed_arcs)
	solution["improvements"] = options.summary()
	return solution


def solve_large_instance_cp_sat(instance: Instance, options: ImprovementOptions) -> Dict:
	"""Solve the large instance with optional cuts using CP-SAT."""

	P = instance.P
	K = instance.N
	SRC = 0
	SNK = P + 1

	d_ext = [0] + instance.d + [0]
	T_ext = build_extended_travel(instance)
	arcs, removed_arcs = _build_arcs(instance, T_ext, options.eliminate_impossible)
	arc_set = set(arcs)

	model = cp_model.CpModel()

	x: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
	for k in range(K):
		for (i, j) in arcs:
			x[(k, i, j)] = model.NewBoolVar(f"x_{k}_{i}_{j}")

	s_vars: List[cp_model.IntVar] = [None]
	for i in range(1, P + 1):
		lb = instance.S[i - 1]
		ub = instance.F[i - 1] - instance.d[i - 1]
		s_vars.append(model.NewIntVar(lb, ub, f"s_{i}"))

	dep = [model.NewIntVar(instance.L, instance.A, f"dep_{k}") for k in range(K)]
	arr = [model.NewIntVar(instance.L, instance.A, f"arr_{k}") for k in range(K)]

	default_M = _default_big_m(instance, T_ext)
	big_m_map = _compute_big_m_map(instance, T_ext) if options.tighten_big_m else None

	for k in range(K):
		outbound = [x[(k, SRC, j)] for j in range(1, P + 2) if (SRC, j) in arc_set]
		inbound = [x[(k, i, SNK)] for i in range(0, P + 1) if (i, SNK) in arc_set]
		model.Add(sum(outbound) == 1)
		model.Add(sum(inbound) == 1)
		model.Add(arr[k] >= dep[k])

	for k in range(K):
		for v in range(1, P + 1):
			incoming = [x[(k, i, v)] for i in range(0, P + 1) if (i, v) in arc_set]
			outgoing = [x[(k, v, j)] for j in range(1, P + 2) if (v, j) in arc_set]
			model.Add(sum(incoming) == sum(outgoing))

	for v in range(1, P + 1):
		model.Add(
			sum(x[(k, i, v)] for k in range(K) for i in range(0, P + 1) if (i, v) in arc_set) == 1
		)
		model.Add(
			sum(x[(k, v, j)] for k in range(K) for j in range(1, P + 2) if (v, j) in arc_set) == 1
		)

	for k in range(K):
		for (i, j) in arcs:
			travel = T_ext[i][j]
			arc_var = x[(k, i, j)]
			M = big_m_map[(i, j)] if big_m_map is not None else default_M
			if i == SRC and j == SNK:
				model.Add(arr[k] >= dep[k] + travel - M * (1 - arc_var))
			elif i == SRC:
				model.Add(s_vars[j] >= dep[k] + travel - M * (1 - arc_var))
			elif j == SNK:
				model.Add(arr[k] >= s_vars[i] + d_ext[i] + travel - M * (1 - arc_var))
			else:
				model.Add(s_vars[j] >= s_vars[i] + d_ext[i] + travel - M * (1 - arc_var))

	if options.lower_bound:
		for k in range(K):
			travel_expr = sum(T_ext[i][j] * x[(k, i, j)] for (i, j) in arcs)
			service_expr = sum(
				instance.d[j - 1]
				* sum(x[(k, i, j)] for i in range(0, P + 1) if (i, j) in arc_set)
				for j in range(1, P + 1)
			)
			model.Add(arr[k] - dep[k] >= travel_expr + service_expr)

	_add_symmetry_breaking_cp_sat(model, x, dep, instance, arcs)

	model.Minimize(sum(arr[k] - dep[k] for k in range(K)))

	solver = cp_model.CpSolver()
	solver.parameters.num_search_workers = 8
	solver.parameters.max_time_in_seconds = TIME_LIMIT_SECONDS

	status = solver.Solve(model)
	solve_time_sec = solver.WallTime() / 1000.0

	if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
		raise RuntimeError(
			"No feasible solution found by CP-SAT with symmetry breaking (status {} after {:.1f}s).".format(
				status, solve_time_sec
			)
		)

	s_vals = [None] + [solver.Value(s_vars[i]) for i in range(1, P + 1)]
	dep_vals = [solver.Value(dep[k]) for k in range(K)]
	arr_vals = [solver.Value(arr[k]) for k in range(K)]

	solution = build_solution(
		instance,
		arcs,
		lambda nurse, i, j: solver.Value(x[(nurse, i, j)]) > 0.5,
		s_vals,
		dep_vals,
		arr_vals,
		T_ext,
		d_ext,
	)
	solution["status_code"] = status
	solution["status_str"] = {
		cp_model.OPTIMAL: "OPTIMAL",
		cp_model.FEASIBLE: "FEASIBLE (time limit)",
		cp_model.UNKNOWN: "UNKNOWN",
	}.get(status, f"STATUS_{status}")
	solution["solve_time_sec"] = solve_time_sec
	solution["engine"] = "CP-SAT (symmetry broken)"
	solution["time_limit_sec"] = TIME_LIMIT_SECONDS
	solution["removed_arc_count"] = len(removed_arcs)
	solution["improvements"] = options.summary()
	return solution



def solve_large_instance(engine: str, options: ImprovementOptions) -> Dict:
	engine = engine.lower()
	instance = get_large_instance()
	if engine == "cbc":
		return solve_large_instance_cbc(instance, options)
	if engine in {"cp_sat", "cpsat", "sat"}:
		return solve_large_instance_cp_sat(instance, options)
	raise ValueError(f"Unknown solver engine '{engine}'")


def parse_args():
	import argparse

	parser = argparse.ArgumentParser(description="Large multi-nurse instance with symmetry breaking and optional strengthenings.")
	parser.add_argument("--solver", choices=["cbc", "cp_sat"], help="Solver engine (prompted if omitted).")
	parser.add_argument(
		"--strengthen",
		help="Comma-separated strengthenings: 1=working-time lower bounds, 2=prune impossible arcs, 3=tight big-M. "
		"Use 'none' for the baseline (prompted if omitted).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	instance = get_large_instance()
	solver_choices = {
		"1": ("CBC MILP (pywraplp) with symmetry breaking", "cbc"),
		"2": ("CP-SAT with symmetry breaking", "cp_sat"),
	}
	print("Large instance (P=20, N=2) with symmetry breaking")
	print("Select solver engine:")
	for key, (desc, _) in solver_choices.items():
		print(f"  {key}) {desc}")
	sel = None
	if args.solver is not None:
		sel = {"cbc": "1", "cp_sat": "2"}[args.solver]
	else:
		try:
			sel = input("Enter solver choice [default 2]: ").strip()
		except EOFError:
			sel = ""
	if not sel:
		sel = "2"
	if sel not in solver_choices:
		print("Unrecognized solver; defaulting to CP-SAT (option 2).")
		sel = "2"
	solver_name, engine_key = solver_choices[sel]
	print(f"Using solver: {solver_name}")
	improvement_menu = {
		"1": ("Add working-time lower bounds", "lower_bound"),
		"2": ("Remove impossible arcs", "eliminate_impossible"),
		"3": ("Tighten big-M coefficients", "tighten_big_m"),
	}
	print("Select optional strengthenings (comma-separated, Enter for none):")
	for key, (desc, _) in improvement_menu.items():
		print(f"  {key}) {desc}")
	choice_line = None
	if args.strengthen is not None:
		choice_line = "" if args.strengthen.lower() == "none" else args.strengthen
	else:
		try:
			choice_line = input("Your selection: ").strip()
		except EOFError:
			choice_line = ""
	choices = {c.strip() for c in choice_line.split(",") if c.strip()} if choice_line else set()
	options = ImprovementOptions(
		lower_bound="1" in choices,
		eliminate_impossible="2" in choices,
		tighten_big_m="3" in choices,
	)
	print(f"Activated strengthenings: {options.summary()}")
	solution = solve_large_instance(engine_key, options)
	if options.eliminate_impossible:
		print(f"Pruned arcs incompatible with windows: {solution.get('removed_arc_count', 0)}")
	print(f"Solver perspective: {solution.get('improvements', options.summary())}")
	print_solution(instance, solution)
	chart_name = f"gantt_multi_{_slugify(instance.name)}_sym.png"
	plot_gantt(instance, solution, out_path=chart_name)
	print(f"Saved Gantt chart to {chart_name}")
	map_name = f"route_map_multi_{_slugify(instance.name)}_sym.png"
	plot_route_map(instance, solution, out_path=map_name)
	print(f"Saved route map to {map_name}")


if __name__ == "__main__":
	main()
