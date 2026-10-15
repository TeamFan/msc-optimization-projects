from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class Instance:
	name: str
	N: int
	P: int
	d: List[int]
	S: List[int]
	F: List[int]
	L: int
	A: int
	T: List[List[int]]


TIME_LIMIT_SECONDS = 300  # 5 minutes wall-clock limit per solve


def minutes_to_hhmm(m: float) -> str:
	m = int(round(m))
	h = m // 60
	mm = m % 60
	return f"{h:02d}:{mm:02d}"


def build_extended_travel(instance: Instance) -> List[List[int]]:
	n = instance.P
	size_ext = n + 2
	T_ext = [[0 for _ in range(size_ext)] for _ in range(size_ext)]
	for i in range(n + 1):
		for j in range(n + 1):
			T_ext[i][j] = instance.T[i][j]
	for i in range(1, n + 1):
		T_ext[i][n + 1] = instance.T[i][0]
	# depot direct return (0 -> sink) already zero by construction
	return T_ext


def build_solution(
	instance: Instance,
	arcs: List[Tuple[int, int]],
	get_x_value: Callable[[int, int, int], bool],
	s_vals: List[float],
	dep_vals: List[float],
	arr_vals: List[float],
	T_ext: List[List[int]],
	d_ext: List[int],
) -> Dict:
	P = instance.P
	K = instance.N
	SRC = 0
	SNK = P + 1

	def route_for_nurse(k: int) -> List[int]:
		route = [SRC]
		current = SRC
		steps = 0
		while current != SNK and steps <= P + 1:
			next_node = None
			for j in range(1, P + 2):
				if (current, j) in arcs and get_x_value(k, current, j):
					next_node = j
					break
			if next_node is None:
				break
			route.append(next_node)
			current = next_node
			steps += 1
		if route[-1] != SNK:
			route.append(SNK)
		return route

	routes = [route_for_nurse(k) for k in range(K)]

	segments = []
	for k, route in enumerate(routes):
		depart_time = dep_vals[k]
		for idx in range(len(route) - 1):
			i = route[idx]
			j = route[idx + 1]
			travel = T_ext[i][j]
			if travel > 1e-6:
				move_start = depart_time if i == SRC else (s_vals[i] + d_ext[i])
				segments.append(
					{
						"nurse": k,
						"kind": "travel",
						"start": move_start,
						"dur": travel,
						"label": f"{i}->{j}",
					}
				)
			if j == SNK:
				break
			arrival = (depart_time if i == SRC else s_vals[i] + d_ext[i]) + travel
			start_j = s_vals[j]
			wait = max(0.0, start_j - arrival)
			if wait > 1e-6:
				segments.append(
					{
						"nurse": k,
						"kind": "wait",
						"start": arrival,
						"dur": wait,
						"label": f"@{j}",
					}
				)
			service_dur = d_ext[j]
			if service_dur > 0:
				segments.append(
					{
						"nurse": k,
						"kind": "service",
						"start": start_j,
						"dur": service_dur,
						"label": f"Patient {j}",
					}
				)
			depart_time = start_j + service_dur

	working = [arr_vals[k] - dep_vals[k] for k in range(instance.N)]
	objective = sum(working)

	return {
		"routes": routes,
		"start_times": s_vals,
		"departures": dep_vals,
		"arrivals": arr_vals,
		"segments": segments,
		"working_times": working,
		"T_ext": T_ext,
		"d_ext": d_ext,
		"objective": objective,
	}


def solve_multi_nurse_cbc(instance: Instance) -> Dict:
	from ortools.linear_solver import pywraplp

	P = instance.P
	K = instance.N
	SRC = 0
	SNK = P + 1
	node_count = P + 2

	d_ext = [0] + instance.d + [0]
	T_ext = build_extended_travel(instance)

	solver = pywraplp.Solver.CreateSolver("CBC")
	if solver is None:
		raise RuntimeError("Could not create CBC solver. Ensure ortools is installed.")

	max_travel = max(max(row) for row in instance.T)
	M = instance.A - instance.L + max_travel + sum(instance.d)

	arcs = []
	for i in range(0, P + 1):
		for j in range(1, P + 2):
			if i == j:
				continue
			arcs.append((i, j))

	x = {}
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

	# Depot start and end per nurse
	for k in range(K):
		outbound = [x[(k, SRC, j)] for j in range(1, P + 2)]
		inbound = [x[(k, i, SNK)] for i in range(0, P + 1)]
		solver.Add(sum(outbound) == 1)
		solver.Add(sum(inbound) == 1)
		solver.Add(arr[k] >= dep[k])

	# Flow conservation for patients at nurse level
	for k in range(K):
		for v in range(1, P + 1):
			incoming = sum(x[(k, i, v)] for i in range(0, P + 1) if (i, v) in arcs)
			outgoing = sum(x[(k, v, j)] for j in range(1, P + 2) if (v, j) in arcs)
			solver.Add(incoming == outgoing)

	# Assign each patient exactly once (incoming and outgoing)
	for v in range(1, P + 1):
		solver.Add(sum(x[(k, i, v)] for k in range(K) for i in range(0, P + 1) if (i, v) in arcs) == 1)
		solver.Add(sum(x[(k, v, j)] for k in range(K) for j in range(1, P + 2) if (v, j) in arcs) == 1)

	# Time window bounds already embedded in variable definitions

	for k in range(K):
		for (i, j) in arcs:
			var = x[(k, i, j)]
			travel = T_ext[i][j]
			if i == SRC and j == SNK:
				solver.Add(arr[k] >= dep[k] + travel - M * (1 - var))
			elif i == SRC:
				solver.Add(s[j] >= dep[k] + travel - M * (1 - var))
			elif j == SNK:
				solver.Add(arr[k] >= s[i] + d_ext[i] + travel - M * (1 - var))
			else:
				solver.Add(s[j] >= s[i] + d_ext[i] + travel - M * (1 - var))

	objective = solver.Objective()
	for k in range(K):
		objective.SetCoefficient(arr[k], 1.0)
		objective.SetCoefficient(dep[k], -1.0)
	objective.SetMinimization()

	time_limit_ms = int(TIME_LIMIT_SECONDS * 1000)
	solver.SetTimeLimit(time_limit_ms)

	status = solver.Solve()
	solve_time_sec = solver.WallTime() / 1000.0

	if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
		raise RuntimeError("No feasible solution found by CBC (status {} after {:.1f}s).".format(status, solve_time_sec))

	def route_for_nurse(k: int) -> List[int]:
		route = [SRC]
		current = SRC
		visited = set()
		while current != SNK:
			next_node = None
			for j in range(1, P + 2):
				if (k, current, j) in x and x[(k, current, j)].solution_value() > 0.5:
					next_node = j
					break
			if next_node is None or next_node in visited:
				break
			route.append(next_node)
			visited.add(next_node)
			current = next_node
		if route[-1] != SNK:
			route.append(SNK)
		return route

	s_vals = [None] + [s[i].solution_value() for i in range(1, P + 1)]
	dep_vals = [dep[k].solution_value() for k in range(K)]
	arr_vals = [arr[k].solution_value() for k in range(K)]

	routes = [route_for_nurse(k) for k in range(K)]

	segments = []
	for k, route in enumerate(routes):
		depart_time = dep_vals[k]
		for idx in range(len(route) - 1):
			i = route[idx]
			j = route[idx + 1]
			travel = T_ext[i][j]
			if travel > 1e-6:
				segments.append({
					"nurse": k,
					"kind": "travel",
					"start": depart_time if i == SRC else (s_vals[i] + d_ext[i]),
					"dur": travel,
					"label": f"{i}->{j}"
				})
			if j == SNK:
				break
			arrival = (depart_time if i == SRC else s_vals[i] + d_ext[i]) + travel
			start_j = s_vals[j]
			wait = max(0.0, start_j - arrival)
			if wait > 1e-6:
				segments.append({
					"nurse": k,
					"kind": "wait",
					"start": arrival,
					"dur": wait,
					"label": f"@{j}"
				})
			service_dur = d_ext[j]
			segments.append({
				"nurse": k,
				"kind": "service",
				"start": start_j,
				"dur": service_dur,
				"label": f"Patient {j}"
			})
			depart_time = start_j + service_dur

	working = [arr_vals[k] - dep_vals[k] for k in range(K)]

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
		pywraplp.Solver.ABNORMAL: "ABNORMAL",
		pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
		pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
		pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
	}.get(status, f"STATUS_{status}")
	solution["solve_time_sec"] = solve_time_sec
	solution["engine"] = "CBC"
	solution["time_limit_sec"] = TIME_LIMIT_SECONDS
	return solution


def solve_multi_nurse_cp_sat(instance: Instance) -> Dict:
	from ortools.sat.python import cp_model

	P = instance.P
	K = instance.N
	SRC = 0
	SNK = P + 1

	d_ext = [0] + instance.d + [0]
	T_ext = build_extended_travel(instance)

	arcs = []
	for i in range(0, P + 1):
		for j in range(1, P + 2):
			if i == j:
				continue
			arcs.append((i, j))

	model = cp_model.CpModel()

	x = {}
	for k in range(K):
		for (i, j) in arcs:
			x[(k, i, j)] = model.NewBoolVar(f"x_{k}_{i}_{j}")

	s_vars = [None]
	for i in range(1, P + 1):
		lb = instance.S[i - 1]
		ub = instance.F[i - 1] - instance.d[i - 1]
		s_vars.append(model.NewIntVar(lb, ub, f"s_{i}"))

	dep = [model.NewIntVar(instance.L, instance.A, f"dep_{k}") for k in range(K)]
	arr = [model.NewIntVar(instance.L, instance.A, f"arr_{k}") for k in range(K)]

	max_travel = max(max(row) for row in instance.T)
	M = instance.A - instance.L + max_travel + sum(instance.d)

	for k in range(K):
		outbound = [x[(k, SRC, j)] for j in range(1, P + 2) if (SRC, j) in arcs]
		inbound = [x[(k, i, SNK)] for i in range(0, P + 1) if (i, SNK) in arcs]
		model.Add(sum(outbound) == 1)
		model.Add(sum(inbound) == 1)
		model.Add(arr[k] >= dep[k])

	for k in range(K):
		for v in range(1, P + 1):
			incoming = [x[(k, i, v)] for i in range(0, P + 1) if (i, v) in arcs]
			outgoing = [x[(k, v, j)] for j in range(1, P + 2) if (v, j) in arcs]
			model.Add(sum(incoming) == sum(outgoing))

	for v in range(1, P + 1):
		model.Add(
			sum(x[(k, i, v)] for k in range(K) for i in range(0, P + 1) if (i, v) in arcs)
			== 1
		)
		model.Add(
			sum(x[(k, v, j)] for k in range(K) for j in range(1, P + 2) if (v, j) in arcs)
			== 1
		)

	for k in range(K):
		for (i, j) in arcs:
			travel = T_ext[i][j]
			arc_var = x[(k, i, j)]
			if i == SRC and j == SNK:
				model.Add(arr[k] >= dep[k] + travel - M * (1 - arc_var))
			elif i == SRC:
				model.Add(s_vars[j] >= dep[k] + travel - M * (1 - arc_var))
			elif j == SNK:
				model.Add(arr[k] >= s_vars[i] + d_ext[i] + travel - M * (1 - arc_var))
			else:
				model.Add(s_vars[j] >= s_vars[i] + d_ext[i] + travel - M * (1 - arc_var))

	model.Minimize(sum(arr[k] - dep[k] for k in range(K)))

	solver = cp_model.CpSolver()
	solver.parameters.num_search_workers = 8
	solver.parameters.max_time_in_seconds = TIME_LIMIT_SECONDS

	status = solver.Solve(model)
	solve_time_sec = solver.WallTime() / 1000.0

	if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
		raise RuntimeError("No feasible solution found by CP-SAT (status {} after {:.1f}s).".format(status, solve_time_sec))

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
		cp_model.INFEASIBLE: "INFEASIBLE",
		cp_model.MODEL_INVALID: "MODEL_INVALID",
		cp_model.UNKNOWN: "UNKNOWN",
	}.get(status, f"STATUS_{status}")
	solution["solve_time_sec"] = solve_time_sec
	solution["engine"] = "CP-SAT"
	solution["time_limit_sec"] = TIME_LIMIT_SECONDS
	return solution


def solve_multi_nurse(instance: Instance, engine: str) -> Dict:
	engine = engine.lower()
	if engine == "cbc":
		return solve_multi_nurse_cbc(instance)
	if engine in {"cp_sat", "cpsat", "sat"}:
		return solve_multi_nurse_cp_sat(instance)
	raise ValueError(f"Unknown solver engine '{engine}'")


def plot_gantt(instance: Instance, solution: Dict, out_path: str) -> None:
	import matplotlib.pyplot as plt

	segments = solution["segments"]
	if not segments:
		return

	nurses = sorted({seg["nurse"] for seg in segments})
	kinds = ["travel", "wait", "service"]
	colors = {
		"travel": "#7f7f7f",
		"wait": "#ffc658",
		"service": "#1f77b4",
	}

	segments_by_nurse: Dict[int, List[Dict]] = {n: [] for n in nurses}
	for seg in segments:
		if seg["dur"] > 1e-6:
			segments_by_nurse[seg["nurse"]].append(seg)

	fig, axes = plt.subplots(len(nurses), 1, sharex=True, figsize=(13, 2.2 * len(nurses) + 2))
	if len(nurses) == 1:
		axes = [axes]

	kind_offsets = {
		"travel": (0.05, 0.25),
		"wait": (0.35, 0.25),
		"service": (0.65, 0.3),
	}
	ytick_positions = [kind_offsets[k][0] + kind_offsets[k][1] / 2 for k in kinds]
	ytick_labels = [k.capitalize() for k in kinds]

	for idx, nurse in enumerate(nurses):
		ax = axes[idx]
		nurse_segs = sorted(segments_by_nurse[nurse], key=lambda seg: seg["start"])
		for seg in nurse_segs:
			x0 = seg["start"]
			dur = seg["dur"]
			kind = seg["kind"]
			y0, height = kind_offsets[kind]
			ax.broken_barh([(x0, dur)], (y0, height), facecolors=colors[kind], edgecolor="k", linewidth=0.6, alpha=0.85)
			ax.text(x0 + dur / 2, y0 + height + 0.02, seg["label"], ha="center", va="bottom", fontsize=7, rotation=25)

		ax.set_ylim(0, 1)
		ax.set_ylabel(f"Nurse {nurse + 1}")
		ax.set_yticks(ytick_positions)
		ax.set_yticklabels(ytick_labels)
		ax.grid(axis="x", linestyle="--", alpha=0.3)

	axes[-1].set_xlabel("Time (minutes from 00:00)")
	axes[-1].set_xlim(max(instance.L - 10, 0), instance.A + 10)

	ticks = list(range((instance.L // 30) * 30, instance.A + 1, 30))
	axes[-1].set_xticks(ticks)
	axes[-1].set_xticklabels([minutes_to_hhmm(t) for t in ticks], rotation=40, ha="right")

	total_work = sum(solution["working_times"])
	fig.suptitle(
		f"Gantt (multi-nurse) - {instance.name}\nTotal working time {minutes_to_hhmm(total_work)}",
		fontsize=12,
	)

	from matplotlib.patches import Patch

	legend_handles = [
		Patch(facecolor=colors[k], edgecolor="k", linewidth=0.6, label=k.capitalize())
		for k in kinds
	]
	axes[0].legend(handles=legend_handles, loc="upper right", fontsize=8)

	fig.tight_layout(rect=(0, 0, 1, 0.92))
	fig.savefig(out_path, dpi=200)
	plt.close(fig)


def estimate_coordinates(instance: Instance) -> Dict[int, Tuple[float, float]]:
	dist = np.array(instance.T, dtype=float)
	n = dist.shape[0]
	if n <= 1:
		return {0: (0.0, 0.0)}
	d2 = dist ** 2
	J = np.eye(n) - np.ones((n, n)) / n
	B = -0.5 * J @ d2 @ J
	vals, vecs = np.linalg.eigh(B)
	idx = np.argsort(vals)[::-1]
	components = []
	for i in idx:
		if vals[i] > 1e-6 and len(components) < 2:
			components.append(np.sqrt(vals[i]) * vecs[:, i])
	if not components:
		components = [vecs[:, idx[0]] * np.sqrt(max(vals[idx[0]], 1e-6))]
	if len(components) == 1:
		components.append(np.zeros(n))
	coords = np.vstack(components).T
	coords -= coords.mean(axis=0, keepdims=True)
	max_span = np.max(np.sqrt((coords ** 2).sum(axis=1)))
	if max_span > 0:
		coords /= max_span
	return {i: (float(coords[i, 0]), float(coords[i, 1])) for i in range(n)}


def plot_route_map(instance: Instance, solution: Dict, out_path: str) -> None:
	import matplotlib.pyplot as plt

	coords = estimate_coordinates(instance)
	fig, ax = plt.subplots(figsize=(7, 7))

	depot_xy = coords.get(0, (0.0, 0.0))
	ax.scatter([depot_xy[0]], [depot_xy[1]], marker="s", s=140, color="#d62728", label="Depot")

	patients = list(range(1, instance.P + 1))
	xs = [coords[i][0] for i in patients]
	ys = [coords[i][1] for i in patients]
	ax.scatter(xs, ys, marker="o", s=90, color="#1f77b4", label="Patients")

	for p in patients:
		x, y = coords[p]
		ax.text(x, y + 0.03, f"{p}", ha="center", va="bottom", fontsize=8)

	color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

	routes = solution["routes"]
	for idx, route in enumerate(routes):
		color = color_cycle[idx % len(color_cycle)]
		for step in range(len(route) - 1):
			start = route[step]
			end = route[step + 1]
			if start == end:
				continue
			start_idx = 0 if start in (0, instance.P + 1) else start
			end_idx = 0 if end in (0, instance.P + 1) else end
			sx, sy = coords.get(start_idx, depot_xy)
			ex, ey = coords.get(end_idx, depot_xy)
			if abs(sx - ex) < 1e-6 and abs(sy - ey) < 1e-6:
				continue
			ax.annotate(
				"",
				xy=(ex, ey),
				xytext=(sx, sy),
				arrowprops=dict(arrowstyle="->", lw=2, color=color, shrinkA=5, shrinkB=5),
			)
		# add label for nurse near depot offset
		ax.text(
			depot_xy[0] - 0.02,
			depot_xy[1] - 0.05 - idx * 0.05,
			f"Nurse {idx + 1}",
			color=color,
			ha="right",
			va="top",
			fontsize=8,
		)

	ax.set_title(f"Spatial routes - {instance.name}")
	ax.set_xlabel("Relative east-west")
	ax.set_ylabel("Relative north-south")
	ax.legend(loc="upper right")
	ax.set_aspect("equal", adjustable="box")
	ax.grid(True, linestyle="--", alpha=0.3)

	fig.tight_layout()
	fig.savefig(out_path, dpi=200)
	plt.close(fig)


def print_solution(instance: Instance, solution: Dict) -> None:
	routes = solution["routes"]
	s_vals = solution["start_times"]
	dep = solution["departures"]
	arr = solution["arrivals"]
	work = solution["working_times"]
	d_ext = solution["d_ext"]
	status_str = solution.get("status_str", "UNKNOWN")
	engine = solution.get("engine")
	solve_time = solution.get("solve_time_sec")
	objective = solution.get("objective", sum(work))

	print(f"Instance: {instance.name}")
	if engine:
		print(f"Solver: {engine} | Status: {status_str}")
	else:
		print(f"Status: {status_str}")
	if solve_time is not None:
		print(f"Solve time: {solve_time:.1f}s (limit {solution.get('time_limit_sec', 'n/a')}s)")
	print(f"Best total working time found: {objective:.1f} min ({minutes_to_hhmm(objective)})")

	total_work = 0.0
	for k, route in enumerate(routes):
		print(f"Nurse {k + 1} route: {' -> '.join(map(str, route))}")
		if len(route) <= 2:
			print(f"  Nurse {k + 1} is idle (working time 0)")
		for node in route:
			if 1 <= node <= instance.P:
				start = s_vals[node]
				end = start + d_ext[node]
				print(f"  Patient {node}: start {minutes_to_hhmm(start)}, end {minutes_to_hhmm(end)}")
		print(
			f"  Depart {minutes_to_hhmm(dep[k])}, return {minutes_to_hhmm(arr[k])}, working {minutes_to_hhmm(work[k])}"
		)
		total_work += work[k]
	print(f"Total working time (from schedule): {total_work:.1f} min ({minutes_to_hhmm(total_work)})")


def get_small_instance() -> Instance:
	return Instance(
		name="Two nurses, ten patients (07:00-11:00 horizon)",
		N=2,
		P=10,
		d=[20, 30, 25, 10, 25, 30, 10, 40, 10, 15],
		S=[420, 420, 420, 480, 480, 540, 540, 540, 540, 600],
		F=[480, 480, 660, 540, 540, 600, 660, 660, 600, 660],
		L=420,
		A=660,
		T=[
			[0, 16, 11, 13, 5, 12, 5, 11, 11, 16, 14],
			[16, 0, 11, 17, 11, 9, 15, 25, 27, 30, 27],
			[11, 11, 0, 5, 7, 15, 14, 16, 20, 27, 25],
			[13, 17, 5, 0, 10, 20, 17, 13, 19, 27, 26],
			[5, 11, 7, 10, 0, 10, 7, 14, 16, 21, 18],
			[12, 9, 15, 20, 10, 0, 9, 23, 23, 23, 19],
			[5, 15, 14, 17, 7, 9, 0, 16, 14, 15, 12],
			[11, 25, 16, 13, 14, 23, 16, 0, 7, 18, 18],
			[11, 27, 20, 19, 16, 23, 14, 7, 0, 11, 12],
			[16, 30, 27, 27, 21, 23, 15, 18, 11, 0, 4],
			[14, 27, 25, 26, 18, 19, 12, 18, 12, 4, 0],
		],
	)


def get_large_instance() -> Instance:
	return Instance(
		name="Two nurses, twenty patients (07:00-15:00 horizon)",
		N=2,
		P=20,
		d=[40, 20, 25, 30, 15, 30, 20, 60, 30, 10, 20, 10, 25, 10, 25, 60, 30, 15, 60, 20],
		S=[420, 540, 600, 660, 420, 540, 600, 540, 600, 660, 420, 480, 480, 540, 600, 540, 720, 420, 420, 480],
		F=[540, 660, 720, 900, 900, 720, 720, 720, 900, 900, 540, 600, 600, 660, 900, 900, 900, 840, 900, 640],
		L=420,
		A=900,
		T=[
			[0, 21, 21, 25, 17, 23, 14, 7, 19, 11, 15, 20, 11, 17, 17, 9, 13, 21, 18, 27, 20],
			[21, 0, 5, 16, 6, 26, 18, 14, 26, 16, 27, 25, 25, 29, 34, 29, 33, 36, 38, 45, 40],
			[21, 5, 0, 10, 4, 31, 22, 15, 30, 13, 30, 21, 23, 25, 35, 28, 33, 33, 39, 47, 39],
			[25, 16, 10, 0, 12, 40, 30, 22, 39, 14, 37, 15, 22, 23, 41, 29, 38, 30, 42, 52, 39],
			[17, 6, 4, 12, 0, 28, 19, 11, 27, 10, 26, 19, 20, 23, 32, 25, 30, 31, 35, 43, 35],
			[23, 26, 31, 40, 28, 0, 10, 21, 5, 31, 11, 42, 34, 40, 20, 31, 25, 44, 29, 30, 39],
			[14, 18, 22, 30, 19, 10, 0, 10, 9, 21, 9, 32, 25, 30, 18, 23, 20, 35, 25, 30, 33],
			[7, 14, 15, 22, 11, 21, 10, 0, 18, 10, 16, 21, 16, 21, 21, 16, 19, 27, 24, 32, 27],
			[19, 26, 30, 39, 27, 5, 9, 18, 0, 28, 6, 39, 30, 36, 14, 26, 20, 40, 23, 25, 34],
			[11, 16, 13, 14, 10, 31, 21, 10, 28, 0, 25, 11, 10, 13, 28, 16, 23, 21, 28, 38, 26],
			[15, 27, 30, 37, 26, 11, 9, 16, 6, 25, 0, 35, 26, 31, 9, 21, 14, 34, 18, 21, 28],
			[20, 25, 21, 15, 19, 42, 32, 21, 39, 11, 35, 0, 11, 9, 36, 19, 30, 16, 34, 44, 26],
			[11, 25, 23, 22, 20, 34, 25, 16, 30, 10, 26, 11, 0, 5, 25, 8, 19, 11, 22, 33, 17],
			[17, 29, 25, 23, 23, 40, 30, 21, 36, 13, 31, 9, 5, 0, 30, 12, 23, 8, 26, 37, 18],
			[17, 34, 35, 41, 32, 20, 18, 21, 14, 28, 9, 36, 25, 30, 0, 18, 8, 31, 9, 12, 22],
			[9, 29, 28, 29, 25, 31, 23, 16, 26, 16, 21, 19, 8, 12, 18, 0, 11, 13, 14, 25, 11],
			[13, 33, 33, 38, 30, 25, 20, 19, 20, 23, 14, 30, 19, 23, 8, 11, 0, 24, 5, 14, 15],
			[21, 36, 33, 30, 31, 44, 35, 27, 40, 21, 34, 16, 11, 8, 31, 13, 24, 0, 25, 35, 13],
			[18, 38, 39, 42, 35, 29, 25, 24, 23, 28, 18, 34, 22, 26, 9, 14, 5, 25, 0, 10, 14],
			[27, 45, 47, 52, 43, 30, 30, 32, 25, 38, 21, 44, 33, 37, 12, 25, 14, 35, 10, 0, 23],
			[20, 40, 39, 39, 35, 39, 33, 27, 34, 26, 28, 26, 17, 18, 22, 11, 15, 13, 14, 23, 0],
		],
	)


def _slugify(name: str) -> str:
	chars = [(c.lower() if c.isalnum() else "_") for c in name]
	slug = "".join(chars)
	return "_".join(filter(None, slug.split("_")))


def parse_args():
	import argparse

	parser = argparse.ArgumentParser(description="Multi-nurse routing and scheduling MILP.")
	parser.add_argument("--instance", choices=["small", "large"], help="Instance to solve (prompted if omitted).")
	parser.add_argument("--solver", choices=["cbc", "cp_sat"], help="Solver engine (prompted if omitted).")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	options = {
		"1": ("Smaller instance (2 nurses, 10 patients, horizon 07:00-11:00)", get_small_instance),
		"2": ("Larger instance (2 nurses, 20 patients, horizon 07:00-15:00)", get_large_instance),
	}
	print("Select instance to solve:")
	for key, (desc, _) in options.items():
		print(f"  {key}) {desc}")
	choice = None
	if args.instance is not None:
		choice = {"small": "1", "large": "2"}[args.instance]
	else:
		try:
			choice = input("Enter choice [default 1]: ").strip()
		except EOFError:
			choice = ""
	if not choice:
		choice = "1"
	if choice not in options:
		print("Unrecognized choice; defaulting to option 1.")
		choice = "1"
	instance = options[choice][1]()
	solver_choices = {
		"1": ("CBC MILP (pywraplp)", "cbc"),
		"2": ("CP-SAT (integer MILP)", "cp_sat"),
	}
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
	solution = solve_multi_nurse(instance, engine_key)
	print_solution(instance, solution)
	chart_name = f"gantt_multi_{_slugify(instance.name)}.png"
	plot_gantt(instance, solution, out_path=chart_name)
	print(f"Saved Gantt chart to {chart_name}")
	map_name = f"route_map_multi_{_slugify(instance.name)}.png"
	plot_route_map(instance, solution, out_path=map_name)
	print(f"Saved route map to {map_name}")


if __name__ == "__main__":
	main()
