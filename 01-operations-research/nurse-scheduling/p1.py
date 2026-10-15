from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import math
import numpy as np


@dataclass
class Instance:
	name: str
	P: int
	d: List[int]  # service durations for patients 1..P
	S: List[int]  # earliest start times for patients 1..P
	F: List[int]  # latest finish times for patients 1..P
	L: int        # earliest departure from depot
	A: int        # latest return to depot
	T: List[List[int]]  # (P+1) x (P+1) matrix with node 0 as depot


def minutes_to_hhmm(m: float) -> str:
	m = int(round(m))
	h = m // 60
	mm = m % 60
	return f"{h:02d}:{mm:02d}"


def build_extended_travel(instance: Instance) -> List[List[int]]:
	"""Extend travel matrix to include sink node P+1.
	Indices: 0..P, P+1. For i in 1..P: T[i][P+1] = T[i][0]; From sink we don't travel.
	"""
	n = instance.P
	size_ext = n + 2
	# initialize with large values
	T_ext = [[0 for _ in range(size_ext)] for _ in range(size_ext)]
	# fill depot+patients block 0..P
	for i in range(n + 1):
		for j in range(n + 1):
			T_ext[i][j] = instance.T[i][j]
	# from patients to sink
	for i in range(1, n + 1):
		T_ext[i][n + 1] = instance.T[i][0]
	# from source to sink unused; from sink to anywhere unused => keep zeros; will not create arcs.
	return T_ext


def solve_single_nurse(instance: Instance) -> Dict:
	from ortools.linear_solver import pywraplp

	P = instance.P
	N = P + 2  # including source 0 and sink P+1
	SRC = 0
	SNK = P + 1

	d = [0] + instance.d + [0]  # service durations with depot nodes
	S = [instance.L] + instance.S + [0]  # lower bounds; sink will be bounded by A separately
	F = [instance.A] + instance.F + [instance.A]  # upper bounds placeholders; we'll only use for patients
	T_ext = build_extended_travel(instance)

	# Create solver
	solver = pywraplp.Solver.CreateSolver("CBC")
	if solver is None:
		raise RuntimeError("Could not create CBC solver. Ensure ortools is installed.")

	# Big-M: safe upper bound on time differences
	max_T = max(max(row) for row in instance.T)
	M = instance.A - instance.L + max_T + sum(instance.d)

	# Feasible arcs: i in 0..P, j in 1..P+1, i != j, and exclude (0->P+1) to force visiting patients
	arcs = []
	for i in range(0, P + 1):
		for j in range(1, P + 2):
			if i == j:
				continue
			if i == SRC and j == SNK:
				continue  # disallow skipping all patients
			arcs.append((i, j))

	# Variables
	x = {(i, j): solver.BoolVar(f"x_{i}_{j}") for (i, j) in arcs}
	s = [solver.NumVar(0.0, float("inf"), f"s_{i}") for i in range(N)]

	# Flow constraints
	# Start from depot
	solver.Add(sum(x[(SRC, j)] for j in range(1, P + 2) if (SRC, j) in x) == 1)
	# End at sink
	solver.Add(sum(x[(i, SNK)] for i in range(0, P + 1) if (i, SNK) in x) == 1)
	# Each patient has exactly one predecessor and one successor
	for k in range(1, P + 1):
		solver.Add(sum(x[(i, k)] for i in range(0, P + 1) if (i, k) in x) == 1)
		solver.Add(sum(x[(k, j)] for j in range(1, P + 2) if (k, j) in x) == 1)

	# Time windows at depot and sink
	solver.Add(s[SRC] >= instance.L)
	solver.Add(s[SNK] <= instance.A)
	# Time windows for patients: S_k <= s_k <= F_k - d_k
	for k in range(1, P + 1):
		solver.Add(s[k] >= instance.S[k - 1])
		solver.Add(s[k] <= instance.F[k - 1] - instance.d[k - 1])

	# Temporal precedence constraints with big-M
	for (i, j) in arcs:
		solver.Add(s[j] >= s[i] + d[i] + T_ext[i][j] - M * (1 - x[(i, j)]))

	# Objective: minimize working time = s_sink - s_src
	objective = solver.Objective()
	objective.SetCoefficient(s[SNK], 1.0)
	objective.SetCoefficient(s[SRC], -1.0)
	objective.SetMinimization()

	status = solver.Solve()

	if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
		raise RuntimeError("No feasible solution found by CBC.")

	# Extract route by following active arcs from source to sink
	def next_of(i: int) -> int:
		for j in range(1, P + 2):
			if (i, j) in x and x[(i, j)].solution_value() > 0.5:
				return j
		return -1

	route = [SRC]
	cur = SRC
	while cur != SNK:
		cur = next_of(cur)
		if cur == -1:
			break
		route.append(cur)

	s_val = [s[i].solution_value() for i in range(N)]

	# Build timeline segments: list of (type, start, duration, from, to)
	segments = []
	# Traverse route to compute arrivals, waits, services, and travels
	for idx in range(len(route) - 1):
		i = route[idx]
		j = route[idx + 1]
		travel = T_ext[i][j]
		depart_i = s_val[i] + d[i]
		arrival_j = depart_i + travel
		start_j = s_val[j]
		wait = max(0.0, start_j - arrival_j)
		# travel segment
		segments.append({
			"kind": "travel", "from": i, "to": j, "start": depart_i, "dur": travel
		})
		# wait segment (if any)
		if wait > 1e-6:
			segments.append({
				"kind": "wait", "at": j, "start": arrival_j, "dur": wait
			})
		# service at patient j (if j is a patient)
		if 1 <= j <= P:
			segments.append({
				"kind": "service", "at": j, "start": start_j, "dur": d[j]
			})

	working_time = s_val[SNK] - s_val[SRC]

	return {
		"status": status,
		"route": route,  # includes 0 ... P+1
		"start_times": s_val,
		"segments": segments,
		"working_time": working_time,
		"T_ext": T_ext,
		"d_ext": d,
	}


def plot_gantt(instance: Instance, solution: Dict, out_path: str) -> None:
	import matplotlib.pyplot as plt
	from matplotlib.patches import Rectangle

	segs = sorted(solution["segments"], key=lambda s: s["start"]) if solution["segments"] else []
	if not segs:
		return

	kind_order = ["travel", "wait", "service"]
	active_kinds = [k for k in kind_order if any(seg["kind"] == k for seg in segs)]
	lane_height = 0.8
	lane_gap = 0.3
	fig_height = 1.5 + len(active_kinds) * (lane_height + lane_gap)
	fig, ax = plt.subplots(figsize=(12, fig_height))

	colors = {
		"travel": "#7f7f7f",
		"wait": "#ffc658",
		"service": "#1f77b4",
	}

	lane_centers = {}
	for idx, kind in enumerate(active_kinds):
		y0 = idx * (lane_height + lane_gap)
		lane_centers[kind] = y0 + lane_height / 2

	for seg in segs:
		kind = seg["kind"]
		x0 = seg["start"]
		w = seg["dur"]
		lane_center = lane_centers[kind]
		y0 = lane_center - lane_height / 2
		label = ""
		if kind == "travel":
			label = f"{seg['from']}->{seg['to']} ({w:.0f} min)"
		elif kind == "wait":
			label = f"Wait @{seg['at']} ({w:.0f} min)"
		else:
			label = f"Patient {seg['at']} ({w:.0f} min)"
		rect = Rectangle((x0, y0), w, lane_height, facecolor=colors[kind], edgecolor="k", linewidth=0.6, alpha=0.85)
		ax.add_patch(rect)
		ax.text(x0 + w / 2, y0 + lane_height + 0.05, label, ha="center", va="bottom", fontsize=8, rotation=25)

	s0 = solution["start_times"][0]
	sN = solution["start_times"][instance.P + 1]
	ax.set_ylim(-lane_gap, len(active_kinds) * (lane_height + lane_gap))
	ax.set_xlim(max(instance.L - 10, 0), instance.A + 10)
	ax.set_xlabel("Time (minutes from 00:00)")
	ax.set_yticks([lane_centers[k] for k in active_kinds])
	ax.set_yticklabels([k.capitalize() for k in active_kinds])
	ax.set_title(
		f"Gantt - {instance.name}\nWorking {minutes_to_hhmm(sN - s0)} (dep {minutes_to_hhmm(s0)} -> arr {minutes_to_hhmm(sN)})"
	)
	ax.grid(axis="x", linestyle="--", alpha=0.3)

	ticks = list(range((instance.L // 30) * 30, instance.A + 1, 30))
	ax.set_xticks(ticks)
	ax.set_xticklabels([minutes_to_hhmm(t) for t in ticks], rotation=40, ha="right")

	legend_handles = [
		Rectangle((0, 0), 1, 1, facecolor=colors[k], edgecolor="k", linewidth=0.6, label=k.capitalize())
		for k in active_kinds
	]
	ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

	fig.tight_layout()
	fig.savefig(out_path, dpi=180)
	plt.close(fig)


def estimate_coordinates(instance: Instance) -> Dict[int, Tuple[float, float]]:
	dist = np.array(instance.T, dtype=float)
	n = dist.shape[0]
	if n <= 1:
		return {0: (0.0, 0.0)}
	# Classical MDS
	d2 = dist ** 2
	J = np.eye(n) - np.ones((n, n)) / n
	B = -0.5 * J @ d2 @ J
	vals, vecs = np.linalg.eigh(B)
	idx = np.argsort(vals)[::-1]
	coords_components = []
	for i in idx:
		if vals[i] > 1e-6 and len(coords_components) < 2:
			coords_components.append(np.sqrt(vals[i]) * vecs[:, i])
	if not coords_components:
		coords_components = [vecs[:, idx[0]] * np.sqrt(max(vals[idx[0]], 1e-6))]
	if len(coords_components) == 1:
		coords_components.append(np.zeros(n))
	coords = np.vstack(coords_components).T
	# Normalize for readability
	coords = coords - coords.mean(axis=0, keepdims=True)
	max_span = np.max(np.sqrt((coords ** 2).sum(axis=1)))
	if max_span > 0:
		coords /= max_span
	return {i: (float(coords[i, 0]), float(coords[i, 1])) for i in range(n)}


def plot_route_map(instance: Instance, solution: Dict, out_path: str) -> None:
	import matplotlib.pyplot as plt

	coords = estimate_coordinates(instance)
	fig, ax = plt.subplots(figsize=(6, 6))

	depot_xy = coords.get(0, (0.0, 0.0))
	ax.scatter([depot_xy[0]], [depot_xy[1]], marker="s", s=120, color="#d62728", label="Depot")

	patient_indices = list(range(1, instance.P + 1))
	patient_x = [coords[i][0] for i in patient_indices]
	patient_y = [coords[i][1] for i in patient_indices]
	ax.scatter(patient_x, patient_y, marker="o", s=90, color="#1f77b4", label="Patients")

	for i in patient_indices:
		x, y = coords[i]
		ax.text(x, y + 0.035, f"{i}", ha="center", va="bottom", fontsize=9)

	# Draw route arrows
	route = solution["route"]
	for idx in range(len(route) - 1):
		start = route[idx]
		end = route[idx + 1]
		start_idx = 0 if start == instance.P + 1 else start
		end_idx = 0 if end == instance.P + 1 else end
		sx, sy = coords[start_idx]
		ex, ey = coords[end_idx]
		arrowprops = dict(arrowstyle="->", color="#ff7f0e", lw=2, shrinkA=5, shrinkB=5)
		ax.annotate("", xy=(ex, ey), xytext=(sx, sy), arrowprops=arrowprops)

	ax.set_title(f"Spatial route - {instance.name}")
	ax.set_xlabel("Relative east-west")
	ax.set_ylabel("Relative north-south")
	ax.legend(loc="upper right")
	ax.set_aspect("equal", adjustable="box")
	ax.grid(True, linestyle="--", alpha=0.3)

	fig.tight_layout()
	fig.savefig(out_path, dpi=180)
	plt.close(fig)


def print_solution(instance: Instance, sol: Dict) -> None:
	P = instance.P
	route = sol["route"]
	s = sol["start_times"]
	d = sol["d_ext"]

	print(f"Instance: {instance.name}")
	print("Optimal route (including depots):", " -> ".join(map(str, route)))
	print("Schedule:")
	for i in route:
		if i == 0:
			print(f"  Node 0 (Depot start): depart at {minutes_to_hhmm(s[i])}")
		elif i == P + 1:
			print(f"  Node {P+1} (Depot end): arrive at {minutes_to_hhmm(s[i])}")
		else:
			print(
				f"  Patient {i}: start {minutes_to_hhmm(s[i])}, end {minutes_to_hhmm(s[i] + d[i])}"
			)
	print(f"Working time: {sol['working_time']:.1f} min ({minutes_to_hhmm(sol['working_time'])})")


def get_small_instance() -> Instance:
	# From the provided data (times in minutes)
	return Instance(
		name="Small P=5 (07:00-11:00)",
		P=5,
		d=[20, 10, 15, 15, 25],
		S=[420, 480, 480, 420, 540],
		F=[480, 540, 660, 540, 660],
		L=420,
		A=660,
		T=[
			[0, 14, 11, 10, 22, 20],
			[14, 0, 15, 22, 36, 32],
			[11, 15, 0, 11, 30, 30],
			[10, 22, 11, 0, 20, 22],
			[22, 36, 30, 20, 0, 10],
			[20, 32, 30, 22, 10, 0],
		],
	)


def get_large_instance() -> Instance:
	# Second dataset (times in minutes)
	return Instance(
		name="Large P=10 (07:00-15:00)",
		P=10,
		d=[20, 10, 15, 15, 25, 10, 25, 15, 10, 20],
		S=[420, 420, 480, 720, 420, 420, 720, 540, 420, 420],
		F=[600, 600, 720, 900, 480, 900, 780, 720, 600, 540],
		L=420,
		A=900,
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


def _slugify(name: str) -> str:
	chars = [
		(c.lower() if c.isalnum() else "_")
		for c in name
	]
	slug = "".join(chars)
	return "_".join(filter(None, slug.split("_")))


def choose_instance(preset: str | None = None) -> Instance:
	options = {
		"1": ("Small instance (5 patients, horizon 07:00-11:00)", get_small_instance),
		"2": ("Large instance (10 patients, horizon 07:00-15:00)", get_large_instance),
	}
	if preset is not None:
		return options[{"small": "1", "large": "2"}[preset]][1]()
	print("Select instance to solve:")
	for key, (desc, _) in options.items():
		print(f"  {key}) {desc}")
	choice = None
	try:
		choice = input("Enter choice [default 1]: ").strip()
	except EOFError:
		choice = ""
	if not choice:
		choice = "1"
	if choice not in options:
		print("Unrecognized choice; defaulting to option 1.")
		choice = "1"
	return options[choice][1]()


def parse_args():
	import argparse

	parser = argparse.ArgumentParser(description="Single-nurse routing and scheduling MILP (CBC).")
	parser.add_argument("--instance", choices=["small", "large"], help="Instance to solve (prompted if omitted).")
	return parser.parse_args()


def main():
	args = parse_args()
	instance = choose_instance(args.instance)
	solution = solve_single_nurse(instance)
	print_solution(instance, solution)
	chart_name = f"gantt_{_slugify(instance.name)}.png"
	plot_gantt(instance, solution, out_path=chart_name)
	print(f"Saved Gantt chart to {chart_name}")
	map_name = f"route_map_{_slugify(instance.name)}.png"
	plot_route_map(instance, solution, out_path=map_name)
	print(f"Saved route map to {map_name}")


if __name__ == "__main__":
	main()
