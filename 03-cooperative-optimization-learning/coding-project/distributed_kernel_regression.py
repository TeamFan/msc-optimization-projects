import argparse
import csv
import math
import os
import pickle
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rc("font", family="sans-serif", size=12)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Numerical tolerance used for residual-based sanity checks.
RESIDUAL_TOL = 1e-8


#Create output directory if it does not exist.
def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


#Save current Matplotlib figure as PDF and close it.
def save_figure(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()


#Resolve a dataset path relative to this script location.
def project_file_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "data" / filename


#Deterministic per-experiment seed offset (Python's built-in hash() of str is salted per process).
def stable_offset(tag: str) -> int:
    return zlib.crc32(tag.encode("utf-8")) % 100000


#Return a consistent curve style to keep multi-line plots readable.
def curve_style(index: int, marker_stride: int) -> Dict[str, object]:
    colors = list(plt.cm.tab10.colors)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    return {
        "color": colors[index % len(colors)],
        "linestyle": "-",
        "marker": markers[index % len(markers)],
        "markevery": max(1, marker_stride),
        "markersize": 4,
        "linewidth": 2.0,
        "alpha": 0.95,
    }


#Plot a styled log-log curve with value floor clipping.
def plot_loglog_curve(ax, x: np.ndarray, y: np.ndarray, label: str, index: int, floor: float = 1e-12) -> None:
    style = curve_style(index, marker_stride=max(1, len(x) // 14))
    ax.loglog(x, np.maximum(y, floor), label=label, **style)


#Plot a styled linear curve with consistent visibility settings.
def plot_linear_curve(ax, x: np.ndarray, y: np.ndarray, label: str, index: int) -> None:
    style = curve_style(index, marker_stride=max(1, len(x) // 16))
    ax.plot(x, y, label=label, **style)


#Compute compact convergence diagnostics from a metric curve.
def convergence_metrics(curve: np.ndarray, axis: Optional[np.ndarray] = None, floor: float = 1e-14) -> Dict[str, float]:
    vals = np.maximum(np.asarray(curve, dtype=float), floor)
    if vals.size < 2:
        return {
            "reduction_factor": float("nan"),
            "tail_ratio_median": float("nan"),
            "tail_log_slope": float("nan"),
        }
    x = np.asarray(axis, dtype=float) if axis is not None else np.arange(vals.size, dtype=float)
    if x.size != vals.size:
        x = np.arange(vals.size, dtype=float)
    reduction = float(vals[0] / vals[-1])
    ratios = vals[1:] / vals[:-1]
    tail_len = min(max(5, vals.size // 8), ratios.size)
    tail_ratio = float(np.median(ratios[-tail_len:]))
    start = max(1, vals.size - (tail_len + 1))
    x_tail = x[start:]
    y_tail = np.log(vals[start:])
    slope = float(np.polyfit(x_tail, y_tail, 1)[0]) if x_tail.size >= 2 and float(np.ptp(x_tail)) > 0.0 else float("nan")
    return {
        "reduction_factor": reduction,
        "tail_ratio_median": tail_ratio,
        "tail_log_slope": slope,
    }


#Attach convergence diagnostics to a metrics row.
def make_metrics_row(base: Dict[str, object], curve: Optional[np.ndarray] = None, axis: Optional[np.ndarray] = None) -> Dict[str, object]:
    row = dict(base)
    if curve is not None:
        row.update(convergence_metrics(curve, axis=axis))
    return row


#Check whether index blocks are sequential, disjoint, and balanced.
def partition_checks_from_index_blocks(index_blocks: List[np.ndarray], n_points: int, n_agents: int) -> Dict[str, object]:
    concat_idx = np.concatenate(index_blocks) if index_blocks else np.array([], dtype=int)
    counts = [int(len(block)) for block in index_blocks]
    unique_count = int(np.unique(concat_idx).size) if concat_idx.size > 0 else 0
    expected_floor = int(n_points // n_agents) if n_agents > 0 else 0
    expected_ceil = int(math.ceil(n_points / float(n_agents))) if n_agents > 0 else 0
    return {
        "policy": "sequential_chunking",
        "sequential": bool(np.array_equal(concat_idx, np.arange(n_points, dtype=int))),
        "disjoint_cover": bool(unique_count == n_points and concat_idx.size == n_points),
        "counts_balanced": bool(all(expected_floor <= c <= expected_ceil for c in counts)),
        "counts": counts,
    }


#RBF kernel matrix K[i,j] = exp(-(x1[i]-x2[j])^2).
def kernel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    d = x1[:, None] - x2[None, :]
    return np.exp(-(d ** 2))


#Build local quadratic Hessian/linear terms for one agent objective.
def local_quadratic_terms(
    x_local: np.ndarray,
    y_local: np.ndarray,
    x_m: np.ndarray,
    kmm: np.ndarray,
    sigma: float,
    nu: float,
    n_agents: int,
) -> Tuple[np.ndarray, np.ndarray]:
    m_i = kernel(x_local, x_m)
    h_i = (sigma ** 2 / n_agents) * kmm + m_i.T @ m_i + (nu / n_agents) * np.eye(len(x_m))
    b_i = m_i.T @ y_local
    return h_i, b_i


#Evaluate full regularized objective value at alpha.
def objective_value(
    alpha: np.ndarray,
    kmm: np.ndarray,
    m_blocks: List[np.ndarray],
    y_blocks: List[np.ndarray],
    sigma: float,
    nu: float,
) -> float:
    reg = 0.5 * (sigma ** 2) * alpha @ kmm @ alpha + 0.5 * nu * alpha @ alpha
    data_fit = 0.0
    for m_i, y_i in zip(m_blocks, y_blocks):
        r = y_i - m_i @ alpha
        data_fit += 0.5 * (r @ r)
    return float(reg + data_fit)


#Aggregate local quadratic terms into global linear system A alpha = b.
def compute_global_system(
    h_list: List[np.ndarray], b_list: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.sum(h_list, axis=0)
    b = np.sum(b_list, axis=0)
    return a, b


#Solve centralized reference model and return residual for validation.
def solve_centralized(
    h_list: List[np.ndarray], b_list: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = compute_global_system(h_list, b_list)
    alpha_star = np.linalg.solve(a, b)
    residual = a @ alpha_star - b
    return alpha_star, a, residual


#Return undirected line graph edges on n nodes.
def build_line_graph(n: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


#Return undirected complete graph edges on n nodes.
def build_complete_graph(n: int) -> List[Tuple[int, int]]:
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return edges


#Check graph connectivity via DFS.
def is_connected(n: int, edges: List[Tuple[int, int]]) -> bool:
    if n == 0:
        return True
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    seen = set([0])
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


#Generate a small-world style graph with rewiring and connectivity check.
def build_small_world_graph(
    n: int, k: int, p: float, rng: np.random.Generator, max_trials: int = 30
) -> List[Tuple[int, int]]:
    if k % 2 != 0:
        raise ValueError("k must be even for Watts-Strogatz style construction.")
    if k >= n:
        raise ValueError("k must be less than n.")

    for _ in range(max_trials):
        edges = set()
        half = k // 2
        for i in range(n):
            for d in range(1, half + 1):
                j = (i + d) % n
                edges.add(tuple(sorted((i, j))))

        rewired = set(edges)
        for e in list(edges):
            if rng.random() < p:
                i, j = e
                candidates = [c for c in range(n) if c != i and tuple(sorted((i, c))) not in rewired]
                if candidates:
                    new_j = int(rng.choice(candidates))
                    rewired.discard(e)
                    rewired.add(tuple(sorted((i, new_j))))

        edge_list = sorted(list(rewired))
        if is_connected(n, edge_list):
            return edge_list

    return sorted([(i, (i + 1) % n) for i in range(n)])


#Convert edge list to symmetric adjacency matrix.
def adjacency_from_edges(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    a = np.zeros((n, n), dtype=int)
    for i, j in edges:
        a[i, j] = 1
        a[j, i] = 1
    return a


#Build lazy Metropolis doubly-stochastic mixing matrix.
def lazy_metropolis_weights(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    deg = adj.sum(axis=1).astype(float)
    w = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                w[i, j] = 1.0 / (2.0 * max(deg[i], deg[j]))
        w[i, i] = 1.0 - np.sum(w[i, :])
    return w


#Build oriented incidence matrix for edge-constrained consensus.
def incidence_matrix(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    e = len(edges)
    b = np.zeros((e, n), dtype=float)
    for idx, (i, j) in enumerate(edges):
        b[idx, i] = 1.0
        b[idx, j] = -1.0
    return b


#Random model initialization for all agents/clients.
def default_init(n_agents: int, m: int, rng: np.random.Generator, scale: float = 0.01) -> np.ndarray:
    return rng.normal(0.0, scale, size=(n_agents, m))


#Gradient of one local quadratic model.
def part1_grad(x_i: np.ndarray, h_i: np.ndarray, b_i: np.ndarray) -> np.ndarray:
    return h_i @ x_i - b_i


#Run decentralized gradient descent and track mean optimality gap.
def run_dgd(
    h_list: List[np.ndarray],
    b_list: List[np.ndarray],
    alpha_star: np.ndarray,
    w: np.ndarray,
    eta: float,
    iters: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    n_agents = len(h_list)
    m = h_list[0].shape[0]
    x = default_init(n_agents, m, rng)
    gaps = np.zeros(iters + 1)
    gaps[0] = np.mean(np.linalg.norm(x - alpha_star, axis=1))
    for k in range(1, iters + 1):
        g = np.vstack([part1_grad(x[i], h_list[i], b_list[i]) for i in range(n_agents)])
        x = w @ x - eta * g
        gaps[k] = np.mean(np.linalg.norm(x - alpha_star, axis=1))
    return {"gaps": gaps, "x_final": x}


#Run DGD and track consensus-to-mean error (fallback large-n metric).
def run_dgd_consensus_metric(
    h_list: List[np.ndarray],
    b_list: List[np.ndarray],
    w: np.ndarray,
    eta: float,
    iters: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    n_agents = len(h_list)
    m = h_list[0].shape[0]
    x = default_init(n_agents, m, rng)
    metrics = np.zeros(iters + 1)
    x_mean = np.mean(x, axis=0)
    metrics[0] = np.mean(np.linalg.norm(x - x_mean, axis=1))
    for k in range(1, iters + 1):
        g = np.vstack([part1_grad(x[i], h_list[i], b_list[i]) for i in range(n_agents)])
        x = w @ x - eta * g
        x_mean = np.mean(x, axis=0)
        metrics[k] = np.mean(np.linalg.norm(x - x_mean, axis=1))
    return {"metrics": metrics, "x_final": x}


#Run gradient tracking with fixed stepsize and return gap curve.
def run_gradient_tracking(
    h_list: List[np.ndarray],
    b_list: List[np.ndarray],
    alpha_star: np.ndarray,
    w: np.ndarray,
    eta: float,
    iters: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    n_agents = len(h_list)
    m = h_list[0].shape[0]
    x = default_init(n_agents, m, rng)
    g = np.vstack([part1_grad(x[i], h_list[i], b_list[i]) for i in range(n_agents)])
    gaps = np.zeros(iters + 1)
    gaps[0] = np.mean(np.linalg.norm(x - alpha_star, axis=1))
    for k in range(1, iters + 1):
        x_next = w @ x - eta * g
        g_next = w @ g
        for i in range(n_agents):
            g_next[i] += part1_grad(x_next[i], h_list[i], b_list[i]) - part1_grad(x[i], h_list[i], b_list[i])
        x, g = x_next, g_next
        gaps[k] = np.mean(np.linalg.norm(x - alpha_star, axis=1))
    return {"gaps": gaps, "x_final": x}


#Run peer-to-peer dual decomposition on edge dual variables.
def run_dual_decomposition(
    h_list: List[np.ndarray],
    b_list: List[np.ndarray],
    alpha_star: np.ndarray,
    edges: List[Tuple[int, int]],
    eta_dual: float,
    iters: int,
) -> Dict[str, np.ndarray]:
    n_agents = len(h_list)
    m = h_list[0].shape[0]
    n_edges = len(edges)
    inv_h = [np.linalg.inv(h_i) for h_i in h_list]
    lam = np.zeros((n_edges, m))
    gaps = np.zeros(iters + 1)

    x = np.zeros((n_agents, m))
    for i in range(n_agents):
        x[i] = inv_h[i] @ b_list[i]
    gaps[0] = np.mean(np.linalg.norm(x - alpha_star, axis=1))

    for k in range(1, iters + 1):
        s = np.zeros((n_agents, m))
        for e_idx, (i, j) in enumerate(edges):
            s[i] += lam[e_idx]
            s[j] -= lam[e_idx]
        for i in range(n_agents):
            x[i] = inv_h[i] @ (b_list[i] - s[i])
        for e_idx, (i, j) in enumerate(edges):
            lam[e_idx] += eta_dual * (x[i] - x[j])
        gaps[k] = np.mean(np.linalg.norm(x - alpha_star, axis=1))

    return {"gaps": gaps, "x_final": x}

#Run peer-to-peer ADMM and track mean optimality gap.
def run_admm(
    h_list: List[np.ndarray],
    b_list: List[np.ndarray],
    alpha_star: np.ndarray,
    edges: List[Tuple[int, int]],
    beta: float,
    iters: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    n_agents = len(h_list)
    m = h_list[0].shape[0]
    edge_index = {tuple(sorted(e)): idx for idx, e in enumerate(edges)}
    neighbors = [[] for _ in range(n_agents)]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)

    # Per-agent linear systems are constant across iterations, so precompute inverses once.
    eye_m = np.eye(m)
    inv_mats = []
    for i in range(n_agents):
        deg_i = len(neighbors[i])
        mat_i = h_list[i] + beta * deg_i * eye_m
        inv_mats.append(np.linalg.inv(mat_i))

    x = default_init(n_agents, m, rng)
    y = rng.normal(0.0, 0.01, size=(len(edges), m))
    lam = np.zeros((n_agents, n_agents, m))
    gaps = np.zeros(iters + 1)
    gaps[0] = np.mean(np.linalg.norm(x - alpha_star, axis=1))

    for k in range(1, iters + 1):
        x_next = np.zeros_like(x)
        for i in range(n_agents):
            rhs = b_list[i].copy()
            for j in neighbors[i]:
                e_idx = edge_index[tuple(sorted((i, j)))]
                rhs += beta * y[e_idx] - lam[i, j]
            x_next[i] = inv_mats[i] @ rhs
        x = x_next

        for (i, j), e_idx in edge_index.items():
            y[e_idx] = 0.5 * (x[i] + x[j]) + (lam[i, j] + lam[j, i]) / (2.0 * beta)

        for (i, j), e_idx in edge_index.items():
            lam[i, j] += beta * (x[i] - y[e_idx])
            lam[j, i] += beta * (x[j] - y[e_idx])

        gaps[k] = np.mean(np.linalg.norm(x - alpha_star, axis=1))

    return {"gaps": gaps, "x_final": x}


#Run DGD under directed, lossy, and asynchronous communication.
def run_dgd_directed_lossy_async(
    h_list: List[np.ndarray],
    b_list: List[np.ndarray],
    alpha_star: np.ndarray,
    w_row: np.ndarray,
    eta: float,
    iters: int,
    p_loss: float,
    p_active: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    n_agents = len(h_list)
    m = h_list[0].shape[0]
    x = default_init(n_agents, m, rng)
    gaps = np.zeros(iters + 1)
    gaps[0] = np.mean(np.linalg.norm(x - alpha_star, axis=1))

    for k in range(1, iters + 1):
        x_new = x.copy()
        for i in range(n_agents):
            if rng.random() > p_active:
                continue
            row = w_row[i].copy()
            for j in range(n_agents):
                if i == j or row[j] == 0.0:
                    continue
                if rng.random() < p_loss:
                    row[j] = 0.0
            row_sum = row.sum()
            if row_sum <= 0.0:
                row[i] = 1.0
                row_sum = 1.0
            row /= row_sum
            mixed = row @ x
            g = part1_grad(x[i], h_list[i], b_list[i])
            x_new[i] = mixed - eta * g
        x = x_new
        gaps[k] = np.mean(np.linalg.norm(x - alpha_star, axis=1))
    return {"gaps": gaps, "x_final": x}


#Construct a simple directed column-stochastic mixing matrix.
def build_directed_col_stochastic(n_agents: int) -> np.ndarray:
    out_neighbors = {}
    for j in range(n_agents):
        out_neighbors[j] = [j, (j + 1) % n_agents]
    c = np.zeros((n_agents, n_agents), dtype=float)
    for j in range(n_agents):
        outs = out_neighbors[j]
        val = 1.0 / len(outs)
        for i in outs:
            c[i, j] += val
    return c


#Run a push-sum corrected gradient method for directed networks.
def run_push_sum_gradient(
    h_list: List[np.ndarray],
    b_list: List[np.ndarray],
    alpha_star: np.ndarray,
    c_col: np.ndarray,
    eta: float,
    iters: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    n_agents = len(h_list)
    m = h_list[0].shape[0]
    z = default_init(n_agents, m, rng)
    w = np.ones(n_agents)
    x = z / w[:, None]
    gaps = np.zeros(iters + 1)
    gaps[0] = np.mean(np.linalg.norm(x - alpha_star, axis=1))

    for k in range(1, iters + 1):
        grads = np.vstack([part1_grad(x[i], h_list[i], b_list[i]) for i in range(n_agents)])
        z = c_col @ z - eta * grads
        w = c_col @ w
        x = z / w[:, None]
        gaps[k] = np.mean(np.linalg.norm(x - alpha_star, axis=1))
    return {"gaps": gaps, "x_final": x}


#Mini-batch gradient of local federated objective with size correction.
def local_grad_batch(
    alpha: np.ndarray,
    x_local: np.ndarray,
    y_local: np.ndarray,
    x_m: np.ndarray,
    kmm: np.ndarray,
    sigma: float,
    nu: float,
    n_clients: int,
    batch_idx: np.ndarray,
) -> np.ndarray:
    m_batch = kernel(x_local[batch_idx], x_m)
    yb = y_local[batch_idx]
    batch_size = len(batch_idx)
    n_loc = len(x_local)
    scale = n_loc / float(batch_size)
    grad_data = scale * (m_batch.T @ (m_batch @ alpha - yb))
    grad_reg = (sigma ** 2 / n_clients) * (kmm @ alpha) + (nu / n_clients) * alpha
    return grad_data + grad_reg


#Run FedAvg with configurable batch/epochs/participation schedule.
def run_fedavg(
    x_clients: List[np.ndarray],
    y_clients: List[np.ndarray],
    x_m: np.ndarray,
    kmm: np.ndarray,
    sigma: float,
    nu: float,
    alpha_star: np.ndarray,
    rounds: int,
    batch_size: int,
    clients_per_round: int,
    epochs: int,
    eta0: float,
    diminishing: bool,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    c_total = len(x_clients)
    m = len(x_m)
    alpha = np.zeros(m)
    gaps = np.zeros(rounds + 1)
    steps = np.zeros(rounds + 1)

    m_blocks = [kernel(x_clients[i], x_m) for i in range(c_total)]
    f_star = objective_value(alpha_star, kmm, m_blocks, y_clients, sigma, nu)
    gaps[0] = objective_value(alpha, kmm, m_blocks, y_clients, sigma, nu) - f_star
    steps[0] = 0.0

    for t in range(1, rounds + 1):
        eta_t = eta0 / (1.0 + 0.01 * t) if diminishing else eta0
        selected = rng.choice(c_total, size=clients_per_round, replace=False)
        local_models = []
        local_sizes = []

        for cid in selected:
            a_local = alpha.copy()
            x_loc = x_clients[cid]
            y_loc = y_clients[cid]
            n_loc = len(x_loc)

            for _ in range(epochs):
                if batch_size >= n_loc:
                    idx = np.arange(n_loc)
                else:
                    idx = rng.choice(n_loc, size=batch_size, replace=False)
                g = local_grad_batch(
                    a_local, x_loc, y_loc, x_m, kmm, sigma, nu, c_total, idx
                )
                a_local = a_local - eta_t * g

            local_models.append(a_local)
            local_sizes.append(n_loc)

        local_models = np.asarray(local_models)
        local_sizes = np.asarray(local_sizes, dtype=float)
        alpha = np.sum(local_models * local_sizes[:, None], axis=0) / np.sum(local_sizes)
        gaps[t] = objective_value(alpha, kmm, m_blocks, y_clients, sigma, nu) - f_star
        steps[t] = t * epochs

    return {"gaps": gaps, "steps": steps, "alpha_final": alpha}

#Run SCAFFOLD with client and server control variates.
def run_scaffold(
    x_clients: List[np.ndarray],
    y_clients: List[np.ndarray],
    x_m: np.ndarray,
    kmm: np.ndarray,
    sigma: float,
    nu: float,
    alpha_star: np.ndarray,
    rounds: int,
    batch_size: int,
    clients_per_round: int,
    epochs: int,
    eta: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    c_total = len(x_clients)
    m = len(x_m)
    x_global = np.zeros(m)
    c_global = np.zeros(m)
    c_clients = np.zeros((c_total, m))

    m_blocks = [kernel(x_clients[i], x_m) for i in range(c_total)]
    f_star = objective_value(alpha_star, kmm, m_blocks, y_clients, sigma, nu)
    gaps = np.zeros(rounds + 1)
    steps = np.zeros(rounds + 1)
    gaps[0] = objective_value(x_global, kmm, m_blocks, y_clients, sigma, nu) - f_star

    for t in range(1, rounds + 1):
        selected = rng.choice(c_total, size=clients_per_round, replace=False)
        deltas_x = []
        deltas_c = []

        for cid in selected:
            x_loc = x_global.copy()
            c_i = c_clients[cid].copy()
            x_data = x_clients[cid]
            y_data = y_clients[cid]
            n_loc = len(x_data)

            for _ in range(epochs):
                if batch_size >= n_loc:
                    idx = np.arange(n_loc)
                else:
                    idx = rng.choice(n_loc, size=batch_size, replace=False)
                g = local_grad_batch(
                    x_loc, x_data, y_data, x_m, kmm, sigma, nu, c_total, idx
                )
                x_loc = x_loc - eta * (g - c_i + c_global)

            c_i_new = c_i - c_global + (x_global - x_loc) / (epochs * eta)
            deltas_x.append(x_loc - x_global)
            deltas_c.append(c_i_new - c_i)
            c_clients[cid] = c_i_new

        delta_x_mean = np.mean(np.asarray(deltas_x), axis=0)
        x_global = x_global + delta_x_mean
        c_global = c_global + (clients_per_round / float(c_total)) * np.mean(np.asarray(deltas_c), axis=0)

        gaps[t] = objective_value(x_global, kmm, m_blocks, y_clients, sigma, nu) - f_star
        steps[t] = t * epochs

    return {"gaps": gaps, "steps": steps, "alpha_final": x_global}


#Run differentially private DGD with Laplacian message noise.
def run_dp_dgd(
    h_list: List[np.ndarray],
    b_list: List[np.ndarray],
    alpha_star: np.ndarray,
    w: np.ndarray,
    iters: int,
    eps: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    n_agents = len(h_list)
    m = h_list[0].shape[0]
    x = default_init(n_agents, m, rng)
    gaps = np.zeros(iters + 1)
    gaps[0] = np.mean(np.linalg.norm(x - alpha_star, axis=1))
    c_emp = 0.0
    sum_alpha_over_nu = 0.0
    sum_alpha2_over_gamma = 0.0
    sum_gamma2_nu = 0.0
    prev_alpha = None
    prev_gamma = None
    prev_nu = None
    alpha_decreasing = True
    gamma_decreasing = True
    nu_increasing = True

    for k in range(1, iters + 1):
        alpha_k = 0.002 / (1.0 + 0.001 * k)
        gamma_k = 1.0 / (1.0 + 0.001 * (k ** 0.9))
        nu_k = (0.01 / eps) * (1.0 + 0.001 * (k ** 0.1))
        b_scale = math.sqrt(nu_k / 2.0)
        sum_alpha_over_nu += alpha_k / nu_k
        sum_alpha2_over_gamma += (alpha_k ** 2) / gamma_k
        sum_gamma2_nu += (gamma_k ** 2) * nu_k
        if prev_alpha is not None:
            alpha_decreasing = alpha_decreasing and bool(alpha_k < prev_alpha)
            gamma_decreasing = gamma_decreasing and bool(gamma_k < prev_gamma)
            nu_increasing = nu_increasing and bool(nu_k > prev_nu)
        prev_alpha = alpha_k
        prev_gamma = gamma_k
        prev_nu = nu_k

        noise = rng.laplace(loc=0.0, scale=b_scale, size=(n_agents, m))
        chi = x + noise
        mixed = w @ chi
        x_new = np.zeros_like(x)
        for i in range(n_agents):
            g = part1_grad(x[i], h_list[i], b_list[i])
            c_emp = max(c_emp, float(np.linalg.norm(g, ord=1)))
            x_new[i] = x[i] + gamma_k * (mixed[i] - x[i] - alpha_k * g)
        x = x_new
        gaps[k] = np.mean(np.linalg.norm(x - alpha_star, axis=1))
    epsilon_bar_emp = 2.0 * c_emp * sum_alpha_over_nu
    return {
        "gaps": gaps,
        "x_final": x,
        "c_emp": c_emp,
        "epsilon_bar_emp": epsilon_bar_emp,
        "sum_alpha_over_nu": sum_alpha_over_nu,
        "sum_alpha2_over_gamma": sum_alpha2_over_gamma,
        "sum_gamma2_nu": sum_gamma2_nu,
        "alpha_decreasing": bool(alpha_decreasing),
        "gamma_decreasing": bool(gamma_decreasing),
        "nu_increasing": bool(nu_increasing),
    }


#Non-private counterpart of DP-DGD for reference comparison.
def run_non_private_reference_dgd(
    h_list: List[np.ndarray],
    b_list: List[np.ndarray],
    alpha_star: np.ndarray,
    w: np.ndarray,
    iters: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    n_agents = len(h_list)
    m = h_list[0].shape[0]
    x = default_init(n_agents, m, rng)
    gaps = np.zeros(iters + 1)
    gaps[0] = np.mean(np.linalg.norm(x - alpha_star, axis=1))
    for k in range(1, iters + 1):
        alpha_k = 0.002 / (1.0 + 0.001 * k)
        gamma_k = 1.0 / (1.0 + 0.001 * (k ** 0.9))
        mixed = w @ x
        x_new = np.zeros_like(x)
        for i in range(n_agents):
            g = part1_grad(x[i], h_list[i], b_list[i])
            x_new[i] = x[i] + gamma_k * (mixed[i] - x[i] - alpha_k * g)
        x = x_new
        gaps[k] = np.mean(np.linalg.norm(x - alpha_star, axis=1))
    return {"gaps": gaps, "x_final": x}


#Compute Part III theorem-alignment diagnostics for one graph/model setting.
def part3_theorem_audit(
    h_list: List[np.ndarray],
    w_mix: np.ndarray,
    iters: int,
    eps: float,
) -> Dict[str, object]:
    n_agents = w_mix.shape[0]
    lmax = max(float(np.linalg.eigvalsh(h).max()) for h in h_list)
    symmetric = bool(np.allclose(w_mix, w_mix.T, atol=1e-12))
    row_sums = w_mix.sum(axis=1)
    col_sums = w_mix.sum(axis=0)
    doubly_stochastic = bool(
        np.allclose(row_sums, np.ones(n_agents), atol=1e-12)
        and np.allclose(col_sums, np.ones(n_agents), atol=1e-12)
    )
    w_paper = w_mix - np.eye(n_agents)
    w_paper_row_zero = bool(np.allclose(w_paper.sum(axis=1), np.zeros(n_agents), atol=1e-12))
    op_norm = float(
        np.linalg.norm(np.eye(n_agents) + w_paper - np.ones((n_agents, n_agents)) / n_agents, ord=2)
    )
    op_norm_lt_one = bool(op_norm < 1.0)
    eigvals = np.linalg.eigvals(w_mix)
    eig_abs_sorted = np.sort(np.abs(eigvals))[::-1]
    second_abs = float(eig_abs_sorted[1]) if len(eig_abs_sorted) > 1 else 0.0
    spectral_gap = float(1.0 - second_abs)

    k = np.arange(1, iters + 1, dtype=float)
    alpha = 0.002 / (1.0 + 0.001 * k)
    gamma = 1.0 / (1.0 + 0.001 * (k ** 0.9))
    nu = (0.01 / eps) * (1.0 + 0.001 * (k ** 0.1))
    sum_alpha = float(np.sum(alpha))
    sum_gamma = float(np.sum(gamma))
    sum_alpha2_over_gamma = float(np.sum((alpha ** 2) / gamma))
    sum_alpha_over_nu = float(np.sum(alpha / nu))
    sum_gamma2_nu = float(np.sum((gamma ** 2) * nu))

    return {
        "lmax": lmax,
        "spectral_gap": spectral_gap,
        "w_symmetric": symmetric,
        "w_doubly_stochastic": doubly_stochastic,
        "w_paper_row_zero": w_paper_row_zero,
        "i_plus_wpaper_minus_proj_norm": op_norm,
        "i_plus_wpaper_minus_proj_norm_lt_one": op_norm_lt_one,
        "sum_alpha_prefix": sum_alpha,
        "sum_gamma_prefix": sum_gamma,
        "sum_alpha2_over_gamma": sum_alpha2_over_gamma,
        "sum_alpha_over_nu": sum_alpha_over_nu,
        "sum_gamma2_nu": sum_gamma2_nu,
        "alpha_decreasing": bool(np.all(np.diff(alpha) < 0.0)),
        "gamma_decreasing": bool(np.all(np.diff(gamma) < 0.0)),
        "nu_increasing": bool(np.all(np.diff(nu) > 0.0)),
        "asymptotic_alpha_diverges": True,
        "asymptotic_gamma_diverges": True,
        "asymptotic_alpha2_over_gamma_converges": True,
        "asymptotic_alpha_over_nu_converges": True,
    }


#Build distributed quadratic model terms for an arbitrary (n,m,a) setting.
def build_distributed_kernel_setting(
    x_all: np.ndarray,
    y_all: np.ndarray,
    n_points: int,
    m_points: int,
    n_agents: int,
    sigma: float,
    nu: float,
    rng: np.random.Generator,
) -> Dict[str, object]:
    x_n = np.asarray(x_all[:n_points], dtype=float)
    y_n = np.asarray(y_all[:n_points], dtype=float)
    selected = rng.choice(n_points, size=m_points, replace=False)
    x_m = x_n[selected]
    kmm = kernel(x_m, x_m)

    index_blocks = [np.asarray(v, dtype=int) for v in np.array_split(np.arange(n_points), n_agents)]
    x_blocks = [x_n[idx] for idx in index_blocks]
    y_blocks = [y_n[idx] for idx in index_blocks]
    partition_meta = partition_checks_from_index_blocks(index_blocks, n_points=n_points, n_agents=n_agents)

    h_list: List[np.ndarray] = []
    b_list: List[np.ndarray] = []
    for i in range(n_agents):
        h_i, b_i = local_quadratic_terms(
            x_blocks[i],
            y_blocks[i],
            x_m,
            kmm,
            sigma,
            nu,
            n_agents,
        )
        h_list.append(h_i)
        b_list.append(b_i)

    alpha_star, _, residual = solve_centralized(h_list, b_list)
    return {
        "x_n": x_n,
        "y_n": y_n,
        "x_m": x_m,
        "kmm": kmm,
        "x_blocks": x_blocks,
        "y_blocks": y_blocks,
        "h_list": h_list,
        "b_list": b_list,
        "alpha_star": alpha_star,
        "residual_norm": float(np.linalg.norm(residual)),
        "partition_meta": partition_meta,
    }


#Write flat experiment metrics table to CSV.
def write_metrics_csv(path: str, rows: List[Dict[str, object]]) -> None:
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


#Parse CLI options for part selection and runtime mode.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cooperative Kernel Regression project runner.")
    p.add_argument("--part", choices=["part1", "part2", "part3", "all"], default="all")
    p.add_argument("--mode", choices=["quick", "full"], default="full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument(
        "--debug_output",
        action="store_true",
        help="Print intermediate validation/analysis summary to console.",
    )
    return p.parse_args()

#Execute requested project parts and export figures/metrics artifacts.
def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    # Global run identity and reproducibility controls.
    seed = int(args.seed)
    rng = np.random.default_rng(seed)

    # Shared model/data constants for all parts.
    sigma = 0.5
    nu = 1.0
    n_agents = 5
    n_base = 100
    m_base = 10

    # Runtime budget presets.
    if args.mode == "quick":
        iters_p1 = 800
        rounds_p2 = 60
        iters_p3 = 1000
    else:
        iters_p1 = 3000
        rounds_p2 = 150
        iters_p3 = 3000

    metrics_rows: List[Dict[str, object]] = []
    summary_lines: List[str] = []
    figure_paths: List[str] = []

    # Load both project datasets.
    with open(project_file_path("first_database.pkl"), "rb") as f:
        x_big, y_big = pickle.load(f)
    x_big = np.asarray(x_big, dtype=float)
    y_big = np.asarray(y_big, dtype=float)

    with open(project_file_path("second_database.pkl"), "rb") as f:
        x_clients_raw, y_clients_raw = pickle.load(f)
    x_clients = [np.asarray(v, dtype=float) for v in x_clients_raw]
    y_clients = [np.asarray(v, dtype=float) for v in y_clients_raw]

    # Baseline Part I subset and Nystrom points.
    x_n = x_big[:n_base]
    y_n = y_big[:n_base]
    indices = rng.choice(n_base, size=m_base, replace=False)
    x_m = x_n[indices]
    kmm = kernel(x_m, x_m)

    block_size = n_base // n_agents
    x_blocks = [x_n[i * block_size : (i + 1) * block_size] for i in range(n_agents)]
    y_blocks = [y_n[i * block_size : (i + 1) * block_size] for i in range(n_agents)]
    m_blocks = [kernel(x_blocks[i], x_m) for i in range(n_agents)]
    idx_blocks_p1 = [np.arange(i * block_size, (i + 1) * block_size, dtype=int) for i in range(n_agents)]
    part1_partition_meta = partition_checks_from_index_blocks(idx_blocks_p1, n_points=n_base, n_agents=n_agents)

    # Build local quadratic models for each agent.
    h_list = []
    b_list = []
    for i in range(n_agents):
        h_i, b_i = local_quadratic_terms(x_blocks[i], y_blocks[i], x_m, kmm, sigma, nu, n_agents)
        h_list.append(h_i)
        b_list.append(b_i)

    # Centralized reference and smooth/strong-convex constants.
    alpha_star, a_global, residual = solve_centralized(h_list, b_list)
    central_residual_norm = float(np.linalg.norm(residual))
    lmax = max(float(np.linalg.eigvalsh(h).max()) for h in h_list)
    mmin = min(float(np.linalg.eigvalsh(h).min()) for h in h_list)

    # Basic numerical sanity checks.
    checks = {}
    checks["kernel_matrix_is_symmetric"] = bool(np.allclose(kmm, kmm.T, atol=1e-12))
    checks["kernel_matrix_has_unit_diagonal"] = bool(np.allclose(np.diag(kmm), np.ones(m_base), atol=1e-12))
    checks["kernel_cross_matrix_is_finite"] = bool(np.all(np.isfinite(np.vstack(m_blocks))))
    checks["central_system_residual_le_tol"] = bool(central_residual_norm <= RESIDUAL_TOL)
    checks["part1_partition_policy_sequential_chunking"] = bool(part1_partition_meta["sequential"])
    checks["part1_partition_disjoint_cover"] = bool(part1_partition_meta["disjoint_cover"])
    checks["part1_partition_counts_balanced"] = bool(part1_partition_meta["counts_balanced"])
    checks["part1_partition_counts_eq_20"] = bool(all(c == 20 for c in part1_partition_meta["counts"]))

    if args.part in ("part1", "all"):
        line_edges = build_line_graph(n_agents)
        sw_edges = build_small_world_graph(n_agents, k=2, p=0.4, rng=rng)
        full_edges = build_complete_graph(n_agents)

        line_adj = adjacency_from_edges(n_agents, line_edges)
        sw_adj = adjacency_from_edges(n_agents, sw_edges)
        full_adj = adjacency_from_edges(n_agents, full_edges)

        w_line = lazy_metropolis_weights(line_adj)
        w_sw = lazy_metropolis_weights(sw_adj)
        w_full = lazy_metropolis_weights(full_adj)

        b_inc_line = incidence_matrix(n_agents, line_edges)
        sigma_max_a = float(np.linalg.svd(b_inc_line, compute_uv=False).max())
        eta_dgd = 0.2 / lmax
        eta_gt = 0.6 / lmax
        eta_dual = 0.9 * (2.0 * mmin / (sigma_max_a ** 2))

        plt.figure(figsize=(6.5, 4.2))
        plt.plot(x_n, y_n, "o", markersize=4, alpha=0.85, label="Noisy data (n=100)")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.title("Part I Data Preview (n=100)")
        plt.grid(True, alpha=0.35)
        plt.legend()
        p = os.path.join(args.outdir, "part1_data_preview.pdf")
        save_figure(p)
        figure_paths.append(p)

        algo_results = {}
        for algo_name, fn, kwargs in [
            ("DGD", run_dgd, {"w": w_line, "eta": eta_dgd, "iters": iters_p1}),
            ("GT", run_gradient_tracking, {"w": w_line, "eta": eta_gt, "iters": iters_p1}),
            (
                "DualDecomposition",
                run_dual_decomposition,
                {"edges": line_edges, "eta_dual": eta_dual, "iters": iters_p1},
            ),
            ("ADMM_beta1", run_admm, {"edges": line_edges, "beta": 1.0, "iters": iters_p1}),
        ]:
            t0 = time.perf_counter()
            if fn in (run_dgd, run_gradient_tracking, run_admm):
                res = fn(
                    h_list=h_list,
                    b_list=b_list,
                    alpha_star=alpha_star,
                    rng=np.random.default_rng(seed + stable_offset(algo_name)),
                    **kwargs,
                )
            else:
                res = fn(
                    h_list=h_list,
                    b_list=b_list,
                    alpha_star=alpha_star,
                    **kwargs,
                )
            dt_s = time.perf_counter() - t0
            algo_results[algo_name] = res
            metrics_rows.append(
                make_metrics_row(
                    {
                        "part": "part1",
                        "algorithm": algo_name,
                        "graph": "line",
                        "iters": iters_p1,
                        "final_gap": float(res["gaps"][-1]),
                        "min_gap": float(np.min(res["gaps"])),
                        "runtime_s": dt_s,
                        "eta": eta_dgd if algo_name == "DGD" else eta_gt if algo_name == "GT" else eta_dual if algo_name == "DualDecomposition" else "",
                        "beta": 1.0 if algo_name.startswith("ADMM") else "",
                        "seed": seed,
                    },
                    curve=res["gaps"],
                )
            )

        plt.figure(figsize=(7.2, 4.6))
        x_axis = np.arange(iters_p1 + 1)
        ax = plt.gca()
        for idx, algo_name in enumerate(["DGD", "GT", "DualDecomposition", "ADMM_beta1"]):
            plot_loglog_curve(ax, x_axis, algo_results[algo_name]["gaps"], algo_name, idx, floor=1e-12)
        plt.xlabel("Iterations")
        plt.ylabel(r"Mean optimality gap $||\alpha_i^k-\alpha^*||$")
        plt.title("Part I: Convergence on Line Graph (n=100, m=10, N=5)")
        plt.grid(True, which="both", alpha=0.35)
        plt.legend(fontsize=9, framealpha=0.9)
        p = os.path.join(args.outdir, "part1_algorithms_line_loglog.pdf")
        save_figure(p)
        figure_paths.append(p)

        beta_sweep = [0.1, 1.0, 10.0, 100.0]
        plt.figure(figsize=(7.2, 4.6))
        ax = plt.gca()
        for idx, beta in enumerate(beta_sweep):
            t0 = time.perf_counter()
            res = run_admm(
                h_list=h_list,
                b_list=b_list,
                alpha_star=alpha_star,
                edges=line_edges,
                beta=beta,
                iters=iters_p1,
                rng=np.random.default_rng(seed + int(beta * 1000)),
            )
            dt_s = time.perf_counter() - t0
            plot_loglog_curve(
                ax,
                np.arange(iters_p1 + 1),
                res["gaps"],
                f"ADMM beta={beta:g}",
                idx,
                floor=1e-12,
            )
            metrics_rows.append(
                make_metrics_row(
                    {
                        "part": "part1",
                        "algorithm": "ADMM",
                        "graph": "line",
                        "iters": iters_p1,
                        "final_gap": float(res["gaps"][-1]),
                        "min_gap": float(np.min(res["gaps"])),
                        "runtime_s": dt_s,
                        "eta": "",
                        "beta": beta,
                        "seed": seed,
                    },
                    curve=res["gaps"],
                )
            )
        plt.xlabel("Iterations")
        plt.ylabel(r"Mean optimality gap $||\alpha_i^k-\alpha^*||$")
        plt.title("Part I: ADMM Penalty Sweep on Line Graph")
        plt.grid(True, which="both", alpha=0.35)
        plt.legend(fontsize=9, framealpha=0.9)
        p = os.path.join(args.outdir, "part1_admm_beta_sweep.pdf")
        save_figure(p)
        figure_paths.append(p)

        graph_defs = [("line", line_edges, w_line), ("small_world", sw_edges, w_sw), ("complete", full_edges, w_full)]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for idx, (name, edges, w) in enumerate(graph_defs):
            res_dgd = run_dgd(
                h_list=h_list,
                b_list=b_list,
                alpha_star=alpha_star,
                w=w,
                eta=eta_dgd,
                iters=iters_p1,
                rng=np.random.default_rng(seed + stable_offset("dgd_" + name)),
            )
            res_gt = run_gradient_tracking(
                h_list=h_list,
                b_list=b_list,
                alpha_star=alpha_star,
                w=w,
                eta=eta_gt,
                iters=iters_p1,
                rng=np.random.default_rng(seed + stable_offset("gt_" + name)),
            )
            x_curve = np.arange(iters_p1 + 1)
            plot_loglog_curve(axes[0], x_curve, res_dgd["gaps"], name, idx, floor=1e-12)
            plot_loglog_curve(axes[1], x_curve, res_gt["gaps"], name, idx, floor=1e-12)

            metrics_rows.append(
                make_metrics_row(
                    {
                        "part": "part1",
                        "algorithm": "DGD",
                        "graph": name,
                        "iters": iters_p1,
                        "final_gap": float(res_dgd["gaps"][-1]),
                        "min_gap": float(np.min(res_dgd["gaps"])),
                        "runtime_s": "",
                        "eta": eta_dgd,
                        "beta": "",
                        "seed": seed,
                    },
                    curve=res_dgd["gaps"],
                )
            )
            metrics_rows.append(
                make_metrics_row(
                    {
                        "part": "part1",
                        "algorithm": "GT",
                        "graph": name,
                        "iters": iters_p1,
                        "final_gap": float(res_gt["gaps"][-1]),
                        "min_gap": float(np.min(res_gt["gaps"])),
                        "runtime_s": "",
                        "eta": eta_gt,
                        "beta": "",
                        "seed": seed,
                    },
                    curve=res_gt["gaps"],
                )
            )

        axes[0].set_title("DGD vs Graph Structure")
        axes[1].set_title("GT vs Graph Structure")
        for ax in axes:
            ax.set_xlabel("Iterations")
            ax.set_ylabel(r"Mean optimality gap")
            ax.grid(True, which="both", alpha=0.35)
            ax.legend(fontsize=9, framealpha=0.9)
        p = os.path.join(args.outdir, "part1_graph_structure_comparison.pdf")
        save_figure(p)
        figure_paths.append(p)

        x_prime = np.linspace(-1.0, 1.0, 250)
        y_star = kernel(x_prime, x_m) @ alpha_star
        alpha_dgd = np.mean(algo_results["DGD"]["x_final"], axis=0)
        alpha_gt = np.mean(algo_results["GT"]["x_final"], axis=0)
        alpha_dual = np.mean(algo_results["DualDecomposition"]["x_final"], axis=0)
        alpha_admm = np.mean(algo_results["ADMM_beta1"]["x_final"], axis=0)

        plt.figure(figsize=(7.2, 4.8))
        plt.plot(x_n, y_n, "o", markersize=3, alpha=0.5, label="Data (n=100)")
        plt.plot(x_prime, y_star, color="black", linewidth=2.5, label="Centralized")
        ax = plt.gca()
        plot_linear_curve(ax, x_prime, kernel(x_prime, x_m) @ alpha_dgd, "DGD", 0)
        plot_linear_curve(ax, x_prime, kernel(x_prime, x_m) @ alpha_gt, "GT", 1)
        plot_linear_curve(ax, x_prime, kernel(x_prime, x_m) @ alpha_dual, "Dual Decomp", 2)
        plot_linear_curve(ax, x_prime, kernel(x_prime, x_m) @ alpha_admm, "ADMM", 3)
        plt.xlabel(r"$x'$ on uniform grid (nt=250)")
        plt.ylabel("Predicted y")
        plt.title("Part I: Function Reconstruction")
        plt.grid(True, alpha=0.35)
        plt.legend(ncol=2, fontsize=9, framealpha=0.9)
        p = os.path.join(args.outdir, "part1_reconstruction_nt250.pdf")
        save_figure(p)
        figure_paths.append(p)

        w_directed_row = np.array(
            [
                [0.70, 0.30, 0.00, 0.00, 0.00],
                [0.00, 0.65, 0.35, 0.00, 0.00],
                [0.00, 0.00, 0.55, 0.45, 0.00],
                [0.15, 0.00, 0.00, 0.85, 0.00],
                [0.40, 0.00, 0.00, 0.00, 0.60],
            ],
            dtype=float,
        )
        c_directed_col = build_directed_col_stochastic(n_agents)

        res_directed_clean = run_dgd(
            h_list=h_list,
            b_list=b_list,
            alpha_star=alpha_star,
            w=w_directed_row,
            eta=eta_dgd,
            iters=iters_p1,
            rng=np.random.default_rng(seed + 3101),
        )
        res_failure = run_dgd_directed_lossy_async(
            h_list=h_list,
            b_list=b_list,
            alpha_star=alpha_star,
            w_row=w_directed_row,
            eta=eta_dgd,
            iters=iters_p1,
            p_loss=0.2,
            p_active=0.8,
            rng=np.random.default_rng(seed + 3102),
        )
        res_pushsum = run_push_sum_gradient(
            h_list=h_list,
            b_list=b_list,
            alpha_star=alpha_star,
            c_col=c_directed_col,
            eta=eta_gt,
            iters=iters_p1,
            rng=np.random.default_rng(seed + 3103),
        )
        res_line_dgd = algo_results["DGD"]

        plt.figure(figsize=(7.2, 4.8))
        it = np.arange(iters_p1 + 1)
        ax = plt.gca()
        plot_loglog_curve(ax, it, res_line_dgd["gaps"], "DGD undirected baseline", 0, floor=1e-12)
        plot_loglog_curve(ax, it, res_directed_clean["gaps"], "DGD directed only", 1, floor=1e-12)
        plot_loglog_curve(ax, it, res_failure["gaps"], "DGD directed + loss + async", 2, floor=1e-12)
        plot_loglog_curve(ax, it, res_pushsum["gaps"], "Push-sum (directed)", 3, floor=1e-12)
        plt.xlabel("Iterations")
        plt.ylabel(r"Mean optimality gap")
        plt.title("Part I: Failure Modes and Push-sum Recovery")
        plt.grid(True, which="both", alpha=0.35)
        plt.legend(fontsize=9, framealpha=0.9)
        p = os.path.join(args.outdir, "part1_failure_and_pushsum.pdf")
        save_figure(p)
        figure_paths.append(p)

        summary_lines.append("## Part I")
        summary_lines.append(
            f"- Data allocation policy: {part1_partition_meta['policy']} with counts={part1_partition_meta['counts']}."
        )
        summary_lines.append(
            f"- Step sizes: eta_dgd={eta_dgd:.3e}, eta_gt={eta_gt:.3e}, eta_dual={eta_dual:.3e}; Lmax={lmax:.3e}, mmin={mmin:.3e}."
        )
        summary_lines.append(
            f"- Final gaps (line graph): DGD={algo_results['DGD']['gaps'][-1]:.3e}, GT={algo_results['GT']['gaps'][-1]:.3e}, "
            f"Dual={algo_results['DualDecomposition']['gaps'][-1]:.3e}, ADMM(beta=1)={algo_results['ADMM_beta1']['gaps'][-1]:.3e}."
        )
        dgd_diag = convergence_metrics(algo_results["DGD"]["gaps"])
        gt_diag = convergence_metrics(algo_results["GT"]["gaps"])
        summary_lines.append(
            f"- Convergence diagnostics (line graph): DGD reduction={dgd_diag['reduction_factor']:.3e}, "
            f"GT reduction={gt_diag['reduction_factor']:.3e}, DGD tail-ratio={dgd_diag['tail_ratio_median']:.4f}, "
            f"GT tail-ratio={gt_diag['tail_ratio_median']:.4f}."
        )
        summary_lines.append(
            f"- Failure comparison final gaps: directed+loss+async DGD={res_failure['gaps'][-1]:.3e}, push-sum={res_pushsum['gaps'][-1]:.3e}."
        )

        # Bonus experiment: increase n while keeping m = ceil(sqrt(n)).
        if args.mode == "quick":
            n_candidates = [100, 300, 1000, 3000]
            iters_scale = 120
        else:
            n_candidates = [100, 300, 1000, 3000, 10000, 30000]
            iters_scale = 200

        bonus_runs = []
        for n_curr in n_candidates:
            m_curr = int(math.ceil(math.sqrt(n_curr)))
            n_agents_curr = min(100, max(5, n_curr // 20))
            x_curr = x_big[:n_curr]
            y_curr = y_big[:n_curr]

            rng_sel = np.random.default_rng(seed + 7000 + n_curr)
            sel = rng_sel.choice(n_curr, size=m_curr, replace=False)
            x_m_curr = x_curr[sel]
            kmm_curr = kernel(x_m_curr, x_m_curr)

            x_blocks_curr = np.array_split(x_curr, n_agents_curr)
            y_blocks_curr = np.array_split(y_curr, n_agents_curr)
            h_curr = []
            b_curr = []
            for i in range(n_agents_curr):
                h_i, b_i = local_quadratic_terms(
                    x_blocks_curr[i],
                    y_blocks_curr[i],
                    x_m_curr,
                    kmm_curr,
                    sigma,
                    nu,
                    n_agents_curr,
                )
                h_curr.append(h_i)
                b_curr.append(b_i)

            # Try centralized reference; if unavailable, fall back to consensus metric.
            alpha_ref = None
            use_consensus_metric = False
            try:
                alpha_ref, _, _ = solve_centralized(h_curr, b_curr)
            except Exception:
                use_consensus_metric = True

            line_edges_curr = build_line_graph(n_agents_curr)
            w_curr = lazy_metropolis_weights(adjacency_from_edges(n_agents_curr, line_edges_curr))
            lmax_curr = max(float(np.linalg.eigvalsh(h).max()) for h in h_curr)
            eta_curr = 0.2 / lmax_curr

            t0 = time.perf_counter()
            if use_consensus_metric:
                res_curr = run_dgd_consensus_metric(
                    h_list=h_curr,
                    b_list=b_curr,
                    w=w_curr,
                    eta=eta_curr,
                    iters=iters_scale,
                    rng=np.random.default_rng(seed + 8000 + n_curr),
                )
                metric_curve = res_curr["metrics"]
                metric_kind = "consensus_error"
            else:
                res_curr = run_dgd(
                    h_list=h_curr,
                    b_list=b_curr,
                    alpha_star=alpha_ref,
                    w=w_curr,
                    eta=eta_curr,
                    iters=iters_scale,
                    rng=np.random.default_rng(seed + 8000 + n_curr),
                )
                metric_curve = res_curr["gaps"]
                metric_kind = "optimality_gap"
            dt_s = time.perf_counter() - t0

            target_eps = 1e-1 if metric_kind == "optimality_gap" else 1e-2
            reached_idx = np.where(metric_curve <= target_eps)[0]
            iters_to_target = int(reached_idx[0]) if len(reached_idx) > 0 else -1

            bonus_runs.append(
                {
                    "n": n_curr,
                    "m": m_curr,
                    "N": n_agents_curr,
                    "eta": eta_curr,
                    "runtime_s": dt_s,
                    "metric_kind": metric_kind,
                    "curve": metric_curve,
                    "final_metric": float(metric_curve[-1]),
                    "iters_to_target": iters_to_target,
                }
            )

            metrics_rows.append(
                make_metrics_row(
                    {
                        "part": "part1_bonus",
                        "algorithm": "DGD",
                        "graph": "line",
                        "iters": iters_scale,
                        "n": n_curr,
                        "m": m_curr,
                        "N": n_agents_curr,
                        "metric_kind": metric_kind,
                        "final_gap": float(metric_curve[-1]),
                        "min_gap": float(np.min(metric_curve)),
                        "iters_to_target": iters_to_target,
                        "runtime_s": dt_s,
                        "eta": eta_curr,
                        "beta": "",
                        "seed": seed,
                    },
                    curve=metric_curve,
                )
            )

        plt.figure(figsize=(7.2, 4.8))
        x_scale = np.arange(iters_scale + 1)
        ax = plt.gca()
        for idx, row in enumerate(bonus_runs):
            plot_loglog_curve(
                ax,
                x_scale,
                row["curve"],
                f"n={row['n']}, m={row['m']}, N={row['N']}",
                idx,
                floor=1e-14,
            )
        plt.xlabel("Iterations")
        plt.ylabel("Metric (optimality gap or consensus error)")
        plt.title("Part I Bonus: DGD Scaling with n")
        plt.grid(True, which="both", alpha=0.35)
        plt.legend(fontsize=8, framealpha=0.9)
        p = os.path.join(args.outdir, "part1_bonus_scaling_convergence.pdf")
        save_figure(p)
        figure_paths.append(p)

        fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.9))
        n_vals = np.array([r["n"] for r in bonus_runs], dtype=float)
        final_vals = np.array([r["final_metric"] for r in bonus_runs], dtype=float)
        rt_vals = np.array([r["runtime_s"] for r in bonus_runs], dtype=float)
        it_to_t = np.array([r["iters_to_target"] if r["iters_to_target"] > 0 else np.nan for r in bonus_runs], dtype=float)

        axes[0].loglog(n_vals, np.maximum(final_vals, 1e-14), "o-", label="Final metric")
        axes[0].set_xlabel("n")
        axes[0].set_ylabel("Final metric")
        axes[0].set_title("Final metric vs n")
        axes[0].grid(True, which="both", alpha=0.35)
        axes[0].legend()

        axes[1].plot(n_vals, rt_vals, "o-", label="Runtime (s)")
        if np.any(np.isfinite(it_to_t)):
            axes[1].plot(n_vals, it_to_t, "s-", label="Iters to target")
        axes[1].set_xlabel("n")
        axes[1].set_ylabel("Runtime / Iterations")
        axes[1].set_title("Runtime and target-accuracy vs n")
        axes[1].grid(True, alpha=0.35)
        axes[1].legend()

        p = os.path.join(args.outdir, "part1_bonus_n_dependency.pdf")
        save_figure(p)
        figure_paths.append(p)

        summary_lines.append("### Part I Bonus (n-scaling)")
        summary_lines.append(
            "- Scaling executed with m=ceil(sqrt(n)) and adaptive agent count N=min(100,max(5,n//20)) over n="
            + ", ".join([str(int(v["n"])) for v in bonus_runs])
            + "."
        )
        summary_lines.append(
            "- Highest tested n="
            + str(int(max(v["n"] for v in bonus_runs)))
            + f" with final metric {bonus_runs[-1]['final_metric']:.3e}."
        )

    if args.part in ("part2", "all"):
        m_fed = 10
        x_m_fed = np.linspace(-1.0, 1.0, m_fed)
        kmm_fed = kernel(x_m_fed, x_m_fed)
        c_total = len(x_clients)

        h_fed = []
        b_fed = []
        for i in range(c_total):
            h_i, b_i = local_quadratic_terms(
                x_clients[i], y_clients[i], x_m_fed, kmm_fed, sigma, nu, c_total
            )
            h_fed.append(h_i)
            b_fed.append(b_i)
        alpha_star_fed, _, residual_fed = solve_centralized(h_fed, b_fed)
        checks["part2_central_residual_le_tol"] = bool(np.linalg.norm(residual_fed) <= RESIDUAL_TOL)

        fed_configs_full = [{"B": 20, "Csel": 5, "E": e, "diminishing": False} for e in [1, 5, 50]]
        fed_configs_dim = [{"B": 20, "Csel": 5, "E": e, "diminishing": True} for e in [1, 5, 50]]
        fed_configs_more_1 = [{"B": 15, "Csel": 5, "E": e, "diminishing": False} for e in [1, 5, 50]]
        fed_configs_more_2 = [{"B": 15, "Csel": 3, "E": e, "diminishing": False} for e in [1, 5, 50]]

        #Run a grouped FedAvg configuration set and export one comparison figure.
        def run_fed_group(configs: List[Dict[str, int]], title: str, fname: str) -> List[Dict[str, object]]:
            out = []
            plt.figure(figsize=(7.2, 4.8))
            ax = plt.gca()
            for idx, cfg in enumerate(configs):
                tag = (
                    f"B={cfg['B']}, C={cfg['Csel']}, E={cfg['E']}, "
                    + ("dim" if cfg["diminishing"] else "const")
                )
                t0 = time.perf_counter()
                res = run_fedavg(
                    x_clients=x_clients,
                    y_clients=y_clients,
                    x_m=x_m_fed,
                    kmm=kmm_fed,
                    sigma=sigma,
                    nu=nu,
                    alpha_star=alpha_star_fed,
                    rounds=rounds_p2,
                    batch_size=cfg["B"],
                    clients_per_round=cfg["Csel"],
                    epochs=cfg["E"],
                    eta0=0.002,
                    diminishing=bool(cfg["diminishing"]),
                    rng=np.random.default_rng(seed + stable_offset(tag)),
                )
                dt_s = time.perf_counter() - t0
                plot_loglog_curve(
                    ax,
                    np.maximum(res["steps"], 1.0),
                    res["gaps"],
                    tag,
                    idx,
                    floor=1e-14,
                )
                metrics_rows.append(
                    make_metrics_row(
                        {
                            "part": "part2",
                            "algorithm": "FedAvg",
                            "graph": "cloud",
                            "iters": rounds_p2,
                            "final_gap": float(res["gaps"][-1]),
                            "min_gap": float(np.min(res["gaps"])),
                            "runtime_s": dt_s,
                            "eta": "0.002/(1+0.01t)" if cfg["diminishing"] else "0.002",
                            "beta": "",
                            "B": cfg["B"],
                            "Csel": cfg["Csel"],
                            "E": cfg["E"],
                            "seed": seed,
                        },
                        curve=res["gaps"],
                        axis=np.maximum(res["steps"], 1.0),
                    )
                )
                out.append({"cfg": cfg, "res": res})
            plt.xlabel("Iterations (T*E)")
            plt.ylabel(r"Objective error $F(\alpha_t)-F(\alpha^*)$")
            plt.title(title)
            plt.grid(True, which="both", alpha=0.35)
            plt.legend(fontsize=8, framealpha=0.9)
            p = os.path.join(args.outdir, fname)
            save_figure(p)
            figure_paths.append(p)
            return out

        full_const = run_fed_group(
            fed_configs_full,
            "Part II FedAvg: B=20, C=5, constant LR",
            "part2_fedavg_b20_c5_constant_lr.pdf",
        )
        full_dim = run_fed_group(
            fed_configs_dim,
            "Part II FedAvg: B=20, C=5, diminishing LR",
            "part2_fedavg_b20_c5_diminishing_lr.pdf",
        )
        more_1 = run_fed_group(
            fed_configs_more_1,
            "Part II FedAvg: B=15, C=5, constant LR",
            "part2_fedavg_b15_c5_constant_lr.pdf",
        )
        more_2 = run_fed_group(
            fed_configs_more_2,
            "Part II FedAvg: B=15, C=3, constant LR",
            "part2_fedavg_b15_c3_constant_lr.pdf",
        )

        for csel in [5, 3]:
            plt.figure(figsize=(7.2, 4.8))
            ax = plt.gca()
            scaffold_finals = []
            for idx, e in enumerate([1, 5, 50]):
                tag = f"SCAFFOLD C={csel}, E={e}"
                t0 = time.perf_counter()
                res_s = run_scaffold(
                    x_clients=x_clients,
                    y_clients=y_clients,
                    x_m=x_m_fed,
                    kmm=kmm_fed,
                    sigma=sigma,
                    nu=nu,
                    alpha_star=alpha_star_fed,
                    rounds=rounds_p2,
                    batch_size=20,
                    clients_per_round=csel,
                    epochs=e,
                    eta=0.002,
                    rng=np.random.default_rng(seed + stable_offset(tag)),
                )
                dt_s = time.perf_counter() - t0
                plot_loglog_curve(
                    ax,
                    np.maximum(res_s["steps"], 1.0),
                    res_s["gaps"],
                    tag,
                    idx,
                    floor=1e-14,
                )
                metrics_rows.append(
                    make_metrics_row(
                        {
                            "part": "part2",
                            "algorithm": "SCAFFOLD",
                            "graph": "cloud",
                            "iters": rounds_p2,
                            "final_gap": float(res_s["gaps"][-1]),
                            "min_gap": float(np.min(res_s["gaps"])),
                            "runtime_s": dt_s,
                            "eta": "0.002",
                            "beta": "",
                            "B": 20,
                            "Csel": csel,
                            "E": e,
                            "seed": seed,
                        },
                        curve=res_s["gaps"],
                        axis=np.maximum(res_s["steps"], 1.0),
                    )
                )
                scaffold_finals.append(float(res_s["gaps"][-1]))

            plt.xlabel("Iterations (T*E)")
            plt.ylabel(r"Objective error $F(\alpha_t)-F(\alpha^*)$")
            plt.title(f"Part II Optional SCAFFOLD: B=20, C={csel}")
            plt.grid(True, which="both", alpha=0.35)
            plt.legend(fontsize=8, framealpha=0.9)
            p = os.path.join(args.outdir, f"part2_scaffold_b20_c{csel}.pdf")
            save_figure(p)
            figure_paths.append(p)
            summary_lines.append(
                f"- Part II SCAFFOLD final objective errors (C={csel}, E=1/5/50): "
                + ", ".join([f"{v:.3e}" for v in scaffold_finals])
            )

        summary_lines.append("## Part II")
        summary_lines.append(
            "- FedAvg mandatory and additional settings were executed: "
            "B=20,C=5; B=15,C=5; B=15,C=3; E in {1,5,50}; constant + diminishing LR."
        )
        summary_lines.append(
            f"- Example final errors (FedAvg B=20,C=5,const): "
            + ", ".join([f"E={r['cfg']['E']} -> {float(r['res']['gaps'][-1]):.3e}" for r in full_const])
        )
        summary_lines.append(
            f"- Example final errors (FedAvg B=20,C=5,dim): "
            + ", ".join([f"E={r['cfg']['E']} -> {float(r['res']['gaps'][-1]):.3e}" for r in full_dim])
        )
        best_const = min(float(r["res"]["gaps"][-1]) for r in full_const)
        best_dim = min(float(r["res"]["gaps"][-1]) for r in full_dim)
        summary_lines.append(
            f"- Convergence note: best FedAvg constant-LR final error={best_const:.3e}, diminishing-LR final error={best_dim:.3e}."
        )
        _ = more_1, more_2

    if args.part in ("part3", "all"):
        # Baseline Part III setting from mandatory task: n=100, m=10, a=5.
        line_edges = build_line_graph(n_agents)
        line_adj = adjacency_from_edges(n_agents, line_edges)
        w_line = lazy_metropolis_weights(line_adj)
        part3_audit = part3_theorem_audit(h_list=h_list, w_mix=w_line, iters=iters_p3, eps=1.0)
        checks["part3_w_symmetric"] = bool(part3_audit["w_symmetric"])
        checks["part3_w_doubly_stochastic"] = bool(part3_audit["w_doubly_stochastic"])
        checks["part3_wpaper_row_sum_zero"] = bool(part3_audit["w_paper_row_zero"])
        checks["part3_i_plus_wpaper_minus_proj_norm_lt_one"] = bool(part3_audit["i_plus_wpaper_minus_proj_norm_lt_one"])
        checks["part3_alpha_decreasing"] = bool(part3_audit["alpha_decreasing"])
        checks["part3_gamma_decreasing"] = bool(part3_audit["gamma_decreasing"])
        checks["part3_nu_increasing"] = bool(part3_audit["nu_increasing"])
        checks["part3_asymptotic_alpha_diverges"] = bool(part3_audit["asymptotic_alpha_diverges"])
        checks["part3_asymptotic_gamma_diverges"] = bool(part3_audit["asymptotic_gamma_diverges"])
        checks["part3_asymptotic_alpha2_over_gamma_converges"] = bool(
            part3_audit["asymptotic_alpha2_over_gamma_converges"]
        )
        checks["part3_asymptotic_alpha_over_nu_converges"] = bool(part3_audit["asymptotic_alpha_over_nu_converges"])

        t0_np = time.perf_counter()
        res_np = run_non_private_reference_dgd(
            h_list=h_list,
            b_list=b_list,
            alpha_star=alpha_star,
            w=w_line,
            iters=iters_p3,
            rng=np.random.default_rng(seed + 5001),
        )
        dt_np = time.perf_counter() - t0_np
        metrics_rows.append(
            make_metrics_row(
                {
                    "part": "part3",
                    "algorithm": "DGD_non_private",
                    "graph": "line",
                    "iters": iters_p3,
                    "final_gap": float(res_np["gaps"][-1]),
                    "min_gap": float(np.min(res_np["gaps"])),
                    "runtime_s": dt_np,
                    "eta": "alpha_k=0.002/(1+0.001k), gamma_k=1/(1+0.001k^0.9)",
                    "beta": "",
                    "epsilon": "none",
                    "Lmax": part3_audit["lmax"],
                    "spectral_gap": part3_audit["spectral_gap"],
                    "w_symmetric": part3_audit["w_symmetric"],
                    "w_doubly_stochastic": part3_audit["w_doubly_stochastic"],
                    "w_paper_row_zero": part3_audit["w_paper_row_zero"],
                    "w_norm_lt_one": part3_audit["i_plus_wpaper_minus_proj_norm_lt_one"],
                    "sum_alpha2_over_gamma": part3_audit["sum_alpha2_over_gamma"],
                    "sum_gamma2_nu": part3_audit["sum_gamma2_nu"],
                    "sum_alpha_over_nu": "",
                    "C_emp": "",
                    "epsilon_bar_emp": "",
                    "sensitivity_assumption_mode": "unclipped_assumption_audit",
                    "seed": seed,
                },
                curve=res_np["gaps"],
            )
        )
        eps_list = [0.1, 1.0, 10.0]
        dp_results = {}
        for eps in eps_list:
            t0 = time.perf_counter()
            res = run_dp_dgd(
                h_list=h_list,
                b_list=b_list,
                alpha_star=alpha_star,
                w=w_line,
                iters=iters_p3,
                eps=eps,
                rng=np.random.default_rng(seed + int(6000 + 10 * eps)),
            )
            dt_s = time.perf_counter() - t0
            dp_results[eps] = res
            metrics_rows.append(
                make_metrics_row(
                    {
                        "part": "part3",
                        "algorithm": "DP-DGD",
                        "graph": "line",
                        "iters": iters_p3,
                        "final_gap": float(res["gaps"][-1]),
                        "min_gap": float(np.min(res["gaps"])),
                        "runtime_s": dt_s,
                        "eta": "alpha_k=0.002/(1+0.001k), gamma_k=1/(1+0.001k^0.9), nu_k=(0.01/eps)*(1+0.001k^0.1)",
                        "beta": "",
                        "epsilon": eps,
                        "Lmax": part3_audit["lmax"],
                        "spectral_gap": part3_audit["spectral_gap"],
                        "w_symmetric": part3_audit["w_symmetric"],
                        "w_doubly_stochastic": part3_audit["w_doubly_stochastic"],
                        "w_paper_row_zero": part3_audit["w_paper_row_zero"],
                        "w_norm_lt_one": part3_audit["i_plus_wpaper_minus_proj_norm_lt_one"],
                        "sum_alpha2_over_gamma": float(res["sum_alpha2_over_gamma"]),
                        "sum_gamma2_nu": float(res["sum_gamma2_nu"]),
                        "sum_alpha_over_nu": float(res["sum_alpha_over_nu"]),
                        "C_emp": float(res["c_emp"]),
                        "epsilon_bar_emp": float(res["epsilon_bar_emp"]),
                        "sensitivity_assumption_mode": "unclipped_assumption_audit",
                        "seed": seed,
                    },
                    curve=res["gaps"],
                )
            )
        checks["part3_dp_alpha_decreasing_all_eps"] = bool(all(dp_results[e]["alpha_decreasing"] for e in eps_list))
        checks["part3_dp_gamma_decreasing_all_eps"] = bool(all(dp_results[e]["gamma_decreasing"] for e in eps_list))
        checks["part3_dp_nu_increasing_all_eps"] = bool(all(dp_results[e]["nu_increasing"] for e in eps_list))
        checks["part3_dp_c_emp_finite_all_eps"] = bool(
            all(np.isfinite(float(dp_results[e]["c_emp"])) and float(dp_results[e]["c_emp"]) > 0.0 for e in eps_list)
        )
        checks["part3_dp_epsilon_bar_emp_finite_all_eps"] = bool(
            all(np.isfinite(float(dp_results[e]["epsilon_bar_emp"])) for e in eps_list)
        )

        plt.figure(figsize=(7.2, 4.8))
        it = np.arange(iters_p3 + 1)
        ax = plt.gca()
        plot_loglog_curve(ax, it, res_np["gaps"], "Non-private DGD reference", 0, floor=1e-12)
        for idx, eps in enumerate(eps_list, start=1):
            plot_loglog_curve(ax, it, dp_results[eps]["gaps"], f"DP-DGD eps={eps:g}", idx, floor=1e-12)
        plt.xlabel("Iterations")
        plt.ylabel(r"Mean optimality gap $||\alpha_i^k-\alpha^*||$")
        plt.title("Part III: DP-DGD Privacy-Utility Trade-off")
        plt.grid(True, which="both", alpha=0.35)
        plt.legend(fontsize=9, framealpha=0.9)
        p = os.path.join(args.outdir, "part3_dp_dgd_tradeoff.pdf")
        save_figure(p)
        figure_paths.append(p)

        summary_lines.append("## Part III")
        summary_lines.append(
            f"- Non-private reference final gap: {res_np['gaps'][-1]:.3e}."
        )
        summary_lines.append(
            "- DP-DGD final gaps: " + ", ".join([f"eps={eps:g} -> {dp_results[eps]['gaps'][-1]:.3e}" for eps in eps_list]) + "."
        )
        summary_lines.append("- As expected, smaller eps produces stronger privacy and weaker optimization accuracy.")
        summary_lines.append(
            "- Part III theorem-alignment (paper 2202.01113v10 / notes 9.5): "
            f"Lmax={part3_audit['lmax']:.3e}, spectral_gap={part3_audit['spectral_gap']:.3e}, "
            f"||I+W_paper-11^T/N||_2={part3_audit['i_plus_wpaper_minus_proj_norm']:.3e}."
        )
        summary_lines.append(
            "- Prefix sequence sums (T horizon): "
            f"sum(alpha)={part3_audit['sum_alpha_prefix']:.3e}, sum(gamma)={part3_audit['sum_gamma_prefix']:.3e}, "
            f"sum(alpha^2/gamma)={part3_audit['sum_alpha2_over_gamma']:.3e}, sum(alpha/nu)|eps=1={part3_audit['sum_alpha_over_nu']:.3e}."
        )
        summary_lines.append(
            "- DP sensitivity/privacy diagnostics (assumption-backed): "
            + ", ".join(
                [
                    f"eps={eps:g}: C_emp={dp_results[eps]['c_emp']:.3e}, epsilon_bar_emp={dp_results[eps]['epsilon_bar_emp']:.3e}"
                    for eps in eps_list
                ]
            )
            + "."
        )
        dp_reduction = {eps: convergence_metrics(dp_results[eps]["gaps"])["reduction_factor"] for eps in eps_list}
        summary_lines.append(
            "- Convergence note: DP-DGD reduction factors "
            + ", ".join([f"eps={eps:g} -> {dp_reduction[eps]:.3e}" for eps in eps_list])
            + "."
        )

        eps_order = [dp_results[10.0]["gaps"][-1], dp_results[1.0]["gaps"][-1], dp_results[0.1]["gaps"][-1]]
        checks["part3_dp_gap_order_eps10_le_eps1_le_eps0p1"] = bool(eps_order[0] <= eps_order[1] <= eps_order[2])

        # Optional large-scale Part III setting: n=1000, m=33, a=100.
        n_opt = 1000
        m_opt = 33
        a_opt = 100
        opt_setup = build_distributed_kernel_setting(
            x_all=x_big,
            y_all=y_big,
            n_points=n_opt,
            m_points=m_opt,
            n_agents=a_opt,
            sigma=sigma,
            nu=nu,
            rng=np.random.default_rng(seed + 76000),
        )
        checks["part3_optional_central_residual_le_tol"] = bool(opt_setup["residual_norm"] <= RESIDUAL_TOL)
        checks["part3_optional_partition_sequential_chunking"] = bool(opt_setup["partition_meta"]["sequential"])
        checks["part3_optional_partition_disjoint_cover"] = bool(opt_setup["partition_meta"]["disjoint_cover"])
        checks["part3_optional_partition_counts_balanced"] = bool(opt_setup["partition_meta"]["counts_balanced"])

        line_edges_opt = build_line_graph(a_opt)
        line_adj_opt = adjacency_from_edges(a_opt, line_edges_opt)
        w_line_opt = lazy_metropolis_weights(line_adj_opt)
        if args.mode == "quick":
            iters_p3_opt = 1200
        else:
            iters_p3_opt = 3000
        part3_audit_opt = part3_theorem_audit(
            h_list=opt_setup["h_list"], w_mix=w_line_opt, iters=iters_p3_opt, eps=1.0
        )
        checks["part3_optional_w_symmetric"] = bool(part3_audit_opt["w_symmetric"])
        checks["part3_optional_w_doubly_stochastic"] = bool(part3_audit_opt["w_doubly_stochastic"])
        checks["part3_optional_wpaper_row_sum_zero"] = bool(part3_audit_opt["w_paper_row_zero"])
        checks["part3_optional_w_norm_lt_one"] = bool(part3_audit_opt["i_plus_wpaper_minus_proj_norm_lt_one"])
        checks["part3_optional_alpha_decreasing"] = bool(part3_audit_opt["alpha_decreasing"])
        checks["part3_optional_gamma_decreasing"] = bool(part3_audit_opt["gamma_decreasing"])
        checks["part3_optional_nu_increasing"] = bool(part3_audit_opt["nu_increasing"])

        t0_np_opt = time.perf_counter()
        res_np_opt = run_non_private_reference_dgd(
            h_list=opt_setup["h_list"],
            b_list=opt_setup["b_list"],
            alpha_star=opt_setup["alpha_star"],
            w=w_line_opt,
            iters=iters_p3_opt,
            rng=np.random.default_rng(seed + 50011),
        )
        dt_np_opt = time.perf_counter() - t0_np_opt
        metrics_rows.append(
            make_metrics_row(
                {
                    "part": "part3_optional",
                    "algorithm": "DGD_non_private",
                    "graph": "line",
                    "iters": iters_p3_opt,
                    "final_gap": float(res_np_opt["gaps"][-1]),
                    "min_gap": float(np.min(res_np_opt["gaps"])),
                    "runtime_s": dt_np_opt,
                    "eta": "alpha_k=0.002/(1+0.001k), gamma_k=1/(1+0.001k^0.9)",
                    "beta": "",
                    "epsilon": "none",
                    "Lmax": part3_audit_opt["lmax"],
                    "spectral_gap": part3_audit_opt["spectral_gap"],
                    "w_symmetric": part3_audit_opt["w_symmetric"],
                    "w_doubly_stochastic": part3_audit_opt["w_doubly_stochastic"],
                    "w_paper_row_zero": part3_audit_opt["w_paper_row_zero"],
                    "w_norm_lt_one": part3_audit_opt["i_plus_wpaper_minus_proj_norm_lt_one"],
                    "sum_alpha2_over_gamma": part3_audit_opt["sum_alpha2_over_gamma"],
                    "sum_gamma2_nu": part3_audit_opt["sum_gamma2_nu"],
                    "sum_alpha_over_nu": "",
                    "C_emp": "",
                    "epsilon_bar_emp": "",
                    "sensitivity_assumption_mode": "unclipped_assumption_audit",
                    "n": n_opt,
                    "m": m_opt,
                    "a": a_opt,
                    "seed": seed,
                },
                curve=res_np_opt["gaps"],
            )
        )
        dp_results_opt: Dict[float, Dict[str, np.ndarray]] = {}
        for eps in eps_list:
            t0 = time.perf_counter()
            res_opt = run_dp_dgd(
                h_list=opt_setup["h_list"],
                b_list=opt_setup["b_list"],
                alpha_star=opt_setup["alpha_star"],
                w=w_line_opt,
                iters=iters_p3_opt,
                eps=eps,
                rng=np.random.default_rng(seed + int(61000 + 100 * eps)),
            )
            dt_s = time.perf_counter() - t0
            dp_results_opt[eps] = res_opt
            metrics_rows.append(
                make_metrics_row(
                    {
                        "part": "part3_optional",
                        "algorithm": "DP-DGD",
                        "graph": "line",
                        "iters": iters_p3_opt,
                        "final_gap": float(res_opt["gaps"][-1]),
                        "min_gap": float(np.min(res_opt["gaps"])),
                        "runtime_s": dt_s,
                        "eta": "alpha_k=0.002/(1+0.001k), gamma_k=1/(1+0.001k^0.9), nu_k=(0.01/eps)*(1+0.001k^0.1)",
                        "beta": "",
                        "epsilon": eps,
                        "Lmax": part3_audit_opt["lmax"],
                        "spectral_gap": part3_audit_opt["spectral_gap"],
                        "w_symmetric": part3_audit_opt["w_symmetric"],
                        "w_doubly_stochastic": part3_audit_opt["w_doubly_stochastic"],
                        "w_paper_row_zero": part3_audit_opt["w_paper_row_zero"],
                        "w_norm_lt_one": part3_audit_opt["i_plus_wpaper_minus_proj_norm_lt_one"],
                        "sum_alpha2_over_gamma": float(res_opt["sum_alpha2_over_gamma"]),
                        "sum_gamma2_nu": float(res_opt["sum_gamma2_nu"]),
                        "sum_alpha_over_nu": float(res_opt["sum_alpha_over_nu"]),
                        "C_emp": float(res_opt["c_emp"]),
                        "epsilon_bar_emp": float(res_opt["epsilon_bar_emp"]),
                        "sensitivity_assumption_mode": "unclipped_assumption_audit",
                        "n": n_opt,
                        "m": m_opt,
                        "a": a_opt,
                        "seed": seed,
                    },
                    curve=res_opt["gaps"],
                )
            )
        checks["part3_optional_dp_alpha_decreasing_all_eps"] = bool(
            all(dp_results_opt[e]["alpha_decreasing"] for e in eps_list)
        )
        checks["part3_optional_dp_gamma_decreasing_all_eps"] = bool(
            all(dp_results_opt[e]["gamma_decreasing"] for e in eps_list)
        )
        checks["part3_optional_dp_nu_increasing_all_eps"] = bool(
            all(dp_results_opt[e]["nu_increasing"] for e in eps_list)
        )
        checks["part3_optional_dp_c_emp_finite_all_eps"] = bool(
            all(np.isfinite(float(dp_results_opt[e]["c_emp"])) and float(dp_results_opt[e]["c_emp"]) > 0.0 for e in eps_list)
        )
        checks["part3_optional_dp_epsilon_bar_emp_finite_all_eps"] = bool(
            all(np.isfinite(float(dp_results_opt[e]["epsilon_bar_emp"])) for e in eps_list)
        )

        plt.figure(figsize=(7.2, 4.8))
        it_opt = np.arange(iters_p3_opt + 1)
        ax = plt.gca()
        plot_loglog_curve(ax, it_opt, res_np_opt["gaps"], "Non-private DGD reference", 0, floor=1e-12)
        for idx, eps in enumerate(eps_list, start=1):
            plot_loglog_curve(ax, it_opt, dp_results_opt[eps]["gaps"], f"DP-DGD eps={eps:g}", idx, floor=1e-12)
        plt.xlabel("Iterations")
        plt.ylabel(r"Mean optimality gap $||\alpha_i^k-\alpha^*||$")
        plt.title("Part III Optional: DP-DGD (n=1000, m=33, a=100)")
        plt.grid(True, which="both", alpha=0.35)
        plt.legend(fontsize=9, framealpha=0.9)
        p = os.path.join(args.outdir, "part3_optional_dp_dgd_tradeoff.pdf")
        save_figure(p)
        figure_paths.append(p)

        # Richer visualization: compare baseline and optional final gaps versus epsilon.
        eps_arr = np.array(eps_list, dtype=float)
        base_final = np.array([dp_results[e]["gaps"][-1] for e in eps_list], dtype=float)
        opt_final = np.array([dp_results_opt[e]["gaps"][-1] for e in eps_list], dtype=float)
        plt.figure(figsize=(7.2, 4.5))
        plt.loglog(
            eps_arr,
            np.maximum(base_final, 1e-12),
            "o-",
            color=plt.cm.tab10.colors[0],
            linewidth=2.2,
            markersize=6,
            label="Baseline n=100,m=10,a=5",
        )
        plt.loglog(
            eps_arr,
            np.maximum(opt_final, 1e-12),
            "s-",
            color=plt.cm.tab10.colors[3],
            linewidth=2.2,
            markersize=6,
            label="Optional n=1000,m=33,a=100",
        )
        plt.xlabel(r"Privacy parameter $\epsilon$")
        plt.ylabel("Final optimality gap")
        plt.title("Part III: Baseline vs Optional Large-Scale DP Setting")
        plt.grid(True, which="both", alpha=0.35)
        plt.legend(fontsize=9, framealpha=0.9)
        p = os.path.join(args.outdir, "part3_baseline_vs_optional_final_gap.pdf")
        save_figure(p)
        figure_paths.append(p)

        # Track optional ordering expectation as in baseline.
        eps_order_opt = [dp_results_opt[10.0]["gaps"][-1], dp_results_opt[1.0]["gaps"][-1], dp_results_opt[0.1]["gaps"][-1]]
        checks["part3_optional_dp_gap_order_eps10_le_eps1_le_eps0p1"] = bool(
            eps_order_opt[0] <= eps_order_opt[1] <= eps_order_opt[2]
        )

        summary_lines.append(
            "- Optional Part III (n=1000,m=33,a=100) final gaps: "
            + ", ".join([f"eps={eps:g} -> {dp_results_opt[eps]['gaps'][-1]:.3e}" for eps in eps_list])
            + "."
        )
        summary_lines.append(
            "- Optional theorem-alignment diagnostics: "
            f"Lmax={part3_audit_opt['lmax']:.3e}, spectral_gap={part3_audit_opt['spectral_gap']:.3e}, "
            f"||I+W_paper-11^T/N||_2={part3_audit_opt['i_plus_wpaper_minus_proj_norm']:.3e}."
        )
        summary_lines.append(
            f"- Optional Part III non-private reference final gap: {res_np_opt['gaps'][-1]:.3e}."
        )

    short_rng_1 = np.random.default_rng(seed + 9001)
    short_rng_2 = np.random.default_rng(seed + 9001)
    line_edges = build_line_graph(n_agents)
    line_adj = adjacency_from_edges(n_agents, line_edges)
    w_line = lazy_metropolis_weights(line_adj)
    r1 = run_dgd(h_list, b_list, alpha_star, w_line, 0.2 / lmax, 40, short_rng_1)
    r2 = run_dgd(h_list, b_list, alpha_star, w_line, 0.2 / lmax, 40, short_rng_2)
    checks["reproducibility_short_run_dgd"] = bool(np.allclose(r1["gaps"], r2["gaps"], atol=1e-15))

    # Keep intermediate result discussion in memory/console (no markdown export).
    generated_figure_names = {os.path.basename(fp) for fp in figure_paths}
    expected_figures = []
    if args.part in ("part1", "all"):
        expected_figures.extend(
            [
                "part1_data_preview.pdf",
                "part1_algorithms_line_loglog.pdf",
                "part1_admm_beta_sweep.pdf",
                "part1_graph_structure_comparison.pdf",
                "part1_reconstruction_nt250.pdf",
                "part1_failure_and_pushsum.pdf",
                "part1_bonus_scaling_convergence.pdf",
                "part1_bonus_n_dependency.pdf",
            ]
        )
    if args.part in ("part2", "all"):
        expected_figures.extend(
            [
                "part2_fedavg_b20_c5_constant_lr.pdf",
                "part2_fedavg_b20_c5_diminishing_lr.pdf",
                "part2_fedavg_b15_c5_constant_lr.pdf",
                "part2_fedavg_b15_c3_constant_lr.pdf",
                "part2_scaffold_b20_c5.pdf",
                "part2_scaffold_b20_c3.pdf",
            ]
        )
    if args.part in ("part3", "all"):
        expected_figures.extend(
            [
                "part3_dp_dgd_tradeoff.pdf",
                "part3_optional_dp_dgd_tradeoff.pdf",
                "part3_baseline_vs_optional_final_gap.pdf",
            ]
        )
    checks["all_expected_figures_generated"] = bool(all(name in generated_figure_names for name in expected_figures))

    summary_lines.insert(0, "## Validation Checks")
    for k, v in checks.items():
        summary_lines.insert(1, f"- {k}: {'PASS' if v else 'FAIL'}")

    metrics_path = os.path.join(args.outdir, "metrics_summary.csv")
    write_metrics_csv(metrics_path, metrics_rows)

    if args.debug_output:
        print("\nResults summary (console only):")
        for line in summary_lines:
            print(line)
    else:
        print("Debug summary suppressed (enable with --debug_output).")

    print("Run completed.")
    print(f"Output directory: {args.outdir}")
    print(f"Figures generated: {len(figure_paths)}")
    print(f"Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()
