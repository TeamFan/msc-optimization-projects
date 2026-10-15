"""
AL-CMA-ES Comparison Study

Comparing constraint handling methods for CMA-ES:
- Part 1: Baseline Comparison (Sphere + Linear Constraint)
- Part 2: Ill-Conditioned Scenario (Ellipsoid)
- Part 3: Non-Smooth Constraint (Piecewise linear)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cma
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import os
import warnings
warnings.filterwarnings('ignore')


# Configuration
NUM_RUNS = 5
SEED_BASE = 42


# =============================================================================
# Test Functions

def sphere(x: np.ndarray) -> float:
    """f(x) = sum(x_i^2)"""
    return np.sum(np.array(x)**2)


def ellipsoid(x: np.ndarray) -> float:
    """Ill-conditioned ellipsoid with condition number 10^6."""
    x = np.array(x)
    n = len(x)
    if n == 1:
        return x[0]**2
    coeffs = 10 ** (6 * np.arange(n) / (n - 1))
    return np.sum(coeffs * x**2)


def linear_constraint_x1_ge_c(x: np.ndarray, c: float = 1.0) -> float:
    """g(x) = c - x[0] <= 0 (i.e., x[0] >= c)"""
    return c - x[0]


def shifted_sphere(x: np.ndarray, center: float = 1.0) -> float:
    """f(x) = (x₁ - center)² + sum(x_i²). Optimum pushed toward infeasible region."""
    x = np.array(x)
    result = (x[0] - center)**2
    if len(x) > 1:
        result += np.sum(x[1:]**2)
    return result


def piecewise_constraint_kink(x: np.ndarray) -> float:
    """
    Piecewise constraint with gradient discontinuity at x[0] = 0.
    g(x) = 2*x₁ if x₁ <= 0, else x₁
    """
    if x[0] <= 0:
        return 2 * x[0]
    else:
        return x[0]


# Constraint Handling Methods


class VanillaCMAESFitness:
    """
    Vanilla CMA-ES: Ignores constraints entirely.
    Just minimizes f(x) without any constraint handling.
    
    This baseline shows what happens without constraint handling:
    - May find very low f(x) but g(x) > 0 (infeasible)
    """
    def __init__(self, objective: Callable, constraint: Callable):
        self.objective = objective
        self.constraint = constraint
        self.f_history = []
        self.g_history = []
        self.x_history = []
        self.eval_count = 0
        
    def __call__(self, x: np.ndarray) -> float:
        f = self.objective(x)
        g = self.constraint(x)
        self.f_history.append(f)
        self.g_history.append(g)
        self.x_history.append(np.array(x).copy())
        self.eval_count += 1
        return f


class DeathPenaltyFitness:
    """Death Penalty: F(x) = f(x) + M if g(x) > 0"""
    def __init__(self, objective: Callable, constraint: Callable, 
                 death_value: float = 1e10):
        self.objective = objective
        self.constraint = constraint
        self.death_value = death_value
        self.f_history = []
        self.g_history = []
        self.x_history = []
        self.eval_count = 0
        
    def __call__(self, x: np.ndarray) -> float:
        f = self.objective(x)
        g = self.constraint(x)
        self.f_history.append(f)
        self.g_history.append(g)
        self.x_history.append(np.array(x).copy())
        self.eval_count += 1
        if g > 0:
            return f + self.death_value
        return f


class StaticPenaltyFitness:
    """Static quadratic penalty: F(x) = f(x) + β * max(0, g(x))²"""
    def __init__(self, objective: Callable, constraint: Callable, beta: float):
        self.objective = objective
        self.constraint = constraint
        self.beta = beta
        self.f_history = []
        self.g_history = []
        self.x_history = []
        self.eval_count = 0
        
    def __call__(self, x: np.ndarray) -> float:
        f = self.objective(x)
        g = self.constraint(x)
        violation = max(0, g)
        penalty = self.beta * violation**2
        self.f_history.append(f)
        self.g_history.append(g)
        self.x_history.append(np.array(x).copy())
        self.eval_count += 1
        return f + penalty


class AdaptivePenaltyFitness:
    """Adaptive quadratic penalty: β increases when mean is infeasible."""
    def __init__(self, objective: Callable, constraint: Callable,
                 beta_init: float = 1.0, kappa: float = 2.0):
        self.objective = objective
        self.constraint = constraint
        self.beta = beta_init
        self.kappa = kappa
        self.f_history = []
        self.g_history = []
        self.x_history = []
        self.beta_history = [beta_init]
        
    def __call__(self, x: np.ndarray) -> float:
        f = self.objective(x)
        g = self.constraint(x)
        violation = max(0, g)
        penalty = self.beta * violation**2
        self.f_history.append(f)
        self.g_history.append(g)
        self.x_history.append(np.array(x).copy())
        return f + penalty
    
    def update(self, es):
        """Increase β if mean is infeasible."""
        g_mean = self.constraint(es.mean)
        if g_mean > 0:
            self.beta *= self.kappa
        self.beta_history.append(self.beta)


class SurrogateALFitness:
    """Surrogate-Assisted AL for non-smooth constraints using linear regression."""
    def __init__(self, objective: Callable, constraint: Callable, 
                 archive_size: int = 50):
        self.objective = objective
        self.constraint = constraint
        self.archive_size = archive_size
        
        # Archive for surrogate model
        self.archive_x = []
        self.archive_g = []
        
        # For AL updates, we use a wrapper that applies the surrogate
        self.cfun = None  # Will be initialized in run
        
        # History tracking
        self.f_history = []
        self.g_history = []
        self.x_history = []
        self.gamma_history = []
        self.g_surrogate_history = []  # Track surrogate predictions
    
    def _fit_surrogate(self):
        """Fit linear regression: g(x) ≈ w^T * x + b"""
        if len(self.archive_x) < 3:
            return None
        X = np.array(self.archive_x)
        G = np.array(self.archive_g)
        X_aug = np.hstack([X, np.ones((len(X), 1))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_aug, G, rcond=None)
            return (coeffs[:-1], coeffs[-1])
        except:
            return None
    
    def surrogate_constraint(self, x: np.ndarray) -> float:
        """Compute surrogate constraint ĝ(x)."""
        params = self._fit_surrogate()
        if params is None:
            return self.constraint(x)
        w, b = params
        return float(np.dot(w, np.array(x)) + b)
    
    def update_archive(self, x: np.ndarray, g: float):
        """Add point to archive."""
        if g > -0.1:
            self.archive_x.append(np.array(x).copy())
            self.archive_g.append(g)
            if len(self.archive_x) > self.archive_size:
                self.archive_x.pop(0)
                self.archive_g.pop(0)


# Data Containers

@dataclass
class ExperimentResult:
    """Single experiment run result."""
    name: str
    f_history: List[float]
    g_history: List[float]
    x_history: List[np.ndarray]
    eval_history: List[int]
    final_x: np.ndarray
    final_f: float
    final_g: float
    gamma_history: List[float] = field(default_factory=list)
    beta_history: List[float] = field(default_factory=list)
    sigma_history: List[float] = field(default_factory=list)
    condition_history: List[float] = field(default_factory=list)
    fevals: int = 0


@dataclass
class AggregatedResult:
    """Aggregated results over multiple runs."""
    name: str
    distances_all: List[List[float]]
    eval_histories_all: List[List[int]]
    final_f_all: List[float]
    final_g_all: List[float]
    fevals_all: List[int]
    gamma_histories: List[List[float]] = field(default_factory=list)


# Experiment Runners


def run_static_penalty(
    objective: Callable,
    constraint: Callable,
    x0: np.ndarray,
    sigma0: float,
    beta: float,
    x_optimal: np.ndarray,
    max_fevals: int = 5000,
    seed: int = None
) -> ExperimentResult:
    """Run CMA-ES with static penalty. Records per-generation best."""
    fitness = StaticPenaltyFitness(objective, constraint, beta)
    
    opts = {
        'maxfevals': max_fevals,
        'verbose': -9,
        'verb_disp': 0,
        'verb_log': 0,
    }
    if seed is not None:
        opts['seed'] = seed
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    # Per-generation tracking
    f_history = []
    g_history = []
    x_history = []
    eval_history = []
    
    while not es.stop():
        solutions = es.ask()
        values = [fitness(x) for x in solutions]
        es.tell(solutions, values)
        
        # Record best-of-generation
        best_idx = np.argmin(values)
        best_x = solutions[best_idx]
        f_history.append(objective(best_x))
        g_history.append(constraint(best_x))
        x_history.append(best_x.copy())
        eval_history.append(es.result.evaluations)
    
    final_x = es.result.xbest
    
    return ExperimentResult(
        name=f"Static (β={beta:.0e})",
        f_history=f_history,
        g_history=g_history,
        x_history=x_history,
        eval_history=eval_history,
        final_x=final_x,
        final_f=objective(final_x),
        final_g=constraint(final_x),
        fevals=es.result.evaluations
    )


def run_vanilla_cmaes(
    objective: Callable,
    constraint: Callable,
    x0: np.ndarray,
    sigma0: float,
    x_optimal: np.ndarray,
    max_fevals: int = 5000,
    seed: int = None
) -> ExperimentResult:
    """Run Vanilla CMA-ES (ignores constraints)."""
    fitness = VanillaCMAESFitness(objective, constraint)
    
    opts = {
        'maxfevals': max_fevals,
        'verbose': -9,
        'verb_disp': 0,
        'verb_log': 0,
    }
    if seed is not None:
        opts['seed'] = seed
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    f_history = []
    g_history = []
    x_history = []
    eval_history = []
    
    while not es.stop():
        solutions = es.ask()
        values = [fitness(x) for x in solutions]
        es.tell(solutions, values)
        
        best_idx = np.argmin(values)
        best_x = solutions[best_idx]
        f_history.append(objective(best_x))
        g_history.append(constraint(best_x))
        x_history.append(best_x.copy())
        eval_history.append(es.result.evaluations)
    
    final_x = es.result.xbest
    
    return ExperimentResult(
        name="Vanilla CMA-ES",
        f_history=f_history,
        g_history=g_history,
        x_history=x_history,
        eval_history=eval_history,
        final_x=final_x,
        final_f=objective(final_x),
        final_g=constraint(final_x),
        fevals=es.result.evaluations
    )


def run_death_penalty(
    objective: Callable,
    constraint: Callable,
    x0: np.ndarray,
    sigma0: float,
    x_optimal: np.ndarray,
    max_fevals: int = 5000,
    seed: int = None
) -> ExperimentResult:
    """Run CMA-ES with Death Penalty."""
    fitness = DeathPenaltyFitness(objective, constraint)
    
    opts = {
        'maxfevals': max_fevals,
        'verbose': -9,
        'verb_disp': 0,
        'verb_log': 0,
    }
    if seed is not None:
        opts['seed'] = seed
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    f_history = []
    g_history = []
    x_history = []
    eval_history = []
    
    while not es.stop():
        solutions = es.ask()
        values = [fitness(x) for x in solutions]
        es.tell(solutions, values)
        
        best_idx = np.argmin(values)
        best_x = solutions[best_idx]
        f_history.append(objective(best_x))
        g_history.append(constraint(best_x))
        x_history.append(best_x.copy())
        eval_history.append(es.result.evaluations)
    
    final_x = es.result.xbest
    
    return ExperimentResult(
        name="Death Penalty",
        f_history=f_history,
        g_history=g_history,
        x_history=x_history,
        eval_history=eval_history,
        final_x=final_x,
        final_f=objective(final_x),
        final_g=constraint(final_x),
        fevals=es.result.evaluations
    )


def run_adaptive_penalty(
    objective: Callable,
    constraint: Callable,
    x0: np.ndarray,
    sigma0: float,
    x_optimal: np.ndarray,
    beta_init: float = 1.0,
    kappa: float = 2.0,
    max_fevals: int = 5000,
    seed: int = None
) -> ExperimentResult:
    """Run CMA-ES with adaptive penalty."""
    fitness = AdaptivePenaltyFitness(objective, constraint, beta_init, kappa)
    
    opts = {
        'maxfevals': max_fevals,
        'verbose': -9,
        'verb_disp': 0,
        'verb_log': 0,
    }
    if seed is not None:
        opts['seed'] = seed
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    # Per-generation tracking
    f_history = []
    g_history = []
    x_history = []
    eval_history = []
    beta_history = [beta_init]
    
    while not es.stop():
        solutions = es.ask()
        values = [fitness(x) for x in solutions]
        es.tell(solutions, values)
        fitness.update(es)
        
        # Record best-of-generation
        best_idx = np.argmin(values)
        best_x = solutions[best_idx]
        f_history.append(objective(best_x))
        g_history.append(constraint(best_x))
        x_history.append(best_x.copy())
        eval_history.append(es.result.evaluations)
        beta_history.append(fitness.beta)
    
    final_x = es.result.xbest
    
    return ExperimentResult(
        name="Adaptive Penalty",
        f_history=f_history,
        g_history=g_history,
        x_history=x_history,
        eval_history=eval_history,
        final_x=final_x,
        final_f=objective(final_x),
        final_g=constraint(final_x),
        beta_history=beta_history,
        fevals=es.result.evaluations
    )


def run_al_cmaes(
    objective: Callable,
    constraint: Callable,
    x0: np.ndarray,
    sigma0: float,
    x_optimal: np.ndarray,
    max_fevals: int = 5000,
    seed: int = None
) -> ExperimentResult:
    """Run AL-CMA-ES (Augmented Lagrangian)."""
    def constraint_wrapper(x):
        return [constraint(x)]
    
    cfun = cma.ConstrainedFitnessAL(objective, constraint_wrapper, logging=False)
    
    opts = {
        'maxfevals': max_fevals,
        'verbose': -9,
        'verb_disp': 0,
        'verb_log': 0,
        'tolstagnation': 0,
    }
    if seed is not None:
        opts['seed'] = seed
    
    f_history = []
    g_history = []
    x_history = []
    eval_history = []
    gamma_history = []
    sigma_history = []
    condition_history = []
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    while not es.stop():
        solutions = es.ask()
        values = [cfun(x) for x in solutions]
        es.tell(solutions, values)
        cfun.update(es)
        
        # Track best in generation
        best_idx = np.argmin(values)
        best_x = solutions[best_idx]
        f_history.append(objective(best_x))
        g_history.append(constraint(best_x))
        x_history.append(best_x.copy())
        eval_history.append(es.result.evaluations)
        
        # Track Lagrange multiplier
        if hasattr(cfun, 'al') and hasattr(cfun.al, 'lam') and len(cfun.al.lam) > 0:
            gamma_history.append(float(cfun.al.lam[0]))
        
        # Track step-size σ and condition number
        sigma_history.append(es.sigma)
        
        try:
            D = es.sm.D if hasattr(es.sm, 'D') else es.D
            condition = (max(D) / min(D))**2 if min(D) > 0 else 1.0
            condition_history.append(condition)
        except:
            condition_history.append(1.0)
    
    final_x = es.result.xfavorite
    
    return ExperimentResult(
        name="AL-CMA-ES",
        f_history=f_history,
        g_history=g_history,
        x_history=x_history,
        eval_history=eval_history,
        final_x=final_x,
        final_f=objective(final_x),
        final_g=constraint(final_x),
        gamma_history=gamma_history,
        sigma_history=sigma_history,
        condition_history=condition_history,
        fevals=es.result.evaluations
    )


def run_surrogate_al(
    objective: Callable,
    constraint: Callable,
    x0: np.ndarray,
    sigma0: float,
    x_optimal: np.ndarray,
    max_fevals: int = 5000,
    seed: int = None,
    archive_size: int = 30
) -> ExperimentResult:
    """Run Surrogate-Assisted AL-CMA-ES."""
    sur = SurrogateALFitness(objective, constraint, archive_size)
    
    def surrogate_constraint_wrapper(x):
        return [sur.surrogate_constraint(x)]
    
    cfun = cma.ConstrainedFitnessAL(objective, surrogate_constraint_wrapper, logging=False)
    
    opts = {
        'maxfevals': max_fevals,
        'verbose': -9,
        'verb_disp': 0,
        'verb_log': 0,
        'tolstagnation': 0,
    }
    if seed is not None:
        opts['seed'] = seed
    
    f_history = []
    g_history = []
    x_history = []
    eval_history = []
    gamma_history = []
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    while not es.stop():
        solutions = es.ask()
        values = [cfun(x) for x in solutions]
        
        # Update archive with true constraint values
        for x in solutions:
            g_true = constraint(x)
            sur.update_archive(x, g_true)
        
        es.tell(solutions, values)
        cfun.update(es)
        
        # Track best in generation
        best_idx = np.argmin(values)
        best_x = solutions[best_idx]
        f_history.append(objective(best_x))
        g_history.append(constraint(best_x))
        x_history.append(best_x.copy())
        eval_history.append(es.result.evaluations)
        
        if hasattr(cfun, 'al') and hasattr(cfun.al, 'lam') and len(cfun.al.lam) > 0:
            gamma_history.append(float(cfun.al.lam[0]))
    
    final_x = es.result.xfavorite  # Use current mean for AL methods
    
    return ExperimentResult(
        name="Surrogate-AL",
        f_history=f_history,
        g_history=g_history,
        x_history=x_history,
        eval_history=eval_history,
        final_x=final_x,
        final_f=objective(final_x),
        final_g=constraint(final_x),
        gamma_history=gamma_history,
        fevals=es.result.evaluations
    )


# =============================================================================
# Multi-Run Aggregation
# =============================================================================

def run_multiple(
    runner_func: Callable,
    runner_kwargs: dict,
    x_optimal: np.ndarray,
    num_runs: int = NUM_RUNS,
    name: str = "Algorithm"
) -> AggregatedResult:
    """Run an algorithm multiple times and aggregate results."""
    distances_all = []
    eval_histories_all = []
    final_f_all = []
    final_g_all = []
    fevals_all = []
    gamma_histories = []
    
    for run in range(num_runs):
        kwargs = runner_kwargs.copy()
        kwargs['seed'] = SEED_BASE + run
        
        result = runner_func(**kwargs)
        
        # Compute distance to optimal for each point in history
        distances = [np.linalg.norm(np.array(x) - x_optimal) for x in result.x_history]
        distances_all.append(distances)
        eval_histories_all.append(result.eval_history)
        
        final_f_all.append(result.final_f)
        final_g_all.append(result.final_g)
        fevals_all.append(result.fevals)
        
        if hasattr(result, 'gamma_history'):
            gamma_histories.append(result.gamma_history)
    
    return AggregatedResult(
        name=name,
        distances_all=distances_all,
        eval_histories_all=eval_histories_all,
        final_f_all=final_f_all,
        final_g_all=final_g_all,
        fevals_all=fevals_all,
        gamma_histories=gamma_histories
    )


def compute_median_trajectory(distances_all: List[List[float]], max_len: int = None) -> np.ndarray:
    """Compute median trajectory across runs, handling different lengths."""
    if max_len is None:
        max_len = max(len(d) for d in distances_all)
    
    # Pad shorter runs with their final value
    padded = []
    for d in distances_all:
        if len(d) < max_len:
            pad = d + [d[-1]] * (max_len - len(d))
        else:
            pad = d[:max_len]
        padded.append(pad)
    
    arr = np.array(padded)
    return np.median(arr, axis=0)


# =============================================================================
# Visualization Functions
# =============================================================================

def ensure_output_dir(subdir: str = '') -> str:
    """Create output directory."""
    output_dir = 'results/alcmaes_comparison'
    if subdir:
        output_dir = f'{output_dir}/{subdir}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_convergence_median(
    agg_results: List[AggregatedResult],
    title: str,
    filename: str,
    output_dir: str
):
    """Plot median convergence trajectories."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
    
    max_len = max(max(len(d) for d in r.distances_all) for r in agg_results)
    
    for idx, agg in enumerate(agg_results):
        # Use first run's eval_history as representative x-axis
        # (all runs have same algorithm so similar evaluation counts)
        if len(agg.eval_histories_all) > 0 and len(agg.eval_histories_all[0]) > 0:
            eval_hist = agg.eval_histories_all[0]
        else:
            # Fallback to index if eval_history not available
            eval_hist = list(range(len(agg.distances_all[0])))
        
        median_traj = compute_median_trajectory(agg.distances_all, len(eval_hist))
        
        # Subsample for cleaner plot
        step = max(1, len(median_traj) // 500)
        indices = list(range(0, len(median_traj), step))
        x_vals = [eval_hist[i] if i < len(eval_hist) else eval_hist[-1] for i in indices]
        y_vals = [median_traj[i] for i in indices]
        
        color = colors[idx % len(colors)]
        ax.semilogy(x_vals, y_vals, '-', color=color, linewidth=2, 
                    label=f"{agg.name} (n={len(agg.distances_all)})", alpha=0.9)
    
    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel('Distance to Optimal ||x - x*|| (median)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()


def plot_single_run_dynamics(
    result: ExperimentResult,
    x_optimal: np.ndarray,
    title: str,
    filename: str,
    output_dir: str
):
    """Plot dynamics for a single run (gamma, beta evolution)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Panel 1: Distance to optimal
    ax1 = axes[0]
    distances = [np.linalg.norm(x - x_optimal) for x in result.x_history]
    step = max(1, len(distances) // 500)
    indices = list(range(0, len(distances), step))

    # Prefer objective evaluation counts when available.
    if len(result.eval_history) == len(distances) and len(result.eval_history) > 0:
        x_vals = [result.eval_history[i] for i in indices]
        x_label = 'Function Evaluations'
    else:
        x_vals = indices
        x_label = 'Generation'

    y_vals = [distances[i] for i in indices]
    ax1.semilogy(x_vals, y_vals, 'b-', linewidth=2)
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel('||x - x*||', fontsize=12)
    ax1.set_title('Distance to Optimal')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: γ (Lagrange multiplier) or β (penalty coefficient)
    ax2 = axes[1]
    if len(result.gamma_history) > 0:
        ax2.plot(result.gamma_history, 'g-', linewidth=2, label='γ (Lagrange mult.)')
        ax2.set_ylabel('γ', fontsize=12)
        ax2.set_title('Lagrange Multiplier Evolution')
    elif len(result.beta_history) > 0:
        ax2.semilogy(result.beta_history, 'r-', linewidth=2, label='β (penalty coeff.)')
        ax2.set_ylabel('β', fontsize=12)
        ax2.set_title('Penalty Coefficient Evolution')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()


def plot_constraint_function(
    constraint: Callable,
    x_range: tuple,
    title: str,
    filename: str,
    output_dir: str,
    n_dims: int = 5
):
    """Visualize constraint function in 1D."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x1_vals = np.linspace(x_range[0], x_range[1], 200)
    g_vals = [constraint(np.array([x1] + [0]*(n_dims-1))) for x1 in x1_vals]
    
    ax.plot(x1_vals, g_vals, 'k-', linewidth=3, label='g(x)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='red', linestyle=':', linewidth=2, label='Boundary')
    
    # Mark the kink
    ax.plot(0, 0, 'ro', markersize=12, label='Kink (gradient discontinuity)')
    
    # Shade regions
    feasible_x = x1_vals[x1_vals <= 0]
    feasible_g = [g_vals[i] for i, x in enumerate(x1_vals) if x <= 0]
    ax.fill_between(feasible_x, 0, feasible_g, alpha=0.2, color='green', label='Feasible (g≤0)')
    
    infeasible_x = x1_vals[x1_vals > 0]
    infeasible_g = [max(0, g_vals[i]) for i, x in enumerate(x1_vals) if x > 0]
    ax.fill_between(infeasible_x, 0, infeasible_g, alpha=0.2, color='red', label='Infeasible (g>0)')
    
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('g(x)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_range)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()


def print_results_table(agg_results: List[AggregatedResult], x_optimal: np.ndarray):
    """Print results as text table (no image)."""
    print("\n" + "-" * 80)
    print(f"{'Algorithm':<25} {'f(x) [median]':>15} {'g(x) [median]':>15} {'Feasible':>10} {'FEvals':>10}")
    print("-" * 80)
    
    for agg in agg_results:
        median_f = np.median(agg.final_f_all)
        median_g = np.median(agg.final_g_all)
        median_fevals = int(np.median(agg.fevals_all))
        feasible_rate = sum(1 for g in agg.final_g_all if g <= 1e-4) / len(agg.final_g_all)
        
        print(f"{agg.name:<25} {median_f:>15.6f} {median_g:>15.6f} {feasible_rate*100:>9.0f}% {median_fevals:>10}")
    
    print("-" * 80)


def plot_internal_state(
    result: ExperimentResult,
    x_optimal: np.ndarray,
    title: str,
    filename: str,
    output_dir: str
):
    """
    Plot internal state dynamics: σ (step-size) and condition number.
    Reference: Atamna et al. (2016) analyze step-size for linear convergence proof.
    """
    has_sigma = len(result.sigma_history) > 0
    has_condition = len(result.condition_history) > 0
    has_gamma = len(result.gamma_history) > 0
    
    n_panels = 1 + has_sigma + has_condition + has_gamma
    fig, axes = plt.subplots(1, n_panels, figsize=(4*n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    panel_idx = 0
    
    # Panel 1: Distance to optimal
    ax = axes[panel_idx]
    distances = [np.linalg.norm(x - x_optimal) for x in result.x_history]
    ax.semilogy(distances, 'b-', linewidth=2)
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('||x - x*||', fontsize=11)
    ax.set_title('Distance to Optimal')
    ax.grid(True, alpha=0.3)
    panel_idx += 1
    
    # Panel 2: Step-size σ
    if has_sigma:
        ax = axes[panel_idx]
        ax.semilogy(result.sigma_history, 'orange', linewidth=2)
        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('σ (step-size)', fontsize=11)
        ax.set_title('Step-Size Evolution')
        ax.grid(True, alpha=0.3)
        panel_idx += 1
    
    # Panel 3: Condition number
    if has_condition:
        ax = axes[panel_idx]
        ax.semilogy(result.condition_history, 'purple', linewidth=2)
        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('Condition Number', fontsize=11)
        ax.set_title('Covariance Condition Number')
        ax.grid(True, alpha=0.3)
        panel_idx += 1
    
    # Panel 4: Lagrange multiplier γ
    if has_gamma:
        ax = axes[panel_idx]
        ax.plot(result.gamma_history, 'green', linewidth=2)
        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('γ', fontsize=11)
        ax.set_title('Lagrange Multiplier γ')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()


def plot_ecdf(
    agg_results: List[AggregatedResult],
    target_distance: float,
    title: str,
    filename: str,
    output_dir: str
):
    """
    Plot ECDF (Empirical Cumulative Distribution Function).
    Shows probability of reaching target distance within budget.
    Reference: Standard BBOB/COCO benchmarking methodology.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
    
    for idx, agg in enumerate(agg_results):
        # For each run, find first time distance < target
        fevals_to_target = []
        
        for distances in agg.distances_all:
            # Find first index where distance < target
            found = False
            for i, d in enumerate(distances):
                if d < target_distance:
                    fevals_to_target.append(i)
                    found = True
                    break
            if not found:
                fevals_to_target.append(np.inf)  # Never reached
        
        # Compute ECDF
        sorted_fevals = np.sort(fevals_to_target)
        ecdf_y = np.arange(1, len(sorted_fevals) + 1) / len(sorted_fevals)
        
        # Filter out inf values for plotting
        valid_mask = sorted_fevals < np.inf
        if np.any(valid_mask):
            color = colors[idx % len(colors)]
            ax.step(sorted_fevals[valid_mask], ecdf_y[valid_mask], 
                   where='post', color=color, linewidth=2, label=agg.name)
    
    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel(f'Proportion reaching ||x-x*|| < {target_distance}', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()


def plot_gamma_oscillation(
    agg_result: AggregatedResult,
    title: str,
    filename: str,
    output_dir: str
):
    """
    Plot Lagrange multiplier γ evolution to detect oscillation.
    Reference: Dufossé & Hansen (2021), Figure 4 - γ oscillates between 1 and 2.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(agg_result.gamma_histories)))
    
    for idx, gamma_hist in enumerate(agg_result.gamma_histories):
        if len(gamma_hist) > 0:
            ax.plot(gamma_hist, '-', color=colors[idx], linewidth=1.5, 
                   alpha=0.7, label=f'Run {idx+1}' if idx < 3 else None)
    
    # Add theoretical oscillation bounds based on KKT condition
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Lower bound (λ=1, slope=2)')
    ax.axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Upper bound (λ=2, slope=1)')
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('γ (Lagrange Multiplier)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()


def plot_constraint_violation(
    results: List[ExperimentResult],
    title: str,
    filename: str,
    output_dir: str
):
    """
    Plot constraint violation g(x) vs. generations for multiple methods.
    
    Shows how different methods handle constraints:
    - Death Penalty: Violation always ≈ 0 (infeasible solutions rejected)
    - Static Penalty: Violation decreases slowly
    - AL-CMA-ES: May go slightly infeasible initially, then pulls back
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']
    
    for idx, result in enumerate(results):
        g_hist = result.g_history
        if len(g_hist) == 0:
            continue
        
        # Use eval_history for x-axis if available
        if len(result.eval_history) > 0:
            x_vals = result.eval_history
        else:
            x_vals = list(range(len(g_hist)))
        
        # Subsample for cleaner plot
        step = max(1, len(g_hist) // 300)
        x_sub = [x_vals[i] for i in range(0, len(x_vals), step)]
        g_sub = [g_hist[i] for i in range(0, len(g_hist), step)]
        
        color = colors[idx % len(colors)]
        ax.plot(x_sub, g_sub, '-', color=color, linewidth=2, 
                label=result.name, alpha=0.8)
    
    # Add feasibility boundary
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, 
               label='Feasibility boundary (g=0)')
    
    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel('Constraint Value g(x)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Shade infeasible region
    ax.axhspan(0, ax.get_ylim()[1], alpha=0.1, color='red', label='_Infeasible')
    ax.axhspan(ax.get_ylim()[0], 0, alpha=0.1, color='green', label='_Feasible')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()


def plot_2d_trajectory(
    results: List[ExperimentResult],
    objective: Callable,
    constraint_boundary: float,
    title: str,
    filename: str,
    output_dir: str,
    x_range: tuple = (-1, 3),
    y_range: tuple = (-2, 2)
):
    """Plot 2D contour with search trajectories."""
    n_results = len(results)
    n_cols = min(3, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_results == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    # Pre-compute contour grid
    x = np.linspace(x_range[0], x_range[1], 80)
    y = np.linspace(y_range[0], y_range[1], 80)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.zeros(10)
            point[0] = X[j, i]
            point[1] = Y[j, i]
            Z[j, i] = objective(point)
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']
    
    for idx, result in enumerate(results):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        # Plot contours
        levels = np.logspace(-1, 1.5, 12)
        ax.contourf(X, Y, Z, levels=levels, cmap='Blues', alpha=0.4)
        contour = ax.contour(X, Y, Z, levels=levels, colors='steelblue', alpha=0.6, linewidths=0.5)
        
        # Shade infeasible region
        ax.axvspan(x_range[0], constraint_boundary, alpha=0.2, color='red')
        
        # Constraint boundary
        ax.axvline(x=constraint_boundary, color='red', linewidth=2, linestyle='--')
        
        if len(result.x_history) > 0:
            # Extract trajectory
            x_traj = [pt[0] for pt in result.x_history]
            y_traj = [pt[1] if len(pt) > 1 else 0 for pt in result.x_history]
            
            color = colors[idx % len(colors)]
            
            # Plot trajectory with color gradient (early=light, late=dark)
            n_points = len(x_traj)
            for i in range(n_points - 1):
                alpha = 0.3 + 0.7 * (i / n_points)
                ax.plot([x_traj[i], x_traj[i+1]], [y_traj[i], y_traj[i+1]], 
                       '-', color=color, linewidth=2, alpha=alpha)
            
            # Mark start (circle) and end (star)
            ax.scatter(x_traj[0], y_traj[0], s=150, c='white', marker='o', 
                      edgecolors=color, linewidths=3, zorder=10, label='Start')
            ax.scatter(x_traj[-1], y_traj[-1], s=250, c=color, marker='*', 
                      edgecolors='black', linewidths=1.5, zorder=11, label='End')
        
        # Optimal point
        ax.scatter(constraint_boundary, 0, s=200, c='gold', marker='*', 
                  edgecolors='black', linewidths=2, zorder=15)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('x₀', fontsize=10)
        ax.set_ylabel('x₁', fontsize=10)
        ax.set_title(result.name, fontsize=12, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        
        # Add annotation
        if len(result.x_history) > 0:
            final_g = result.final_g
            status = "Feasible ✓" if final_g <= 0 else f"Infeasible (g={final_g:.2f})"
            ax.text(0.02, 0.98, status, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused axes
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# PART 1: Baseline Comparison
# =============================================================================

def run_part1_baseline():
    """Part 1: Sphere + Linear Constraint (x₁ >= 1)"""
    print("\n" + "=" * 80)
    print("PART 1: Baseline Comparison")
    print("=" * 80)
    
    n = 10
    x0 = np.zeros(n)
    sigma0 = 1.0
    max_fevals = 10000
    
    x_optimal = np.zeros(n)
    x_optimal[0] = 1.0
    
    constraint = lambda x: linear_constraint_x1_ge_c(x, c=1.0)
    
    output_dir = ensure_output_dir('part1_baseline')
    
    # Common kwargs
    base_kwargs = {
        'objective': sphere,
        'constraint': constraint,
        'x0': x0,
        'sigma0': sigma0,
        'x_optimal': x_optimal,
        'max_fevals': max_fevals,
    }
    
    agg_results = []
    
    # 1. Vanilla CMA-ES (ignores constraints - shows why we need constraint handling)
    print("\n[1/6] Vanilla CMA-ES (no constraint handling)")
    agg_vanilla = run_multiple(
        run_vanilla_cmaes,
        base_kwargs,
        x_optimal,
        name="Vanilla (no CH)"
    )
    agg_results.append(agg_vanilla)
    
    # 2. Death Penalty
    print("[2/6] Death Penalty")
    agg_death = run_multiple(
        run_death_penalty,
        base_kwargs,
        x_optimal,
        name="Death Penalty"
    )
    agg_results.append(agg_death)
    
    # 3. Static Penalty (Low) - expected infeasible
    print("[3/6] Static Penalty (β=1)")
    agg_low = run_multiple(
        run_static_penalty,
        {**base_kwargs, 'beta': 1.0},
        x_optimal,
        name="Static (β=1)"
    )
    agg_results.append(agg_low)
    
    # 4. Adaptive Penalty (Dufossé & Hansen, 2021)
    print("[4/6] Adaptive Penalty")
    agg_adaptive = run_multiple(
        run_adaptive_penalty,
        base_kwargs,
        x_optimal,
        name="Adaptive Penalty"
    )
    agg_results.append(agg_adaptive)
    
    # 5. AL-CMA-ES (Atamna et al., 2016)
    print("[5/6] AL-CMA-ES")
    agg_al = run_multiple(
        run_al_cmaes,
        base_kwargs,
        x_optimal,
        name="AL-CMA-ES"
    )
    agg_results.append(agg_al)
    
    # Print results table
    print_results_table(agg_results, x_optimal)
    
    # Generate plots
    plot_convergence_median(
        agg_results,
        title="Part 1: Convergence Comparison (Sphere + Linear Constraint)",
        filename="01_convergence_median.png",
        output_dir=output_dir
    )
    
    # Single run dynamics for AL-CMA-ES
    single_result = run_al_cmaes(**base_kwargs, seed=SEED_BASE)
    plot_single_run_dynamics(
        single_result, x_optimal,
        title="AL-CMA-ES Dynamics (Single Run)",
        filename="02_al_dynamics.png",
        output_dir=output_dir
    )
    
    # Internal state dynamics (σ, condition number, γ)
    plot_internal_state(
        single_result, x_optimal,
        title="AL-CMA-ES Internal State (σ, Condition Number, γ)",
        filename="03_internal_state.png",
        output_dir=output_dir
    )
    
    # Single run dynamics for Adaptive Penalty
    single_adaptive = run_adaptive_penalty(**base_kwargs, seed=SEED_BASE)
    plot_single_run_dynamics(
        single_adaptive, x_optimal,
        title="Adaptive Penalty Dynamics (Single Run)",
        filename="04_adaptive_dynamics.png",
        output_dir=output_dir
    )
    
    # =========================================================================
    # SENSITIVITY ANALYSIS: β values from 10⁰ to 10⁹
    # =========================================================================
    print("\n--- Sensitivity Analysis ---")
    
    beta_values = [1, 1e3, 1e6, 1e9]
    sensitivity_results = []
    
    for beta in beta_values:
        result = run_static_penalty(**base_kwargs, beta=beta, seed=SEED_BASE)
        sensitivity_results.append(result)
    
    # Add AL-CMA-ES for comparison
    al_result = run_al_cmaes(**base_kwargs, seed=SEED_BASE)
    
    # Plot sensitivity analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sensitivity Analysis: Static Penalty β vs AL-CMA-ES', fontsize=14, fontweight='bold')
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    
    # Panel 1: Convergence for different β
    ax1 = axes[0]
    for idx, (beta, result) in enumerate(zip(beta_values, sensitivity_results)):
        distances = [np.linalg.norm(x - x_optimal) for x in result.x_history]
        step = max(1, len(distances) // 300)
        x_vals = np.arange(0, len(distances), step)
        y_vals = [max(distances[i], 1e-10) for i in x_vals]
        label = f'β=10^{int(np.log10(beta))}' if beta > 1 else 'β=1'
        ax1.semilogy(x_vals, y_vals, '-', color=colors[idx], linewidth=2, label=label, alpha=0.8)
    
    # AL-CMA-ES
    al_distances = [np.linalg.norm(x - x_optimal) for x in al_result.x_history]
    step = max(1, len(al_distances) // 300)
    x_vals = np.arange(0, len(al_distances), step)
    y_vals = [max(al_distances[i], 1e-10) for i in x_vals]
    ax1.semilogy(x_vals, y_vals, 'k-', linewidth=3, label='AL-CMA-ES', alpha=0.9)
    
    ax1.set_xlabel('Function Evaluations', fontsize=12)
    ax1.set_ylabel('||x - x*||', fontsize=12)
    ax1.set_title('Distance to Optimal vs Penalty Coefficient')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Final distance vs β (bar chart)
    ax2 = axes[1]
    labels = [f'β=10^{int(np.log10(b))}' if b > 1 else 'β=1' for b in beta_values] + ['AL-CMA-ES']
    final_dists = [np.linalg.norm(r.final_x - x_optimal) for r in sensitivity_results]
    final_dists.append(np.linalg.norm(al_result.final_x - x_optimal))
    
    bar_colors = colors + ['black']
    bars = ax2.bar(labels, final_dists, color=bar_colors, alpha=0.8)
    ax2.set_ylabel('Final ||x - x*||', fontsize=12)
    ax2.set_title('Final Distance vs Penalty Approach')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add feasibility annotation
    for idx, bar in enumerate(bars):
        g = sensitivity_results[idx].final_g if idx < len(sensitivity_results) else al_result.final_g
        status = '✓' if g <= 1e-4 else '✗'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), status, 
                ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Constraint Violation Plot
    single_runs = [
        run_vanilla_cmaes(**base_kwargs, seed=SEED_BASE),
        run_death_penalty(**base_kwargs, seed=SEED_BASE),
        run_static_penalty(**base_kwargs, beta=1.0, seed=SEED_BASE),
        run_adaptive_penalty(**base_kwargs, seed=SEED_BASE),
        run_al_cmaes(**base_kwargs, seed=SEED_BASE)
    ]
    
    plot_constraint_violation(
        single_runs,
        title="Constraint Violation g(x) Over Search",
        filename="06_constraint_violation.png",
        output_dir=output_dir
    )
    
    # 2D Trajectory Plot
    
    plot_2d_trajectory(
        single_runs,
        objective=sphere,
        constraint_boundary=1.0,
        title="2D Search Trajectory: Death Penalty vs AL-CMA-ES (x₀, x₁ slice)",
        filename="07_2d_trajectory.png",
        output_dir=output_dir,
        x_range=(-0.5, 2.5),
        y_range=(-1.5, 1.5)
    )
    
    print(f"\nPlots saved to {output_dir}/")
    
    return agg_results


# =============================================================================
# PART 2: Ill-Conditioned Scenario (Ellipsoid)
# =============================================================================

def run_part2_illconditioned():
    """Part 2: Ellipsoid + Linear Constraint (x₁ >= 10)"""
    print("\n" + "=" * 80)
    print("PART 2: Ill-Conditioned Scenario (Ellipsoid)")
    print("=" * 80)
    
    n = 10
    x0 = np.zeros(n)
    sigma0 = 5.0
    max_fevals = 20000
    
    x_optimal = np.zeros(n)
    x_optimal[0] = 10.0
    
    constraint = lambda x: linear_constraint_x1_ge_c(x, c=10.0)
    
    print(f"\nProblem: min ellipsoid(x) s.t. x₁ >= 10")
    print(f"Condition number: 10^6")
    print(f"Number of runs: {NUM_RUNS}")
    
    output_dir = ensure_output_dir('part2_illconditioned')
    
    base_kwargs = {
        'objective': ellipsoid,
        'constraint': constraint,
        'x0': x0,
        'sigma0': sigma0,
        'x_optimal': x_optimal,
        'max_fevals': max_fevals,
    }
    
    agg_results = []
    
    # 1. Adaptive Penalty
    print("\n[1/2] Adaptive Penalty...")
    agg_adaptive = run_multiple(
        run_adaptive_penalty,
        base_kwargs,
        x_optimal,
        name="Adaptive Penalty"
    )
    agg_results.append(agg_adaptive)
    
    # 2. AL-CMA-ES
    print("[2/2] AL-CMA-ES...")
    agg_al = run_multiple(
        run_al_cmaes,
        base_kwargs,
        x_optimal,
        name="AL-CMA-ES"
    )
    agg_results.append(agg_al)
    
    # Print results
    print_results_table(agg_results, x_optimal)
    
    # Generate convergence plot
    plot_convergence_median(
        agg_results,
        title="Part 2: Ill-Conditioned Ellipsoid (Condition Number = 10⁶)",
        filename="01_convergence_ellipsoid.png",
        output_dir=output_dir
    )
    # Internal state dynamics
    single_al = run_al_cmaes(**base_kwargs, seed=SEED_BASE)
    plot_internal_state(
        single_al, x_optimal,
        title="AL-CMA-ES on Ellipsoid: Internal State",
        filename="02_internal_state_al.png",
        output_dir=output_dir
    )
    
    return agg_results


# =============================================================================
# PART 3: Non-Smooth Constraint
# =============================================================================

def run_part3_nonsmooth():
    """Part 3: Piecewise constraint with gradient discontinuity."""
    print("\n" + "=" * 80)
    print("PART 3: Non-Smooth Constraint")
    print("=" * 80)
    
    n = 5
    x0 = np.ones(n) * 0.5
    sigma0 = 0.5
    max_fevals = 10000
    
    x_optimal = np.zeros(n)
    
    output_dir = ensure_output_dir('part3_nonsmooth')
    
    base_kwargs = {
        'objective': shifted_sphere,
        'constraint': piecewise_constraint_kink,
        'x0': x0,
        'sigma0': sigma0,
        'x_optimal': x_optimal,
        'max_fevals': max_fevals,
    }
    
    agg_results = []
    
    # 1. Adaptive Penalty
    print("\n[1/3] Adaptive Penalty...")
    agg_adaptive = run_multiple(
        run_adaptive_penalty,
        base_kwargs,
        x_optimal,
        name="Adaptive Penalty"
    )
    agg_results.append(agg_adaptive)
    
    # 2. AL-CMA-ES
    print("[2/3] AL-CMA-ES...")
    agg_al = run_multiple(
        run_al_cmaes,
        base_kwargs,
        x_optimal,
        name="AL-CMA-ES"
    )
    agg_results.append(agg_al)
    
    # 3. Surrogate-AL
    print("[3/3] Surrogate-AL...")
    agg_surrogate = run_multiple(
        run_surrogate_al,
        base_kwargs,
        x_optimal,
        name="Surrogate-AL"
    )
    agg_results.append(agg_surrogate)
    
    # Print results
    print_results_table(agg_results, x_optimal)
    
    # Generate plots
    plot_convergence_median(
        agg_results,
        title="Part 3: Non-Smooth Constraint (Standard AL vs Surrogate-AL)",
        filename="01_convergence_nonsmooth.png",
        output_dir=output_dir
    )
    
    # Visualize the constraint function
    plot_constraint_function(
        piecewise_constraint_kink,
        x_range=(-2, 3),
        title="Piecewise Constraint",
        filename="02_constraint_function.png",
        output_dir=output_dir,
        n_dims=n
    )
    
    # Gamma oscillation for Standard AL
    if len(agg_al.gamma_histories) > 0:
        plot_gamma_oscillation(
            agg_al,
            title="Standard AL: γ Oscillation",
            filename="03_gamma_oscillation_al.png",
            output_dir=output_dir
        )
    
    # Gamma for Surrogate AL
    if len(agg_surrogate.gamma_histories) > 0:
        plot_gamma_oscillation(
            agg_surrogate,
            title="Surrogate-AL: γ Convergence",
            filename="04_gamma_surrogate.png",
            output_dir=output_dir
        )
    
    # ECDF
    plot_ecdf(
        agg_results,
        target_distance=1e-2,
        title="Part 3: ECDF",
        filename="05_ecdf.png",
        output_dir=output_dir
    )
    
    return agg_results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("AL-CMA-ES COMPARISON STUDY")
    print("=" * 80)
    
    part1_results = run_part1_baseline()
    part2_results = run_part2_illconditioned()
    part3_results = run_part3_nonsmooth()
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    
    return {
        'part1': part1_results,
        'part2': part2_results,
        'part3': part3_results
    }


if __name__ == "__main__":
    results = main()
