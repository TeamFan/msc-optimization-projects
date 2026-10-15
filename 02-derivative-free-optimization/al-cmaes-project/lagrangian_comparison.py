"""
Comparison: Standard Lagrangian (Dual Ascent) vs Augmented Lagrangian

Two-stage comparison for constrained optimization:
    Case A: Convex Quadratic Programming
    Case B: Non-Convex Double Well
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, List, Optional
import os


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TestProblem:
    """Constrained optimization test problem container."""
    name: str
    objective: Callable[[np.ndarray], float]
    grad_objective: Callable[[np.ndarray], np.ndarray]
    constraint: Callable[[np.ndarray], float]
    grad_constraint: Callable[[np.ndarray], np.ndarray]
    x0: np.ndarray
    x_optimal: np.ndarray
    f_optimal: float
    lambda_optimal: float
    alpha_x: float
    alpha_lambda: float
    rho: float
    x_range: tuple
    y_range: tuple
    description: str


@dataclass
class OptimizationResult:
    """Optimization results container."""
    x_history: List[np.ndarray]
    lambda_history: List[float]
    f_history: List[float]
    constraint_history: List[float]
    converged: bool
    iterations: int
    final_x: np.ndarray
    final_lambda: float
    final_f: float
    final_constraint: float


# =============================================================================
# Test Problems
# =============================================================================

def create_convex_qp() -> TestProblem:
    """Convex QP: minimize (x1-2)² + (x2-1)², subject to x1 + x2 = 1"""
    def objective(x):
        return (x[0] - 2)**2 + (x[1] - 1)**2
    
    def grad_objective(x):
        return np.array([2*(x[0] - 2), 2*(x[1] - 1)])
    
    def constraint(x):
        return x[0] + x[1] - 1
    
    def grad_constraint(x):
        return np.array([1.0, 1.0])
    
    return TestProblem(
        name="Convex QP",
        objective=objective,
        grad_objective=grad_objective,
        constraint=constraint,
        grad_constraint=grad_constraint,
        x0=np.array([0.0, 0.0]),
        x_optimal=np.array([1.0, 0.0]),
        f_optimal=2.0,
        lambda_optimal=2.0,
        alpha_x=0.1,
        alpha_lambda=0.3,
        rho=1.0,
        x_range=(-1, 4),
        y_range=(-3, 3),
        description="Convex QP: Both methods work, ALM is faster"
    )


def create_double_well() -> TestProblem:
    """Non-Convex Double Well: minimize (x1² - 1)² + x2², subject to x1 = 0"""
    def objective(x):
        return (x[0]**2 - 1)**2 + x[1]**2
    
    def grad_objective(x):
        return np.array([4*x[0]*(x[0]**2 - 1), 2*x[1]])
    
    def constraint(x):
        return x[0]
    
    def grad_constraint(x):
        return np.array([1.0, 0.0])
    
    return TestProblem(
        name="Non-Convex Double Well",
        objective=objective,
        grad_objective=grad_objective,
        constraint=constraint,
        grad_constraint=grad_constraint,
        x0=np.array([1.5, 0.5]),
        x_optimal=np.array([0.0, 0.0]),
        f_optimal=1.0,
        lambda_optimal=0.0,
        alpha_x=0.01,
        alpha_lambda=0.1,
        rho=10.0,
        x_range=(-2, 2),
        y_range=(-1.5, 1.5),
        description="Double Well: Standard FAILS, ALM succeeds via convexification"
    )


# =============================================================================
# Optimization Algorithms
# =============================================================================

def dual_ascent(
    problem: TestProblem,
    alpha_x: Optional[float] = None,
    alpha_lambda: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False
) -> OptimizationResult:
    """Standard Lagrangian with Dual Ascent: L(x, λ) = f(x) + λ * g(x)"""
    alpha_x = alpha_x or problem.alpha_x
    alpha_lambda = alpha_lambda or problem.alpha_lambda
    
    x = problem.x0.copy()
    lam = 0.0
    
    x_history = [x.copy()]
    lambda_history = [lam]
    f_history = [problem.objective(x)]
    constraint_history = [problem.constraint(x)]
    
    for k in range(max_iter):
        grad_L = problem.grad_objective(x) + lam * problem.grad_constraint(x)
        x_new = x - alpha_x * grad_L
        
        g_x = problem.constraint(x_new)
        lam_new = lam + alpha_lambda * g_x
        
        x_history.append(x_new.copy())
        lambda_history.append(lam_new)
        f_history.append(problem.objective(x_new))
        constraint_history.append(g_x)
        
        if np.linalg.norm(x_new - x) < tol and abs(g_x) < tol:
            if verbose:
                print(f"  [Dual Ascent] Converged at iteration {k+1}")
            return OptimizationResult(
                x_history=x_history, lambda_history=lambda_history,
                f_history=f_history, constraint_history=constraint_history,
                converged=True, iterations=k+1,
                final_x=x_new, final_lambda=lam_new,
                final_f=problem.objective(x_new), final_constraint=g_x
            )
        
        x = x_new
        lam = lam_new
    
    if verbose:
        print(f"  [Dual Ascent] Did not converge in {max_iter} iterations")
    
    return OptimizationResult(
        x_history=x_history, lambda_history=lambda_history,
        f_history=f_history, constraint_history=constraint_history,
        converged=False, iterations=max_iter,
        final_x=x, final_lambda=lam,
        final_f=problem.objective(x), final_constraint=problem.constraint(x)
    )


def augmented_lagrangian(
    problem: TestProblem,
    alpha_x: Optional[float] = None,
    rho: Optional[float] = None,
    inner_iter: int = 50,
    max_outer_iter: int = 50,
    tol: float = 1e-6,
    verbose: bool = False
) -> OptimizationResult:
    """Augmented Lagrangian Method: L_A(x, λ, ρ) = f(x) + λ * g(x) + (ρ/2) * g(x)²"""
    alpha_x = alpha_x or problem.alpha_x
    rho = rho or problem.rho
    
    x = problem.x0.copy()
    lam = 0.0
    
    x_history = [x.copy()]
    lambda_history = [lam]
    f_history = [problem.objective(x)]
    constraint_history = [problem.constraint(x)]
    
    for outer_k in range(max_outer_iter):
        for inner_k in range(inner_iter):
            g_x = problem.constraint(x)
            grad_L_A = problem.grad_objective(x) + (lam + rho * g_x) * problem.grad_constraint(x)
            x_new = x - alpha_x * grad_L_A
            
            if np.linalg.norm(grad_L_A) < tol * 0.1:
                break
            x = x_new
        
        g_x = problem.constraint(x)
        lam_new = lam + rho * g_x
        
        x_history.append(x.copy())
        lambda_history.append(lam_new)
        f_history.append(problem.objective(x))
        constraint_history.append(g_x)
        
        if abs(g_x) < tol and abs(lam_new - lam) < tol:
            if verbose:
                print(f"  [Aug. Lagrangian] Converged at outer iteration {outer_k+1}")
            return OptimizationResult(
                x_history=x_history, lambda_history=lambda_history,
                f_history=f_history, constraint_history=constraint_history,
                converged=True, iterations=outer_k+1,
                final_x=x, final_lambda=lam_new,
                final_f=problem.objective(x), final_constraint=g_x
            )
        
        lam = lam_new
    
    if verbose:
        print(f"  [Aug. Lagrangian] Did not converge in {max_outer_iter} outer iterations")
    
    return OptimizationResult(
        x_history=x_history, lambda_history=lambda_history,
        f_history=f_history, constraint_history=constraint_history,
        converged=False, iterations=max_outer_iter,
        final_x=x, final_lambda=lam,
        final_f=problem.objective(x), final_constraint=problem.constraint(x)
    )


# =============================================================================
# Visualization Utilities
# =============================================================================

def ensure_output_dir(subdir: str = ''):
    """Create output directory."""
    output_dir = 'results/lagrangian_comparison'
    if subdir:
        output_dir = f'{output_dir}/{subdir}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# =============================================================================
# Case A: Convex QP Visualization
# =============================================================================

def plot_case_a_trajectory(problem: TestProblem, result_da: OptimizationResult, 
                           result_al: OptimizationResult):
    """Case A: Trajectory comparison in design space."""
    output_dir = ensure_output_dir('case_a_convex')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x1 = np.linspace(problem.x_range[0], problem.x_range[1], 150)
    x2 = np.linspace(problem.y_range[0], problem.y_range[1], 150)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = problem.objective(np.array([X1[i, j], X2[i, j]]))
    
    levels = np.linspace(Z.min(), Z.min() + 10, 25)
    ax.contourf(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.3)
    ax.contour(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.6)
    
    ax.plot(x1, 1 - x1, 'k-', linewidth=2.5, label='Constraint: x₁ + x₂ = 1')
    
    x_da = np.array(result_da.x_history)
    x_al = np.array(result_al.x_history)
    
    ax.plot(x_da[:, 0], x_da[:, 1], 'o-', color='#e74c3c', markersize=4, 
            linewidth=1.5, label=f'Standard Lagrangian ({result_da.iterations} iters)', alpha=0.8)
    ax.plot(x_al[:, 0], x_al[:, 1], 's-', color='#27ae60', markersize=5, 
            linewidth=2, label=f'Augmented Lagrangian ({result_al.iterations} iters)', alpha=0.9)
    
    ax.plot(problem.x0[0], problem.x0[1], 'ko', markersize=12, label='Start', zorder=10)
    ax.plot(problem.x_optimal[0], problem.x_optimal[1], 'b*', markersize=18, 
            label=f'Optimal ({problem.x_optimal[0]:.1f}, {problem.x_optimal[1]:.1f})', zorder=10)
    
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Convex QP - Optimization Trajectory', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(problem.x_range)
    ax.set_ylim(problem.y_range)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_case_a_convergence(problem: TestProblem, result_da: OptimizationResult, 
                            result_al: OptimizationResult):
    """Case A: Convergence analysis (4 panels)."""
    output_dir = ensure_output_dir('case_a_convex')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Convex QP - Convergence Analysis', fontsize=14, fontweight='bold')
    
    c_da = '#e74c3c'
    c_al = '#27ae60'
    
    ax1 = axes[0, 0]
    f_err_da = [abs(f - problem.f_optimal) for f in result_da.f_history]
    f_err_al = [abs(f - problem.f_optimal) for f in result_al.f_history]
    ax1.semilogy(f_err_da, 'o-', color=c_da, markersize=3, label='Standard Lagrangian')
    ax1.semilogy(f_err_al, 's-', color=c_al, markersize=4, label='Augmented Lagrangian')
    ax1.axhline(y=1e-6, color='gray', linestyle=':', label='Tolerance')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('|f(x) - f*|')
    ax1.set_title('Objective Value Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    viol_da = [abs(c) for c in result_da.constraint_history]
    viol_al = [abs(c) for c in result_al.constraint_history]
    ax2.semilogy(viol_da, 'o-', color=c_da, markersize=3, label='Standard Lagrangian')
    ax2.semilogy(viol_al, 's-', color=c_al, markersize=4, label='Augmented Lagrangian')
    ax2.axhline(y=1e-6, color='gray', linestyle=':', label='Tolerance')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('|g(x)|')
    ax2.set_title('Constraint Violation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    dist_da = [np.linalg.norm(x - problem.x_optimal) for x in result_da.x_history]
    dist_al = [np.linalg.norm(x - problem.x_optimal) for x in result_al.x_history]
    ax3.semilogy(dist_da, 'o-', color=c_da, markersize=3, label='Standard Lagrangian')
    ax3.semilogy(dist_al, 's-', color=c_al, markersize=4, label='Augmented Lagrangian')
    ax3.axhline(y=1e-6, color='gray', linestyle=':', label='Tolerance')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('||x - x*||')
    ax3.set_title('Distance to Optimal Solution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(result_da.lambda_history, 'o-', color=c_da, markersize=3, label='Standard Lagrangian')
    ax4.plot(result_al.lambda_history, 's-', color=c_al, markersize=4, label='Augmented Lagrangian')
    ax4.axhline(y=problem.lambda_optimal, color='blue', linestyle='--', 
                linewidth=2, label=f'λ* = {problem.lambda_optimal}')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('λ')
    ax4.set_title('Lagrange Multiplier Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_case_a_comparison_panel(problem: TestProblem, result_da: OptimizationResult, 
                                  result_al: OptimizationResult):
    """Case A: Side-by-side trajectory comparison."""
    output_dir = ensure_output_dir('case_a_convex')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Convex QP - Method Comparison', fontsize=14, fontweight='bold')
    
    x1 = np.linspace(problem.x_range[0], problem.x_range[1], 100)
    x2 = np.linspace(problem.y_range[0], problem.y_range[1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = problem.objective(np.array([X1[i, j], X2[i, j]]))
    levels = np.linspace(Z.min(), Z.min() + 10, 20)
    
    x_da = np.array(result_da.x_history)
    x_al = np.array(result_al.x_history)
    
    ax1 = axes[0]
    ax1.contourf(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.3)
    ax1.contour(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.5)
    ax1.plot(x1, 1 - x1, 'k-', linewidth=2, label='Constraint')
    ax1.plot(x_da[:, 0], x_da[:, 1], 'o-', color='#e74c3c', markersize=4, linewidth=1.5)
    ax1.plot(problem.x0[0], problem.x0[1], 'ko', markersize=10, label='Start')
    ax1.plot(problem.x_optimal[0], problem.x_optimal[1], 'b*', markersize=15, label='Optimal')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title(f'Standard Lagrangian\n({result_da.iterations} iterations)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(problem.x_range)
    ax1.set_ylim(problem.y_range)
    
    ax2 = axes[1]
    ax2.contourf(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.3)
    ax2.contour(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.5)
    ax2.plot(x1, 1 - x1, 'k-', linewidth=2, label='Constraint')
    ax2.plot(x_al[:, 0], x_al[:, 1], 's-', color='#27ae60', markersize=5, linewidth=2)
    ax2.plot(problem.x0[0], problem.x0[1], 'ko', markersize=10, label='Start')
    ax2.plot(problem.x_optimal[0], problem.x_optimal[1], 'b*', markersize=15, label='Optimal')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title(f'Augmented Lagrangian\n({result_al.iterations} iterations)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(problem.x_range)
    ax2.set_ylim(problem.y_range)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Case B: Non-Convex Double Well Visualization
# =============================================================================

def plot_case_b_trajectory(problem: TestProblem, result_da: OptimizationResult, 
                           result_al: OptimizationResult):
    """Case B: Trajectory comparison showing failure vs success."""
    output_dir = ensure_output_dir('case_b_nonconvex')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x1 = np.linspace(problem.x_range[0], problem.x_range[1], 150)
    x2 = np.linspace(problem.y_range[0], problem.y_range[1], 150)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = problem.objective(np.array([X1[i, j], X2[i, j]]))
    
    levels = np.linspace(0, 3, 25)
    ax.contourf(X1, X2, Z, levels=levels, cmap='plasma', alpha=0.4)
    ax.contour(X1, X2, Z, levels=levels, cmap='plasma', alpha=0.6)
    
    ax.axvline(x=0, color='k', linewidth=2.5, label='Constraint: x₁ = 0')
    
    ax.plot(1, 0, 'ko', markersize=10, label='Well bottom (1, 0)')
    ax.plot(-1, 0, 'ko', markersize=10, label='Well bottom (-1, 0)')
    
    x_da = np.array(result_da.x_history)
    x_al = np.array(result_al.x_history)
    
    ax.plot(x_da[:, 0], x_da[:, 1], 'o-', color='#e74c3c', markersize=3, 
            linewidth=1.5, label=f'Standard Lagrangian ({result_da.iterations} iters)', alpha=0.8)
    ax.plot(x_al[:, 0], x_al[:, 1], 's-', color='#27ae60', markersize=4, 
            linewidth=2, label=f'Augmented Lagrangian ({result_al.iterations} iters)', alpha=0.9)
    
    ax.plot(problem.x0[0], problem.x0[1], 'ko', markersize=12, zorder=10)
    ax.plot(problem.x_optimal[0], problem.x_optimal[1], 'b*', markersize=18, 
            label=f'Optimal ({problem.x_optimal[0]:.1f}, {problem.x_optimal[1]:.1f})', zorder=10)
    
    ax.plot(x_da[-1, 0], x_da[-1, 1], 'X', color='#e74c3c', markersize=15, 
            markeredgewidth=3, label=f'Standard Final: ({x_da[-1, 0]:.2f}, {x_da[-1, 1]:.2f})')
    ax.plot(x_al[-1, 0], x_al[-1, 1], 'X', color='#27ae60', markersize=15, 
            markeredgewidth=3, label=f'ALM Final: ({x_al[-1, 0]:.2f}, {x_al[-1, 1]:.2f})')
    
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Double Well - Optimization Trajectory', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(problem.x_range)
    ax.set_ylim(problem.y_range)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_case_b_slowmotion(problem: TestProblem, result_da: OptimizationResult, 
                           result_al: OptimizationResult):
    """Case B: Slow-motion convergence analysis."""
    output_dir = ensure_output_dir('case_b_nonconvex')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Double Well - Convergence Analysis', fontsize=14, fontweight='bold')
    
    c_da = '#e74c3c'
    c_al = '#27ae60'
    
    ax1 = axes[0]
    viol_da = [abs(c) for c in result_da.constraint_history]
    viol_al = [abs(c) for c in result_al.constraint_history]
    ax1.semilogy(viol_da, 'o-', color=c_da, markersize=2, linewidth=1.5, label='Standard Lagrangian')
    ax1.semilogy(viol_al, 's-', color=c_al, markersize=3, linewidth=2, label='Augmented Lagrangian')
    ax1.axhline(y=1e-6, color='gray', linestyle=':', linewidth=1.5, label='Tolerance (1e-6)')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('|g(x)| = |x₁|', fontsize=12)
    ax1.set_title('Constraint Violation Over Iterations')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    final_da = viol_da[-1]
    final_al = viol_al[-1]
    ax1.annotate(f'|g| = {final_da:.3f}', xy=(len(viol_da)-1, final_da), 
                 xytext=(-60, 20), textcoords='offset points', fontsize=10,
                 color=c_da, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=c_da))
    ax1.annotate(f'|g| = {final_al:.2e}', xy=(len(viol_al)-1, max(final_al, 1e-9)), 
                 xytext=(-60, -30), textcoords='offset points', fontsize=10,
                 color=c_al, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=c_al))
    
    ax2 = axes[1]
    x1_da = [x[0] for x in result_da.x_history]
    x1_al = [x[0] for x in result_al.x_history]
    ax2.plot(x1_da, 'o-', color=c_da, markersize=2, linewidth=1.5, label='Standard Lagrangian')
    ax2.plot(x1_al, 's-', color=c_al, markersize=3, linewidth=2, label='Augmented Lagrangian')
    ax2.axhline(y=0, color='blue', linestyle='--', linewidth=2, label='Optimal x₁ = 0')
    ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.7, label='Well bottoms (x₁ = ±1)')
    ax2.axhline(y=-1, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('x₁', fontsize=12)
    ax2.set_title('x₁ Evolution: Does it reach the constraint?')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_slowmotion.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_case_b_convergence(problem: TestProblem, result_da: OptimizationResult, 
                            result_al: OptimizationResult):
    """Case B: Convergence metrics (4 panels)."""
    output_dir = ensure_output_dir('case_b_nonconvex')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Double Well - Convergence Metrics', fontsize=14, fontweight='bold')
    
    c_da = '#e74c3c'
    c_al = '#27ae60'
    
    ax1 = axes[0, 0]
    ax1.plot(result_da.f_history, 'o-', color=c_da, markersize=2, label='Standard Lagrangian')
    ax1.plot(result_al.f_history, 's-', color=c_al, markersize=3, label='Augmented Lagrangian')
    ax1.axhline(y=problem.f_optimal, color='blue', linestyle='--', linewidth=2, label=f'f* = {problem.f_optimal}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Objective Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    viol_da = [abs(c) for c in result_da.constraint_history]
    viol_al = [abs(c) for c in result_al.constraint_history]
    ax2.semilogy(viol_da, 'o-', color=c_da, markersize=2, label='Standard Lagrangian')
    ax2.semilogy(viol_al, 's-', color=c_al, markersize=3, label='Augmented Lagrangian')
    ax2.axhline(y=1e-6, color='gray', linestyle=':', label='Tolerance')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('|g(x)| = |x₁|')
    ax2.set_title('Constraint Violation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    dist_da = [np.linalg.norm(x - problem.x_optimal) for x in result_da.x_history]
    dist_al = [np.linalg.norm(x - problem.x_optimal) for x in result_al.x_history]
    ax3.semilogy(dist_da, 'o-', color=c_da, markersize=2, label='Standard Lagrangian')
    ax3.semilogy(dist_al, 's-', color=c_al, markersize=3, label='Augmented Lagrangian')
    ax3.axhline(y=1e-6, color='gray', linestyle=':', label='Tolerance')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('||x - x*||')
    ax3.set_title('Distance to Optimal Solution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(result_da.lambda_history, 'o-', color=c_da, markersize=2, label='Standard Lagrangian')
    ax4.plot(result_al.lambda_history, 's-', color=c_al, markersize=3, label='Augmented Lagrangian')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('λ')
    ax4.set_title('Lagrange Multiplier Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_case_b_convexification_contour(problem: TestProblem):
    """Case B: 2D contour visualization of convexification effect."""
    output_dir = ensure_output_dir('case_b_nonconvex')
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Double Well Function - Contour Comparison', fontsize=14, fontweight='bold')
    
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-1.5, 1.5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    F = (X1**2 - 1)**2 + X2**2
    G = X1
    
    ax1 = axes[0]
    levels = np.linspace(0, 4, 20)
    ax1.contourf(X1, X2, F, levels=levels, cmap='viridis', alpha=0.7)
    ax1.contour(X1, X2, F, levels=levels, colors='white', alpha=0.3, linewidths=0.5)
    ax1.axvline(x=0, color='red', linewidth=2.5, label='Constraint: x₁=0')
    ax1.plot(1, 0, 'ko', markersize=10, label='Well bottom (±1, 0)')
    ax1.plot(-1, 0, 'ko', markersize=10)
    ax1.plot(0, 0, 'r*', markersize=15, label='Constrained optimum')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title('Original Objective f(x)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_xlim([-2, 2])
    ax1.set_ylim([-1.5, 1.5])
    
    ax2 = axes[1]
    lam = 2.0
    L_std = F + lam * G
    ax2.contourf(X1, X2, L_std, levels=np.linspace(-2, 6, 20), cmap='plasma', alpha=0.7)
    ax2.contour(X1, X2, L_std, levels=np.linspace(-2, 6, 20), colors='white', alpha=0.3, linewidths=0.5)
    ax2.axvline(x=0, color='red', linewidth=2.5)
    ax2.plot(1, 0, 'ko', markersize=10)
    ax2.plot(-1, 0, 'ko', markersize=10)
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title(f'Standard Lagrangian L (λ={lam})')
    ax2.set_xlim([-2, 2])
    ax2.set_ylim([-1.5, 1.5])
    
    ax3 = axes[2]
    rho = 10.0
    lam_aug = 0
    L_aug = F + lam_aug * G + (rho / 2) * G**2
    levels_aug = np.linspace(0, 8, 20)
    ax3.contourf(X1, X2, L_aug, levels=levels_aug, cmap='coolwarm', alpha=0.7)
    ax3.contour(X1, X2, L_aug, levels=levels_aug, colors='white', alpha=0.3, linewidths=0.5)
    ax3.axvline(x=0, color='red', linewidth=2.5)
    ax3.plot(0, 0, 'r*', markersize=15)
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.set_title(f'Augmented: Lₐ = f + λg + (ρ/2)g²')
    ax3.set_xlim([-2, 2])
    ax3.set_ylim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_contour_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_case_b_3d_surface(problem: TestProblem):
    """Case B: Double Well Function 3D visualization."""
    output_dir = ensure_output_dir('case_b_nonconvex')
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle('Double Well Function 3D Visualization', fontsize=14, fontweight='bold')
    
    x1 = np.linspace(-2, 2, 80)
    x2 = np.linspace(-1.5, 1.5, 80)
    X1, X2 = np.meshgrid(x1, x2)
    
    F = (X1**2 - 1)**2 + X2**2
    G = X1
    
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(X1, X2, F, cmap='viridis', alpha=0.8, edgecolor='none')
    x2_line = np.linspace(-1.5, 1.5, 50)
    z_constraint = x2_line**2 + 1
    ax1.plot(np.zeros_like(x2_line), x2_line, z_constraint, 'r-', linewidth=3, label='Constraint x₁=0')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_zlabel('f(x)')
    ax1.set_title('Original: f(x) = (x₁²-1)² + x₂²')
    ax1.view_init(25, -60)
    ax1.set_zlim([0, 5])
    
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    lam = 2.0
    L_std = F + lam * G
    ax2.plot_surface(X1, X2, L_std, cmap='plasma', alpha=0.8, edgecolor='none')
    ax2.plot(np.zeros_like(x2_line), x2_line, 1 + x2_line**2, 'r-', linewidth=3)
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_zlabel('L')
    ax2.set_title(f'Standard: L = f + λg (λ={lam})')
    ax2.view_init(25, -60)
    ax2.set_zlim([-3, 5])
    
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    rho = 10.0
    lam_aug = 0
    L_aug = F + lam_aug * G + (rho / 2) * G**2
    L_aug_clipped = np.clip(L_aug, 0, 8)
    ax3.plot_surface(X1, X2, L_aug_clipped, cmap='coolwarm', alpha=0.8, edgecolor='none')
    ax3.plot(np.zeros_like(x2_line), x2_line, 1 + x2_line**2, 'r-', linewidth=3)
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.set_zlabel('Lₐ')
    ax3.set_title(f'Augmented: Lₐ = f + λg + (ρ/2)g² (ρ={rho})')
    ax3.view_init(25, -60)
    ax3.set_zlim([0, 8])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_3d_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_case_b_1d_slice(problem: TestProblem):
    """Case B: 1D Slice at x₂=0 showing convexification effect."""
    output_dir = ensure_output_dir('case_b_nonconvex')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x1 = np.linspace(-2, 2, 200)
    f_original = (x1**2 - 1)**2
    
    ax.plot(x1, f_original, 'k-', linewidth=3, label='Original: f(x₁) = (x₁²-1)²')
    
    ax.plot(1, 0, 'ko', markersize=12)
    ax.plot(-1, 0, 'ko', markersize=12)
    ax.plot(0, 1, 'k^', markersize=10, label='Saddle point (unstable)')
    
    lam = 2.0
    L_std = f_original + lam * x1
    ax.plot(x1, L_std, 'r--', linewidth=2.5, label=f'Standard: L = f + λx₁ (λ={lam})')
    
    min_idx_std = np.argmin(L_std)
    ax.plot(x1[min_idx_std], L_std[min_idx_std], 'rv', markersize=12)
    
    rho = 10.0
    lam_aug = 0
    L_aug = f_original + lam_aug * x1 + (rho / 2) * x1**2
    ax.plot(x1, L_aug, 'g-', linewidth=3, label=f'Augmented: Lₐ = f + λg + (ρ/2)g² (ρ={rho})')
    
    ax.plot(0, 1, 'g*', markersize=18, label='ALM minimum at x₁=0')
    
    ax.axvline(x=0, color='blue', linestyle=':', linewidth=2, alpha=0.7, label='Constraint: x₁=0')
    
    ax.set_xlabel('x₁', fontsize=14)
    ax.set_ylabel('Function Value', fontsize=14)
    ax.set_title('1D Slice at x₂=0: Double Well Function Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-1, 6])
    
    ax.annotate('Well minimum\n(local opt)', xy=(1, 0), xytext=(1.3, 1.5),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('Well minimum\n(local opt)', xy=(-1, 0), xytext=(-1.3, 1.5),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('Standard tilts W\nbut minimum stays\nin well', 
                xy=(x1[min_idx_std], L_std[min_idx_std]), 
                xytext=(0.8, -0.5),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.annotate('ALM fills wells\nminimum at constraint!', 
                xy=(0, 1), xytext=(-1.5, 4),
                fontsize=10, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_1d_slice.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_case_b_lambda_sweep_1d():
    """Case B: 1D Slice showing Standard Lagrangian vs Augmented Lagrangian sweep."""
    output_dir = ensure_output_dir('case_b_nonconvex')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x1 = np.linspace(-2, 2, 200)
    f_original = (x1**2 - 1)**2
    
    ax1 = axes[0]
    ax1.plot(x1, f_original, 'k-', linewidth=3, label='Original f(x₁)')
    
    lambda_values = [-3, -1, 0, 1, 3]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(lambda_values)))
    
    for lam, color in zip(lambda_values, colors):
        L = f_original + lam * x1
        ax1.plot(x1, L, '--', color=color, linewidth=1.5, label=f'L (λ={lam})')
        min_idx = np.argmin(L)
        ax1.plot(x1[min_idx], L[min_idx], 'v', color=color, markersize=8)
    
    ax1.axvline(x=0, color='blue', linestyle=':', linewidth=2, label='Constraint x₁=0')
    ax1.set_xlabel('x₁', fontsize=12)
    ax1.set_ylabel('Function Value', fontsize=12)
    ax1.set_title('Standard Lagrangian: No λ Reaches x₁=0', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-2, 2])
    ax1.set_ylim([-4, 6])
    
    ax2 = axes[1]
    ax2.plot(x1, f_original, 'k-', linewidth=3, label='Original f(x₁)')
    
    rho_values = [1, 3, 5, 10, 20]
    colors_rho = plt.cm.Greens(np.linspace(0.3, 0.9, len(rho_values)))
    
    for rho, color in zip(rho_values, colors_rho):
        lam_aug = 0
        L_aug = f_original + lam_aug * x1 + (rho / 2) * x1**2
        ax2.plot(x1, L_aug, '-', color=color, linewidth=1.5, label=f'Lₐ (ρ={rho})')
    
    ax2.axvline(x=0, color='blue', linestyle=':', linewidth=2, label='Constraint x₁=0')
    ax2.plot(0, 1, 'g*', markersize=15, zorder=10)
    ax2.set_xlabel('x₁', fontsize=12)
    ax2.set_ylabel('Function Value', fontsize=12)
    ax2.set_title('Augmented Lagrangian: All ρ Reach x₁=0', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-2, 2])
    ax2.set_ylim([0, 10])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_lambda_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_case_b_comparison_panel(problem: TestProblem, result_da: OptimizationResult, 
                                  result_al: OptimizationResult):
    """Case B: Side-by-side trajectory comparison."""
    output_dir = ensure_output_dir('case_b_nonconvex')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Double Well - Method Comparison', fontsize=14, fontweight='bold')
    
    x1 = np.linspace(problem.x_range[0], problem.x_range[1], 100)
    x2 = np.linspace(problem.y_range[0], problem.y_range[1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = problem.objective(np.array([X1[i, j], X2[i, j]]))
    levels = np.linspace(0, 3, 20)
    
    x_da = np.array(result_da.x_history)
    x_al = np.array(result_al.x_history)
    
    ax1 = axes[0]
    ax1.contourf(X1, X2, Z, levels=levels, cmap='plasma', alpha=0.4)
    ax1.contour(X1, X2, Z, levels=levels, cmap='plasma', alpha=0.5)
    ax1.axvline(x=0, color='k', linewidth=2, label='Constraint')
    ax1.plot(x_da[:, 0], x_da[:, 1], 'o-', color='#e74c3c', markersize=3, linewidth=1.5)
    ax1.plot(problem.x0[0], problem.x0[1], 'ko', markersize=10, label='Start')
    ax1.plot(problem.x_optimal[0], problem.x_optimal[1], 'b*', markersize=15, label='Optimal')
    ax1.plot(1, 0, 'kx', markersize=10, label='Well bottom')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title(f'Standard Lagrangian\n({result_da.iterations} iterations, |g|={abs(result_da.final_constraint):.3f})', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(problem.x_range)
    ax1.set_ylim(problem.y_range)
    
    ax2 = axes[1]
    ax2.contourf(X1, X2, Z, levels=levels, cmap='plasma', alpha=0.4)
    ax2.contour(X1, X2, Z, levels=levels, cmap='plasma', alpha=0.5)
    ax2.axvline(x=0, color='k', linewidth=2, label='Constraint')
    ax2.plot(x_al[:, 0], x_al[:, 1], 's-', color='#27ae60', markersize=4, linewidth=2)
    ax2.plot(problem.x0[0], problem.x0[1], 'ko', markersize=10, label='Start')
    ax2.plot(problem.x_optimal[0], problem.x_optimal[1], 'b*', markersize=15, label='Optimal')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title(f'Augmented Lagrangian\n({result_al.iterations} iterations, |g|={abs(result_al.final_constraint):.2e})', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(problem.x_range)
    ax2.set_ylim(problem.y_range)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("TWO-STAGE COMPARISON: Standard Lagrangian vs Augmented Lagrangian")
    print("=" * 70)
    
    prob_convex = create_convex_qp()
    prob_nonconvex = create_double_well()
    
    # Case A: Convex QP
    print("\n" + "-" * 50)
    print("CASE A: Convex QP (Baseline)")
    print("-" * 50)
    print(f"Problem: {prob_convex.description}")
    print(f"Start: {prob_convex.x0}, Optimal: {prob_convex.x_optimal}")
    
    result_convex_da = dual_ascent(prob_convex, max_iter=200, verbose=True)
    result_convex_al = augmented_lagrangian(prob_convex, max_outer_iter=30, verbose=True)
    
    print(f"\n  Standard:  x={result_convex_da.final_x}, |g|={abs(result_convex_da.final_constraint):.2e}")
    print(f"  ALM:       x={result_convex_al.final_x}, |g|={abs(result_convex_al.final_constraint):.2e}")
    
    plot_case_a_trajectory(prob_convex, result_convex_da, result_convex_al)
    plot_case_a_convergence(prob_convex, result_convex_da, result_convex_al)
    plot_case_a_comparison_panel(prob_convex, result_convex_da, result_convex_al)
    
    # Case B: Non-Convex Double Well
    print("\n" + "-" * 50)
    print("CASE B: Non-Convex Double Well (Critical Failure Case)")
    print("-" * 50)
    print(f"Problem: {prob_nonconvex.description}")
    print(f"Start: {prob_nonconvex.x0}, Optimal: {prob_nonconvex.x_optimal}")
    print(f"Using small step size (alpha={prob_nonconvex.alpha_x}) for slow-motion analysis")
    
    result_nonconvex_da = dual_ascent(prob_nonconvex, max_iter=500, verbose=True)
    result_nonconvex_al = augmented_lagrangian(prob_nonconvex, max_outer_iter=50, verbose=True)
    
    print(f"\n  Standard:  x={result_nonconvex_da.final_x}, |g|={abs(result_nonconvex_da.final_constraint):.2e}")
    print(f"  ALM:       x={result_nonconvex_al.final_x}, |g|={abs(result_nonconvex_al.final_constraint):.2e}")
    
    plot_case_b_trajectory(prob_nonconvex, result_nonconvex_da, result_nonconvex_al)
    plot_case_b_slowmotion(prob_nonconvex, result_nonconvex_da, result_nonconvex_al)
    plot_case_b_convergence(prob_nonconvex, result_nonconvex_da, result_nonconvex_al)
    plot_case_b_convexification_contour(prob_nonconvex)
    plot_case_b_3d_surface(prob_nonconvex)
    plot_case_b_1d_slice(prob_nonconvex)
    plot_case_b_lambda_sweep_1d()
    plot_case_b_comparison_panel(prob_nonconvex, result_nonconvex_da, result_nonconvex_al)
    
    return {
        'convex': (result_convex_da, result_convex_al),
        'nonconvex': (result_nonconvex_da, result_nonconvex_al)
    }


if __name__ == "__main__":
    results = main()
