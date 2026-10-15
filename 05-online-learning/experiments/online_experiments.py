from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
LEGACY_TABLE_EXPORTS = [
    "svm_schedule_table.tex",
    "algorithm_family_table.tex",
    "geometry_table.tex",
    "stochastic_noisy_table.tex",
]

ORIGINAL_BETA_ORDER = ["0", "0.5", "1-1/sqrt(t)"]
ORIGINAL_GAMMA_ORDER = ["c/sqrt(T)", "c/sqrt(t)", "c/t"]
PILOT_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
STAGE_CHOICES = {
    "1": "svm",
    "2": "geometry",
    "3": "stochastic",
    "4": "all",
    "svm": "svm",
    "geometry": "geometry",
    "stochastic": "stochastic",
    "all": "all",
}


@dataclass
class Problem:
    name: str
    task: str
    loss: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray | None
    y_test: np.ndarray | None
    reg_lambda: float
    constraint: str
    radius: float | None = None
    low: float | None = None
    high: float | None = None
    delta: float = 1.0
    metric_diag: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    def initial_point(self) -> np.ndarray:
        if self.constraint == "simplex":
            return np.full(self.dim, 1.0 / self.dim)
        return np.zeros(self.dim)

    def project(self, x: np.ndarray) -> np.ndarray:
        if self.constraint == "ball":
            assert self.radius is not None
            norm = np.linalg.norm(x)
            if norm <= self.radius or norm == 0.0:
                return x
            return x * (self.radius / norm)
        if self.constraint == "box":
            assert self.low is not None and self.high is not None
            return np.clip(x, self.low, self.high)
        if self.constraint == "simplex":
            return project_simplex(x)
        raise ValueError(f"Unknown constraint {self.constraint}")

    def objective(self, w: np.ndarray, split: str = "train") -> float:
        X, y = self._data(split)
        linear = X @ w
        regularizer = 0.5 * self.reg_lambda * float(np.dot(w, w))

        if self.loss == "hinge":
            losses = np.maximum(0.0, 1.0 - y * linear)
            return regularizer + float(np.mean(losses))
        if self.loss == "logistic":
            losses = np.logaddexp(0.0, -y * linear)
            return regularizer + float(np.mean(losses))
        if self.loss == "huber":
            residual = linear - y
            abs_residual = np.abs(residual)
            losses = np.where(
                abs_residual <= self.delta,
                0.5 * residual**2,
                self.delta * (abs_residual - 0.5 * self.delta),
            )
            return regularizer + float(np.mean(losses))
        if self.loss == "lad":
            residual = linear - y
            return regularizer + float(np.mean(np.abs(residual)))
        raise ValueError(f"Unknown loss {self.loss}")

    def subgradient(self, w: np.ndarray, batch_indices: np.ndarray | None = None) -> np.ndarray:
        X = self.X_train if batch_indices is None else self.X_train[batch_indices]
        y = self.y_train if batch_indices is None else self.y_train[batch_indices]
        n = float(X.shape[0])
        linear = X @ w

        if self.loss == "hinge":
            active = (y * linear) < 1.0
            grad = self.reg_lambda * w
            if np.any(active):
                grad = grad - (X[active].T @ y[active]) / n
            return grad

        if self.loss == "logistic":
            yz = y * linear
            coeff = -y / (1.0 + np.exp(np.clip(yz, -60.0, 60.0)))
            return self.reg_lambda * w + (X.T @ coeff) / n

        if self.loss == "huber":
            residual = linear - y
            psi = np.clip(residual, -self.delta, self.delta)
            return self.reg_lambda * w + (X.T @ psi) / n

        if self.loss == "lad":
            residual = linear - y
            psi = np.sign(residual)
            return self.reg_lambda * w + (X.T @ psi) / n

        raise ValueError(f"Unknown loss {self.loss}")

    def accuracy(self, w: np.ndarray, split: str = "test") -> float:
        if self.task != "classification":
            return float("nan")
        X, y = self._data(split)
        if X is None or y is None:
            return float("nan")
        pred = np.where((X @ w) >= 0.0, 1.0, -1.0)
        return float(np.mean(pred == y))

    def lipschitz_estimate(self) -> float:
        if self.loss in {"hinge", "logistic", "huber"}:
            radius = float(self.radius or 0.0)
            return self.reg_lambda * radius + float(np.max(np.linalg.norm(self.X_train, axis=1)))
        if self.loss == "lad":
            return float(np.max(np.linalg.norm(self.X_train, axis=0)))
        return 1.0

    def _data(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        if split == "train":
            return self.X_train, self.y_train
        if self.X_test is None or self.y_test is None:
            return self.X_train, self.y_train
        return self.X_test, self.y_test


@dataclass
class RunConfig:
    stage: str
    study: str
    problem_name: str
    algorithm: str
    geometry: str
    beta_name: str
    gamma_name: str
    base_scale: float
    horizon: int
    batch_size: int
    seed: int
    noise_mode: str = "none"
    noise_scale: float = 0.0

    def config_id(self) -> str:
        parts = [
            self.stage,
            self.study,
            self.problem_name,
            self.algorithm,
            self.geometry,
            self.gamma_name,
            self.beta_name,
            f"bs{self.batch_size}",
            f"seed{self.seed}",
            self.noise_mode,
        ]
        return "__".join(sanitize_token(part) for part in parts)


def sanitize_token(token: str) -> str:
    return (
        token.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "p")
        .replace("-", "_")
    )


def remove_legacy_table_exports() -> None:
    for filename in LEGACY_TABLE_EXPORTS:
        path = RESULTS_DIR / filename
        if path.exists():
            path.unlink()


def save_stage_exports(stage: str, summary_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    summary_cols = [
        "stage",
        "study",
        "problem",
        "algorithm",
        "geometry",
        "regime",
        "gamma_name",
        "beta_name",
        "batch_size",
        "noise_mode",
        "seed",
        "base_scale",
        "noise_scale",
        "reference_objective",
        "final_train_objective",
        "best_train_objective",
        "final_objective_gap",
        "best_objective_gap",
        "final_test_accuracy",
        "final_stability",
        "final_grad_norm",
        "wall_clock_sec",
        "is_reference",
        "reference_source",
        "reference_winner",
        "config_id",
    ]
    history_cols = [
        "stage",
        "study",
        "problem",
        "algorithm",
        "geometry",
        "regime",
        "gamma_name",
        "beta_name",
        "iteration",
        "gamma_t",
        "beta_t",
        "batch_size",
        "noise_mode",
        "seed",
        "reference_objective",
        "train_objective",
        "objective_gap",
        "test_accuracy",
        "stability",
        "grad_norm",
        "elapsed_sec",
        "config_id",
    ]
    summary_export = summary_df[[col for col in summary_cols if col in summary_df.columns]].sort_values(
        ["study", "problem", "algorithm", "regime", "gamma_name", "beta_name", "seed", "is_reference"]
    )
    history_export = history_df[[col for col in history_cols if col in history_df.columns]].sort_values(
        ["study", "problem", "algorithm", "regime", "beta_name", "seed", "iteration"]
    )
    summary_export.to_csv(RESULTS_DIR / f"{stage}_summary.csv", index=False)
    history_export.to_csv(RESULTS_DIR / f"{stage}_history.csv", index=False)
    summary_export[summary_export["is_reference"]].to_csv(RESULTS_DIR / f"{stage}_reference.csv", index=False)


def project_simplex(v: np.ndarray) -> np.ndarray:
    # Euclidean projection onto the probability simplex.
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    idx = np.arange(1, v.size + 1)
    cond = u - cssv / idx > 0.0
    rho = int(idx[cond][-1])
    theta = cssv[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


def standardize_from_train(X_train: np.ndarray, X_test: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0.0] = 1.0
    X_train_std = (X_train - mean) / std
    X_test_std = None if X_test is None else (X_test - mean) / std
    return X_train_std, X_test_std


def standardize_target_from_train(y_train: np.ndarray, y_test: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if std == 0.0:
        std = 1.0
    y_train_std = (y_train - mean) / std
    y_test_std = None if y_test is None else (y_test - mean) / std
    return y_train_std, y_test_std


def build_core_problems() -> dict[str, Problem]:
    # Recast digits as an even-vs-odd binary classification problem.
    data = load_digits()
    X = data.data.astype(float)
    y = np.where((data.target % 2) == 0, 1.0, -1.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )
    X_train, X_test = standardize_from_train(X_train, X_test)

    svm = Problem(
        name="svm_digits_even_odd",
        task="classification",
        loss="hinge",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        reg_lambda=1e-2,
        constraint="ball",
        radius=2.0,
    )
    logistic = Problem(
        name="logistic_digits_even_odd",
        task="classification",
        loss="logistic",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        reg_lambda=1e-2,
        constraint="ball",
        radius=2.0,
    )

    # Use synthetic regression task for the Huber benchmark.
    rng = np.random.default_rng(0)
    n = 2000
    d = 25
    X = rng.normal(size=(n, d))
    w_star = rng.normal(size=d)
    y = X @ w_star + 0.5 * rng.standard_t(df=3, size=n)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
    )
    X_train, X_test = standardize_from_train(X_train, X_test)
    y_train, y_test = standardize_target_from_train(y_train, y_test)
    huber = Problem(
        name="huber_synthetic_medium",
        task="regression",
        loss="huber",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        reg_lambda=1e-2,
        constraint="ball",
        radius=6.0,
        delta=1.0,
    )

    return {
        svm.name: svm,
        logistic.name: logistic,
        huber.name: huber,
    }


def build_simplex_problem(seed: int = 0) -> Problem:
    rng = np.random.default_rng(seed)
    # Normalize columns so the LAD geometry comparison is not dominated by scaling.
    A = rng.normal(size=(800, 80))
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0.0] = 1.0
    A = A / col_norms
    w_star = rng.dirichlet(0.2 * np.ones(80))
    b = A @ w_star + rng.laplace(scale=0.05, size=800)
    return Problem(
        name="simplex_lad_synthetic",
        task="regression",
        loss="lad",
        X_train=A,
        y_train=b,
        X_test=None,
        y_test=None,
        reg_lambda=0.0,
        constraint="simplex",
    )


def build_ill_conditioned_logistic(seed: int = 0) -> Problem:
    rng = np.random.default_rng(seed)
    n = 2500
    d = 30
    # Spread feature scales across several orders of magnitude.
    scales = 10.0 ** np.linspace(0.0, 3.0, d)
    X = rng.normal(size=(n, d)) * scales
    w_star = rng.normal(size=d) / scales
    logits = X @ w_star + 0.5 * rng.normal(size=n)
    y = np.where(logits >= 0.0, 1.0, -1.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    metric_diag = np.mean(X_train**2, axis=0) + 1e-8
    return Problem(
        name="ill_conditioned_logistic",
        task="classification",
        loss="logistic",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        reg_lambda=1e-2,
        constraint="box",
        low=-5.0,
        high=5.0,
        metric_diag=metric_diag,
    )


def beta_value(beta_name: str, t: int) -> float:
    if beta_name == "0":
        return 0.0
    if beta_name == "0.5":
        return 0.5
    if beta_name == "1-1/sqrt(t)":
        return 1.0 - 1.0 / math.sqrt(t)
    raise ValueError(f"Unknown beta schedule {beta_name}")


def assert_feasible(problem: Problem, x: np.ndarray) -> None:
    if problem.constraint == "ball":
        assert problem.radius is not None
        if np.linalg.norm(x) > problem.radius + 1e-8:
            raise AssertionError(f"Ball constraint violated for {problem.name}")
    elif problem.constraint == "box":
        assert problem.low is not None and problem.high is not None
        if np.any(x < problem.low - 1e-8) or np.any(x > problem.high + 1e-8):
            raise AssertionError(f"Box constraint violated for {problem.name}")
    elif problem.constraint == "simplex":
        if np.min(x) < -1e-8 or abs(float(np.sum(x)) - 1.0) > 1e-8:
            raise AssertionError(f"Simplex constraint violated for {problem.name}")


def averaging_weight(config: RunConfig, problem: Problem, t: int) -> float:
    # Dual averaging and AdaGrad use uniform averaging weights.
    if config.algorithm in {"dual_averaging", "adagrad_norm"}:
        return 1.0
    if config.study == "original_svm":
        if config.gamma_name == "c/sqrt(T)":
            return config.base_scale / math.sqrt(config.horizon)
        if config.gamma_name == "c/sqrt(t)":
            return config.base_scale / math.sqrt(t)
        if config.gamma_name == "c/t":
            return config.base_scale / t
        raise ValueError(f"Unknown original gamma schedule {config.gamma_name}")
    return config.base_scale / math.sqrt(t)


def update_iterate(
    x: np.ndarray,
    grad: np.ndarray,
    config: RunConfig,
    problem: Problem,
    t: int,
    grad_sum: np.ndarray,
    grad_sq_sum: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if config.algorithm == "projected_subgradient":
        step = averaging_weight(config, problem, t)
        x_next = problem.project(x - step * grad)
        return x_next, grad_sum, grad_sq_sum

    if config.algorithm == "dual_averaging":
        grad_sum = grad_sum + grad
        eta_next = config.base_scale / math.sqrt(t + 1)
        x_next = problem.project(-eta_next * grad_sum)
        return x_next, grad_sum, grad_sq_sum

    if config.algorithm == "adagrad_norm":
        # Adapt the step size to the cumulative gradient norm.
        grad_sq_sum += float(np.dot(grad, grad))
        step = config.base_scale / math.sqrt(grad_sq_sum + 1e-12)
        x_next = problem.project(x - step * grad)
        return x_next, grad_sum, grad_sq_sum

    if config.algorithm == "entropy_mirror_descent":
        # Multiplicative update followed by simplex renormalization.
        step = averaging_weight(config, problem, t)
        log_weights = np.log(np.clip(x, 1e-12, None)) - step * grad
        log_weights = log_weights - np.max(log_weights)
        x_next = np.exp(log_weights)
        x_next = x_next / np.sum(x_next)
        return x_next, grad_sum, grad_sq_sum

    if config.algorithm == "mahalanobis_mirror_descent":
        step = averaging_weight(config, problem, t)
        assert problem.metric_diag is not None
        x_next = problem.project(x - step * grad / problem.metric_diag)
        return x_next, grad_sum, grad_sq_sum

    raise ValueError(f"Unknown algorithm {config.algorithm}")


def run_configuration(problem: Problem, config: RunConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(config.seed)
    x = problem.initial_point()
    xbar = x.copy()
    gamma_total = 0.0
    grad_sum = np.zeros_like(x)
    grad_sq_sum = 0.0
    history: list[dict[str, Any]] = []
    best_objective = float("inf")
    start = time.perf_counter()

    for t in range(1, config.horizon + 1):
        gamma_t = averaging_weight(config, problem, t)
        gamma_next = gamma_total + gamma_t
        regime = stochastic_regime_label(config.batch_size, config.noise_mode, problem.n_train)
        if gamma_total == 0.0:
            xbar_t = x.copy()
        else:
            # Update the weighted average before querying the oracle.
            xbar_t = (gamma_total * xbar + gamma_t * x) / gamma_next

        beta_t = beta_value(config.beta_name, t)
        z_t = beta_t * xbar_t + (1.0 - beta_t) * x
        if config.beta_name == "0" and not np.allclose(z_t, x):
            raise AssertionError("beta_t = 0 should sample the oracle at x_t")

        if config.batch_size >= problem.n_train:
            batch_idx = None
        else:
            # Sample with replacement to match the stochastic-update setting.
            batch_idx = rng.integers(0, problem.n_train, size=config.batch_size)

        grad = problem.subgradient(z_t, batch_idx)
        if config.noise_mode == "gaussian" and config.noise_scale > 0.0:
            # Add explicit oracle noise on top of mini-batch variability.
            grad = grad + config.noise_scale * rng.standard_normal(problem.dim)

        x_next, grad_sum, grad_sq_sum = update_iterate(
            x=x,
            grad=grad,
            config=config,
            problem=problem,
            t=t,
            grad_sum=grad_sum,
            grad_sq_sum=grad_sq_sum,
        )
        assert_feasible(problem, x_next)

        stability = float(np.linalg.norm(x_next - x))
        grad_norm = float(np.linalg.norm(grad))
        train_objective = problem.objective(xbar_t, split="train")
        best_objective = min(best_objective, train_objective)
        test_accuracy = problem.accuracy(xbar_t, split="test")
        elapsed = time.perf_counter() - start

        history.append(
            {
                "config_id": config.config_id(),
                "stage": config.stage,
                "study": config.study,
                "problem": problem.name,
                "algorithm": config.algorithm,
                "geometry": config.geometry,
                "regime": regime,
                "gamma_name": config.gamma_name,
                "beta_name": config.beta_name,
                "iteration": t,
                "gamma_t": gamma_t,
                "beta_t": beta_t,
                "train_objective": train_objective,
                "test_accuracy": test_accuracy,
                "stability": stability,
                "grad_norm": grad_norm,
                "elapsed_sec": elapsed,
                "batch_size": config.batch_size,
                "seed": config.seed,
                "noise_mode": config.noise_mode,
            }
        )

        x = x_next
        xbar = xbar_t
        gamma_total = gamma_next

    final_history = history[-1]
    summary = {
        "config_id": config.config_id(),
        "stage": config.stage,
        "study": config.study,
        "problem": problem.name,
        "algorithm": config.algorithm,
        "geometry": config.geometry,
        "regime": stochastic_regime_label(config.batch_size, config.noise_mode, problem.n_train),
        "gamma_name": config.gamma_name,
        "beta_name": config.beta_name,
        "base_scale": config.base_scale,
        "horizon": config.horizon,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "noise_mode": config.noise_mode,
        "noise_scale": config.noise_scale,
        "final_train_objective": final_history["train_objective"],
        "best_train_objective": best_objective,
        "final_test_accuracy": final_history["test_accuracy"],
        "final_stability": final_history["stability"],
        "final_grad_norm": final_history["grad_norm"],
        "wall_clock_sec": final_history["elapsed_sec"],
        "is_reference": False,
    }
    return summary, history


def tune_base_scale(
    problem: Problem,
    algorithm: str,
    horizon: int,
    geometry: str = "euclidean",
    batch_size: int | None = None,
    noise_mode: str = "none",
    noise_scale: float = 0.0,
    beta_name: str = "0",
    seed: int = 0,
) -> float:
    best_scale = PILOT_GRID[0]
    best_value = float("inf")
    pilot_horizon = max(100, horizon // 10)
    for scale in PILOT_GRID:
        candidate = RunConfig(
            stage="pilot",
            study="pilot",
            problem_name=problem.name,
            algorithm=algorithm,
            geometry=geometry,
            beta_name=beta_name,
            gamma_name="pilot",
            base_scale=scale,
            horizon=pilot_horizon,
            batch_size=problem.n_train if batch_size is None else batch_size,
            seed=seed,
            noise_mode=noise_mode,
            noise_scale=noise_scale,
        )
        summary, _ = run_configuration(problem, candidate)
        if summary["final_train_objective"] < best_value:
            best_value = summary["final_train_objective"]
            best_scale = scale
    return best_scale


def estimate_noise_scale(problem: Problem, base_scale: float, horizon: int = 100) -> float:
    cfg = RunConfig(
        stage="noise",
        study="noise_probe",
        problem_name=problem.name,
        algorithm="projected_subgradient",
        geometry="euclidean",
        beta_name="0",
        gamma_name="c/sqrt(t)",
        base_scale=base_scale,
        horizon=horizon,
        batch_size=problem.n_train,
        seed=0,
    )
    _, history = run_configuration(problem, cfg)
    grad_norms = np.array([row["grad_norm"] for row in history], dtype=float)
    return 0.1 * float(np.median(grad_norms))


def scale_lookup_key(problem_name: str, algorithm: str) -> str:
    return f"{problem_name}|{algorithm}"


def stochastic_scale_lookup_key(problem_name: str, algorithm: str, batch_size: int, noise_mode: str) -> str:
    return f"{problem_name}|{algorithm}|bs{batch_size}|{noise_mode}"


def build_reference_bank_configs(
    stage: str,
    problems: dict[str, Problem],
    horizon: int,
    scale_lookup: dict[str, float],
) -> list[RunConfig]:
    configs: list[RunConfig] = []
    if stage in {"svm", "stochastic"}:
        for problem_name, problem in problems.items():
            for algorithm in ["projected_subgradient", "dual_averaging", "adagrad_norm"]:
                scale = scale_lookup[scale_lookup_key(problem_name, algorithm)]
                for beta_name in ORIGINAL_BETA_ORDER:
                    configs.append(
                        RunConfig(
                            stage=stage,
                            study="reference",
                            problem_name=problem_name,
                            algorithm=algorithm,
                            geometry="euclidean",
                            beta_name=beta_name,
                            gamma_name="reference",
                            base_scale=scale,
                            horizon=horizon,
                            batch_size=problem.n_train,
                            seed=0,
                        )
                    )
    elif stage == "geometry":
        geometry_candidates = {
            "simplex_lad_synthetic": [
                ("projected_subgradient", "euclidean"),
                ("entropy_mirror_descent", "entropy"),
            ],
            "ill_conditioned_logistic": [
                ("projected_subgradient", "euclidean"),
                ("mahalanobis_mirror_descent", "diagonal_mahalanobis"),
            ],
        }
        for problem_name, candidates in geometry_candidates.items():
            problem = problems[problem_name]
            for algorithm, geometry in candidates:
                scale = scale_lookup[scale_lookup_key(problem_name, algorithm)]
                for beta_name in ["0", "0.5"]:
                    configs.append(
                        RunConfig(
                            stage=stage,
                            study="reference",
                            problem_name=problem_name,
                            algorithm=algorithm,
                            geometry=geometry,
                            beta_name=beta_name,
                            gamma_name="reference",
                            base_scale=scale,
                            horizon=horizon,
                            batch_size=problem.n_train,
                            seed=0,
                        )
                    )
    else:
        raise ValueError(f"Unknown reference-bank stage {stage}")
    return configs


def original_svm_configs(problem: Problem, horizon: int) -> list[RunConfig]:
    R = float(problem.radius or 1.0)
    l_est = problem.lipschitz_estimate()
    c = R * math.sqrt(2.0) / l_est
    configs: list[RunConfig] = []
    for gamma_name in ORIGINAL_GAMMA_ORDER:
        for beta_name in ORIGINAL_BETA_ORDER:
            configs.append(
                RunConfig(
                    stage="svm",
                    study="original_svm",
                    problem_name=problem.name,
                    algorithm="projected_subgradient",
                    geometry="euclidean",
                    beta_name=beta_name,
                    gamma_name=gamma_name,
                    base_scale=c,
                    horizon=horizon,
                    batch_size=problem.n_train,
                    seed=0,
                )
            )
    return configs


def run_configs(problems: dict[str, Problem], configs: list[RunConfig]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for config in configs:
        problem = problems[config.problem_name]
        summary, history = run_configuration(problem, config)
        summaries.append(summary)
        histories.extend(history)
    return pd.DataFrame(summaries), pd.DataFrame(histories)


def compute_reference_rows(problems: dict[str, Problem], reference_configs: list[RunConfig]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cfg in reference_configs:
        problem = problems[cfg.problem_name]
        summary, _ = run_configuration(problem, cfg)
        summary["is_reference"] = True
        summary["reference_source"] = "reference_bank"
        summary["reference_winner"] = False
        rows.append(summary)
    reference_df = pd.DataFrame(rows)
    if not reference_df.empty:
        winner_idx = reference_df.groupby("problem")["best_train_objective"].idxmin()
        reference_df.loc[winner_idx, "reference_winner"] = True
    return reference_df


def enrich_with_reference(
    summary_df: pd.DataFrame,
    history_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Convert raw objectives into comparable gaps using only the independent reference runs.
    ref_values = reference_df.groupby("problem")["best_train_objective"].min().rename("reference_objective")

    combined = pd.concat([summary_df, reference_df], ignore_index=True)
    combined = combined.merge(ref_values, on="problem", how="left")
    combined["final_objective_gap"] = combined["final_train_objective"] - combined["reference_objective"]
    combined["best_objective_gap"] = combined["best_train_objective"] - combined["reference_objective"]
    if "reference_source" in combined:
        combined["reference_source"] = combined["reference_source"].fillna("reference_bank")
    else:
        combined["reference_source"] = "reference_bank"
    if "reference_winner" in combined:
        combined["reference_winner"] = combined["reference_winner"].fillna(False)
    else:
        combined["reference_winner"] = False

    history_df = history_df.merge(ref_values, on="problem", how="left")
    history_df["objective_gap"] = history_df["train_objective"] - history_df["reference_objective"]
    return combined, history_df

def select_best_rows(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    idx = df.groupby(group_cols)[value_col].idxmin()
    return df.loc[idx].copy()


def stochastic_regime_label(batch_size: int, noise_mode: str, full_batch_size: int) -> str:
    if batch_size >= full_batch_size and noise_mode == "none":
        return "full-batch"
    if batch_size == 32 and noise_mode == "none":
        return "mini-batch 32"
    if batch_size == 32 and noise_mode == "gaussian":
        return "mini-batch 32 + noise"
    if batch_size == 1 and noise_mode == "none":
        return "single sample"
    noise_suffix = " + noise" if noise_mode != "none" else ""
    return f"batch {batch_size}{noise_suffix}"


def create_original_svm_assets(summary_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    subset = summary_df[(summary_df["study"] == "original_svm") & (~summary_df["is_reference"])].copy()
    subset["gamma_order"] = subset["gamma_name"].map({name: idx for idx, name in enumerate(ORIGINAL_GAMMA_ORDER)})
    subset["beta_order"] = subset["beta_name"].map({name: idx for idx, name in enumerate(ORIGINAL_BETA_ORDER)})
    subset = subset.sort_values(["gamma_order", "beta_order"])
    hist = history_df[history_df["study"] == "original_svm"]
    beta_colors = {
        "0": "#1f77b4",
        "0.5": "#ff7f0e",
        "1-1/sqrt(t)": "#2ca02c",
    }
    gamma_titles = {
        "c/sqrt(T)": r"$\gamma_t = c/\sqrt{T}$",
        "c/sqrt(t)": r"$\gamma_t = c/\sqrt{t}$",
        "c/t": r"$\gamma_t = c/t$",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for ax, gamma_name in zip(axes, ORIGINAL_GAMMA_ORDER):
        gamma_subset = subset[subset["gamma_name"] == gamma_name]
        for beta_name in ORIGINAL_BETA_ORDER:
            row = gamma_subset[gamma_subset["beta_name"] == beta_name].iloc[0]
            group = hist[hist["config_id"] == row["config_id"]]
            ax.plot(
                group["iteration"],
                np.maximum(group["objective_gap"], 1e-8),
                label=rf"$\beta_t={pretty_beta(beta_name)}$",
                color=beta_colors[beta_name],
                linewidth=1.7,
            )
        ax.set_title(gamma_titles[gamma_name])
        ax.set_xlabel("Iteration")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=True)
    axes[0].set_ylabel("Objective gap")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_svm_convergence.png", dpi=180)
    plt.close(fig)

    metrics = [
        ("Objective gap", True, lambda group: np.maximum(group["objective_gap"].to_numpy(dtype=float), 1e-8)),
        ("Test accuracy (%)", False, lambda group: 100.0 * group["test_accuracy"].to_numpy(dtype=float)),
        ("Iterate stability", True, lambda group: np.maximum(group["stability"].to_numpy(dtype=float), 1e-8)),
        ("Gradient norm", True, lambda group: np.maximum(group["grad_norm"].to_numpy(dtype=float), 1e-8)),
    ]

    gamma_colors = {
        "c/sqrt(T)": "#1f77b4",
        "c/sqrt(t)": "#ff7f0e",
        "c/t": "#2ca02c",
    }
    beta_styles = {
        "0": "-",
        "0.5": "--",
        "1-1/sqrt(t)": "-.",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4), sharex=True)
    for config_id, group in hist.groupby("config_id"):
        label_row = subset[subset["config_id"] == config_id].iloc[0]
        color = gamma_colors[str(label_row["gamma_name"])]
        linestyle = beta_styles[str(label_row["beta_name"])]
        for ax, (_, log_scale, value_fn) in zip(axes.ravel(), metrics):
            ax.plot(
                group["iteration"],
                value_fn(group),
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
            )
            if log_scale:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

    for ax, (title, _, _) in zip(axes.ravel(), metrics):
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(title)

    gamma_handles = [
        Line2D([0], [0], color=gamma_colors[name], linewidth=2, label=pretty_gamma(name))
        for name in ORIGINAL_GAMMA_ORDER
    ]
    beta_handles = [
        Line2D([0], [0], color="black", linestyle=beta_styles[name], linewidth=2, label=rf"$\beta_t={pretty_beta(name)}$")
        for name in ORIGINAL_BETA_ORDER
    ]
    gamma_legend = fig.legend(
        gamma_handles,
        [handle.get_label() for handle in gamma_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        fontsize=10,
        frameon=True,
        title=r"Step-size schedule",
        title_fontsize=10,
    )
    fig.add_artist(gamma_legend)
    fig.legend(
        beta_handles,
        [handle.get_label() for handle in beta_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        fontsize=10,
        frameon=True,
        title=r"Interpolation schedule",
        title_fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.savefig(RESULTS_DIR / "fig_svm_diagnostics.png", dpi=180)
    plt.close(fig)

    table_df = subset[
        ["gamma_name", "beta_name", "final_train_objective", "final_objective_gap", "final_test_accuracy", "wall_clock_sec"]
    ].copy()
    table_df["accuracy_pct"] = 100.0 * table_df["final_test_accuracy"]
    table_df.to_csv(RESULTS_DIR / "svm_schedule_table.csv", index=False)


def create_algorithm_family_assets(summary_df: pd.DataFrame) -> None:
    subset = summary_df[(summary_df["study"] == "algorithm_family") & (~summary_df["is_reference"])].copy()
    best = select_best_rows(subset, ["problem", "algorithm"], "final_objective_gap").sort_values(
        ["problem", "algorithm"]
    )

    algorithms = ["projected_subgradient", "dual_averaging", "adagrad_norm"]
    problem_order = list(best["problem"].drop_duplicates())
    cls_best = best[best["final_test_accuracy"].notna()].copy()
    cls_problem_order = list(cls_best["problem"].drop_duplicates())
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.arange(len(problem_order))
    for offset, algorithm in enumerate(algorithms):
        algo_rows = best[best["algorithm"] == algorithm].set_index("problem").reindex(problem_order)
        values = np.maximum(algo_rows["final_objective_gap"].to_numpy(dtype=float), 1e-8)
        axes[0].bar(x + (offset - 1) * width, values, width=width, label=pretty_algorithm(algorithm))
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([pretty_problem(name) for name in problem_order], rotation=15, ha="right")
    axes[0].set_ylabel("Final objective gap")
    axes[0].set_title("Best deterministic gap by algorithm")
    axes[0].grid(True, axis="y", alpha=0.3)

    x_cls = np.arange(len(cls_problem_order))
    for offset, algorithm in enumerate(algorithms):
        algo_rows = cls_best[cls_best["algorithm"] == algorithm].set_index("problem").reindex(cls_problem_order)
        values = 100.0 * algo_rows["final_test_accuracy"].to_numpy(dtype=float)
        positions = x_cls + (offset - 1) * width
        mask = ~np.isnan(values)
        axes[1].bar(positions[mask], values[mask], width=width, label=pretty_algorithm(algorithm))
    axes[1].set_xticks(x_cls)
    axes[1].set_xticklabels([pretty_problem(name) for name in cls_problem_order], rotation=15, ha="right")
    axes[1].set_ylabel("Final test accuracy (%)")
    axes[1].set_title("Best deterministic accuracy by algorithm")
    axes[1].grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(RESULTS_DIR / "fig_algorithm_family_overview.png", dpi=180)
    plt.close(fig)

    table_df = best[
        ["problem", "algorithm", "beta_name", "final_train_objective", "final_objective_gap", "final_test_accuracy"]
    ].copy()
    table_df["accuracy_pct"] = 100.0 * table_df["final_test_accuracy"]
    table_df.to_csv(RESULTS_DIR / "algorithm_family_table.csv", index=False)


def create_geometry_assets(summary_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    subset = summary_df[(summary_df["stage"] == "geometry") & (~summary_df["is_reference"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    plot_specs = [
        ("geometry_simplex", "simplex_lad_synthetic", axes[0], "Simplex LAD"),
        ("geometry_ill_conditioned", "ill_conditioned_logistic", axes[1], "Ill-conditioned logistic"),
    ]
    for study, problem_name, ax, title in plot_specs:
        study_summary = subset[(subset["study"] == study) & (subset["problem"] == problem_name)]
        if study_summary.empty:
            continue
        best = select_best_rows(study_summary, ["geometry"], "final_objective_gap")
        for _, row in best.iterrows():
            group = history_df[history_df["config_id"] == row["config_id"]]
            ax.plot(
                group["iteration"],
                np.maximum(group["objective_gap"], 1e-8),
                label=pretty_geometry(str(row["geometry"])),
                linewidth=1.8,
            )
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective gap")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_geometry_comparison.png", dpi=180)
    plt.close(fig)

    best_rows: list[pd.DataFrame] = []
    for study in ["geometry_simplex", "geometry_ill_conditioned"]:
        study_summary = subset[subset["study"] == study]
        if not study_summary.empty:
            best_rows.append(select_best_rows(study_summary, ["geometry"], "final_objective_gap"))
    table_df = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()
    if not table_df.empty:
        table_df = table_df[
            ["problem", "geometry", "algorithm", "beta_name", "final_objective_gap", "final_test_accuracy"]
        ].copy()
        table_df["accuracy_pct"] = 100.0 * table_df["final_test_accuracy"]
        table_df.to_csv(RESULTS_DIR / "geometry_table.csv", index=False)


def create_stochastic_assets(summary_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    subset = summary_df[(summary_df["stage"] == "stochastic") & (~summary_df["is_reference"])].copy()

    grouped = (
        subset.groupby(["problem", "algorithm", "batch_size", "noise_mode", "beta_name"])
        .agg(
            mean_final_gap=("final_objective_gap", "mean"),
            std_final_gap=("final_objective_gap", "std"),
            mean_accuracy=("final_test_accuracy", "mean"),
            std_accuracy=("final_test_accuracy", "std"),
            mean_time=("wall_clock_sec", "mean"),
            std_time=("wall_clock_sec", "std"),
            mean_stability=("final_stability", "mean"),
            std_stability=("final_stability", "std"),
            mean_grad_norm=("final_grad_norm", "mean"),
            std_grad_norm=("final_grad_norm", "std"),
            num_runs=("seed", "nunique"),
        )
        .reset_index()
    )
    grouped["mean_accuracy_pct"] = 100.0 * grouped["mean_accuracy"]
    grouped["std_accuracy_pct"] = 100.0 * grouped["std_accuracy"].fillna(0.0)
    grouped = grouped.drop(columns=["mean_accuracy", "std_accuracy"]).sort_values(
        ["problem", "batch_size", "noise_mode", "algorithm", "beta_name"]
    )
    grouped.to_csv(RESULTS_DIR / "stochastic_grouped_summary.csv", index=False)

    # Show one best noisy batch-32 trajectory per algorithm.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=False)
    algorithm_labels = {
        "projected_subgradient": "Proj. subgrad.",
        "dual_averaging": "Dual avg.",
        "adagrad_norm": "AdaGrad",
    }
    for ax, problem_name, title in zip(
        axes,
        ["svm_digits_even_odd", "logistic_digits_even_odd"],
        ["Stochastic SVM", "Stochastic logistic regression"],
    ):
        problem_subset = subset[
            (subset["problem"] == problem_name)
            & (subset["batch_size"] == 32)
            & (subset["noise_mode"] == "gaussian")
        ]
        if problem_subset.empty:
            continue
        selected = select_best_rows(
            grouped[
                (grouped["problem"] == problem_name)
                & (grouped["batch_size"] == 32)
                & (grouped["noise_mode"] == "gaussian")
            ],
            ["algorithm"],
            "mean_final_gap",
        )[["algorithm", "beta_name"]]
        for _, row in selected.iterrows():
            hist = history_df[
                (history_df["problem"] == problem_name)
                & (history_df["algorithm"] == row["algorithm"])
                & (history_df["batch_size"] == 32)
                & (history_df["noise_mode"] == "gaussian")
                & (history_df["beta_name"] == row["beta_name"])
            ]
            mean_std = hist.groupby("iteration")["objective_gap"].agg(["mean", "std"]).reset_index()
            mean = np.maximum(mean_std["mean"].to_numpy(dtype=float), 1e-8)
            std = mean_std["std"].fillna(0.0).to_numpy(dtype=float)
            lower = np.maximum(mean - std, 1e-8)
            upper = np.maximum(mean + std, 1e-8)
            label = f"{algorithm_labels[str(row['algorithm'])]}, " + rf"$\beta={pretty_beta(str(row['beta_name']))}$"
            ax.plot(mean_std["iteration"], mean, label=label, linewidth=1.8)
            ax.fill_between(mean_std["iteration"], lower, upper, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective gap")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_stochastic_mean_std.png", dpi=180)
    plt.close(fig)

    # Compare algorithms by regime after selecting the best beta inside each regime.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    algorithms = ["projected_subgradient", "dual_averaging", "adagrad_norm"]
    for ax, problem_name, title in zip(
        axes,
        ["svm_digits_even_odd", "logistic_digits_even_odd"],
        ["SVM stochastic regimes", "Logistic stochastic regimes"],
    ):
        full_batch_size = int(subset[subset["problem"] == problem_name]["batch_size"].max())
        problem_grouped = grouped[grouped["problem"] == problem_name].copy()
        problem_grouped["regime"] = problem_grouped.apply(
            lambda row: stochastic_regime_label(int(row["batch_size"]), str(row["noise_mode"]), full_batch_size),
            axis=1,
        )
        regime_order = ["full-batch", "mini-batch 32", "mini-batch 32 + noise", "single sample"]
        problem_grouped = problem_grouped[problem_grouped["regime"].isin(regime_order)]
        if problem_grouped.empty:
            continue
        best_by_regime = select_best_rows(problem_grouped, ["algorithm", "regime"], "mean_final_gap")
        x = np.arange(len(regime_order))
        width = 0.24
        for offset, algorithm in enumerate(algorithms):
            algo_rows = best_by_regime[best_by_regime["algorithm"] == algorithm].set_index("regime").reindex(regime_order)
            values = algo_rows["mean_final_gap"].to_numpy(dtype=float)
            errors = algo_rows["std_final_gap"].fillna(0.0).to_numpy(dtype=float)
            positions = x + (offset - 1) * width
            mask = ~np.isnan(values)
            ax.bar(
                positions[mask],
                np.maximum(values[mask], 1e-8),
                width=width,
                yerr=errors[mask],
                capsize=3,
                label=pretty_algorithm(algorithm),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(regime_order, rotation=15, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("Mean final objective gap")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(RESULTS_DIR / "fig_stochastic_regimes.png", dpi=180)
    plt.close(fig)

    noisy_best = select_best_rows(
        grouped[(grouped["batch_size"] == 32) & (grouped["noise_mode"] == "gaussian")],
        ["problem", "algorithm"],
        "mean_final_gap",
    ).sort_values(["problem", "algorithm"])
    noisy_best.to_csv(RESULTS_DIR / "stochastic_noisy_table.csv", index=False)

    regime_rows: list[pd.DataFrame] = []
    for problem_name in ["svm_digits_even_odd", "logistic_digits_even_odd"]:
        full_batch_size = int(subset[subset["problem"] == problem_name]["batch_size"].max())
        problem_grouped = grouped[grouped["problem"] == problem_name].copy()
        problem_grouped["regime"] = problem_grouped.apply(
            lambda row: stochastic_regime_label(int(row["batch_size"]), str(row["noise_mode"]), full_batch_size),
            axis=1,
        )
        regime_rows.append(select_best_rows(problem_grouped, ["algorithm", "regime"], "mean_final_gap"))
    pd.concat(regime_rows, ignore_index=True).to_csv(RESULTS_DIR / "stochastic_regime_best.csv", index=False)


def pretty_problem(name: str) -> str:
    return {
        "svm_digits_even_odd": "SVM (digits even/odd)",
        "logistic_digits_even_odd": "Logistic regression (digits even/odd)",
        "huber_synthetic_medium": "Huber regression (synthetic)",
        "simplex_lad_synthetic": "Simplex LAD",
        "ill_conditioned_logistic": "Ill-conditioned logistic regression",
    }.get(name, name)


def pretty_algorithm(name: str) -> str:
    return {
        "projected_subgradient": "Projected subgradient",
        "dual_averaging": "Dual averaging",
        "adagrad_norm": "AdaGrad-Norm",
        "entropy_mirror_descent": "Entropy mirror descent",
        "mahalanobis_mirror_descent": "Diagonal Mahalanobis mirror descent",
    }.get(name, name)


def pretty_geometry(name: str) -> str:
    return {
        "euclidean": "Euclidean",
        "entropy": "Entropy",
        "diagonal_mahalanobis": "Diagonal Mahalanobis",
    }.get(name, name)


def pretty_gamma(name: str) -> str:
    return {
        "c/sqrt(T)": r"$c/\sqrt{T}$",
        "c/sqrt(t)": r"$c/\sqrt{t}$",
        "c/t": r"$c/t$",
    }.get(name, str(name))


def pretty_beta(name: str) -> str:
    return {
        "0": "0",
        "0.5": "0.5",
        "1-1/sqrt(t)": r"1-1/\sqrt{t}",
    }.get(name, str(name))


def run_svm_stage(quick: bool = False) -> dict[str, float]:
    problems = build_core_problems()
    horizon = 300 if quick else 2000
    reference_horizon = 2000 if quick else 50000

    # SVM sweep and the deterministic family benchmark.
    configs = original_svm_configs(problems["svm_digits_even_odd"], horizon)
    scale_lookup: dict[str, float] = {}
    for problem_name in ["svm_digits_even_odd", "logistic_digits_even_odd", "huber_synthetic_medium"]:
        problem = problems[problem_name]
        for algorithm in ["projected_subgradient", "dual_averaging", "adagrad_norm"]:
            scale_lookup[scale_lookup_key(problem_name, algorithm)] = tune_base_scale(problem, algorithm, horizon)

    for problem_name in ["svm_digits_even_odd", "logistic_digits_even_odd", "huber_synthetic_medium"]:
        for algorithm in ["projected_subgradient", "dual_averaging", "adagrad_norm"]:
            base_scale = scale_lookup[scale_lookup_key(problem_name, algorithm)]
            for beta_name in ORIGINAL_BETA_ORDER:
                configs.append(
                    RunConfig(
                        stage="svm",
                        study="algorithm_family",
                        problem_name=problem_name,
                        algorithm=algorithm,
                        geometry="euclidean",
                        beta_name=beta_name,
                        gamma_name="c/sqrt(t)" if algorithm == "projected_subgradient" else "1",
                        base_scale=base_scale,
                        horizon=horizon,
                        batch_size=problems[problem_name].n_train,
                        seed=0,
                    )
                )

    summary_df, history_df = run_configs(problems, configs)
    reference_configs = build_reference_bank_configs("svm", problems, reference_horizon, scale_lookup)
    reference_df = compute_reference_rows(problems, reference_configs)
    summary_df, history_df = enrich_with_reference(summary_df, history_df, reference_df)
    save_stage_exports("svm", summary_df, history_df)
    create_original_svm_assets(summary_df, history_df)
    create_algorithm_family_assets(summary_df)
    return scale_lookup


def run_geometry_stage(quick: bool = False) -> None:
    problems = {
        "simplex_lad_synthetic": build_simplex_problem(seed=0),
        "ill_conditioned_logistic": build_ill_conditioned_logistic(seed=0),
    }
    horizon = 250 if quick else 1500
    reference_horizon = 2000 if quick else 50000

    scale_lookup: dict[str, float] = {}
    configs: list[RunConfig] = []

    # Simplex benchmark: Euclidean projection versus entropy geometry.
    for geometry in ["euclidean", "entropy"]:
        problem = problems["simplex_lad_synthetic"]
        algorithm = "projected_subgradient" if geometry == "euclidean" else "entropy_mirror_descent"
        key = scale_lookup_key(problem.name, algorithm)
        scale_lookup[key] = tune_base_scale(problem, algorithm, horizon, geometry=geometry)
        for beta_name in ["0", "0.5"]:
            configs.append(
                RunConfig(
                    stage="geometry",
                    study="geometry_simplex",
                    problem_name=problem.name,
                    algorithm=algorithm,
                    geometry=geometry,
                    beta_name=beta_name,
                    gamma_name="c/sqrt(t)",
                    base_scale=scale_lookup[key],
                    horizon=horizon,
                    batch_size=problem.n_train,
                    seed=0,
                )
            )

    # Ill-conditioned logistic benchmark: Euclidean versus diagonal Mahalanobis geometry.
    for geometry in ["euclidean", "diagonal_mahalanobis"]:
        problem = problems["ill_conditioned_logistic"]
        algorithm = "projected_subgradient" if geometry == "euclidean" else "mahalanobis_mirror_descent"
        key = scale_lookup_key(problem.name, algorithm)
        scale_lookup[key] = tune_base_scale(problem, algorithm, horizon, geometry=geometry)
        for beta_name in ["0", "0.5"]:
            configs.append(
                RunConfig(
                    stage="geometry",
                    study="geometry_ill_conditioned",
                    problem_name=problem.name,
                    algorithm=algorithm,
                    geometry=geometry,
                    beta_name=beta_name,
                    gamma_name="c/sqrt(t)",
                    base_scale=scale_lookup[key],
                    horizon=horizon,
                    batch_size=problem.n_train,
                    seed=0,
                )
            )

    summary_df, history_df = run_configs(problems, configs)
    reference_configs = build_reference_bank_configs("geometry", problems, reference_horizon, scale_lookup)
    reference_df = compute_reference_rows(problems, reference_configs)
    summary_df, history_df = enrich_with_reference(summary_df, history_df, reference_df)
    save_stage_exports("geometry", summary_df, history_df)
    create_geometry_assets(summary_df, history_df)


def run_stochastic_stage(quick: bool = False) -> None:
    problems = build_core_problems()
    problems = {name: problems[name] for name in ["svm_digits_even_odd", "logistic_digits_even_odd"]}
    horizon = 250 if quick else 1000
    reference_horizon = 2000 if quick else 50000

    scale_lookup: dict[str, float] = {}
    for problem_name, problem in problems.items():
        for algorithm in ["projected_subgradient", "dual_averaging", "adagrad_norm"]:
            scale_lookup[scale_lookup_key(problem_name, algorithm)] = tune_base_scale(problem, algorithm, horizon)

    noise_scales = {
        problem_name: estimate_noise_scale(
            problem,
            scale_lookup[scale_lookup_key(problem_name, "projected_subgradient")],
            horizon=min(100, horizon),
        )
        for problem_name, problem in problems.items()
    }

    configs: list[RunConfig] = []
    # Cover full-batch, mini-batch, and single-sample regimes, with an extra noisy mini-batch case.
    batch_specs = [
        (problems["svm_digits_even_odd"].n_train, "none"),
        (32, "none"),
        (32, "gaussian"),
        (1, "none"),
    ]
    regime_scale_lookup: dict[str, float] = {}
    for problem_name, problem in problems.items():
        for algorithm in ["projected_subgradient", "dual_averaging", "adagrad_norm"]:
            for batch_size, noise_mode in batch_specs:
                regime_scale_lookup[
                    stochastic_scale_lookup_key(problem_name, algorithm, batch_size, noise_mode)
                ] = tune_base_scale(
                    problem,
                    algorithm,
                    horizon,
                    batch_size=batch_size,
                    noise_mode=noise_mode,
                    noise_scale=noise_scales[problem_name] if noise_mode == "gaussian" else 0.0,
                )

    for problem_name, problem in problems.items():
        for algorithm in ["projected_subgradient", "dual_averaging", "adagrad_norm"]:
            for batch_size, noise_mode in batch_specs:
                scale = regime_scale_lookup[stochastic_scale_lookup_key(problem_name, algorithm, batch_size, noise_mode)]
                for beta_name in ORIGINAL_BETA_ORDER:
                    for seed in [0, 1, 2]:
                        configs.append(
                            RunConfig(
                                stage="stochastic",
                                study="stochastic",
                                problem_name=problem_name,
                                algorithm=algorithm,
                                geometry="euclidean",
                                beta_name=beta_name,
                                gamma_name="1" if algorithm != "projected_subgradient" else "c/sqrt(t)",
                                base_scale=scale,
                                horizon=horizon,
                                batch_size=batch_size,
                                seed=seed,
                                noise_mode=noise_mode,
                                noise_scale=noise_scales[problem_name] if noise_mode == "gaussian" else 0.0,
                            )
                        )

    summary_df, history_df = run_configs(problems, configs)
    full_batch_scale_lookup = {
        problem_name: {
            algorithm: regime_scale_lookup[
                stochastic_scale_lookup_key(problem_name, algorithm, problems[problem_name].n_train, "none")
            ]
            for algorithm in ["projected_subgradient", "dual_averaging", "adagrad_norm"]
        }
        for problem_name in problems
    }
    reference_scale_lookup = {
        scale_lookup_key(problem_name, algorithm): scale
        for problem_name, algorithm_scales in full_batch_scale_lookup.items()
        for algorithm, scale in algorithm_scales.items()
    }
    reference_configs = build_reference_bank_configs("stochastic", problems, reference_horizon, reference_scale_lookup)
    reference_df = compute_reference_rows(problems, reference_configs)
    summary_df, history_df = enrich_with_reference(summary_df, history_df, reference_df)
    save_stage_exports("stochastic", summary_df, history_df)
    create_stochastic_assets(summary_df, history_df)


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the online optimization experiments.")
    parser.add_argument(
        "--stage",
        choices=["svm", "geometry", "stochastic", "all"],
        help="Stage to run. If omitted, the script falls back to interactive prompts.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced horizons for a quick smoke test.",
    )
    return parser.parse_args()


def prompt_stage() -> str:
    print("Select experiment stage:")
    print("  1. svm")
    print("  2. geometry")
    print("  3. stochastic")
    print("  4. all")
    while True:
        try:
            choice = input("Enter choice [1-4]: ").strip().lower()
        except EOFError as exc:
            raise SystemExit("Interactive input is required to select a stage.") from exc
        stage = STAGE_CHOICES.get(choice)
        if stage is not None:
            return stage
        print("Invalid choice. Enter 1, 2, 3, 4, svm, geometry, stochastic, or all.")


def prompt_quick_mode() -> bool:
    while True:
        try:
            choice = input("Run quick mode? [y/N]: ").strip().lower()
        except EOFError as exc:
            raise SystemExit("Interactive input is required to select quick mode.") from exc
        if choice in {"", "n", "no"}:
            return False
        if choice in {"y", "yes"}:
            return True
        print("Invalid choice. Enter y or n.")


def run_stage(stage: str, quick: bool) -> None:
    mode = "quick" if quick else "full"
    if stage == "svm":
        print(f"Running svm stage ({mode} mode)...")
        run_svm_stage(quick=quick)
        return
    if stage == "geometry":
        print(f"Running geometry stage ({mode} mode)...")
        run_geometry_stage(quick=quick)
        return
    if stage == "stochastic":
        print(f"Running stochastic stage ({mode} mode)...")
        run_stochastic_stage(quick=quick)
        return

    print(f"Running all stages ({mode} mode)...")
    for stage_name, stage_fn in [
        ("svm", run_svm_stage),
        ("geometry", run_geometry_stage),
        ("stochastic", run_stochastic_stage),
    ]:
        print(f"Running {stage_name} stage ({mode} mode)...")
        stage_fn(quick=quick)


def main() -> None:
    ensure_results_dir()
    args = parse_args()
    stage = args.stage if args.stage is not None else prompt_stage()
    quick = args.quick if args.stage is not None else prompt_quick_mode()
    run_stage(stage, quick)
    print(f"Completed. Results and images are available in {RESULTS_DIR}.")


if __name__ == "__main__":
    main()
