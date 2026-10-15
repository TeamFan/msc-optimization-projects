from __future__ import annotations

import argparse
import csv
import math
import pickle
import warnings
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Configuration dataclasses keep experiment settings explicit and easy to serialize.
@dataclass(frozen=True)
class QPInstanceSpec:
    dimension: int
    condition_number: float
    lambda_min: float = 1.0
    seed: int = 0

    @property
    def lambda_max(self) -> float:
        return float(self.lambda_min * self.condition_number)


@dataclass(frozen=True)
class PaperSweepSpec:
    dimensions: tuple[int, ...] = (2, 4, 8)
    condition_numbers: tuple[float, ...] = (1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0)
    repetitions: int = 100
    figure2_repetitions: int = 100
    depth_budget: int = 18
    figure2_dimension: int = 2
    figure2_condition_number: float = 2.0
    figure2_optimum: tuple[float, float] = (1.0, 1.0)
    figure2_initial: tuple[float, float] = (3.0, 3.0)
    lambda_min: float = 1.0

    @classmethod
    def for_mode(cls, mode: str) -> "PaperSweepSpec":
        if mode == "quick":
            return cls(repetitions=6, figure2_repetitions=12)
        return cls()


@dataclass(frozen=True)
class KernelSpec:
    n_points: int = 100
    m_points: int = 10
    sigma: float = 0.5
    nu: float = 1.0
    grid_points: int = 250
    synthetic_noise: float = 0.03
    plain_iterations: int = 80

    @classmethod
    def for_mode(cls, mode: str) -> "KernelSpec":
        if mode == "quick":
            return cls(plain_iterations=30)
        return cls()


@dataclass
class RunArtifacts:
    name: str
    outdir: Path
    figure_paths: list[str] = field(default_factory=list)
    table_paths: list[str] = field(default_factory=list)
    extra_paths: list[str] = field(default_factory=list)
    summary_json: str = ""

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["outdir"] = str(self.outdir)
        return payload

# The simulated backend keeps approximate payloads while enforcing a multiplicative-depth budget.
class DepthBudgetExceeded(RuntimeError):
    pass


@dataclass
class Ciphertext:
    value: np.ndarray
    scale: float
    depth: int
    noise_norm: float = 0.0
    label: str = ""


@dataclass
class OperationTally:
    encryptions: int = 0
    decryptions: int = 0
    additions: int = 0
    subtractions: int = 0
    ciphertext_multiplications: int = 0
    plaintext_multiplications: int = 0
    rotations: int = 0
    relinearizations: int = 0
    max_depth_seen: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class SimulatedCKKSBackend:
    def __init__(
        self,
        *,
        seed: int = 0,
        depth_budget: int = 18,
        base_scale: float = 2.0**40,
        noise_scale: float = 5.0e-7,
    ) -> None:
        self.depth_budget = int(depth_budget)
        self.base_scale = float(base_scale)
        self.noise_scale = float(noise_scale)
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._stats = OperationTally()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._stats = OperationTally()

    def stats(self) -> dict[str, int]:
        return self._stats.to_dict()

    def _record_depth(self, depth: int) -> None:
        self._stats.max_depth_seen = max(self._stats.max_depth_seen, int(depth))
        if depth > self.depth_budget:
            raise DepthBudgetExceeded(
                f"Depth budget exceeded: depth={depth}, budget={self.depth_budget}."
            )

    def _noise(self, value: np.ndarray, depth: int, op_factor: float) -> tuple[np.ndarray, float]:
        arr = np.asarray(value, dtype=float)
        if self.noise_scale <= 0.0:
            return arr.copy(), 0.0
        ref = max(1.0, float(np.linalg.norm(arr)) / max(1, arr.size))
        sigma = self.noise_scale * op_factor * (1.0 + float(depth)) * ref
        perturbation = self._rng.normal(0.0, sigma, size=arr.shape)
        return arr + perturbation, float(np.linalg.norm(perturbation))

    def encrypt(self, value: Any, *, label: str = "") -> Ciphertext:
        self._stats.encryptions += 1
        arr = np.asarray(value, dtype=float)
        self._record_depth(0)
        return Ciphertext(value=arr.copy(), scale=self.base_scale, depth=0, label=label)

    def decrypt(self, ciphertext: Ciphertext) -> np.ndarray:
        self._stats.decryptions += 1
        return np.asarray(ciphertext.value, dtype=float).copy()

    def add(self, left: Ciphertext, right: Ciphertext, *, label: str = "") -> Ciphertext:
        depth = max(left.depth, right.depth)
        self._stats.additions += 1
        value, noise_norm = self._noise(left.value + right.value, depth, 0.10)
        self._record_depth(depth)
        return Ciphertext(value=value, scale=max(left.scale, right.scale), depth=depth, noise_norm=noise_norm, label=label)

    def subtract(self, left: Ciphertext, right: Ciphertext, *, label: str = "") -> Ciphertext:
        depth = max(left.depth, right.depth)
        self._stats.subtractions += 1
        value, noise_norm = self._noise(left.value - right.value, depth, 0.10)
        self._record_depth(depth)
        return Ciphertext(value=value, scale=max(left.scale, right.scale), depth=depth, noise_norm=noise_norm, label=label)

    def multiply(self, left: Ciphertext, right: Ciphertext, *, label: str = "") -> Ciphertext:
        depth = max(left.depth, right.depth) + 1
        self._stats.ciphertext_multiplications += 1
        value, noise_norm = self._noise(left.value * right.value, depth, 1.00)
        self._record_depth(depth)
        return Ciphertext(
            value=value,
            scale=left.scale * right.scale,
            depth=depth,
            noise_norm=noise_norm,
            label=label,
        )

    def multiply_plain(self, ciphertext: Ciphertext, plain: Any, *, label: str = "") -> Ciphertext:
        depth = ciphertext.depth + 1
        self._stats.plaintext_multiplications += 1
        plain_arr = np.asarray(plain, dtype=float)
        value, noise_norm = self._noise(ciphertext.value * plain_arr, depth, 0.55)
        self._record_depth(depth)
        return Ciphertext(
            value=value,
            scale=ciphertext.scale,
            depth=depth,
            noise_norm=noise_norm,
            label=label,
        )

    def rotate(self, ciphertext: Ciphertext, steps: int, *, label: str = "") -> Ciphertext:
        self._stats.rotations += 1
        value = np.roll(np.asarray(ciphertext.value, dtype=float), -int(steps))
        self._record_depth(ciphertext.depth)
        return Ciphertext(
            value=value,
            scale=ciphertext.scale,
            depth=ciphertext.depth,
            noise_norm=ciphertext.noise_norm,
            label=label,
        )

    def relinearize(self, ciphertext: Ciphertext, *, label: str = "") -> Ciphertext:
        self._stats.relinearizations += 1
        value, noise_norm = self._noise(ciphertext.value, ciphertext.depth, 0.08)
        self._record_depth(ciphertext.depth)
        return Ciphertext(
            value=value,
            scale=ciphertext.scale,
            depth=ciphertext.depth,
            noise_norm=noise_norm,
            label=label,
        )

# Row-order encoding and diagonal transforms follow the paper's matrix multiplication setup.
@dataclass(frozen=True)
class MatrixMethodCost:
    name: str
    ciphertexts: str
    complexity: str
    total_depth: int
    relinearizations: int
    plaintext_multiplications: int
    ciphertext_multiplications: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def row_encode(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("row_encode expects a square matrix.")
    return arr.reshape(-1).copy()


def row_decode(encoded: np.ndarray, dimension: int) -> np.ndarray:
    arr = np.asarray(encoded, dtype=float)
    expected = int(dimension * dimension)
    if arr.size != expected:
        raise ValueError(f"Encoded array has size {arr.size}, expected {expected}.")
    return arr.reshape((dimension, dimension)).copy()


def vector_to_repeated_column_matrix(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=float).reshape(-1)
    return np.tile(vec[:, None], (1, vec.size))


def repeated_column_matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Expected a square repeated-column matrix.")
    return arr[:, 0].copy()


@lru_cache(maxsize=None)
def build_vk_matrix(dimension: int, shift: int, scale: float) -> np.ndarray:
    d = int(dimension)
    k = int(shift)
    mat = np.zeros((d * d, d * d), dtype=float)
    for i in range(d):
        for j in range(d):
            row = d * i + j
            col = d * i + ((i + j + k) % d)
            mat[row, col] = float(scale)
    return mat


@lru_cache(maxsize=None)
def build_wk_matrix(dimension: int, shift: int, scale: float = 1.0) -> np.ndarray:
    d = int(dimension)
    k = int(shift)
    mat = np.zeros((d * d, d * d), dtype=float)
    for i in range(d):
        for j in range(d):
            row = d * i + j
            col = d * ((i + j + k) % d) + j
            mat[row, col] = float(scale)
    return mat


def _diagonals_from_matrix(transform: np.ndarray) -> tuple[np.ndarray, ...]:
    n = transform.shape[0]
    diagonals = []
    for shift in range(n):
        diag = np.array([transform[i, (i + shift) % n] for i in range(n)], dtype=float)
        diagonals.append(diag)
    return tuple(diagonals)


@lru_cache(maxsize=None)
def vk_diagonals(dimension: int, shift: int, scale: float) -> tuple[np.ndarray, ...]:
    return _diagonals_from_matrix(build_vk_matrix(dimension, shift, scale))


@lru_cache(maxsize=None)
def wk_diagonals(dimension: int, shift: int, scale: float = 1.0) -> tuple[np.ndarray, ...]:
    return _diagonals_from_matrix(build_wk_matrix(dimension, shift, scale))


def halevi_shoup_lintrans(
    ciphertext: Ciphertext,
    diagonals: tuple[np.ndarray, ...],
    backend: SimulatedCKKSBackend,
    *,
    label: str = "",
) -> Ciphertext:
    acc: Ciphertext | None = None
    for shift, diag in enumerate(diagonals):
        rotated = backend.rotate(ciphertext, shift, label=f"{label}_rot_{shift}")
        term = backend.multiply_plain(rotated, diag, label=f"{label}_diag_{shift}")
        acc = term if acc is None else backend.add(acc, term, label=f"{label}_sum_{shift}")
    if acc is None:
        raise ValueError("Linear transform requires at least one diagonal.")
    return acc


def proposed_mmult(
    left: Ciphertext,
    right: Ciphertext,
    dimension: int,
    backend: SimulatedCKKSBackend,
    *,
    left_scale: float = 1.0,
    label: str = "",
) -> Ciphertext:
    acc: Ciphertext | None = None
    for shift in range(int(dimension)):
        a_k = halevi_shoup_lintrans(
            left,
            vk_diagonals(dimension, shift, float(left_scale)),
            backend,
            label=f"{label}_vk_{shift}",
        )
        b_k = halevi_shoup_lintrans(
            right,
            wk_diagonals(dimension, shift, 1.0),
            backend,
            label=f"{label}_wk_{shift}",
        )
        prod = backend.multiply(a_k, b_k, label=f"{label}_mul_{shift}")
        prod = backend.relinearize(prod, label=f"{label}_relin_{shift}")
        acc = prod if acc is None else backend.add(acc, prod, label=f"{label}_acc_{shift}")
    if acc is None:
        raise ValueError("Matrix multiplication requires positive dimension.")
    return acc


def matrix_method_costs() -> dict[str, MatrixMethodCost]:
    return {
        "halevi_shoup": MatrixMethodCost(
            name="Halevi-Shoup",
            ciphertexts="d",
            complexity="O(d^2)",
            total_depth=1,
            relinearizations=1,
            plaintext_multiplications=0,
            ciphertext_multiplications=1,
        ),
        "jkls": MatrixMethodCost(
            name="JKLS",
            ciphertexts="1",
            complexity="O(d)",
            total_depth=3,
            relinearizations=3,
            plaintext_multiplications=2,
            ciphertext_multiplications=1,
        ),
        "proposed": MatrixMethodCost(
            name="Proposed",
            ciphertexts="1",
            complexity="O(d)",
            total_depth=2,
            relinearizations=2,
            plaintext_multiplications=1,
            ciphertext_multiplications=1,
        ),
    }


def per_iteration_depth(algorithm: str, method: str = "proposed") -> int:
    method_cost = matrix_method_costs()[method]
    if algorithm.lower() == "gd":
        return method_cost.total_depth
    if algorithm.lower() == "agd":
        return method_cost.total_depth + 1
    raise ValueError(f"Unsupported algorithm {algorithm!r}.")


def max_iterations_for_budget(depth_budget: int, algorithm: str, method: str = "proposed") -> int:
    depth = per_iteration_depth(algorithm, method)
    return int(depth_budget) // int(depth)

# Problem and trace containers separate instance generation from optimization outputs.
@dataclass
class QPInstance:
    q: np.ndarray
    p: np.ndarray
    x_star: np.ndarray
    x0: np.ndarray
    lambda_min: float
    lambda_max: float
    condition_number: float
    seed: int


@dataclass
class OptimizationTrace:
    algorithm: str
    backend: str
    states: np.ndarray
    objective_gaps: np.ndarray
    coefficient_errors: np.ndarray
    cumulative_depth: np.ndarray
    step_size: float
    momentum: float
    backend_stats: dict[str, int]


@dataclass
class KernelProblem:
    q: np.ndarray
    p: np.ndarray
    x_star: np.ndarray
    x0: np.ndarray
    x_data: np.ndarray
    y_data: np.ndarray
    x_m: np.ndarray
    sigma: float
    nu: float
    source: str


# Kernel helpers turn the thesis regression objective into the same quadratic form used by the paper.
def rbf_kernel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    a = np.asarray(x1, dtype=float).reshape(-1)
    b = np.asarray(x2, dtype=float).reshape(-1)
    diff = a[:, None] - b[None, :]
    return np.exp(-(diff**2))


def orthogonal_matrix(dimension: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, size=(dimension, dimension))
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def build_spd_qp(
    spec: QPInstanceSpec,
    *,
    rng: np.random.Generator,
    optimum: np.ndarray | None = None,
    initial: np.ndarray | None = None,
) -> QPInstance:
    lambdas = np.geomspace(spec.lambda_min, spec.lambda_max, spec.dimension)
    basis = orthogonal_matrix(spec.dimension, rng)
    q = basis @ np.diag(lambdas) @ basis.T
    q = 0.5 * (q + q.T)
    if optimum is None:
        optimum = rng.normal(0.0, 1.0, size=spec.dimension)
    x_star = np.asarray(optimum, dtype=float).reshape(spec.dimension)
    p = -(q @ x_star)
    if initial is None:
        direction = rng.normal(0.0, 1.0, size=spec.dimension)
        direction /= max(np.linalg.norm(direction), 1.0e-12)
        initial = x_star + direction
    x0 = np.asarray(initial, dtype=float).reshape(spec.dimension)
    return QPInstance(
        q=q,
        p=p,
        x_star=x_star,
        x0=x0,
        lambda_min=float(lambdas.min()),
        lambda_max=float(lambdas.max()),
        condition_number=float(lambdas.max() / lambdas.min()),
        seed=spec.seed,
    )


def objective_value(q: np.ndarray, p: np.ndarray, x: np.ndarray) -> float:
    vec = np.asarray(x, dtype=float).reshape(-1)
    return float(0.5 * vec @ q @ vec + p @ vec)


def build_trace(
    *,
    algorithm: str,
    backend: str,
    states: list[np.ndarray],
    q: np.ndarray,
    p: np.ndarray,
    x_star: np.ndarray,
    cumulative_depth: list[float],
    step_size: float,
    momentum: float = 0.0,
    backend_stats: dict[str, int] | None = None,
) -> OptimizationTrace:
    state_arr = np.vstack([np.asarray(s, dtype=float).reshape(1, -1) for s in states])
    f_star = objective_value(q, p, x_star)
    objective_gaps = np.array([max(objective_value(q, p, s) - f_star, 0.0) for s in state_arr], dtype=float)
    coefficient_errors = np.array([np.linalg.norm(s - x_star) for s in state_arr], dtype=float)
    return OptimizationTrace(
        algorithm=algorithm,
        backend=backend,
        states=state_arr,
        objective_gaps=objective_gaps,
        coefficient_errors=coefficient_errors,
        cumulative_depth=np.asarray(cumulative_depth, dtype=float),
        step_size=float(step_size),
        momentum=float(momentum),
        backend_stats=backend_stats or {},
    )


def run_plain_gd(instance: QPInstance, iterations: int) -> OptimizationTrace:
    eta = 2.0 / (instance.lambda_min + instance.lambda_max)
    x = instance.x0.copy()
    states = [x.copy()]
    depth_axis = [0.0]
    for step in range(int(iterations)):
        x = x - eta * (instance.q @ x + instance.p)
        states.append(x.copy())
        depth_axis.append(float(step + 1))
    return build_trace(
        algorithm="GD",
        backend="plain",
        states=states,
        q=instance.q,
        p=instance.p,
        x_star=instance.x_star,
        cumulative_depth=depth_axis,
        step_size=eta,
    )


def run_plain_agd(instance: QPInstance, iterations: int) -> OptimizationTrace:
    eta = 1.0 / instance.lambda_max
    sqrt_kappa = math.sqrt(instance.condition_number)
    beta = (sqrt_kappa - 1.0) / (sqrt_kappa + 1.0)
    x = instance.x0.copy()
    y_prev = instance.x0.copy()
    states = [x.copy()]
    depth_axis = [0.0]
    for step in range(int(iterations)):
        y = x - eta * (instance.q @ x + instance.p)
        x = (1.0 + beta) * y - beta * y_prev
        y_prev = y
        states.append(x.copy())
        depth_axis.append(float(step + 1))
    return build_trace(
        algorithm="AGD",
        backend="plain",
        states=states,
        q=instance.q,
        p=instance.p,
        x_star=instance.x_star,
        cumulative_depth=depth_axis,
        step_size=eta,
        momentum=beta,
    )


def _encrypt_qp_operands(instance: QPInstance, backend: SimulatedCKKSBackend) -> tuple[Any, Any, Any]:
    q_ct = backend.encrypt(row_encode(instance.q), label="Q")
    p_matrix = vector_to_repeated_column_matrix(instance.p)
    x_matrix = vector_to_repeated_column_matrix(instance.x0)
    p_ct = backend.encrypt(row_encode(p_matrix), label="p")
    x_ct = backend.encrypt(row_encode(x_matrix), label="x0")
    return q_ct, p_ct, x_ct


def run_simulated_he_gd(instance: QPInstance, *, seed: int, depth_budget: int = 18) -> OptimizationTrace:
    backend = SimulatedCKKSBackend(seed=seed, depth_budget=depth_budget)
    q_ct, p_ct, x_ct = _encrypt_qp_operands(instance, backend)
    eta = 2.0 / (instance.lambda_min + instance.lambda_max)
    iterations = max_iterations_for_budget(depth_budget, "gd", "proposed")
    states = [instance.x0.copy()]
    depth_axis = [0.0]
    for _ in range(iterations):
        grad_ct = proposed_mmult(q_ct, x_ct, instance.q.shape[0], backend, left_scale=eta, label="gd_grad")
        p_step = backend.multiply_plain(p_ct, eta, label="gd_p")
        tmp = backend.subtract(x_ct, grad_ct, label="gd_sub_grad")
        x_ct = backend.subtract(tmp, p_step, label="gd_step")
        mat = row_decode(backend.decrypt(x_ct), instance.q.shape[0])
        states.append(repeated_column_matrix_to_vector(mat))
        depth_axis.append(float(x_ct.depth))
    return build_trace(
        algorithm="GD",
        backend="simulated_ckks",
        states=states,
        q=instance.q,
        p=instance.p,
        x_star=instance.x_star,
        cumulative_depth=depth_axis,
        step_size=eta,
        backend_stats=backend.stats(),
    )


def run_simulated_he_agd(instance: QPInstance, *, seed: int, depth_budget: int = 18) -> OptimizationTrace:
    backend = SimulatedCKKSBackend(seed=seed, depth_budget=depth_budget)
    q_ct = backend.encrypt(row_encode(instance.q), label="Q")
    p_matrix = vector_to_repeated_column_matrix(instance.p)
    x_matrix = vector_to_repeated_column_matrix(instance.x0)
    p_ct = backend.encrypt(row_encode(p_matrix), label="p")
    x_ct = backend.encrypt(row_encode(x_matrix), label="x0")
    y_ct = backend.encrypt(row_encode(x_matrix), label="y0")

    eta = 1.0 / instance.lambda_max
    sqrt_kappa = math.sqrt(instance.condition_number)
    beta = (sqrt_kappa - 1.0) / (sqrt_kappa + 1.0)
    iterations = max_iterations_for_budget(depth_budget, "agd", "proposed")
    states = [instance.x0.copy()]
    depth_axis = [0.0]
    for _ in range(iterations):
        grad_ct = proposed_mmult(q_ct, x_ct, instance.q.shape[0], backend, left_scale=eta, label="agd_grad")
        p_step = backend.multiply_plain(p_ct, eta, label="agd_p")
        tmp = backend.subtract(x_ct, grad_ct, label="agd_sub_grad")
        y_next = backend.subtract(tmp, p_step, label="agd_y")
        term1 = backend.multiply_plain(y_next, 1.0 + beta, label="agd_term1")
        term2 = backend.multiply_plain(y_ct, beta, label="agd_term2")
        x_next = backend.subtract(term1, term2, label="agd_x")
        x_ct = backend.relinearize(x_next, label="agd_relin_x")
        y_ct = y_next
        mat = row_decode(backend.decrypt(x_ct), instance.q.shape[0])
        states.append(repeated_column_matrix_to_vector(mat))
        depth_axis.append(float(x_ct.depth))
    return build_trace(
        algorithm="AGD",
        backend="simulated_ckks",
        states=states,
        q=instance.q,
        p=instance.p,
        x_star=instance.x_star,
        cumulative_depth=depth_axis,
        step_size=eta,
        momentum=beta,
        backend_stats=backend.stats(),
    )


def project_file_path(filename: str) -> Path:
    return Path(__file__).resolve().with_name(filename)


def load_kernel_problem(
    *,
    spec: KernelSpec,
    seed: int,
    dataset_path: Path | None = None,
) -> KernelProblem:
    rng = np.random.default_rng(seed)
    db_path = dataset_path or (Path(__file__).resolve().parent.parent / "coding-project" / "data" / "first_database.pkl")
    if db_path.exists():
        with open(db_path, "rb") as handle:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                x_raw, y_raw = pickle.load(handle)
        x = np.asarray(x_raw, dtype=float)[: spec.n_points]
        y = np.asarray(y_raw, dtype=float)[: spec.n_points]
        source = db_path.name
    else:
        x = np.sort(rng.uniform(-1.0, 1.0, size=spec.n_points))
        y = (
            np.sin(3.0 * x)
            + 0.25 * np.cos(5.0 * x)
            + 0.15 * x**3
            + rng.normal(0.0, spec.synthetic_noise, size=spec.n_points)
        )
        source = "synthetic_1d_kernel_dataset"

    chosen = np.sort(rng.choice(x.size, size=spec.m_points, replace=False))
    x_m = x[chosen]
    k_mm = rbf_kernel(x_m, x_m)
    m_mat = rbf_kernel(x, x_m)
    q = (spec.sigma**2) * k_mm + m_mat.T @ m_mat + spec.nu * np.eye(spec.m_points)
    p = -(m_mat.T @ y)
    x_star = np.linalg.solve(q, -p)
    x0 = np.zeros(spec.m_points, dtype=float)
    return KernelProblem(
        q=q,
        p=p,
        x_star=x_star,
        x0=x0,
        x_data=x,
        y_data=y,
        x_m=x_m,
        sigma=spec.sigma,
        nu=spec.nu,
        source=source,
    )


def kernel_problem_to_qp(problem: KernelProblem, seed: int) -> QPInstance:
    eigvals = np.linalg.eigvalsh(problem.q)
    return QPInstance(
        q=problem.q.copy(),
        p=problem.p.copy(),
        x_star=problem.x_star.copy(),
        x0=problem.x0.copy(),
        lambda_min=float(eigvals.min()),
        lambda_max=float(eigvals.max()),
        condition_number=float(eigvals.max() / eigvals.min()),
        seed=seed,
    )


def kernel_prediction(problem: KernelProblem, coeffs: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    return rbf_kernel(np.asarray(x_query, dtype=float), problem.x_m) @ np.asarray(coeffs, dtype=float)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

matplotlib.rc("font", family="sans-serif", size=10)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["axes.labelsize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 9
matplotlib.rcParams["ytick.labelsize"] = 9
matplotlib.rcParams["legend.fontsize"] = 9
matplotlib.rcParams["figure.titlesize"] = 10


# Output helpers keep the export surface compact and submission-friendly.
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def artifact_ref(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def save_figure(fig: Any, path: Path) -> None:
    fig.tight_layout(pad=0.45)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def normalized_outdir(base_outdir: Path, command: str) -> Path:
    if command == "all":
        return base_outdir
    suffix = {
        "paper": "paper",
        "kernel": "kernel",
        "validate": "validate",
    }[command]
    if base_outdir.name.lower() == suffix:
        return base_outdir
    return base_outdir / suffix


def print_run_summary(artifacts: RunArtifacts) -> None:
    summary_path = artifacts.summary_json or "(no summary file written)"
    print(f"[{artifacts.name}] wrote artifacts to {artifacts.outdir}")
    print(
        "  "
        f"figures={len(artifacts.figure_paths)}, "
        f"tables={len(artifacts.table_paths)}, "
        f"extras={len(artifacts.extra_paths)}, "
        f"summary={summary_path}"
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def style(index: int) -> dict[str, object]:
    colors = list(plt.cm.tab10.colors)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    return {
        "color": colors[index % len(colors)],
        "marker": markers[index % len(markers)],
        "linewidth": 2.0,
        "markersize": 4.0,
        "markevery": 1,
        "alpha": 0.95,
    }


def trace_summary(trace: OptimizationTrace) -> dict[str, object]:
    return {
        "algorithm": trace.algorithm,
        "backend": trace.backend,
        "iterations": int(trace.states.shape[0] - 1),
        "final_gap": float(trace.objective_gaps[-1]),
        "final_error": float(trace.coefficient_errors[-1]),
        "final_depth": float(trace.cumulative_depth[-1]),
        "step_size": float(trace.step_size),
        "momentum": float(trace.momentum),
    }


def _plot_grouped_boxplots(
    ax: Any,
    grouped: dict[tuple[int, float], list[float]],
    dimensions: tuple[int, ...],
    kappas: tuple[float, ...],
    title: str,
) -> None:
    base = np.arange(len(dimensions), dtype=float)
    width = 0.10
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(kappas)))
    for idx, kappa in enumerate(kappas):
        positions = base + (idx - (len(kappas) - 1) / 2.0) * width
        data = [grouped[(dimension, kappa)] for dimension in dimensions]
        box = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[idx])
            patch.set_alpha(0.75)
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.1)
    ax.set_xticks(base)
    ax.set_xticklabels([str(dimension) for dimension in dimensions])
    ax.set_yscale("log")
    ax.set_xlabel("Dimension d")
    ax.set_ylabel(r"Optimality gap $f(x)-f(x^*)$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.30)
    handles = [plt.Line2D([0], [0], color=colors[idx], lw=6) for idx in range(len(kappas))]
    labels = [f"kappa={kappa:g}" for kappa in kappas]
    ax.legend(handles, labels, framealpha=0.9, ncol=2)

# The paper pipeline reproduces the depth-limited GD versus AGD comparison and writes report-ready artifacts.
def run_paper_pipeline(
    *,
    outdir: Path,
    mode: str,
    seed: int,
    backend_name: str,
) -> RunArtifacts:
    if backend_name != "simulated_ckks":
        raise ValueError("Only the simulated_ckks backend is implemented in this environment.")

    ensure_dir(outdir)
    artifacts = RunArtifacts(name="paper", outdir=outdir)
    spec = PaperSweepSpec.for_mode(mode)
    metrics_rows: list[dict[str, object]] = []
    table2_rows: list[dict[str, object]] = []
    figure2_seeds = [seed + rep for rep in range(spec.figure2_repetitions)]

    method_costs = matrix_method_costs()
    for key in ["halevi_shoup", "jkls", "proposed"]:
        cost = method_costs[key]
        row = {
            "method": cost.name,
            "ciphertexts": cost.ciphertexts,
            "complexity": cost.complexity,
            "matrix_mult_depth": cost.total_depth,
            "plaintext_multiplications": cost.plaintext_multiplications,
            "ciphertext_multiplications": cost.ciphertext_multiplications,
            "relinearizations": cost.relinearizations,
            "gd_iterations_at_budget18": max_iterations_for_budget(spec.depth_budget, "gd", key if key != "halevi_shoup" else "proposed"),
            "agd_iterations_at_budget18": max_iterations_for_budget(spec.depth_budget, "agd", key if key != "halevi_shoup" else "proposed"),
        }
        if key == "halevi_shoup":
            row["gd_iterations_at_budget18"] = ""
            row["agd_iterations_at_budget18"] = ""
        table2_rows.append(row)

    table2_csv = outdir / "table2_matrix_method_costs.csv"
    write_csv(table2_csv, table2_rows)
    artifacts.table_paths.append(artifact_ref(table2_csv, outdir))

    fig2_rng = np.random.default_rng(seed + 991)
    fig2_instance = build_spd_qp(
        QPInstanceSpec(
            dimension=spec.figure2_dimension,
            condition_number=spec.figure2_condition_number,
            lambda_min=spec.lambda_min,
            seed=seed + 991,
        ),
        rng=fig2_rng,
        optimum=np.asarray(spec.figure2_optimum, dtype=float),
        initial=np.asarray(spec.figure2_initial, dtype=float),
    )
    trajectories = []
    for rep_seed in figure2_seeds:
        trace = run_simulated_he_agd(fig2_instance, seed=rep_seed, depth_budget=spec.depth_budget)
        trajectories.append(trace.states)
    fig = plt.figure(figsize=(7.2, 5.8))
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1], color="#5b8ff9", alpha=0.18, linewidth=1.0)
        plt.scatter(traj[:, 0], traj[:, 1], color="#5b8ff9", alpha=0.12, s=10)
    plt.scatter([fig2_instance.x_star[0]], [fig2_instance.x_star[1]], color="black", s=40, label="Optimal point")
    plt.scatter([fig2_instance.x0[0]], [fig2_instance.x0[1]], color="#d62728", s=35, label="Initial point")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("Figure 2 Analogue: Decrypted HE-AGD Trajectories")
    plt.grid(True, alpha=0.30)
    plt.legend(framealpha=0.9)
    fig2_path = outdir / "paper_figure2_he_agd_trajectories.pdf"
    save_figure(fig, fig2_path)
    artifacts.figure_paths.append(artifact_ref(fig2_path, outdir))

    grouped_agd: dict[tuple[int, float], list[float]] = {}
    grouped_gd: dict[tuple[int, float], list[float]] = {}
    heatmap = np.zeros((len(spec.dimensions), len(spec.condition_numbers)), dtype=float)
    depth_curve_curves: list[dict[str, object]] = []

    for d_idx, dimension in enumerate(spec.dimensions):
        for k_idx, kappa in enumerate(spec.condition_numbers):
            gd_gaps = []
            agd_gaps = []
            gd_depth_curves = []
            agd_depth_curves = []
            for rep in range(spec.repetitions):
                local_seed = seed + 10000 * d_idx + 100 * k_idx + rep
                rng = np.random.default_rng(local_seed)
                instance = build_spd_qp(
                    QPInstanceSpec(
                        dimension=dimension,
                        condition_number=kappa,
                        lambda_min=spec.lambda_min,
                        seed=local_seed,
                    ),
                    rng=rng,
                )
                gd_trace = run_simulated_he_gd(instance, seed=local_seed + 500000, depth_budget=spec.depth_budget)
                agd_trace = run_simulated_he_agd(instance, seed=local_seed + 600000, depth_budget=spec.depth_budget)
                gd_gap = float(gd_trace.objective_gaps[-1])
                agd_gap = float(agd_trace.objective_gaps[-1])
                gd_gaps.append(gd_gap)
                agd_gaps.append(agd_gap)
                gd_depth_curves.append(gd_trace.objective_gaps)
                agd_depth_curves.append(agd_trace.objective_gaps)
                metrics_rows.extend(
                    [
                        {
                            "dimension": dimension,
                            "condition_number": kappa,
                            "rep": rep,
                            "algorithm": "GD",
                            "backend": "simulated_ckks",
                            "final_gap": gd_gap,
                            "final_error": float(gd_trace.coefficient_errors[-1]),
                            "final_depth": float(gd_trace.cumulative_depth[-1]),
                            "iterations": int(gd_trace.states.shape[0] - 1),
                        },
                        {
                            "dimension": dimension,
                            "condition_number": kappa,
                            "rep": rep,
                            "algorithm": "AGD",
                            "backend": "simulated_ckks",
                            "final_gap": agd_gap,
                            "final_error": float(agd_trace.coefficient_errors[-1]),
                            "final_depth": float(agd_trace.cumulative_depth[-1]),
                            "iterations": int(agd_trace.states.shape[0] - 1),
                        },
                    ]
                )
            grouped_gd[(dimension, kappa)] = gd_gaps
            grouped_agd[(dimension, kappa)] = agd_gaps
            heatmap[d_idx, k_idx] = math.log10(max(np.median(agd_gaps), 1.0e-16) / max(np.median(gd_gaps), 1.0e-16))
            if kappa in (2.0, 10.0, 50.0) and dimension == 8:
                depth_curve_curves.append(
                    {
                        "dimension": dimension,
                        "condition_number": kappa,
                        "algorithm": "GD",
                        "depth": np.arange(len(np.median(np.vstack(gd_depth_curves), axis=0))) * per_iteration_depth("gd"),
                        "gap": np.median(np.vstack(gd_depth_curves), axis=0),
                    }
                )
                depth_curve_curves.append(
                    {
                        "dimension": dimension,
                        "condition_number": kappa,
                        "algorithm": "AGD",
                        "depth": np.arange(len(np.median(np.vstack(agd_depth_curves), axis=0))) * per_iteration_depth("agd"),
                        "gap": np.median(np.vstack(agd_depth_curves), axis=0),
                    }
                )

    fig3_path = outdir / "paper_figure3_gd_agd_boxplots.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), sharey=True)
    _plot_grouped_boxplots(axes[0], grouped_agd, spec.dimensions, spec.condition_numbers, "HE-AGD at depth-limited final iterate")
    _plot_grouped_boxplots(axes[1], grouped_gd, spec.dimensions, spec.condition_numbers, "HE-GD at depth-limited final iterate")
    save_figure(fig, fig3_path)
    artifacts.figure_paths.append(artifact_ref(fig3_path, outdir))

    table3_rows: list[dict[str, object]] = []
    for dimension in spec.dimensions:
        for kappa in spec.condition_numbers:
            med_gd = float(np.median(grouped_gd[(dimension, kappa)]))
            med_agd = float(np.median(grouped_agd[(dimension, kappa)]))
            table3_rows.append(
                {
                    "dimension": dimension,
                    "condition_number": kappa,
                    "median_gd_gap": med_gd,
                    "median_agd_gap": med_agd,
                    "preferred": "AGD" if med_agd < med_gd else "GD",
                }
            )
    table3_csv = outdir / "table3_gd_vs_agd_medians.csv"
    write_csv(table3_csv, table3_rows)
    artifacts.table_paths.append(artifact_ref(table3_csv, outdir))

    fig = plt.figure(figsize=(8.8, 4.8))
    ax = fig.gca()
    im = ax.imshow(heatmap, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(spec.condition_numbers)))
    ax.set_xticklabels([f"{kappa:g}" for kappa in spec.condition_numbers])
    ax.set_yticks(np.arange(len(spec.dimensions)))
    ax.set_yticklabels([str(d) for d in spec.dimensions])
    ax.set_xlabel("Condition number kappa")
    ax.set_ylabel("Dimension d")
    ax.set_title("Win-region heatmap: log10(median AGD gap / median GD gap)")
    plt.colorbar(im, ax=ax, label="Negative means AGD is better")
    heatmap_path = outdir / "paper_gd_vs_agd_win_region_heatmap.pdf"
    save_figure(fig, heatmap_path)
    artifacts.figure_paths.append(artifact_ref(heatmap_path, outdir))

    fig = plt.figure(figsize=(8.4, 5.4))
    ax = fig.gca()
    for idx, curve in enumerate(depth_curve_curves):
        ax.plot(
            curve["depth"],
            np.maximum(curve["gap"], 1.0e-14),
            label=f"{curve['algorithm']} kappa={curve['condition_number']:g}, d={curve['dimension']}",
            **style(idx),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Consumed multiplicative depth")
    ax.set_ylabel(r"Median optimality gap")
    ax.set_title("Depth consumed versus solution quality")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(framealpha=0.9)
    depth_path = outdir / "paper_depth_consumed_vs_error.pdf"
    save_figure(fig, depth_path)
    artifacts.figure_paths.append(artifact_ref(depth_path, outdir))

    fig = plt.figure(figsize=(8.8, 5.4))
    methods = ["halevi_shoup", "jkls", "proposed"]
    names = [matrix_method_costs()[name].name for name in methods]
    depths = [matrix_method_costs()[name].total_depth for name in methods]
    relins = [matrix_method_costs()[name].relinearizations for name in methods]
    gd_caps = [0, max_iterations_for_budget(spec.depth_budget, "gd", "jkls"), max_iterations_for_budget(spec.depth_budget, "gd", "proposed")]
    x_axis = np.arange(len(methods), dtype=float)
    bar_w = 0.25
    plt.bar(x_axis - bar_w, depths, width=bar_w, label="Depth per matrix multiply")
    plt.bar(x_axis, relins, width=bar_w, label="Relinearizations")
    plt.bar(x_axis + bar_w, gd_caps, width=bar_w, label="GD iterations @ budget 18")
    plt.xticks(x_axis, names)
    plt.ylabel("Count")
    plt.title("Matrix multiplication method comparison")
    plt.grid(True, axis="y", alpha=0.30)
    plt.legend(framealpha=0.9)
    mmult_path = outdir / "paper_matrix_method_comparison.pdf"
    save_figure(fig, mmult_path)
    artifacts.figure_paths.append(artifact_ref(mmult_path, outdir))

    metrics_csv = outdir / "paper_metrics.csv"
    write_csv(metrics_csv, metrics_rows)
    artifacts.extra_paths.append(artifact_ref(metrics_csv, outdir))

    low_pref = [row for row in table3_rows if row["condition_number"] <= 5.0]
    high_pref = [row for row in table3_rows if row["condition_number"] >= 10.0]
    low_gd_majority = sum(1 for row in low_pref if row["preferred"] == "GD") >= max(1, len(low_pref) // 2)
    high_agd_majority = sum(1 for row in high_pref if row["preferred"] == "AGD") >= max(1, len(high_pref) // 2)
    summary = {
        "mode": mode,
        "seed": seed,
        "backend": backend_name,
        "depth_budget": spec.depth_budget,
        "figure2_repetitions": spec.figure2_repetitions,
        "sweep_repetitions": spec.repetitions,
        "gd_iteration_cap": max_iterations_for_budget(spec.depth_budget, "gd", "proposed"),
        "agd_iteration_cap": max_iterations_for_budget(spec.depth_budget, "agd", "proposed"),
        "qualitative_result_verified": bool(low_gd_majority and high_agd_majority),
        "table2_paths": [artifact_ref(table2_csv, outdir)],
        "table3_paths": [artifact_ref(table3_csv, outdir)],
        "figure_paths": artifacts.figure_paths,
        "metrics_csv": artifact_ref(metrics_csv, outdir),
    }
    summary_json = outdir / "paper_summary.json"
    write_json(summary_json, summary)
    artifacts.summary_json = artifact_ref(summary_json, outdir)
    artifacts.extra_paths.append(artifact_ref(summary_json, outdir))
    return artifacts


def _build_kernel_trace_rows(traces: Iterable[OptimizationTrace], label: str) -> list[dict[str, object]]:
    rows = []
    for trace in traces:
        row = trace_summary(trace)
        row["label"] = label
        rows.append(row)
    return rows


# The kernel pipeline reuses the same solver core on the thesis kernel-regression quadratic program.
def run_kernel_pipeline(
    *,
    outdir: Path,
    mode: str,
    seed: int,
    backend_name: str,
) -> RunArtifacts:
    if backend_name != "simulated_ckks":
        raise ValueError("Only the simulated_ckks backend is implemented in this environment.")

    ensure_dir(outdir)
    artifacts = RunArtifacts(name="kernel", outdir=outdir)
    spec = KernelSpec.for_mode(mode)
    kernel_problem = load_kernel_problem(spec=spec, seed=seed)
    qp = kernel_problem_to_qp(kernel_problem, seed)

    plain_gd = run_plain_gd(qp, iterations=spec.plain_iterations)
    plain_agd = run_plain_agd(qp, iterations=spec.plain_iterations)
    he_gd = run_simulated_he_gd(qp, seed=seed + 700000)
    he_agd = run_simulated_he_agd(qp, seed=seed + 800000)

    traces = [plain_gd, plain_agd, he_gd, he_agd]
    metrics_rows = _build_kernel_trace_rows(traces, kernel_problem.source)
    metrics_csv = outdir / "kernel_metrics.csv"
    write_csv(metrics_csv, metrics_rows)
    artifacts.extra_paths.append(artifact_ref(metrics_csv, outdir))

    fig_curves = outdir / "kernel_objective_and_error_curves.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8))
    for idx, trace in enumerate(traces):
        axes[0].plot(
            np.arange(trace.states.shape[0]),
            np.maximum(trace.objective_gaps, 1.0e-14),
            label=f"{trace.backend} {trace.algorithm}",
            **style(idx),
        )
        axes[1].plot(
            np.arange(trace.states.shape[0]),
            np.maximum(trace.coefficient_errors, 1.0e-14),
            label=f"{trace.backend} {trace.algorithm}",
            **style(idx),
        )
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_xlabel("Iterations")
    axes[1].set_xlabel("Iterations")
    axes[0].set_ylabel(r"Objective gap $F(\alpha)-F(\alpha^*)$")
    axes[1].set_ylabel(r"Coefficient error $\|\alpha-\alpha^*\|_2$")
    axes[0].set_title("Kernel QP objective gap")
    axes[1].set_title("Kernel QP coefficient error")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.30)
        ax.legend(framealpha=0.9)
    save_figure(fig, fig_curves)
    artifacts.figure_paths.append(artifact_ref(fig_curves, outdir))

    fig_depth = outdir / "kernel_depth_budget_comparison.pdf"
    fig = plt.figure(figsize=(8.0, 5.6))
    ax = fig.gca()
    for idx, trace in enumerate([he_gd, he_agd]):
        ax.plot(
            trace.cumulative_depth,
            np.maximum(trace.objective_gaps, 1.0e-14),
            label=f"{trace.algorithm} simulated CKKS",
            **style(idx),
        )
    ax.axvline(18.0, color="black", linestyle="--", linewidth=1.1, label="Depth budget")
    ax.set_yscale("log")
    ax.set_xlabel("Consumed multiplicative depth")
    ax.set_ylabel(r"Objective gap $F(\alpha)-F(\alpha^*)$")
    ax.set_title("Kernel extension: depth budget comparison")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(framealpha=0.9)
    save_figure(fig, fig_depth)
    artifacts.figure_paths.append(artifact_ref(fig_depth, outdir))

    if kernel_problem.x_data.ndim == 1:
        x_grid = np.linspace(float(kernel_problem.x_data.min()), float(kernel_problem.x_data.max()), spec.grid_points)
        curves = [
            ("Optimal closed form", kernel_problem.x_star, "black"),
            ("Plain GD", plain_gd.states[-1], "#1f77b4"),
            ("Plain AGD", plain_agd.states[-1], "#ff7f0e"),
            ("HE GD", he_gd.states[-1], "#2ca02c"),
            ("HE AGD", he_agd.states[-1], "#d62728"),
        ]
        fig = plt.figure(figsize=(8.2, 5.6))
        plt.scatter(kernel_problem.x_data, kernel_problem.y_data, s=10, alpha=0.35, color="gray", label="Samples")
        for idx, (label, coeffs, color) in enumerate(curves):
            y_hat = kernel_prediction(kernel_problem, coeffs, x_grid)
            plt.plot(x_grid, y_hat, color=color, linewidth=2.2 if idx == 0 else 1.9, alpha=0.95, label=label)
        plt.xlabel("x")
        plt.ylabel("Predicted y")
        plt.title("Kernel extension: 1D reconstruction")
        plt.grid(True, alpha=0.30)
        plt.legend(framealpha=0.9, ncol=2)
        recon_path = outdir / "kernel_reconstruction.pdf"
        save_figure(fig, recon_path)
        artifacts.figure_paths.append(artifact_ref(recon_path, outdir))

    summary = {
        "mode": mode,
        "seed": seed,
        "backend": backend_name,
        "source": kernel_problem.source,
        "condition_number": float(qp.condition_number),
        "dimension": int(qp.q.shape[0]),
        "sigma": float(kernel_problem.sigma),
        "nu": float(kernel_problem.nu),
        "plain_gd": trace_summary(plain_gd),
        "plain_agd": trace_summary(plain_agd),
        "he_gd": trace_summary(he_gd),
        "he_agd": trace_summary(he_agd),
        "figure_paths": artifacts.figure_paths,
        "metrics_csv": artifact_ref(metrics_csv, outdir),
    }
    summary_json = outdir / "kernel_summary.json"
    write_json(summary_json, summary)
    artifacts.summary_json = artifact_ref(summary_json, outdir)
    artifacts.extra_paths.append(artifact_ref(summary_json, outdir))
    return artifacts


def run_validation_pipeline(
    *,
    outdir: Path,
    mode: str,
    seed: int,
    backend_name: str,
) -> RunArtifacts:
    ensure_dir(outdir)
    artifacts = RunArtifacts(name="validate", outdir=outdir)
    vk = build_vk_matrix(4, 1, 0.25)
    wk = build_wk_matrix(4, 1, 1.0)
    cost_model = matrix_method_costs()
    cost_checks = {
        "proposed_plain_mults_lt_jkls": cost_model["proposed"].plaintext_multiplications == cost_model["jkls"].plaintext_multiplications - 1,
        "proposed_gd_cap_eq_9": max_iterations_for_budget(18, "gd", "proposed") == 9,
        "proposed_agd_cap_eq_6": max_iterations_for_budget(18, "agd", "proposed") == 6,
        "jkls_gd_cap_eq_6": max_iterations_for_budget(18, "gd", "jkls") == 6,
        "jkls_agd_cap_eq_4": max_iterations_for_budget(18, "agd", "jkls") == 4,
    }
    qp = build_spd_qp(
        QPInstanceSpec(dimension=4, condition_number=10.0, seed=seed),
        rng=np.random.default_rng(seed),
    )
    eigvals = np.linalg.eigvalsh(qp.q)
    qp_checks = {
        "condition_number_match": bool(np.isclose(eigvals.max() / eigvals.min(), 10.0, rtol=1.0e-10)),
        "stationarity_match": bool(np.allclose(qp.q @ qp.x_star + qp.p, np.zeros_like(qp.x_star), atol=1.0e-10)),
    }
    summary = {
        "mode": mode,
        "backend": backend_name,
        "vk_shape": list(vk.shape),
        "wk_shape": list(wk.shape),
        "vk_row_sums_unique": sorted({float(v) for v in np.unique(vk.sum(axis=1))}),
        "wk_row_sums_unique": sorted({float(v) for v in np.unique(wk.sum(axis=1))}),
        "cost_checks": cost_checks,
        "qp_checks": qp_checks,
    }
    summary_json = outdir / "validation_summary.json"
    write_json(summary_json, summary)
    artifacts.summary_json = artifact_ref(summary_json, outdir)
    artifacts.extra_paths.append(artifact_ref(summary_json, outdir))
    return artifacts

# The CLI keeps submission usage to one command surface.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone reproduction package for homomorphic-encrypted GD/AGD QP experiments."
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mode", choices=["quick", "full"], default="full")
    common.add_argument("--backend", choices=["simulated_ckks"], default="simulated_ckks")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument(
        "--outdir",
        type=str,
        default="results",
        help="Root directory for generated artifacts. Single-command runs create paper/kernel/validate subdirectories unless the directory already ends with that name.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("paper", parents=[common], help="Run the paper reproduction pipeline.")
    subparsers.add_parser("kernel", parents=[common], help="Run the kernel-extension pipeline.")
    subparsers.add_parser("all", parents=[common], help="Run paper, kernel, and validation outputs.")
    subparsers.add_parser("validate", parents=[common], help="Run internal deterministic validation checks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    base_outdir = Path(args.outdir).resolve()
    if args.command == "paper":
        artifacts = run_paper_pipeline(
            outdir=normalized_outdir(base_outdir, "paper"),
            mode=args.mode,
            seed=args.seed,
            backend_name=args.backend,
        )
        print_run_summary(artifacts)
        return 0
    if args.command == "kernel":
        artifacts = run_kernel_pipeline(
            outdir=normalized_outdir(base_outdir, "kernel"),
            mode=args.mode,
            seed=args.seed,
            backend_name=args.backend,
        )
        print_run_summary(artifacts)
        return 0
    if args.command == "validate":
        artifacts = run_validation_pipeline(
            outdir=normalized_outdir(base_outdir, "validate"),
            mode=args.mode,
            seed=args.seed,
            backend_name=args.backend,
        )
        print_run_summary(artifacts)
        return 0
    if args.command == "all":
        paper_artifacts = run_paper_pipeline(
            outdir=base_outdir / "paper",
            mode=args.mode,
            seed=args.seed,
            backend_name=args.backend,
        )
        kernel_artifacts = run_kernel_pipeline(
            outdir=base_outdir / "kernel",
            mode=args.mode,
            seed=args.seed,
            backend_name=args.backend,
        )
        validation_artifacts = run_validation_pipeline(
            outdir=base_outdir / "validate",
            mode=args.mode,
            seed=args.seed,
            backend_name=args.backend,
        )
        for artifacts in [paper_artifacts, kernel_artifacts, validation_artifacts]:
            print_run_summary(artifacts)
        return 0
    parser.error(f"Unsupported command {args.command!r}.")
    return 2



if __name__ == "__main__":
    raise SystemExit(main())

