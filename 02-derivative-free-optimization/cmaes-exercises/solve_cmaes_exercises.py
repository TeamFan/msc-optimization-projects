from __future__ import annotations

import json
from pathlib import Path

import cma
import matplotlib.pyplot as plt
import numpy as np


N_DIM = 10
SIGMA0 = 1.0
SEED = 42
ROTATION_SEED = 2026
X0 = np.ones(N_DIM, dtype=float)
Q7_TARGET = 1e-6
Q7_N_RUNS = 6
Q7_BASE_SEED = 1000
Q7_CMA_MAXFEVALS = 2_000_000
Q7_ONEPLUS_MAXFEVALS = 500_000
Q7_ONEPLUS_C = 0.85
Q7_ONEPLUS_WINDOW = N_DIM
Q9_SIGMA0 = 10.0
Q9_TARGET = 1e-8
Q9_N_RUNS = 5
Q9_X0_SEED = 2026
Q9_BASE_CMA_SEED = 9000
Q9_MAXFEVALS = 200_000
Q9_LAMBDA_MULTIPLIERS = (1, 2, 4, 8, 16, 32)


def sample_uniform_rotation(n: int, seed: int) -> np.ndarray:
    """Sample a random rotation matrix uniformly (Haar measure on SO(n))."""
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((n, n))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


P_ELLIROT = sample_uniform_rotation(N_DIM, ROTATION_SEED)


def elli_value(x: np.ndarray, condition_number: float) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 1:
        scales = np.array([1.0], dtype=float)
    else:
        exponents = np.arange(n, dtype=float) / (n - 1)
        scales = np.power(condition_number, exponents)
    return float(np.sum((scales * x) ** 2))


def f_elli(x: np.ndarray) -> float:
    """f_elli(x) = sum_i (((10^3)^((i-1)/(n-1)) * x_i)^2), i starts from 1."""
    return elli_value(x, 1e3)


def f_elli_2(x: np.ndarray) -> float:
    """f_elli_2(x) = sum_i (((10^2)^((i-1)/(n-1)) * x_i)^2), i starts from 1."""
    return elli_value(x, 1e2)


def f_tablet(x: np.ndarray) -> float:
    """f_tablet(x) = 10^6*x_1^2 + sum_{i=2}^n x_i^2."""
    x = np.asarray(x, dtype=float)
    return float(1e6 * x[0] ** 2 + np.sum(x[1:] ** 2))


def f_ellirot(x: np.ndarray) -> float:
    """f_ellirot(x) = f_elli(Px), where P is a random rotation matrix."""
    x = np.asarray(x, dtype=float)
    return f_elli(P_ELLIROT @ x)


def f_rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(10.0 * x.size + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def load_fit_log(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="%")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def write_q6_comparison(results_dir: Path, all_results: dict) -> None:
    fit_elli = load_fit_log(results_dir / "f_elli" / "outcmaesfit.dat")
    fit_ellirot = load_fit_log(results_dir / "f_ellirot" / "outcmaesfit.dat")

    plt.figure(figsize=(8, 4.6))
    plt.semilogy(fit_elli[:, 1], fit_elli[:, 4], label="f_elli")
    plt.semilogy(fit_ellirot[:, 1], fit_ellirot[:, 4], label="f_ellirot")
    plt.xlabel("Function evaluations")
    plt.ylabel("Best-ever objective value")
    plt.title("CMA-ES convergence: f_elli vs f_ellirot")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "q6_elli_vs_ellirot_fbest.png", dpi=180)
    plt.close("all")

    lines = [
        "| metric | f_elli | f_ellirot |",
        "|---|---:|---:|",
        f"| f(x0) | {all_results['f_elli']['f_x0']:.3e} | {all_results['f_ellirot']['f_x0']:.3e} |",
        f"| best f | {all_results['f_elli']['fbest']:.3e} | {all_results['f_ellirot']['fbest']:.3e} |",
        f"| evaluations | {all_results['f_elli']['evaluations']} | {all_results['f_ellirot']['evaluations']} |",
        f"| iterations | {all_results['f_elli']['iterations']} | {all_results['f_ellirot']['iterations']} |",
        f"| final sigma | {fit_elli[-1, 2]:.3e} | {fit_ellirot[-1, 2]:.3e} |",
        f"| final axis ratio | {fit_elli[-1, 3]:.3e} | {fit_ellirot[-1, 3]:.3e} |",
    ]
    with (results_dir / "q6_metrics.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_cma_to_target(seed: int, target: float, maxfevals: int) -> dict:
    _, es = cma.fmin2(
        f_elli_2,
        X0.copy(),
        SIGMA0,
        options={
            "seed": seed,
            "ftarget": target,
            "maxfevals": maxfevals,
            "verbose": -9,
            "verb_log": 0,
            "verb_disp": 0,
        },
    )
    return {
        "seed": seed,
        "evaluations": int(es.result.evaluations),
        "iterations": int(es.result.iterations),
        "fbest": float(es.result.fbest),
        "reached_target": bool(es.result.fbest <= target),
        "stop": dict(es.result.stop),
    }


def run_oneplusone_to_target(seed: int, target: float, maxfevals: int) -> dict:
    rng = np.random.default_rng(seed)
    x = X0.copy()
    fx = float(f_elli_2(x))
    sigma = float(SIGMA0)
    evals = 1
    iterations = 0
    success_count = 0

    while evals < maxfevals and fx > target:
        y = x + sigma * rng.standard_normal(N_DIM)
        fy = float(f_elli_2(y))
        evals += 1
        iterations += 1

        if fy <= fx:
            x = y
            fx = fy
            success_count += 1

        if iterations % Q7_ONEPLUS_WINDOW == 0:
            success_rate = success_count / Q7_ONEPLUS_WINDOW
            if success_rate > 0.2:
                sigma /= Q7_ONEPLUS_C
            elif success_rate < 0.2:
                sigma *= Q7_ONEPLUS_C
            success_count = 0

    return {
        "seed": seed,
        "evaluations": int(evals),
        "iterations": int(iterations),
        "fbest": float(fx),
        "final_sigma": float(sigma),
        "reached_target": bool(fx <= target),
    }


def run_q7_comparison(results_dir: Path) -> dict:
    q7_dir = results_dir / "q7"
    q7_dir.mkdir(parents=True, exist_ok=True)

    seeds = [Q7_BASE_SEED + i for i in range(1, Q7_N_RUNS + 1)]
    runs = []
    cma_evals = []
    oneplus_evals = []

    for run_id, seed in enumerate(seeds, start=1):
        cma_result = run_cma_to_target(seed, Q7_TARGET, Q7_CMA_MAXFEVALS)
        oneplus_result = run_oneplusone_to_target(seed, Q7_TARGET, Q7_ONEPLUS_MAXFEVALS)
        ratio = oneplus_result["evaluations"] / cma_result["evaluations"]
        runs.append(
            {
                "run": run_id,
                "seed": seed,
                "cma_es": cma_result,
                "one_plus_one_es": oneplus_result,
                "eval_ratio_oneplus_over_cma": ratio,
            }
        )
        cma_evals.append(cma_result["evaluations"])
        oneplus_evals.append(oneplus_result["evaluations"])

    summary = {
        "dimension": N_DIM,
        "target": Q7_TARGET,
        "x0": X0.tolist(),
        "sigma0": SIGMA0,
        "one_plus_one_rule": "1/5 success rule",
        "one_plus_one_window": Q7_ONEPLUS_WINDOW,
        "one_plus_one_c": Q7_ONEPLUS_C,
        "runs": runs,
        "cma_eval_mean": float(np.mean(cma_evals)),
        "cma_eval_median": float(np.median(cma_evals)),
        "oneplus_eval_mean": float(np.mean(oneplus_evals)),
        "oneplus_eval_median": float(np.median(oneplus_evals)),
        "mean_ratio_oneplus_over_cma": float(np.mean(np.array(oneplus_evals) / np.array(cma_evals))),
    }

    with (q7_dir / "q7_runs.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "| run | seed | CMA-ES evals to 1e-6 | (1+1)-ES evals to 1e-6 | ratio ((1+1)/CMA) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in runs:
        lines.append(
            f"| {item['run']} | {item['seed']} | {item['cma_es']['evaluations']} | "
            f"{item['one_plus_one_es']['evaluations']} | {item['eval_ratio_oneplus_over_cma']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"- mean CMA-ES evals: {summary['cma_eval_mean']:.1f}",
            f"- mean (1+1)-ES evals: {summary['oneplus_eval_mean']:.1f}",
            f"- mean ratio ((1+1)/CMA): {summary['mean_ratio_oneplus_over_cma']:.2f}",
        ]
    )
    with (q7_dir / "q7_metrics.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    x = np.arange(1, Q7_N_RUNS + 1)
    width = 0.38
    plt.figure(figsize=(8, 4.6))
    plt.bar(x - width / 2, cma_evals, width=width, label="CMA-ES")
    plt.bar(x + width / 2, oneplus_evals, width=width, label="(1+1)-ES (1/5)")
    plt.yscale("log")
    plt.xlabel("Run")
    plt.ylabel("Evaluations to reach 1e-6 (log scale)")
    plt.title("Q7: CMA-ES vs (1+1)-ES")
    plt.xticks(x)
    plt.grid(True, which="both", axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(q7_dir / "q7_evals_comparison.png", dpi=180)
    plt.close("all")
    return summary


def run_q9_rastrigin_population_experiment(results_dir: Path) -> dict:
    q9_dir = results_dir / "q9"
    q9_dir.mkdir(parents=True, exist_ok=True)

    default_lambda = 4 + int(np.floor(3.0 * np.log(N_DIM)))
    x0_rng = np.random.default_rng(Q9_X0_SEED)
    x0_list = [x0_rng.random(N_DIM) for _ in range(Q9_N_RUNS)]

    groups = []
    success_probabilities = []
    mean_evaluations = []

    for multiplier in Q9_LAMBDA_MULTIPLIERS:
        popsize = default_lambda * multiplier
        run_results = []

        for run_id, x0 in enumerate(x0_list, start=1):
            seed = Q9_BASE_CMA_SEED + run_id
            _, es = cma.fmin2(
                f_rastrigin,
                x0.copy(),
                Q9_SIGMA0,
                options={
                    "seed": seed,
                    "popsize": popsize,
                    "ftarget": Q9_TARGET,
                    "maxfevals": Q9_MAXFEVALS,
                    "verbose": -9,
                    "verb_log": 0,
                    "verb_disp": 0,
                },
            )
            fbest = float(es.result.fbest)
            run_results.append(
                {
                    "run": run_id,
                    "seed": seed,
                    "x0": x0.tolist(),
                    "popsize": popsize,
                    "evaluations": int(es.result.evaluations),
                    "iterations": int(es.result.iterations),
                    "fbest": fbest,
                    "reached_target": bool(fbest <= Q9_TARGET),
                    "stop": dict(es.result.stop),
                }
            )

        successes = int(sum(1 for r in run_results if r["reached_target"]))
        probability = successes / Q9_N_RUNS
        groups.append(
            {
                "lambda_multiplier": multiplier,
                "population_size": popsize,
                "n_success": successes,
                "n_runs": Q9_N_RUNS,
                "success_probability": probability,
                "mean_evaluations": float(np.mean([r["evaluations"] for r in run_results])),
                "median_evaluations": float(np.median([r["evaluations"] for r in run_results])),
                "runs": run_results,
            }
        )
        success_probabilities.append(probability)
        mean_evaluations.append(float(np.mean([r["evaluations"] for r in run_results])))

    summary = {
        "dimension": N_DIM,
        "objective": "rastrigin",
        "sigma0": Q9_SIGMA0,
        "target": Q9_TARGET,
        "maxfevals": Q9_MAXFEVALS,
        "default_lambda_formula": "4 + floor(3*log(n))",
        "default_lambda": default_lambda,
        "lambda_multipliers": list(Q9_LAMBDA_MULTIPLIERS),
        "groups": groups,
    }

    with (q9_dir / "q9_runs.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "| lambda multiplier | population size | success / 5 | success probability | mean evals |",
        "|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        lines.append(
            f"| {group['lambda_multiplier']} | {group['population_size']} | "
            f"{group['n_success']}/{group['n_runs']} | {group['success_probability']:.2f} | "
            f"{group['mean_evaluations']:.1f} |"
        )
    with (q9_dir / "q9_metrics.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    x = np.arange(len(Q9_LAMBDA_MULTIPLIERS))
    labels = [str(m) for m in Q9_LAMBDA_MULTIPLIERS]
    fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
    bars = ax1.bar(x, success_probabilities, color="#3a86ff", alpha=0.8)
    ax1.set_ylabel("Success probability")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(x, labels)
    ax1.set_xlabel("Lambda multiplier")
    ax1.set_title("Q9: Rastrigin success probability vs population size")
    ax1.grid(True, axis="y", alpha=0.25)
    for bar, p in zip(bars, success_probabilities):
        ax1.text(bar.get_x() + bar.get_width() / 2, p + 0.03, f"{p:.2f}", ha="center", va="bottom", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, mean_evaluations, color="#fb5607", marker="o", linewidth=1.8, label="mean evals")
    ax2.set_ylabel("Mean evaluations")
    ax2.set_yscale("log")

    fig.tight_layout()
    plt.savefig(q9_dir / "q9_success_probability.png", dpi=180)
    plt.close("all")
    return summary


def run_case(name: str, objective) -> dict:
    out_dir = Path("results") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = str(out_dir / "outcmaes")
    options = {
        "seed": SEED,
        "verb_filenameprefix": prefix,
        "verb_log": 1,   # log every iteration for post-plotting
        "verb_disp": 50,
        "verbose": 1,
        "ftarget": 1e-12,
        "maxfevals": 500000,
    }

    xbest, es = cma.fmin2(objective, X0.copy(), SIGMA0, options=options)

    # Equivalent to MATLAB LogPlot='on': load log files and draw CMA-ES diagnostics.
    cma.plot(prefix, xsemilog=0, plot_mean=1)
    for i, fig_num in enumerate(plt.get_fignums(), start=1):
        fig_path = out_dir / f"logplot_{i}.png"
        plt.figure(fig_num).savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close("all")

    result = {
        "name": name,
        "dimension": N_DIM,
        "seed": SEED,
        "x0": X0.tolist(),
        "f_x0": float(objective(X0.copy())),
        "sigma0": SIGMA0,
        "xbest": np.asarray(xbest, dtype=float).tolist(),
        "fbest": float(es.result.fbest),
        "evaluations": int(es.result.evaluations),
        "iterations": int(es.result.iterations),
        "stop": dict(es.result.stop),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    all_results = {
        "f_elli": run_case("f_elli", f_elli),
        "f_ellirot": run_case("f_ellirot", f_ellirot),
        "f_tablet": run_case("f_tablet", f_tablet),
    }

    results_dir = Path("results")
    np.save(results_dir / "q6_rotation_matrix.npy", P_ELLIROT)
    with (results_dir / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    lines = [
        "| function | fbest | evaluations | iterations |",
        "|---|---:|---:|---:|",
    ]
    for key in ("f_elli", "f_ellirot", "f_tablet"):
        item = all_results[key]
        lines.append(
            f"| {key} | {item['fbest']:.3e} | {item['evaluations']} | {item['iterations']} |"
        )

    with (results_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    write_q6_comparison(results_dir, all_results)
    q7_summary = run_q7_comparison(results_dir)
    q9_summary = run_q9_rastrigin_population_experiment(results_dir)

    print("\n".join(lines))
    print(
        f"Q7 mean evals -> CMA-ES: {q7_summary['cma_eval_mean']:.1f}, "
        f"(1+1)-ES: {q7_summary['oneplus_eval_mean']:.1f}"
    )
    best_q9 = max(q9_summary["groups"], key=lambda g: g["success_probability"])
    print(
        f"Q9 best success probability: {best_q9['success_probability']:.2f} "
        f"at lambda multiplier {best_q9['lambda_multiplier']}"
    )


if __name__ == "__main__":
    main()
