"""Monte Carlo illustration of the thin-shell phenomenon (companion to the PGMO report).

For isotropic log-concave random vectors X in R^n (E X = 0, Cov X = I_n) the
thin-shell bound states Var(|X|^2) <= C n, i.e. |X| concentrates around sqrt(n)
with O(1) fluctuations, and most one-dimensional marginals are close to N(0, 1).

Uniform samples from three convex bodies are put in isotropic position:

* cube       [-sqrt(3), sqrt(3)]^n                       (independent coordinates)
* ball       sqrt(n + 2) * B_2^n                         (rotation invariant)
* simplex    centred, rescaled regular simplex           (Dirichlet(1, ..., 1) samples)

and compared with a non-thin-shell isotropic example, an equal mixture of two
Gaussians N(0, s_1^2 I) and N(0, s_2^2 I) with s_1^2 + s_2^2 = 2, for which
Var(|X|^2) grows like n^2.
"""

from __future__ import annotations

import os

import numpy as np

DIMS = [10, 30, 100, 300, 1000]
N_SAMPLES = 20_000
rng = np.random.default_rng(0)


def cube(n, m):
    return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(m, n))


def ball(n, m):
    g = rng.standard_normal((m, n))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    r = rng.random(m) ** (1.0 / n)
    return np.sqrt(n + 2.0) * g * r[:, None]


def simplex(n, m):
    """Uniform on the n-simplex {p >= 0, sum p = 1} in R^{n+1}, mapped isotropically to R^n."""
    p = rng.dirichlet(np.ones(n + 1), size=m)
    centred = p - 1.0 / (n + 1)
    # Orthonormal basis of the hyperplane sum = 0 (Helmert-type, via QR).
    q, _ = np.linalg.qr(np.vstack([np.ones(n + 1), np.eye(n + 1)[:-1]]).T)
    x = centred @ q[:, 1:]
    var = 1.0 / ((n + 1) * (n + 2))  # covariance of the Dirichlet(1) on the hyperplane is var * I
    return x / np.sqrt(var)


def gaussian_mixture(n, m, s1=0.5, s2=np.sqrt(1.75)):
    scale = np.where(rng.random(m) < 0.5, s1, s2)
    return rng.standard_normal((m, n)) * scale[:, None]


BODIES = {"cube": cube, "ball": ball, "simplex": simplex, "two-Gaussian mixture": gaussian_mixture}


def main() -> None:
    os.makedirs("results", exist_ok=True)
    table = {name: [] for name in BODIES}
    marg = {}
    print(f"{'body':>22s} " + " ".join(f"{'n=' + str(n):>10s}" for n in DIMS) + "   (Var|X|^2 / n)")
    for name, sampler in BODIES.items():
        for n in DIMS:
            X = sampler(n, N_SAMPLES if n <= 300 else N_SAMPLES // 4)
            sq = np.sum(X**2, axis=1)
            table[name].append(np.var(sq) / n)
            if n == DIMS[-1]:
                theta = rng.standard_normal(n)
                marg[name] = X @ (theta / np.linalg.norm(theta))
        print(f"{name:>22s} " + " ".join(f"{v:10.3f}" for v in table[name]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, vals in table.items():
        axes[0].loglog(DIMS, vals, marker="o", label=name)
    axes[0].set_xlabel("dimension $n$")
    axes[0].set_ylabel(r"$\mathrm{Var}(|X|^2)/n$")
    axes[0].set_title("Thin shell: bounded for convex bodies")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    t = np.linspace(-4, 4, 200)
    for name, m in marg.items():
        axes[1].hist(m, bins=80, density=True, histtype="step", label=name)
    axes[1].plot(t, np.exp(-t**2 / 2) / np.sqrt(2 * np.pi), "k--", label="N(0,1)")
    axes[1].set_xlim(-4, 4)
    axes[1].set_title(rf"Random marginal $\langle X,\theta\rangle$, $n={DIMS[-1]}$")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join("results", "thin_shell.png"), dpi=150)
    print("Saved results/thin_shell.png")


if __name__ == "__main__":
    main()
