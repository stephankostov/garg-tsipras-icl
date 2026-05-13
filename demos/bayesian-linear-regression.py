"""
Bayesian linear regression with a Normal-Inverse-Gamma prior on (w, sigma^2).

For scalar output noise, the multivariate Normal-Inverse-Wishart reduces to a
Normal-Inverse-Gamma:

    sigma^2  ~ InvGamma(a0, b0)
    w | s^2  ~ N(mu0, sigma^2 * Lambda0^{-1})

With a Gaussian likelihood y = w^T x + eps, eps ~ N(0, sigma^2), the posterior
after n observations (X in R^{n x d}, y in R^n) is conjugate:

    Lambda_n = Lambda0 + X^T X
    mu_n     = Lambda_n^{-1} (Lambda0 mu0 + X^T y)
    a_n      = a0 + n/2
    b_n      = b0 + 0.5 (mu0^T Lambda0 mu0 + y^T y - mu_n^T Lambda_n mu_n)

The posterior predictive at a new x* is Student-t:

    y* | x*, D ~ t_{2 a_n}( mu_n^T x*, (b_n / a_n) (1 + x*^T Lambda_n^{-1} x*) )

Three figures:
    - blr_sausage_1d.png: d=1 sausage plot as more data arrives.
    - blr_heatmap_2d.png: d=2 predictive mean + std heatmaps as more data arrives.
    - blr_metrics.png:    k-shot test NLL and MSE curves averaged over seeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ---------- model ----------

def sample_prior(mu0, Lambda0, a0, b0, rng):
    """Draw (w, sigma^2) from NIG(mu0, Lambda0, a0, b0)."""
    sigma2 = stats.invgamma.rvs(a=a0, scale=b0, random_state=rng)
    cov = sigma2 * np.linalg.inv(np.atleast_2d(Lambda0))
    w = rng.multivariate_normal(np.atleast_1d(mu0), cov)
    return w, sigma2


def posterior(X, y, mu0, Lambda0, a0, b0):
    """Conjugate NIG posterior given data (X, y) and NIG prior."""
    n = X.shape[0]
    Lambda_n = Lambda0 + X.T @ X
    rhs = Lambda0 @ mu0 + X.T @ y
    mu_n = np.linalg.solve(Lambda_n, rhs)
    a_n = a0 + 0.5 * n
    b_n = b0 + 0.5 * (mu0 @ Lambda0 @ mu0 + y @ y - mu_n @ Lambda_n @ mu_n)
    return mu_n, Lambda_n, a_n, b_n


def predictive_params(X_star, mu_n, Lambda_n, a_n, b_n):
    """Student-t predictive params at rows of X_star."""
    Lambda_n_inv = np.linalg.inv(Lambda_n)
    loc = X_star @ mu_n
    quad = np.einsum("ni,ij,nj->n", X_star, Lambda_n_inv, X_star)
    scale2 = (b_n / a_n) * (1.0 + quad)
    df = 2.0 * a_n
    return df, loc, np.sqrt(scale2)


def generate_data(d, N, mu0, Lambda0, a0, b0, rng, sigma2_override=None):
    """If sigma2_override is given, draw (w, sigma^2) from the prior *conditioned*
    on sigma^2 = override — i.e. w ~ N(mu0, override * Lambda0^{-1}). This keeps
    the model well-specified, just sampled from a fixed slice of the prior."""
    if sigma2_override is not None:
        sigma2_true = float(sigma2_override)
        cov = sigma2_true * np.linalg.inv(np.atleast_2d(Lambda0))
        w_true = rng.multivariate_normal(np.atleast_1d(mu0), cov)
    else:
        w_true, sigma2_true = sample_prior(mu0, Lambda0, a0, b0, rng)
    X = rng.normal(size=(N, d))
    y = X @ w_true + rng.normal(scale=np.sqrt(sigma2_true), size=N)
    return w_true, sigma2_true, X, y


# ---------- 1D sausage plot ----------

def plot_1d(out_path, ks, rng, n_datasets=4):
    d = 1
    sigma2, w_var = 0.1, 1.0
    mu0 = np.zeros(d)
    Lambda0 = (sigma2 / w_var) * np.eye(d)   # so Var[w_i | sigma^2] = w_var
    a0, b0 = 3.0, 1.0
    N = max(ks)

    datasets = []
    for j in range(n_datasets):
        w_true, sigma2_true, X_all, y_all = generate_data(
            d, N, mu0, Lambda0, a0, b0, rng, sigma2_override=sigma2
        )
        datasets.append((w_true, sigma2_true, X_all, y_all))
        print(f"[1D dataset {j}] w_true = {w_true[0]:+.3f},  sigma^2 = {sigma2_true:.4f}")

    t_grid = np.linspace(-3.0, 3.0, 200)
    X_star = t_grid[:, None]

    fig, axes = plt.subplots(
        len(ks), n_datasets,
        figsize=(2.6 * n_datasets, 1.8 * len(ks)),
        sharex=True, sharey=True,
    )
    if n_datasets == 1:
        axes = axes[:, None]

    for col, (w_true, _, X_all, y_all) in enumerate(datasets):
        for row, k in enumerate(ks):
            X_k, y_k = X_all[:k], y_all[:k]
            mu_n, Lambda_n, a_n, b_n = posterior(X_k, y_k, mu0, Lambda0, a0, b0)
            df, loc, scale = predictive_params(X_star, mu_n, Lambda_n, a_n, b_n)
            lo, hi = stats.t.interval(0.95, df=df, loc=loc, scale=scale)

            ax = axes[row, col]
            show_legend = row == 0 and col == 0
            ax.fill_between(t_grid, lo, hi, alpha=0.25,
                            label="95% predictive" if show_legend else None)
            ax.plot(t_grid, loc, lw=2,
                    label="posterior mean" if show_legend else None)
            ax.plot(t_grid, t_grid * w_true[0], "k--", lw=1,
                    label="true mean" if show_legend else None)
            if k > 0:
                ax.scatter(X_k[:, 0], y_k, c="C3", s=20, edgecolors="none",
                           label="train" if show_legend else None)
            if col == 0:
                ax.set_ylabel(f"k = {k}")
            if row == 0:
                ax.set_title(f"dataset {col}: w*={w_true[0]:+.2f}", fontsize=9)
            ax.set_ylim(-4, 4)

    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    axes[0, 0].legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.suptitle(
        f"1D Bayesian linear regression  "
        f"(σ²={sigma2}, Var[w|σ²]={w_var})",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


# ---------- 2D heatmap ----------

def plot_2d_heatmap(out_path, ks, rng):
    d = 2
    sigma2, w_var = 0.1, 1.0
    mu0 = np.zeros(d)
    Lambda0 = (sigma2 / w_var) * np.eye(d)
    a0, b0 = 3.0, 1.0
    N = max(ks)

    w_true, sigma2_true, X_all, y_all = generate_data(
        d, N, mu0, Lambda0, a0, b0, rng, sigma2_override=sigma2
    )
    print(f"[2D] w_true = {w_true},  sigma^2_true = {sigma2_true:.4f}")

    # Evaluation grid
    g = np.linspace(-3.0, 3.0, 100)
    G1, G2 = np.meshgrid(g, g)
    X_grid = np.stack([G1.ravel(), G2.ravel()], axis=1)

    # Colorbar ranges
    y_lim = float(np.max(np.abs(G1 * w_true[0] + G2 * w_true[1]))) * 1.1
    # Prior predictive std at the grid corner controls the std vmax
    _, _, prior_scale = predictive_params(
        np.array([[3.0, 3.0]]), mu0, Lambda0, a0, b0
    )
    std_vmax = float(prior_scale[0]) * 1.05

    fig, axes = plt.subplots(
        len(ks), 2, figsize=(9, 3.6 * len(ks)),
        sharex=True, sharey=True, constrained_layout=True,
    )

    im_mean = im_std = None
    for row, k in enumerate(ks):
        X_k, y_k = X_all[:k], y_all[:k]
        mu_n, Lambda_n, a_n, b_n = posterior(X_k, y_k, mu0, Lambda0, a0, b0)
        df, loc, scale = predictive_params(X_grid, mu_n, Lambda_n, a_n, b_n)
        mean_g = loc.reshape(G1.shape)
        std_g = scale.reshape(G1.shape)

        ax_m = axes[row, 0]
        im_mean = ax_m.pcolormesh(
            G1, G2, mean_g, cmap="RdBu_r",
            vmin=-y_lim, vmax=y_lim, shading="auto",
        )
        ax_m.contour(G1, G2, mean_g, levels=10, colors="k",
                     linewidths=0.4, alpha=0.5)
        # True direction of w*: predictive mean is zero on the line perpendicular to w*
        # Draw w* arrow from origin for orientation.
        ax_m.annotate("", xy=tuple(w_true), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
        if k > 0:
            ax_m.scatter(X_k[:, 0], X_k[:, 1], c=y_k, cmap="RdBu_r",
                         vmin=-y_lim, vmax=y_lim,
                         edgecolors="k", linewidths=0.5, s=35)
        ax_m.set_ylabel(f"k = {k}")
        ax_m.set_aspect("equal")

        ax_s = axes[row, 1]
        im_std = ax_s.pcolormesh(
            G1, G2, std_g, cmap="viridis",
            vmin=0.0, vmax=std_vmax, shading="auto",
        )
        ax_s.contour(G1, G2, std_g, levels=8, colors="w",
                     linewidths=0.4, alpha=0.5)
        if k > 0:
            ax_s.scatter(X_k[:, 0], X_k[:, 1], c="white",
                         edgecolors="k", linewidths=0.5, s=25)
        ax_s.set_aspect("equal")

    axes[0, 0].set_title("predictive mean   E[y | x, D]")
    axes[0, 1].set_title("predictive std   sqrt Var[y | x, D]")
    for ax in axes[-1, :]:
        ax.set_xlabel("x_1")
    axes[-1, 0].set_xlabel("x_1")

    fig.colorbar(im_mean, ax=axes[:, 0].tolist(), shrink=0.6, label="mean")
    fig.colorbar(im_std, ax=axes[:, 1].tolist(), shrink=0.6, label="std")
    fig.suptitle(
        f"2D Bayesian linear regression  "
        f"(w*={w_true.round(2).tolist()}, sigma^2*={sigma2_true:.3f})  "
        f"— white arrow = w*",
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


# ---------- k-shot test metrics ----------

def plot_metric_curves(out_path, N, S, rng, dims=(2, 5, 10, 20)):
    """Test NLL and MSE as a function of k, averaged over S random (w*, sigma^2*) seeds.

    For each seed we sample (w*, sigma^2*) from the prior, generate a train stream
    of size N and a held-out test set of size N_test, and for each k = 0..N compute
    the posterior predictive's NLL (under the Student-t) and squared-error loss on
    the same test set. The model is well-specified here (no sigma^2 override), so
    curves should approach the Bayes-optimal floor.
    """
    sigma2, w_var = 0.01, 1.0
    a0, b0 = 3.0, 1.0
    N_test = 1000
    ks = np.arange(0, N + 1)

    # sigma^2 is fixed across seeds under the override, so Bayes optimum is a
    # single number for every dimension — Bayes optimality depends only on the
    # noise distribution, not on d.
    bayes_nll = float(0.5 * np.log(2.0 * np.pi * np.e * sigma2))

    fig, axes = plt.subplots(
        len(dims), 2, figsize=(11, 2.8 * len(dims)),
        sharex=True, constrained_layout=True,
    )
    if len(dims) == 1:
        axes = axes[None, :]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(dims)))

    for row, (color, d) in enumerate(zip(colors, dims)):
        mu0 = np.zeros(d)
        Lambda0 = (sigma2 / w_var) * np.eye(d)
        nlls = np.zeros((S, len(ks)))
        mses = np.zeros((S, len(ks)))

        for s in range(S):
            w_true, sigma2_true, X_all, y_all = generate_data(
                d, N, mu0, Lambda0, a0, b0, rng, sigma2_override=sigma2
            )
            X_test = rng.normal(size=(N_test, d))
            y_test = X_test @ w_true + rng.normal(
                scale=np.sqrt(sigma2_true), size=N_test
            )
            for i, k in enumerate(ks):
                X_k, y_k = X_all[:k], y_all[:k]
                mu_n, Lambda_n, a_n, b_n = posterior(X_k, y_k, mu0, Lambda0, a0, b0)
                df, loc, scale = predictive_params(X_test, mu_n, Lambda_n, a_n, b_n)
                nlls[s, i] = -stats.t.logpdf(
                    y_test, df=df, loc=loc, scale=scale
                ).mean()
                mses[s, i] = float(np.mean((y_test - loc) ** 2))

        for col, (vals, ylabel, ref, ref_label) in enumerate([
            (nlls, "test NLL (nats)", bayes_nll, f"Bayes opt = {bayes_nll:.3f}"),
            (mses, "test MSE",        sigma2,    f"σ² = {sigma2}"),
        ]):
            ax = axes[row, col]
            mean = vals.mean(0)
            std = vals.std(0)
            ax.plot(ks, mean, lw=2, color=color, label="mean")
            ax.fill_between(ks, mean - 2 * std, mean + 2 * std,
                            color=color, alpha=0.25, label="±2 std")
            ax.axhline(ref, color="k", ls="--", lw=1, label=ref_label)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"d = {d}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            if col == 1:
                ax.set_yscale("log")
            if row == len(dims) - 1:
                ax.set_xlabel("k  (training examples)")

        print(f"d={d:2d}: NLL[k={N}]={nlls[:, -1].mean():.3f}, "
              f"MSE[k={N}]={mses[:, -1].mean():.4f}")

    fig.suptitle(
        f"k-shot test metrics  "
        f"(mean ± 2σ over {S} seeds, σ²={sigma2}, Var[wᵢ|σ²]={w_var})"
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


# ---------- main ----------

def main():
    ks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 96, 128]
    plot_1d("demos/blr_sausage_1d.png", ks, rng=np.random.default_rng(6))
    plot_2d_heatmap("demos/blr_heatmap_2d.png", ks, rng=np.random.default_rng(3))
    plot_metric_curves("demos/blr_metrics.png", N=200, S=100,
                       rng=np.random.default_rng(1),
                       dims=(2, 5, 10, 20))


if __name__ == "__main__":
    main()
