"""
bootstrap_stats.py
==================
General-purpose bootstrap methods covering two core analyses.

Contents
--------
1. Bootstrap              -- resampling class with per-replicate summary stats
2. confidence_interval    -- percentile CI: two-sided, left-sided, right-sided
3. ci_width_analysis      -- how CI width evolves with bootstrap sample size
                            (replicates the seminarium.ipynb result)
4. reg_stats              -- OLS intercept + slope via scipy.stats.linregress
5. mc_bootstrap           -- Monte Carlo bootstrap for linear regression coefs
6. mse                    -- mean squared error for a linear predictor
7. bootstrap_regression_demo -- full NBA DRIBBLES -> TOUCH_TIME analysis

Data expected
-------------
    nba_cleaned.csv  (place in same directory as this file)

Usage
-----
    python bootstrap_stats.py               # runs regression demo
    from bootstrap_stats import Bootstrap, mc_bootstrap, confidence_interval
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KernelDensity


# ── 1. Generic Bootstrap class ─────────────────────────────────────────────────

class Bootstrap:
    """
    Resample-based bootstrap for arbitrary 1-D data.

    Parameters
    ----------
    data : array-like  -- the original sample
    """

    def __init__(self, data):
        self.data = np.asanyarray(data)
        self.n = len(self.data)
        self.statistics: pd.DataFrame | None = None

    def boot_sampling(self, replicates: int) -> np.ndarray:
        """
        Draw `replicates` bootstrap samples (each of size n, with replacement).

        Returns
        -------
        np.ndarray of shape (replicates, n)
        """
        indices = np.random.randint(0, self.n, size=(replicates, self.n))
        return self.data[indices]

    def samples_statistics(self, replicates: int = 1) -> pd.DataFrame:
        """
        Compute per-replicate summary statistics.

        Returns a DataFrame with columns: mean, std, 0.025, median, 0.975.
        """
        resamples = self.boot_sampling(replicates)
        df = pd.DataFrame({
            "mean":   np.mean(resamples, axis=1),
            "std":    np.std(resamples, axis=1),
            "0.025":  np.quantile(resamples, 0.025, axis=1),
            "median": np.median(resamples, axis=1),
            "0.975":  np.quantile(resamples, 0.975, axis=1),
        })
        self.statistics = df
        return df


# ── 2. Confidence intervals ────────────────────────────────────────────────────

def confidence_interval(
    data: np.ndarray,
    alpha: float = 0.05,
    side: str = "2sided",
) -> str:
    """
    Compute a bootstrap percentile confidence interval.

    Parameters
    ----------
    data  : bootstrap distribution (array of statistic values)
    alpha : significance level (default 0.05 -> 95% CI)
    side  : "2sided" | "Lsided" | "Rsided"

    Returns
    -------
    Formatted string describing the interval.
    """
    if side == "2sided":
        lo, hi = np.quantile(data, [alpha / 2, 1 - alpha / 2])
        return f"Przedział ufności na poziomie {1 - alpha}: ({lo:.4f}, {hi:.4f})"
    if side == "Lsided":
        lo = np.quantile(data, alpha)
        return (
            f"Lewostronny przedział ufności na poziomie {1 - alpha}: "
            f"(∞, {lo:.4f})"
        )
    hi = np.quantile(data, 1 - alpha)
    return (
        f"Prawostronny przedział ufności na poziomie {1 - alpha}: "
        f"({hi:.4f}, ∞)"
    )


# ── 3. CI width analysis (seminarium.ipynb) ────────────────────────────────────

def ci_width_analysis(
    data: np.ndarray,
    sample_sizes: range | list[int] | None = None,
    true_mean: float | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """
    Analyse how bootstrap CI width varies with bootstrap sample size.

    For each s in sample_sizes:
      - Draw s random indices from the data, then resample from those with
        replacement to form s bootstrap replicates of length s.
      - Record (q_0.025, q_0.975) of the per-replicate means.

    Reproduces the coverage plot from seminarium.ipynb:
        indices = range(10, 400, 2)
        tails   = list(map(boot_mean, indices))

    Parameters
    ----------
    data         : original 1-D sample
    sample_sizes : iterable of bootstrap sample sizes to test
    true_mean    : if provided, prints empirical coverage rate

    Returns
    -------
    (sizes, lowers, uppers) -- parallel lists suitable for plotting
    """
    if sample_sizes is None:
        sample_sizes = range(10, 400, 2)

    data = np.asarray(data)

    def _boot_mean(s: int) -> tuple[float, float]:
        idx = np.random.randint(0, len(data), size=s)
        boot_idx = np.random.choice(idx, replace=True, size=(s, s))
        boot_means = data[boot_idx].mean(axis=1)
        return float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))

    tails = [_boot_mean(s) for s in sample_sizes]
    lowers, uppers = zip(*tails)

    if true_mean is not None:
        coverage = sum(lo < true_mean < hi for lo, hi in tails) / len(tails)
        print(f"Empirical coverage ({1 - 0.05:.0%} nominal): {coverage:.4f}")

    return list(sample_sizes), list(lowers), list(uppers)


def plot_ci_width(
    sample_sizes: list[int],
    lowers: list[float],
    uppers: list[float],
    true_mean: float | None = None,
    save_path: str | None = None,
) -> None:
    """Plot CI bounds and width for the ci_width_analysis output."""
    dist = [hi - lo for lo, hi in zip(lowers, uppers)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.vlines(sample_sizes, lowers, uppers, alpha=0.4, color="steelblue")
    ax1.plot(sample_sizes, uppers, color="steelblue", linewidth=1.2)
    ax1.plot(sample_sizes, lowers, color="steelblue", linewidth=1.2)
    if true_mean is not None:
        ax1.axhline(true_mean, color="red", linewidth=1.5, label=f"true mean = {true_mean:.2f}")
        ax1.legend()
    ax1.set_title("Przedziały ufności")
    ax1.set_xlabel("rozmiar próbki bootstrapowej")
    ax1.set_ylabel("wartość")

    ax2.plot(sample_sizes, dist, color="darkorange", linewidth=1.5)
    ax2.set_title("Długość przedziału ufności")
    ax2.set_xlabel("rozmiar próby bootstrapowej")
    ax2.set_ylabel("szerokość CI")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110)
    plt.close()


# ── 4. OLS regression helpers ──────────────────────────────────────────────────

def reg_stats(X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """Return (intercept, slope) from a simple OLS regression Y ~ X."""
    slope, intercept, *_ = stats.linregress(X, Y)
    return float(intercept), float(slope)


# ── 5. Monte Carlo bootstrap for regression ───────────────────────────────────

def mc_bootstrap(
    X: np.ndarray,
    Y: np.ndarray,
    B: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo bootstrap for simple linear regression coefficients.

    For each of B replicates, a size-n sample is drawn with replacement from
    (X, Y) and OLS is fit. The empirical distribution of the coefficients is
    used to form percentile confidence intervals.

    Parameters
    ----------
    X, Y  : predictor and response arrays (same length)
    B     : number of bootstrap replicates
    alpha : significance level for the CI
    seed  : RNG seed for reproducibility

    Returns
    -------
    dict with keys "b0" (intercept) and "b1" (slope), each containing:
        estimate      -- OLS coefficient from the original sample
        conf_interval -- percentile CI at level (1 - alpha)
        distribution  -- array of B bootstrap estimates
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    b0 = np.empty(B)
    b1 = np.empty(B)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        b0[b], b1[b] = reg_stats(X[idx], Y[idx])

    b0_est, b1_est = reg_stats(X, Y)
    return {
        "b0": {
            "estimate":      b0_est,
            "conf_interval": np.quantile(b0, [alpha / 2, 1 - alpha / 2]),
            "distribution":  b0,
        },
        "b1": {
            "estimate":      b1_est,
            "conf_interval": np.quantile(b1, [alpha / 2, 1 - alpha / 2]),
            "distribution":  b1,
        },
    }


# ── 6. MSE ──────────────────────────────────────────────────────────────────────

def mse(Y: np.ndarray, X: np.ndarray, intercept: float, slope: float) -> float:
    """Mean squared error of a linear predictor: mean((Y - b0 - b1*X)^2)."""
    return float(np.mean((Y - intercept - slope * X) ** 2))


# ── 7. KDE plot helper ─────────────────────────────────────────────────────────

def kde_plot(data: np.ndarray, ax: plt.Axes | None = None) -> None:
    """Plot a Gaussian KDE of 1-D `data` using scikit-learn."""
    data2d = data[:, np.newaxis]
    kde = KernelDensity(bandwidth=1.0, kernel="gaussian").fit(data2d)
    x_plot = np.linspace(data.min() - 1, data.max() + 1, 1000)[:, np.newaxis]
    log_dens = kde.score_samples(x_plot)
    target = ax if ax is not None else plt.subplot()
    target.fill(x_plot[:, 0], np.exp(log_dens), alpha=0.5, color="blue")


# ── 8. Full regression demo ────────────────────────────────────────────────────

def bootstrap_regression_demo(csv_path: str = "nba_cleaned.csv") -> None:
    """
    Reproduce the DRIBBLES -> TOUCH_TIME bootstrap regression analysis.

    Prints the OLS line and confidence intervals, then saves:
      - bootstrap_regression_coefs.png  (b0 and b1 distributions)
      - bootstrap_mse_diff.png          (MSE_bootstrap - MSE_observed)
    """
    df = pd.read_csv(csv_path)
    X = df["DRIBBLES"].reset_index(drop=True).values
    y = df["TOUCH_TIME"].reset_index(drop=True).values

    coef = reg_stats(X, y)
    print(f"Regression line: Y = {coef[1]:.4f}*X + {coef[0]:.4f}")

    data = mc_bootstrap(X, y, B=10_000, alpha=0.05, seed=42)

    # --- coefficient distribution plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, key in zip(axes, ["b0", "b1"]):
        r = data[key]
        ax.hist(r["distribution"], bins=50, alpha=0.5,
                color="darkblue", density=True, edgecolor="black")
        ax.axvline(r["estimate"], color="red",
                   label=f"Wartość: {r['estimate']:.4f}")
        ax.axvline(r["conf_interval"][0], color="orange",
                   label=(
                       f"95% CI: [{r['conf_interval'][0]:.4f},"
                       f" {r['conf_interval'][1]:.4f}]"
                   ))
        ax.axvline(r["conf_interval"][1], color="orange")
        ax.set_title(f"Rozkład {key}")
        ax.set_xlabel("wartość parametru")
        ax.set_ylabel("Gęstość")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("bootstrap_regression_coefs.png", dpi=110)
    plt.close()
    print("Saved: bootstrap_regression_coefs.png")

    # --- MSE excess distribution ---
    real_error = mse(y, X, coef[0], coef[1])
    errors_diff = [
        mse(y, X, i, s) - real_error
        for i, s in zip(data["b0"]["distribution"], data["b1"]["distribution"])
    ]
    plt.figure(figsize=(8, 6))
    plt.hist(errors_diff, color="darkblue", bins=100, edgecolor="black",
             label=f"Średnia: {np.mean(errors_diff):.4f}")
    plt.title("Różnica błędów średniokwadratowych")
    plt.xlabel("MSE_bootstrap - MSE_OLS")
    plt.ylabel("Liczba replikacji")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("bootstrap_mse_diff.png", dpi=110)
    plt.close()
    print("Saved: bootstrap_mse_diff.png")

    print(f"\nOLS MSE:                  {real_error:.6f}")
    print(f"Mean bootstrap MSE excess: {np.mean(errors_diff):.6f}")
    print(f"\nb0 95% CI: {data['b0']['conf_interval']}")
    print(f"b1 95% CI: {data['b1']['conf_interval']}")


if __name__ == "__main__":
    np.random.seed(42)
    bootstrap_regression_demo()
