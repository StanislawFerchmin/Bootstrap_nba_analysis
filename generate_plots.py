"""
generate_plots.py
=================
Reproduces all five figures from Chapter 4 of the paper using the same
datasets and parameters described in the text.

Output files
------------
  ch4_fig1_ci_analysis.png         -- Section 4.1: CI bounds + CI width
  ch4_fig2_params_correlated.png   -- Section 4.2: b0/b1 distributions (r=0.942)
  ch4_fig3_rss_correlated.png      -- Section 4.2: RSS* - RSS (r=0.942)
  ch4_fig4_params_uncorrelated.png -- Section 4.2: b0/b1 distributions (r~-0.016)
  ch4_fig5_rss_uncorrelated.png    -- Section 4.2: RSS* - RSS (r~-0.016)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

CSV = "nba_cleaned.csv"
B   = 10_000
np.random.seed(42)

df = pd.read_csv(CSV)
print(f"Loaded {len(df):,} rows from {CSV}")


# ── helpers ────────────────────────────────────────────────────────────────────

def reg_stats(X, Y):
    slope, intercept, *_ = stats.linregress(X, Y)
    return float(intercept), float(slope)

def rss(Y, X, intercept, slope):
    return float(np.sum((Y - intercept - slope * X) ** 2))

def run_bootstrap(X, Y, B=10_000, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(X)
    b0  = np.empty(B)
    b1  = np.empty(B)
    for b in range(B):
        idx     = rng.integers(0, n, size=n)
        b0[b], b1[b] = reg_stats(X[idx], Y[idx])
    return b0, b1


# ── Figure 1: CI analysis (Section 4.1) ───────────────────────────────────────
# shots per game, games with ≥ 50 shots  →  n = 894

print("\n[Fig 1] Bootstrap CI analysis...")

shots_per_game = (
    df.groupby("GAME_ID").size()
    .reset_index(name="shot_count")["shot_count"]
)
nba = shots_per_game[shots_per_game >= 50].values
print(f"  Games with >=50 shots: {len(nba)}")
true_mean = nba.mean()

indices = list(range(10, 400, 2))

def boot_mean_ci(data, s):
    idx      = np.random.randint(0, len(data), size=s)
    boot_idx = np.random.choice(idx, replace=True, size=(s, s))
    boot_m   = data[boot_idx].mean(axis=1)
    return float(np.quantile(boot_m, 0.025)), float(np.quantile(boot_m, 0.975))

np.random.seed(15)
tails = [boot_mean_ci(nba, s) for s in indices]
low, high = zip(*tails)
widths    = [h - l for l, h in tails]

coverage = sum(l < true_mean < h for l, h in tails) / len(tails)
print(f"  Empirical coverage: {coverage:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.vlines(indices, low, high, alpha=0.35, color="steelblue", linewidth=0.8)
ax1.plot(indices, high, color="steelblue", linewidth=1.2)
ax1.plot(indices, low,  color="steelblue", linewidth=1.2)
ax1.axhline(true_mean, color="red", linewidth=1.5,
            label=f"Średnia z próby: {true_mean:.2f}")
ax1.set_title("Przedziały ufności")
ax1.set_xlabel("rozmiar próbki bootstrapowej")
ax1.set_ylabel("wartość")
ax1.legend(fontsize=9)

ax2.plot(indices, widths, color="darkorange", linewidth=1.5)
ax2.set_title("Długość przedziałów ufności")
ax2.set_xlabel("rozmiar próby bootstrapowej")
ax2.set_ylabel("szerokość CI")

plt.tight_layout()
plt.savefig("ch4_fig1_ci_analysis.png", dpi=130)
plt.close()
print("  Saved: ch4_fig1_ci_analysis.png")


# ── Figures 2 & 3: correlated data — DRIBBLES → TOUCH_TIME (n ≈ 27 000) ──────

print(f"\n[Fig 2 & 3] Correlated regression (DRIBBLES -> TOUCH_TIME)...")

X_corr = df["DRIBBLES"].values.astype(float)
y_corr = df["TOUCH_TIME"].values.astype(float)
print(f"  n = {len(X_corr):,}  |  r = {np.corrcoef(X_corr, y_corr)[0,1]:.4f}")

coef_corr = reg_stats(X_corr, y_corr)
print(f"  Regression: Y = {coef_corr[1]:.4f}*X + {coef_corr[0]:.4f}")

b0_c, b1_c = run_bootstrap(X_corr, y_corr, B=B)

# Figure 2
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, key, vals, est in zip(axes, ["b0", "b1"], [b0_c, b1_c], coef_corr):
    ci = np.quantile(vals, [0.025, 0.975])
    ax.hist(vals, bins=50, alpha=0.5, color="darkblue", density=True,
            edgecolor="black")
    ax.axvline(est,   color="red",    linestyle="-",
               label=f"Wartość parametru: {est:.4f}")
    ax.axvline(ci[0], color="orange", linestyle="-",
               label=f"95% przedział ufności: [{ci[0]:.4f}, {ci[1]:.4f}]")
    ax.axvline(ci[1], color="orange", linestyle="-")
    ax.set_title(f"Rozkład {key}")
    ax.set_xlabel("wartość parametru")
    ax.set_ylabel("Gęstość")
    ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig("ch4_fig2_params_correlated.png", dpi=130)
plt.close()
print("  Saved: ch4_fig2_params_correlated.png")

# Figure 3
real_rss_c   = rss(y_corr, X_corr, coef_corr[0], coef_corr[1])
diff_c       = [rss(y_corr, X_corr, i, s) - real_rss_c
                for i, s in zip(b0_c, b1_c)]
print(f"  RSS = {real_rss_c:.2f}  |  Mean(RSS* - RSS) = {np.mean(diff_c):.4f}")

plt.figure(figsize=(8, 6))
plt.hist(diff_c, color="darkblue", bins=100, edgecolor="black",
         label=f"Średnia: {np.mean(diff_c):.4f}")
plt.title("Różnica błędów resztowych RSS")
plt.xlabel("RSS* − RSS")
plt.ylabel("Liczba replikacji")
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("ch4_fig3_rss_correlated.png", dpi=130)
plt.close()
print("  Saved: ch4_fig3_rss_correlated.png")


# ── Figures 4 & 5: uncorrelated data — total shots vs accuracy (Q4, n=281) ───

print("\n[Fig 4 & 5] Uncorrelated regression (shots vs accuracy, Q4)...")

df4 = df[df["PERIOD"] == 4]
players = (
    df4.groupby("player_id")["SHOT_RESULT"]
    .value_counts()
    .unstack(fill_value=0)
)
players["accuracy"]     = (players["made"] / (players["made"] + players["missed"]) * 100).round(2)
players["total_shots"]  = players["made"] + players["missed"]
players = players.dropna(subset=["accuracy"])

X_unc = players["total_shots"].values.astype(float)
y_unc = players["accuracy"].values.astype(float)
print(f"  n = {len(X_unc)}  |  r = {np.corrcoef(X_unc, y_unc)[0,1]:.4f}")

coef_unc = reg_stats(X_unc, y_unc)
print(f"  Regression: Y = {coef_unc[1]:.4f}*X + {coef_unc[0]:.4f}")

b0_u, b1_u = run_bootstrap(X_unc, y_unc, B=B)

# Figure 4
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, key, vals, est in zip(axes, ["b0", "b1"], [b0_u, b1_u], coef_unc):
    ci = np.quantile(vals, [0.025, 0.975])
    ax.hist(vals, bins=50, alpha=0.5, color="darkblue", density=True,
            edgecolor="black")
    ax.axvline(est,   color="red",    linestyle="-",
               label=f"Wartość parametru: {est:.4f}")
    ax.axvline(ci[0], color="orange", linestyle="-",
               label=f"95% przedział ufności: [{ci[0]:.4f}, {ci[1]:.4f}]")
    ax.axvline(ci[1], color="orange", linestyle="-")
    ax.set_title(f"Rozkład {key} — dane nieskorelowane")
    ax.set_xlabel("wartość parametru")
    ax.set_ylabel("Gęstość")
    ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig("ch4_fig4_params_uncorrelated.png", dpi=130)
plt.close()
print("  Saved: ch4_fig4_params_uncorrelated.png")

# Figure 5
real_rss_u = rss(y_unc, X_unc, coef_unc[0], coef_unc[1])
diff_u     = [rss(y_unc, X_unc, i, s) - real_rss_u
              for i, s in zip(b0_u, b1_u)]
print(f"  RSS = {real_rss_u:.2f}  |  Mean(RSS* - RSS) = {np.mean(diff_u):.4f}")

plt.figure(figsize=(8, 6))
plt.hist(diff_u, color="darkblue", bins=100, edgecolor="black",
         label=f"Średnia: {np.mean(diff_u):.4f}")
plt.title("Różnica błędów resztowych RSS — dane nieskorelowane")
plt.xlabel("RSS* − RSS")
plt.ylabel("Liczba replikacji")
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("ch4_fig5_rss_uncorrelated.png", dpi=130)
plt.close()
print("  Saved: ch4_fig5_rss_uncorrelated.png")

print("\nDone. All 5 figures saved.")
