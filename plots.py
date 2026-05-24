"""
plots.py
========
Matplotlib/Seaborn figure factories for the Flask dashboard.

Every function returns a base64-encoded PNG string suitable for embedding
directly in an HTML <img> src attribute: data:image/png;base64,<...>

Design conventions
------------------
- Dark background (facecolor #0f1117) to match the dashboard theme.
- Accent palette: observed stat -> red (#e63946), mean/null -> green (#2ec4b6),
  confidence bounds -> amber (#fca311).
- Each function is fully self-contained: receives a BootstrapResult (and
  optional extras) and returns a str.
"""

from __future__ import annotations

import io
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde

from bootstrap_engine import BootstrapResult

# ── Palette & style ────────────────────────────────────────────────────────────

BG    = "#0f1117"
PANEL = "#1a1d2e"
RED   = "#e63946"
GREEN = "#2ec4b6"
AMBER = "#fca311"
BLUE  = "#4361ee"
TEXT  = "#e0e0e0"
GRID  = "#2a2d3e"


def _base_fig(w: float = 9, h: float = 5) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, which="both")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="--")
    return fig, ax


def _encode(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=BG, dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── 1. Bootstrap null distribution ────────────────────────────────────────────

def plot_bootstrap_distribution(result: BootstrapResult) -> str:
    """
    KDE + histogram of the empirical null distribution DeltaZ*.

    Vertical lines:
      - Green : null mean (0 under H0)
      - Red   : observed DeltaZ (test statistic)
    Annotated with the two-tailed empirical p-value.
    """
    fig, ax = _base_fig(9, 5)

    dist = result.delta_z_boot
    obs  = result.delta_z_obs

    ax.hist(dist, bins=60, density=True, color=BLUE, alpha=0.45, edgecolor="none")
    sns.kdeplot(dist, ax=ax, color=BLUE, linewidth=2.0)

    ax.axvline(0.0, color=GREEN, linewidth=2.0, linestyle="--", label="Null mean (0)")
    ax.axvline(obs, color=RED,   linewidth=2.5, linestyle="-",
               label=f"Observed DeltaZ = {obs:.3f}")

    xs  = np.linspace(dist.min(), dist.max(), 1000)
    kde = gaussian_kde(dist)
    ys  = kde(xs)
    ax.fill_between(xs, ys, where=np.abs(xs) >= abs(obs),
                    color=RED, alpha=0.25, label="p-value region")

    ax.set_xlabel("DeltaZ*  (Z_PO* - Z_RS*)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"{result.player_name}  |  Parametric Bootstrap Null Distribution\n"
        f"p = {result.p_value:.4f}   B = {len(dist):,}   "
        f"RS games = {result.n_rs_games}   PO games = {result.n_po_games}",
        fontsize=11,
    )
    ax.legend(facecolor=PANEL, labelcolor=TEXT, framealpha=0.8, fontsize=9)
    fig.tight_layout()
    return _encode(fig)


# ── 2. Z-score comparison bar chart ───────────────────────────────────────────

def plot_z_comparison(result: BootstrapResult) -> str:
    """Horizontal bar chart comparing Z_RS and Z_PO."""
    fig, ax = _base_fig(6, 3.5)

    labels = ["Z_RS  (Regular Season)", "Z_PO  (Playoffs)"]
    values = [result.z_rs, result.z_po]
    colors = [GREEN if v >= 0 else RED for v in values]

    bars = ax.barh(labels, values, color=colors, height=0.4, edgecolor=GRID)
    ax.axvline(0, color=TEXT, linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            val + (0.03 if val >= 0 else -0.03),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            color=TEXT, fontsize=10,
        )

    ax.set_xlabel("Aggregated Wald-Wolfowitz Z", fontsize=10)
    ax.set_title(
        f"{result.player_name}  |  Season-Context Z Comparison\n"
        f"DeltaZ = {result.delta_z_obs:+.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    return _encode(fig)


# ── 3. Per-game Z-score timeline ───────────────────────────────────────────────

def plot_game_timeline(game_records: list[dict], player_name: str) -> str:
    """
    Scatter + line plot of per-game Z_i scores, colour-coded by season type.

    Parameters
    ----------
    game_records : list of dicts with keys {game_id, season_type, n, p, z}.
    player_name  : label for the title.
    """
    if not game_records:
        fig, ax = _base_fig()
        ax.text(0.5, 0.5, "No game data available",
                ha="center", va="center", color=TEXT, fontsize=13,
                transform=ax.transAxes)
        return _encode(fig)

    fig, ax = _base_fig(10, 4.5)

    rs = [r for r in game_records if r["season_type"] == "RS"]
    po = [r for r in game_records if r["season_type"] == "PO"]

    def _plot_segment(records: list[dict], color: str, label: str) -> None:
        if not records:
            return
        zs = [r["z"] for r in records]
        xs = list(range(len(zs)))
        ax.plot(xs, zs, color=color, linewidth=1.2, alpha=0.6)
        ax.scatter(xs, zs, color=color, s=55, zorder=3,
                   label=f"{label}  (n={len(records)} games)")

    _plot_segment(rs, GREEN, "Regular Season")
    _plot_segment(po, RED,   "Playoffs")

    ax.axhline(0, color=TEXT, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Game index (within context)", fontsize=10)
    ax.set_ylabel("Z_i  (Wald-Wolfowitz)", fontsize=10)
    ax.set_title(
        f"{player_name}  |  Per-Game Streakiness Z-scores\n"
        "Negative Z -> fewer runs (more streaky); Positive Z -> more alternating",
        fontsize=10,
    )
    ax.legend(facecolor=PANEL, labelcolor=TEXT, framealpha=0.8, fontsize=9)
    fig.tight_layout()
    return _encode(fig)


# ── 4. Shot-percentage vs Z scatter ───────────────────────────────────────────

def plot_fg_vs_z(game_records: list[dict], player_name: str) -> str:
    """
    Scatter of per-game FG% (p_i) vs streakiness (Z_i), bubble size proportional to n_i.
    Reveals whether high-percentage games correlate with streakiness.
    """
    if not game_records:
        fig, ax = _base_fig()
        ax.text(0.5, 0.5, "No game data available",
                ha="center", va="center", color=TEXT, fontsize=13,
                transform=ax.transAxes)
        return _encode(fig)

    fig, ax = _base_fig(7, 5)

    for r in game_records:
        color = GREEN if r["season_type"] == "RS" else RED
        size  = max(30, r["n"] * 8)
        ax.scatter(r["p"] * 100, r["z"], color=color, s=size,
                   alpha=0.65, edgecolors="none")

    ax.axhline(0, color=TEXT, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Field Goal %  (p_i x 100)", fontsize=10)
    ax.set_ylabel("Z_i  (Wald-Wolfowitz)", fontsize=10)
    ax.set_title(
        f"{player_name}  |  FG% vs Streakiness per Game\n"
        "Bubble size proportional to shots taken",
        fontsize=10,
    )
    patches = [
        mpatches.Patch(color=GREEN, label="Regular Season"),
        mpatches.Patch(color=RED,   label="Playoffs"),
    ]
    ax.legend(handles=patches, facecolor=PANEL, labelcolor=TEXT,
              framealpha=0.8, fontsize=9)
    fig.tight_layout()
    return _encode(fig)


# ── 5. Multi-player DeltaZ summary bar ────────────────────────────────────────

def plot_all_players_delta_z(results: dict[str, BootstrapResult]) -> str:
    """
    Horizontal bar chart of DeltaZ for all analysed players.
    Bars coloured by significance: red = p < 0.05, grey = not significant.
    """
    if not results:
        fig, ax = _base_fig()
        ax.text(0.5, 0.5, "No results yet",
                ha="center", va="center", color=TEXT, fontsize=13,
                transform=ax.transAxes)
        return _encode(fig)

    names  = list(results.keys())
    deltas = [results[n].delta_z_obs for n in names]
    pvals  = [results[n].p_value     for n in names]
    colors = [RED if p < 0.05 else "#606070" for p in pvals]

    sorted_idx = np.argsort(deltas)
    names  = [names[i]  for i in sorted_idx]
    deltas = [deltas[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]
    pvals  = [pvals[i]  for i in sorted_idx]

    h = max(4, len(names) * 0.42)
    fig, ax = _base_fig(9, h)

    bars = ax.barh(names, deltas, color=colors, height=0.5, edgecolor=GRID)
    ax.axvline(0, color=TEXT, linewidth=0.8)

    for bar, val, pv in zip(bars, deltas, pvals):
        ax.text(
            val + (0.02 if val >= 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}  (p={pv:.3f})",
            va="center",
            ha="left" if val >= 0 else "right",
            color=TEXT, fontsize=8,
        )

    ax.set_xlabel("DeltaZ  (Z_PO - Z_RS)", fontsize=10)
    ax.set_title(
        "All Players  |  Streakiness Shift: Playoffs vs Regular Season\n"
        "Red bars = statistically significant (p < 0.05)",
        fontsize=11,
    )
    sig_patch = mpatches.Patch(color=RED,       label="p < 0.05 (significant)")
    ns_patch  = mpatches.Patch(color="#606070", label="p >= 0.05")
    ax.legend(handles=[sig_patch, ns_patch],
              facecolor=PANEL, labelcolor=TEXT, framealpha=0.8, fontsize=9)
    fig.tight_layout()
    return _encode(fig)
