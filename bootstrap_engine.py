"""
bootstrap_engine.py
===================
Game-level parametric bootstrap for basketball streakiness analysis.

Mathematical pipeline
---------------------
1. Wald-Wolfowitz runs statistic per game:
       E[R_i] = 2 * n_i * p_i * (1 - p_i) + 1
       Var[R_i] = 4 * n_i * p_i * (1-p_i) * [1 - 3*p_i*(1-p_i)]
       Z_i = (R_i - E[R_i]) / sqrt(Var[R_i])

2. Volume-weighted meta-analytic aggregation per season context:
       Z_ctx = sum(n_i * Z_i) / sum(n_i)

3. Observed test statistic:
       DeltaZ = Z_PO - Z_RS

4. Parametric bootstrap null distribution (B >= 5000 reps):
   For each rep b:
       Simulate n_i Bernoulli(p_i) shots for every game i (RS and PO),
       recompute Z_RS*, Z_PO*, store DeltaZ* = Z_PO* - Z_RS*.

5. Two-tailed empirical p-value:
       p = (1/B) * sum( |DeltaZ*| >= |DeltaZ| )

Usage
-----
    python bootstrap_engine.py          # runs a self-test with toy data
    from bootstrap_engine import run_bootstrap, analyse_all_players
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ── Core statistical primitives ────────────────────────────────────────────────

def count_runs(shots: np.ndarray) -> int:
    """
    Count the number of Wald-Wolfowitz runs in a binary sequence.

    A run is a maximal consecutive subsequence of identical values.
    Time complexity O(n), space O(1) beyond the input array.
    """
    if len(shots) < 2:
        return int(len(shots))
    return int(np.sum(shots[1:] != shots[:-1]) + 1)


def ww_z_score(shots: np.ndarray) -> tuple[float, float, float] | None:
    """
    Compute the Wald-Wolfowitz Z-score for a single game's shot sequence.

    Returns (n, p, Z) or None when Var(R) <= 0 (degenerate game).

    Degeneracy arises when:
        - n < 3  (too few shots for the asymptotic formula)
        - p = 0 or p = 1 (all misses / all makes -> single run, zero variance)
    """
    n = len(shots)
    if n < 3:
        return None

    p = shots.mean()
    q = 1.0 - p

    var_r = 4.0 * n * p * q * (1.0 - 3.0 * p * q)
    if var_r <= 0.0:
        return None

    e_r   = 2.0 * n * p * q + 1.0
    r_obs = count_runs(shots)
    z     = (r_obs - e_r) / np.sqrt(var_r)
    return float(n), float(p), float(z)


def aggregate_z(game_stats: list[tuple[float, float, float]]) -> float:
    """
    Volume-weighted meta-analytic aggregation:
        Z_ctx = sum(n_i * Z_i) / sum(n_i)

    game_stats is a list of (n_i, p_i, Z_i) tuples.
    """
    if not game_stats:
        return 0.0
    ns = np.array([g[0] for g in game_stats])
    zs = np.array([g[2] for g in game_stats])
    return float((ns * zs).sum() / ns.sum())


# ── Per-game bootstrap simulation ─────────────────────────────────────────────

def _simulate_context_z(
    game_params: list[tuple[float, float]],
    rng: np.random.Generator,
) -> float:
    """
    Simulate one bootstrap replicate of a season context.

    For each game i, draw n_i Bernoulli(p_i) shots, compute Z_i*, and
    return the volume-weighted aggregate Z*. Games that produce Var(R) <= 0
    in the simulated sequence are silently skipped.
    """
    valid: list[tuple[float, float, float]] = []
    for n, p in game_params:
        sim = rng.binomial(1, p, size=int(n)).astype(np.int8)
        result = ww_z_score(sim)
        if result is not None:
            valid.append(result)
    return aggregate_z(valid) if valid else 0.0


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class GameRecord:
    game_id:     str
    season_type: str   # "RS" | "PO"
    n:           float
    p:           float
    z:           float


@dataclass
class BootstrapResult:
    player_name:   str
    delta_z_obs:   float
    z_rs:          float
    z_po:          float
    p_value:       float
    n_rs_games:    int
    n_po_games:    int
    delta_z_boot:  np.ndarray = field(repr=False)

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> dict:
        return {
            "player":      self.player_name,
            "Z_RS":        round(self.z_rs, 4),
            "Z_PO":        round(self.z_po, 4),
            "delta_Z":     round(self.delta_z_obs, 4),
            "p_value":     round(self.p_value, 4),
            "significant": self.significant,
            "RS_games":    self.n_rs_games,
            "PO_games":    self.n_po_games,
        }


# ── Main analysis pipeline ─────────────────────────────────────────────────────

def build_game_records(player_df: pd.DataFrame) -> list[GameRecord]:
    """
    Group player_df (columns: game_id, season_type, shot_result) by game,
    compute Wald-Wolfowitz stats, and return valid GameRecord objects.

    Shots within each game must already be sorted chronologically
    (by event_num ascending).
    """
    records: list[GameRecord] = []
    for (game_id, season_type), grp in player_df.groupby(
        ["game_id", "season_type"], sort=False
    ):
        shots = grp["shot_result"].to_numpy(dtype=np.int8)
        result = ww_z_score(shots)
        if result is None:
            continue
        n, p, z = result
        records.append(GameRecord(str(game_id), str(season_type), n, p, z))
    return records


def run_bootstrap(
    player_df: pd.DataFrame,
    player_name: str,
    B: int = 5000,
    seed: int = 42,
) -> BootstrapResult | None:
    """
    Full analysis pipeline for a single player.

    Parameters
    ----------
    player_df   : DataFrame with columns [game_id, season_type, shot_result].
                  shot_result must be 0/1 integers sorted by event_num.
    player_name : Display name (string label only).
    B           : Bootstrap replications (>= 5000 recommended).
    seed        : NumPy RNG seed for reproducibility.

    Returns
    -------
    BootstrapResult or None if the player lacks both RS and PO games.
    """
    records = build_game_records(player_df)

    rs_records = [r for r in records if r.season_type == "RS"]
    po_records = [r for r in records if r.season_type == "PO"]

    if not rs_records or not po_records:
        return None

    z_rs = aggregate_z([(r.n, r.p, r.z) for r in rs_records])
    z_po = aggregate_z([(r.n, r.p, r.z) for r in po_records])
    delta_z_obs = z_po - z_rs

    rs_params = [(r.n, r.p) for r in rs_records]
    po_params = [(r.n, r.p) for r in po_records]

    rng = np.random.default_rng(seed)
    delta_z_boot = np.empty(B)
    for b in range(B):
        z_rs_star = _simulate_context_z(rs_params, rng)
        z_po_star = _simulate_context_z(po_params, rng)
        delta_z_boot[b] = z_po_star - z_rs_star

    p_value = float(np.mean(np.abs(delta_z_boot) >= abs(delta_z_obs)))

    return BootstrapResult(
        player_name  = player_name,
        delta_z_obs  = delta_z_obs,
        z_rs         = z_rs,
        z_po         = z_po,
        p_value      = p_value,
        n_rs_games   = len(rs_records),
        n_po_games   = len(po_records),
        delta_z_boot = delta_z_boot,
    )


def analyse_all_players(
    df: pd.DataFrame,
    B: int = 5000,
    seed: int = 42,
) -> dict[str, BootstrapResult]:
    """
    Run run_bootstrap for every unique player in df.

    Parameters
    ----------
    df   : Full shots DataFrame with columns
           [player_name, game_id, season_type, shot_result].
    B    : Bootstrap replications per player.
    seed : Master seed; each player gets a derived seed for independence.

    Returns
    -------
    dict mapping player_name -> BootstrapResult (only non-None results).
    """
    results: dict[str, BootstrapResult] = {}
    rng_master = np.random.default_rng(seed)

    for player_name, player_df in df.groupby("player_name"):
        player_seed = int(rng_master.integers(0, 2**31))
        result = run_bootstrap(player_df, str(player_name), B=B, seed=player_seed)
        if result is not None:
            results[str(player_name)] = result

    return results


# ── Self-test with toy data ────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = []
    for game_id in range(1, 21):
        stype = "RS" if game_id <= 15 else "PO"
        p_game = rng.uniform(0.35, 0.55)
        n_shots = rng.integers(8, 22)
        for ev in range(n_shots):
            rows.append({
                "player_name": "Test Player",
                "game_id":     game_id,
                "season_type": stype,
                "shot_result": int(rng.binomial(1, p_game)),
                "event_num":   ev,
            })

    toy_df = pd.DataFrame(rows).sort_values(["game_id", "event_num"])
    result = run_bootstrap(toy_df, "Test Player", B=5000)
    if result:
        print("Bootstrap result:")
        for k, v in result.summary().items():
            print(f"  {k:15s}: {v}")
    else:
        print("Not enough RS + PO games in toy data.")
