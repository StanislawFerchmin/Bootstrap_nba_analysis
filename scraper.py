"""
scraper.py
==========
Fetches shot-by-shot sequences for the top-5 scorers of SAS, OKC, NYK, CLE
for the 2025-26 NBA season (Regular Season + Playoffs).

Returned DataFrame schema
-------------------------
    player_name   str   e.g. "Shai Gilgeous-Alexander"
    player_id     int   NBA Stats player ID
    team_abbr     str   "OKC" | "SAS" | "NYK" | "CLE"
    game_id       str   10-digit NBA game ID
    game_date     str   "YYYY-MM-DD"
    event_num     int   within-game event order
    shot_result   int   1 = made, 0 = missed
    season_type   str   "RS" | "PO"

Usage
-----
    python scraper.py                   # force-refresh from NBA API
    from scraper import fetch_shot_sequences
    df = fetch_shot_sequences()         # uses cache if available
"""

import time
import random
import logging
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, shotchartdetail

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SEASON     = "2025-26"
CACHE_PATH = Path(__file__).parent / "shots_cache.csv"
TOP_N      = 5
API_SLEEP  = 1.0
API_JITTER = 0.4

TARGET_TEAMS: dict[str, int] = {
    "SAS": 1610612759,
    "OKC": 1610612760,
    "NYK": 1610612752,
    "CLE": 1610612739,
}

_SEASON_TYPE_MAP = {
    "Regular Season": "RS",
    "Playoffs":       "PO",
}


def _sleep() -> None:
    time.sleep(API_SLEEP + random.uniform(0, API_JITTER))


def _top_scorers(team_id: int, team_abbr: str, season_type: str) -> list[dict]:
    """Return top-N scorers (by PPG) for `team_id` as [{player_id, player_name}]."""
    _sleep()
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        season_type_all_star=season_type,
        per_mode_simple="PerGame",
        team_id_nullable=team_id,
        timeout=60,
    )
    df = resp.get_data_frames()[0]
    if df.empty:
        log.warning("No players returned for %s (%s)", team_abbr, season_type)
        return []
    df = df.nlargest(TOP_N, "PTS")[["PLAYER_ID", "PLAYER_NAME"]]
    return [{"player_id": int(r.PLAYER_ID), "player_name": r.PLAYER_NAME}
            for r in df.itertuples()]


def _fetch_shots(
    player_id: int,
    player_name: str,
    team_abbr: str,
    season_type: str,
) -> pd.DataFrame | None:
    """Pull all FGA for `player_id` via ShotChartDetail. Returns None if empty."""
    _sleep()
    resp = shotchartdetail.ShotChartDetail(
        team_id=TARGET_TEAMS[team_abbr],
        player_id=player_id,
        season_nullable=SEASON,
        season_type_all_star=season_type,
        context_measure_simple="FGA",
        timeout=60,
    )
    df = resp.get_data_frames()[0]
    if df.empty:
        log.info("    No shots for %s (%s)", player_name, season_type)
        return None

    out = pd.DataFrame({
        "player_name": player_name,
        "player_id":   player_id,
        "team_abbr":   team_abbr,
        "game_id":     df["GAME_ID"].astype(str),
        "game_date":   df["GAME_DATE"],
        "event_num":   df["GAME_EVENT_ID"].astype(int),
        "shot_result": df["SHOT_MADE_FLAG"].astype(int),
        "season_type": _SEASON_TYPE_MAP[season_type],
    })
    return out.sort_values(["game_id", "event_num"]).reset_index(drop=True)


def fetch_shot_sequences(force_refresh: bool = False) -> pd.DataFrame:
    """
    Main entry point. Returns a DataFrame with one row per FGA.

    Reads from shots_cache.csv when available to avoid re-hitting the API.
    Pass force_refresh=True to bypass the cache.
    """
    if not force_refresh and CACHE_PATH.exists():
        log.info("Loading shots from cache: %s", CACHE_PATH)
        return pd.read_csv(CACHE_PATH, dtype={"game_id": str})

    frames: list[pd.DataFrame] = []

    for season_type_label in ("Regular Season", "Playoffs"):
        log.info("=" * 60)
        log.info("Season type: %s", season_type_label)
        log.info("=" * 60)

        for team_abbr, team_id in TARGET_TEAMS.items():
            log.info("  Team: %s", team_abbr)

            try:
                scorers = _top_scorers(team_id, team_abbr, season_type_label)
            except Exception as exc:
                log.error("Scorer list failed for %s (%s): %s",
                          team_abbr, season_type_label, exc)
                continue

            if not scorers:
                continue

            log.info("  Top-%d scorers: %s", TOP_N,
                     ", ".join(p["player_name"] for p in scorers))

            for player in scorers:
                try:
                    shots = _fetch_shots(
                        player["player_id"], player["player_name"],
                        team_abbr, season_type_label,
                    )
                    if shots is not None:
                        frames.append(shots)
                        log.info("    -> %d shots collected for %s",
                                 len(shots), player["player_name"])
                except Exception as exc:
                    log.error("Shot fetch failed for %s: %s",
                              player["player_name"], exc)

    if not frames:
        log.error("No data collected.")
        return pd.DataFrame(columns=[
            "player_name", "player_id", "team_abbr",
            "game_id", "game_date", "event_num",
            "shot_result", "season_type",
        ])

    result = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["player_id", "game_id", "event_num"])
        .sort_values(["season_type", "team_abbr", "player_name", "game_id", "event_num"])
        .reset_index(drop=True)
    )

    result.to_csv(CACHE_PATH, index=False)
    log.info("Saved %d rows to %s", len(result), CACHE_PATH)
    return result


if __name__ == "__main__":
    df = fetch_shot_sequences(force_refresh=True)
    print(df.head(10))
    print(f"\nTotal shots: {len(df)}")
    print(
        df.groupby(["team_abbr", "season_type", "player_name"])
        .size()
        .rename("shots")
        .reset_index()
        .to_string(index=False)
    )
