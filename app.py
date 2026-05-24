"""
app.py
======
Flask dashboard for NBA shooting streakiness analysis.

Routes
------
GET  /                  -> landing page (team + player search)
GET  /player/<name>     -> full analysis page for one player
GET  /overview          -> multi-player DeltaZ summary
GET  /team/<abbr>       -> DeltaZ summary filtered to one team
POST /refresh           -> re-fetch from NBA API (force_refresh=True)
GET  /api/players       -> JSON list of available player names (autocomplete)

Usage
-----
    python app.py
    # then open http://localhost:5050
"""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

from bootstrap_engine import (
    build_game_records,
    run_bootstrap,
    analyse_all_players,
    BootstrapResult,
)
from plots import (
    plot_bootstrap_distribution,
    plot_z_comparison,
    plot_game_timeline,
    plot_fg_vs_z,
    plot_all_players_delta_z,
)

# ── App setup ──────────────────────────────────────────────────────────────────

app = Flask(__name__)

CACHE_PATH = Path(__file__).parent / "shots_cache.csv"

_state_lock = threading.Lock()
_shots_df:  pd.DataFrame | None = None
_results:   dict[str, BootstrapResult] = {}


def _load_data() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH, dtype={"game_id": str})
    return pd.DataFrame(columns=[
        "player_name", "player_id", "team_abbr",
        "game_id", "game_date", "event_num",
        "shot_result", "season_type",
    ])


def _ensure_loaded() -> None:
    global _shots_df, _results
    with _state_lock:
        if _shots_df is None:
            _shots_df = _load_data()
            if not _shots_df.empty:
                _results = analyse_all_players(_shots_df, B=5000)


def _player_summary(name: str) -> dict | None:
    _ensure_loaded()
    if _shots_df is None or _shots_df.empty:
        return None

    player_df = _shots_df[_shots_df["player_name"] == name]
    if player_df.empty:
        return None

    result = _results.get(name)
    if result is None:
        result = run_bootstrap(player_df, name, B=5000)
        if result is None:
            return None
        _results[name] = result

    records_raw = build_game_records(player_df)
    game_records = [
        {"game_id": r.game_id, "season_type": r.season_type,
         "n": r.n, "p": r.p, "z": r.z}
        for r in records_raw
    ]

    return {
        "summary":        result.summary(),
        "plot_bootstrap": plot_bootstrap_distribution(result),
        "plot_z_compare": plot_z_comparison(result),
        "plot_timeline":  plot_game_timeline(game_records, name),
        "plot_fg_vs_z":   plot_fg_vs_z(game_records, name),
        "game_records":   game_records,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    _ensure_loaded()
    players = sorted(_results.keys()) if _results else []
    teams = []
    if _shots_df is not None and not _shots_df.empty:
        teams = sorted(_shots_df["team_abbr"].unique().tolist())
    return render_template("index.html", players=players, teams=teams,
                           data_loaded=not (_shots_df is None or _shots_df.empty))


@app.route("/player/<path:player_name>")
def player_page(player_name: str):
    data = _player_summary(player_name)
    if data is None:
        return render_template(
            "error.html",
            message=f"No data found for '{player_name}'. "
                    "Run /refresh to fetch from the NBA API.",
        )
    return render_template("player.html", player_name=player_name, **data)


@app.route("/overview")
def overview():
    _ensure_loaded()
    if not _results:
        return render_template(
            "error.html",
            message="No player data available. Visit /refresh to fetch from the NBA API.",
        )
    plot_img = plot_all_players_delta_z(_results)
    summaries = sorted(
        [r.summary() for r in _results.values()],
        key=lambda x: x["delta_Z"],
    )
    return render_template("overview.html", plot_img=plot_img, summaries=summaries)


@app.route("/team/<team_abbr>")
def team_page(team_abbr: str):
    _ensure_loaded()
    if _shots_df is None or _shots_df.empty:
        return render_template("error.html",
                               message="No data available. Visit /refresh first.")
    players_in_team = (
        _shots_df[_shots_df["team_abbr"] == team_abbr]["player_name"]
        .unique().tolist()
    )
    team_results = {n: _results[n] for n in players_in_team if n in _results}
    plot_img = plot_all_players_delta_z(team_results)
    summaries = sorted(
        [r.summary() for r in team_results.values()],
        key=lambda x: x["delta_Z"],
    )
    return render_template("overview.html", plot_img=plot_img,
                           summaries=summaries, team_filter=team_abbr)


@app.route("/api/players")
def api_players():
    _ensure_loaded()
    query = request.args.get("q", "").lower()
    if _shots_df is None or _shots_df.empty:
        return jsonify([])
    names = _shots_df["player_name"].unique().tolist()
    if query:
        names = [n for n in names if query in n.lower()]
    return jsonify(sorted(names))


@app.route("/refresh", methods=["GET", "POST"])
def refresh():
    """Re-fetch from NBA API. Shows a confirmation page on GET."""
    if request.method == "GET":
        return render_template("refresh.html")

    global _shots_df, _results
    try:
        from scraper import fetch_shot_sequences
        df = fetch_shot_sequences(force_refresh=True)
        with _state_lock:
            _shots_df = df
            _results  = analyse_all_players(df, B=5000) if not df.empty else {}
        return redirect(url_for("overview"))
    except Exception as exc:
        return render_template("error.html", message=str(exc))


if __name__ == "__main__":
    app.run(debug=True, port=5050)
