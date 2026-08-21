# core/nfl_matchup_context.py
# Game-level "how do these teams compare / how have they been playing"
# context for Matchup Center — built entirely from data nflreadpy already
# provides (team_stats, schedules) and the existing cached defense
# summary (core/nfl_defense_data.py). No new external data source.
#
# `season` is accepted as an explicit parameter (mirroring
# build_team_defense_summary's own signature) rather than resolved
# internally via get_current_season() — Matchup Center calls these
# functions once per team per game, and get_current_season() currently
# has a debug-caption side effect (nflverse_data.DEBUG_SEASON) that would
# otherwise spam the page when called that many times in one render.
from core.nflverse_data import _load_team_stats, _load_teams, _load_schedules, _team_abbr_for
from core.nfl_defense_data import build_team_defense_summary

TEAM_COMPARISON_METRICS = [
    ("points_per_game", "Points / Game"),
    ("points_allowed_per_game", "Points Allowed / Game"),
    ("total_yards_per_game", "Total Yards / Game"),
    ("passing_yards_per_game", "Passing Yards / Game"),
    ("rushing_yards_per_game", "Rushing Yards / Game"),
    ("pass_yards_allowed_per_game", "Pass Yards Allowed / Game"),
    ("rush_yards_allowed_per_game", "Rush Yards Allowed / Game"),
    ("turnover_margin", "Turnover Margin"),
]


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def _avg(rows, field) -> float | None:
    return _mean([r.get(field) for r in rows])


def get_team_comparison_stats(season: int, team_full_name: str) -> dict | None:
    """Season-to-date offense/defense/turnover snapshot for one team.
    Offense (passing/rushing yards, turnovers) comes from nflreadpy's
    team_stats own-team weekly rows. Points per game come from
    nflreadpy's schedules (actual final scores — team_stats has no
    points field). Yards allowed reuse the already-cached
    build_team_defense_summary (same function Lineup Analysis/Prop
    Research already compute) rather than re-deriving it here. Returns
    None if the team can't be resolved or it has no games played yet."""
    if season is None:
        return None
    teams_df = _load_teams()
    team_abbr = _team_abbr_for(team_full_name, teams_df)
    if not team_abbr:
        return None

    team_stats = _load_team_stats(season)
    if team_stats is None:
        return None
    try:
        own_rows = team_stats.filter(team_stats["team"] == team_abbr).to_dicts()
    except Exception:
        return None
    games_played = len(own_rows)
    if games_played == 0:
        return None

    passing_yards_pg = _avg(own_rows, "passing_yards")
    rushing_yards_pg = _avg(own_rows, "rushing_yards")
    total_yards_pg = (
        round(passing_yards_pg + rushing_yards_pg, 1)
        if passing_yards_pg is not None and rushing_yards_pg is not None else None
    )

    takeaways = sum(
        (r.get("def_interceptions") or 0) + (r.get("fumble_recovery_opp") or 0)
        for r in own_rows
    )
    giveaways = sum(
        (r.get("passing_interceptions") or 0) + (r.get("sack_fumbles_lost") or 0)
        + (r.get("rushing_fumbles_lost") or 0) + (r.get("receiving_fumbles_lost") or 0)
        for r in own_rows
    )
    turnover_margin = takeaways - giveaways

    defense_summary = build_team_defense_summary(season, "PPR")
    team_defense = defense_summary.get(team_abbr, {})

    points_pg = points_allowed_pg = None
    schedules = _load_schedules(season)
    if schedules is not None:
        try:
            mask = (schedules["home_team"] == team_abbr) | (schedules["away_team"] == team_abbr)
            team_games = schedules.filter(mask)
            team_games = team_games.filter(team_games["home_score"].is_not_null())
            games = team_games.to_dicts()
            scored = [g["home_score"] if g["home_team"] == team_abbr else g["away_score"] for g in games]
            allowed = [g["away_score"] if g["home_team"] == team_abbr else g["home_score"] for g in games]
            points_pg = _mean(scored)
            points_allowed_pg = _mean(allowed)
        except Exception:
            pass

    return {
        "games_played": games_played,
        "points_per_game": points_pg,
        "points_allowed_per_game": points_allowed_pg,
        "total_yards_per_game": total_yards_pg,
        "passing_yards_per_game": passing_yards_pg,
        "rushing_yards_per_game": rushing_yards_pg,
        "pass_yards_allowed_per_game": team_defense.get("pass_yards_allowed_per_game"),
        "rush_yards_allowed_per_game": team_defense.get("rush_yards_allowed_per_game"),
        "turnover_margin": turnover_margin,
    }


def get_team_recent_form(season: int, team_full_name: str, n: int = 5) -> dict | None:
    """Last-N completed-game record/scoring summary from nflreadpy's
    schedules — actual final scores only, no betting-trend calculation
    (e.g. ATS) is derived even though schedules carries a spread_line
    column, since that would be new methodology, not a reuse of an
    existing one. Returns None if the team can't be resolved or it has
    no completed games yet this season."""
    if season is None:
        return None
    teams_df = _load_teams()
    team_abbr = _team_abbr_for(team_full_name, teams_df)
    if not team_abbr:
        return None
    schedules = _load_schedules(season)
    if schedules is None:
        return None

    try:
        mask = (schedules["home_team"] == team_abbr) | (schedules["away_team"] == team_abbr)
        team_games = schedules.filter(mask)
        team_games = team_games.filter(team_games["home_score"].is_not_null())
        games = team_games.to_dicts()
    except Exception:
        return None
    if not games:
        return None

    games.sort(key=lambda g: (g.get("season") or 0, g.get("week") or 0))
    recent = games[-n:]

    rows = []
    wins = losses = ties = 0
    for g in recent:
        is_home = g.get("home_team") == team_abbr
        own_score = g.get("home_score") if is_home else g.get("away_score")
        opp_score = g.get("away_score") if is_home else g.get("home_score")
        opponent = g.get("away_team") if is_home else g.get("home_team")
        if own_score > opp_score:
            result, wins = "W", wins + 1
        elif own_score < opp_score:
            result, losses = "L", losses + 1
        else:
            result, ties = "T", ties + 1
        rows.append({
            "week": g.get("week"), "opponent": opponent, "location": "vs" if is_home else "@",
            "result": result, "own_score": own_score, "opp_score": opp_score,
        })

    n_games = len(recent)
    avg_scored = round(sum(r["own_score"] for r in rows) / n_games, 1)
    avg_allowed = round(sum(r["opp_score"] for r in rows) / n_games, 1)

    return {
        "games": rows,
        "record": f"{wins}-{losses}" + (f"-{ties}" if ties else ""),
        "avg_scored": avg_scored,
        "avg_allowed": avg_allowed,
        "avg_margin": round(avg_scored - avg_allowed, 1),
    }
