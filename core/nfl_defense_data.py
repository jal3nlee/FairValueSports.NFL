# core/nfl_defense_data.py
# Opponent defensive context, derived from nflverse team_stats/player_stats.
# "Yards allowed" = the OPPONENT's own stat line in that game — team_stats
# already carries an 'opponent_team' column, so no play-by-play needed.
import streamlit as st

from core.nflverse_data import (
    _load_player_stats, _load_team_stats, _load_teams,
    _team_abbr_for, get_current_season,
)

POSITION_DEFENSE_METRICS = {
    "WR": [
        ("pass_yards_allowed_per_game", "Pass Yds Allowed / G"),
        ("wr_rec_yards_allowed_per_game", "WR Rec Yds Allowed / G"),
        ("pass_td_allowed_per_game", "Pass TD Allowed / G"),
        ("wr_fantasy_pts_allowed_per_game", "WR Fantasy Pts Allowed / G"),
    ],
    "TE": [
        ("pass_yards_allowed_per_game", "Pass Yds Allowed / G"),
        ("te_rec_yards_allowed_per_game", "TE Rec Yds Allowed / G"),
        ("pass_td_allowed_per_game", "Pass TD Allowed / G"),
        ("te_fantasy_pts_allowed_per_game", "TE Fantasy Pts Allowed / G"),
    ],
    "RB": [
        ("rush_yards_allowed_per_game", "Rush Yds Allowed / G"),
        ("yards_per_carry_allowed", "Yards / Carry Allowed"),
        ("rush_td_allowed_per_game", "Rush TD Allowed / G"),
        ("rb_fantasy_pts_allowed_per_game", "RB Fantasy Pts Allowed / G"),
    ],
    "QB": [
        ("pass_yards_allowed_per_game", "Pass Yds Allowed / G"),
        ("pass_td_allowed_per_game", "Pass TD Allowed / G"),
        ("interceptions_forced_per_game", "INT / Game"),
        ("sacks_per_game", "Sacks / Game"),
        ("qb_fantasy_pts_allowed_per_game", "QB Fantasy Pts Allowed / G"),
    ],
}


def _half_ppr(fantasy_points, receptions):
    if fantasy_points is None:
        return None
    return fantasy_points + 0.5 * (receptions or 0)


def _fantasy_points_for(row: dict, scoring: str):
    if scoring == "PPR":
        return row.get("fantasy_points_ppr")
    if scoring == "Half PPR":
        return _half_ppr(row.get("fantasy_points"), row.get("receptions"))
    return row.get("fantasy_points")  # Standard


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def build_team_defense_summary(season: int, scoring: str) -> dict:
    """
    One shared, cached computation per (season, scoring) pair — not
    recalculated per player selection. Returns {team_abbr: {metrics}}.
    """
    team_stats = _load_team_stats(season)
    player_stats = _load_player_stats(season)
    if team_stats is None or player_stats is None:
        return {}

    try:
        team_rows = team_stats.to_dicts()
        player_rows = player_stats.to_dicts()
    except Exception:
        return {}

    teams = sorted(set(r["team"] for r in team_rows if r.get("team")))
    summary = {}

    for team in teams:
        games_faced = [r for r in team_rows if r.get("opponent_team") == team]  # opponent's own lines vs this team
        own_def_rows = [r for r in team_rows if r.get("team") == team]
        games_played = len(own_def_rows)
        if games_played == 0:
            continue

        def _avg(rows, field):
            vals = [r.get(field) for r in rows if r.get(field) is not None]
            return round(sum(vals) / len(vals), 1) if vals else None

        pass_yards_allowed = _avg(games_faced, "passing_yards")
        rush_yards_allowed = _avg(games_faced, "rushing_yards")
        pass_td_allowed = _avg(games_faced, "passing_tds")
        rush_td_allowed = _avg(games_faced, "rushing_tds")
        sacks_per_game = _avg(own_def_rows, "def_sacks")
        ints_forced = _avg(own_def_rows, "def_interceptions")

        total_carries_allowed = sum((r.get("carries") or 0) for r in games_faced)
        total_rush_yards_allowed = sum((r.get("rushing_yards") or 0) for r in games_faced)
        ypc_allowed = round(total_rush_yards_allowed / total_carries_allowed, 1) if total_carries_allowed else None

        # Position-specific: opposing players' own game lines, in games played AGAINST this team.
        opp_player_rows = [r for r in player_rows if r.get("opponent_team") == team]

        def _position_allowed(position: str, yard_field: str | None):
            pos_rows = [r for r in opp_player_rows if r.get("position") == position]
            by_game = {}
            for r in pos_rows:
                gid = r.get("game_id")
                by_game.setdefault(gid, {"fp": 0.0, "yds": 0.0})
                fp = _fantasy_points_for(r, scoring)
                if fp is not None:
                    by_game[gid]["fp"] += fp
                if yard_field and r.get(yard_field) is not None:
                    by_game[gid]["yds"] += r[yard_field]
            n = len(by_game) or 1
            fp_avg = round(sum(g["fp"] for g in by_game.values()) / n, 1) if by_game else None
            yds_avg = round(sum(g["yds"] for g in by_game.values()) / n, 1) if by_game and yard_field else None
            return fp_avg, yds_avg

        wr_fp, wr_yds = _position_allowed("WR", "receiving_yards")
        te_fp, te_yds = _position_allowed("TE", "receiving_yards")
        rb_fp, _ = _position_allowed("RB", None)
        qb_fp, _ = _position_allowed("QB", None)

        summary[team] = {
            "games_played": games_played,
            "pass_yards_allowed_per_game": pass_yards_allowed,
            "rush_yards_allowed_per_game": rush_yards_allowed,
            "pass_td_allowed_per_game": pass_td_allowed,
            "rush_td_allowed_per_game": rush_td_allowed,
            "sacks_per_game": sacks_per_game,
            "interceptions_forced_per_game": ints_forced,
            "yards_per_carry_allowed": ypc_allowed,
            "wr_rec_yards_allowed_per_game": wr_yds,
            "te_rec_yards_allowed_per_game": te_yds,
            "wr_fantasy_pts_allowed_per_game": wr_fp,
            "te_fantasy_pts_allowed_per_game": te_fp,
            "rb_fantasy_pts_allowed_per_game": rb_fp,
            "qb_fantasy_pts_allowed_per_game": qb_fp,
        }
    return summary


def get_opponent_defense(opponent_team_full_name: str | None, position: str, scoring: str) -> dict | None:
    """Returns None if there's no opponent (bye week) or defense data unavailable."""
    if not opponent_team_full_name:
        return None
    season = get_current_season()
    if season is None:
        return None
    teams_df = _load_teams()
    team_abbr = _team_abbr_for(opponent_team_full_name, teams_df)
    if not team_abbr:
        return None
    summary = build_team_defense_summary(season, scoring)
    return summary.get(team_abbr)
