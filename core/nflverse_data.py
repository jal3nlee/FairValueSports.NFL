# core/nflverse_data.py
import streamlit as st

try:
    import nflreadpy as nfl
    _NFLVERSE_AVAILABLE = True
except Exception:
    _NFLVERSE_AVAILABLE = False

RECENCY_DECAY = 0.85

POSITION_METRICS = {
    "WR": ["targets_per_game", "target_share", "receiving_yards_per_game"],
    "TE": ["targets_per_game", "target_share", "receiving_yards_per_game"],
    "RB": ["carries_per_game", "carry_share", "targets_per_game"],
    "QB": ["attempts_per_game", "rush_attempts_per_game", "passing_yards_per_game"],
}

METRIC_LABELS = {
    "targets_per_game":        "Targets / Game",
    "target_share":            "Target Share",
    "receiving_yards_per_game":"Receiving Yards / Game",
    "carries_per_game":        "Carries / Game",
    "carry_share":             "Carry Share",
    "attempts_per_game":       "Pass Attempts / Game",
    "rush_attempts_per_game":  "Rush Attempts / Game",
    "passing_yards_per_game":  "Passing Yards / Game",
}
PERCENT_METRICS = {"target_share", "carry_share"}

_BLEND_SCHEDULE = {0: 0.40, 1: 0.55, 2: 0.70, 3: 0.80, 4: 0.90, 5: 0.90}
_BLEND_FLOOR_GAMES = 6

# Maps Odds API market keys (from core/lineup_data.py) to nflverse's
# per-game field names, so a prop line can be checked against real
# historical results without a second, separate stat vocabulary.
MARKET_TO_STAT_FIELD = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_pass_interceptions": "passing_interceptions",
    "player_pass_attempts": "attempts",
    "player_pass_completions": "completions",
    "player_rush_yds": "rushing_yards",
    "player_rush_attempts": "carries",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
    "player_anytime_td": "_any_td",  # derived, not a raw column
}


def _norm_name(name: str) -> str:
    if not name:
        return ""
    n = name.lower().strip()
    for suffix in (" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_player_stats(season: int):
    if not _NFLVERSE_AVAILABLE:
        return None
    try:
        return nfl.load_player_stats(seasons=season, summary_level="week")
    except Exception:
        return None


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_team_stats(season: int):
    if not _NFLVERSE_AVAILABLE:
        return None
    try:
        return nfl.load_team_stats(seasons=season, summary_level="week")
    except Exception:
        return None


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _load_teams():
    if not _NFLVERSE_AVAILABLE:
        return None
    try:
        return nfl.load_teams()
    except Exception:
        return None


def get_current_season() -> int:
    if not _NFLVERSE_AVAILABLE:
        return None
    try:
        return nfl.get_current_season()
    except Exception:
        return None


def recency_weighted_average(values: list, decay: float = RECENCY_DECAY):
    """values: chronological order, OLDEST first, most recent LAST."""
    pairs = [(i, v) for i, v in enumerate(values) if v is not None]
    if not pairs:
        return None
    n = len(values)
    weighted_sum, weight_total = 0.0, 0.0
    for i, v in pairs:
        games_ago = (n - 1) - i
        w = decay ** games_ago
        weighted_sum += v * w
        weight_total += w
    return round(weighted_sum / weight_total, 2) if weight_total > 0 else None


def blend_prior_season(current_value, prior_value, games_played: int, continuity: bool = True):
    if current_value is None and prior_value is None:
        return None
    if not continuity or prior_value is None or games_played >= _BLEND_FLOOR_GAMES:
        return current_value
    if current_value is None:
        return prior_value
    current_weight = _BLEND_SCHEDULE.get(games_played, 0.90)
    return round(current_value * current_weight + prior_value * (1 - current_weight), 2)


def _build_weekly_rows(player_stats, team_stats, player_name: str, team_abbr: str):
    if player_stats is None:
        return []
    try:
        target_name = _norm_name(player_name)
        rows = player_stats.filter(player_stats["team"] == team_abbr).to_dicts()
        rows = [r for r in rows if _norm_name(r.get("player_display_name") or r.get("player_name") or "") == target_name]
        rows.sort(key=lambda r: (r.get("season", 0), r.get("week", 0)))

        team_carries_by_week = {}
        if team_stats is not None:
            for r in team_stats.filter(team_stats["team"] == team_abbr).to_dicts():
                team_carries_by_week[(r["season"], r["week"])] = r.get("carries")

        for r in rows:
            _team_carries = team_carries_by_week.get((r.get("season"), r.get("week")))
            r["_carry_share"] = (
                round(r["carries"] / _team_carries, 3)
                if r.get("carries") is not None and _team_carries else None
            )
        return rows
    except Exception:
        return []


def _team_abbr_for(team_full_name: str, teams_df) -> str | None:
    if teams_df is None or not team_full_name:
        return None
    try:
        matches = teams_df.filter(teams_df["team_name"] == team_full_name)
        if matches.height == 0:
            return None
        return matches["team_abbr"][0]
    except Exception:
        return None


def get_player_usage(player_name: str, team_full_name: str, position: str, prior_team_full_name: str | None = None) -> dict:
    """Returns {} on any failure — callers must keep working even if nflverse is unreachable."""
    if not _NFLVERSE_AVAILABLE:
        return {}
    try:
        season = get_current_season()
        if season is None:
            return {}
        teams_df = _load_teams()
        team_abbr = _team_abbr_for(team_full_name, teams_df)
        if not team_abbr:
            return {}

        cur_stats = _load_player_stats(season)
        cur_team_stats = _load_team_stats(season)
        cur_rows = _build_weekly_rows(cur_stats, cur_team_stats, player_name, team_abbr)
        games_played = len(cur_rows)

        continuity = True
        if prior_team_full_name and prior_team_full_name != team_full_name:
            continuity = False

        prior_rows = []
        if games_played < _BLEND_FLOOR_GAMES:
            prior_stats = _load_player_stats(season - 1)
            prior_team_stats = _load_team_stats(season - 1)
            prior_rows = _build_weekly_rows(prior_stats, prior_team_stats, player_name, team_abbr)

        def _series(field):
            return [r.get(field) for r in cur_rows]

        def _prior_avg(field):
            vals = [r.get(field) for r in prior_rows if r.get(field) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        def _season_avg(field):
            vals = [r.get(field) for r in cur_rows if r.get(field) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        def _metric(field, per_game_source=None):
            source = field if per_game_source is None else per_game_source
            season_val = _season_avg(source)
            current_val = recency_weighted_average(_series(source))
            prior_val = _prior_avg(source)
            blended = blend_prior_season(current_val, prior_val, games_played, continuity)
            return {"season": season_val, "current_role": blended, "games_played": games_played}

        result = {"games_played": games_played, "position": position}
        if position in ("WR", "TE"):
            result["targets_per_game"] = _metric("targets")
            result["target_share"] = _metric("target_share")
            result["receiving_yards_per_game"] = _metric("receiving_yards")
        elif position == "RB":
            result["carries_per_game"] = _metric("carries")
            result["carry_share"] = _metric("_carry_share")
            result["targets_per_game"] = _metric("targets")
        elif position == "QB":
            result["attempts_per_game"] = _metric("attempts")
            result["rush_attempts_per_game"] = _metric("carries")
            result["passing_yards_per_game"] = _metric("passing_yards")
        return result
    except Exception:
        return {}


def get_player_game_log(player_name: str, team_full_name: str, market_key: str, n_games: int = 10) -> list[dict]:
    """
    Returns up to n_games most-recent CURRENT-SEASON games as
    [{"week": int, "opponent": str, "value": float}], most recent first.
    Pass n_games=None for the full season. Empty list on any failure.
    """
    if not _NFLVERSE_AVAILABLE:
        return []
    try:
        season = get_current_season()
        if season is None:
            return []
        teams_df = _load_teams()
        team_abbr = _team_abbr_for(team_full_name, teams_df)
        if not team_abbr:
            return []
        stats = _load_player_stats(season)
        rows = _build_weekly_rows(stats, _load_team_stats(season), player_name, team_abbr)
        if not rows:
            return []

        field = MARKET_TO_STAT_FIELD.get(market_key)
        out = []
        for r in rows:
            if field == "_any_td":
                rush_td = r.get("rushing_tds") or 0
                rec_td = r.get("receiving_tds") or 0
                pass_td = r.get("passing_tds") or 0
                val = 1.0 if (rush_td + rec_td + pass_td) > 0 else 0.0
            else:
                val = r.get(field)
            if val is None:
                continue
            out.append({"week": r.get("week"), "opponent": r.get("opponent_team", "—"), "value": float(val)})
        out.sort(key=lambda x: x["week"] or 0, reverse=True)
        return out[:n_games] if n_games else out
    except Exception:
        return []


def calculate_hit_rate(game_log: list[dict], line: float, side: str = "Over") -> dict | None:
    """side: 'Over' or 'Under'. Returns None if there's no game log to check."""
    if not game_log:
        return None
    hits = sum(
        1 for g in game_log
        if (g["value"] > line if side == "Over" else g["value"] < line)
    )
    return {"hits": hits, "total": len(game_log)}
