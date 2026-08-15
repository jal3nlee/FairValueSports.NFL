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

# ── Single shared prop vocabulary — used by BOTH the Leaderboard and
# Player Search subviews, so a stat resolves to the exact same nflverse
# column no matter which workflow the user came in through. ──────────
PROP_STAT_MAP = {
    "Passing Yards":     "passing_yards",
    "Passing TDs":       "passing_tds",
    "Interceptions":     "passing_interceptions",
    "Pass Attempts":     "attempts",
    "Completions":       "completions",
    "Rushing Yards":     "rushing_yards",
    "Rushing Attempts":  "carries",
    "Receiving Yards":   "receiving_yards",
    "Receptions":        "receptions",
    "Targets":           "targets",
}
PROP_POSITION_MAP = {
    "Passing Yards":    ["QB"],
    "Passing TDs":      ["QB"],
    "Interceptions":    ["QB"],
    "Pass Attempts":    ["QB"],
    "Completions":      ["QB"],
    "Rushing Yards":    ["RB", "QB", "WR"],
    "Rushing Attempts": ["RB", "QB", "WR"],
    "Receiving Yards":  ["WR", "TE", "RB"],
    "Receptions":       ["WR", "TE", "RB"],
    "Targets":          ["WR", "TE", "RB"],
}
PROP_AVG_LABEL = {
    "Passing Yards":    "Avg Pass Yds",
    "Passing TDs":      "Avg Pass TDs",
    "Interceptions":    "Avg INT",
    "Pass Attempts":    "Avg Pass Att",
    "Completions":      "Avg Completions",
    "Rushing Yards":    "Avg Rush Yds",
    "Rushing Attempts": "Avg Carries",
    "Receiving Yards":  "Avg Rec Yds",
    "Receptions":       "Avg Rec",
    "Targets":          "Avg Targets",
}
SAMPLE_OPTIONS = {"Last 3 Games": 3, "Last 5 Games": 5, "Last 10 Games": 10, "Season": None}
_SEASON_MIN_GAMES = 3

# Player Search only — Anytime TD has no reliable single nflverse column
# (it's derived from three TD types combined), and isn't part of the
# core shared vocabulary the Leaderboard uses, so it's kept separate
# rather than forced into PROP_STAT_MAP.
PLAYER_SEARCH_EXTRA_STATS = {"Anytime TD": "_any_td"}

# Bridges PROP_STAT_MAP's labels to The Odds API's market keys — a
# genuinely different vocabulary (a betting market, not a stat column),
# used only for pulling CURRENT sportsbook lines in Player Search.
# None = no sportsbook market exists for that stat (historical research
# still works, there's just nothing to show under "Current Market").
PROP_LABEL_TO_ODDS_MARKET = {
    "Passing Yards":    "player_pass_yds",
    "Passing TDs":      "player_pass_tds",
    "Interceptions":    "player_pass_interceptions",
    "Pass Attempts":    "player_pass_attempts",
    "Completions":      "player_pass_completions",
    "Rushing Yards":    "player_rush_yds",
    "Rushing Attempts": "player_rush_attempts",
    "Receiving Yards":  "player_reception_yds",
    "Receptions":       "player_receptions",
    "Targets":          None,
    "Anytime TD":       "player_anytime_td",
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


def get_player_game_log(player_name: str, team_full_name: str, stat_field: str, n_games: int | None = 10) -> list[dict]:
    """
    stat_field is an nflverse column name (from PROP_STAT_MAP's values, or
    "_any_td" for the Player-Search-only Anytime TD derivation) — the SAME
    field vocabulary the Leaderboard uses, so both subviews read identical
    underlying numbers. Pass n_games=None for the full season.
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

        out = []
        for r in rows:
            if stat_field == "_any_td":
                rush_td = r.get("rushing_tds") or 0
                rec_td = r.get("receiving_tds") or 0
                pass_td = r.get("passing_tds") or 0
                val = 1.0 if (rush_td + rec_td + pass_td) > 0 else 0.0
            else:
                val = r.get(stat_field)
            if val is None:
                continue
            out.append({"week": r.get("week"), "opponent": r.get("opponent_team", "—"), "value": float(val)})
        out.sort(key=lambda x: x["week"] or 0, reverse=True)
        return out[:n_games] if n_games else out
    except Exception:
        return []


def calculate_hit_rate(game_log: list[dict], line: float, side: str = "Over") -> dict | None:
    """
    THE single shared hit-rate engine — Leaderboard and Player Search both
    call this, never a separate inline calculation. Pushes (exact line
    matches) are excluded from both the hit count AND the denominator.
    Returns None if there's no game log at all, or every game was a push.
    """
    if not game_log:
        return None
    hits = misses = pushes = 0
    for g in game_log:
        if g["value"] == line:
            pushes += 1
        elif (g["value"] > line if side == "Over" else g["value"] < line):
            hits += 1
        else:
            misses += 1
    decided = hits + misses
    if decided == 0:
        return None
    return {"hits": hits, "total": decided, "pushes": pushes}


# ── Prop Leaderboard ────────────────────────────────────────────
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _all_player_weekly_rows(season: int) -> list[dict]:
    """One bulk pull, cached — every player's every game this season, in one list."""
    stats = _load_player_stats(season)
    if stats is None:
        return []
    try:
        return stats.to_dicts()
    except Exception:
        return []


def build_prop_leaderboard(stat_label: str, side: str, line: float, sample_label: str) -> list[dict]:
    """
    Scans nflverse's full current-season player-week table directly — no
    per-team loop needed. Uses the SAME calculate_hit_rate engine Player
    Search uses, per-player, so results are guaranteed consistent between
    the two subviews for identical inputs.
    """
    season = get_current_season()
    if season is None:
        return []
    field = PROP_STAT_MAP.get(stat_label)
    eligible_positions = PROP_POSITION_MAP.get(stat_label, [])
    n_games = SAMPLE_OPTIONS.get(sample_label)
    if not field or not eligible_positions:
        return []

    rows = _all_player_weekly_rows(season)
    if not rows:
        return []

    by_player: dict[tuple, list[dict]] = {}
    positions_by_player: dict[tuple, str] = {}
    for r in rows:
        if r.get("position") not in eligible_positions:
            continue
        val = r.get(field)
        if val is None:
            continue
        key = (r.get("player_display_name") or r.get("player_name"), r.get("team"))
        by_player.setdefault(key, []).append({"week": r.get("week"), "value": float(val)})
        positions_by_player[key] = r.get("position")

    results = []
    for (name, team), games in by_player.items():
        games.sort(key=lambda g: g["week"] or 0, reverse=True)

        if n_games is not None:
            if len(games) < n_games:
                continue
            sample = games[:n_games]
        else:
            if len(games) < _SEASON_MIN_GAMES:
                continue
            sample = games

        hr = calculate_hit_rate(sample, line, side)
        if not hr:
            continue

        results.append({
            "player": name,
            "team": team,
            "position": positions_by_player.get((name, team), ""),
            "hits": hr["hits"],
            "games": hr["total"],
            "pushes": hr["pushes"],
            "hit_rate": hr["hits"] / hr["total"] * 100,
            "avg": round(sum(g["value"] for g in sample) / len(sample), 1),
        })

    results.sort(key=lambda r: (r["hit_rate"], r["hits"], r["games"]), reverse=True)
    return results[:10]
