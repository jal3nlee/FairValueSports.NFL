# core/nfl_injuries_data.py
# Official NFL weekly injury report data, from nflreadpy's load_injuries
# (nflverse) — the same package/trust-tier already relied on elsewhere in
# this app (Lineup Analysis, Prop Research). No new external data source.
from core.nflverse_data import _load_injuries, _load_teams, _team_abbr_for


def get_team_injuries(season: int, team_full_name: str) -> list[dict]:
    """Player/Position/Status from this team's most recent official
    injury report this season (the highest `week` value present for that
    team, rather than trying to reconcile nflverse's week numbering
    against the app's own week-index logic). report_status is shown
    verbatim (e.g. "Out", "Doubtful", "Questionable") — there is no
    existing status-normalization convention in this project to follow,
    so none is invented here. Returns [] if the team can't be resolved,
    the data is unavailable, or no players are currently listed — an
    empty result should be treated as "nothing to show," not an error."""
    if season is None:
        return []
    teams_df = _load_teams()
    team_abbr = _team_abbr_for(team_full_name, teams_df)
    if not team_abbr:
        return []
    injuries = _load_injuries(season)
    if injuries is None:
        return []

    try:
        team_rows = injuries.filter(injuries["team"] == team_abbr).to_dicts()
    except Exception:
        return []
    if not team_rows:
        return []

    latest_week = max((r.get("week") or 0) for r in team_rows)
    latest_rows = [r for r in team_rows if r.get("week") == latest_week]

    out = []
    for r in latest_rows:
        status = r.get("report_status")
        if not status:
            continue
        out.append({
            "Player": r.get("full_name", "—"),
            "Position": r.get("position", "—"),
            "Status": status,
        })
    out.sort(key=lambda r: r["Player"])
    return out
