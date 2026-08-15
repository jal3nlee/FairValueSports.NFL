# core/nfl_player_search.py
# One standardized NFL player-selection pattern, used everywhere a single
# player needs picking: Lineup Analysis and Prop Leaderboard's Player
# Search subview. Team -> Position -> Player, no free-text search.
import streamlit as st

from core.lineup_data import get_players_by_team, get_players_by_position, NFL_TEAMS

_POSITION_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "K": 4, "DST": 5}
DEFAULT_NFL_TEAM = sorted(NFL_TEAMS.keys())[0]  # "Arizona Cardinals" — first alphabetically, centralized here


def _sort_key(p: dict):
    return (_POSITION_ORDER.get(p.get("position", ""), 99), p.get("name", ""))


def render_nfl_player_search(
    key_prefix: str,
    allowed_positions: list[str] | None = None,
    taken_names: set[str] | None = None,
) -> dict | None:
    """
    Renders the standardized Team | Position | Player controls and returns
    the selected player as {"name", "team", "position", "headshot_url"},
    or None if no eligible player exists for the current Team/Position.

    allowed_positions restricts the Position dropdown's non-"All" options
    (and, when Position="All", which positions are actually eligible) —
    e.g. ["QB","RB","WR","TE"] for skill-position tools. Defaults to those
    four if not given, since no current NFL tool needs K/DST.
    taken_names excludes already-selected players (for multi-player
    comparisons where duplicates aren't allowed).
    """
    allowed_positions = allowed_positions or ["QB", "RB", "WR", "TE"]
    taken_names = taken_names or set()

    st.markdown(
        "<div style='font-size:0.95rem;font-weight:600;margin:0 0 4px 0'>Player Search</div>",
        unsafe_allow_html=True,
    )

    _c1, _c2, _c3 = st.columns([2.2, 1.2, 2.6])
    with _c1:
        team_name = st.selectbox(
            "Team", sorted(NFL_TEAMS.keys()),
            index=sorted(NFL_TEAMS.keys()).index(DEFAULT_NFL_TEAM),
            key=f"{key_prefix}_team", label_visibility="collapsed",
        )
    team_abbr = NFL_TEAMS.get(team_name)

    with _c2:
        position = st.selectbox(
            "Position", ["All"] + allowed_positions,
            key=f"{key_prefix}_position", label_visibility="collapsed",
        )

    roster = get_players_by_team(team_abbr) if team_abbr else []
    if position != "All":
        roster = [p for p in roster if p.get("position") == position]
    else:
        roster = [p for p in roster if p.get("position") in allowed_positions]
    roster = [p for p in roster if p.get("name") and p["name"] not in taken_names]
    roster = sorted(roster, key=_sort_key)

    with _c3:
        if not roster:
            st.selectbox("Player", ["No eligible players"], key=f"{key_prefix}_player_empty",
                         label_visibility="collapsed", disabled=True)
            return None
        _labels = {f"{p['name']} — {p['position']}": p for p in roster}
        _picked_label = st.selectbox(
            "Player", list(_labels.keys()), key=f"{key_prefix}_player", label_visibility="collapsed",
        )
        p = _labels[_picked_label]

    return {"name": p["name"], "team": team_name, "position": p.get("position", ""), "headshot_url": p.get("headshot_url")}
