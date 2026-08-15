# core/nfl_player_context.py
# Shared NFL context rendering used by BOTH Lineup Analysis and Prop
# Research's Player Context — one implementation, not two copies that
# can drift apart. Covers: season-aware week labeling, and the
# opponent-defense table (which never needs the player's name, only
# their position and upcoming opponent).
import streamlit as st
import pandas as pd

from core.nfl_defense_data import get_opponent_defense, POSITION_DEFENSE_METRICS


def format_nfl_week(week, season, current_season, compact: bool = False) -> str:
    """
    Current-season game: 'W8'. Prior-season game: 'W17 2025' (or
    'W17 '25' if compact=True, for chart labels). Uses the game's own
    recorded season — never inferred from the week number.
    """
    if season is not None and current_season is not None and season != current_season:
        return f"W{week} '{str(season)[-2:]}" if compact else f"W{week} {season}"
    return f"W{week}"


def render_opponent_defense_single(opponent: str | None, position: str, scoring: str = "PPR"):
    """
    One player's opponent-defense table — general season-long defensive
    stats for the opponent, aggregated across ALL offensive players they've
    faced league-wide. Not the selected player's personal history against
    that team. Never takes a player name — the header is built entirely
    from the opponent + position.
    """
    if not opponent:
        st.caption("No opponent this week (bye week).")
        return

    _def = get_opponent_defense(opponent, position, scoring)
    _metric_set = POSITION_DEFENSE_METRICS.get(position, [])
    if not _def or not _metric_set:
        st.caption("Opponent defensive data is not available yet.")
        return

    _header = f"{opponent} Defense vs {position}"
    _rows = [{"Metric": label, _header: _def.get(field, "—")} for field, label in _metric_set]
    st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
    st.caption("Defensive data: nflverse")


def render_opponent_defense_multi(players_with_opponents: list[dict], scoring: str = "PPR"):
    """
    players_with_opponents: [{"name", "position", "opponent"}, ...] — used
    for Lineup Analysis's multi-player comparison. Same general opponent
    defensive stats as above, matched to each selected player's position.
    """
    if all(not p.get("opponent") for p in players_with_opponents):
        st.caption("No opponent this week (bye week).")
        return

    positions = {p["position"] for p in players_with_opponents}
    if len(positions) != 1:
        st.caption("Select players at the same position to see matched defensive context.")
        return
    position = next(iter(positions))
    metric_set = POSITION_DEFENSE_METRICS.get(position, [])
    if not metric_set:
        st.caption("Opponent defensive data is not available yet.")
        return

    def_by_player = {
        p["name"]: get_opponent_defense(p.get("opponent"), position, scoring) for p in players_with_opponents
    }
    headers = [
        f"{p['opponent']} Defense vs {position}" if p.get("opponent") else "Bye Week"
        for p in players_with_opponents
    ]

    rows = []
    for field, label in metric_set:
        row = [label]
        for p in players_with_opponents:
            d = def_by_player.get(p["name"])
            v = d.get(field) if d else None
            row.append(v if v is not None else "—")
        rows.append(row)

    st.dataframe(pd.DataFrame(rows, columns=["Metric"] + headers), use_container_width=True, hide_index=True)
    st.caption("Defensive data: nflverse")
