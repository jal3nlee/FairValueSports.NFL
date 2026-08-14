# tabs/fantasy_draft.py
import streamlit as st
import pandas as pd

from core.fantasy_data import (
    load_fantasy_rankings,
    build_ranking_table,
    filter_fantasy_rankings,
    get_platform_columns,
)

POSITIONS = ["Overall", "QB", "RB", "WR", "TE", "K", "DST"]


def render():
    st.markdown("## Fantasy Draft")
    st.caption(
        "Consensus draft rankings pulled from ESPN, Sleeper, CBS, NFL, RTSports, and Fantrax — "
        "see where the fantasy market agrees, and where it doesn't."
    )

    raw = load_fantasy_rankings()
    if raw.empty:
        st.info(
            "No fantasy rankings data found. Add the ADP file to the repo at "
            "`data/fantasy_adp.xlsx` to populate this tab."
        )
        return

    df, platform_cols = build_ranking_table(raw)
    if df.empty:
        st.warning("Rankings data couldn't be processed. Check the file formatting.")
        return

    with st.container(border=True):
        st.markdown(
            "**Consensus ADP** is the average draft position across every platform that ranked a "
            "player — missing platforms are skipped, never counted as a 0. **Range** is the gap "
            "between the highest and lowest platform ranking for that player, a quick read on how "
            "much the fantasy market actually agrees on their value."
        )

    _pos_col, _search_col = st.columns([2.5, 2])
    with _pos_col:
        _position = st.segmented_control("Position", POSITIONS, default="Overall", key="fd_position") or "Overall"
    with _search_col:
        _search = st.text_input(
            "Search player or team...", key="fd_search",
            label_visibility="collapsed", placeholder="Search player or team...",
        )

    _view = st.segmented_control(
        "View", ["Consensus", "Biggest Disagreement", "Best Available"], default="Consensus", key="fd_view",
    ) or "Consensus"

    _filtered = filter_fantasy_rankings(df, _position, _search)

    if _view == "Biggest Disagreement":
        _filtered = _filtered.sort_values("Range", ascending=False, na_position="last").reset_index(drop=True)
    elif _view == "Best Available":
        # Same order as Consensus for now — the drafted-player hide/track
        # logic below is wired up so this becomes a real "who's left"
        # view once live-draft tracking is built out.
        st.session_state.setdefault("fd_drafted_players", set())
        _drafted = st.multiselect(
            "Mark players as drafted (removes them from Best Available)",
            options=sorted(df["Name"].tolist()),
            key="fd_drafted_players_select",
        )
        st.session_state["fd_drafted_players"] = set(_drafted)
        _filtered = _filtered[~_filtered["Name"].isin(st.session_state["fd_drafted_players"])]
        _filtered = _filtered.sort_values("AvgADP", ascending=True, na_position="last").reset_index(drop=True)

    if _filtered.empty:
        st.info("No players match the current filters.")
        return

    def _fmt_player(row):
        loc = row["Team"] or ""
        if row["Bye"]:
            loc = f"{loc} ({row['Bye']})" if loc else f"({row['Bye']})"
        return f"{row['Name']} — {loc}" if loc else row["Name"]

    _display = pd.DataFrame({
        "Rank": _filtered["ConsensusRank"],
        "Player": _filtered.apply(_fmt_player, axis=1),
        "Pos": _filtered["PosRank"],
    })
    for col in platform_cols:
        _display[col] = _filtered[col]
    _display["Avg ADP"] = _filtered["AvgADP"]
    _display["Range"] = _filtered["Range"]
    _display["Agreement"] = _filtered["Agreement"]

    st.dataframe(
        _display,
        use_container_width=True,
        hide_index=True,
        height=min(700, 46 + 35 * len(_display)),
        column_config={
            "Rank":       st.column_config.NumberColumn("Rank", width="small"),
            "Player":     st.column_config.TextColumn("Player", width="large"),
            "Pos":        st.column_config.TextColumn("Pos", width="small"),
            "Avg ADP":    st.column_config.NumberColumn("Avg ADP", format="%.1f"),
            "Range":      st.column_config.NumberColumn("Range"),
            "Agreement":  st.column_config.TextColumn(
                "Agreement", help="Flagged when platforms disagree more than the typical spread in this dataset.",
            ),
        },
    )
    st.caption(
        "Click any column header to sort — defaults to Consensus Rank. "
        "On narrow screens, scroll the table horizontally to see every platform column."
    )
    st.caption(f"ADP Sources: {', '.join(platform_cols)}")
