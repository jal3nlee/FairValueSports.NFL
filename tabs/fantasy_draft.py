# tabs/fantasy_draft.py
import streamlit as st
import pandas as pd

from core.fantasy_data import (
    load_fantasy_rankings,
    build_ranking_table,
    filter_fantasy_rankings,
)

POSITIONS = ["Overall", "QB", "RB", "WR", "TE", "K", "DST"]


def _fmt_missing(v):
    """Never show NaN/None — always a clean value or an em dash."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return v


def render():
    st.markdown("## Fantasy Draft")

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

    st.caption(
        f"Compare fantasy football draft rankings and ADP across "
        f"{', '.join(platform_cols)} in one place."
    )

    # ── One-row toolbar: position filters left, search right ──────
    _tb1, _tb2 = st.columns([3, 1.3])
    with _tb1:
        _position = st.segmented_control("Position", POSITIONS, default="Overall", key="fd_position") or "Overall"
    with _tb2:
        _search = st.text_input(
            "Search player or team...", key="fd_search",
            label_visibility="collapsed", placeholder="Search player or team...",
        )

    _view = st.segmented_control("View", ["Consensus", "Best Available"], default="Consensus", key="fd_view") or "Consensus"

    _filtered = filter_fantasy_rankings(df, _position, _search)

    if _view == "Best Available":
        # Same ordering as Consensus for now — architecture ready for
        # drafted-player removal once live-draft tracking is built.
        st.session_state.setdefault("fd_drafted_players", set())
        _drafted = st.multiselect(
            "Mark players as drafted", options=sorted(df["Name"].tolist()),
            key="fd_drafted_players_select", label_visibility="collapsed",
            placeholder="Mark players as drafted...",
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
        _display[col] = _filtered[col].apply(_fmt_missing)
    _display["Avg ADP"] = _filtered["AvgADP"].apply(_fmt_missing)

    _col_config = {
        "Rank":    st.column_config.NumberColumn("RANK", width="small"),
        "Player":  st.column_config.TextColumn("PLAYER", width="large"),
        "Pos":     st.column_config.TextColumn("POS", width="small"),
        "Avg ADP": st.column_config.TextColumn("AVG ADP", width="small"),
    }
    for col in platform_cols:
        _col_config[col] = st.column_config.TextColumn(col.upper(), width="small")

    st.dataframe(
        _display,
        use_container_width=True,
        hide_index=True,
        height=min(760, 46 + 35 * len(_display)),
        column_config=_col_config,
    )
    st.caption(f"ADP Sources: {' · '.join(platform_cols)}")
