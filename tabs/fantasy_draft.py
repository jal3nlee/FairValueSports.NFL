# tabs/fantasy_draft.py
import streamlit as st
import pandas as pd

from core.fantasy_data import load_fantasy_rankings, build_ranking_table, filter_fantasy_rankings

POSITIONS = ["Overall", "QB", "RB", "WR", "TE", "K", "DST"]


def _fmt_missing(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return v


def render():
    st.markdown("## Fantasy Draft")
    st.caption(
        "Compare fantasy football draft rankings and ADP across ESPN, Sleeper, CBS, NFL, RTSports, "
        "and Fantrax in one place."
    )

    st.session_state.setdefault("fd_drafted_ids", set())
    st.session_state.setdefault("fd_editor_version", 0)
    st.session_state.setdefault("fd_editor_row_ids", [])

    # ── Scoring + Position, one row ──────────────────────
    _sc1, _sc2 = st.columns([1.6, 3])
    with _sc1:
        st.caption("Scoring")
        _scoring = st.segmented_control("Scoring", ["PPR", "Half PPR", "Standard"], default="PPR",
                                         key="fd_scoring", label_visibility="collapsed") or "PPR"
    with _sc2:
        st.caption("Position")
        _position = st.segmented_control("Position", POSITIONS, default="Overall",
                                          key="fd_position", label_visibility="collapsed") or "Overall"

    raw = load_fantasy_rankings(scoring=_scoring)
    if raw.empty:
        st.info("No fantasy rankings data found. Add the ADP file to the repo at `data/fantasy_adp.xlsx`.")
        return
    df, platform_cols = build_ranking_table(raw)
    if df.empty:
        st.warning("Rankings data couldn't be processed. Check the file formatting.")
        return

    # ── Apply Update/Reset from the PREVIOUS render's staged edits ──
    # (button code sits above the editor, but the editor's staged
    # checkbox state from last run is already sitting in session_state
    # before this script starts executing — that's what lets the
    # button "read what was just unchecked" even though it renders first.)
    _editor_key = f"fd_editor_{st.session_state.fd_editor_version}"

    _status_col, _reset_col, _update_col = st.columns([2.2, 1, 1.2])

    def _apply_pending_drafts():
        _edits = st.session_state.get(_editor_key, {}).get("edited_rows", {})
        _row_ids = st.session_state.get("fd_editor_row_ids", [])
        for _idx, _change in _edits.items():
            if _change.get("Available") is False and _idx < len(_row_ids):
                st.session_state.fd_drafted_ids.add(_row_ids[_idx])
        st.session_state.fd_editor_version += 1  # force a fresh widget, discard stale edited_rows

    def _reset_board():
        st.session_state.fd_drafted_ids = set()
        st.session_state.fd_editor_version += 1

    with _update_col:
        if st.button("Update Draft Board", type="primary", use_container_width=True, key="fd_update_btn"):
            _apply_pending_drafts()
            st.rerun()
    with _reset_col:
        if st.button("Reset Draft Board", use_container_width=True, key="fd_reset_btn"):
            _reset_board()
            st.rerun()

    # ── Drafted removed BEFORE position filter, per spec ─────────
    _available_full = df[~df["PlayerID"].isin(st.session_state.fd_drafted_ids)].reset_index(drop=True)
    _available = filter_fantasy_rankings(_available_full, _position)

    with _status_col:
        st.caption(f"{len(_available_full)} Available · {len(st.session_state.fd_drafted_ids)} Drafted")

    if _available.empty:
        st.info("No players match the current filters, or all matching players have been drafted.")
        return

    def _fmt_player(row):
        return f"{row['Name']} — {row['Team']}" if row["Team"] else row["Name"]

    _display = pd.DataFrame({
        "Available": True,  # everyone still visible is, by definition, currently available
        "Rank": _available["ConsensusRank"],
        "Player": _available.apply(_fmt_player, axis=1),
        "Pos": _available["Position"],
    })
    for col in platform_cols:
        _display[col] = _available[col].apply(_fmt_missing)
    _display["Avg ADP"] = _available["AvgADP"].apply(_fmt_missing)
    _display["Bye"] = _available["Bye"].apply(lambda b: b if b else "—")

    # Row order for THIS render — stored so a later Update click can map
    # edited_rows' integer indices back to a real player.
    st.session_state["fd_editor_row_ids"] = _available["PlayerID"].tolist()

    _col_config = {
        "Available": st.column_config.CheckboxColumn("AVAILABLE", width="small",
                        help="Uncheck drafted players, then click Update Draft Board."),
        "Rank":       st.column_config.NumberColumn("RANK", width="small"),
        "Player":     st.column_config.TextColumn("PLAYER", width="large"),
        "Pos":        st.column_config.TextColumn("POS", width="small"),
        "Avg ADP":    st.column_config.TextColumn("AVG ADP", width="small"),
        "Bye":        st.column_config.TextColumn("BYE", width="small"),
    }
    for col in platform_cols:
        _col_config[col] = st.column_config.TextColumn(col.upper(), width="small")

    st.data_editor(
        _display,
        key=_editor_key,
        use_container_width=True,
        hide_index=True,
        height=min(760, 46 + 35 * len(_display)),
        column_config=_col_config,
        disabled=[c for c in _display.columns if c != "Available"],
    )

    st.caption(f"ADP Sources: {' · '.join(platform_cols)}")
