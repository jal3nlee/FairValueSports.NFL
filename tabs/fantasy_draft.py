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


def _tight_label(text: str):
    st.markdown(
        f"<div style='font-size:0.78rem;opacity:0.65;margin:0 0 2px 0;line-height:1'>{text}</div>",
        unsafe_allow_html=True,
    )


def render():
    st.markdown("## Fantasy Draft")
    st.caption(
        "Compare fantasy football draft rankings and ADP across ESPN, Sleeper, CBS, NFL, RTSports, "
        "and Fantrax in one place."
    )

    st.session_state.setdefault("fd_drafted_ids", set())
    st.session_state.setdefault("fd_editor_version", 0)
    st.session_state.setdefault("fd_editor_row_ids", [])

    # ── Compact toolbar: Scoring · Position · View, one row ──────
    _c1, _c2, _c3 = st.columns([1.6, 3, 1.6])
    with _c1:
        _tight_label("Scoring")
        _scoring = st.segmented_control("Scoring", ["PPR", "Half PPR", "Standard"], default="PPR",
                                         key="fd_scoring", label_visibility="collapsed") or "PPR"
    with _c2:
        _tight_label("Position")
        _position = st.segmented_control("Position", POSITIONS, default="Overall",
                                          key="fd_position", label_visibility="collapsed") or "Overall"
    with _c3:
        _tight_label("View")
        _view = st.segmented_control("View", ["Consensus", "Best Available"], default="Consensus",
                                      key="fd_view", label_visibility="collapsed") or "Consensus"

    raw = load_fantasy_rankings(scoring=_scoring)
    if raw.empty:
        st.info("No fantasy rankings data found. Add the ADP file to the repo at `data/fantasy_adp.xlsx`.")
        return
    df, platform_cols = build_ranking_table(raw)
    if df.empty:
        st.warning("Rankings data couldn't be processed. Check the file formatting.")
        return

    _editor_key = f"fd_editor_{st.session_state.fd_editor_version}"

    def _apply_pending_drafts():
        _edits = st.session_state.get(_editor_key, {}).get("edited_rows", {})
        _row_ids = st.session_state.get("fd_editor_row_ids", [])
        for _idx, _change in _edits.items():
            if _idx >= len(_row_ids):
                continue
            _pid = _row_ids[_idx]
            if "Available" in _change:
                if _change["Available"] is False:
                    st.session_state.fd_drafted_ids.add(_pid)
                else:
                    st.session_state.fd_drafted_ids.discard(_pid)
        st.session_state.fd_editor_version += 1

    def _reset_board():
        st.session_state.fd_drafted_ids = set()
        st.session_state.fd_editor_version += 1

    _reset_col, _update_col, _spacer = st.columns([1.1, 1.3, 3])
    with _reset_col:
        if st.button("Reset Draft Board", use_container_width=True, key="fd_reset_btn"):
            _reset_board()
            st.rerun()
    with _update_col:
        if st.button("Update Draft Board", type="primary", use_container_width=True, key="fd_update_btn"):
            _apply_pending_drafts()
            st.rerun()

    # ── Base set depends on view — Consensus shows everyone (with drafted
    # state visible via the checkbox), Best Available filters drafted out
    # entirely. Neither ever renumbers ConsensusRank. ──────────────────
    if _view == "Best Available":
        _base = df[~df["PlayerID"].isin(st.session_state.fd_drafted_ids)].reset_index(drop=True)
    else:
        _base = df

    _filtered = filter_fantasy_rankings(_base, _position)

    if _filtered.empty:
        st.info("No players match the current filters, or all matching players have been drafted.")
        return

    def _fmt_player(row):
        return f"{row['Name']} — {row['Team']}" if row["Team"] else row["Name"]

    _display = pd.DataFrame({
        "Available": ~_filtered["PlayerID"].isin(st.session_state.fd_drafted_ids),
        "Rank": _filtered["ConsensusRank"],
        "Player": _filtered.apply(_fmt_player, axis=1),
        "Pos": _filtered["Position"],
    })
    for col in platform_cols:
        _display[col] = _filtered[col].apply(_fmt_missing)
    _display["Avg ADP"] = _filtered["AvgADP"].apply(_fmt_missing)
    _display["Bye"] = _filtered["Bye"].apply(lambda b: b if b else "—")

    st.session_state["fd_editor_row_ids"] = _filtered["PlayerID"].tolist()

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
