# tabs/lineup_comparison.py
import streamlit as st
import pandas as pd

from core.lineup_data import (
    espn_search_players,
    espn_team_roster,
    build_player_comparison,
    generate_comparison_notes,
    PROP_LABELS,
)

ROLES = ["Roster", "Bench", "Waiver"]


def _player_picker(slot_idx: int):
    st.markdown(f"**Player {slot_idx + 1}**")
    _q = st.text_input("Search player", key=f"lc_search_{slot_idx}", placeholder="Search player name...")
    _results = espn_search_players(_q) if _q else []
    if not _results:
        if _q:
            st.caption("No match found via search — try full name, or select manually below.")
        return None
    _labels = {f"{r['name']} ({r['team']})": r for r in _results}
    _picked = st.selectbox("Result", list(_labels.keys()), key=f"lc_pick_{slot_idx}", label_visibility="collapsed")
    return _labels[_picked]


def render(supabase, now_utc):
    st.markdown("## Lineup Comparison")
    st.caption(
        "Compare players side by side using real game odds, market player props, and matchup context — "
        "no proprietary score, just the information to make your own call."
    )

    _mode = st.segmented_control("Comparison Mode", ["Start/Sit", "FLEX", "Waiver Compare"], default="Start/Sit", key="lc_mode") or "Start/Sit"
    _scoring = st.segmented_control("Scoring", ["PPR", "Half PPR", "Standard"], default="PPR", key="lc_scoring") or "PPR"

    st.divider()

    _n_slots = st.session_state.get("lc_n_slots", 2)
    _slot_cols = st.columns(_n_slots)
    _players = []
    for i, col in enumerate(_slot_cols):
        with col:
            _p = _player_picker(i)
            if _p:
                _pos = st.selectbox("Position", ["QB", "RB", "WR", "TE"], key=f"lc_pos_{i}")
                _role = st.selectbox("Role", ROLES, key=f"lc_role_{i}")
                _team = _p["team"].split(" ")[-1] if _p.get("team") else ""
                _players.append({"name": _p["name"], "team": _team, "position": _pos, "role": _role})

    _bc1, _bc2 = st.columns(2)
    with _bc1:
        if st.button("+ Add Player", disabled=_n_slots >= 4, use_container_width=True):
            st.session_state["lc_n_slots"] = min(4, _n_slots + 1)
            st.rerun()
    with _bc2:
        if st.button("Remove Last", disabled=_n_slots <= 2, use_container_width=True):
            st.session_state["lc_n_slots"] = max(2, _n_slots - 1)
            st.rerun()

    st.divider()

    if len(_players) < 2:
        st.info(
            "**Compare your lineup options**\n\n"
            "Search for two or more players above to compare game environment, market props, "
            "and weekly context."
        )
        return

    with st.spinner("Loading market data..."):
        enriched = build_player_comparison(supabase, _players, now_utc)

    # ── Game Environment — real data ──────────────────────────
    st.markdown("### Game Environment")
    _env_rows = {"Metric": ["Opponent", "Spread", "Game Total", "Team Implied Total", "Home/Away"]}
    for p in enriched:
        ctx = p["context"]
        _env_rows[p["name"]] = [
            ctx.get("opponent", "—"),
            ctx.get("spread", "—"),
            ctx.get("game_total", "—"),
            ctx.get("team_implied_total", "—"),
            "Home" if ctx.get("is_home") else "Away" if ctx else "—",
        ]
    st.dataframe(pd.DataFrame(_env_rows), use_container_width=True, hide_index=True)

    # ── Betting Market (player props) — real data where available ──
    st.markdown("### Betting Market")
    st.caption("Consensus (median) line across books currently offering this prop.")
    _all_markets = sorted(set(m for p in enriched for m in p["props"].keys()))
    if not _all_markets:
        st.info("No player prop lines currently available for these players' games.")
    else:
        _prop_rows = {"Prop": [PROP_LABELS.get(m, m) for m in _all_markets]}
        for p in enriched:
            _prop_rows[p["name"]] = [
                p["props"].get(m, "—") if p["props"].get(m) is not None else "—" for m in _all_markets
            ]
        st.dataframe(pd.DataFrame(_prop_rows), use_container_width=True, hide_index=True)

    # ── Sections with no real data source yet — honest, not fabricated ──
    st.markdown("### Fantasy Outlook")
    st.info("Projected points, floor/ceiling, and positional rank require a fantasy projections provider — not yet connected.")

    st.markdown("### Usage / Opportunity")
    st.info("Snap share, targets, and red-zone usage require a play-by-play/usage data provider — not yet connected.")

    st.markdown("### Matchup")
    st.info("Opponent defensive rankings and points allowed by position require a stats provider — not yet connected.")

    # ── Comparison Notes — deterministic, no verdict ──────────
    st.markdown("### Comparison Notes")
    for note in generate_comparison_notes(enriched):
        st.markdown(f"- {note}")

    with st.expander("Roles"):
        for p in enriched:
            st.caption(f"**{p['name']}** — {p['role']}")
