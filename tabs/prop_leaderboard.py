# tabs/prop_leaderboard.py
import streamlit as st
import pandas as pd

from core.nflverse_data import (
    PROP_STAT_MAP, PROP_AVG_LABEL, SAMPLE_OPTIONS, build_prop_leaderboard,
)
from core.lineup_data import get_team_game_context, NFL_TEAMS

_ABBR_TO_FULLNAME = {v.upper(): k for k, v in NFL_TEAMS.items()}


def _opponent_for(team_abbr: str, now_utc):
    """Best-effort only — a failure here never breaks the main leaderboard."""
    try:
        full_name = _ABBR_TO_FULLNAME.get(team_abbr)
        if not full_name:
            return "—"
        ctx = get_team_game_context(st.session_state.get("_pl_supabase"), full_name, now_utc)
        return ctx.get("opponent", "—") if ctx else "—"
    except Exception:
        return "—"


def render(supabase, now_utc):
    st.markdown("## Prop Leaderboard")
    st.markdown(
        "<div style='opacity:0.7;font-size:0.95rem;margin:0 0 6px 0'>"
        "See which players have hit a selected prop threshold most often over recent games."
        "</div>",
        unsafe_allow_html=True,
    )

    _c1, _c2, _c3, _c4 = st.columns([1.8, 1.2, 1.2, 1.6])
    with _c1:
        stat_label = st.selectbox("Stat", list(PROP_STAT_MAP.keys()), key="pl_stat")
    with _c2:
        side = st.selectbox("Over/Under", ["Over", "Under"], key="pl_side")
    with _c3:
        line = st.number_input("Prop Line", min_value=0.0, value=49.5, step=0.5, key="pl_line")
    with _c4:
        sample_label = st.selectbox("Sample", list(SAMPLE_OPTIONS.keys()), index=1, key="pl_sample")

    _run = st.button("Find Top 10", type="primary", key="pl_run")

    st.markdown(f"### {side} {line:g} {stat_label}")
    st.caption(sample_label)

    if not _run:
        st.info("Set your filters above and click **Find Top 10** to run the search.")
        return

    st.session_state["_pl_supabase"] = supabase  # scoped stash for the opponent-lookup helper above

    with st.spinner("Scanning current-season player data..."):
        results = build_prop_leaderboard(stat_label, side, line, sample_label)

    if not results:
        st.info(
            "No players qualified with a full sample for this stat, line, and sample size. "
            "Try a shorter sample or a different threshold."
        )
        return

    avg_label = PROP_AVG_LABEL.get(stat_label, "Avg")
    rows = []
    for i, r in enumerate(results, 1):
        opp = _opponent_for(r["team"], now_utc)
        rows.append({
            "Rank": i,
            "Player": r["player"],
            "Team": r["team"],
            "Pos": r["position"],
            "Opp": opp,
            "Hit Rate": r["hit_rate"] / 100.0,
            "Record": f"{r['hits']} / {r['games']}",
            avg_label: r["avg"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hit Rate": st.column_config.ProgressColumn("Hit Rate", format="%.0f%%", min_value=0.0, max_value=1.0),
        },
    )
    if any(r["pushes"] > 0 for r in results):
        st.caption("Pushes (exact line matches) are excluded from both hits and the sample denominator.")
