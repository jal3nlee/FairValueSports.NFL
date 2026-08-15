# tabs/prop_leaderboard.py
import streamlit as st
import pandas as pd

from core.nfl_player_search import render_nfl_player_search
from core.nfl_player_card import render_nfl_player_card
from core.nflverse_data import (
    PROP_STAT_MAP, PROP_POSITION_MAP, PROP_AVG_LABEL, SAMPLE_OPTIONS,
    PLAYER_SEARCH_EXTRA_STATS, PROP_LABEL_TO_ODDS_MARKET,
    build_prop_leaderboard, get_player_game_log, calculate_hit_rate,
    get_player_usage, POSITION_METRICS, METRIC_LABELS, PERCENT_METRICS,
)
from core.nfl_defense_data import get_opponent_defense, POSITION_DEFENSE_METRICS
from core.lineup_data import (
    get_team_game_context, fetch_player_props_for_event, get_consensus_prop_line, NFL_TEAMS,
)

_ABBR_TO_FULLNAME = {v.upper(): k for k, v in NFL_TEAMS.items()}

SPORTSBOOK_DISPLAY = {
    "fanduel": "FanDuel", "draftkings": "DraftKings", "betmgm": "BetMGM",
    "caesars": "Caesars", "espnbet": "ESPN Bet", "fanatics": "Fanatics",
    "hardrockbet": "Hard Rock Bet", "betrivers": "BetRivers", "bovada": "Bovada",
}


def _sc_name(book: str) -> str:
    return SPORTSBOOK_DISPLAY.get(str(book).lower(), str(book).replace("_", " ").title())


def _fmt_odds(price):
    if price is None:
        return "—"
    return f"+{int(price)}" if price > 0 else str(int(price))


def _opponent_for(team_abbr: str, supabase, now_utc):
    try:
        full_name = _ABBR_TO_FULLNAME.get(team_abbr)
        if not full_name:
            return "—"
        ctx = get_team_game_context(supabase, full_name, now_utc)
        return ctx.get("opponent", "—") if ctx else "—"
    except Exception:
        return "—"


# =======================
# LEADERBOARD SUBVIEW — unchanged
# =======================
def render_leaderboard_view(supabase, now_utc):
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
        opp = _opponent_for(r["team"], supabase, now_utc)
        rows.append({
            "Rank": i, "Player": r["player"], "Team": r["team"], "Pos": r["position"], "Opp": opp,
            "Hit Rate": r["hit_rate"] / 100.0, "Record": f"{r['hits']} / {r['games']}", avg_label: r["avg"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={"Hit Rate": st.column_config.ProgressColumn("Hit Rate", format="%.0f%%", min_value=0.0, max_value=1.0)},
    )
    if any(r["pushes"] > 0 for r in results):
        st.caption("Pushes (exact line matches) are excluded from both hits and the sample denominator.")


# =======================
# PLAYER SEARCH SUBVIEW
# =======================
def render_player_search_view(supabase, now_utc):
    player = render_nfl_player_search("ps_slot", allowed_positions=["QB", "RB", "WR", "TE"])
    if not player:
        st.caption("No eligible players for this team/position.")
        return

    ctx = get_team_game_context(supabase, player["team"], now_utc)
    render_nfl_player_card(player, ctx, compact=True)

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    _available = [s for s, positions in PROP_POSITION_MAP.items() if player["position"] in positions]
    if player["position"] in ("WR", "TE", "RB"):
        _available = _available + list(PLAYER_SEARCH_EXTRA_STATS.keys())
    if not _available:
        st.info("No supported prop stats for this position.")
        return

    _sc1, _sc2, _sc3 = st.columns([1.35, 0.9, 0.75], gap="small")
    with _sc1:
        _picked_label = st.selectbox("Stat", _available, key="ps_stat_pick", label_visibility="collapsed")
    stat_field = PROP_STAT_MAP.get(_picked_label) or PLAYER_SEARCH_EXTRA_STATS.get(_picked_label)

    odds_market_key = PROP_LABEL_TO_ODDS_MARKET.get(_picked_label)
    prop_rows = []
    consensus_line = None
    if odds_market_key and ctx.get("event_id"):
        prop_rows = fetch_player_props_for_event(ctx["event_id"], player["position"])
        consensus_line = get_consensus_prop_line(prop_rows, player["name"], odds_market_key)

    with _sc2:
        _threshold = st.number_input(
            "Prop Line", min_value=0.0,
            value=float(consensus_line) if consensus_line is not None else 0.5,
            step=0.5, key="ps_threshold", label_visibility="collapsed",
        )
    with _sc3:
        _side = st.segmented_control("Side", ["Over", "Under"], default="Over",
                                      key="ps_side", label_visibility="collapsed") or "Over"

    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:1.05rem;font-weight:700;margin:0 0 2px 0'>Current Market</div>", unsafe_allow_html=True)
    if not odds_market_key:
        st.caption(f"{_picked_label} isn't tracked by sportsbooks — research the line above manually.")
    elif consensus_line is None:
        st.caption("Player props are not available yet. Check back closer to kickoff.")
    else:
        st.markdown(f"**{_picked_label} — {consensus_line:g}** (consensus)")
        _book_rows = {}
        for r in prop_rows:
            if r["player"].strip().lower() != player["name"].strip().lower() or r["market"] != odds_market_key:
                continue
            b = r["book"]
            _book_rows.setdefault(b, {"Sportsbook": _sc_name(b), "Line": r.get("line"), "Over": None, "Under": None})
            if r.get("side") in ("Over", "Yes"):
                _book_rows[b]["Over"] = _fmt_odds(r.get("price"))
            elif r.get("side") in ("Under", "No"):
                _book_rows[b]["Under"] = _fmt_odds(r.get("price"))
        if _book_rows:
            st.dataframe(pd.DataFrame(list(_book_rows.values())), use_container_width=True, hide_index=True)
        else:
            st.caption("No individual sportsbook prices available for this market yet.")

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:1.05rem;font-weight:700;margin:0 0 4px 0'>Historical Hit Rates</div>", unsafe_allow_html=True)
    _full_log = get_player_game_log(player["name"], player["team"], stat_field, n_games=None)
    if not _full_log:
        st.caption("No current-season game log available yet for this player.")
    else:
        _windows = [("Season", _full_log), ("Last 10", _full_log[:10]), ("Last 5", _full_log[:5])]
        _hc = st.columns(len(_windows), gap="small")
        for col, (label, log) in zip(_hc, _windows):
            hr = calculate_hit_rate(log, _threshold, _side)
            with col:
                st.metric(label, f"{hr['hits']} / {hr['total']} — {hr['hits']/hr['total']*100:.0f}%" if hr else "—")

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        st.markdown("**Recent Results**")
        st.caption("Line = your selected research line, not a historical sportsbook line.")
        _log_rows = [
            {
                "Week": g["week"], "Opponent": g["opponent"], _picked_label: g["value"], "Line": _threshold,
                "Result": ("Over" if g["value"] > _threshold else "Push" if g["value"] == _threshold else "Under"),
            }
            for g in _full_log[:10]
        ]
        st.dataframe(pd.DataFrame(_log_rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    st.markdown("### Usage & Role")
    usage = get_player_usage(player["name"], player["team"], player["position"])
    metrics = POSITION_METRICS.get(player["position"], [])
    if not usage or not metrics:
        st.caption("Usage data temporarily unavailable.")
    else:
        _rows = []
        for m in metrics:
            label = METRIC_LABELS.get(m, m)
            is_pct = m in PERCENT_METRICS
            u = usage.get(m, {})
            sv, cv = u.get("season"), u.get("current_role")
            _rows.append({
                "Metric": label,
                "Season": (f"{sv * 100:.0f}%" if is_pct and sv is not None else f"{sv:.1f}" if sv is not None else "—"),
                "Current Role": (f"{cv * 100:.0f}%" if is_pct and cv is not None else f"{cv:.1f}" if cv is not None else "—"),
            })
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
        st.caption("Usage data: nflverse")

    if ctx:
        st.markdown("### Game Environment")
        _env = {
            "Opponent": ctx.get("opponent", "—"), "Home/Away": "Home" if ctx.get("is_home") else "Away",
            "Spread": ctx.get("spread", "—"), "Game Total": ctx.get("game_total", "—"),
            "Team Implied Total": ctx.get("team_implied_total", "—"),
        }
        st.dataframe(pd.DataFrame([_env]), use_container_width=True, hide_index=True)

    st.markdown("### Opponent Defense")
    _def = get_opponent_defense(ctx.get("opponent"), player["position"], "PPR")
    _metric_set = POSITION_DEFENSE_METRICS.get(player["position"], [])
    if not ctx.get("opponent"):
        st.caption("No opponent this week (bye week).")
    elif not _def or not _metric_set:
        st.caption("Opponent defensive data is not available yet.")
    else:
        _rows = [{"Metric": label, f"vs {ctx.get('opponent')}": _def.get(field, "—")} for field, label in _metric_set]
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
        st.caption("Defensive data: nflverse")


def render(supabase, now_utc):
    st.markdown("## Prop Leaderboard")
    st.markdown(
        "<div style='opacity:0.7;font-size:0.95rem;margin:0 0 6px 0'>"
        "Research NFL player props by historical hit rates or by individual player."
        "</div>",
        unsafe_allow_html=True,
    )

    _view = st.segmented_control("View", ["Leaderboard", "Player Search"], default="Leaderboard",
                                  key="pl_view", label_visibility="collapsed") or "Leaderboard"

    if _view == "Leaderboard":
        render_leaderboard_view(supabase, now_utc)
    else:
        render_player_search_view(supabase, now_utc)
