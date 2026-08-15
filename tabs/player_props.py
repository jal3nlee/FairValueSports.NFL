# tabs/player_props.py
import streamlit as st
import pandas as pd

from core.lineup_data import (
    espn_search_players,
    get_players_by_team,
    get_players_by_position,
    get_team_game_context,
    fetch_player_props_for_event,
    get_consensus_prop_line,
    get_relevant_props_for_position,
    PROP_LABELS,
    NFL_TEAMS,
    POSITIONS,
)
from core.nflverse_data import (
    get_player_usage, get_player_game_log, calculate_hit_rate,
    POSITION_METRICS, METRIC_LABELS, PERCENT_METRICS,
)

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


def _player_picker():
    _method = st.segmented_control("Method", ["Search Player", "Browse Team"], default="Search Player",
                                    key="pp_method", label_visibility="collapsed") or "Search Player"
    if _method == "Search Player":
        _q = st.text_input("Search player name...", key="pp_search",
                            placeholder="Search player name...", label_visibility="collapsed")
        if not _q or len(_q.strip()) < 3:
            return None
        _results = espn_search_players(_q)
        if not _results:
            st.caption("No match — try Browse Team instead.")
            return None
        _labels = {f"{r['name']} — {r['position']}, {r['team']}": r for r in _results}
        _picked = st.selectbox("Result", list(_labels.keys()), label_visibility="collapsed")
        r = _labels[_picked]
        return {"name": r["name"], "team": r["team"], "position": r.get("position") or ""}
    else:
        _c1, _c2 = st.columns(2)
        with _c1:
            _team_name = st.selectbox("Team", sorted(NFL_TEAMS.keys()), key="pp_team", label_visibility="collapsed")
        _team_abbr = NFL_TEAMS.get(_team_name)
        with _c2:
            _position = st.selectbox("Position", POSITIONS, key="pp_pos", label_visibility="collapsed")
        _roster = get_players_by_position(_team_abbr, _position) if _team_abbr else []
        if not _roster:
            st.caption("No matching players.")
            return None
        _options = {p["name"]: p for p in _roster}
        _picked_name = st.selectbox("Player", sorted(_options.keys()), key="pp_player")
        p = _options[_picked_name]
        return {"name": p["name"], "team": _team_name, "position": p.get("position") or _position}


def render(supabase, now_utc):
    st.markdown("## Player Props")
    st.markdown(
        "<div style='opacity:0.7;font-size:0.95rem;margin:0 0 6px 0'>"
        "Research sportsbook player props against historical performance, current usage, and game context."
        "</div>",
        unsafe_allow_html=True,
    )

    player = _player_picker()
    if not player:
        st.caption("Search for or browse to a player to begin.")
        return

    st.markdown(f"### {player['name']} — {player['position']} · {player['team']}")

    _markets = get_relevant_props_for_position(player["position"])
    if not _markets:
        st.info("No supported prop markets for this position.")
        return
    _market_labels = {PROP_LABELS.get(m, m): m for m in _markets}
    _picked_label = st.selectbox("Market", list(_market_labels.keys()), key="pp_market")
    market_key = _market_labels[_picked_label]

    ctx = get_team_game_context(supabase, player["team"], now_utc)
    prop_rows = []
    consensus_line = None
    if ctx.get("event_id"):
        prop_rows = fetch_player_props_for_event(ctx["event_id"], player["position"])
        consensus_line = get_consensus_prop_line(prop_rows, player["name"], market_key)

    # ── Current Market ─────────────────────────────────
    st.markdown("### Current Market")
    if consensus_line is None:
        st.caption("Player props are not available yet. Check back closer to kickoff.")
    else:
        st.markdown(f"**{_picked_label} — {consensus_line:g}** (consensus)")
        _book_rows = {}
        for r in prop_rows:
            if r["player"].strip().lower() != player["name"].strip().lower() or r["market"] != market_key:
                continue
            b = r["book"]
            _book_rows.setdefault(b, {"Sportsbook": _sc_name(b), "Line": r.get("line"), "Over": None, "Under": None})
            if r.get("side") in ("Over", "Yes"):
                _book_rows[b]["Over"] = _fmt_odds(r.get("price"))
            elif r.get("side") in ("Under", "No"):
                _book_rows[b]["Under"] = _fmt_odds(r.get("price"))
        if _book_rows:
            _df = pd.DataFrame(list(_book_rows.values()))
            st.dataframe(_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No individual sportsbook prices available for this market yet.")

    # ── Historical Results ─────────────────────────────
    st.markdown("### Historical Results")
    st.caption(
        "How often this player has cleared today's market line in past games — "
        "context on recent performance, not a probability estimate for the next game."
    )
    if consensus_line is None:
        st.caption("No current line to check historical results against.")
    else:
        _side = st.segmented_control("Side", ["Over", "Under"], default="Over", key="pp_side", label_visibility="collapsed") or "Over"
        _full_log = get_player_game_log(player["name"], player["team"], market_key, n_games=None)
        if not _full_log:
            st.caption("No current-season game log available yet for this player.")
        else:
            _windows = [("Season", _full_log), ("Last 10", _full_log[:10]), ("Last 5", _full_log[:5])]
            _hc = st.columns(len(_windows))
            for col, (label, log) in zip(_hc, _windows):
                hr = calculate_hit_rate(log, consensus_line, _side)
                with col:
                    if hr:
                        st.metric(label, f"{hr['hits']} / {hr['total']} {_side}")
                    else:
                        st.metric(label, "—")

            st.markdown("#### Recent Games")
            _log_rows = [
                {
                    "Week": g["week"], "Opponent": g["opponent"], _picked_label: g["value"],
                    "Line": consensus_line,
                    "Result": ("Over" if g["value"] > consensus_line else "Under")
                              if _side == "Over" else ("Under" if g["value"] < consensus_line else "Over"),
                }
                for g in _full_log[:10]
            ]
            st.dataframe(pd.DataFrame(_log_rows), use_container_width=True, hide_index=True)

    # ── Usage & Role ────────────────────────────────────
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

    # ── Game Environment ─────────────────────────────────
    if ctx:
        st.markdown("### Game Environment")
        _env = {
            "Opponent": ctx.get("opponent", "—"),
            "Home/Away": "Home" if ctx.get("is_home") else "Away",
            "Spread": ctx.get("spread", "—"),
            "Game Total": ctx.get("game_total", "—"),
            "Team Implied Total": ctx.get("team_implied_total", "—"),
        }
        st.dataframe(pd.DataFrame([_env]), use_container_width=True, hide_index=True)
