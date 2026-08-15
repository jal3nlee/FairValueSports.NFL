# tabs/lineup_comparison.py
# User-facing name: "Lineup Analysis" — module filename kept as-is to
# avoid unnecessary import-risk across the app (per explicit instruction).
import streamlit as st
import pandas as pd

from core.lineup_data import (
    espn_search_players,
    get_players_by_team,
    get_players_by_position,
    build_player_comparison,
    PROP_LABELS,
    NFL_TEAMS,
    POSITIONS,
    FLEX_POSITIONS,
)
from core.nflverse_data import get_usage_samples, get_recent_games, LINEUP_USAGE_METRICS, METRIC_LABELS, PERCENT_METRICS
from core.nfl_defense_data import get_opponent_defense, POSITION_DEFENSE_METRICS

ROLES = ["Roster", "Bench", "Waiver"]
HEADSHOT_SIZE = 72  # single value within the requested 64–80px range — Streamlit has no real desktop/mobile breakpoint hook


def _dash(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return v


def _tight_label(text: str):
    st.markdown(
        f"<div style='font-size:0.78rem;opacity:0.65;margin:0 0 1px 0;line-height:1'>{text}</div>",
        unsafe_allow_html=True,
    )


def _section_heading(text: str):
    st.markdown(
        f"<div style='font-size:1.15rem;font-weight:700;margin:4px 0 6px 0'>{text}</div>",
        unsafe_allow_html=True,
    )


def _selected_names(exclude_idx: int, n_slots: int) -> set[str]:
    names = set()
    for i in range(n_slots):
        if i == exclude_idx:
            continue
        p = st.session_state.get(f"lc_selected_{i}")
        if p:
            names.add(p["name"])
    return names


def render_selected_player_header(p: dict, mode: str, slot_idx: int, n_slots: int):
    ctx = p.get("context", {})
    _photo_col, _info_col = st.columns([0.95, 3.05])
    with _photo_col:
        if p.get("headshot_url"):
            st.markdown(
                f"<img src='{p['headshot_url']}' style='width:{HEADSHOT_SIZE}px;height:{HEADSHOT_SIZE}px;"
                f"object-fit:contain;'/>",
                unsafe_allow_html=True,
            )
    with _info_col:
        st.markdown(
            f"<div style='font-weight:700;font-size:1rem;line-height:1.2'>{p['name']}</div>"
            f"<div style='opacity:0.65;font-size:0.85rem;line-height:1.3'>{p['position']} · {p['team']}</div>",
            unsafe_allow_html=True,
        )
        if ctx.get("opponent"):
            _side = "vs" if ctx.get("is_home") else "@"
            st.markdown(
                f"<div style='opacity:0.6;font-size:0.8rem;margin-top:1px'>{_side} {ctx['opponent']}</div>",
                unsafe_allow_html=True,
            )
        if mode == "Waiver":
            st.markdown(
                f"<div style='opacity:0.75;font-size:0.8rem;font-weight:600;margin-top:1px'>{p.get('role', 'Roster')}</div>",
                unsafe_allow_html=True,
            )
    _bc1, _bc2 = st.columns([1, 1])
    with _bc1:
        if st.button("Change", key=f"lc_change_{slot_idx}", use_container_width=True):
            st.session_state.pop(f"lc_selected_{slot_idx}", None)
            st.rerun()
    with _bc2:
        if slot_idx > 0 and n_slots > 1:
            if st.button("Remove", key=f"lc_removebtn_{slot_idx}", use_container_width=True):
                st.session_state.pop(f"lc_selected_{slot_idx}", None)
                st.session_state["lc_n_slots"] = n_slots - 1
                st.rerun()


def render_player_search(slot_idx: int, allowed_positions: list[str] | None, taken_names: set[str]):
    _q = st.text_input(
        "Search player name...", key=f"lc_search_{slot_idx}",
        placeholder="Search player name...", label_visibility="collapsed",
    )
    if not _q or len(_q.strip()) < 3:
        return None
    _results = espn_search_players(_q)
    _results = [r for r in _results if r["name"] not in taken_names]
    if allowed_positions:
        _results = [r for r in _results if not r.get("position") or r["position"] in allowed_positions]
    if not _results:
        st.caption("No match — try Browse Team instead.")
        return None
    _labels = {}
    for r in _results:
        _ctx = f"{r['position']}, {r['team']}" if r.get("position") else r["team"]
        _labels[f"{r['name']} — {_ctx}"] = r
    _picked = st.selectbox("Result", list(_labels.keys()), key=f"lc_pick_{slot_idx}", label_visibility="collapsed")
    r = _labels[_picked]
    return {"name": r["name"], "team": r["team"], "position": r.get("position") or "", "headshot_url": r.get("headshot_url")}


def render_team_position_search(slot_idx: int, allowed_positions: list[str] | None, taken_names: set[str]):
    _c1, _c2 = st.columns(2)
    with _c1:
        _team_name = st.selectbox("Team", sorted(NFL_TEAMS.keys()), key=f"lc_team_{slot_idx}", label_visibility="collapsed")
    _team_abbr = NFL_TEAMS.get(_team_name)
    _pos_options = ["All"] + (allowed_positions or POSITIONS)
    with _c2:
        _position = st.selectbox("Position", _pos_options, key=f"lc_teampos_{slot_idx}", label_visibility="collapsed")

    _roster = get_players_by_position(_team_abbr, _position) if _team_abbr else []
    _roster = [p for p in _roster if p["name"] not in taken_names]
    if not _roster:
        st.caption("No matching players for this team/position.")
        return None
    _options = {p["name"]: p for p in _roster if p.get("name")}
    _picked_name = st.selectbox("Player", sorted(_options.keys()), key=f"lc_teamplayer_{slot_idx}")
    p = _options[_picked_name]
    return {
        "name": p["name"], "team": _team_name,
        "position": p.get("position") or _position if _position != "All" else p.get("position", ""),
        "headshot_url": p.get("headshot_url"),
    }


def render_player_selector(slot_idx: int, mode: str, n_slots: int):
    _tight_label(f"Player {slot_idx + 1}")

    _confirmed = st.session_state.get(f"lc_selected_{slot_idx}")
    if _confirmed:
        render_selected_player_header(_confirmed, mode, slot_idx, n_slots)
        return _confirmed

    _allowed = FLEX_POSITIONS if mode == "FLEX" else None
    _taken = _selected_names(slot_idx, n_slots)

    _method = st.segmented_control(
        "Method", ["Search Player", "Browse Team"], default="Search Player",
        key=f"lc_method_{slot_idx}", label_visibility="collapsed",
    ) or "Search Player"

    if _method == "Search Player":
        _p = render_player_search(slot_idx, _allowed, _taken)
    else:
        _p = render_team_position_search(slot_idx, _allowed, _taken)

    if _p and _p["name"] not in _taken:
        if mode == "Waiver":
            _role = st.selectbox("Role", ROLES, key=f"lc_role_{slot_idx}")
            _p["role"] = _role
            if st.button("Confirm", key=f"lc_confirm_{slot_idx}", use_container_width=True):
                st.session_state[f"lc_selected_{slot_idx}"] = _p
                st.rerun()
        else:
            _p["role"] = "Roster"
            st.session_state[f"lc_selected_{slot_idx}"] = _p
            st.rerun()
    return None


def _fmt_val(v, is_pct):
    if v is None:
        return "—"
    return f"{v * 100:.0f}%" if is_pct else f"{v:.1f}"


def render_usage_role(enriched: list[dict], names: list[str]):
    _section_heading("Usage & Role")
    positions = {p["position"] for p in enriched}
    metrics = LINEUP_USAGE_METRICS.get(next(iter(positions)), []) if len(positions) == 1 else []
    if not metrics:
        if len(positions) > 1:
            st.caption("Select players at the same position to see matched usage metrics.")
        else:
            st.caption("Usage data temporarily unavailable.")
        return

    usage_by_player = {p["name"]: get_usage_samples(p["name"], p["team"], p["position"]) for p in enriched}
    if not any(usage_by_player.values()):
        st.caption("Usage data temporarily unavailable.")
        return

    if len(enriched) == 1:
        # Single player — wide Season | Last 5 | Last 3 table, per spec #20.
        p = enriched[0]
        u = usage_by_player.get(p["name"], {})
        _rows = []
        for m in metrics:
            is_pct = m in PERCENT_METRICS
            entry = u.get(m, {})
            row = {"Metric": METRIC_LABELS.get(m, m)}
            for wkey, wlabel_base, wreq in [("season", "Season", None), ("last5", "Last 5", 5), ("last3", "Last 3", 3)]:
                w = entry.get(wkey, {})
                games = w.get("games", 0)
                col_label = wlabel_base if (wreq is None or games >= wreq) else f"{wlabel_base} ({games})"
                row[col_label] = _fmt_val(w.get("value"), is_pct)
            _rows.append(row)
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
    else:
        # Multiple players — Option A: one row per metric per window, player columns.
        _rows = []
        for m in metrics:
            is_pct = m in PERCENT_METRICS
            label = METRIC_LABELS.get(m, m)
            for wkey, wlabel in [("season", "Season"), ("last5", "Last 5"), ("last3", "Last 3")]:
                row = [f"{label} — {wlabel}"]
                for p in enriched:
                    w = usage_by_player.get(p["name"], {}).get(m, {}).get(wkey, {})
                    row.append(_fmt_val(w.get("value"), is_pct))
                _rows.append(row)
        st.dataframe(pd.DataFrame(_rows, columns=["Metric"] + names), use_container_width=True, hide_index=True)
    st.caption("Usage data: nflverse — current season only.")


def render_recent_games(enriched: list[dict]):
    _section_heading("Recent Games")
    _any = False
    for p in enriched:
        games = get_recent_games(p["name"], p["team"], p["position"], n=5)
        if not games:
            continue
        _any = True
        if len(enriched) > 1:
            st.markdown(f"**{p['name']}**")
        st.dataframe(pd.DataFrame(games), use_container_width=True, hide_index=True)
    if not _any:
        st.caption("No current-season game log available yet.")


def render(supabase, now_utc):
    st.markdown("## Lineup Analysis")
    st.markdown(
        "<div style='opacity:0.7;font-size:0.95rem;margin:0 0 6px 0'>"
        "Research weekly usage, player props, game environment, and matchup context for your fantasy lineup decisions."
        "</div>",
        unsafe_allow_html=True,
    )

    _tc1, _tc2, _spacer = st.columns([1.5, 1.5, 3], gap="small")
    with _tc1:
        _tight_label("Mode")
        _mode = st.segmented_control("Mode", ["Start/Sit", "FLEX", "Waiver"], default="Start/Sit",
                                      key="lc_mode", label_visibility="collapsed") or "Start/Sit"
    with _tc2:
        _tight_label("Scoring")
        _scoring = st.segmented_control("Scoring", ["PPR", "Half PPR", "Standard"], default="PPR",
                                         key="lc_scoring", label_visibility="collapsed") or "PPR"

    st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)

    _n_slots = st.session_state.get("lc_n_slots", 1)  # defaults to ONE player, not two
    _show_vs = _n_slots == 2

    if _show_vs:
        _slot_cols = st.columns([5, 0.5, 5], gap="small")
        _player_cols = [_slot_cols[0], _slot_cols[2]]
        with _slot_cols[1]:
            st.markdown(
                "<div style='display:flex;align-items:center;justify-content:center;height:100%;"
                "opacity:0.4;font-size:0.8rem;font-weight:600;margin-top:34px'>VS</div>",
                unsafe_allow_html=True,
            )
    elif _n_slots == 1:
        _player_cols = [st.container()]
    else:
        _player_cols = st.columns(_n_slots, gap="small")

    _confirmed_players = []
    for i, col in enumerate(_player_cols):
        with col:
            _p = render_player_selector(i, _mode, _n_slots)
            if _p:
                _confirmed_players.append(_p)

    st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
    if _confirmed_players:
        _add_col, _ = st.columns([1.1, 4.9])
        with _add_col:
            if st.button("+ Add Player", disabled=_n_slots >= 4, key="lc_add"):
                st.session_state["lc_n_slots"] = min(4, _n_slots + 1)
                st.rerun()

    if not _confirmed_players:
        st.caption("Select a player to begin your lineup analysis.")
        return

    with st.spinner("Loading market data..."):
        enriched = build_player_comparison(supabase, _confirmed_players, now_utc)

    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

    def _row(label, values, higher_is_better=True):
        _real = [v for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))]
        best = max(_real) if higher_is_better and _real else (min(_real) if _real else None)
        cells = []
        for v in values:
            disp = _dash(v)
            if v is not None and best is not None and v == best and len(_real) > 1:
                cells.append(f"**{disp}**")
            else:
                cells.append(str(disp))
        return [label] + cells

    _names = [p["name"] for p in enriched]

    # ── Player Props ──────────────────────────────────
    _all_markets = sorted(set(m for p in enriched for m in p["props"].keys()))
    _has_any_prop_value = any(p["props"].get(m) is not None for p in enriched for m in _all_markets)
    _section_heading("Player Props")
    if not _all_markets or not _has_any_prop_value:
        st.caption("Player props are not available yet. Check back closer to kickoff.")
    else:
        if len(enriched) == 1:
            _rows = [{"Prop": PROP_LABELS.get(m, m), "Line": enriched[0]["props"].get(m, "—")} for m in _all_markets]
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
        else:
            _rows = [_row(PROP_LABELS.get(m, m), [p["props"].get(m) for p in enriched]) for m in _all_markets]
            st.dataframe(pd.DataFrame(_rows, columns=["Metric"] + _names), use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Usage & Role ────────────────────────────────────
    render_usage_role(enriched, _names)

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Recent Games ────────────────────────────────────
    render_recent_games(enriched)

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Game Environment ──────────────────────────────
    _has_env = any(p["context"] for p in enriched)
    if _has_env:
        _section_heading("Game Environment")
        if len(enriched) == 1:
            ctx = enriched[0]["context"]
            _env = {
                "Opponent": ctx.get("opponent", "—"), "Home/Away": "Home" if ctx.get("is_home") else "Away",
                "Spread": ctx.get("spread", "—"), "Game Total": ctx.get("game_total", "—"),
                "Team Implied Total": ctx.get("team_implied_total", "—"),
            }
            st.dataframe(pd.DataFrame([_env]), use_container_width=True, hide_index=True)
        else:
            _env_metrics = [
                ("Opponent", [p["context"].get("opponent") for p in enriched], None),
                ("Spread", [p["context"].get("spread") for p in enriched], False),
                ("Game Total", [p["context"].get("game_total") for p in enriched], True),
                ("Team Implied Total", [p["context"].get("team_implied_total") for p in enriched], True),
            ]
            _rows = []
            for label, vals, higher in _env_metrics:
                _rows.append(_row(label, vals, higher_is_better=higher) if higher is not None else [label] + [str(_dash(v)) for v in vals])
            st.dataframe(pd.DataFrame(_rows, columns=["Metric"] + _names), use_container_width=True, hide_index=True)
        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Opponent Defense ──────────────────────────────
    _section_heading("Opponent Defense")
    _def_by_player = {p["name"]: get_opponent_defense(p["context"].get("opponent"), p["position"], _scoring) for p in enriched}

    if all(v is None for v in _def_by_player.values()):
        _any_bye = any(not p["context"].get("opponent") for p in enriched)
        st.caption("No opponent this week (bye week)." if _any_bye else "Opponent defensive data is not available yet.")
    else:
        positions_in_view = {p["position"] for p in enriched}
        metric_set = POSITION_DEFENSE_METRICS.get(next(iter(positions_in_view)), []) if len(positions_in_view) == 1 else []
        if not metric_set:
            st.caption("Select players at the same position to see matched defensive context.")
        else:
            _headers = [f"{p['name']} vs {p['context'].get('opponent')}" if p["context"].get("opponent") else f"{p['name']} — Bye Week" for p in enriched]
            _rows = []
            for field, label in metric_set:
                row = [label]
                for p in enriched:
                    d = _def_by_player.get(p["name"])
                    row.append(_dash(d.get(field)) if d else "—")
                _rows.append(row)
            st.dataframe(pd.DataFrame(_rows, columns=["Metric"] + _headers), use_container_width=True, hide_index=True)
        st.caption("Defensive data: nflverse")
