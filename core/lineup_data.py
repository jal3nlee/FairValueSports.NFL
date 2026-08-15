# tabs/lineup_comparison.py
import streamlit as st
import pandas as pd

from core.lineup_data import (
    espn_search_players,
    get_players_by_team,
    get_players_by_position,
    build_player_comparison,
    generate_key_differences,
    PROP_LABELS,
    NFL_TEAMS,
    POSITIONS,
    FLEX_POSITIONS,
)
from core.nflverse_data import get_player_usage, POSITION_METRICS, METRIC_LABELS, PERCENT_METRICS

ROLES = ["Roster", "Bench", "Waiver"]
TREND_THRESHOLD = 0.15


def _dash(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return v


def _tight_label(text: str):
    st.markdown(
        f"<div style='font-size:0.78rem;opacity:0.65;margin:0 0 1px 0;line-height:1'>{text}</div>",
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


def render_selected_player_header(p: dict, mode: str, slot_idx: int):
    ctx = p.get("context", {})
    st.markdown(f"**{p['name']}**")
    st.caption(f"{p['position']} · {p['team']}")
    if ctx.get("opponent"):
        _side = "vs" if ctx.get("is_home") else "@"
        st.caption(f"{_side} {ctx['opponent']}")
    if mode == "Waiver":
        st.caption(f"**{p.get('role', 'Roster')}**")
    if st.button("Change", key=f"lc_change_{slot_idx}", use_container_width=True):
        st.session_state.pop(f"lc_selected_{slot_idx}", None)
        st.rerun()


def render_player_search(slot_idx: int, allowed_positions: list[str] | None, taken_names: set[str]):
    _q = st.text_input(
        "Search player name...", key=f"lc_search_{slot_idx}",
        placeholder="Search player name...", label_visibility="collapsed",
    )
    # Don't show "no match" while the user is still typing a short prefix —
    # give the search a real chance before declaring failure.
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
    _team_abbr = r["team"].split(" ")[-1] if r.get("team") else ""
    return {"name": r["name"], "team": _team_abbr, "position": r.get("position") or ""}


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
    return {"name": p["name"], "team": _team_name, "position": p.get("position") or _position if _position != "All" else p.get("position", "")}


def render_player_selector(slot_idx: int, mode: str, n_slots: int):
    _tight_label(f"Player {slot_idx + 1}")

    _confirmed = st.session_state.get(f"lc_selected_{slot_idx}")
    if _confirmed:
        render_selected_player_header(_confirmed, mode, slot_idx)
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
        _role = "Roster"
        if mode == "Waiver":
            _role = st.selectbox("Role", ROLES, key=f"lc_role_{slot_idx}")
        _p["role"] = _role
        if st.button("Confirm", key=f"lc_confirm_{slot_idx}", use_container_width=True):
            st.session_state[f"lc_selected_{slot_idx}"] = _p
            st.rerun()
    return None


def render(supabase, now_utc):
    st.markdown("## Lineup Comparison")
    st.markdown(
        "<div style='opacity:0.7;font-size:0.95rem;margin:0 0 10px 0'>"
        "Compare fantasy projections, player props, game environment, and matchup context side by side."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Mode + Scoring, tight, left-aligned, not full width ──
    _tc1, _tc2, _spacer = st.columns([1.6, 1.6, 2.8], gap="small")
    with _tc1:
        _tight_label("Mode")
        _mode = st.segmented_control("Mode", ["Start/Sit", "FLEX", "Waiver"], default="Start/Sit",
                                      key="lc_mode", label_visibility="collapsed") or "Start/Sit"
    with _tc2:
        _tight_label("Scoring")
        _scoring = st.segmented_control("Scoring", ["PPR", "Half PPR", "Standard"], default="PPR",
                                         key="lc_scoring", label_visibility="collapsed") or "PPR"

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    # ── Player grid — VS only for exactly 2 players ──────
    _n_slots = st.session_state.get("lc_n_slots", 2)
    _show_vs = _n_slots == 2

    if _show_vs:
        _slot_cols = st.columns([5, 0.6, 5], gap="small")
        _player_cols = [_slot_cols[0], _slot_cols[2]]
        with _slot_cols[1]:
            st.markdown(
                "<div style='display:flex;align-items:center;justify-content:center;height:100%;"
                "opacity:0.4;font-size:0.85rem;font-weight:600;margin-top:40px'>VS</div>",
                unsafe_allow_html=True,
            )
    else:
        _player_cols = st.columns(_n_slots, gap="small")

    _confirmed_players = []
    for i, col in enumerate(_player_cols):
        with col:
            _p = render_player_selector(i, _mode, _n_slots)
            if i == _n_slots - 1 and _n_slots > 2:
                if st.button("Remove", key=f"lc_remove_{i}", use_container_width=True):
                    st.session_state.pop(f"lc_selected_{i}", None)
                    st.session_state["lc_n_slots"] = _n_slots - 1
                    st.rerun()
            if _p:
                _confirmed_players.append(_p)

    # ── Shared Add Player, compact, below the whole grid ──
    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
    _add_col, _ = st.columns([1.3, 4.7])
    with _add_col:
        if st.button("+ Add Player", disabled=_n_slots >= 4, key="lc_add", use_container_width=True):
            st.session_state["lc_n_slots"] = min(4, _n_slots + 1)
            st.rerun()

    if len(_confirmed_players) < 2:
        st.caption("Select at least two players to begin comparing.")
        return

    with st.spinner("Loading market data..."):
        enriched = build_player_comparison(supabase, _confirmed_players, now_utc)

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

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
    if _all_markets:
        st.markdown("### Player Props")
        _rows = [_row(PROP_LABELS.get(m, m), [p["props"].get(m) for p in enriched]) for m in _all_markets]
        st.dataframe(pd.DataFrame(_rows, columns=["Metric"] + _names), use_container_width=True, hide_index=True)

    # ── Usage & Role (nflverse) ────────────────────────
    st.markdown("### Usage & Role")
    positions = {p["position"] for p in enriched}
    metrics = POSITION_METRICS.get(next(iter(positions)), []) if len(positions) == 1 else ["targets_per_game"]
    usage_by_player = {p["name"]: get_player_usage(p["name"], p["team"], p["position"]) for p in enriched}
    if not any(usage_by_player.values()):
        st.info("Usage data temporarily unavailable.")
    else:
        _rows = []
        for m in metrics:
            label = METRIC_LABELS.get(m, m)
            is_pct = m in PERCENT_METRICS
            season_row = [label + " (Season)"]
            role_row = [label + " (Current Role)"]
            for p in enriched:
                u = usage_by_player.get(p["name"], {}).get(m, {})
                sv, cv = u.get("season"), u.get("current_role")
                season_row.append(f"{sv * 100:.0f}%" if (is_pct and sv is not None) else (f"{sv:.1f}" if sv is not None else "—"))
                cv_str = f"{cv * 100:.0f}%" if (is_pct and cv is not None) else (f"{cv:.1f}" if cv is not None else "—")
                if sv and cv and abs((cv - sv) / sv) >= TREND_THRESHOLD:
                    cv_str += " ↑" if cv > sv else " ↓"
                role_row.append(cv_str)
            _rows.append(season_row)
            _rows.append(role_row)
        st.dataframe(pd.DataFrame(_rows, columns=["Metric"] + _names), use_container_width=True, hide_index=True)
        st.caption("Usage data: nflverse")

    # ── Game Environment ──────────────────────────────
    _has_env = any(p["context"] for p in enriched)
    if _has_env:
        st.markdown("### Game Environment")
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

    # ── Key Differences ───────────────────────────────
    st.markdown("### Key Differences")
    notes = generate_key_differences(enriched)
    for p in enriched:
        u = usage_by_player.get(p["name"], {})
        for metric_key, meta in u.items():
            if not isinstance(meta, dict):
                continue
            sv, cv = meta.get("season"), meta.get("current_role")
            if sv and cv and abs((cv - sv) / sv) >= TREND_THRESHOLD:
                direction = "risen" if cv > sv else "declined"
                label = METRIC_LABELS.get(metric_key, metric_key).lower()
                notes.append(f"{p['name']}'s {label} has {direction} recently ({sv} → {cv}).")
    for note in notes[:6]:
        st.markdown(f"- {note}")
