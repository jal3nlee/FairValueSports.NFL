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
TREND_THRESHOLD = 0.15  # 15% change minimum before showing an arrow


def _dash(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return v


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
    if st.button("Change Player", key=f"lc_change_{slot_idx}", use_container_width=True):
        st.session_state.pop(f"lc_selected_{slot_idx}", None)
        st.rerun()


def render_player_search(slot_idx: int, allowed_positions: list[str] | None, taken_names: set[str]):
    _q = st.text_input(
        "Search player name...", key=f"lc_search_{slot_idx}",
        placeholder="Search player name...", label_visibility="collapsed",
    )
    if not _q:
        return None
    _results = espn_search_players(_q)
    _results = [r for r in _results if r["name"] not in taken_names]
    if allowed_positions:
        _results = [r for r in _results if not r.get("position") or r["position"] in allowed_positions]
    if not _results:
        st.caption("No match found — try the full name, or switch to Browse Team.")
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
        st.caption("No matching players found for this team/position.")
        return None
    _options = {p["name"]: p for p in _roster if p.get("name")}
    _picked_name = st.selectbox("Player", sorted(_options.keys()), key=f"lc_teamplayer_{slot_idx}")
    p = _options[_picked_name]
    return {"name": p["name"], "team": _team_name, "position": p.get("position") or _position if _position != "All" else p.get("position", "")}


def render_player_selector(slot_idx: int, mode: str, n_slots: int):
    st.markdown(f"**Player {slot_idx + 1}**")

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


def _trend_str(season, current):
    if season is None or current is None or season == 0:
        return ""
    change = (current - season) / season
    if abs(change) < TREND_THRESHOLD:
        return ""
    return " ↑" if change > 0 else " ↓"


def _fmt_usage_val(v, is_percent):
    if v is None:
        return "—"
    return f"{v * 100:.0f}%" if is_percent else f"{v:.1f}"


def render_usage_and_role(enriched_players: list[dict]):
    st.markdown("### Usage & Role")

    positions = {p["position"] for p in enriched_players}
    if len(positions) == 1:
        metrics = POSITION_METRICS.get(next(iter(positions)), [])
    else:
        metrics = ["targets_per_game"]  # cross-position FLEX comparisons kept minimal

    usage_by_player = {}
    for p in enriched_players:
        usage_by_player[p["name"]] = get_player_usage(p["name"], p["team"], p["position"])

    if not any(usage_by_player.values()):
        st.info("Usage data temporarily unavailable.")
        return {}

    rows = []
    for m in metrics:
        label = METRIC_LABELS.get(m, m)
        is_pct = m in PERCENT_METRICS
        season_row = [label + " (Season)"]
        role_row = [label + " (Current Role)"]
        for p in enriched_players:
            u = usage_by_player.get(p["name"], {}).get(m, {})
            season_v, current_v = u.get("season"), u.get("current_role")
            season_row.append(_fmt_usage_val(season_v, is_pct))
            role_row.append(_fmt_usage_val(current_v, is_pct) + _trend_str(season_v, current_v))
        rows.append(season_row)
        rows.append(role_row)

    cols = ["Metric"] + [p["name"] for p in enriched_players]
    st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True, hide_index=True)
    st.caption("Usage data: nflverse")
    return usage_by_player


def render(supabase, now_utc):
    st.markdown("## Lineup Comparison")
    st.caption("Compare fantasy projections, player props, game environment, and matchup context side by side.")

    _tc1, _tc2 = st.columns([2, 1.4])
    with _tc1:
        _mode = st.segmented_control("Mode", ["Start/Sit", "FLEX", "Waiver"], default="Start/Sit", key="lc_mode") or "Start/Sit"
    with _tc2:
        _scoring = st.segmented_control("Scoring", ["PPR", "Half PPR", "Standard"], default="PPR", key="lc_scoring") or "PPR"

    _n_slots = st.session_state.get("lc_n_slots", 2)
    _slot_cols = st.columns(_n_slots, gap="medium")

    _confirmed_players = []
    for i, col in enumerate(_slot_cols):
        with col:
            _p = render_player_selector(i, _mode, _n_slots)
            if i == _n_slots - 1 and _n_slots > 2:
                if st.button("Remove", key=f"lc_remove_{i}", use_container_width=True):
                    st.session_state.pop(f"lc_selected_{i}", None)
                    st.session_state["lc_n_slots"] = _n_slots - 1
                    st.rerun()
            if _p:
                _confirmed_players.append(_p)

    if st.button("+ Add Player", disabled=_n_slots >= 4, key="lc_add"):
        st.session_state["lc_n_slots"] = min(4, _n_slots + 1)
        st.rerun()

    if len(_confirmed_players) < 2:
        st.caption("Select at least two players to begin comparing.")
        return

    with st.spinner("Loading market data..."):
        enriched = build_player_comparison(supabase, _confirmed_players, now_utc)

    st.markdown("---")

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
        _rows = [_row(PROP_LABELS.get(m, m), [p["props"].get(m) for m in enriched]) for m in _all_markets]
        st.dataframe(pd.DataFrame(_rows, columns=["Metric"] + _names), use_container_width=True, hide_index=True)

    # ── Usage & Role (nflverse) ────────────────────────
    usage_by_player = render_usage_and_role(enriched)

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
            season_v, current_v = meta.get("season"), meta.get("current_role")
            if season_v and current_v and abs((current_v - season_v) / season_v) >= TREND_THRESHOLD:
                direction = "risen" if current_v > season_v else "declined"
                label = METRIC_LABELS.get(metric_key, metric_key).lower()
                notes.append(f"{p['name']}'s {label} has {direction} recently ({season_v} → {current_v}).")
    for note in notes[:6]:
        st.markdown(f"- {note}")
