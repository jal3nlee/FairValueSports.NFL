# tabs/fair_value_model.py
import pandas as pd
import streamlit as st

from core.odds_math import (
    american_to_decimal, american_to_implied_prob, expected_value_pct, fmt_odds, fmt_ev,
)
from core.pipeline import MARKETS, run_market_pipeline, PipelineTrace
from core.data_sources import fetch_market_lines, filter_by_window, get_date_window
from tabs import market_movers


def _sc_name(book: str) -> str:
    _display = {
        "eu_pinnacle": "Pinnacle", "pinnacle": "Pinnacle", "fanduel": "FanDuel",
        "draftkings": "DraftKings", "caesars": "Caesars", "bet365": "bet365", "kalshi": "Kalshi",
        "betmgm": "BetMGM", "espnbet": "ESPN Bet", "fanatics": "Fanatics",
        "hardrockbet": "Hard Rock Bet", "betrivers": "BetRivers", "ballybet": "Bally Bet",
        "betparx": "betParx", "betonline": "BetOnline", "lowvig": "LowVig", "fliff": "Fliff",
        "rebet": "Rebet", "betanysports": "BetAnySports", "bovada": "Bovada",
        "mybookie": "MyBookie", "betus": "BetUS", "underdog": "Underdog", "prophetx": "ProphetX",
    }
    return _display.get(str(book).lower(), str(book).replace("_", " ").title())


TIPS = {
    "ev":         "Expected Value — your edge vs. the fair price. Positive = favorable bet.",
    "fair_win":   "Consensus fair probability — the no-vig estimate of this outcome's true likelihood.",
    "fair_odds":  "Fair Value odds — the American odds equivalent of the consensus fair probability.",
    "best_odds":  "Best available odds — the highest American odds found across all sportsbooks.",
}


def _odds_value_in_format(american_price, fmt: str):
    if american_price is None or pd.isna(american_price):
        return None
    if fmt == "American":
        return float(american_price)
    elif fmt == "Decimal":
        return american_to_decimal(int(american_price))
    else:
        p = american_to_implied_prob(int(american_price))
        return round(p * 100, 1) if p else None


def _uncheck_no_min_odds(checkbox_key: str) -> None:
    """on_change callback for the Minimum Odds number_input — editing the
    value is itself the user's signal that they want the filter active, so
    flip the paired "No minimum" checkbox off on that same rerun."""
    st.session_state[checkbox_key] = False


def _fmt_best_odds(american_price, fmt: str) -> str:
    if american_price is None or pd.isna(american_price):
        return "—"
    price = int(american_price)
    if fmt == "American":
        return fmt_odds(price)
    elif fmt == "Decimal":
        return f"{american_to_decimal(price):.2f}x"
    else:
        p = american_to_implied_prob(price)
        return f"{p * 100:.1f}%" if p else "—"


# Same team-abbreviation map already used by Sportsbook Screener / Matchup
# Center / Parlay Builder / Market Movers — reused as-is for the compact
# "Bet" column below rather than introducing a second mapping.
NFL_TEAM_ABBR = {
    "Arizona Cardinals": "ari", "Atlanta Falcons": "atl", "Baltimore Ravens": "bal",
    "Buffalo Bills": "buf", "Carolina Panthers": "car", "Chicago Bears": "chi",
    "Cincinnati Bengals": "cin", "Cleveland Browns": "cle", "Dallas Cowboys": "dal",
    "Denver Broncos": "den", "Detroit Lions": "det", "Green Bay Packers": "gb",
    "Houston Texans": "hou", "Indianapolis Colts": "ind", "Jacksonville Jaguars": "jax",
    "Kansas City Chiefs": "kc", "Las Vegas Raiders": "lv", "Los Angeles Chargers": "lac",
    "Los Angeles Rams": "lar", "Miami Dolphins": "mia", "Minnesota Vikings": "min",
    "New England Patriots": "ne", "New Orleans Saints": "no", "New York Giants": "nyg",
    "New York Jets": "nyj", "Philadelphia Eagles": "phi", "Pittsburgh Steelers": "pit",
    "San Francisco 49ers": "sf", "Seattle Seahawks": "sea", "Tampa Bay Buccaneers": "tb",
    "Tennessee Titans": "ten", "Washington Commanders": "wsh",
}


def _abbr(team_name) -> str:
    mapped = NFL_TEAM_ABBR.get(team_name)
    return mapped.upper() if mapped else (team_name or "—")


def _bet_compact_label(row) -> str:
    """One mobile-friendly string identifying the exact bet, e.g.
    'GB ML @ DEN', 'MIA +2.5 @ NYG', or 'Over 36.5 · CIN @ CHI'."""
    game = row.get("Game", "")
    teams = game.split(" vs ") if isinstance(game, str) else []
    home_team = teams[0] if len(teams) == 2 else None
    away_team = teams[1] if len(teams) == 2 else None
    market = row.get("Market", "")
    pick = row.get("Pick", "—")
    line = row.get("Line")
    has_line = pd.notna(line) and str(line).strip()

    if market == "Total":
        line_part = f" {line}" if has_line else ""
        return f"{pick}{line_part} · {_abbr(away_team)} @ {_abbr(home_team)}"

    opponent = away_team if pick == home_team else home_team
    suffix = "ML" if market == "Moneyline" else (str(line) if has_line else "")
    return f"{_abbr(pick)} {suffix} @ {_abbr(opponent)}".replace("  ", " ").strip()


def render(supabase, now_utc, eff_bankroll, eff_kelly, authed, debug_mode=False):
    market_movers.render(supabase, now_utc, eff_bankroll, eff_kelly)

    if not authed:
        st.info("Sign in to access the Fair Value Model.")
        return

    st.markdown("## Fair Value Model")
    st.caption(
        "Compare our fair odds estimates to the best available sportsbook prices "
        "to identify positive expected value betting opportunities."
    )

    _format_options = ["American", "Decimal", "Implied %"]
    _current_week_label = "NFL Week X"  # replaced below once window is known

    # ── Row 1: Date Range · Market · Odds Format ────────────
    _fr1, _fr2, _fr3 = st.columns(3)
    with _fr1:
        _now_local_week = None
        from core.data_sources import infer_current_week_index
        _wk = infer_current_week_index(now_utc)
        _week_label = "NFL Preseason" if _wk == 0 else f"NFL Week {_wk}"
        _window_choice = st.selectbox(
            "Date Range", ["Today", _week_label, "Next 7 Days"], index=1, key="fvm_window_choice",
        )
    window_start, window_end, sport_keys, caption_label = get_date_window(now_utc, _window_choice)

    with _fr3:
        _odds_format = st.selectbox("Odds Format", _format_options, key="fvm_odds_format")

    # ── Load all three markets ───────────────────────────────
    all_raw: dict[str, pd.DataFrame] = {}
    all_display: dict[str, pd.DataFrame] = {}
    all_traces: dict[str, PipelineTrace] = {}
    pulled_times: list = []

    with st.spinner("Loading NFL odds..."):
        for mkt_key, mkt_cfg in MARKETS.items():
            trace = PipelineTrace(market=mkt_cfg.name)
            all_traces[mkt_key] = trace
            raw, pulled = fetch_market_lines(supabase, sport_keys, mkt_cfg.db_market_key)
            trace.raw = len(raw)
            raw_filtered = filter_by_window(raw, window_start, window_end)
            trace.after_window = len(raw_filtered)
            all_raw[mkt_key] = raw_filtered
            pulled_times.extend(pulled)
            df = run_market_pipeline(
                raw_lines=raw_filtered, cfg=mkt_cfg, bankroll=eff_bankroll, kelly=eff_kelly,
                min_ev=0.0, min_fair_pct=0.0, show_all=True, trace=trace,
            )
            all_display[mkt_key] = df

    df_all = (
        pd.concat([df for df in all_display.values() if not df.empty], ignore_index=True)
        if any(not df.empty for df in all_display.values())
        else pd.DataFrame()
    )

    if debug_mode:
        for _mkt_key, _trace in all_traces.items():
            st.caption(
                f"debug [{_mkt_key}]: raw={_trace.raw} after_window={_trace.after_window} "
                f"after_build={_trace.after_build} after_consensus={_trace.after_consensus} "
                f"after_best={_trace.after_best} after_merge={_trace.after_merge} "
                f"after_mi={_trace.after_mi} after_display={_trace.after_display} "
                f"after_filter={_trace.after_filter} | error={_trace.error or 'none'}"
            )
        _raw_game_count = 0
        for _raw in all_raw.values():
            if not _raw.empty:
                _raw_game_count = max(
                    _raw_game_count,
                    _raw[["home_team", "away_team"]].drop_duplicates().shape[0],
                )
        st.caption(f"debug: caption_label={caption_label} | unique games in raw data: {_raw_game_count}")

    with _fr2:
        _mkt_opts = ["All"] + (sorted(df_all["Market"].dropna().unique().tolist()) if not df_all.empty else [])
        _mkt_sel = st.selectbox("Market", _mkt_opts, key="fvm_mkt")

    # ── Row 2: Minimum Expected Value · Minimum Odds ────────
    _sr1, _sr2 = st.columns(2)
    with _sr1:
        _min_ev_sel = st.number_input(
            "Minimum Expected Value (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.5,
            key="fvm_min_ev",
        )
    with _sr2:
        # Bounds/steps match the old select_slider exactly per format — only
        # the control changed. Widget keys are format-specific so switching
        # Odds Format always starts fresh rather than carrying over a
        # numeric value (or a stale "No minimum" state) in the wrong units.
        if _odds_format == "American":
            _odds_fmt_key = "american"
            _min_odds_kwargs = dict(min_value=-500, max_value=500, value=-110, step=5)
        elif _odds_format == "Decimal":
            _odds_fmt_key = "decimal"
            _min_odds_kwargs = dict(min_value=0.0, max_value=10.0, value=1.91, step=0.05, format="%.2f")
        else:
            _odds_fmt_key = "implied"
            _min_odds_kwargs = dict(min_value=0.0, max_value=100.0, value=52.4, step=0.1, format="%.1f")

        _no_min_odds_key = f"fvm_min_odds_no_min_{_odds_fmt_key}"
        # Always visible, never disabled — editing it is handled by the
        # on_change callback below rather than by hiding/disabling the input.
        _min_odds_raw = st.number_input(
            "Minimum Odds", key=f"fvm_min_odds_val_{_odds_fmt_key}",
            on_change=_uncheck_no_min_odds, args=(_no_min_odds_key,),
            **_min_odds_kwargs,
        )

    # ── Sportsbook filter ────────────────────────────────────
    _all_books_raw = sorted(set(
        b for raw in all_raw.values() if not raw.empty for b in raw["book"].dropna().unique()
    ))
    _book_display_map = {b: _sc_name(b) for b in _all_books_raw}
    _book_disp_to_keys: dict[str, set[str]] = {}
    for _raw_book, _disp in _book_display_map.items():
        _book_disp_to_keys.setdefault(_disp, set()).add(_raw_book.lower())
    _label_to_cfg = {c.market_label: c for c in MARKETS.values()}

    st.markdown("Sportsbooks")
    _all_books_disp = sorted(_book_disp_to_keys.keys())
    _selected_count = sum(1 for _b in _all_books_disp if st.session_state.get(f"fvm_book_{_b}", True))
    with st.popover(f"{_selected_count} of {len(_all_books_disp)} selected", use_container_width=True):
        _sa, _sb = st.columns(2)
        if _sa.button("Select all", use_container_width=True, key="fvm_books_all"):
            for _b in _all_books_disp:
                st.session_state[f"fvm_book_{_b}"] = True
            st.rerun()
        if _sb.button("Clear all", use_container_width=True, key="fvm_books_none"):
            for _b in _all_books_disp:
                st.session_state[f"fvm_book_{_b}"] = False
            st.rerun()
        st.divider()
        _book_sel_disp = []
        for _b in _all_books_disp:
            _checked = st.checkbox(_b, value=st.session_state.get(f"fvm_book_{_b}", True), key=f"fvm_book_{_b}")
            if _checked:
                _book_sel_disp.append(_b)
    _book_sel_keys: set[str] = set()
    for d in _book_sel_disp:
        _book_sel_keys |= _book_disp_to_keys.get(d, set())

    _show_col, _no_min_col = st.columns(2)
    with _show_col:
        _show_all_bets = st.checkbox("Show all bets (include negative EV)", value=False, key="fvm_show_all")
    with _no_min_col:
        _no_min_odds = st.checkbox("No minimum", value=True, key=_no_min_odds_key)
    _min_odds_sel = None if _no_min_odds else float(_min_odds_raw)
    st.divider()

    if df_all.empty:
        st.info(f"No games found for {caption_label}.")
        return
    if not _book_sel_keys:
        st.info("Select at least one sportsbook to see betting opportunities.")
        return

    def _recompute_row(row):
        cfg = _label_to_cfg.get(row.get("Market"))
        book_table = row.get("mi_book_table")
        if cfg is None or not isinstance(book_table, list) or not book_table:
            return pd.Series({"_new_best_odds": None, "_new_best_book": None, "_new_ev_num": None, "_new_fair_odds": None})
        game_parts = str(row.get("Game", "")).split(" vs ")
        home_team = game_parts[0] if len(game_parts) == 2 else None
        pick_label = row.get("Pick", "")
        pick_is_a = (
            pick_label == "Over" if pick_label in ("Over", "Under") else pick_label == home_team
        )
        # mi_fair_odds_a/mi_fair_odds_b are both attached to every row of a
        # game (build_market_intelligence merges them at the game level, not
        # per side) — select the one matching this row's own Pick using the
        # same pick_is_a test already used for Best Odds/EV below, rather
        # than defaulting to side A whenever it happens to be truthy.
        fair_odds_val = row.get("mi_fair_odds_a") if pick_is_a else row.get("mi_fair_odds_b")
        odds_key = "_odds_a_raw" if pick_is_a else "_odds_b_raw"
        best_price, best_book_key = None, None
        for b in book_table:
            if b.get("_book_key") not in _book_sel_keys:
                continue
            p = b.get(odds_key)
            if p is None:
                continue
            if best_price is None or p > best_price:
                best_price = p
                best_book_key = b.get("_book_key")
        if best_price is None:
            return pd.Series({"_new_best_odds": None, "_new_best_book": None, "_new_ev_num": None, "_new_fair_odds": None})
        # Use the pipeline's full-precision fair probability directly rather
        # than reconstructing it from the already-1-decimal-rounded "Fair
        # Win %" display string — that rounding previously caused this
        # recomputed EV to drift slightly from the canonical EV%/Top Plays
        # value for the same pick/price.
        fw_raw = row.get("_fair_raw")
        fair_raw = float(fw_raw) if pd.notna(fw_raw) else None
        new_ev = expected_value_pct(fair_raw, best_price) if fair_raw is not None else None
        return pd.Series({
            "_new_best_odds": best_price,
            "_new_best_book": _sc_name(best_book_key) if best_book_key else None,
            "_new_ev_num": new_ev,
            "_new_fair_odds": fair_odds_val,
        })

    _filt = df_all.copy()
    if _mkt_sel != "All":
        _filt = _filt[_filt["Market"] == _mkt_sel]
    _filt = pd.concat([_filt, _filt.apply(_recompute_row, axis=1)], axis=1)

    if debug_mode:
        st.caption(
            f"debug: {len(_filt)} rows before book match, "
            f"{_filt['_new_best_odds'].notna().sum()} have a matching book"
        )
    _filt = _filt.dropna(subset=["_new_best_odds"])
    if debug_mode:
        st.caption(
            f"debug: {len(_filt)} rows after book match | "
            f"positive EV: {(_filt['_new_ev_num'] > 0).sum()} | "
            f"negative EV: {(_filt['_new_ev_num'] <= 0).sum()} | "
            f"markets present: {sorted(_filt['Market'].unique().tolist())}"
        )
    if not _show_all_bets:
        _filt = _filt[_filt["_new_ev_num"] > 0]
    if _min_ev_sel > 0:
        _filt = _filt[_filt["_new_ev_num"] >= _min_ev_sel]
    _filt["_odds_val_fmt"] = _filt["_new_best_odds"].apply(lambda v: _odds_value_in_format(v, _odds_format))
    if _min_odds_sel is not None:
        _filt = _filt[_filt["_odds_val_fmt"] >= _min_odds_sel]
    _filt = _filt.sort_values("_new_ev_num", ascending=False).reset_index(drop=True)

    if _filt.empty:
        st.info("No betting opportunities match the current filters.")
        return

    _filt["Best Odds"] = _filt["_new_best_odds"].apply(lambda v: _fmt_best_odds(v, _odds_format))
    _filt["Fair Odds"] = _filt["_new_fair_odds"].apply(lambda v: _fmt_best_odds(v, _odds_format))
    _filt["EV%"] = _filt["_new_ev_num"].apply(fmt_ev)
    _filt["Bet"] = _filt.apply(_bet_compact_label, axis=1)
    _filt["Best Odds (Fair)"] = _filt["Best Odds"] + " (" + _filt["Fair Odds"] + ")"

    with st.expander("Fair Value Model Results", expanded=True):
        _tbl_cols = ["Bet", "Best Odds (Fair)", "EV%"]
        _tbl_display = _filt[_tbl_cols].reset_index(drop=True)
        st.dataframe(
            _tbl_display, use_container_width=True, hide_index=True,
            height=min(600, 38 + 35 * len(_tbl_display)),
            column_config={
                "Bet":               st.column_config.TextColumn("Bet", width="large"),
                "Best Odds (Fair)":  st.column_config.TextColumn(
                    "Best Odds (Fair)",
                    help=f"{TIPS['best_odds']} Fair Odds in parentheses — {TIPS['fair_odds']}",
                    width="medium",
                ),
                "EV%":               st.column_config.TextColumn("EV%", help=TIPS["ev"], width="small"),
            },
        )
        st.caption(
            f"{len(_tbl_display)} result{'s' if len(_tbl_display) != 1 else ''} "
            f"· {len(_book_sel_keys)} sportsbook{'s' if len(_book_sel_keys) != 1 else ''}"
        )
