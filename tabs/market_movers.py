# tabs/market_movers.py
import pandas as pd
import streamlit as st

from core.odds_math import parse_iso_dt_utc, EASTERN
from core.pipeline import MARKETS, run_market_pipeline
from core.data_sources import fetch_market_lines, filter_by_window, get_date_window


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


def render(supabase, now_utc, eff_bankroll, eff_kelly):
    st.markdown("## Market Movers")
    st.caption(
        "A high-level view of today's NFL betting market. "
        "Highlights the most significant activity across games, markets, and sportsbooks."
    )
    st.divider()

    # ── Today-only data — independent of the Fair Value Model's Date Range ──
    _today_start, _today_end, _sport_keys, _ = get_date_window(now_utc, "Today")
    _today_raw: dict[str, pd.DataFrame] = {}
    _today_display: dict[str, pd.DataFrame] = {}
    _today_pulled: list = []

    for _mkt_key, _mkt_cfg in MARKETS.items():
        _raw_t, _pulled_t = fetch_market_lines(supabase, _sport_keys, _mkt_cfg.db_market_key)
        _raw_t_filtered = filter_by_window(_raw_t, _today_start, _today_end)
        _today_raw[_mkt_key] = _raw_t_filtered
        _today_pulled.extend(_pulled_t)
        _today_display[_mkt_key] = run_market_pipeline(
            raw_lines=_raw_t_filtered, cfg=_mkt_cfg, bankroll=eff_bankroll, kelly=eff_kelly,
            min_ev=0.0, min_fair_pct=0.0, show_all=True,
        )

    df_today = (
        pd.concat([df for df in _today_display.values() if not df.empty], ignore_index=True)
        if any(not df.empty for df in _today_display.values())
        else pd.DataFrame()
    )

    # ── Market Snapshot ──────────────────────────────────────────
    st.markdown("### Today's Market Snapshot")

    _snap_games = 0
    _snap_books = 0
    _snap_markets = 0
    _latest_pull = None

    for _mkt_key, _raw in _today_raw.items():
        if not _raw.empty:
            _snap_games = max(_snap_games, _raw[["home_team", "away_team"]].drop_duplicates().shape[0])
            _snap_books = max(_snap_books, _raw["book"].nunique())
            _snap_markets += 1

    for _p in _today_pulled:
        _dt = parse_iso_dt_utc(_p)
        if _dt and (_latest_pull is None or _dt > _latest_pull):
            _latest_pull = _dt

    _pull_str = (
        _latest_pull.astimezone(EASTERN).strftime("%b %d  %I:%M %p ET") if _latest_pull else "—"
    )

    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("Games Today", _snap_games if _snap_games else "—")
    _m2.metric("Active Markets", _snap_markets if _snap_markets else "—")
    _m3.metric("Sportsbooks", _snap_books if _snap_books else "—")
    _m4.metric("Odds Last Updated", _pull_str)

    st.divider()

    # ── Today's Top EV Plays ─────────────────────────────────────
    st.markdown("### Today's Top EV Plays")
    st.caption(
        "The three highest expected value betting opportunities identified by "
        "the Fair Value Model for today's NFL slate."
    )

    if df_today.empty:
        st.info("No positive EV opportunities were identified at this time.")
    else:
        _ev_col = pd.to_numeric(df_today["EV%"].astype(str).str.replace("%", "", regex=False), errors="coerce")
        _top_ev = (
            df_today.assign(_ev_num=_ev_col)
            .loc[lambda d: d["_ev_num"] > 0]
            .sort_values("_ev_num", ascending=False)
            .head(3)
            .reset_index(drop=True)
        )

        if _top_ev.empty:
            st.info("No positive EV opportunities were identified at this time.")
        else:
            _rows = []
            for _i, (_, _r) in enumerate(_top_ev.iterrows(), 1):
                _fo_val = _r.get("mi_fair_odds_a") or _r.get("mi_fair_odds_b")
                from core.odds_math import fmt_odds
                _fo_str = fmt_odds(_fo_val) if _fo_val else "—"
                _rows.append({
                    "Rank": _i,
                    "Game": _r.get("Game", "—"),
                    "Pick": _r.get("Pick", "—"),
                    "Market": _r.get("Market", "—"),
                    "Best Book": _sc_name(_r.get("Best Book", "—")),
                    "Best Odds": _r.get("Best Odds", "—"),
                    "Fair Odds": _fo_str,
                    "Fair Win %": _r.get("Fair Win %", "—"),
                    "EV%": _r.get("EV%", "—"),
                })
            st.dataframe(pd.DataFrame(_rows).set_index("Rank"), use_container_width=True)

        st.caption("View the complete list of positive EV opportunities in the **Fair Value Model** tab.")

    st.divider()
    st.caption(
        "Use **Fair Value Model** to find positive EV opportunities.  "
        "Use **Matchup Center** to research a specific game."
    )
