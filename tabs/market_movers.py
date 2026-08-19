# tabs/market_movers.py
import pandas as pd
import streamlit as st

from core.odds_math import parse_iso_dt_utc, EASTERN
from core.pipeline import MARKETS, run_market_pipeline
from core.data_sources import fetch_market_lines, filter_by_window, get_date_window, infer_current_week_index


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


# Same team-abbreviation map and ESPN CDN logo URL pattern already used by
# Sportsbook Screener / Matchup Center / Parlay Builder — reused as-is rather
# than introducing a second logo dataset.
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


def _logo_url(team_name: str) -> str | None:
    abbr = NFL_TEAM_ABBR.get(team_name)
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{abbr}.png" if abbr else None


def render(supabase, now_utc, eff_bankroll, eff_kelly):
    # ── This-week data — independent of the Fair Value Model's Date Range.
    # NFL games cluster on Thu/Sun/Mon, so a literal "Today" window (the
    # MLB pattern this was ported from) shows empty on most days. Use the
    # current NFL week instead.
    _current_week = infer_current_week_index(now_utc)
    _week_label = "NFL Preseason" if _current_week == 0 else f"NFL Week {_current_week}"
    _week_start, _week_end, _sport_keys, _week_caption = get_date_window(now_utc, _week_label)
    _week_raw: dict[str, pd.DataFrame] = {}
    _week_display: dict[str, pd.DataFrame] = {}
    _week_pulled: list = []

    for _mkt_key, _mkt_cfg in MARKETS.items():
        _raw_t, _pulled_t = fetch_market_lines(supabase, _sport_keys, _mkt_cfg.db_market_key)
        _raw_t_filtered = filter_by_window(_raw_t, _week_start, _week_end)
        _week_raw[_mkt_key] = _raw_t_filtered
        _week_pulled.extend(_pulled_t)
        _week_display[_mkt_key] = run_market_pipeline(
            raw_lines=_raw_t_filtered, cfg=_mkt_cfg, bankroll=eff_bankroll, kelly=eff_kelly,
            min_ev=0.0, min_fair_pct=0.0, show_all=True,
        )

    df_week = (
        pd.concat([df for df in _week_display.values() if not df.empty], ignore_index=True)
        if any(not df.empty for df in _week_display.values())
        else pd.DataFrame()
    )

    with st.expander("Market Movers", expanded=True):
        st.caption(
            "A high-level view of this week's NFL betting market. "
            "Highlights the most significant activity across games, markets, and sportsbooks."
        )

        # ── Market Snapshot ──────────────────────────────────────
        st.markdown(f"**{_week_label} Market Snapshot**")

        _snap_games = 0
        _snap_books = 0
        _snap_markets = 0
        _latest_pull = None

        for _mkt_key, _raw in _week_raw.items():
            if not _raw.empty:
                _snap_games = max(_snap_games, _raw[["home_team", "away_team"]].drop_duplicates().shape[0])
                _snap_books = max(_snap_books, _raw["book"].nunique())
                _snap_markets += 1

        for _p in _week_pulled:
            _dt = parse_iso_dt_utc(_p)
            if _dt and (_latest_pull is None or _dt > _latest_pull):
                _latest_pull = _dt

        _pull_str = (
            _latest_pull.astimezone(EASTERN).strftime("%b %d  %I:%M %p ET") if _latest_pull else "—"
        )

        _snapshot_rows = [
            {"Metric": "Games This Week", "Value": str(_snap_games) if _snap_games else "—"},
            {"Metric": "Active Markets", "Value": str(_snap_markets) if _snap_markets else "—"},
            {"Metric": "Sportsbooks", "Value": str(_snap_books) if _snap_books else "—"},
            {"Metric": "Odds Last Updated", "Value": _pull_str},
        ]
        st.dataframe(
            pd.DataFrame(_snapshot_rows), use_container_width=True, hide_index=True,
            height=38 + 35 * len(_snapshot_rows),
        )

        st.divider()

        # ── Top EV Plays ─────────────────────────────────────
        st.markdown("**This Week's Top EV Plays**")
        st.caption(
            "The three highest expected value betting opportunities identified by "
            f"the Fair Value Model for {_week_label.lower()}."
        )

        if df_week.empty:
            st.info("No positive EV opportunities were identified at this time.")
        else:
            _ev_col = pd.to_numeric(df_week["EV%"].astype(str).str.replace("%", "", regex=False), errors="coerce")
            _top_ev = (
                df_week.assign(_ev_num=_ev_col)
                .loc[lambda d: d["_ev_num"] > 0]
                .sort_values("_ev_num", ascending=False)
                .head(3)
                .reset_index(drop=True)
            )

            if _top_ev.empty:
                st.info("No positive EV opportunities were identified at this time.")
            else:
                for _, _r in _top_ev.iterrows():
                    _game = _r.get("Game", "—")
                    _teams = _game.split(" vs ") if isinstance(_game, str) else []
                    _home_team = _teams[0] if len(_teams) == 2 else None
                    _away_team = _teams[1] if len(_teams) == 2 else None
                    _away_logo = _logo_url(_away_team) if _away_team else None
                    _home_logo = _logo_url(_home_team) if _home_team else None

                    with st.container(border=True):
                        if _away_logo and _home_logo:
                            st.markdown(
                                f"<img src='{_away_logo}' width='22' style='vertical-align:middle;margin-right:5px'/>"
                                f"**{_away_team}** @ "
                                f"<img src='{_home_logo}' width='22' style='vertical-align:middle;margin:0 5px'/>"
                                f"**{_home_team}**",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(f"**{_game}**")

                        st.caption(f"{_r.get('Market', '—')} · Pick: {_r.get('Pick', '—')}")

                        st.dataframe(
                            pd.DataFrame([{
                                "Best Book": _sc_name(_r.get("Best Book", "—")),
                                "Best Odds": _r.get("Best Odds", "—"),
                                "EV%": _r.get("EV%", "—"),
                            }]),
                            use_container_width=True, hide_index=True, height=38 + 35,
                        )

            st.caption("See the full list of positive EV opportunities below.")

        st.divider()
        st.caption("Use **Matchup Center** to research a specific game.")

    st.divider()
