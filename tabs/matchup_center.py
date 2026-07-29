# tabs/matchup_center.py
import pandas as pd
import streamlit as st

from core.odds_math import parse_iso_dt_utc, EASTERN
from core.pipeline import MARKETS, run_market_pipeline
from core.data_sources import fetch_market_lines, filter_by_window, get_date_window

TIPS = {
    "ev":        "Expected Value — your edge vs. the fair price. Positive = favorable bet.",
    "fair_win":  "Consensus fair probability — the no-vig estimate of this outcome's true likelihood.",
    "best_odds": "Best available odds — the highest American odds found across all sportsbooks.",
}

# Full team name (as it appears in odds data) -> ESPN abbreviation, for logos.
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


def _logo_url(team_name: str) -> str | None:
    abbr = NFL_TEAM_ABBR.get(team_name)
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{abbr}.png" if abbr else None


def _build_snap_rows(source: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in source.iterrows():
        rows.append({
            "Pick":             r.get("Pick", "—"),
            "Fair Probability": r.get("Fair Win %", "—"),
            "Best Book":        _sc_name(r.get("Best Book", "—")),
            "Best Odds":        r.get("Best Odds", "—"),
            "EV%":              r.get("EV%", "—"),
        })
    return rows


def render(supabase, now_utc, eff_bankroll, eff_kelly):
    st.markdown("## Matchup Center")

    # ── Matchup Center owns its own Date Range — independent of Fair Value Model ──
    _wk_col = st.columns(1)[0]
    from core.data_sources import infer_current_week_index
    _wk = infer_current_week_index(now_utc)
    _week_label = "NFL Preseason" if _wk == 0 else f"NFL Week {_wk}"
    _window_choice = st.selectbox(
        "Date Range", ["Today", _week_label, "Next 7 Days"], index=1, key="mc_window_choice",
    )
    window_start, window_end, sport_keys, caption_label = get_date_window(now_utc, _window_choice)

    st.caption(
        f"Every game for {caption_label}, grouped by start time. "
        "Click a matchup to open the full research breakdown."
    )

    _mc_display: dict[str, pd.DataFrame] = {}
    for _mkt_key, _mkt_cfg in MARKETS.items():
        _raw_mc, _ = fetch_market_lines(supabase, sport_keys, _mkt_cfg.db_market_key)
        _raw_mc = filter_by_window(_raw_mc, window_start, window_end)
        _mc_display[_mkt_key] = run_market_pipeline(
            raw_lines=_raw_mc, cfg=_mkt_cfg, bankroll=eff_bankroll, kelly=eff_kelly,
            min_ev=0.0, min_fair_pct=0.0, show_all=True,
        )
    _df_mc = (
        pd.concat([df for df in _mc_display.values() if not df.empty], ignore_index=True)
        if any(not df.empty for df in _mc_display.values())
        else pd.DataFrame()
    )

    if _df_mc.empty:
        st.info(f"No games available for {caption_label}.")
        return

    _games_ordered = (
        _df_mc.assign(_sort_dt=_df_mc["commence_time"].apply(parse_iso_dt_utc))
        .dropna(subset=["_sort_dt"])
        [["Game", "commence_time", "_sort_dt"]]
        .drop_duplicates(subset=["Game", "commence_time"])
        .sort_values("_sort_dt")
        [["Game", "commence_time"]]
        .to_records(index=False)
        .tolist()
    )

    _last_bucket = None
    for _game_label, _commence_iso in _games_ordered:
        _game_rows = _df_mc[
            (_df_mc["Game"] == _game_label) & (_df_mc["commence_time"] == _commence_iso)
        ].reset_index(drop=True)
        if _game_rows.empty:
            continue
        _date_str = _game_rows["Date"].iloc[0] or ""
        _dt = parse_iso_dt_utc(_commence_iso)
        if not _dt:
            continue
        _et = _dt.astimezone(EASTERN)
        _bucket = _et.replace(minute=0, second=0, microsecond=0)

        if _bucket != _last_bucket:
            st.markdown(f"##### {_bucket.strftime('%a %I:%M %p ET').lstrip('0').replace(' 0', ' ')}")
            _last_bucket = _bucket

        _teams = _game_label.split(" vs ")
        _ht = _teams[0] if len(_teams) == 2 else _game_label
        _at = _teams[1] if len(_teams) == 2 else ""
        _away_logo = _logo_url(_at)
        _home_logo = _logo_url(_ht)

        _card_title = f"**{_at} @ {_ht}**  |  {_date_str}"
        with st.expander(_card_title, expanded=False):
            if _away_logo and _home_logo:
                st.markdown(
                    f"<img src='{_away_logo}' width='24' style='vertical-align:middle;margin-right:6px'/>"
                    f"**{_at}** @ "
                    f"<img src='{_home_logo}' width='24' style='vertical-align:middle;margin:0 6px'/>"
                    f"**{_ht}**",
                    unsafe_allow_html=True,
                )

            # ── 1. Market Snapshot ────────────────────────────
            st.markdown("#### Market Snapshot")
            st.caption("Fair probabilities and best EV across all markets for this game.")
            _snap_source = _game_rows.copy()
            _has_line = "Line" in _snap_source.columns and _snap_source["Line"].notna().any()
            _snap_col_config = {
                "EV%": st.column_config.TextColumn("EV%", help=TIPS["ev"]),
                "Fair Probability": st.column_config.TextColumn("Fair Probability", help=TIPS["fair_win"]),
                "Best Odds": st.column_config.TextColumn("Best Odds", help=TIPS["best_odds"]),
            }
            if not _has_line:
                _snap_rows = _build_snap_rows(_snap_source)
                if _snap_rows:
                    st.dataframe(pd.DataFrame(_snap_rows), use_container_width=True, hide_index=True,
                                 column_config=_snap_col_config)
            else:
                _ml_source = _snap_source[_snap_source["Line"].isna() | (_snap_source["Line"] == "")]
                _lined_source = _snap_source[_snap_source["Line"].notna() & (_snap_source["Line"] != "")]
                if not _ml_source.empty:
                    st.caption("Moneyline")
                    st.dataframe(pd.DataFrame(_build_snap_rows(_ml_source)), use_container_width=True,
                                 hide_index=True, column_config=_snap_col_config)
                for _mkt_name in _lined_source["Market"].unique():
                    _mkt_group = _lined_source[_lined_source["Market"] == _mkt_name]
                    for _line_val in sorted(_mkt_group["Line"].unique(), key=lambda x: str(x)):
                        _line_group = _mkt_group[_mkt_group["Line"] == _line_val]
                        st.caption(f"{_mkt_name} — Line {_line_val}")
                        st.dataframe(pd.DataFrame(_build_snap_rows(_line_group)), use_container_width=True,
                                     hide_index=True, column_config=_snap_col_config)
            st.divider()

            # ── 2. Betting Splits ─────────────────────────────
            st.markdown("#### Betting Splits")
            st.info(
                "Betting splits require a dedicated data provider and aren't "
                "available through the current odds feed."
            )
            st.divider()

            # ── 3. Team Comparison ─────────────────────────────
            st.markdown("#### Team Comparison")
            st.info(
                "Team stats require an NFL stats provider — pending a data source decision."
            )
            st.divider()

            # ── 4. Recent Form ─────────────────────────────────
            st.markdown("#### Recent Form")
            st.info(
                "Recent form requires an NFL stats provider — pending a data source decision."
            )
            st.divider()

            # ── 5. Injuries ─────────────────────────────────────
            st.markdown("#### Injuries")
            st.info(
                "Injury reports require a dedicated data provider and aren't "
                "available through the current odds/stats feeds."
            )
