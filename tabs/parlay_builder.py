# tabs/parlay_builder.py
import pandas as pd
import streamlit as st

from core.odds_math import american_to_decimal, fmt_odds, parse_iso_dt_utc, EASTERN
from core.pipeline import MARKETS, run_market_pipeline
from core.data_sources import fetch_market_lines, filter_by_window, get_date_window, infer_current_week_index

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


def _fragment_rerun() -> None:
    """Rerun after a session_state mutation so the already-updated state is
    reflected on the very next pass, instead of only becoming visible on
    some later, unrelated interaction. Scoped to the fragment so an Add/
    Remove click doesn't force every other tab's data loading to re-run —
    matches the reason this whole body is wrapped in @st.fragment. Falls
    back to a plain rerun on older Streamlit versions without fragment
    scoping."""
    try:
        st.rerun(scope="fragment")
    except TypeError:
        st.rerun()


def render(supabase, now_utc, eff_bankroll, eff_kelly, authed):
    if not authed:
        st.warning("Sign in to use the Parlay Builder.")
        return

    st.subheader("NFL Parlay Builder")
    stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=1.0, key="pb_stake")
    st.session_state.setdefault("pb_parlay_legs", [])

    _wk = infer_current_week_index(now_utc)
    _week_label = "NFL Preseason" if _wk == 0 else f"NFL Week {_wk}"
    _window_choice = st.selectbox(
        "Date Range", ["Today", _week_label, "Next 7 Days"], index=1, key="pb_window_choice",
    )
    window_start, window_end, sport_keys, caption_label = get_date_window(now_utc, _window_choice)

    @st.fragment
    def _parlay_builder_body():
        st.markdown("### Current Parlay")
        if not st.session_state.pb_parlay_legs:
            st.info("Add at least two legs by clicking Add on any pick below.")
        else:
            for _leg_i, _leg in enumerate(st.session_state.pb_parlay_legs):
                _line_part = (
                    f" {_leg['Line']}" if _leg.get("Market") != "Moneyline" and _leg.get("Line") else ""
                )
                _leg_c1, _leg_c2 = st.columns([5, 1])
                with _leg_c1:
                    st.write(f"{_leg['Pick']}{_line_part} — {_leg['Game']} ({_leg['Market']})")
                with _leg_c2:
                    if st.button("Remove", key=f"pb_leg_remove_{_leg_i}", use_container_width=True):
                        st.session_state.pb_parlay_legs.pop(_leg_i)
                        _fragment_rerun()

            colA, colB = st.columns(2)
            compare = colA.button("Compare Parlay Odds", use_container_width=True, key="pb_compare")
            if colB.button("Clear All Legs", use_container_width=True,
                           disabled=not st.session_state.pb_parlay_legs, key="pb_clear_all"):
                st.session_state.pb_parlay_legs = []
                _fragment_rerun()

            if compare and len(st.session_state.pb_parlay_legs) >= 2:
                _markets_pb = {}
                for _mkt_key, _mkt_cfg in MARKETS.items():
                    _raw_p, _ = fetch_market_lines(supabase, sport_keys, _mkt_cfg.db_market_key)
                    _raw_p = filter_by_window(_raw_p, window_start, window_end)
                    from core.pipeline import build_books_df
                    _markets_pb[_mkt_cfg.market_label] = build_books_df(_raw_p, _mkt_cfg)

                every_book = sorted(set(
                    b for df in _markets_pb.values() if not df.empty for b in df["book"].unique()
                ))

                results = []
                for book in every_book:
                    combined_dec = 1.0
                    line_labels = []
                    valid = True
                    for leg in st.session_state.pb_parlay_legs:
                        src = _markets_pb.get(leg["Market"])
                        if src is None or src.empty:
                            valid = False; break
                        s = src[
                            (src["book"] == book) &
                            ((src["home_team"] + " vs " + src["away_team"]) == leg["Game"])
                        ]
                        if s.empty:
                            valid = False; break
                        row = s.iloc[0]
                        if leg["Market"] == "Moneyline":
                            price = row["home_price"] if leg["Pick"] == row["home_team"] else row["away_price"]
                            line_labels.append("ML")
                        elif leg["Market"] == "Spread":
                            price = row["home_price"] if leg["Pick"] == row["home_team"] else row["away_price"]
                            line_labels.append(leg["Line"])
                        else:
                            price = row["over_price"] if leg["Pick"].lower() == "over" else row["under_price"]
                            line_labels.append(leg["Line"])
                        try:
                            combined_dec *= american_to_decimal(price)
                        except Exception:
                            valid = False; break
                    if valid and combined_dec > 1.0:
                        pa = int((combined_dec - 1) * 100) if combined_dec >= 2 else int(-100 / (combined_dec - 1))
                        results.append({
                            "Sportsbook": book,
                            "American Odds": fmt_odds(pa),
                            "Payout ($)": f"${round(stake * combined_dec, 2):,.2f}",
                            "Lines": ", ".join(line_labels),
                        })

                if not results:
                    st.warning("No sportsbook has all selected legs available.")
                else:
                    st.markdown("### Parlay Comparison")
                    st.dataframe(
                        pd.DataFrame(results).sort_values("Payout ($)", ascending=False),
                        use_container_width=True, hide_index=True,
                    )

        st.divider()
        st.markdown("### Browse Games")
        st.caption("Scroll through every game and market. Click Add to build your parlay. "
                   "Only one pick per market, per game is allowed.")

        _pb_display: dict[str, pd.DataFrame] = {}
        for _mkt_key, _mkt_cfg in MARKETS.items():
            _raw_pb, _ = fetch_market_lines(supabase, sport_keys, _mkt_cfg.db_market_key)
            _raw_pb = filter_by_window(_raw_pb, window_start, window_end)
            _pb_display[_mkt_key] = run_market_pipeline(
                raw_lines=_raw_pb, cfg=_mkt_cfg, bankroll=eff_bankroll, kelly=eff_kelly,
                min_ev=0.0, min_fair_pct=0.0, show_all=True,
            )
        df_all = (
            pd.concat([df for df in _pb_display.values() if not df.empty], ignore_index=True)
            if any(not df.empty for df in _pb_display.values())
            else pd.DataFrame()
        )

        if df_all.empty:
            st.info(f"No games found for {caption_label}.")
            return

        _locked_markets = {
            (l["Game"], l["_commence"], l["Market"]): l
            for l in st.session_state.pb_parlay_legs
            if "_commence" in l
        }

        _games_ordered = (
            df_all.assign(_sort_dt=df_all["commence_time"].apply(parse_iso_dt_utc))
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
            _game_rows = df_all[
                (df_all["Game"] == _game_label) & (df_all["commence_time"] == _commence_iso)
            ].reset_index(drop=True)
            if _game_rows.empty:
                continue
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

            with st.container(border=True):
                if _away_logo and _home_logo:
                    st.markdown(
                        f"<img src='{_away_logo}' width='20' style='vertical-align:middle;margin-right:4px'/>"
                        f"**{_at}** @ "
                        f"<img src='{_home_logo}' width='20' style='vertical-align:middle;margin:0 4px'/>"
                        f"**{_ht}**",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"**{_at} @ {_ht}**")

                for _mkt_name in ["Moneyline", "Spread", "Total"]:
                    _mkt_rows = _game_rows[_game_rows["Market"] == _mkt_name]
                    if _mkt_rows.empty:
                        continue
                    _lock_key = (_game_label, _commence_iso, _mkt_name)
                    _locked_leg = _locked_markets.get(_lock_key)

                    _has_line = "Line" in _mkt_rows.columns and _mkt_rows["Line"].notna().any()
                    _line_vals = (
                        sorted(_mkt_rows["Line"].dropna().unique(), key=lambda x: str(x)) if _has_line else [None]
                    )
                    for _lv in _line_vals:
                        _sub = _mkt_rows[_mkt_rows["Line"] == _lv] if _lv is not None else _mkt_rows
                        _label = _mkt_name + (f" — line {_lv}" if _lv is not None else "")
                        st.caption(_label)
                        for _, _r in _sub.iterrows():
                            _pick_label = _r.get("Pick", "—")
                            _leg_line = "ML" if _mkt_name == "Moneyline" else _r.get("Line", "")
                            _leg = {
                                "Market": _mkt_name, "Game": _game_label,
                                "Pick": _pick_label, "Line": _leg_line, "_commence": _commence_iso,
                            }
                            _is_this_leg_added = (
                                _locked_leg is not None
                                and _locked_leg["Pick"] == _pick_label
                                and _locked_leg["Line"] == _leg_line
                            )
                            _rc1, _rc2 = st.columns([4, 1])
                            _rc1.write(_pick_label)
                            _btn_key = f"pb_add_{_game_label}_{_commence_iso}_{_mkt_name}_{_pick_label}_{_lv}"
                            if _is_this_leg_added:
                                if _rc2.button("Remove", key=_btn_key, use_container_width=True):
                                    st.session_state.pb_parlay_legs = [
                                        l for l in st.session_state.pb_parlay_legs
                                        if not (
                                            l.get("_commence") == _commence_iso
                                            and l["Market"] == _mkt_name
                                            and l["Pick"] == _pick_label
                                            and l["Line"] == _leg_line
                                        )
                                    ]
                                    _fragment_rerun()
                            elif _locked_leg is not None:
                                _rc2.button("Locked", key=_btn_key, disabled=True, use_container_width=True)
                            else:
                                if _rc2.button("Add", key=_btn_key, use_container_width=True):
                                    st.session_state.pb_parlay_legs.append(_leg)
                                    _fragment_rerun()

    _parlay_builder_body()
