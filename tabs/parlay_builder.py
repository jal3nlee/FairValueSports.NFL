# tabs/parlay_builder.py
import pandas as pd
import streamlit as st

from core.odds_math import american_to_decimal, fmt_odds, parse_iso_dt_utc, EASTERN
from core.pipeline import MARKETS, run_market_pipeline
from core.data_sources import fetch_market_lines, filter_by_window, get_date_window


def render(supabase, now_utc, eff_bankroll, eff_kelly, authed):
    if not authed:
        st.warning("Sign in to use the Parlay Builder.")
        return

    st.subheader("NFL Parlay Builder")
    stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=1.0, key="pb_stake")
    st.session_state.setdefault("pb_parlay_legs", [])

    from core.data_sources import infer_current_week_index
    _wk = infer_current_week_index(now_utc)
    _week_label = "NFL Preseason" if _wk == 0 else f"NFL Week {_wk}"
    _window_choice = st.selectbox(
        "Date Range", ["Today", _week_label, "Next 7 Days"], index=1, key="pb_window_choice",
    )
    window_start, window_end, sport_keys, caption_label = get_date_window(now_utc, _window_choice)

    @st.fragment
    def _parlay_builder_body():
        st.markdown("### Selected Legs")
        if not st.session_state.pb_parlay_legs:
            st.info("Add at least two legs by clicking Add on any pick below.")
        else:
            _legs_df = pd.DataFrame([
                {"Market": l["Market"], "Game": l["Game"], "Pick": l["Pick"], "Line": l["Line"]}
                for l in st.session_state.pb_parlay_legs
            ])
            st.dataframe(_legs_df, use_container_width=True, hide_index=True)

            colA, colB = st.columns(2)
            compare = colA.button("Compare Parlay Odds", use_container_width=True, key="pb_compare")
            if colB.button("Clear All Legs", use_container_width=True,
                           disabled=not st.session_state.pb_parlay_legs, key="pb_clear_all"):
                st.session_state.pb_parlay_legs = []
                st.rerun()

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

            with st.container(border=True):
                st.markdown(f"**{_game_label}**")

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
                            elif _locked_leg is not None:
                                _rc2.button("Locked", key=_btn_key, disabled=True, use_container_width=True)
                            else:
                                if _rc2.button("Add", key=_btn_key, use_container_width=True):
                                    st.session_state.pb_parlay_legs.append(_leg)

    _parlay_builder_body()
