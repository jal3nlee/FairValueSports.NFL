# tabs/arbitrage_tracker.py
import pandas as pd
import streamlit as st

from core.odds_math import parse_iso_dt_utc, EASTERN
from core.pipeline import MARKETS, run_market_pipeline
from core.data_sources import fetch_market_lines, filter_by_window, get_date_window
from core.arbitrage_engine import find_arbitrage_opportunities


def render(supabase, now_utc, eff_bankroll, eff_kelly):
    st.markdown("## Arbitrage Tracker")
    st.markdown(
        "<div style='opacity:0.7;font-size:0.95rem;margin:0 0 6px 0'>"
        "Find markets where the best price on each side, across different sportsbooks, "
        "guarantees a profit regardless of outcome."
        "</div>",
        unsafe_allow_html=True,
    )
    st.info(
        "⚠️ Real arbitrage windows are usually small and close within minutes as books "
        "adjust. This scans on-demand — click Refresh to check current prices. Treat this "
        "as research and education, not a live execution feed.",
        icon="ℹ️",
    )

    _c1, _c2, _c3 = st.columns([1.6, 1.2, 1.2])
    with _c1:
        _window_choice = st.selectbox("Date Range", ["Today", "This Week", "Next 7 Days"], index=2, key="arb_window")
    with _c2:
        _stake_bankroll = st.number_input("Stake Bankroll ($)", min_value=10.0, value=100.0, step=10.0, key="arb_bankroll")
    with _c3:
        st.markdown("<div style='margin-top:1.6rem'></div>", unsafe_allow_html=True)
        _refresh = st.button("🔄 Refresh Arbitrage Scan", type="primary", use_container_width=True)

    if not _refresh and "arb_results" not in st.session_state:
        st.caption("Click **Refresh Arbitrage Scan** to check current prices for arbitrage opportunities.")
        return

    if _refresh:
        window_start, window_end, sport_keys, caption_label = get_date_window(now_utc, _window_choice)

        with st.spinner("Scanning current odds across all markets..."):
            _all_display = []
            for _mkt_key, _mkt_cfg in MARKETS.items():
                _raw, _pulled = fetch_market_lines(supabase, sport_keys, _mkt_cfg.db_market_key)
                _raw = filter_by_window(_raw, window_start, window_end)
                _display = run_market_pipeline(
                    raw_lines=_raw, cfg=_mkt_cfg, bankroll=eff_bankroll, kelly=eff_kelly,
                    min_ev=-999.0, min_fair_pct=0.0, show_all=True,
                )
                if not _display.empty:
                    _all_display.append(_display)

            _df_all = pd.concat(_all_display, ignore_index=True) if _all_display else pd.DataFrame()
            _opportunities = find_arbitrage_opportunities(_df_all, _stake_bankroll) if not _df_all.empty else []

            st.session_state["arb_results"] = _opportunities
            st.session_state["arb_scanned_at"] = now_utc
            st.session_state["arb_caption"] = caption_label

    opportunities = st.session_state.get("arb_results", [])
    caption_label = st.session_state.get("arb_caption", "")
    scanned_at = st.session_state.get("arb_scanned_at")

    if scanned_at:
        st.caption(f"Last scanned: {scanned_at.astimezone(EASTERN).strftime('%I:%M %p ET').lstrip('0')} · {caption_label}")

    if not opportunities:
        st.info("No arbitrage opportunities found in the current price set. Try refreshing again shortly, or widen the date range.")
        return

    st.success(f"Found {len(opportunities)} arbitrage opportunit{'y' if len(opportunities) == 1 else 'ies'}.")

    for opp in opportunities:
        _dt = parse_iso_dt_utc(opp["commence_time"])
        _time_str = _dt.astimezone(EASTERN).strftime("%a %I:%M %p ET").lstrip("0") if _dt else ""
        _line_str = f" · Line {opp['line']}" if opp["line"] else ""

        with st.container(border=True):
            _hc1, _hc2 = st.columns([3, 1])
            with _hc1:
                st.markdown(f"**{opp['game']}**")
                st.caption(f"{opp['market']}{_line_str} · {_time_str}")
            with _hc2:
                st.markdown(
                    f"<div style='text-align:right;font-size:1.4rem;font-weight:700;color:#2ecc71'>"
                    f"+{opp['profit_pct']:.2f}%</div>",
                    unsafe_allow_html=True,
                )
                st.caption("Guaranteed margin")

            _rows = [
                {"Pick": leg["pick"], "Sportsbook": leg["book"], "Odds": leg["odds"], "Stake": f"${leg['stake']:,.2f}"}
                for leg in opp["legs"]
            ]
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
            st.caption(
                f"Total stake: ${sum(l['stake'] for l in opp['legs']):,.2f}  ·  "
                f"Guaranteed profit: ${opp['guaranteed_profit']:,.2f} regardless of outcome"
            )
