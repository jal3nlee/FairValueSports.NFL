# core/nfl_player_card.py
# Shared, compact "profile header" card for a selected NFL player —
# identity + this-week context only. No analytics, no recommendations.
import streamlit as st

from core.odds_math import parse_iso_dt_utc, EASTERN
from core.lineup_data import NFL_TEAMS

_ABBR = {full: abbr.upper() for full, abbr in NFL_TEAMS.items()}


def _team_abbr(team_full_name: str) -> str:
    return _ABBR.get(team_full_name, team_full_name)


def _fmt_game_time(commence_iso: str | None) -> str | None:
    if not commence_iso:
        return None
    dt = parse_iso_dt_utc(commence_iso)
    if not dt:
        return None
    et = dt.astimezone(EASTERN)
    return et.strftime("%a %I:%M %p ET").replace(" 0", " ")


def render_nfl_player_card(player: dict, context: dict | None = None, compact: bool = False):
    """
    player: {"name", "team", "position", "headshot_url"} — team is the
    full display name (e.g. "Minnesota Vikings"), abbreviated here for display.
    context: the dict from core.lineup_data.get_team_game_context(), or
    None/{} for a bye week / context not yet available.
    compact=True omits the game-context chips (used in Prop Leaderboard's
    Player Search, where the next action is the prop controls, not context).
    """
    context = context or {}
    team_abbr = _team_abbr(player.get("team", ""))
    headshot = player.get("headshot_url")

    opponent = context.get("opponent")
    opp_line = None
    if opponent:
        side = "vs" if context.get("is_home") else "@"
        opp_abbr = _team_abbr(opponent)
        game_time = _fmt_game_time(context.get("commence_time"))
        opp_line = f"{side} {opp_abbr}" + (f" · {game_time}" if game_time else "")
    elif "opponent" in context:  # context was fetched but genuinely empty -> bye week
        opp_line = "Bye Week"

    chips = []
    if not compact and opponent:
        spread = context.get("spread")
        game_total = context.get("game_total")
        team_total = context.get("team_implied_total")
        if team_total is not None:
            chips.append(f"Team Total {team_total:g}")
        if spread is not None:
            chips.append(f"Spread {spread:+g}")
        if game_total is not None:
            chips.append(f"Game Total {game_total:g}")

    headshot_size = 80 if not compact else 64  # single value each — no real desktop/mobile breakpoint in Streamlit

    with st.container(border=True):
        _photo_col, _info_col = st.columns([0.85, 3.15])
        with _photo_col:
            if headshot:
                st.markdown(
                    f"<div style='width:{headshot_size}px;height:{headshot_size}px;display:flex;"
                    f"align-items:center;justify-content:center;overflow:hidden;'>"
                    f"<img src='{headshot}' style='width:100%;height:100%;object-fit:contain;'/></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='width:{headshot_size}px;height:{headshot_size}px;border-radius:8px;"
                    f"background:rgba(128,128,128,0.15);'></div>",
                    unsafe_allow_html=True,
                )
        with _info_col:
            st.markdown(
                f"<div style='font-weight:700;font-size:1.05rem;line-height:1.25;margin-top:2px'>{player['name']}</div>"
                f"<div style='opacity:0.65;font-size:0.85rem;line-height:1.3'>{player.get('position','')} · {team_abbr}</div>",
                unsafe_allow_html=True,
            )
            if opp_line:
                st.markdown(
                    f"<div style='opacity:0.6;font-size:0.8rem;margin-top:1px'>{opp_line}</div>",
                    unsafe_allow_html=True,
                )
            if chips:
                st.markdown(
                    f"<div style='opacity:0.75;font-size:0.8rem;margin-top:4px'>{'   ·   '.join(chips)}</div>",
                    unsafe_allow_html=True,
                )
