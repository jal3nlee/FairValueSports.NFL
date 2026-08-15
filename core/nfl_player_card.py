# core/nfl_player_card.py
# Shared, compact NFL player identity card: headshot, name, position/team,
# opponent + game time (or Bye Week), and — non-compact only — a 3-stat
# position-relevant season snapshot. No betting-market data (spread,
# game total, team total) anymore; that lives in Game Environment.
import streamlit as st

from core.odds_math import parse_iso_dt_utc, EASTERN
from core.lineup_data import NFL_TEAMS
from core.nflverse_data import get_card_season_stats

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


def render_nfl_player_card(
    player: dict,
    context: dict | None = None,
    compact: bool = False,
    spread: float | None = None,      # accepted for call-site compatibility, no longer rendered
    game_total: float | None = None,  # accepted for call-site compatibility, no longer rendered
    team_total: float | None = None,  # accepted for call-site compatibility, no longer rendered
):
    """
    player: {"name", "team", "position", "headshot_url"} — team is the
    full display name, abbreviated here for display.
    context: dict from core.lineup_data.get_team_game_context(), or
    None/{} for a bye week / context not yet available.
    compact=True (Prop Leaderboard's Player Search): tight, width-
    constrained flex-row card — headshot and identity text grouped
    close together, card doesn't stretch across the full page.
    compact=False (Lineup Analysis): unchanged — full-width st.container
    + st.columns layout, plus the 3-stat season snapshot.
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

    if compact:
        headshot_size = 72
        img_html = (
            f"<img src='{headshot}' style='width:{headshot_size}px;height:{headshot_size}px;"
            f"object-fit:contain;flex-shrink:0;'/>"
            if headshot else
            f"<div style='width:{headshot_size}px;height:{headshot_size}px;border-radius:8px;"
            f"background:rgba(128,128,128,0.15);flex-shrink:0;'></div>"
        )
        lines = [
            f"<div style='font-weight:700;font-size:1.05rem;line-height:1.25'>{player['name']}</div>",
            f"<div style='opacity:0.65;font-size:0.85rem;line-height:1.3'>{player.get('position','')} · {team_abbr}</div>",
        ]
        if opp_line:
            lines.append(f"<div style='opacity:0.6;font-size:0.8rem;line-height:1.3'>{opp_line}</div>")

        st.markdown(
            f"<div style='display:flex;align-items:center;gap:16px;max-width:520px;"
            f"border:1px solid rgba(128,128,128,0.25);border-radius:12px;padding:18px;margin:2px 0;'>"
            f"{img_html}"
            f"<div style='min-width:0;'>{''.join(lines)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Full variant — unchanged from before ─────────────────
    season_stats = get_card_season_stats(player["name"], player.get("team", ""), player.get("position", ""))
    headshot_size = 80

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
            if season_stats:
                _stat_str = "   ·   ".join(f"{s['label']} {s['value']:g}" for s in season_stats)
                st.markdown(
                    f"<div style='opacity:0.75;font-size:0.8rem;margin-top:4px'>{_stat_str}</div>",
                    unsafe_allow_html=True,
                )
