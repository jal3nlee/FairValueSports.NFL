# core/nfl_live_scores.py
# Live NFL game status via ESPN's public scoreboard endpoint. The Odds API
# (this app's only other NFL data source) has no score/status data at all —
# this is a genuinely new, separate source, not something already present
# and simply unrendered.
#
# NOTE: response shape below is based on consistent, cross-referenced public
# documentation of this endpoint (not yet live-tested against this specific
# deployment) — same honest caveat as every other ESPN-endpoint integration
# already in this app. Worth a quick sanity check on a real live game once
# deployed.
import requests
import streamlit as st

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"


@st.cache_data(ttl=45, show_spinner=False)
def fetch_nfl_scoreboard() -> list[dict]:
    """One shared fetch for ALL users within the 45s window — not per-page-view."""
    try:
        r = requests.get(SCOREBOARD_URL, timeout=10)
        if r.status_code != 200:
            return []
        return r.json().get("events", [])
    except Exception:
        return []


def get_game_status(events: list[dict], home_abbr: str, away_abbr: str) -> dict:
    """
    Matches on team ABBREVIATION, not display-name string matching, per
    the stable-identifier preference — home_abbr/away_abbr should be the
    same uppercase codes already used for logos elsewhere in this file.
    Returns {} if no match (game not found on today's scoreboard, or the
    fetch failed) — callers must fall back to pregame display gracefully.
    """
    if not events or not home_abbr or not away_abbr:
        return {}
    try:
        for event in events:
            comp = (event.get("competitions") or [{}])[0]
            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue
            abbrs = {c.get("team", {}).get("abbreviation", "").upper() for c in competitors}
            if abbrs != {home_abbr.upper(), away_abbr.upper()}:
                continue

            status = event.get("status", {})
            state = (status.get("type") or {}).get("state", "")  # 'pre' / 'in' / 'post'
            home_c = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), {})

            return {
                "state": state,
                "description": (status.get("type") or {}).get("shortDetail", ""),
                "period": status.get("period"),
                "clock": status.get("displayClock"),
                "home_score": home_c.get("score"),
                "away_score": away_c.get("score"),
            }
        return {}
    except Exception:
        return {}


def format_live_line(status: dict) -> str | None:
    """Returns a compact display string, or None if there's nothing live/final to show."""
    if not status or status.get("state") not in ("in", "post"):
        return None
    home_s = status.get("home_score")
    away_s = status.get("away_score")
    if home_s is None or away_s is None:
        return None
    if status["state"] == "post":
        return f"Final — {away_s}-{home_s}"
    period = status.get("period")
    clock = status.get("clock")
    if period and clock:
        return f"LIVE — {away_s}-{home_s} (Q{period} · {clock})"
    return f"LIVE — {away_s}-{home_s}"
