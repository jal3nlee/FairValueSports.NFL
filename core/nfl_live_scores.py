# core/nfl_live_scores.py
import requests
import streamlit as st

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

DEBUG_LIVE = False  # temporary — remove once live scores are confirmed working


@st.cache_data(ttl=45, show_spinner=False)
def fetch_nfl_scoreboard() -> list[dict]:
    """One shared fetch for ALL users within the 45s window."""
    try:
        r = requests.get(SCOREBOARD_URL, timeout=10)
        if DEBUG_LIVE:
            st.caption(f"debug: scoreboard HTTP status = {r.status_code}")
        if r.status_code != 200:
            return []
        events = r.json().get("events", [])
        if DEBUG_LIVE:
            st.caption(f"debug: scoreboard returned {len(events)} event(s)")
            for e in events[:5]:
                comp = (e.get("competitions") or [{}])[0]
                competitors = comp.get("competitors", [])
                abbrs = [c.get("team", {}).get("abbreviation") for c in competitors]
                state = (e.get("status", {}).get("type") or {}).get("state")
                st.caption(f"debug: event — teams={abbrs} state={state!r}")
        return events
    except Exception as e:
        if DEBUG_LIVE:
            st.error(f"debug: fetch_nfl_scoreboard failed — {type(e).__name__}: {e}")
        return []


def get_game_status(events: list[dict], home_abbr: str, away_abbr: str) -> dict:
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
            state = (status.get("type") or {}).get("state", "")
            home_c = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), {})

            result = {
                "state": state,
                "description": (status.get("type") or {}).get("shortDetail", ""),
                "period": status.get("period"),
                "clock": status.get("displayClock"),
                "home_score": home_c.get("score"),
                "away_score": away_c.get("score"),
            }
            if DEBUG_LIVE:
                st.caption(f"debug: matched {home_abbr}/{away_abbr} -> {result}")
            return result
        if DEBUG_LIVE:
            st.caption(f"debug: no scoreboard match for home={home_abbr!r} away={away_abbr!r}")
        return {}
    except Exception as e:
        if DEBUG_LIVE:
            st.error(f"debug: get_game_status failed — {type(e).__name__}: {e}")
        return {}


def format_live_line(status: dict) -> str | None:
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
