# core/lineup_data.py
import os
import requests
import streamlit as st
import pandas as pd

from core.data_sources import fetch_market_lines, get_date_window

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_SPORT_KEY = "americanfootball_nfl"

POSITION_PROP_MARKETS = {
    "QB": [
        "player_pass_yds", "player_pass_tds", "player_pass_interceptions",
        "player_pass_attempts", "player_pass_completions",
        "player_rush_yds", "player_anytime_td",
    ],
    "RB": [
        "player_rush_yds", "player_rush_attempts",
        "player_reception_yds", "player_receptions", "player_anytime_td",
    ],
    "WR": ["player_reception_yds", "player_receptions", "player_anytime_td"],
    "TE": ["player_reception_yds", "player_receptions", "player_anytime_td"],
}

PROP_LABELS = {
    "player_pass_yds": "Passing Yards",
    "player_pass_tds": "Passing TDs",
    "player_pass_interceptions": "Interceptions",
    "player_pass_attempts": "Pass Attempts",
    "player_pass_completions": "Completions",
    "player_rush_yds": "Rushing Yards",
    "player_rush_attempts": "Rush Attempts",
    "player_reception_yds": "Receiving Yards",
    "player_receptions": "Receptions",
    "player_anytime_td": "Anytime TD",
}

# Same abbreviation map used elsewhere in the app (matchup_center.py's
# logo lookups) — kept identical rather than introducing a second version.
NFL_TEAMS = {
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

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]

DEBUG_LINEUP = True  # temporary — remove once search is confirmed working


# =======================
# ESPN — free player identity/search source (no key required)
# =======================
@st.cache_data(ttl=3600, show_spinner=False)
def espn_search_players(query: str) -> list[dict]:
    """Search NFL players via ESPN's public site API. Free, unofficial, no key."""
    if not query or len(query.strip()) < 2:
        return []
    try:
        r = requests.get(
            "https://site.web.api.espn.com/apis/common/v3/search",
            params={"query": query.strip(), "limit": 10, "type": "player", "sport": "football", "league": "nfl"},
            timeout=10,
        )
        if r.status_code != 200:
            if DEBUG_LINEUP:
                st.error(f"debug: ESPN search returned status {r.status_code} — {r.text[:300]}")
            return []
        payload = r.json()
        if DEBUG_LINEUP:
            st.caption(f"debug: ESPN search top-level keys = {list(payload.keys())}")
        results = []
        for item in payload.get("items", []):
            for res in item.get("results", []):
                if res.get("type") != "player":
                    continue
                results.append({
                    "id": res.get("uid", res.get("id")),
                    "name": res.get("displayName", ""),
                    "team": (res.get("subtitle") or "").strip(),
                })
        if DEBUG_LINEUP and not results:
            st.caption(f"debug: ESPN search returned 0 parsed players — raw payload sample: {str(payload)[:500]}")
        return results
    except Exception as e:
        if DEBUG_LINEUP:
            st.error(f"debug: espn_search_players failed — {type(e).__name__}: {e}")
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def espn_team_roster(team_abbr: str) -> list[dict]:
    """Full roster for one team — the browse-by-team fallback path."""
    try:
        r = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_abbr}/roster",
            timeout=10,
        )
        if r.status_code != 200:
            if DEBUG_LINEUP:
                st.error(f"debug: ESPN roster returned status {r.status_code} for team={team_abbr}")
            return []
        out = []
        for group in r.json().get("athletes", []):
            for p in group.get("items", []):
                out.append({
                    "id": p.get("id"),
                    "name": p.get("fullName", ""),
                    "position": (p.get("position") or {}).get("abbreviation", ""),
                    "team": team_abbr,
                })
        return out
    except Exception as e:
        if DEBUG_LINEUP:
            st.error(f"debug: espn_team_roster failed — {type(e).__name__}: {e}")
        return []


# =======================
# Game environment — reuses the existing odds pipeline, no new fetch
# =======================
def get_team_game_context(supabase, team_name: str, now_utc) -> dict:
    window_start, window_end, sport_keys, _ = get_date_window(now_utc, "Next 7 Days")

    raw, _ = fetch_market_lines(supabase, sport_keys, "spread")
    if raw.empty:
        return {}
    games = raw[raw["home_team"].eq(team_name) | raw["away_team"].eq(team_name)]
    if games.empty:
        return {}
    g = games.sort_values("commence_time").iloc[0]
    is_home = g["home_team"] == team_name
    opponent = g["away_team"] if is_home else g["home_team"]

    total_raw, _ = fetch_market_lines(supabase, sport_keys, "total")
    total_game = total_raw[total_raw["event_id"] == g["event_id"]]
    game_total = None
    if not total_game.empty and "line" in total_game.columns:
        vals = total_game["line"].dropna()
        if not vals.empty:
            game_total = float(vals.iloc[0])

    team_spread = float(g["line"]) if pd.notna(g.get("line")) else None
    if team_spread is not None and not is_home:
        team_spread = -team_spread

    return {
        "event_id": g["event_id"],
        "opponent": opponent,
        "is_home": is_home,
        "commence_time": g["commence_time"],
        "spread": team_spread,
        "game_total": game_total,
        "team_implied_total": calculate_team_implied_total(game_total, team_spread),
    }


def calculate_team_implied_total(game_total: float | None, team_spread: float | None):
    if game_total is None or team_spread is None:
        return None
    return round((game_total / 2) - (team_spread / 2), 1)


# =======================
# Player props — live, per-event, cached
# =======================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_player_props_for_event(event_id: str, position: str) -> list[dict]:
    markets = POSITION_PROP_MARKETS.get(position, [])
    if not markets or not ODDS_API_KEY:
        return []
    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{ODDS_API_SPORT_KEY}/events/{event_id}/odds",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "markets": ",".join(markets),
                "oddsFormat": "american",
            },
            timeout=15,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        rows = []
        for book in data.get("bookmakers", []):
            for market in book.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "market": market["key"],
                        "player": outcome.get("description", ""),
                        "book": book["key"],
                        "line": outcome.get("point"),
                        "side": outcome.get("name"),
                        "price": outcome.get("price"),
                    })
        return rows
    except Exception:
        return []


def get_consensus_prop_line(prop_rows: list[dict], player_name: str, market_key: str):
    vals = [
        r["line"] for r in prop_rows
        if r["player"].strip().lower() == player_name.strip().lower()
        and r["market"] == market_key
        and r.get("side") in ("Over", "Yes")
        and r.get("line") is not None
    ]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else round((vals[n // 2 - 1] + vals[n // 2]) / 2, 1)


def get_relevant_props_for_position(position: str) -> list[str]:
    return POSITION_PROP_MARKETS.get(position, [])


def build_player_comparison(supabase, players: list[dict], now_utc) -> list[dict]:
    enriched = []
    for p in players:
        ctx = get_team_game_context(supabase, p["team"], now_utc) if p.get("team") else {}
        props = {}
        if ctx.get("event_id"):
            raw_props = fetch_player_props_for_event(ctx["event_id"], p["position"])
            for market_key in get_relevant_props_for_position(p["position"]):
                props[market_key] = get_consensus_prop_line(raw_props, p["name"], market_key)
        enriched.append({**p, "context": ctx, "props": props})
    return enriched


def generate_comparison_notes(enriched_players: list[dict]) -> list[str]:
    notes = []
    totals = [(p["name"], p["context"].get("team_implied_total")) for p in enriched_players if p["context"].get("team_implied_total") is not None]
    if len(totals) >= 2:
        totals.sort(key=lambda x: x[1], reverse=True)
        notes.append(f"{totals[0][0]}'s team has the higher implied point total ({totals[0][1]} vs {totals[-1][1]}).")

    for market_key, label in [("player_reception_yds", "receiving yards"), ("player_rush_yds", "rushing yards")]:
        vals = [(p["name"], p["props"].get(market_key)) for p in enriched_players if p["props"].get(market_key) is not None]
        if len(vals) >= 2:
            vals.sort(key=lambda x: x[1], reverse=True)
            notes.append(f"{vals[0][0]} has the higher {label} line ({vals[0][1]} vs {vals[-1][1]}).")

    if not notes:
        notes.append("Not enough market data available yet to compare these players — check back closer to kickoff as more books post lines.")
    return notes
