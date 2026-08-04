# fetch_odds_nfl.py
# Pulls NFL odds (moneyline, spread, total) from The Odds API into
# Supabase. Mirrors fetch_odds_mlb.py's structure exactly.
import os
import uuid
import requests
from datetime import datetime, timezone
from supabase import create_client, Client

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Markets to pull
markets = ["h2h", "spreads", "totals"]

# The Odds API's sport key for the request URL
ODDS_API_SPORT_KEY = "americanfootball_nfl"

# What we store internally — matches sport_key_for_week() in core/data_sources.py.
# Not the same string as ODDS_API_SPORT_KEY; this is the app's own label.
SPORT = "NFL"

url = f"https://api.the-odds-api.com/v4/sports/{ODDS_API_SPORT_KEY}/odds"
params = {
    "apiKey":     ODDS_API_KEY,
    "regions":    "us",   # single region — matches region="us" lookups in data_sources.py
    "markets":    ",".join(markets),
    "oddsFormat": "american",
}


def run_pull():
    print(f"Fetching NFL odds ({ODDS_API_SPORT_KEY})...")
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print("Error:", resp.status_code, resp.text)
        return
    data = resp.json()
    if not data:
        print("No games returned from Odds API. Season may be off or no lines posted yet.")
        return

    rows_inserted = 0
    for market_key in markets:
        snapshot_id = str(uuid.uuid4())
        supabase.table("odds_snapshots").insert({
            "id":        snapshot_id,
            "market":    market_key,
            "pulled_at": datetime.now(timezone.utc).isoformat(),
            "payload":   data,
            "sport":     SPORT,
            "region":    "us",
        }).execute()

        for game in data:
            event_id      = game["id"]
            commence_time = game["commence_time"]
            home_team     = game.get("home_team")
            away_team     = game.get("away_team")

            for book in game.get("bookmakers", []):
                book_key = book["key"]
                for market in book.get("markets", []):
                    if market["key"] != market_key:
                        continue
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name")
                        if name in ["Over", "Under"]:
                            side = name.lower()
                        elif name == home_team:
                            side = "home"
                        elif name == away_team:
                            side = "away"
                        else:
                            side = None

                        line  = outcome.get("point")
                        price = outcome.get("price")

                        supabase.table("odds_lines").insert({
                            "snapshot_id":   snapshot_id,
                            "event_id":      event_id,
                            "commence_time": commence_time,
                            "home_team":     home_team,
                            "away_team":     away_team,
                            "book":          book_key,
                            "sport":         SPORT,
                            "market":        market_key,
                            "side":          side,
                            "line":          line,
                            "price":         price,
                        }).execute()
                        rows_inserted += 1

    print(f"Done. Inserted {rows_inserted} rows across {len(markets)} snapshots.")
    print(f"Games pulled: {len(data)}")


if __name__ == "__main__":
    run_pull()
