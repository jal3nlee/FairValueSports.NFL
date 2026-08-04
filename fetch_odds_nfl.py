# fetch_odds_nfl.py
import os
import uuid
import requests
from datetime import datetime, timezone
from supabase import create_client, Client

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

markets = ["h2h", "spreads", "totals"]

# The Odds API treats preseason as a separate sport entirely from
# regular season — pull both, write them into the same internal "NFL" bucket.
ODDS_API_SPORT_KEYS = ["americanfootball_nfl", "americanfootball_nfl_preseason"]
SPORT = "NFL"


def run_pull():
    total_rows = 0
    total_games = 0
    for odds_api_sport_key in ODDS_API_SPORT_KEYS:
        print(f"Fetching NFL odds ({odds_api_sport_key})...")
        url = f"https://api.the-odds-api.com/v4/sports/{odds_api_sport_key}/odds"
        params = {
            "apiKey":     ODDS_API_KEY,
            "regions":    "us",
            "markets":    ",".join(markets),
            "oddsFormat": "american",
        }
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print(f"Error for {odds_api_sport_key}:", resp.status_code, resp.text)
            continue
        data = resp.json()
        if not data:
            print(f"No games returned for {odds_api_sport_key}.")
            continue
        total_games += len(data)

        # ── Temporary debug: show exactly what came back for preseason ──
        if odds_api_sport_key == "americanfootball_nfl_preseason":
            for g in data:
                print(
                    f"  preseason game: {g.get('away_team')} @ {g.get('home_team')} — "
                    f"{g.get('commence_time')} — {len(g.get('bookmakers', []))} books"
                )

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
                            total_rows += 1

    print(f"Done. Inserted {total_rows} rows across {len(ODDS_API_SPORT_KEYS) * len(markets)} snapshots.")
    print(f"Total games pulled across both sport keys: {total_games}")


if __name__ == "__main__":
    if not ODDS_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
        print("Missing ODDS_API_KEY, SUPABASE_URL, or SUPABASE_KEY environment variables.")
    else:
        run_pull()
