# fetch_odds_nfl.py
import os
import time
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
BATCH_SIZE = 500

# ── Retry/backoff for the Odds API request ──
REQUEST_TIMEOUT = 15        # seconds, per attempt
MAX_ATTEMPTS = 4            # total attempts, including the first — not "4 retries"
BACKOFF_BASE_SECONDS = 1.0  # doubles each retry: 1s, 2s, 4s, ...
BACKOFF_MAX_SECONDS = 30.0  # cap on any single computed delay
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _parse_retry_after(resp):
    """Defensively parse a Retry-After header (seconds or HTTP-date not
    supported — only the common numeric-seconds form). Returns None if
    absent or unparseable, so the caller falls back to normal backoff."""
    raw = resp.headers.get("Retry-After") if resp is not None else None
    if not raw:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(seconds, BACKOFF_MAX_SECONDS)


def _fetch_odds_with_retries(url: str, params: dict, label: str):
    """
    GETs the Odds API with bounded retries for transient failures only:
    network errors, timeouts, HTTP 429, and 5xx. Permanent 4xx errors
    (bad key, bad request, etc.) are returned immediately without retry,
    same as the prior non-retrying behavior. Returns the Response on
    success or on a non-retryable status; returns None if every attempt
    was exhausted on a transient failure.
    """
    last_resp = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as e:
            print(f"Request error for {label} (attempt {attempt}/{MAX_ATTEMPTS}): {type(e).__name__}: {e}")
            if attempt == MAX_ATTEMPTS:
                print(f"Giving up on {label} after {MAX_ATTEMPTS} attempts — skipping this sport key.")
                return None
            delay = min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)
            time.sleep(delay)
            continue

        last_resp = resp
        if resp.status_code == 200:
            return resp

        if resp.status_code not in RETRYABLE_STATUS_CODES:
            # Permanent client error (e.g. 401/403/400) — do not retry.
            return resp

        if attempt == MAX_ATTEMPTS:
            print(
                f"Giving up on {label} after {MAX_ATTEMPTS} attempts — "
                f"last status {resp.status_code}: {resp.text}"
            )
            return None

        retry_after = _parse_retry_after(resp)
        delay = retry_after if retry_after is not None else min(
            BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS
        )
        print(
            f"Transient error for {label} (attempt {attempt}/{MAX_ATTEMPTS}): "
            f"status {resp.status_code} — retrying in {delay:.1f}s"
        )
        time.sleep(delay)

    return last_resp


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
        resp = _fetch_odds_with_retries(url, params, label=odds_api_sport_key)
        if resp is None or resp.status_code != 200:
            if resp is not None:
                print(f"Error for {odds_api_sport_key}:", resp.status_code, resp.text)
            continue
        data = resp.json()
        if not data:
            print(f"No games returned for {odds_api_sport_key}.")
            continue
        total_games += len(data)

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

            lines = []
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

                            lines.append({
                                "snapshot_id":   snapshot_id,
                                "event_id":      event_id,
                                "commence_time": commence_time,
                                "home_team":     home_team,
                                "away_team":     away_team,
                                "book":          book_key,
                                "sport":         SPORT,
                                "market":        market_key,
                                "side":          side,
                                "line":          outcome.get("point"),
                                "price":         outcome.get("price"),
                            })

            for i in range(0, len(lines), BATCH_SIZE):
                batch = lines[i:i + BATCH_SIZE]
                supabase.table("odds_lines").insert(batch).execute()
                total_rows += len(batch)

    print(f"Done. Inserted {total_rows} rows across {len(ODDS_API_SPORT_KEYS) * len(markets)} snapshots.")
    print(f"Total games pulled across both sport keys: {total_games}")


if __name__ == "__main__":
    if not ODDS_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
        print("Missing ODDS_API_KEY, SUPABASE_URL, or SUPABASE_KEY environment variables.")
    else:
        run_pull()
