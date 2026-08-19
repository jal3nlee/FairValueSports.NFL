# fetch_odds_nfl.py
import os
import time
import uuid
import requests
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client

from core.odds_math import parse_iso_dt_utc, EASTERN

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

# ── Dynamic refresh cadence ──
# Odds refresh frequency depends on how close/active NFL games are. Values
# are picked at the conservative end of the requested ranges (3 min, not 2;
# 30 min, not 15) to limit Odds API usage during less time-sensitive states —
# the live-window requirement (60s) is fixed and drives most of the quota
# cost regardless of the other three values.
LIVE_INTERVAL_SECONDS = 60
PREGAME_INTERVAL_SECONDS = 180    # 3 minutes
GAMEDAY_INTERVAL_SECONDS = 300    # 5 minutes
OFFPEAK_INTERVAL_SECONDS = 1800   # 30 minutes

# Game-window definition, per game, built from its scheduled commence_time
# alone (no live-score source consulted — see module docstring below).
LIVE_PRE_KICKOFF_BUFFER = timedelta(minutes=15)   # lines move fast just before kickoff
LIVE_POST_KICKOFF_WINDOW = timedelta(hours=4)     # covers a full game + OT with margin
PREGAME_WINDOW = timedelta(hours=2)                # "approaching" = within ~2h of kickoff

_STATE_PRIORITY = {"live": 0, "pregame": 1, "gameday": 2}
_STATE_INTERVAL = {
    "live": LIVE_INTERVAL_SECONDS,
    "pregame": PREGAME_INTERVAL_SECONDS,
    "gameday": GAMEDAY_INTERVAL_SECONDS,
}


def _classify_game_state(now_utc: datetime, commence_utc: datetime):
    """
    Classifies a single game's relationship to `now_utc` using only its
    scheduled kickoff time (no separate live-score lookup — see note on
    _determine_cadence_seconds). Returns None if the game isn't relevant
    to the current moment at all (not today, not upcoming soon).
    """
    live_start = commence_utc - LIVE_PRE_KICKOFF_BUFFER
    live_end = commence_utc + LIVE_POST_KICKOFF_WINDOW
    pregame_start = commence_utc - PREGAME_WINDOW

    if live_start <= now_utc <= live_end:
        return "live"
    if pregame_start <= now_utc < live_start:
        return "pregame"
    if now_utc.astimezone(EASTERN).date() == commence_utc.astimezone(EASTERN).date():
        return "gameday"
    return None


def _determine_cadence_seconds(now_utc: datetime, game_times_utc: list):
    """
    Picks the refresh interval for right now, given the set of known game
    kickoff times (drawn from previously-ingested Odds API payloads already
    stored in Supabase — see _get_known_game_times). Whichever game is in
    the most urgent state wins (live > pregame > gameday). If no games are
    known for today or soon, cadence is "off-peak". If no game-time data is
    available at all (e.g. very first run), returns (None, "unknown") so
    the caller treats this as "not enough info — fetch now" rather than
    blocking ingestion.

    This uses scheduled kickoff time + an assumed game-duration window
    rather than a live-score feed, per guidance that approximate,
    kickoff-time-based windows are an acceptable substitute for exact live
    status — it keeps this script free of any new dependency (no Streamlit,
    no additional external API) and avoids coupling the standalone ingestion
    script to the app's live-score module.
    """
    if not game_times_utc:
        return None, "unknown"

    best_state = None
    for commence_utc in game_times_utc:
        state = _classify_game_state(now_utc, commence_utc)
        if state is None:
            continue
        if best_state is None or _STATE_PRIORITY[state] < _STATE_PRIORITY[best_state]:
            best_state = state

    if best_state is None:
        return OFFPEAK_INTERVAL_SECONDS, "off-peak"
    return _STATE_INTERVAL[best_state], best_state


def _get_last_pull_time(supabase):
    """Timestamp of the most recent successful snapshot write, or None if
    unknown/unavailable. A read failure here must never block ingestion —
    it just means the throttle is bypassed and a fetch proceeds."""
    try:
        res = (
            supabase.table("odds_snapshots")
            .select("pulled_at")
            .eq("sport", SPORT)
            .order("pulled_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return None
        return parse_iso_dt_utc(rows[0].get("pulled_at"))
    except Exception as e:
        print(f"Could not determine last pull time (defaulting to fetch now): {type(e).__name__}: {e}")
        return None


def _get_known_game_times(supabase):
    """
    Distinct game kickoff times drawn from the most recently stored Odds
    API payloads (already-ingested data — no new external call). Covers
    both sport keys since each fetch cycle writes snapshot rows for both
    in quick succession. Returns [] on any failure or if nothing has been
    ingested yet, which the caller treats as "unknown state — fetch now".
    """
    try:
        res = (
            supabase.table("odds_snapshots")
            .select("payload,pulled_at")
            .eq("sport", SPORT)
            .order("pulled_at", desc=True)
            .limit(6)
            .execute()
        )
        rows = res.data or []
    except Exception as e:
        print(f"Could not load known game schedule (defaulting to fetch now): {type(e).__name__}: {e}")
        return []

    times = set()
    for row in rows:
        for game in (row.get("payload") or []):
            ct = parse_iso_dt_utc(game.get("commence_time"))
            if ct:
                times.add(ct)
    return sorted(times)


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
    now_utc = datetime.now(timezone.utc)
    last_pulled = _get_last_pull_time(supabase)
    game_times = _get_known_game_times(supabase)
    interval_seconds, cadence_state = _determine_cadence_seconds(now_utc, game_times)

    if interval_seconds is not None and last_pulled is not None:
        elapsed = (now_utc - last_pulled).total_seconds()
        if elapsed < interval_seconds:
            print(
                f"Skipping fetch — cadence state is '{cadence_state}' (target interval "
                f"{interval_seconds}s), only {elapsed:.0f}s since last successful pull "
                f"at {last_pulled.isoformat()}."
            )
            return

    print(
        f"Cadence state: {cadence_state} "
        f"(target interval: {interval_seconds if interval_seconds is not None else 'n/a — no schedule data yet'}s)"
    )

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
