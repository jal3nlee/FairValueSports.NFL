# app.py
import os, math
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client, Client
from pathlib import Path
from PIL import Image

# =======================
# Auth (Supabase)
# =======================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Auth not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY in your environment.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ===== Branding =====
ROOT = Path(__file__).parent.resolve()
ASSET_DIRS = [ROOT / "assets", ROOT / ".streamlit" / "assets"]

def find_asset(name: str):
    for d in ASSET_DIRS:
        p = d / name
        if p.is_file():
            return p
    return None

def newest_favicon():
    cands = []
    for d in ASSET_DIRS:
        if d.is_dir():
            cands += list(d.glob("favicon*.png"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

LOGO_PATH = find_asset("logo.png")
FAVICON_PATH = newest_favicon()

favicon_img = None
if FAVICON_PATH:
    try:
        favicon_img = Image.open(FAVICON_PATH)
    except Exception:
        favicon_img = None

st.set_page_config(
    page_title="Fair Value Betting",
    page_icon=(favicon_img if favicon_img else "🏈"),
    layout="wide",
    initial_sidebar_state="expanded",
)

HEADER_W = 560
SIDEBAR_W = 320

if LOGO_PATH:
    st.image(str(LOGO_PATH), width=HEADER_W)

# ---- Session state priming ----
st.session_state.setdefault("sb_session", None)
st.session_state.setdefault("sb_access_token", None)
st.session_state.setdefault("sb_refresh_token", None)
st.session_state.setdefault("show_auth", False)

authed = st.session_state.sb_session is not None

# =======================
# NFL config
# =======================
EASTERN = ZoneInfo("America/New_York")

def american_to_implied_prob(odds):
    try:
        o = int(odds)
    except Exception:
        return None
    return 100/(o+100) if o > 0 else (-o)/(-o+100)

def american_to_decimal(odds):
    o = int(odds)
    return 1 + (o/100 if o > 0 else 100/(-o))

def expected_value_pct(true_prob: float, american_odds: int) -> float:
    d = american_to_decimal(american_odds)
    return 100.0 * (true_prob * (d - 1.0) - (1.0 - true_prob))

def kelly_fraction(true_prob: float, american_odds: int) -> float:
    d = american_to_decimal(american_odds)
    b = d - 1.0
    p = float(true_prob); q = 1.0 - p
    if b <= 0:
        return 0.0
    return max(0.0, (b*p - q) / b)

def devig_two_way(p_a: float, p_b: float):
    a = (p_a or 0.0); b = (p_b or 0.0); s = a + b
    if s <= 0:
        return None, None
    return a/s, b/s

def parse_iso_dt_utc(iso_s: str):
    try:
        return datetime.fromisoformat(str(iso_s).replace("Z","+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

def fmt_date_et_str(iso_s: str):
    dt = parse_iso_dt_utc(iso_s)
    if not dt:
        return None
    et = dt.astimezone(EASTERN)
    dow = et.strftime("%a")
    md  = f"{et.month}/{et.day}"
    tm  = et.strftime("%I:%M %p").lstrip("0")
    return f"{dow} {md} {tm} ET"

# =======================
# Supabase odds readers
# =======================
PAGE_SIZE = 1000

@st.cache_data(ttl=60, show_spinner=False)
def get_latest_snapshot_meta(sport: str, market: str, region: str = "us"):
    """
    Returns (snapshot_id, pulled_at_iso) for the latest row in odds_snapshots
    for (sport, market, region).
    """
    try:
        res = supabase.table("odds_snapshots") \
            .select("id,pulled_at") \
            .eq("sport", sport) \
            .eq("market", market) \
            .eq("region", region) \
            .order("pulled_at", desc=True) \
            .limit(1) \
            .execute()
        data = res.data or []
        if not data:
            return None, None
        row = data[0]
        return row["id"], row.get("pulled_at")
    except Exception:
        return None, None

@st.cache_data(ttl=60, show_spinner=False)
def get_lines_for_snapshot(snapshot_id: str):
    """Fetch all normalized rows from odds_lines for a given snapshot_id."""
    rows, start = [], 0
    while True:
        page = supabase.table("odds_lines") \
            .select("event_id,home_team,away_team,commence_time,book,market,side,line,price") \
            .eq("snapshot_id", snapshot_id) \
            .range(start, start + PAGE_SIZE - 1) \
            .execute()
        chunk = page.data or []
        rows.extend(chunk)
        if len(chunk) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return pd.DataFrame(rows)

def fetch_market_lines(sport_keys: set[str], market_db: str):
    """
    market_db in {'moneyline','spread','total'}.
    Returns (df_lines_all_sports, latest_pulled_at_list)
    """
    all_lines = []
    pulled_ats = []
    for sport in sorted(sport_keys):
        snap_id, pulled_at = get_latest_snapshot_meta(sport, market_db, region="us")
        if not snap_id:
            continue
        df = get_lines_for_snapshot(snap_id)
        if not df.empty:
            all_lines.append(df)
        if pulled_at:
            pulled_ats.append(pulled_at)
    if all_lines:
        return pd.concat(all_lines, ignore_index=True), pulled_ats
    return pd.DataFrame(), pulled_ats

# =======================
# Main app
# =======================
def run_app():
    st.title("NFL Expected Value Model")

    # Example fetch: moneyline
    df_ml_lines, pulled_ml = fetch_market_lines({"NFL"}, "moneyline")

    if df_ml_lines.empty:
        st.info("No data available.")
    else:
        st.dataframe(df_ml_lines.head(20))

# ---- Run app ----
if __name__ == "__main__":
    run_app()
