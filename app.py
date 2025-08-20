# app.py
import os, math, requests
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client, Client
from pathlib import Path

# ===== BRANDING (supports assets/ and .streamlit/assets) =====
from pathlib import Path
from PIL import Image
import streamlit as st

ROOT = Path(__file__).parent.resolve()
ASSET_DIRS = [ROOT / "assets", ROOT / ".streamlit" / "assets"]

def find_asset(name: str) -> Path | None:
    for d in ASSET_DIRS:
        p = d / name
        if p.is_file():
            return p
    return None

LOGO_PATH = find_asset("logo.png")
FAVICON_PATH = find_asset("favicon.png")

favicon_img = None
if FAVICON_PATH:
    try:
        favicon_img = Image.open(FAVICON_PATH)
    except Exception:
        favicon_img = None

# MUST be first Streamlit call
st.set_page_config(
    page_title="Fair Value Sports • v4",
    page_icon=(favicon_img if favicon_img else "🏈"),
    layout="wide",
)

# Show logos (fallback if missing)
if Path(LOGO_PATH).is_file():
    st.image(str(LOGO_PATH), width=400)   # was 200

with st.sidebar:
    if Path(LOGO_PATH).is_file():
        st.image(str(LOGO_PATH), width=240)  # was 160
    else:
        st.write("Fair Value Sports")
# ===== END BRANDING =====

# =======================
# Auth (Supabase)
# =======================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Auth not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY in your environment.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

if "sb_session" not in st.session_state:
    st.session_state.sb_session = None

def auth_view():
    st.title("Fair Value Sports")
    tabs = st.tabs(["Sign in", "Create account"])

    with tabs[0]:
        email = st.text_input("Email", key="signin_email")
        password = st.text_input("Password", type="password", key="signin_pw")
        if st.button("Sign in"):
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.sb_session = res.session
                st.experimental_rerun()
            except Exception:
                st.error("Sign-in failed. Check your email/password or verify your email.")

    with tabs[1]:
        email2 = st.text_input("Email", key="signup_email")
        pw2 = st.text_input("Password", type="password", key="signup_pw")
        if st.button("Create account"):
            try:
                supabase.auth.sign_up({"email": email2, "password": pw2})
                st.success("Account created. Check your email to verify, then sign in.")
            except Exception:
                st.error("Sign-up failed. Try a different email or password.")

def require_auth():
    if st.session_state.sb_session is None:
        auth_view()
        st.stop()

# =======================
# NFL app config
# =======================
API_BASE = "https://api.the-odds-api.com/v4"
API_KEY  = os.getenv("ODDS_API_KEY", "")
EASTERN = ZoneInfo("America/New_York")

# ---------- helpers ----------
def american_to_implied_prob(odds):
    try: o = int(odds)
    except Exception: return None
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
    if b <= 0: return 0.0
    return max(0.0, (b*p - q) / b)

def devig_two_way(p_home: float, p_away: float):
    a = (p_home or 0.0); b = (p_away or 0.0); s = a + b
    if s <= 0: return None, None
    return a/s, b/s

def parse_iso_dt_utc(iso_s: str):
    try: return datetime.fromisoformat(iso_s.replace("Z","+00:00")).astimezone(timezone.utc)
    except Exception: return None

def fmt_date_est_str(iso_s: str, snap_odd_minutes: bool = True):
    dt = parse_iso_dt_utc(iso_s)
    if not dt: return None
    et = dt.astimezone(EASTERN)
    if snap_odd_minutes:
        if et.minute == 1:  et = et - timedelta(minutes=1)
        elif et.minute == 59: et = et + timedelta(minutes=1)
    dow = et.strftime("%a")
    md  = f"{et.month}/{et.day}"
    tm  = et.strftime("%I:%M %p").lstrip("0")
    return f"{dow} {md} {tm} ET"

# ---- Hard-coded NFL weeks (Week 1 = Sep 4 00:00 UTC; Week 0 = before that) ----
def nfl_week1_kickoff_thursday_utc(year: int) -> datetime:
    return datetime(year, 9, 4, 0, 0, 0, tzinfo=timezone.utc)

def nfl_week_window_utc(week_index: int, now_utc: datetime):
    wk1 = nfl_week1_kickoff_thursday_utc(now_utc.year)
    if week_index == 0:
        start = datetime(now_utc.year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end   = wk1 - timedelta(seconds=1)
        return start, end
    start = wk1 + timedelta(days=7*(week_index-1))
    end   = start + timedelta(days=5, hours=23, minutes=59, seconds=59)
    return start, end

def infer_current_week_index(now_utc: datetime) -> int:
    wk1 = nfl_week1_kickoff_thursday_utc(now_utc.year)
    if now_utc < wk1: return 0
    weeks = (now_utc - wk1).days // 7 + 1
    return max(1, min(19, weeks))

def sport_key_for_week(week_index: int) -> str:
    return "americanfootball_nfl_preseason" if week_index == 0 else "americanfootball_nfl"

def pretty_book_title(bm: dict) -> str:
    title = bm.get("title")
    if title: return title
    key = (bm.get("key") or "").replace("_", " ").title()
    return key or "Sportsbook"

def build_market(events, selected_books=None):
    rows = []
    for ev in events:
        home, away = ev.get("home_team"), ev.get("away_team")
        eid, t0 = ev.get("id"), ev.get("commence_time")
        for bm in ev.get("bookmakers", []):
            if selected_books and bm.get("key") not in selected_books: continue
            price_home, price_away = None, None
            for mk in bm.get("markets", []):
                if mk.get("key") == "h2h":
                    side_map = {o.get("name",""): o.get("price") for o in mk.get("outcomes", [])}
                    price_home = side_map.get(home); price_away = side_map.get(away)
                    break
            if price_home is None and price_away is None: continue
            rows.append({
                "event_id": eid, "home_team": home, "away_team": away,
                "book": pretty_book_title(bm),
                "home_price": price_home, "away_price": price_away,
                "commence_time": t0
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def compute_consensus_fair_probs(df_evt_books: pd.DataFrame):
    if df_evt_books.empty: return pd.DataFrame()
    tmp = df_evt_books.copy()
    tmp["home_imp_vig"] = tmp["home_price"].apply(american_to_implied_prob)
    tmp["away_imp_vig"] = tmp["away_price"].apply(american_to_implied_prob)
    agg = tmp.groupby(["event_id","home_team","away_team"]).agg(
        home_imp_vig=("home_imp_vig","mean"),
        away_imp_vig=("away_imp_vig","mean"),
        commence_time=("commence_time","first")
    ).reset_index()
    agg[["home_fair","away_fair"]] = agg.apply(
        lambda r: pd.Series(devig_two_way(r["home_imp_vig"], r["away_imp_vig"])),
        axis=1
    )
    return agg

def best_prices(df_evt_books: pd.DataFrame):
    if df_evt_books.empty:
        return pd.DataFrame(columns=["event_id","home_team","away_team","home_book","home_price","away_book","away_price"])
    home_best = df_evt_books.dropna(subset=["home_price"]).sort_values(
        ["event_id","home_price"], ascending=[True, False]
    ).groupby(["event_id","home_team","away_team"]).first().reset_index()[
        ["event_id","home_team","away_team","book","home_price"]
    ].rename(columns={"book":"home_book"})
    away_best = df_evt_books.dropna(subset=["away_price"]).sort_values(
        ["event_id","away_price"], ascending=[True, False]
    ).groupby(["event_id","home_team","away_team"]).first().reset_index()[
        ["event_id","home_team","away_team","book","away_price"]
    ].rename(columns={"book":"away_book"})
    return pd.merge(home_best, away_best, on=["event_id","home_team","away_team"], how="outer")

# =======================
# Main app
# =======================
def run_app():
    if not API_KEY:
        st.error("Missing ODDS_API_KEY environment variable."); st.stop()

    st.title("NFL Market EV Model")
    region = "us"

    # Window dropdown
    now = datetime.now(timezone.utc)
    current_week = infer_current_week_index(now)
    window_choice = st.selectbox("Window", ["Today", f"Week {current_week}"], index=1)

    if window_choice == "Today":
        window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        window_end   = window_start + timedelta(hours=23, minutes=59, seconds=59)
        sport_key    = sport_key_for_week(0 if current_week == 0 else current_week)
        caption_label = f"Today ({window_start.strftime('%Y-%m-%d')} UTC)"
    else:
        week_index   = current_week
        window_start, window_end = nfl_week_window_utc(week_index, now)
        sport_key    = sport_key_for_week(week_index)
        caption_label = ("Week 0 (before Sep 4) — "
                         f"{window_start.strftime('%Y-%m-%d')} → {window_end.strftime('%Y-%m-%d')} UTC") if week_index == 0 \
                        else (f"NFL Week {week_index} — {window_start.strftime('%Y-%m-%d')} → {window_end.strftime('%Y-%m-%d')} UTC")

    # Inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        weekly_bankroll = st.number_input("Weekly Bankroll ($)", min_value=0.0, value=1000.0, step=50.0)
    with c2:
        kelly_factor = st.slider("Kelly Factor (0.0–1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    with c3:
        min_ev = st.number_input("Minimum EV% to display", value=0.0, step=0.5)

    show_all = st.checkbox("Show all games (ignore EV% filter)", value=False)

    # Fetch odds
    params = {"apiKey": API_KEY, "regions": region, "markets": "h2h", "oddsFormat": "american"}
    try:
        events = requests.get(f"{API_BASE}/sports/{sport_key}/odds", params=params, timeout=30).json()
    except Exception as e:
        st.error(f"API error: {e}"); st.stop()

    if not isinstance(events, list) or not events:
        st.warning("No events returned."); st.stop()

    # Filter to chosen window
    events = [ev for ev in events if (dt := parse_iso_dt_utc(ev.get("commence_time"))) and window_start <= dt <= window_end]
    if not events:
        st.info(f"No NFL games in the selected window ({caption_label})."); st.stop()

    # Build -> devig -> best
    df_books = build_market(events, selected_books=None)
    if df_books.empty:
        st.warning("No odds available in this window."); st.stop()

    cons = compute_consensus_fair_probs(df_books)
    bests = best_prices(df_books)
    merged = pd.merge(bests, cons, on=["event_id","home_team","away_team"], how="inner")

    # Rows
    rows = []
    for _, r in merged.iterrows():
        date_str = fmt_date_est_str(r.get("commence_time"))
        game_label = f"{r['home_team']} vs {r['away_team']}"

        # Home
        if pd.notna(r.get("home_price")) and pd.notna(r.get("home_fair")):
            fair_p, price = float(r["home_fair"]), int(r["home_price"])
            ev_pct = expected_value_pct(fair_p, price)
            kelly = kelly_fraction(fair_p, price)
            rows.append({
                "Date": date_str, "Game": game_label, "Pick": r["home_team"],
                "Best Odds": price, "Best Book": r.get("home_book"),
                "Implied Probability": fair_p, "EV%": ev_pct, "Kelly %": kelly*100.0,
                "Stake ($)": round(weekly_bankroll * kelly_factor * kelly, 2)
            })

        # Away
        if pd.notna(r.get("away_price")) and pd.notna(r.get("away_fair")):
            fair_p, price = float(r["away_fair"]), int(r["away_price"])
            ev_pct = expected_value_pct(fair_p, price)
            kelly = kelly_fraction(fair_p, price)
            rows.append({
                "Date": date_str, "Game": game_label, "Pick": r["away_team"],
                "Best Odds": price, "Best Book": r.get("away_book"),
                "Implied Probability": fair_p, "EV%": ev_pct, "Kelly %": kelly*100.0,
                "Stake ($)": round(weekly_bankroll * kelly_factor * kelly, 2)
            })

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No sides available."); st.stop()

    if not show_all:
        df = df[df["EV%"] >= float(min_ev)].reset_index(drop=True)
        if df.empty:
            st.info("No games meet the EV% filter for this window."); st.stop()

    df = df.sort_values(["EV%","Implied Probability"], ascending=[False, False]).reset_index(drop=True)

    # Pretty display
    def fmt_odds(o):
        try:
            o = int(o)
            return f"+{o}" if o > 0 else str(o)
        except Exception:
            return str(o)

    show = df.copy()
    show["Best Odds"] = show["Best Odds"].map(fmt_odds)
    show["Best Book"] = show["Best Book"].astype(str)
    show["Implied Probability"] = show["Implied Probability"].map(lambda x: f"{x*100:.1f}%")
    show["Kelly %"] = show["Kelly %"].map(lambda x: f"{x:.2f}")

    st.subheader("Games & EV")
    st.caption(f"Window: {caption_label}")
    st.dataframe(
        show[["Date","Game","Pick","Best Odds","Best Book","Implied Probability","EV%","Kelly %","Stake ($)"]],
        use_container_width=True, hide_index=True,
    )

    total_stake = float(df["Stake ($)"].sum())
    util_pct = 100.0 * (total_stake / weekly_bankroll) if weekly_bankroll > 0 else 0.0
    color = "green" if util_pct < 50 else ("orange" if util_pct < 70 else "red")
    st.markdown(
        f"**Total Suggested Stake:** ${total_stake:,.2f}  |  **Utilization:** "
        f"<span style='color:{color}'>{util_pct:.1f}% of weekly bankroll</span>",
        unsafe_allow_html=True
    )

# ===== Require login, then run app =====
require_auth()

with st.sidebar:
    st.write("Logged in")
    if st.button("Log out"):
        try:
            supabase.auth.sign_out()
        except:
            pass
        st.session_state.sb_session = None
        st.experimental_rerun()

run_app()
