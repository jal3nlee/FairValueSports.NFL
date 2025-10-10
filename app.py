# app.py
import os, math
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client, Client
from pathlib import Path
from PIL import Image
from streamlit_cookies_manager import EncryptedCookieManager

# =======================
# Auth (Supabase)
# =======================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Auth not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY in your environment.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Cookie manager setup ---
cookies = EncryptedCookieManager(prefix="fvb_", password="your-very-strong-secret-key")
if not cookies.ready():
    st.stop()

# --- Try to restore Supabase session from cookies ---
if "access_token" in cookies and "refresh_token" in cookies:
    try:
        supabase.auth.set_session(
            access_token=cookies["access_token"],
            refresh_token=cookies["refresh_token"]
        )
        st.session_state.sb_access_token = cookies["access_token"]
        st.session_state.sb_refresh_token = cookies["refresh_token"]
    except Exception as e:
        st.warning(f"Session reattach failed: {e}")

# --- When user logs in successfully ---
def save_session(session, remember=False):
    """Store Supabase session in both session_state and optionally cookies"""
    access_token = session["access_token"]
    refresh_token = session["refresh_token"]

    # Always save to Streamlit session
    st.session_state.sb_access_token = access_token
    st.session_state.sb_refresh_token = refresh_token

    # Save to cookies only if user checked "Remember me"
    if remember:
        cookies["access_token"] = access_token
        cookies["refresh_token"] = refresh_token
        cookies.save()

# --- Example login UI ---
with st.expander("Login", expanded=True):
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    remember_me = st.checkbox("Keep me logged in", value=True)

    if st.button("Login"):
        try:
            result = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if result and result.session:
                save_session(result.session, remember=remember_me)
                st.success("Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials. Try again.")
        except Exception as e:
            st.error(f"Login failed: {e}")

# --- Example logout ---
if st.button("Logout"):
    try:
        supabase.auth.sign_out()
        st.session_state.pop("sb_access_token", None)
        st.session_state.pop("sb_refresh_token", None)
        cookies.clear()
        cookies.save()
        st.success("Logged out successfully.")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Logout failed: {e}")


# ===== BRANDING =====
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

def _store_session(sess):
    st.session_state.sb_session = sess
    try:
        st.session_state.sb_access_token = getattr(sess, "access_token", None) or sess.get("access_token")
        st.session_state.sb_refresh_token = getattr(sess, "refresh_token", None) or sess.get("refresh_token")
    except Exception:
        pass

def _clear_session():
    st.session_state.sb_session = None
    st.session_state.sb_access_token = None
    st.session_state.sb_refresh_token = None

def _maybe_refresh_session():
    if st.session_state.sb_session is None and st.session_state.sb_refresh_token:
        try:
            res = supabase.auth.refresh_session()
            if res and getattr(res, "session", None):
                _store_session(res.session)
        except Exception:
            _clear_session()

_maybe_refresh_session()

def auth_view():
    tabs = st.tabs(["Sign in", "Create account"])
    with tabs[0]:
        with st.form("signin_form", clear_on_submit=False):
            email = st.text_input("Email", key="signin_email")
            password = st.text_input("Password", type="password", key="signin_pw")
            submit = st.form_submit_button("Sign in")
        if submit:
            try:
                try:
                    supabase.auth.sign_out()
                except Exception:
                    pass
                res = supabase.auth.sign_in_with_password({"email": (email or "").strip(), "password": password})
                sess = getattr(res, "session", None) or getattr(res, "session", {}) or None
                if not sess:
                    try:
                        res2 = supabase.auth.refresh_session()
                        sess = getattr(res2, "session", None) or getattr(res2, "session", {}) or None
                    except Exception:
                        pass
                if sess:
                    _store_session(sess)
                    st.success("Signed in.")
                    st.session_state.show_auth = False
                    st.rerun()
                else:
                    st.error("Sign-in succeeded but no session was returned. Please try again.")
            except Exception as e:
                msg = str(e)
                if "Invalid login credentials" in msg:
                    st.error("Sign-in failed: invalid email or password.")
                elif "Email not confirmed" in msg or "confirmed_at" in msg:
                    st.error("Please verify your email address, then sign in.")
                else:
                    st.error(f"Sign-in failed: {msg or 'Unknown error.'}")
    with tabs[1]:
        with st.form("signup_form", clear_on_submit=False):
            full_name = st.text_input("Name", key="signup_name")
            email2 = st.text_input("Email", key="signup_email")
            pw2 = st.text_input("Password", type="password", key="signup_pw")
            submit2 = st.form_submit_button("Create account")
        if submit2:
            if not full_name.strip():
                st.warning("Please enter your full name.")
            elif not email2 or not pw2:
                st.warning("Email and password are required.")
            else:
                try:
                    supabase.auth.sign_up(
                        {"email": email2.strip(), "password": pw2, "options": {"data": {"full_name": full_name.strip()}}}
                    )
                    st.success("Account created. Check your email to verify, then sign in.")
                except Exception as e:
                    st.error(f"Sign-up failed: {str(e) or 'Try a different email or password.'}")

authed = st.session_state.sb_session is not None

# --- Sidebar ---
with st.sidebar:
    if LOGO_PATH:
        st.image(str(LOGO_PATH), width=SIDEBAR_W)
    else:
        st.write("Fair Value Betting")

st.sidebar.divider()

# Account block
with st.sidebar:
    if authed:
        u = getattr(st.session_state.sb_session, "user", None)
        user_email = getattr(u, "email", None) or (getattr(u, "user_metadata", {}) or {}).get("email")
        st.success(f"Signed in{f' as {user_email}' if user_email else ''}.")
        if st.button("Log out"):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            _clear_session()
            st.rerun()
    else:
        st.info("Free full access in September—create a free account to unlock filters and sorting.")
        if st.button("Sign in / Create account"):
            st.session_state.show_auth = True

    if st.session_state.show_auth and not authed:
        with st.expander("Account", expanded=True):
            auth_view()

with st.sidebar.expander("How to use", expanded=False):
    st.markdown(
        """
1. **Pick a Window**: **Today**, **NFL Week X**, or **Next 7 Days**.
2. **Set inputs**:
   - **Weekly Bankroll ($)** — your total budget for the week.
   - **Kelly Factor (0–1)** — risk scaling (e.g., 0.5 = half Kelly).
   - **Minimum EV%** — filter picks above this expected value.
3. **Review the table**:
   - **Fair Win %** = de-vigged market consensus.
   - **EV%** = edge versus best available odds.
   - **Stake ($)** = recommended bet size based on Kelly & bankroll.
        """
    )

with st.sidebar.expander("Glossary", expanded=False):
    st.markdown(
        """
**EV% (Expected Value %)** — How favorable the offered price is versus the fair baseline (no-vig).  
**Kelly Factor** — Scales bet size to edge; 1.0 = full Kelly, 0.5 = half Kelly.
        """
    )

with st.sidebar.expander("Feedback", expanded=False):
    user = None
    try:
        user = getattr(st.session_state.get("sb_session", None), "user", None)
    except Exception:
        user = None

    if not user:
        st.info("You must be signed in to leave feedback.")
    else:
        with st.form("feedback_form", clear_on_submit=True):
            full_name = (user.user_metadata or {}).get("full_name") or (user.user_metadata or {}).get("name") or ""
            email_addr = getattr(user, "email", "") or (user.user_metadata or {}).get("email", "")
            st.markdown(f"**Submitting as:** {full_name or 'Unknown'}  \n**Email:** {email_addr or 'Unknown'}")
            feedback_text = st.text_area("Share your thoughts, ideas, or issues:")
            submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            txt = (feedback_text or "").strip()
            if not txt:
                st.warning("Please enter feedback before submitting.")
            else:
                try:
                    payload = {
                        "message": txt,
                        "name": full_name.strip() or None,
                        "email": (email_addr or "").strip() or None,
                        "user_id": user.id,
                    }
                    supabase.table("feedback").insert(payload).execute()
                    st.success("Thanks for your feedback!")
                except Exception as e:
                    st.error(f"Error saving feedback: {e}")

with st.sidebar.expander("Disclaimer", expanded=False):
    st.markdown(
        """
**Fair Value Betting** is for **education and entertainment** only — not financial or betting advice.
        """
    )

# =======================
# NFL app config
# =======================
EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")  # used for window math only

# ---------- helpers ----------
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

def fmt_date_et_str(iso_s: str, snap_odd_minutes: bool = True):
    dt = parse_iso_dt_utc(iso_s)
    if not dt:
        return None
    et = dt.astimezone(EASTERN)
    if snap_odd_minutes:
        if et.minute == 1:
            et = et - timedelta(minutes=1)
        elif et.minute == 59:
            et = et + timedelta(minutes=1)
    dow = et.strftime("%a")
    md  = f"{et.month}/{et.day}"
    tm  = et.strftime("%I:%M %p").lstrip("0")
    return f"{dow} {md} {tm} ET"

def thursday_after_labor_day_utc(year: int) -> datetime:
    """Thursday after Labor Day at 00:00 ET, converted to UTC."""
    d = datetime(year, 9, 1, tzinfo=EASTERN)
    while d.weekday() != 0:  # 0 = Monday
        d += timedelta(days=1)
    opener_et_midnight = (d + timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
    return opener_et_midnight.astimezone(timezone.utc)

def nfl_week_window_utc(week_index: int, now_utc: datetime):
    """Thu 00:00 ET → Tue 23:59:59 ET window, returned in UTC."""
    yr = now_utc.astimezone(EASTERN).year
    wk1 = thursday_after_labor_day_utc(yr)
    if now_utc < wk1:
        return 0
    if week_index == 0:
        start = datetime(yr, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end   = wk1 - timedelta(seconds=1)
        return start, end
    start = thursday_after_labor_day_utc(yr) + timedelta(days=7 * (week_index - 1))
    end   = start + timedelta(days=5, hours=23, minutes=59, seconds=59)
    return start, end

def infer_current_week_index(now_utc: datetime) -> int:
    """Return 0 before Week 1; otherwise clamp to 1..18 (regular season)."""
    yr = now_utc.astimezone(EASTERN).year
    wk1 = thursday_after_labor_day_utc(yr)
    if now_utc < wk1:
        return 0
    weeks = (now_utc - wk1).days // 7 + 1
    return max(1, min(18, weeks))

def sport_key_for_week(week_index: int) -> str:
    # All snapshots are saved as sport="NFL"
    return "NFL"

# =======================
# Supabase odds readers
# =======================
PAGE_SIZE = 1000

@st.cache_data(ttl=60, show_spinner=False)
def get_latest_snapshot_meta(sport: str, market: str, region: str = "us"):
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

# Map app labels → DB values
MARKET_MAP = {
    "moneyline": "h2h",
    "spread": "spreads",
    "total": "totals",
}

def fetch_market_lines(sport_keys: set[str], market_label: str):
    """
    sport_keys: {"NFL"} usually
    market_label: one of "moneyline", "spread", "total" (app labels)
    """
    all_lines = []
    pulled_ats = []

    db_market = MARKET_MAP.get(market_label, market_label)

    for sport in sorted(sport_keys):
        snap_id, pulled_at = get_latest_snapshot_meta(sport, db_market, region="us")
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
# Builders for each market
# =======================
def build_market_from_lines_moneyline(df_lines: pd.DataFrame) -> pd.DataFrame:
    if df_lines.empty: return pd.DataFrame()
    df = df_lines[(df_lines["market"]=="h2h") & (df_lines["side"].isin(["home","away"]))].copy()
    df["home_price"] = df.apply(lambda r: r["price"] if r["side"]=="home" else None, axis=1)
    df["away_price"] = df.apply(lambda r: r["price"] if r["side"]=="away" else None, axis=1)
    agg = df.groupby(["event_id","home_team","away_team","book","commence_time"], as_index=False).agg(
        home_price=("home_price","max"),
        away_price=("away_price","max"),
    )
    return agg

def build_market_from_lines_spread(df_lines: pd.DataFrame) -> pd.DataFrame:
    if df_lines.empty: return pd.DataFrame()
    df = df_lines[(df_lines["market"]=="spreads") & (df_lines["side"].isin(["home","away"]))].copy()
    df["home_price"] = df.apply(lambda r: r["price"] if r["side"]=="home" else None, axis=1)
    df["away_price"] = df.apply(lambda r: r["price"] if r["side"]=="away" else None, axis=1)
    agg = df.groupby(["event_id","home_team","away_team","book","commence_time","line"], as_index=False).agg(
        home_price=("home_price","max"),
        away_price=("away_price","max"),
    )
    return agg

def build_market_from_lines_total(df_lines: pd.DataFrame) -> pd.DataFrame:
    if df_lines.empty: return pd.DataFrame()
    df = df_lines[(df_lines["market"]=="totals") & (df_lines["side"].isin(["over","under"]))].copy()
    df["over_price"]  = df.apply(lambda r: r["price"] if r["side"]=="over"  else None, axis=1)
    df["under_price"] = df.apply(lambda r: r["price"] if r["side"]=="under" else None, axis=1)
    agg = df.groupby(["event_id","home_team","away_team","book","commence_time","line"], as_index=False).agg(
        over_price=("over_price","max"),
        under_price=("under_price","max"),
    ).rename(columns={"line":"total"})
    return agg

# =======================
# Consensus / Best price
# =======================
def compute_consensus_fair_probs_h2h(df_evt_books: pd.DataFrame):
    """
    Compute consensus fair win probabilities (no-vig) for moneylines.

    - Uses all available sportsbook lines for each event.
    - Converts odds → implied probabilities.
    - Averages across books.
    - Removes vig so the two sides sum to 1.
    """
    if df_evt_books.empty:
        return pd.DataFrame()

    df = df_evt_books.copy()

    # Convert American odds to implied probabilities (with vig)
    df["home_imp_vig"] = df["home_price"].apply(american_to_implied_prob)
    df["away_imp_vig"] = df["away_price"].apply(american_to_implied_prob)

    # Average across all books for each event
    agg = df.groupby(["event_id", "home_team", "away_team"]).agg(
        home_imp_vig=("home_imp_vig", "mean"),
        away_imp_vig=("away_imp_vig", "mean"),
        commence_time=("commence_time", "first")
    ).reset_index()

    # Remove vig — normalize probabilities so they sum to 1
    agg["home_fair"], agg["away_fair"] = zip(
        *agg.apply(lambda r: devig_two_way(r["home_imp_vig"], r["away_imp_vig"]), axis=1)
    )

    return agg

def best_prices_h2h(df_evt_books: pd.DataFrame):
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

def compute_consensus_fair_probs_spread(df_spreads: pd.DataFrame):
    if df_spreads.empty:
        return pd.DataFrame()

    tmp = df_spreads.copy()
    tmp["home_imp_vig"] = tmp["home_price"].apply(american_to_implied_prob)
    tmp["away_imp_vig"] = tmp["away_price"].apply(american_to_implied_prob)

    # Weight lines by how close odds are to even money (-110ish)
    tmp["home_dist"] = tmp["home_price"].apply(lambda o: abs(abs(int(o)) - 110) if pd.notna(o) else 999)
    tmp["away_dist"] = tmp["away_price"].apply(lambda o: abs(abs(int(o)) - 110) if pd.notna(o) else 999)

    # Anchor line = whichever side is closer to even money
    tmp["anchor_line"] = tmp.apply(
        lambda r: r["line"] if r["home_dist"] <= r["away_dist"] else -r["line"], axis=1
    )
    tmp["anchor_weight"] = tmp.apply(lambda r: 1.0 / (1.0 + min(r["home_dist"], r["away_dist"])), axis=1)

    agg = tmp.groupby(["event_id", "home_team", "away_team"]).apply(
        lambda g: pd.Series({
            "consensus_line": (g["anchor_line"] * g["anchor_weight"]).sum() / g["anchor_weight"].sum(),
            "home_imp_vig": g["home_imp_vig"].mean(),
            "away_imp_vig": g["away_imp_vig"].mean(),
            "commence_time": g["commence_time"].iloc[0]
        })
    ).reset_index()

    # At consensus line: break-even 50/50
    agg["home_fair"] = 0.5
    agg["away_fair"] = 0.5
    return agg

def best_prices_spread(df_spreads: pd.DataFrame):
    if df_spreads.empty:
        return pd.DataFrame()
    home_best = df_spreads.dropna(subset=["home_price"]).sort_values(
        ["event_id","line","home_price"], ascending=[True, True, False]
    ).groupby(["event_id","home_team","away_team","line"]).first().reset_index()[
        ["event_id","home_team","away_team","line","book","home_price"]
    ].rename(columns={"book":"home_book"})
    away_best = df_spreads.dropna(subset=["away_price"]).sort_values(
        ["event_id","line","away_price"], ascending=[True, True, False]
    ).groupby(["event_id","home_team","away_team","line"]).first().reset_index()[
        ["event_id","home_team","away_team","line","book","away_price"]
    ].rename(columns={"book":"away_book"})
    return pd.merge(home_best, away_best, on=["event_id","home_team","away_team","line"], how="outer")

def compute_consensus_fair_probs_totals(df_totals: pd.DataFrame):
    if df_totals.empty:
        return pd.DataFrame()

    tmp = df_totals.copy()
    tmp["over_imp_vig"] = tmp["over_price"].apply(american_to_implied_prob)
    tmp["under_imp_vig"] = tmp["under_price"].apply(american_to_implied_prob)

    tmp["over_dist"] = tmp["over_price"].apply(lambda o: abs(abs(int(o)) - 110) if pd.notna(o) else 999)
    tmp["under_dist"] = tmp["under_price"].apply(lambda o: abs(abs(int(o)) - 110) if pd.notna(o) else 999)

    tmp["anchor_total"] = tmp["total"]
    tmp["anchor_weight"] = tmp.apply(lambda r: 1.0 / (1.0 + min(r["over_dist"], r["under_dist"])), axis=1)

    agg = tmp.groupby(["event_id", "home_team", "away_team"]).apply(
        lambda g: pd.Series({
            "consensus_total": (g["anchor_total"] * g["anchor_weight"]).sum() / g["anchor_weight"].sum(),
            "over_imp_vig": g["over_imp_vig"].mean(),
            "under_imp_vig": g["under_imp_vig"].mean(),
            "commence_time": g["commence_time"].iloc[0]
        })
    ).reset_index()

    # At consensus total: break-even 50/50
    agg["over_fair"] = 0.5
    agg["under_fair"] = 0.5
    return agg

def best_prices_totals(df_totals: pd.DataFrame):
    if df_totals.empty:
        return pd.DataFrame()
    over_best = df_totals.dropna(subset=["over_price"]).sort_values(
        ["event_id","total","over_price"], ascending=[True, True, False]
    ).groupby(["event_id","home_team","away_team","total"]).first().reset_index()[
        ["event_id","home_team","away_team","total","book","over_price"]
    ].rename(columns={"book":"over_book"})
    under_best = df_totals.dropna(subset=["under_price"]).sort_values(
        ["event_id","total","under_price"], ascending=[True, True, False]
    ).groupby(["event_id","home_team","away_team","total"]).first().reset_index()[
        ["event_id","home_team","away_team","total","book","under_price"]
    ].rename(columns={"book":"under_book"})
    return pd.merge(over_best, under_best, on=["event_id","home_team","away_team","total"], how="outer")

# =======================
# Main app
# =======================
def run_app():
    tabs = st.tabs(["NFL Expected Value Model", "Best Odds by Sportsbook", "Parlay Builder"])

    with tabs[0]:
        if not authed:
            st.info("Preview Mode: showing today's top pick — **Sign in** to see all picks, filters, and sorting.")
    
        # --- Market Filter ---
        market_choice = st.radio(
            "Market",
            ["Moneyline", "Spread", "Total"],
            index=0,
            horizontal=True,
            help="Toggle between Moneyline, Spread, and Total markets.",
            key="market_choice"
        )
    
        # --- Window dropdown ---
        now_utc = datetime.now(timezone.utc)
        current_week = infer_current_week_index(now_utc)
        week_label = "NFL Preseason" if current_week == 0 else f"NFL Week {current_week}"
    
        window_options = ["Today", week_label, "Next 7 Days"]
        window_choice = st.selectbox(
            "Date Window",
            window_options,
            index=1,
            key="window_choice",
            help="Choose a time window. “Next 7 Days” shows games from today through the next full week.",
            disabled=False,  # keep visible to everyone
        )
    
        # Determine window
        def _short_day_md(dt_utc):
            local = dt_utc.astimezone(EASTERN)
            return f"{local.strftime('%a')} {local.month}/{local.day}"
    
        def window_next_7_days(now_utc, tz=EASTERN):
            local = now_utc.astimezone(tz)
            start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
            end_local = (start_local + timedelta(days=8)).replace(hour=23, minute=59, second=59, microsecond=0)
            return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)
    
        if window_choice == "Today":
            now_local = datetime.now(EASTERN)
            window_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
            window_end   = (window_start + timedelta(days=1)) - timedelta(seconds=1)
            sport_keys   = {sport_key_for_week(current_week)}
            caption_label = f"Today"
        elif window_choice == "Next 7 Days":
            window_start, window_end = window_next_7_days(now_utc, tz=EASTERN)
            sport_keys = {
                sport_key_for_week(infer_current_week_index(window_start)),
                sport_key_for_week(infer_current_week_index(window_end)),
            }
            caption_label = f"{_short_day_md(window_start)} – {_short_day_md(window_end)}"
        else:
            week_index   = current_week
            window_start, window_end = nfl_week_window_utc(week_index, now_utc)
            sport_keys   = {sport_key_for_week(week_index)}
            caption_label = f"{week_label} — {_short_day_md(window_start)} – {_short_day_md(window_end)}"
    
        # =======================
        # Fetch lines
        # =======================
    
        
        df_ml_lines,   pulled_ml  = fetch_market_lines(sport_keys, "moneyline")
        df_spread_lines, pulled_sp = fetch_market_lines(sport_keys, "spread")
        df_total_lines, pulled_tot = fetch_market_lines(sport_keys, "total")

        # Filter by window
        def filter_by_window_df(df_any: pd.DataFrame) -> pd.DataFrame:
            if df_any.empty: return df_any
            df = df_any.copy()
            df["__t0"] = df["commence_time"].apply(parse_iso_dt_utc)
            df = df[(df["__t0"] >= window_start) & (df["__t0"] <= window_end)]
            return df.drop(columns=["__t0"])
    
        df_ml_lines     = filter_by_window_df(df_ml_lines)
        df_spread_lines = filter_by_window_df(df_spread_lines)
        df_total_lines  = filter_by_window_df(df_total_lines)
    
        if df_ml_lines.empty and df_spread_lines.empty and df_total_lines.empty:
            st.info(f"No NFL games in the selected window ({caption_label}).")
            st.stop()
    
        # --- Sportsbook filter ---
        def _books_from(df: pd.DataFrame) -> set[str]:
            if df is None or df.empty or "book" not in df.columns:
                return set()
            return set(df["book"].dropna().astype(str).unique().tolist())
    
        all_books = sorted(_books_from(df_ml_lines) | _books_from(df_spread_lines) | _books_from(df_total_lines))
    
        if all_books:
            selected_books = st.multiselect(
                "Sportsbooks",
                options=all_books,
                default=all_books,
                help="Uncheck sportsbooks you don’t want to include in screening."
            )
        else:
            selected_books = []
    
        # --- Inputs ---
        st.subheader("Filters & Bankroll")
    
        r1c1, r1c2 = st.columns([1, 1])
        with r1c1:
            min_ev = st.slider(
                "Minimum Expected Value (%)",
                min_value=0.0, max_value=20.0, value=0.0, step=0.1,
                format="%0.1f%%",
                help="Filter out plays below this expected value.",
                disabled=not authed,
                key="min_ev",
            )
        with r1c2:
            fair_win_min = st.slider(
                "Minimum Fair Win Probability (%)",
                min_value=0.0, max_value=100.0, value=0.0, step=0.5,
                format="%0.1f%%",
                help="Hide picks with a fair (no-vig) win probability below this percentage.",
                disabled=not authed,
                key="min_fair_win_pct",
            )
    
        r2c1, r2c2 = st.columns([1, 1])
        with r2c1:
            weekly_bankroll = st.number_input(
                "Weekly Bankroll ($)",
                min_value=0.0, value=1000.0, step=50.0,
                help="Total budget for this week.",
                disabled=not authed,
                key="weekly_bankroll",
            )
        with r2c2:
            kelly_pct = st.slider(
                "Kelly Factor (%)",
                min_value=0.0, max_value=100.0, value=50.0, step=0.5,
                format="%0.1f%%",
                help="Controls risk factor. 50% = half Kelly, 100% = full Kelly.",
                disabled=not authed,
                key="kelly_factor",
            )
        kelly_factor = (kelly_pct / 100.0)
    
        show_all = st.toggle(
            "Show all matchups",
            value=False,
            help="Ignore EV% and Fair Win % filters.",
            disabled=not authed,
            key="show_all",
        )
    
        # Build book-level tables
        df_ml_books     = build_market_from_lines_moneyline(df_ml_lines)
        df_spread_books = build_market_from_lines_spread(df_spread_lines)
        df_total_books  = build_market_from_lines_total(df_total_lines)
    
        def _apply_book_filter(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            if not selected_books:
                return df.iloc[0:0].copy()
            return df[df["book"].isin(selected_books)].copy()
    
        df_ml_books     = _apply_book_filter(df_ml_books)
        df_spread_books = _apply_book_filter(df_spread_books)
        df_total_books  = _apply_book_filter(df_total_books)
    
        df_ml_cons   = compute_consensus_fair_probs_h2h(df_ml_books) if not df_ml_books.empty else pd.DataFrame()
        df_ml_best   = best_prices_h2h(df_ml_books) if not df_ml_books.empty else pd.DataFrame()      
        df_ml        = pd.merge(df_ml_best, df_ml_cons, on=["event_id","home_team","away_team"], how="inner") if (not df_ml_best.empty and not df_ml_cons.empty) else pd.DataFrame()
        
        # --- Spread handling ---
        df_sp_cons = pd.DataFrame()
        df_sp_best = pd.DataFrame()
        df_spread  = pd.DataFrame()
        
        def _adjust_spread_fair(row):
            """
            Adjust fair win probabilities based on how far the sportsbook's spread line
            differs from the market consensus line.
        
            Consensus line is for the home team (negative = favorite).
            Uses absolute-value comparison so +14 vs −14 are treated correctly.
            """
        
            if pd.isna(row.get("line")) or pd.isna(row.get("consensus_line")):
                return 0.5, 0.5  # baseline if missing
        
            line = float(row["line"])
            consensus = float(row["consensus_line"])
        
            # Use absolute values so +14 vs −14 aren't treated as 28 apart
            diff = abs(abs(line) - abs(consensus))
        
            # Base sensitivity (percent per point)
            step_per_point = 0.02
            # Increase sensitivity near key numbers
            if any(abs(round(consensus) - k) <= 0.5 for k in [3, 7]):
                step_per_point = 0.025
        
            # Apply a cap to avoid unrealistic extremes
            max_adj = 0.30
            adj = min(diff * step_per_point, max_adj)
        
            # Direction logic:
            # If sportsbook line gives more points to underdog (larger abs(line)),
            # home favorite’s fair win % should fall.
            # Direction: if sportsbook line is more negative (harder for the favorite),
            # lower home fair win %. If it's less negative (easier), raise it.
            # Direction logic: treat favorites (negative consensus) and underdogs (positive consensus) correctly
          # Direction logic: handle both favorite (negative) and underdog (positive) cases
            if consensus < 0:
                # Home team is favorite
                if abs(line) > abs(consensus):
                    # sportsbook line gives more points (harder to cover)
                    home_fair = 0.5 - adj
                else:
                    # sportsbook line gives fewer points (easier)
                    home_fair = 0.5 + adj
            else:
                # Home team is underdog
                if abs(line) > abs(consensus):
                    # sportsbook line gives more points (easier to cover)
                    home_fair = 0.5 + adj
                else:
                    # fewer points to dog (harder)
                    home_fair = 0.5 - adj

            away_fair = 1 - home_fair
        
            # Clamp to valid range
            home_fair = max(0.01, min(0.99, home_fair))
            away_fair = max(0.01, min(0.99, away_fair))
        
            return home_fair, away_fair


        if not df_spread_books.empty:
            df_sp_cons = compute_consensus_fair_probs_spread(df_spread_books)
            df_sp_best = best_prices_spread(df_spread_books)
        
            if not df_sp_best.empty and not df_sp_cons.empty:
                df_spread = pd.merge(
                    df_sp_best,
                    df_sp_cons,
                    on=["event_id", "home_team", "away_team"],
                    how="inner"
                )
                df_spread[["home_fair", "away_fair"]] = df_spread.apply(
                    _adjust_spread_fair, axis=1, result_type="expand"
                )

    
        def _adjust_total_fair(row):
            if pd.isna(row.get("total")) or pd.isna(row.get("consensus_total")):
                return row["over_fair"], row["under_fair"]
        
            diff = float(row["total"]) - float(row["consensus_total"])
            adj_over, adj_under = row["over_fair"], row["under_fair"]
        
            # Flat ~1% per point adjustment
            step = 0.01
            adj = diff * step
        
            adj_over -= adj
            adj_under += adj
        
            return max(0, min(1, adj_over)), max(0, min(1, adj_under))
        
        df_tot_cons = compute_consensus_fair_probs_totals(df_total_books) if not df_total_books.empty else pd.DataFrame()
        df_tot_best = best_prices_totals(df_total_books) if not df_total_books.empty else pd.DataFrame()
        
        if not df_tot_best.empty and not df_tot_cons.empty:
            df_total = pd.merge(
                df_tot_best,
                df_tot_cons,
                on=["event_id", "home_team", "away_team"],
                how="inner"
            )
            df_total[["over_fair", "under_fair"]] = df_total.apply(
                _adjust_total_fair, axis=1, result_type="expand"
            )
    
        # Format helpers
        def fmt_odds(o):
            try:
                o = int(o)
                return f"+{o}" if o > 0 else str(o)
            except Exception:
                return str(o)
    
        def fmt_ev(val) -> str:
            try:
                return f"{val:+.1f}%"
            except Exception:
                return str(val)
    
        # Display row makers
        def make_rows_moneyline(df_in: pd.DataFrame):
            rows = []
            for _, r in df_in.iterrows():
                date_str = fmt_date_et_str(r.get("commence_time"))
                game_label = f"{r['home_team']} vs {r['away_team']}"
                if pd.notna(r.get("home_price")) and pd.notna(r.get("home_fair")):
                    fair_p, price = float(r["home_fair"]), int(r["home_price"])
                    ev_pct = expected_value_pct(fair_p, price)
                    kelly = kelly_fraction(fair_p, price)
                    rows.append({
                        "Market": "Moneyline",
                        "Date": date_str,
                        "Game": game_label, "Pick": r["home_team"],
                        "Best Odds": price, "Best Book": r.get("home_book"),
                        "Fair Win %": fair_p, "EV%": ev_pct, "Kelly (u)": kelly,
                        "Stake ($)": round((weekly_bankroll if authed else 1000.0) * (kelly_factor if authed else 0.5) * kelly, 2)
                    })
                if pd.notna(r.get("away_price")) and pd.notna(r.get("away_fair")):
                    fair_p, price = float(r["away_fair"]), int(r["away_price"])
                    ev_pct = expected_value_pct(fair_p, price)
                    kelly = kelly_fraction(fair_p, price)
                    rows.append({
                        "Market": "Moneyline",
                        "Date": date_str,
                        "Game": game_label, "Pick": r["away_team"],
                        "Best Odds": price, "Best Book": r.get("away_book"),
                        "Fair Win %": fair_p, "EV%": ev_pct, "Kelly (u)": kelly,
                        "Stake ($)": round((weekly_bankroll if authed else 1000.0) * (kelly_factor if authed else 0.5) * kelly, 2)
                    })
            return pd.DataFrame(rows)
    
        def make_rows_spread(df_in: pd.DataFrame):
            rows = []
            for _, r in df_in.iterrows():
                date_str = fmt_date_et_str(r.get("commence_time"))
                game_label = f"{r['home_team']} vs {r['away_team']}"
                
                # 🛠 Safely handle missing line
                try:
                    line = float(r.get("line")) if r.get("line") is not None else None
                except Exception:
                    line = None
        
                # Home side
                if pd.notna(r.get("home_price")) and pd.notna(r.get("home_fair")):
                    fair_p, price = float(r["home_fair"]), int(r["home_price"])
                    ev_pct = expected_value_pct(fair_p, price)
                    kelly = kelly_fraction(fair_p, price)
                    rows.append({
                        "Market": "Spread",
                        "Date": date_str,
                        "Game": game_label, "Pick": r["home_team"],
                        "Line": f"{line:+g}" if line is not None else "",
                        "Best Odds": price, "Best Book": r.get("home_book"),
                        "Fair Win %": fair_p, "EV%": ev_pct, "Kelly (u)": kelly,
                        "Stake ($)": round((weekly_bankroll if authed else 1000.0) * 
                                           (kelly_factor if authed else 0.5) * kelly, 2)
                    })
        
                # Away side
                if pd.notna(r.get("away_price")) and pd.notna(r.get("away_fair")):
                    fair_p, price = float(r["away_fair"]), int(r["away_price"])
                    ev_pct = expected_value_pct(fair_p, price)
                    kelly = kelly_fraction(fair_p, price)
                    rows.append({
                        "Market": "Spread",
                        "Date": date_str,
                        "Game": game_label, "Pick": r["away_team"],
                        "Line": f"{line:+g}" if line is not None else "",
                        "Best Odds": price, "Best Book": r.get("away_book"),
                        "Fair Win %": fair_p, "EV%": ev_pct, "Kelly (u)": kelly,
                        "Stake ($)": round((weekly_bankroll if authed else 1000.0) * 
                                           (kelly_factor if authed else 0.5) * kelly, 2)
                    })
            return pd.DataFrame(rows)
    
        def make_rows_total(df_in: pd.DataFrame):
            rows = []
            for _, r in df_in.iterrows():
                date_str = fmt_date_et_str(r.get("commence_time"))
                game_label = f"{r['home_team']} vs {r['away_team']}"
                total = float(r.get("total"))
                if pd.notna(r.get("over_price")) and pd.notna(r.get("over_fair")):
                    fair_p, price = float(r["over_fair"]), int(r["over_price"])
                    ev_pct = expected_value_pct(fair_p, price)
                    kelly = kelly_fraction(fair_p, price)
                    rows.append({
                        "Market": "Total",                    
                        "Date": date_str,
                        "Game": game_label, "Pick": "Over",
                        "Line": f"{total:.1f}",
                        "Best Odds": price, "Best Book": r.get("over_book"),
                        "Fair Win %": fair_p, "EV%": ev_pct, "Kelly (u)": kelly,
                        "Stake ($)": round((weekly_bankroll if authed else 1000.0) * (kelly_factor if authed else 0.5) * kelly, 2)
                    })
                if pd.notna(r.get("under_price")) and pd.notna(r.get("under_fair")):
                    fair_p, price = float(r["under_fair"]), int(r["under_price"])
                    ev_pct = expected_value_pct(fair_p, price)
                    kelly = kelly_fraction(fair_p, price)
                    rows.append({
                        "Market": "Total",                    
                        "Date": date_str,
                        "Game": game_label, "Pick": "Under",
                        "Line": f"{total:.1f}",
                        "Best Odds": price, "Best Book": r.get("under_book"),
                        "Fair Win %": fair_p, "EV%": ev_pct, "Kelly (u)": kelly,
                        "Stake ($)": round((weekly_bankroll if authed else 1000.0) * (kelly_factor if authed else 0.5) * kelly, 2)
                    })
            return pd.DataFrame(rows)
    
        df_ml_disp = make_rows_moneyline(df_ml) if not df_ml.empty else pd.DataFrame()
        df_sp_disp = make_rows_spread(df_spread) if not df_spread.empty else pd.DataFrame()
        df_tot_disp= make_rows_total(df_total) if not df_total.empty else pd.DataFrame()
    
        st.subheader("Screened Picks")
    
        # Show latest odds pulled timestamp
        pulled_list = []
        pulled_list.extend(pulled_ml or [])
        pulled_list.extend(pulled_sp or [])
        pulled_list.extend(pulled_tot or [])
        latest_pull = max((parse_iso_dt_utc(p) for p in pulled_list if p), default=None)
    
        if latest_pull:
            pulled_at_et = latest_pull.astimezone(EASTERN)
            st.markdown(
                f"**Odds last pulled:** {pulled_at_et.strftime('%b %d, %Y %I:%M %p ET')}"
            )
        else:
            st.markdown("**Odds last pulled:** (no snapshot available)")
    
        # Keep the caption for window info
        st.caption(
            f"Window: {caption_label}  |  All times ET. Fair Win % is no-vig."
        )
        
        if market_choice == "Moneyline":
            df_disp = df_ml_disp
        elif market_choice == "Spread":
            df_disp = df_sp_disp
        else:
            df_disp = df_tot_disp
    
        if df_disp.empty:
            st.info("No data for this market.")
            return
    
        df_disp["Best Odds"] = df_disp["Best Odds"].apply(fmt_odds)
        df_disp["Fair Win %"] = (df_disp["Fair Win %"]*100).map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
        df_disp["EV%"] = df_disp["EV%"].map(fmt_ev)
        df_disp["Kelly (u)"] = df_disp["Kelly (u)"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    
        if not show_all:
            def _flt(row):
                evp = None
                try: evp = float(str(row["EV%"]).replace("%",""))
                except: pass
                try: fwp = float(str(row["Fair Win %"]).replace("%",""))
                except: fwp = None
                if evp is None or fwp is None: return False
                return (evp >= min_ev) and (fwp >= fair_win_min)
            df_disp = df_disp[df_disp.apply(_flt, axis=1)]
    
        if not authed:
            if df_disp.empty:
                st.warning("Please sign in to see all plays.")
            else:
                st.dataframe(df_disp.head(1).set_index("Market"))
                st.warning("Sign in to see the full table and filters.")
        else:
            # Allow user to choose sorting metric
            sort_option = st.radio(
                "Sort by:",
                ["Stake ($)", "EV%"],
                horizontal=True,
                index=0,
                help="Sort by Kelly-based stake (recommended) or by raw Expected Value."
            )
        
            # Prepare numeric column for sorting
            if sort_option == "EV%":
                df_disp["_sort"] = (
                    df_disp["EV%"]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .astype(float)
                )
            else:
                df_disp["_sort"] = pd.to_numeric(df_disp["Stake ($)"], errors="coerce")
        
            # Sort descending (largest first)
            df_disp = df_disp.sort_values("_sort", ascending=False).drop(columns=["_sort"])
        
            # Display sorted table
            st.dataframe(df_disp.set_index("Market"), use_container_width=True)


            
    with tabs[1]:
        st.subheader("Best Odds by Sportsbook")
    
        # Merge best prices with consensus for clarity
        df_best_disp = pd.merge(df_ml_best, df_ml_cons, on=["event_id","home_team","away_team"], how="inner")
    
        if df_best_disp.empty:
            st.info("No best odds available.")
        else:
            # Build Matchup column
            df_best_disp["Matchup (Home vs. Away)"] = df_best_disp["home_team"] + " vs " + df_best_disp["away_team"]
    
            # Format odds with "+" for positives
            def fmt_odds(o):
                try:
                    o = int(o)
                    return f"+{o}" if o > 0 else str(o)
                except:
                    return str(o)
    
            df_best_disp["Home Odds"] = df_best_disp["home_price"].apply(fmt_odds)
            df_best_disp["Away Odds"] = df_best_disp["away_price"].apply(fmt_odds)
    
            # Keep only needed columns
            df_best_disp = df_best_disp[[
                "commence_time", "Matchup (Home vs. Away)", "Home Odds", "home_book", "Away Odds", "away_book"
            ]].rename(columns={
                "commence_time": "Date",
                "home_book": "Best Home Book",
                "away_book": "Best Away Book",
            })
    
            # Format date
            df_best_disp["Date"] = df_best_disp["Date"].apply(fmt_date_et_str)
    
            # Show table with tooltip
            st.dataframe(
                df_best_disp,
                use_container_width=True,
                column_config={
                    "Matchup": st.column_config.TextColumn(
                        "Matchup",
                        help="first team listed = Home, second team listed = Away",
                    )
                }
            )
            
    with tabs[2]:
        st.subheader("Parlay Builder")
    
        # --- Stake input ---
        stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=1.0)
    
        # --- Market data dictionary ---
        markets = {
            "Moneyline": df_ml_books,
            "Spread": df_spread_books,
            "Total": df_total_books,
        }
    
        # --- Select Market ---
        market_choice = st.selectbox("Market", list(markets.keys()))
        df_market = markets.get(market_choice)
        if df_market is None or df_market.empty:
            st.info("No data available for this market.")
            st.stop()
    
        # --- First dropdown: Game ---
        games_for_market = sorted(list(set(df_market["home_team"] + " vs " + df_market["away_team"])))
        game_choice = st.selectbox("Game", games_for_market)
    
        # --- Second dropdown: Pick (team/line) ---
        picks = []
        subset = df_market[(df_market["home_team"] + " vs " + df_market["away_team"]) == game_choice]
    
        if market_choice == "Moneyline":
            teams = [subset["home_team"].iloc[0], subset["away_team"].iloc[0]]
            for team in teams:
                picks.append({"label": f"{team} Moneyline", "pick": team, "line": "Moneyline"})
    
        elif market_choice == "Spread":
            if "line" not in subset.columns or subset.empty:
                st.warning("No spread data found for this game.")
                lines = [0.0]
            else:
                # Collect all unique non-null spread lines
                lines = sorted([float(l) for l in subset["line"].unique() if pd.notna(l)])
        
            home_team = subset["home_team"].iloc[0]
            away_team = subset["away_team"].iloc[0]
        
            # Separate negative (favorites) and positive (underdogs)
            neg_lines = sorted([l for l in lines if l < 0])
            pos_lines = sorted([l for l in lines if l > 0])
        
            # If no negatives exist, assume away favorite and flip
            if not neg_lines and pos_lines:
                neg_lines = [-l for l in pos_lines]
        
            # Build dropdown entries for every line
            for line in neg_lines:
                picks.append({
                    "label": f"{home_team} {line:+.1f}",
                    "pick": home_team,
                    "line": f"{line:+.1f}"
                })
                picks.append({
                    "label": f"{away_team} {(-line):+.1f}",
                    "pick": away_team,
                    "line": f"{(-line):+.1f}"
                })


    
        elif market_choice == "Total":
            totals = sorted(subset["total"].unique()) if "total" in subset.columns else [0]
            for total in totals:
                picks.append({"label": f"Over {total:.1f}", "pick": "Over", "line": f"{total:.1f}"})
                picks.append({"label": f"Under {total:.1f}", "pick": "Under", "line": f"{total:.1f}"})
    
        pick_labels = [p["label"] for p in picks]
        pick_choice = st.selectbox("Pick", pick_labels)
        selected_pick = next((p for p in picks if p["label"] == pick_choice), None)
    
        # --- Session state init ---
        if "selected_legs" not in st.session_state:
            st.session_state.selected_legs = []
    
        # --- Add Leg button ---
        if st.button("Add Leg to Parlay", use_container_width=True) and selected_pick:
            leg_entry = {
                "Market": market_choice,
                "Game": game_choice,
                "Pick": selected_pick["pick"],
                "Line": selected_pick["line"],
            }
            if leg_entry not in st.session_state.selected_legs:
                st.session_state.selected_legs.append(leg_entry)
            else:
                st.warning("That leg is already added.")
    
        # --- Display Selected Legs ---
        st.markdown("### Selected Legs")
    
        if not st.session_state.selected_legs:
            st.info("Add at least two legs to compare parlay odds.")
            st.stop()
        else:
            # Header
            header_cols = st.columns([2, 3, 2, 1])
            header_cols[0].markdown("**Market**")
            header_cols[1].markdown("**Game**")
            header_cols[2].markdown("**Pick**")
            header_cols[3].markdown("**Line**")
    
            # Rows
            for leg in st.session_state.selected_legs:
                col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
                col1.write(leg["Market"])
                col2.write(leg["Game"])
                col3.write(leg["Pick"])
                col4.write(leg["Line"])
    
            # Action buttons below table
            colA, colB = st.columns([1, 1])
            with colA:
                compare_clicked = st.button("Compare Parlay Odds", use_container_width=True)
            with colB:
                if len(st.session_state.selected_legs) > 0:
                    if st.button("Clear Last Selection", use_container_width=True):
                        st.session_state.selected_legs.pop(-1)
                        st.rerun()
                else:
                    st.button("Clear Last Selection", use_container_width=True, disabled=True)
    
            # Stop until Compare is pressed
            if not compare_clicked:
                st.stop()
    
        # --- Sportsbook Comparison Logic ---
        every_book = sorted(
            set(
                df_ml_books["book"].unique().tolist()
                + df_spread_books["book"].unique().tolist()
                + df_total_books["book"].unique().tolist()
            )
        )
    
        sportsbook_results = []
        for book in every_book:
            combined_dec = 1.0
            all_lines = []
            valid = True
    
            for leg in st.session_state.selected_legs:
                df_src = markets.get(leg["Market"])
                if df_src is None or df_src.empty:
                    valid = False
                    break
    
                subset = df_src[
                    (df_src["book"] == book)
                    & ((df_src["home_team"] + " vs " + df_src["away_team"]) == leg["Game"])
                ]
                if subset.empty:
                    valid = False
                    break
    
                if leg["Market"] == "Moneyline":
                    row = subset.iloc[0]
                    if leg["Pick"] == row["home_team"]:
                        price = row["home_price"]
                    else:
                        price = row["away_price"]
                    dec = american_to_decimal(price)
                    line_display = "Moneyline"
    
                elif leg["Market"] == "Spread":
                    avg_line = subset["line"].mean()
                    row = subset.iloc[0]
                    if leg["Pick"] == row["home_team"]:
                        price = row["home_price"]
                        line_display = f"{avg_line:+.1f}"
                    else:
                        price = row["away_price"]
                        line_display = f"{-avg_line:+.1f}"
                    dec = american_to_decimal(price)
    
                elif leg["Market"] == "Total":
                    avg_total = subset["total"].mean() if "total" in subset.columns else 0
                    row = subset.iloc[0]
                    if leg["Pick"].lower() == "over":
                        price = row["over_price"]
                    else:
                        price = row["under_price"]
                    dec = american_to_decimal(price)
                    line_display = f"{avg_total:.1f}"
                else:
                    valid = False
                    break
    
                combined_dec *= dec
                all_lines.append(line_display)
    
            if valid and combined_dec > 1.0:
                parlay_american = (
                    int((combined_dec - 1) * 100)
                    if combined_dec >= 2
                    else int(-100 / (combined_dec - 1))
                )
    
                # Add '+' prefix for positive odds
                parlay_american_str = f"+{parlay_american}" if parlay_american > 0 else str(parlay_american)
    
                payout = round(stake * combined_dec, 2)
                sportsbook_results.append(
                    {
                        "Sportsbook": book,
                        "Decimal Odds": round(combined_dec, 3),
                        "American Odds": parlay_american_str,
                        "Payout ($)": f"${payout:,.2f}",
                        "Lines Used": ", ".join(all_lines),
                    }
                )
    
        # --- Display Results ---
        if not sportsbook_results:
            st.warning("No sportsbook offers all selected legs.")
        else:
            df_results = pd.DataFrame(sportsbook_results).sort_values("Payout ($)", ascending=False)
            st.markdown("### Parlay Comparison Across Sportsbooks")
            st.dataframe(df_results, use_container_width=True)




if __name__ == "__main__":
    run_app()
