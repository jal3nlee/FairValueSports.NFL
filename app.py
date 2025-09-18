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

# Reattach session tokens if present
if st.session_state.get("sb_access_token") and st.session_state.get("sb_refresh_token"):
    supabase.auth.set_session(
        access_token=st.session_state.sb_access_token,
        refresh_token=st.session_state.sb_refresh_token,
    )

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

# Account block (concise)
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
PACIFIC = ZoneInfo("America/Los_Angeles")  # used for window math only (label unchanged per your preference)

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
    return "americanfootball_nfl_preseason" if week_index == 0 else "americanfootball_nfl"

# =======================
# Supabase odds readers (UPDATED)
# =======================
PAGE_SIZE = 1000

@st.cache_data(ttl=60, show_spinner=False)
def get_latest_snapshot_meta(sport: str, market: str, region: str = "us"):
    """
    Returns (snapshot_id, pulled_at_iso) for the latest row in odds_snapshots
    for (sport, market, region). None if not found.
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
# Main app (soft-gated)
# =======================
def run_app():
    st.title("NFL Expected Value Model")
    # (UNCHANGED REST OF YOUR APP...)
    # ...
    # ---- Run app (soft gate; no hard require_auth) ----

run_app()

