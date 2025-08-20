# app.py
import os, math, requests
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client, Client
from pathlib import Path

# ===== BRANDING (clean, no debug) =====
from pathlib import Path
from PIL import Image
import streamlit as st

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
    page_title="Fair Value Sports",
    page_icon=(favicon_img if favicon_img else "🏈"),
    layout="wide",
)

HEADER_W = 560
SIDEBAR_W = 320

if LOGO_PATH:
    st.image(str(LOGO_PATH), width=HEADER_W)

with st.sidebar:
    if LOGO_PATH:
        st.image(str(LOGO_PATH), width=SIDEBAR_W)
    else:
        st.write("Fair Value Sports")
# ===== END BRANDING =====

# Make sure the sidebar opens by default (optional)
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# --- Sidebar ---
st.sidebar.divider()

with st.sidebar.expander("How to use", expanded=False):
    st.markdown(
        """
1. **Pick a Window**: choose **Today** or **NFL Week X** from the dropdown.
2. **Set inputs**:
   - **Weekly Bankroll ($)** — your total budget for the week.
   - **Kelly Factor (0–1)** — risk scaling (e.g., 0.5 = half Kelly).
   - **Minimum EV%** — filter picks below this expected value.
3. **Review the table**:
   - **Implied Probability** = de-vigged market consensus.
   - **EV%** = edge versus best available odds.
   - **Stake ($)** = how much you should bet based on Kelly Factor and Weekly Bankroll inputs.
        """
    )

# --- Sidebar Glossary ---
with st.sidebar.expander("Glossary", expanded=False):
    st.markdown(
        """
**EV% (Expected Value %)**  
Represents the percentage edge you have over the market.  
- Betting only positive EV% plays with a Kelly factor is mathematically profitable long-term, **BUT** only if the model’s probabilities are accurate and you can withstand short-term variance. 
- Conservative bettors typically only take bets with +2-3% EV or higher, focusing on fewer plays with stronger edges and lower variance.
- Aggressive bettors may take bets with +0.5-1% EV or higher, accepting more volume and variance in exchange for higher potential long-run growth.

**Kelly Factor**  
A bankroll management formula that adjusts bet size based on edge and probability.  
- **1.0** = full Kelly (aggressive, max growth but higher risk).  
- **0.5** = half Kelly (more conservative, balances growth & variance).  
- Lower values scale down risk even further.  
        """
    )

# --- Sidebar Feedback ---
with st.sidebar.expander("💬 Feedback", expanded=False):
    feedback_text = st.text_area("Share your thoughts, ideas, or issues:", key="feedback_input")
    if st.button("Submit Feedback", key="feedback_submit"):
        if feedback_text.strip():
            # Right now just show a success message
            # Later you could send this to Supabase, Google Sheet, or email
            st.success("Thanks for your feedback!")
            st.session_state.feedback_input = ""  # clears box
        else:
            st.warning("Please enter some feedback before submitting.")

with st.sidebar.expander("Disclaimer", expanded=False):
    st.markdown(
        """
**Fair Value Sports** is for **education and entertainment** only.  
Not to be used as financial or betting advice.
        """
    )


# =======================
# Auth (Supabase) — stable, single-submit forms
# =======================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Auth not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY in your environment.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---- Session state priming ----
st.session_state.setdefault("sb_session", None)
st.session_state.setdefault("sb_access_token", None)
st.session_state.setdefault("sb_refresh_token", None)

def _store_session(sess):
    """Persist session + tokens in session_state."""
    st.session_state.sb_session = sess
    try:
        st.session_state.sb_access_token = getattr(sess, "access_token", None) or sess.get("access_token")
        st.session_state.sb_refresh_token = getattr(sess, "refresh_token", None) or sess.get("refresh_token")
    except Exception:
        # sess may be an object with attributes (gotrue Session) or a dict depending on lib version
        pass

def _clear_session():
    st.session_state.sb_session = None
    st.session_state.sb_access_token = None
    st.session_state.sb_refresh_token = None

def _maybe_refresh_session():
    """
    Best-effort: if we have tokens but sb_session is None (after a rerun), try to refresh.
    This avoids the 'first click fails, second works' pattern caused by stale local state.
    """
    if st.session_state.sb_session is None and st.session_state.sb_refresh_token:
        try:
            res = supabase.auth.refresh_session()
            if res and getattr(res, "session", None):
                _store_session(res.session)
        except Exception:
            # If refresh fails, clear so user is prompted to log in cleanly
            _clear_session()

# Try refreshing on load if we previously had tokens
_maybe_refresh_session()

def auth_view():
    tabs = st.tabs(["Sign in", "Create account"])

    # --- Sign in (form = single atomic submit) ---
    with tabs[0]:
        with st.form("signin_form", clear_on_submit=False):
            email = st.text_input("Email", key="signin_email")
            password = st.text_input("Password", type="password", key="signin_pw")
            submit = st.form_submit_button("Sign in")
        if submit:
            try:
                # Clear any stale session to prevent token conflicts
                try:
                    supabase.auth.sign_out()
                except Exception:
                    pass
                res = supabase.auth.sign_in_with_password(
                    {"email": (email or "").strip(), "password": password}
                )
                # Some drivers return .session, others dict-like
                sess = getattr(res, "session", None) or getattr(res, "session", {}) or None
                if not sess:
                    # Occasionally a verified email flow or weird state returns no session on first pass
                    # Try one immediate refresh attempt to stabilize
                    try:
                        res2 = supabase.auth.refresh_session()
                        sess = getattr(res2, "session", None) or getattr(res2, "session", {}) or None
                    except Exception:
                        pass

                if sess:
                    _store_session(sess)
                    st.success("Signed in.")
                    st.rerun()
                else:
                    st.error("Sign-in succeeded but no session was returned. Please try again.")
            except Exception as e:
                msg = str(e)
                # Make common errors clearer
                if "Invalid login credentials" in msg:
                    st.error("Sign-in failed: invalid email or password.")
                elif "Email not confirmed" in msg or "confirmed_at" in msg:
                    st.error("Please verify your email address, then sign in.")
                else:
                    st.error(f"Sign-in failed: {msg or 'Unknown error.'}")

    # --- Create account (form) ---
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
                    res = supabase.auth.sign_up(
                        {
                            "email": email2.strip(),
                            "password": pw2,
                            "options": {"data": {"full_name": full_name.strip()}},
                        }
                    )
                    # For email-confirmation flows, session may be None until the user confirms
                    st.success("Account created. Check your email to verify, then sign in.")
                except Exception as e:
                    st.error(f"Sign-up failed: {str(e) or 'Try a different email or password.'}")

def require_auth():
    # If session dropped after a rerun, try to refresh once more before prompting UI
    if st.session_state.sb_session is None:
        _maybe_refresh_session()
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

    st.title("NFL Expected Value Model")
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
