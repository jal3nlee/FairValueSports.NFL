# app.py
import os, math, requests
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
API_BASE = "https://api.the-odds-api.com/v4"
API_KEY  = os.getenv("ODDS_API_KEY", "")
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

def devig_two_way(p_home: float, p_away: float):
    a = (p_home or 0.0); b = (p_away or 0.0); s = a + b
    if s <= 0:
        return None, None
    return a/s, b/s

def parse_iso_dt_utc(iso_s: str):
    try:
        return datetime.fromisoformat(iso_s.replace("Z","+00:00")).astimezone(timezone.utc)
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
    if week_index == 0:
        start = datetime(yr, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end   = wk1 - timedelta(seconds=1)
        return start, end
    start = wk1 + timedelta(days=7 * (week_index - 1))
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

def pretty_book_title(bm: dict) -> str:
    title = bm.get("title")
    if title:
        return title
    key = (bm.get("key") or "").replace("_", " ").title()
    return key or "Sportsbook"

def build_market(events, selected_books=None):
    rows = []
    for ev in events:
        home, away = ev.get("home_team"), ev.get("away_team")
        eid, t0 = ev.get("id"), ev.get("commence_time")
        for bm in ev.get("bookmakers", []):
            if selected_books and bm.get("key") not in selected_books:
                continue
            price_home, price_away = None, None
            for mk in bm.get("markets", []):
                if mk.get("key") == "h2h":
                    side_map = {o.get("name",""): o.get("price") for o in mk.get("outcomes", [])}
                    price_home = side_map.get(home); price_away = side_map.get(away)
                    break
            if price_home is None and price_away is None:
                continue
            rows.append({
                "event_id": eid, "home_team": home, "away_team": away,
                "book": pretty_book_title(bm),
                "home_price": price_home, "away_price": price_away,
                "commence_time": t0
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def compute_consensus_fair_probs(df_evt_books: pd.DataFrame):
    if df_evt_books.empty:
        return pd.DataFrame()
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

# --- Cached odds fetch (60s TTL) ---
@st.cache_data(ttl=60, show_spinner=False)
def fetch_odds_payload(sport_key: str, params: dict):
    url = f"{API_BASE}/sports/{sport_key}/odds"
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            raise RuntimeError("Rate limited by odds API. Please try again shortly.")
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError("Unexpected API payload.")
        return data
    except Exception as e:
        raise

# =======================
# Main app (soft-gated)
# =======================
def run_app():
    if not API_KEY:
        st.error("Missing ODDS_API_KEY environment variable.")
        st.stop()

    st.title("NFL Expected Value Model")

    # Top nudge
    if not authed:
        st.info("Preview mode: example rows shown. **Sign in** to adjust filters and sort.")

    # --- Window dropdown (disabled when not signed in) ---
    now_utc = datetime.now(timezone.utc)
    current_week = infer_current_week_index(now_utc)
    week_label = "NFL Preseason" if current_week == 0 else f"NFL Week {current_week}"

    window_options = ["Today", week_label, "Next 7 Days"]
    window_choice = st.selectbox(
        "Window",
        window_options,
        index=1,
        key="window_choice",
        help="Choose a time window. “Next 7 Days” shows games from today through the next full week.",
        disabled=not authed,
    )

    # Determine time window + sport key(s)
    # (Window math still uses PACIFIC per your original logic; label stays “Next 7 Days”)
    def _short_md(dt_utc):
        local = dt_utc.astimezone(EASTERN)
        return f"{local.month}/{local.day}"

    def _short_day_md(dt_utc):
        local = dt_utc.astimezone(EASTERN)
        return f"{local.strftime('%a')} {local.month}/{local.day}"

    def window_next_7_days(now_utc, tz=PACIFIC):
        local = now_utc.astimezone(tz)
        start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_local = (start_local + timedelta(days=8)).replace(hour=23, minute=59, second=59, microsecond=0)
        return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)

    if window_choice == "Today":
        now_local = datetime.now(PACIFIC)
        window_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        window_end   = (window_start + timedelta(days=1)) - timedelta(seconds=1)
        sport_keys   = {sport_key_for_week(current_week)}
        caption_label = f"Today ({_short_md(window_start)})"
    elif window_choice == "Next 7 Days":
        window_start, window_end = window_next_7_days(now_utc, tz=PACIFIC)
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

    # --- Inputs (disabled until signed in) ---
    st.subheader("Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        weekly_bankroll = st.number_input(
            "Weekly Bankroll ($)",
            min_value=0.0, value=1000.0, step=50.0,
            help="Total budget for this week.",
            disabled=not authed,
            key="weekly_bankroll",
        )
    with c2:
        kelly_factor = st.slider(
            "Kelly Factor (0.0–1.0)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Controls bet size. Lower = safer; higher = riskier.",
            disabled=not authed,
            key="kelly_factor",
        )
    with c3:
        min_ev = st.number_input(
            "Minimum EV% to display",
            value=0.0, step=0.5,
            help="Filter out plays below this expected value.",
            disabled=not authed,
            key="min_ev",
        )

    show_all = st.checkbox(
        "Show all games (ignore EV% filter)",
        value=False,
        help="Display every matchup regardless of EV%.",
        disabled=not authed,
        key="show_all",
    )

    # --- Fetch odds (cached), record exact pull time in ET ---
    params = {"apiKey": API_KEY, "regions": "us", "markets": "h2h", "oddsFormat": "american"}

    events = []
    try:
        for key in sorted(sport_keys):
            payload = fetch_odds_payload(key, params)
            events.extend(payload)
    except Exception as e:
        msg = str(e)
        if "Rate limited" in msg or "429" in msg:
            st.warning("Rate limit hit. Please try again shortly.")
            st.stop()
        st.error(f"API error: {msg}")
        st.stop()

    if not isinstance(events, list) or not events:
        st.warning("No events returned.")
        st.stop()

    # Filter to chosen window (commence_time is UTC)
    events = [
        ev for ev in events
        if (dt := parse_iso_dt_utc(ev.get("commence_time"))) and window_start <= dt <= window_end
    ]
    if not events:
        st.info(f"No NFL games in the selected window ({caption_label}).")
        st.stop()

    # Build -> de-vig -> best
    df_books = build_market(events, selected_books=None)
    if df_books.empty:
        st.warning("No odds available in this window.")
        st.stop()

    cons = compute_consensus_fair_probs(df_books)
    bests = best_prices(df_books)
    merged = pd.merge(bests, cons, on=["event_id","home_team","away_team"], how="inner")

    # Rows for display
    rows = []
    for _, r in merged.iterrows():
        date_str = fmt_date_et_str(r.get("commence_time"))
        game_label = f"{r['home_team']} vs {r['away_team']}"

        # Home
        if pd.notna(r.get("home_price")) and pd.notna(r.get("home_fair")):
            fair_p, price = float(r["home_fair"]), int(r["home_price"])
            ev_pct = expected_value_pct(fair_p, price)
            kelly = kelly_fraction(fair_p, price)
            rows.append({
                "Game": game_label, "Pick": r["home_team"],
                "Best Odds": price, "Best Book": r.get("home_book"),
                "Fair Win %": fair_p, "EV%": ev_pct, "Kelly (u)": kelly,  # store as raw; format later
                "Stake ($)": round((weekly_bankroll if authed else 1000.0) * (kelly_factor if authed else 0.5) * kelly, 2),
                "Date": date_str
            })

        # Away
        if pd.notna(r.get("away_price")) and pd.notna(r.get("away_fair")):
            fair_p, price = float(r["away_fair"]), int(r["away_price"])
            ev_pct = expected_value_pct(fair_p, price)
            kelly = kelly_fraction(fair_p, price)
            rows.append({
                "Game": game_label, "Pick": r["away_team"],
                "Best Odds": price, "Best Book": r.get("away_book"),
                "Fair Win %": fair_p, "EV%": ev_pct, "Kelly (u)": kelly,
                "Stake ($)": round((weekly_bankroll if authed else 1000.0) * (kelly_factor if authed else 0.5) * kelly, 2),
                "Date": date_str
            })

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No sides available.")
        st.stop()

    # Apply EV filter only when authed
    if authed and not show_all:
        df = df[df["EV%"] >= float(min_ev)].reset_index(drop=True)
        if df.empty:
            st.info("No games meet the EV% filter for this window.")
            st.stop()

    # Sort strongest first (static for preview; interactive for authed)
    df = df.sort_values(["EV%","Fair Win %"], ascending=[False, False]).reset_index(drop=True)

    # Pretty display helpers
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

    # Prepare view df
    show = df.copy()
    show["Best Odds"] = show["Best Odds"].map(fmt_odds)
    show["Best Book"] = show["Best Book"].astype(str)
    show["Fair Win %"] = show["Fair Win %"].map(lambda x: f"{x*100:.1f}%")
    show["EV%"] = show["EV%"].map(fmt_ev)
    show["Kelly (u)"] = show["Kelly (u)"].map(lambda x: f"{x:.2f}u")

    st.subheader("Games & EV")
    pulled_at_et = datetime.now(EASTERN)
    st.caption(f"Window: {caption_label}  |  Data pulled: {pulled_at_et.strftime('%b %d, %Y %I:%M %p ET')}  |  All times ET. Fair Win % is no-vig.")

    columns_order = ["Game","Pick","Best Odds","Best Book","Fair Win %","EV%","Kelly (u)","Stake ($)","Date"]

    if authed:
        # Interactive view
        st.dataframe(
            show[columns_order],
            use_container_width=True,
            hide_index=True,
        )
    else:
        # Static, non-sortable preview
        st.table(show[columns_order])

    # Utilization summary (uses actual bankroll if authed, else preview math)
    bankroll_used = float(df["Stake ($)"].sum())
    wk_bankroll = weekly_bankroll if authed else 1000.0
    util_pct = 100.0 * (bankroll_used / wk_bankroll) if wk_bankroll > 0 else 0.0

    if util_pct < 50:
        util_color, util_label = "#166534", "Low"
    elif util_pct < 70:
        util_color, util_label = "#a16207", "Moderate"
    else:
        util_color, util_label = "#b91c1c", "High"

    st.markdown(
        f"**Total Suggested Stake:** ${bankroll_used:,.2f}  |  **Utilization:** "
        f"<span style='color:{util_color}'>{util_pct:.1f}%</span> ({util_label}) of weekly bankroll",
        unsafe_allow_html=True
    )

# ---- Run app (soft gate; no hard require_auth) ----
run_app()
