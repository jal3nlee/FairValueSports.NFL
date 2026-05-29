# app.py
import os
import math
import pandas as pd
import streamlit as st
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client, Client
from pathlib import Path
from PIL import Image
from streamlit_cookies_manager import EncryptedCookieManager

# =======================
# AUTH
# =======================

SUPABASE_URL      = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
COOKIE_SECRET     = os.getenv("COOKIE_SECRET", "")  # FIX: was hardcoded — must be set in env

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Auth not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY to environment.")
    st.stop()

if not COOKIE_SECRET:
    st.error("COOKIE_SECRET is not set. Add it to your environment variables.")
    st.stop()

# Create Supabase client once
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Cookie manager
cookies = EncryptedCookieManager(prefix="fvb_", password=COOKIE_SECRET)
if not cookies.ready():
    st.stop()

# Session state defaults
st.session_state.setdefault("sb_access_token", None)
st.session_state.setdefault("sb_refresh_token", None)
st.session_state.setdefault("sb_session", None)

# Restore session from cookies if present
if (
    cookies.get("access_token")
    and cookies.get("refresh_token")
    and not st.session_state.get("sb_access_token")
):
    try:
        supabase.auth.set_session(
            access_token=cookies.get("access_token"),
            refresh_token=cookies.get("refresh_token"),
        )
        st.session_state.sb_access_token = cookies.get("access_token")
        st.session_state.sb_refresh_token = cookies.get("refresh_token")
    except Exception as e:
        st.warning(f"Could not restore login session: {e}")


def save_session(sess, remember=False):
    """Store Supabase session object."""
    st.session_state.sb_session = sess
    st.session_state.sb_access_token = sess.access_token
    st.session_state.sb_refresh_token = sess.refresh_token
    if remember:
        cookies["access_token"] = sess.access_token
        cookies["refresh_token"] = sess.refresh_token
        cookies.save()


def clear_session():
    st.session_state.sb_session = None
    st.session_state.sb_access_token = None
    st.session_state.sb_refresh_token = None
    cookies.clear()
    cookies.save()


authed = bool(
    st.session_state.sb_access_token and st.session_state.sb_refresh_token
)

# =======================
# BRANDING
# =======================
ROOT       = Path(__file__).parent.resolve()
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


LOGO_PATH    = find_asset("logo.png")
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

# =======================
# SIDEBAR UI
# =======================
with st.sidebar:
    if LOGO_PATH:
        st.image(str(LOGO_PATH), width=SIDEBAR_W)
    else:
        st.title("Fair Value Betting")

    st.sidebar.divider()

    if authed:
        user = supabase.auth.get_user()
        user_email = (
            getattr(user.user, "email", None)
            if user and getattr(user, "user", None)
            else None
        )

        st.success(f"Signed in{f' as {user_email}' if user_email else ''}.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Log out", use_container_width=True):
                try:
                    supabase.auth.sign_out()
                except Exception:
                    pass
                st.session_state.clear()
                cookies.clear()
                cookies.save()
                st.rerun()

    else:
        st.info(
            "Free full access in September — create a free account to unlock filters and sorting."
        )

        with st.form("login_form_sidebar", clear_on_submit=False, border=True):
            email       = st.text_input("Email", key="signin_email_sidebar")
            password    = st.text_input("Password", type="password", key="signin_pw_sidebar")
            remember_me = st.checkbox("Keep me logged in", value=True)
            submit      = st.form_submit_button("Sign in", use_container_width=True)

        if submit:
            try:
                res  = supabase.auth.sign_in_with_password(
                    {"email": (email or "").strip(), "password": password}
                )
                sess = getattr(res, "session", None)
                if sess:
                    save_session(sess, remember=remember_me)
                    st.success("Signed in successfully.")
                    st.rerun()
                else:
                    st.error("Sign-in succeeded but no session returned.")
            except Exception as e:
                if "Invalid login credentials" in str(e):
                    st.error("Invalid email or password.")
                else:
                    st.error(f"Sign-in failed: {e}")

        with st.expander("Create account", expanded=False):
            full_name = st.text_input("Name", key="signup_name_sidebar")
            email2    = st.text_input("Email", key="signup_email_sidebar")
            pw2       = st.text_input("Password", type="password", key="signup_pw_sidebar")
            submit2   = st.button("Create Account", use_container_width=True)
            if submit2:
                if not full_name.strip():
                    st.warning("Please enter your full name.")
                elif not email2 or not pw2:
                    st.warning("Email and password are required.")
                else:
                    try:
                        supabase.auth.sign_up(
                            {
                                "email": email2.strip(),
                                "password": pw2,
                                "options": {"data": {"full_name": full_name.strip()}},
                            }
                        )
                        st.success("Account created! Check your email to verify, then sign in.")
                    except Exception as e:
                        st.error(f"Sign-up failed: {str(e) or 'Try again.'}")


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
    _fb_user = None
    try:
        _fb_user = getattr(st.session_state.get("sb_session", None), "user", None)
    except Exception:
        _fb_user = None

    if not _fb_user:
        st.info("You must be signed in to leave feedback.")
    else:
        with st.form("feedback_form", clear_on_submit=True):
            _full_name  = (_fb_user.user_metadata or {}).get("full_name") or (_fb_user.user_metadata or {}).get("name") or ""
            _email_addr = getattr(_fb_user, "email", "") or (_fb_user.user_metadata or {}).get("email", "")
            st.markdown(f"**Submitting as:** {_full_name or 'Unknown'}  \n**Email:** {_email_addr or 'Unknown'}")
            feedback_text = st.text_area("Share your thoughts, ideas, or issues:")
            submitted     = st.form_submit_button("Submit Feedback")

        if submitted:
            txt = (feedback_text or "").strip()
            if not txt:
                st.warning("Please enter feedback before submitting.")
            else:
                try:
                    supabase.table("feedback").insert(
                        {
                            "message": txt,
                            "name":    _full_name.strip() or None,
                            "email":   (_email_addr or "").strip() or None,
                            "user_id": _fb_user.id,
                        }
                    ).execute()
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
# NFL CONFIG & HELPERS
# =======================
EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")


def american_to_implied_prob(odds):
    try:
        o = int(odds)
    except Exception:
        return None
    return 100 / (o + 100) if o > 0 else (-o) / (-o + 100)


def american_to_decimal(odds):
    o = int(odds)
    return 1 + (o / 100 if o > 0 else 100 / (-o))


def expected_value_pct(true_prob: float, american_odds: int) -> float:
    d = american_to_decimal(american_odds)
    return 100.0 * (true_prob * (d - 1.0) - (1.0 - true_prob))


def kelly_fraction(true_prob: float, american_odds: int) -> float:
    d = american_to_decimal(american_odds)
    b = d - 1.0
    p = float(true_prob)
    q = 1.0 - p
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - q) / b)


def devig_two_way(p_a: float, p_b: float):
    a = p_a or 0.0
    b = p_b or 0.0
    s = a + b
    if s <= 0:
        return None, None
    return a / s, b / s


def parse_iso_dt_utc(iso_s: str):
    try:
        return datetime.fromisoformat(str(iso_s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def fmt_date_et_str(iso_s: str, snap_odd_minutes: bool = True):
    dt = parse_iso_dt_utc(iso_s)
    if not dt:
        return None
    et = dt.astimezone(EASTERN)
    if snap_odd_minutes:
        if et.minute == 1:
            et -= timedelta(minutes=1)
        elif et.minute == 59:
            et += timedelta(minutes=1)
    dow = et.strftime("%a")
    md  = f"{et.month}/{et.day}"
    tm  = et.strftime("%I:%M %p").lstrip("0")
    return f"{dow} {md} {tm} ET"


# FIX: shared fmt_odds — defined once at module level (was duplicated inside tabs)
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


def thursday_after_labor_day_utc(year: int) -> datetime:
    """Thursday after Labor Day at 00:00 ET, converted to UTC."""
    d = datetime(year, 9, 1, tzinfo=EASTERN)
    while d.weekday() != 0:  # 0 = Monday
        d += timedelta(days=1)
    opener_et = (d + timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
    return opener_et.astimezone(timezone.utc)


def nfl_week_window_utc(week_index: int, now_utc: datetime):
    """
    Returns (start_utc, end_utc) for the given week index.
    FIX: previously returned bare int 0 before Week 1, causing unpack error.
    Now always returns a 2-tuple.
    """
    yr  = now_utc.astimezone(EASTERN).year
    wk1 = thursday_after_labor_day_utc(yr)

    if week_index == 0:
        # Pre-season: Jan 1 → day before Week 1
        start = datetime(yr, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end   = wk1 - timedelta(seconds=1)
        return start, end

    start = thursday_after_labor_day_utc(yr) + timedelta(days=7 * (week_index - 1))
    end   = start + timedelta(days=5, hours=23, minutes=59, seconds=59)
    return start, end


def infer_current_week_index(now_utc: datetime) -> int:
    """Return 0 before Week 1; otherwise clamp to 1..18."""
    yr  = now_utc.astimezone(EASTERN).year
    wk1 = thursday_after_labor_day_utc(yr)
    if now_utc < wk1:
        return 0
    weeks = (now_utc - wk1).days // 7 + 1
    return max(1, min(18, weeks))


def sport_key_for_week(week_index: int) -> str:
    return "NFL"


# =======================
# SUPABASE ODDS READERS
# =======================
PAGE_SIZE = 1000


@st.cache_data(ttl=300, show_spinner=False)  # FIX: bumped from 60s → 300s to reduce DB load
def get_latest_snapshot_meta(sport: str, market: str, region: str = "us"):
    try:
        res = (
            supabase.table("odds_snapshots")
            .select("id,pulled_at")
            .eq("sport", sport)
            .eq("market", market)
            .eq("region", region)
            .order("pulled_at", desc=True)
            .limit(1)
            .execute()
        )
        data = res.data or []
        if not data:
            return None, None
        row = data[0]
        return row["id"], row.get("pulled_at")
    except Exception:
        return None, None


@st.cache_data(ttl=300, show_spinner=False)  # FIX: bumped from 60s → 300s
def get_lines_for_snapshot(snapshot_id: str):
    rows, start = [], 0
    while True:
        page = (
            supabase.table("odds_lines")
            .select("event_id,home_team,away_team,commence_time,book,market,side,line,price")
            .eq("snapshot_id", snapshot_id)
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )
        chunk = page.data or []
        rows.extend(chunk)
        if len(chunk) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return pd.DataFrame(rows)


MARKET_MAP = {
    "moneyline": "h2h",
    "spread":    "spreads",
    "total":     "totals",
}


def fetch_market_lines(sport_keys: set, market_label: str):
    all_lines  = []
    pulled_ats = []
    db_market  = MARKET_MAP.get(market_label, market_label)

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
# MARKET BUILDERS
# =======================
def build_market_from_lines_moneyline(df_lines: pd.DataFrame) -> pd.DataFrame:
    if df_lines.empty:
        return pd.DataFrame()
    df = df_lines[(df_lines["market"] == "h2h") & (df_lines["side"].isin(["home", "away"]))].copy()
    df["home_price"] = df.apply(lambda r: r["price"] if r["side"] == "home" else None, axis=1)
    df["away_price"] = df.apply(lambda r: r["price"] if r["side"] == "away" else None, axis=1)
    return df.groupby(
        ["event_id", "home_team", "away_team", "book", "commence_time"], as_index=False
    ).agg(home_price=("home_price", "max"), away_price=("away_price", "max"))


def build_market_from_lines_spread(df_lines: pd.DataFrame) -> pd.DataFrame:
    if df_lines.empty:
        return pd.DataFrame()
    df = df_lines[(df_lines["market"] == "spreads") & (df_lines["side"].isin(["home", "away"]))].copy()
    df["home_price"] = df.apply(lambda r: r["price"] if r["side"] == "home" else None, axis=1)
    df["away_price"] = df.apply(lambda r: r["price"] if r["side"] == "away" else None, axis=1)
    return df.groupby(
        ["event_id", "home_team", "away_team", "book", "commence_time", "line"], as_index=False
    ).agg(home_price=("home_price", "max"), away_price=("away_price", "max"))


def build_market_from_lines_total(df_lines: pd.DataFrame) -> pd.DataFrame:
    if df_lines.empty:
        return pd.DataFrame()
    df = df_lines[(df_lines["market"] == "totals") & (df_lines["side"].isin(["over", "under"]))].copy()
    df["over_price"]  = df.apply(lambda r: r["price"] if r["side"] == "over"  else None, axis=1)
    df["under_price"] = df.apply(lambda r: r["price"] if r["side"] == "under" else None, axis=1)
    return df.groupby(
        ["event_id", "home_team", "away_team", "book", "commence_time", "line"], as_index=False
    ).agg(
        over_price=("over_price", "max"),
        under_price=("under_price", "max"),
    ).rename(columns={"line": "total"})


# =======================
# CONSENSUS / BEST PRICE
# =======================
def compute_consensus_fair_probs_h2h(df_evt_books: pd.DataFrame) -> pd.DataFrame:
    if df_evt_books.empty:
        return pd.DataFrame()

    df = df_evt_books.copy()
    df["home_imp_vig"] = df["home_price"].apply(american_to_implied_prob)
    df["away_imp_vig"] = df["away_price"].apply(american_to_implied_prob)

    bench_books      = {"FanDuel", "DraftKings", "bet365"}
    df["is_benchmark"] = df["book"].isin(bench_books)

    agg = (
        df.groupby(["event_id", "home_team", "away_team"])
        .apply(lambda g: pd.Series({
            "home_imp_vig": (
                g.loc[g["is_benchmark"], "home_imp_vig"].mean()
                if g["is_benchmark"].any()
                else g["home_imp_vig"].mean()
            ),
            "away_imp_vig": (
                g.loc[g["is_benchmark"], "away_imp_vig"].mean()
                if g["is_benchmark"].any()
                else g["away_imp_vig"].mean()
            ),
            "commence_time":        g["commence_time"].iloc[0],
            "num_benchmark_books":  g["is_benchmark"].sum(),
            "num_total_books":      len(g),
        }))
        .reset_index()
    )

    agg["home_fair"], agg["away_fair"] = zip(
        *agg.apply(lambda r: devig_two_way(r["home_imp_vig"], r["away_imp_vig"]), axis=1)
    )
    return agg


def best_prices_h2h(df_evt_books: pd.DataFrame) -> pd.DataFrame:
    if df_evt_books.empty:
        return pd.DataFrame(
            columns=["event_id", "home_team", "away_team", "home_book", "home_price", "away_book", "away_price"]
        )
    home_best = (
        df_evt_books.dropna(subset=["home_price"])
        .sort_values(["event_id", "home_price"], ascending=[True, False])
        .groupby(["event_id", "home_team", "away_team"]).first().reset_index()
        [["event_id", "home_team", "away_team", "book", "home_price"]]
        .rename(columns={"book": "home_book"})
    )
    away_best = (
        df_evt_books.dropna(subset=["away_price"])
        .sort_values(["event_id", "away_price"], ascending=[True, False])
        .groupby(["event_id", "home_team", "away_team"]).first().reset_index()
        [["event_id", "home_team", "away_team", "book", "away_price"]]
        .rename(columns={"book": "away_book"})
    )
    return pd.merge(home_best, away_best, on=["event_id", "home_team", "away_team"], how="outer")


def compute_consensus_fair_probs_spread(df_spreads: pd.DataFrame) -> pd.DataFrame:
    if df_spreads.empty:
        return pd.DataFrame()

    tmp = df_spreads.copy()
    tmp["home_imp_vig"] = tmp["home_price"].apply(american_to_implied_prob)
    tmp["away_imp_vig"] = tmp["away_price"].apply(american_to_implied_prob)
    tmp["home_dist"]    = tmp["home_price"].apply(lambda o: abs(abs(int(o)) - 110) if pd.notna(o) else 999)
    tmp["away_dist"]    = tmp["away_price"].apply(lambda o: abs(abs(int(o)) - 110) if pd.notna(o) else 999)
    tmp["anchor_line"]  = tmp.apply(
        lambda r: r["line"] if r["home_dist"] <= r["away_dist"] else -r["line"], axis=1
    )
    tmp["anchor_weight"] = tmp.apply(lambda r: 1.0 / (1.0 + min(r["home_dist"], r["away_dist"])), axis=1)

    agg = (
        tmp.groupby(["event_id", "home_team", "away_team"])
        .apply(lambda g: pd.Series({
            "consensus_line": (g["anchor_line"] * g["anchor_weight"]).sum() / g["anchor_weight"].sum(),
            "home_imp_vig":   g["home_imp_vig"].mean(),
            "away_imp_vig":   g["away_imp_vig"].mean(),
            "commence_time":  g["commence_time"].iloc[0],
        }))
        .reset_index()
    )
    agg["home_fair"] = 0.5
    agg["away_fair"] = 0.5
    return agg


def best_prices_spread(df_spreads: pd.DataFrame) -> pd.DataFrame:
    if df_spreads.empty:
        return pd.DataFrame()
    home_best = (
        df_spreads.dropna(subset=["home_price"])
        .sort_values(["event_id", "line", "home_price"], ascending=[True, True, False])
        .groupby(["event_id", "home_team", "away_team", "line"]).first().reset_index()
        [["event_id", "home_team", "away_team", "line", "book", "home_price"]]
        .rename(columns={"book": "home_book"})
    )
    away_best = (
        df_spreads.dropna(subset=["away_price"])
        .sort_values(["event_id", "line", "away_price"], ascending=[True, True, False])
        .groupby(["event_id", "home_team", "away_team", "line"]).first().reset_index()
        [["event_id", "home_team", "away_team", "line", "book", "away_price"]]
        .rename(columns={"book": "away_book"})
    )
    return pd.merge(home_best, away_best, on=["event_id", "home_team", "away_team", "line"], how="outer")


def compute_consensus_fair_probs_totals(df_totals: pd.DataFrame) -> pd.DataFrame:
    if df_totals.empty:
        return pd.DataFrame()

    tmp = df_totals.copy()
    tmp["over_imp_vig"]  = tmp["over_price"].apply(american_to_implied_prob)
    tmp["under_imp_vig"] = tmp["under_price"].apply(american_to_implied_prob)
    tmp["over_dist"]     = tmp["over_price"].apply(lambda o: abs(abs(int(o)) - 110) if pd.notna(o) else 999)
    tmp["under_dist"]    = tmp["under_price"].apply(lambda o: abs(abs(int(o)) - 110) if pd.notna(o) else 999)
    tmp["anchor_total"]  = tmp["total"]
    tmp["anchor_weight"] = tmp.apply(lambda r: 1.0 / (1.0 + min(r["over_dist"], r["under_dist"])), axis=1)

    agg = (
        tmp.groupby(["event_id", "home_team", "away_team"])
        .apply(lambda g: pd.Series({
            "consensus_total": (g["anchor_total"] * g["anchor_weight"]).sum() / g["anchor_weight"].sum(),
            "over_imp_vig":    g["over_imp_vig"].mean(),
            "under_imp_vig":   g["under_imp_vig"].mean(),
            "commence_time":   g["commence_time"].iloc[0],
        }))
        .reset_index()
    )
    agg["over_fair"]  = 0.5
    agg["under_fair"] = 0.5
    return agg


def best_prices_totals(df_totals: pd.DataFrame) -> pd.DataFrame:
    if df_totals.empty:
        return pd.DataFrame()
    over_best = (
        df_totals.dropna(subset=["over_price"])
        .sort_values(["event_id", "total", "over_price"], ascending=[True, True, False])
        .groupby(["event_id", "home_team", "away_team", "total"]).first().reset_index()
        [["event_id", "home_team", "away_team", "total", "book", "over_price"]]
        .rename(columns={"book": "over_book"})
    )
    under_best = (
        df_totals.dropna(subset=["under_price"])
        .sort_values(["event_id", "total", "under_price"], ascending=[True, True, False])
        .groupby(["event_id", "home_team", "away_team", "total"]).first().reset_index()
        [["event_id", "home_team", "away_team", "total", "book", "under_price"]]
        .rename(columns={"book": "under_book"})
    )
    return pd.merge(over_best, under_best, on=["event_id", "home_team", "away_team", "total"], how="outer")


# =======================
# VECTORIZED ROW MAKERS
# FIX: replaced slow iterrows() loops with vectorized DataFrame ops
# =======================

def make_rows_moneyline(df_in: pd.DataFrame, bankroll: float, kf: float) -> pd.DataFrame:
    if df_in.empty:
        return pd.DataFrame()

    def _side(price_col, fair_col, team_col, book_col):
        sub = df_in[[price_col, fair_col, team_col, book_col, "home_team", "away_team", "commence_time"]].copy()
        sub = sub.dropna(subset=[price_col, fair_col])
        sub["price"] = sub[price_col].astype(int)
        sub["fair"]  = sub[fair_col].astype(float)
        sub["EV%_raw"]    = sub.apply(lambda r: expected_value_pct(r["fair"], r["price"]), axis=1)
        sub["Kelly_raw"]  = sub.apply(lambda r: kelly_fraction(r["fair"], r["price"]), axis=1)
        sub["Stake ($)"]  = (bankroll * kf * sub["Kelly_raw"]).round(2)
        sub["Market"]     = "Moneyline"
        sub["Date"]       = sub["commence_time"].apply(fmt_date_et_str)
        sub["Game"]       = sub["home_team"] + " vs " + sub["away_team"]
        sub["Pick"]       = sub[team_col]
        sub["Best Odds"]  = sub["price"]
        sub["Best Book"]  = sub[book_col]
        sub["Fair Win %"] = sub["fair"]
        sub["EV%"]        = sub["EV%_raw"]
        sub["Kelly (u)"]  = sub["Kelly_raw"]
        return sub[["Market", "Date", "Game", "Pick", "Best Odds", "Best Book", "Fair Win %", "EV%", "Kelly (u)", "Stake ($)"]]

    return pd.concat([_side("home_price", "home_fair", "home_team", "home_book"),
                      _side("away_price", "away_fair", "away_team", "away_book")],
                     ignore_index=True)


def make_rows_spread(df_in: pd.DataFrame, bankroll: float, kf: float) -> pd.DataFrame:
    if df_in.empty:
        return pd.DataFrame()

    def _side(price_col, fair_col, team_col, book_col):
        sub = df_in[[price_col, fair_col, team_col, book_col, "home_team", "away_team", "commence_time", "line"]].copy()
        sub = sub.dropna(subset=[price_col, fair_col])
        sub["price"]      = sub[price_col].astype(int)
        sub["fair"]       = sub[fair_col].astype(float)
        sub["EV%_raw"]    = sub.apply(lambda r: expected_value_pct(r["fair"], r["price"]), axis=1)
        sub["Kelly_raw"]  = sub.apply(lambda r: kelly_fraction(r["fair"], r["price"]), axis=1)
        sub["Stake ($)"]  = (bankroll * kf * sub["Kelly_raw"]).round(2)
        sub["Market"]     = "Spread"
        sub["Date"]       = sub["commence_time"].apply(fmt_date_et_str)
        sub["Game"]       = sub["home_team"] + " vs " + sub["away_team"]
        sub["Pick"]       = sub[team_col]
        sub["Line"]       = sub["line"].apply(lambda l: f"{float(l):+g}" if pd.notna(l) else "")
        sub["Best Odds"]  = sub["price"]
        sub["Best Book"]  = sub[book_col]
        sub["Fair Win %"] = sub["fair"]
        sub["EV%"]        = sub["EV%_raw"]
        sub["Kelly (u)"]  = sub["Kelly_raw"]
        return sub[["Market", "Date", "Game", "Pick", "Line", "Best Odds", "Best Book", "Fair Win %", "EV%", "Kelly (u)", "Stake ($)"]]

    return pd.concat([_side("home_price", "home_fair", "home_team", "home_book"),
                      _side("away_price", "away_fair", "away_team", "away_book")],
                     ignore_index=True)


def make_rows_total(df_in: pd.DataFrame, bankroll: float, kf: float) -> pd.DataFrame:
    if df_in.empty:
        return pd.DataFrame()

    def _side(price_col, fair_col, pick_label, book_col):
        sub = df_in[[price_col, fair_col, book_col, "home_team", "away_team", "commence_time", "total"]].copy()
        sub = sub.dropna(subset=[price_col, fair_col])
        sub["price"]      = sub[price_col].astype(int)
        sub["fair"]       = sub[fair_col].astype(float)
        sub["EV%_raw"]    = sub.apply(lambda r: expected_value_pct(r["fair"], r["price"]), axis=1)
        sub["Kelly_raw"]  = sub.apply(lambda r: kelly_fraction(r["fair"], r["price"]), axis=1)
        sub["Stake ($)"]  = (bankroll * kf * sub["Kelly_raw"]).round(2)
        sub["Market"]     = "Total"
        sub["Date"]       = sub["commence_time"].apply(fmt_date_et_str)
        sub["Game"]       = sub["home_team"] + " vs " + sub["away_team"]
        sub["Pick"]       = pick_label
        sub["Line"]       = sub["total"].apply(lambda t: f"{float(t):.1f}")
        sub["Best Odds"]  = sub["price"]
        sub["Best Book"]  = sub[book_col]
        sub["Fair Win %"] = sub["fair"]
        sub["EV%"]        = sub["EV%_raw"]
        sub["Kelly (u)"]  = sub["Kelly_raw"]
        return sub[["Market", "Date", "Game", "Pick", "Line", "Best Odds", "Best Book", "Fair Win %", "EV%", "Kelly (u)", "Stake ($)"]]

    return pd.concat([_side("over_price",  "over_fair",  "Over",  "over_book"),
                      _side("under_price", "under_fair", "Under", "under_book")],
                     ignore_index=True)


# =======================
# MAIN APP
# =======================
def run_app():
    tabs = st.tabs(["NFL Expected Value Model", "Best Odds by Sportsbook", "Parlay Builder", "Player Props"])

    # ── shared window/data state so Tab 1 data is reusable in other tabs ──
    with tabs[0]:
        if not authed:
            st.info("Preview Mode: showing today's top pick — **Sign in** to see all picks, filters, and sorting.")

        # Market filter
        market_choice = st.radio(
            "Market",
            ["Moneyline", "Spread", "Total"],
            index=0,
            horizontal=True,
            help="Toggle between Moneyline, Spread, and Total markets.",
            key="market_choice",
        )

        # Window dropdown
        now_utc       = datetime.now(timezone.utc)
        current_week  = infer_current_week_index(now_utc)
        week_label    = "NFL Preseason" if current_week == 0 else f"NFL Week {current_week}"
        window_options = ["Today", week_label, "Next 7 Days"]

        window_choice = st.selectbox(
            "Date Window",
            window_options,
            index=1,
            key="window_choice",
            help="Choose a time window. 'Next 7 Days' shows games from today through the next full week.",
        )

        def _short_day_md(dt_utc):
            local = dt_utc.astimezone(EASTERN)
            return f"{local.strftime('%a')} {local.month}/{local.day}"

        def window_next_7_days(now_utc, tz=EASTERN):
            local       = now_utc.astimezone(tz)
            start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
            end_local   = (start_local + timedelta(days=8)).replace(hour=23, minute=59, second=59, microsecond=0)
            return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)

        if window_choice == "Today":
            now_local     = datetime.now(EASTERN)
            window_start  = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
            window_end    = (window_start + timedelta(days=1)) - timedelta(seconds=1)
            sport_keys    = {sport_key_for_week(current_week)}
            caption_label = "Today"
        elif window_choice == "Next 7 Days":
            window_start, window_end = window_next_7_days(now_utc, tz=EASTERN)
            sport_keys = {
                sport_key_for_week(infer_current_week_index(window_start)),
                sport_key_for_week(infer_current_week_index(window_end)),
            }
            caption_label = f"{_short_day_md(window_start)} – {_short_day_md(window_end)}"
        else:
            week_index   = current_week
            # FIX: nfl_week_window_utc now always returns a tuple — no more unpack error
            window_start, window_end = nfl_week_window_utc(week_index, now_utc)
            sport_keys    = {sport_key_for_week(week_index)}
            caption_label = f"{week_label} — {_short_day_md(window_start)} – {_short_day_md(window_end)}"

        # FIX: lazy market fetching — only fetch the selected market to reduce DB calls
        with st.spinner("Loading odds data…"):
            market_label = market_choice.lower()
            if market_label == "moneyline":
                df_ml_lines,    pulled_ml  = fetch_market_lines(sport_keys, "moneyline")
                df_spread_lines = pd.DataFrame(); pulled_sp  = []
                df_total_lines  = pd.DataFrame(); pulled_tot = []
            elif market_label == "spread":
                df_spread_lines, pulled_sp  = fetch_market_lines(sport_keys, "spread")
                df_ml_lines     = pd.DataFrame(); pulled_ml  = []
                df_total_lines  = pd.DataFrame(); pulled_tot = []
            else:
                df_total_lines,  pulled_tot = fetch_market_lines(sport_keys, "total")
                df_ml_lines     = pd.DataFrame(); pulled_ml  = []
                df_spread_lines = pd.DataFrame(); pulled_sp  = []

        def filter_by_window_df(df_any: pd.DataFrame) -> pd.DataFrame:
            if df_any.empty:
                return df_any
            df        = df_any.copy()
            df["__t0"] = df["commence_time"].apply(parse_iso_dt_utc)
            df         = df[(df["__t0"] >= window_start) & (df["__t0"] <= window_end)]
            return df.drop(columns=["__t0"])

        df_ml_lines     = filter_by_window_df(df_ml_lines)
        df_spread_lines = filter_by_window_df(df_spread_lines)
        df_total_lines  = filter_by_window_df(df_total_lines)

        if df_ml_lines.empty and df_spread_lines.empty and df_total_lines.empty:
            st.info(f"No NFL games in the selected window ({caption_label}).")
            st.stop()

        def _books_from(df: pd.DataFrame) -> set:
            if df is None or df.empty or "book" not in df.columns:
                return set()
            return set(df["book"].dropna().astype(str).unique().tolist())

        all_books = sorted(
            _books_from(df_ml_lines) | _books_from(df_spread_lines) | _books_from(df_total_lines)
        )

        selected_books = (
            st.multiselect(
                "Sportsbooks",
                options=all_books,
                default=all_books,
                help="Uncheck sportsbooks you don't want included.",
            )
            if all_books
            else []
        )

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
                help="Hide picks with a fair win probability below this percentage.",
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
                help="50% = half Kelly, 100% = full Kelly.",
                disabled=not authed,
                key="kelly_factor",
            )
        kelly_factor = kelly_pct / 100.0

        show_all = st.toggle(
            "Show all matchups",
            value=False,
            help="Ignore EV% and Fair Win % filters.",
            disabled=not authed,
            key="show_all",
        )

        eff_bankroll = weekly_bankroll if authed else 1000.0
        eff_kelly    = kelly_factor    if authed else 0.5

        # Build book-level tables
        def _apply_book_filter(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            if not selected_books:
                return df.iloc[0:0].copy()
            return df[df["book"].isin(selected_books)].copy()

        df_ml_books     = _apply_book_filter(build_market_from_lines_moneyline(df_ml_lines))
        df_spread_books = _apply_book_filter(build_market_from_lines_spread(df_spread_lines))
        df_total_books  = _apply_book_filter(build_market_from_lines_total(df_total_lines))

        # Moneyline consensus + best
        df_ml_cons = compute_consensus_fair_probs_h2h(df_ml_books) if not df_ml_books.empty else pd.DataFrame()
        df_ml_best = best_prices_h2h(df_ml_books) if not df_ml_books.empty else pd.DataFrame()
        df_ml = (
            pd.merge(df_ml_best, df_ml_cons, on=["event_id", "home_team", "away_team"], how="inner")
            if (not df_ml_best.empty and not df_ml_cons.empty)
            else pd.DataFrame()
        )

        # Spread consensus + best
        df_spread = pd.DataFrame()

        def _adjust_spread_fair(row):
            if pd.isna(row.get("line")) or pd.isna(row.get("consensus_line")):
                return 0.5, 0.5
            line      = float(row["line"])
            consensus = float(row["consensus_line"])
            diff      = abs(abs(line) - abs(consensus))
            step_per_point = 0.025 if any(abs(round(consensus) - k) <= 0.5 for k in [3, 7]) else 0.02
            adj  = min(diff * step_per_point, 0.30)
            if consensus < 0:
                home_fair = 0.5 - adj if abs(line) > abs(consensus) else 0.5 + adj
            else:
                home_fair = 0.5 + adj if abs(line) > abs(consensus) else 0.5 - adj
            home_fair = max(0.01, min(0.99, home_fair))
            return home_fair, max(0.01, min(0.99, 1 - home_fair))

        if not df_spread_books.empty:
            df_sp_cons = compute_consensus_fair_probs_spread(df_spread_books)
            df_sp_best = best_prices_spread(df_spread_books)
            if not df_sp_best.empty and not df_sp_cons.empty:
                df_spread = pd.merge(df_sp_best, df_sp_cons, on=["event_id", "home_team", "away_team"], how="inner")
                df_spread[["home_fair", "away_fair"]] = df_spread.apply(
                    _adjust_spread_fair, axis=1, result_type="expand"
                )

        # Totals consensus + best
        df_total = pd.DataFrame()

        def _adjust_total_fair(row):
            if pd.isna(row.get("total")) or pd.isna(row.get("consensus_total")):
                return row["over_fair"], row["under_fair"]
            diff      = float(row["total"]) - float(row["consensus_total"])
            adj_over  = max(0, min(1, row["over_fair"]  - diff * 0.01))
            adj_under = max(0, min(1, row["under_fair"] + diff * 0.01))
            return adj_over, adj_under

        if not df_total_books.empty:
            df_tot_cons = compute_consensus_fair_probs_totals(df_total_books)
            df_tot_best = best_prices_totals(df_total_books)
            if not df_tot_best.empty and not df_tot_cons.empty:
                df_total = pd.merge(df_tot_best, df_tot_cons, on=["event_id", "home_team", "away_team"], how="inner")
                df_total[["over_fair", "under_fair"]] = df_total.apply(
                    _adjust_total_fair, axis=1, result_type="expand"
                )

        # Build display DataFrames (vectorized)
        df_ml_disp  = make_rows_moneyline(df_ml,     eff_bankroll, eff_kelly) if not df_ml.empty     else pd.DataFrame()
        df_sp_disp  = make_rows_spread(df_spread,    eff_bankroll, eff_kelly) if not df_spread.empty  else pd.DataFrame()
        df_tot_disp = make_rows_total(df_total,      eff_bankroll, eff_kelly) if not df_total.empty   else pd.DataFrame()

        st.subheader("Screened Picks")

        pulled_list = list(pulled_ml or []) + list(pulled_sp or []) + list(pulled_tot or [])
        latest_pull = max((parse_iso_dt_utc(p) for p in pulled_list if p), default=None)

        if latest_pull:
            pulled_at_et = latest_pull.astimezone(EASTERN)
            st.markdown(f"**Odds last pulled:** {pulled_at_et.strftime('%b %d, %Y %I:%M %p ET')}")
        else:
            st.markdown("**Odds last pulled:** (no snapshot available)")

        st.caption(f"Window: {caption_label}  |  All times ET. Fair Win % is no-vig.")

        if market_choice == "Moneyline":
            df_disp = df_ml_disp
        elif market_choice == "Spread":
            df_disp = df_sp_disp
        else:
            df_disp = df_tot_disp

        if df_disp.empty:
            st.info("No data for this market.")
            return

        # FIX: filter on raw numeric values BEFORE formatting strings
        if not show_all:
            mask = (df_disp["EV%"] >= min_ev) & (df_disp["Fair Win %"] * 100 >= fair_win_min)
            df_disp = df_disp[mask].copy()

        # Now format for display
        df_disp["Best Odds"]  = df_disp["Best Odds"].apply(fmt_odds)
        df_disp["Fair Win %"] = (df_disp["Fair Win %"] * 100).map(lambda x: f"{x:.1f}%")
        df_disp["EV%"]        = df_disp["EV%"].map(fmt_ev)
        df_disp["Kelly (u)"]  = df_disp["Kelly (u)"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

        if not authed:
            if df_disp.empty:
                st.warning("Please sign in to see all plays.")
            else:
                st.dataframe(df_disp.head(1).set_index("Market"))
                st.warning("Sign in to see the full table and filters.")
        else:
            sort_option = st.radio(
                "Sort by:",
                ["Stake ($)", "EV%"],
                horizontal=True,
                index=0,
                help="Sort by Kelly-based stake or raw Expected Value.",
            )

            if sort_option == "EV%":
                df_disp["_sort"] = (
                    df_disp["EV%"].astype(str).str.replace("%", "", regex=False).astype(float)
                )
            else:
                df_disp["_sort"] = pd.to_numeric(df_disp["Stake ($)"], errors="coerce")

            df_disp = df_disp.sort_values("_sort", ascending=False).drop(columns=["_sort"])
            st.dataframe(df_disp.set_index("Market"), use_container_width=True)

    # ─────────────────────────────────────────
    # TAB 1 — Best Odds by Sportsbook
    # ─────────────────────────────────────────
    with tabs[1]:
        if not authed:
            st.warning("🔒 Please sign in to view Best Odds by Sportsbook.")
            st.stop()

        st.subheader("Best Odds by Sportsbook")

        # Re-fetch moneyline lines for this tab (cached, so no extra DB round trip)
        _ml_lines_tab1, _ = fetch_market_lines({sport_key_for_week(current_week)}, "moneyline")
        _ml_lines_tab1    = filter_by_window_df(_ml_lines_tab1) if not _ml_lines_tab1.empty else _ml_lines_tab1
        _ml_books_tab1    = build_market_from_lines_moneyline(_ml_lines_tab1)
        _ml_cons_tab1     = compute_consensus_fair_probs_h2h(_ml_books_tab1) if not _ml_books_tab1.empty else pd.DataFrame()
        _ml_best_tab1     = best_prices_h2h(_ml_books_tab1) if not _ml_books_tab1.empty else pd.DataFrame()

        df_best_disp = (
            pd.merge(_ml_best_tab1, _ml_cons_tab1, on=["event_id", "home_team", "away_team"], how="inner")
            if (not _ml_best_tab1.empty and not _ml_cons_tab1.empty)
            else pd.DataFrame()
        )

        if df_best_disp.empty:
            st.info("No best odds available.")
        else:
            df_best_disp["Matchup (Home vs. Away)"] = df_best_disp["home_team"] + " vs " + df_best_disp["away_team"]
            df_best_disp["Home Odds"] = df_best_disp["home_price"].apply(fmt_odds)
            df_best_disp["Away Odds"] = df_best_disp["away_price"].apply(fmt_odds)
            df_best_disp = df_best_disp[
                ["commence_time", "Matchup (Home vs. Away)", "Home Odds", "home_book", "Away Odds", "away_book"]
            ].rename(columns={"commence_time": "Date", "home_book": "Best Home Book", "away_book": "Best Away Book"})
            df_best_disp["Date"] = df_best_disp["Date"].apply(fmt_date_et_str)
            st.dataframe(df_best_disp, use_container_width=True)

    # ─────────────────────────────────────────
    # TAB 2 — Parlay Builder
    # ─────────────────────────────────────────
    with tabs[2]:
        if not authed:
            st.warning("🔒 Please sign in to use the Parlay Builder.")
            st.stop()

        st.subheader("Parlay Builder")

        stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=1.0)

        # Re-fetch all markets for parlay builder (cached)
        _now = datetime.now(timezone.utc)
        _wk  = infer_current_week_index(_now)
        _sp  = {sport_key_for_week(_wk)}

        _ml_p,  _ = fetch_market_lines(_sp, "moneyline")
        _sp_p,  _ = fetch_market_lines(_sp, "spread")
        _tot_p, _ = fetch_market_lines(_sp, "total")

        _ml_p  = filter_by_window_df(_ml_p)  if not _ml_p.empty  else _ml_p
        _sp_p  = filter_by_window_df(_sp_p)  if not _sp_p.empty  else _sp_p
        _tot_p = filter_by_window_df(_tot_p) if not _tot_p.empty else _tot_p

        pb_ml_books  = build_market_from_lines_moneyline(_ml_p)
        pb_sp_books  = build_market_from_lines_spread(_sp_p)
        pb_tot_books = build_market_from_lines_total(_tot_p)

        markets = {"Moneyline": pb_ml_books, "Spread": pb_sp_books, "Total": pb_tot_books}

        market_choice_pb = st.selectbox("Market", list(markets.keys()), key="parlay_market")
        df_market = markets.get(market_choice_pb)

        if df_market is None or df_market.empty:
            st.info("No data available for this market.")
        else:
            games_for_market = sorted(list(set(df_market["home_team"] + " vs " + df_market["away_team"])))
            game_choice      = st.selectbox("Game", games_for_market)

            picks  = []
            subset = df_market[(df_market["home_team"] + " vs " + df_market["away_team"]) == game_choice]

            if market_choice_pb == "Moneyline":
                for team in [subset["home_team"].iloc[0], subset["away_team"].iloc[0]]:
                    picks.append({"label": f"{team} Moneyline", "pick": team, "line": "Moneyline"})

            elif market_choice_pb == "Spread":
                lines     = sorted([float(l) for l in subset["line"].unique() if pd.notna(l)]) if "line" in subset.columns else [0.0]
                home_team = subset["home_team"].iloc[0]
                away_team = subset["away_team"].iloc[0]

                def _imp(odds):
                    try:
                        o = float(odds)
                        return 100 / (o + 100) if o > 0 else (-o) / (-o + 100)
                    except Exception:
                        return None

                home_probs = [_imp(o) for o in subset["home_price"] if pd.notna(o)]
                away_probs = [_imp(o) for o in subset["away_price"] if pd.notna(o)]
                home_is_fav = (sum(home_probs) / len(home_probs) if home_probs else 0.5) > (sum(away_probs) / len(away_probs) if away_probs else 0.5)

                for line in lines:
                    hl = -abs(line) if home_is_fav else abs(line)
                    al =  abs(line) if home_is_fav else -abs(line)
                    picks.append({"label": f"{home_team} {hl:+.1f}", "pick": home_team, "line": f"{hl:+.1f}"})
                    picks.append({"label": f"{away_team} {al:+.1f}", "pick": away_team, "line": f"{al:+.1f}"})

            elif market_choice_pb == "Total":
                totals = sorted(subset["total"].unique()) if "total" in subset.columns else [0]
                for total in totals:
                    picks.append({"label": f"Over {total:.1f}",  "pick": "Over",  "line": f"{total:.1f}"})
                    picks.append({"label": f"Under {total:.1f}", "pick": "Under", "line": f"{total:.1f}"})

            pick_labels  = [p["label"] for p in picks]
            pick_choice  = st.selectbox("Pick", pick_labels)
            selected_pick = next((p for p in picks if p["label"] == pick_choice), None)

            st.session_state.setdefault("selected_legs", [])

            if st.button("Add Leg to Parlay", use_container_width=True) and selected_pick:
                leg_entry = {"Market": market_choice_pb, "Game": game_choice, "Pick": selected_pick["pick"], "Line": selected_pick["line"]}
                if leg_entry not in st.session_state.selected_legs:
                    st.session_state.selected_legs.append(leg_entry)
                else:
                    st.warning("That leg is already added.")

            st.markdown("### Selected Legs")

            if not st.session_state.selected_legs:
                st.info("Add at least two legs to compare parlay odds.")
            else:
                header_cols = st.columns([2, 3, 2, 1])
                for col, lbl in zip(header_cols, ["**Market**", "**Game**", "**Pick**", "**Line**"]):
                    col.markdown(lbl)

                for leg in st.session_state.selected_legs:
                    c1, c2, c3, c4 = st.columns([2, 3, 2, 1])
                    c1.write(leg["Market"]); c2.write(leg["Game"]); c3.write(leg["Pick"]); c4.write(leg["Line"])

                colA, colB = st.columns([1, 1])
                compare_clicked = colA.button("Compare Parlay Odds", use_container_width=True)
                has_legs        = len(st.session_state.selected_legs) > 0
                if colB.button("Clear Last Selection", use_container_width=True, disabled=not has_legs):
                    st.session_state.selected_legs.pop(-1)
                    st.rerun()

                if compare_clicked:
                    every_book = sorted(
                        set(
                            (pb_ml_books["book"].unique().tolist() if not pb_ml_books.empty else [])
                            + (pb_sp_books["book"].unique().tolist() if not pb_sp_books.empty else [])
                            + (pb_tot_books["book"].unique().tolist() if not pb_tot_books.empty else [])
                        )
                    )

                    sportsbook_results = []
                    for book in every_book:
                        combined_dec = 1.0
                        all_lines    = []
                        valid        = True

                        for leg in st.session_state.selected_legs:
                            df_src = markets.get(leg["Market"])
                            if df_src is None or df_src.empty:
                                valid = False; break

                            sub = df_src[
                                (df_src["book"] == book)
                                & ((df_src["home_team"] + " vs " + df_src["away_team"]) == leg["Game"])
                            ]
                            if sub.empty:
                                valid = False; break

                            row = sub.iloc[0]
                            if leg["Market"] == "Moneyline":
                                price        = row["home_price"] if leg["Pick"] == row["home_team"] else row["away_price"]
                                line_display = "Moneyline"
                            elif leg["Market"] == "Spread":
                                price        = row["home_price"] if leg["Pick"] == row["home_team"] else row["away_price"]
                                line_display = f"{sub['line'].mean():+.1f}"
                            elif leg["Market"] == "Total":
                                price        = row["over_price"] if leg["Pick"].lower() == "over" else row["under_price"]
                                line_display = f"{sub['total'].mean():.1f}" if "total" in sub.columns else "0"
                            else:
                                valid = False; break

                            combined_dec *= american_to_decimal(price)
                            all_lines.append(line_display)

                        if valid and combined_dec > 1.0:
                            pa  = int((combined_dec - 1) * 100) if combined_dec >= 2 else int(-100 / (combined_dec - 1))
                            pa_str = f"+{pa}" if pa > 0 else str(pa)
                            sportsbook_results.append({
                                "Sportsbook":    book,
                                "Decimal Odds":  round(combined_dec, 3),
                                "American Odds": pa_str,
                                "Payout ($)":    f"${stake * combined_dec:,.2f}",
                                "Lines Used":    ", ".join(all_lines),
                            })

                    if not sportsbook_results:
                        st.warning("No sportsbook offers all selected legs.")
                    else:
                        df_results = (
                            pd.DataFrame(sportsbook_results)
                            .assign(_sort=lambda d: d["Decimal Odds"])
                            .sort_values("_sort", ascending=False)
                            .drop(columns=["_sort"])
                        )
                        st.markdown("### Parlay Comparison Across Sportsbooks")
                        st.dataframe(df_results, use_container_width=True)

    # ─────────────────────────────────────────
    # TAB 3 — Player Props
    # ─────────────────────────────────────────
    with tabs[3]:
        st.title("NFL Player Lookup — Team, Player & Stats Search")

        API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
        API_BASE       = "https://v1.american-football.api-sports.io"

        if not API_SPORTS_KEY:
            st.error("Missing API_SPORTS_KEY in environment variables.")
            # FIX: use st.stop() inside a conditional block won't kill other tabs
            # But in Streamlit, st.stop() is global — so use return instead
            return

        HEADERS = {
            "x-rapidapi-key":  API_SPORTS_KEY,
            "x-rapidapi-host": "v1.american-football.api-sports.io",
        }

        NFL_TEAMS = [
            {"id": 1,  "name": "Las Vegas Raiders"},    {"id": 2,  "name": "Jacksonville Jaguars"},
            {"id": 3,  "name": "New England Patriots"},  {"id": 4,  "name": "New York Giants"},
            {"id": 5,  "name": "Baltimore Ravens"},      {"id": 6,  "name": "Tennessee Titans"},
            {"id": 7,  "name": "Detroit Lions"},         {"id": 8,  "name": "Atlanta Falcons"},
            {"id": 9,  "name": "Cleveland Browns"},      {"id": 10, "name": "Cincinnati Bengals"},
            {"id": 11, "name": "Arizona Cardinals"},     {"id": 12, "name": "Philadelphia Eagles"},
            {"id": 13, "name": "New York Jets"},         {"id": 14, "name": "San Francisco 49ers"},
            {"id": 15, "name": "Green Bay Packers"},     {"id": 16, "name": "Chicago Bears"},
            {"id": 17, "name": "Kansas City Chiefs"},    {"id": 18, "name": "Washington Commanders"},
            {"id": 19, "name": "Carolina Panthers"},     {"id": 20, "name": "Buffalo Bills"},
            {"id": 21, "name": "Indianapolis Colts"},    {"id": 22, "name": "Pittsburgh Steelers"},
            {"id": 23, "name": "Seattle Seahawks"},      {"id": 24, "name": "Tampa Bay Buccaneers"},
            {"id": 25, "name": "Miami Dolphins"},        {"id": 26, "name": "Houston Texans"},
            {"id": 27, "name": "New Orleans Saints"},    {"id": 28, "name": "Denver Broncos"},
            {"id": 29, "name": "Dallas Cowboys"},        {"id": 30, "name": "Los Angeles Chargers"},
            {"id": 31, "name": "Los Angeles Rams"},      {"id": 32, "name": "Minnesota Vikings"},
        ]

        team_names    = [t["name"] for t in NFL_TEAMS]
        team_map      = {t["name"]: t["id"] for t in NFL_TEAMS}
        selected_team = st.selectbox("Select NFL Team", team_names)
        team_id       = team_map[selected_team]

        player_name = st.text_input(
            "Enter Player Name (e.g., Baker Mayfield, Stefon Diggs, Bijan Robinson)", ""
        )

        # FIX: replaced st.stop() with early return so other tabs aren't killed
        if not player_name.strip():
            st.info("Enter a player name to continue.")
            return

        with st.spinner("Searching for player..."):
            r = requests.get(f"{API_BASE}/players", headers=HEADERS, params={"name": player_name, "team": team_id})

        if r.status_code != 200:
            st.error(f"Player search failed: {r.status_code}")
            return

        resp = r.json().get("response", [])
        if not resp:
            st.warning("No players found with that name on this team.")
            return

        player_block = resp[0]
        if "player" not in player_block:
            st.error("Unexpected API response. Missing 'player' field.")
            st.json(resp)
            return

        player    = player_block["player"]
        player_id = player.get("id")

        st.subheader(f"Player Found: {player.get('name')}")
        photo_url = player.get("photo") or player.get("image")
        if photo_url:
            st.image(photo_url, width=150)

        st.markdown("---")
        st.subheader("Season Statistics")

        with st.spinner("Fetching player statistics..."):
            r2 = requests.get(
                f"{API_BASE}/players/statistics",
                headers=HEADERS,
                params={"id": player_id, "team": team_id, "season": 2025},
            )

        if r2.status_code != 200:
            st.error(f"Statistics fetch failed: {r2.status_code}")
            return

        stats_data = r2.json().get("response", [])
        if not stats_data:
            st.warning("No statistics available for this player this season.")
            return

        stat_block = stats_data[0]
        if "teams" not in stat_block or not stat_block["teams"]:
            st.warning("No team statistics in response.")
            return

        for g in stat_block["teams"][0].get("groups", []):
            st.markdown(f"### {g.get('name')}")
            clean_dict = {item["name"]: item["value"] for item in g.get("statistics", [])}
            st.dataframe(pd.DataFrame(list(clean_dict.items()), columns=["Stat", "Value"]), use_container_width=True)

        st.markdown("---")
        st.subheader(f"{selected_team} — Season Team Statistics")

        with st.spinner("Fetching team season statistics..."):
            t = requests.get(
                f"{API_BASE}/teams/statistics",
                headers=HEADERS,
                params={"team": team_id, "season": 2025, "league": 12},
            )

        if t.status_code != 200:
            st.error(f"Team statistics fetch failed: {t.status_code}")
            return

        team_stats_raw = t.json().get("response", {})

        # FIX: display team stats as formatted table instead of raw st.json() dump
        if isinstance(team_stats_raw, dict):
            flat = {}
            for k, v in team_stats_raw.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        flat[f"{k} — {k2}"] = v2
                else:
                    flat[k] = v
            if flat:
                st.dataframe(
                    pd.DataFrame(list(flat.items()), columns=["Stat", "Value"]),
                    use_container_width=True,
                )
            else:
                st.info("No team stats available for this season.")
        else:
            st.info("No team stats available for this season.")

        st.success("Player + Team stats loaded successfully.")


if __name__ == "__main__":
    run_app()
