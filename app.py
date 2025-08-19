import os, math, requests
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

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

def fmt_date_est_str(iso_s: str):
    dt = parse_iso_dt_utc(iso_s)
    if not dt: return None
    et = dt.astimezone(EASTERN)
    dow = et.strftime("%a")              # e.g., Thu
    md  = f"{et.month}/{et.day}"         # e.g., 8/14
    tm  = et.strftime("%I:%M %p").lstrip("0")  # e.g., 7:30 PM
    return f"{dow} {md} {tm} ET"

# ---------- HARD-CODED NFL WEEK CALENDAR ----------
def nfl_week1_kickoff_thursday_utc(year: int) -> datetime:
    """Hard-code Week 1 kickoff as Sep 4 (00:00 UTC) for the given year."""
    return datetime(year, 9, 4, 0, 0, 0, tzinfo=timezone.utc)

def nfl_week_window_utc(week_index: int, now_utc: datetime):
    """
    Week 0: any date before Sep 4 this year (Jan 1 00:00 -> Sep 3 23:59:59 UTC).
    Week >=1: Thu 00:00 -> Tue 23:59:59 blocks starting Sep 4.
    """
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
    if now_utc < wk1:
        return 0
    weeks = (now_utc - wk1).days // 7 + 1
    return max(1, min(19, weeks))

def sport_key_for_week(week_index: int) -> str:
    return "americanfootball_nfl_preseason" if week_index == 0 else "americanfootball_nfl"

# ---------- odds shaping ----------
def pretty_book_title(bm: dict) -> str:
    # Prefer API's 'title'; fallback to cleaned 'key'
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
                "book": pretty_book_title(bm),  # use nice title
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

# ---------- UI ----------
st.set_page_config(page_title="Fair Value Sports - NFL", page_icon="🏈", layout="wide")
st.title("NFL Market EV Model")

if not API_KEY:
    st.error("Missing ODDS_API_KEY environment variable."); st.stop()

region = "us"

# Dropdown with exactly: Today, Week X (X = current hard-coded week index)
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
c1, c2 = st.columns(2)
with c1:
    weekly_bankroll = st.number_input("Weekly Bankroll ($)", min_value=0.0, value=1000.0, step=50.0)
with c2:
    kelly_factor = st.slider("Kelly Factor (0.0–1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# TEMP: show-all toggle (if OFF, keep only EV% >= 0)
show_all = st.checkbox("Show all games (ignore EV filter)", value=True)

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

# Build output rows (initially include all)
rows = []
for _, r in merged.iterrows():
    date_str = fmt_date_est_str(r.get("commence_time"))
    game_label = f"{r['home_team']} vs {r['away_team']}"

    # Home side
    if pd.notna(r.get("home_price")) and pd.notna(r.get("home_fair")):
        fair_p, price = float(r["home_fair"]), int(r["home_price"])
        ev_pct = expected_value_pct(fair_p, price)
        kelly = kelly_fraction(fair_p, price)
        rows.append({
            "Date": date_str,
            "Game": game_label,
            "Pick": r["home_team"],
            "Best Odds": price,                  # numeric; we'll pretty-print with + later
            "Best Book": r.get("home_book"),
            "Implied Probability": fair_p,       # decimal now; show as % later
            "EV%": ev_pct,
            "Kelly %": kelly*100.0,
            "Stake ($)": round(weekly_bankroll * kelly_factor * kelly, 2)
        })

    # Away side
    if pd.notna(r.get("away_price")) and pd.notna(r.get("away_fair")):
        fair_p, price = float(r["away_fair"]), int(r["away_price"])
        ev_pct = expected_value_pct(fair_p, price)
        kelly = kelly_fraction(fair_p, price)
        rows.append({
            "Date": date_str,
            "Game": game_label,
            "Pick": r["away_team"],
            "Best Odds": price,
            "Best Book": r.get("away_book"),
            "Implied Probability": fair_p,
            "EV%": ev_pct,
            "Kelly %": kelly*100.0,
            "Stake ($)": round(weekly_bankroll * kelly_factor * kelly, 2)
        })

df = pd.DataFrame(rows)
if df.empty:
    st.info("No sides available."); st.stop()

# Apply temp EV filter if requested (EV% >= 0)
if not show_all:
    df = df[df["EV%"] >= 0].reset_index(drop=True)
    if df.empty:
        st.info("No games meet EV% ≥ 0 with this window."); st.stop()

# Sort
df = df.sort_values(["EV%","Implied Probability"], ascending=[False, False]).reset_index(drop=True)

# Pretty formatting for display
show = df.copy()

def fmt_odds(o):
    try:
        o = int(o)
        return f"+{o}" if o > 0 else str(o)
    except Exception:
        return str(o)

show["Best Odds"] = show["Best Odds"].map(fmt_odds)
show["Best Book"] = show["Best Book"].astype(str)
show["Implied Probability"] = show["Implied Probability"].map(lambda x: f"{x*100:.1f}%")
show["Kelly %"] = show["Kelly %"].map(lambda x: f"{x:.2f}")

# Render
st.caption(f"Window: {caption_label}")
st.subheader("Games & EV")
st.dataframe(
    show[["Date","Game","Pick","Best Odds","Best Book","Implied Probability","EV%","Kelly %","Stake ($)"]],
    use_container_width=True,
    hide_index=True,
)

# Totals
total_stake = float(df["Stake ($)"].sum())
util_pct = 100.0 * (total_stake / weekly_bankroll) if weekly_bankroll > 0 else 0.0
color = "green" if util_pct < 50 else ("orange" if util_pct < 70 else "red")
st.markdown(
    f"**Total Suggested Stake:** ${total_stake:,.2f}  |  **Utilization:** "
    f"<span style='color:{color}'>{util_pct:.1f}% of weekly bankroll</span>",
    unsafe_allow_html=True
)
