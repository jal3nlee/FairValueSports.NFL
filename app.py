import os, math, requests
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta

API_BASE = "https://api.the-odds-api.com/v4"
API_KEY  = os.getenv("ODDS_API_KEY", "")

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

def fmt_date_utc_str(iso_s: str):
    dt = parse_iso_dt_utc(iso_s)
    return dt.strftime('%Y-%m-%d %H:%M UTC') if dt else None

# ---------- NFL week calendar ----------
def first_monday_of_september_utc(year: int) -> datetime:
    d = datetime(year, 9, 1, tzinfo=timezone.utc)
    delta = (0 - d.weekday()) % 7  # Monday=0
    return d + timedelta(days=delta)

def nfl_week1_kickoff_thursday_utc(year: int) -> datetime:
    # NFL kickoff is the Thursday after the first Monday in September
    mon = first_monday_of_september_utc(year)
    return mon + timedelta(days=3)   # Thursday 00:00 UTC

def nfl_week_window_utc(week_index: int, now_utc: datetime) -> tuple[datetime, datetime]:
    """
    Week 0 = the 7-day block BEFORE Week 1 (preseason).
    Weeks 1–18 = regular season (Thu–Tue). Week 19 = first postseason block after Week 18.
    Window spans Thu 00:00 UTC -> Tue 23:59:59 UTC.
    """
    wk1 = nfl_week1_kickoff_thursday_utc(now_utc.year)
    if week_index == 0:
        start = wk1 - timedelta(days=7)
    else:
        start = wk1 + timedelta(days=7*(week_index-1))
    end = start + timedelta(days=5, hours=23, minutes=59, seconds=59)
    return start, end

def infer_current_week_index(now_utc: datetime) -> int:
    wk1 = nfl_week1_kickoff_thursday_utc(now_utc.year)
    if now_utc < wk1:
        return 0  # the week before Week 1
    weeks = int((now_utc - wk1).days // 7) + 1  # Week 1..∞
    return max(0, min(19, weeks))  # clamp to 19

def sport_key_for_week(week_index: int) -> str:
    return "americanfootball_nfl_preseason" if week_index == 0 else "americanfootball_nfl"

# odds shaping
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
                "book": bm.get("key"), "home_price": price_home, "away_price": price_away,
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

# Dropdown with exactly: Today, Week X (X is current week)
now = datetime.now(timezone.utc)
current_week = infer_current_week_index(now)
window_options = ["Today", f"Week {current_week}"]
window_choice = st.selectbox("Window", window_options, index=1)

if window_choice == "Today":
    window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    window_end   = window_start + timedelta(hours=23, minutes=59, seconds=59)
    # If today is before Week 1, use preseason key; else regular key
    sport_key = sport_key_for_week(0 if current_week == 0 else current_week)
    caption_label = f"Today ({window_start.strftime('%Y-%m-%d')} UTC)"
else:
    # Parse week index from label "Week X"
    week_index = current_week
    window_start, window_end = nfl_week_window_utc(week_index, now)
    sport_key = sport_key_for_week(week_index)
    caption_label = f"NFL Week {week_index} — {window_start.strftime('%Y-%m-%d')} → {window_end.strftime('%Y-%m-%d')} UTC"

# Inputs
c1, c2, c3 = st.columns(3)
with c1:
    weekly_bankroll = st.number_input("Weekly Bankroll ($)", min_value=0.0, value=1000.0, step=50.0)
with c2:
    kelly_factor = st.slider("Kelly Factor (0.0–1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
with c3:
    min_ev = st.number_input("Minimum EV% to display", value=4.0, step=0.5)

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

rows = []
for _, r in merged.iterrows():
    date_str = fmt_date_utc_str(r.get("commence_time"))
    event_label = f"{r['home_team']} vs {r['away_team']}"
    # home
    if pd.notna(r.get("home_price")) and pd.notna(r.get("home_fair")):
        fair_p, price = float(r["home_fair"]), int(r["home_price"])
        ev_pct = expected_value_pct(fair_p, price)
        kelly = kelly_fraction(fair_p, price)
        if ev_pct >= float(min_ev):
            rows.append({
                "Date": date_str, "Event": event_label, "Pick": r["home_team"], "Side": "Home",
                "Best Odds": price, "Best Book": r.get("home_book"),
                "Fair Prob": fair_p, "EV%": ev_pct, "Kelly %": kelly*100.0,
                "Stake ($)": round(weekly_bankroll * kelly_factor * kelly, 2)
            })
    # away
    if pd.notna(r.get("away_price")) and pd.notna(r.get("away_fair")):
        fair_p, price = float(r["away_fair"]), int(r["away_price"])
        ev_pct = expected_value_pct(fair_p, price)
        kelly = kelly_fraction(fair_p, price)
        if ev_pct >= float(min_ev):
            rows.append({
                "Date": date_str, "Event": event_label, "Pick": r["away_team"], "Side": "Away",
                "Best Odds": price, "Best Book": r.get("away_book"),
                "Fair Prob": fair_p, "EV%": ev_pct, "Kelly %": kelly*100.0,
                "Stake ($)": round(weekly_bankroll * kelly_factor * kelly, 2)
            })

df = pd.DataFrame(rows)
if df.empty:
    st.info("No picks pass the EV% filter."); st.stop()

df = df.sort_values(["EV%","Fair Prob"], ascending=[False, False]).reset_index(drop=True)

total_stake = float(df["Stake ($)"].sum())
util_pct = 100.0 * (total_stake / weekly_bankroll) if weekly_bankroll > 0 else 0.0

st.caption(f"Window: {caption_label}")
st.subheader("Qualified Picks")

show = df.copy()
show["Fair Prob"] = show["Fair Prob"].map(lambda x: f"{x:.3f}")
show["Kelly %"] = show["Kelly %"].map(lambda x: f"{x:.2f}")
st.dataframe(
    show[["Date","Event","Pick","Side","Best Odds","Best Book","Fair Prob","EV%","Kelly %","Stake ($)"]],
    use_container_width=True,
    hide_index=True,
)

color = "green" if util_pct < 50 else ("orange" if util_pct < 70 else "red")
st.markdown(
    f"**Total Suggested Stake:** ${total_stake:,.2f}  |  **Utilization:** "
    f"<span style='color:{color}'>{util_pct:.1f}% of weekly bankroll</span>",
    unsafe_allow_html=True
)
