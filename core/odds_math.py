# core/odds_math.py
# Pure odds math and formatting helpers — no Streamlit, no pipeline logic.
# Shared by core/pipeline.py and every tab.
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


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
    md = f"{et.month}/{et.day}"
    tm = et.strftime("%I:%M %p").lstrip("0")
    return f"{dow} {md} {tm} ET"


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
