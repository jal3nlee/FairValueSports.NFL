# core/fantasy_data.py
# Data loading/cleaning for the Fantasy Draft tab. Kept separate from the
# UI on purpose — the source file can later be swapped for an API feed or
# a database without tabs/fantasy_draft.py needing to change at all.
import re
import pandas as pd
import streamlit as st

# Columns in the source file that are NOT a ranking platform. Everything
# else in the ORIGINAL file header is treated as a platform column
# automatically — captured before any cleaning step adds derived fields
# like Name/Team/Bye/Position, so those never get mistaken for a source.
_META_COLS = {"Rank", "Player (Bye)", "POS", "AVG", "Real-Time"}

_PLAYER_FIELD_RE = re.compile(r"^(.*?)\s{2,}([A-Z]{2,4})\s*(?:\((\d+)\))?\s*$")
_BYE_ONLY_RE = re.compile(r"^(.*?)\s*\((\d+)\)\s*$")
_POS_RE = re.compile(r"^([A-Za-z]+?)(\d*)$")


def get_platform_columns(raw_df: pd.DataFrame) -> list[str]:
    """Must be called on the ORIGINAL, un-cleaned dataframe — clean_fantasy_rankings
    adds columns (Name, Team, Bye, Position, PositionRankRaw) that must never
    be mistaken for ranking platforms."""
    return [c for c in raw_df.columns if c not in _META_COLS]


def _parse_player_field(raw) -> tuple[str | None, str | None, str | None]:
    """
    Splits the combined "Player   TEAM (Bye)" field into (name, team, bye).
    Returns (None, None, None) for a genuinely blank/NaN cell so callers
    can drop that row entirely rather than ever rendering "nan".
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None, None
    raw = str(raw).strip()
    if not raw or raw.lower() == "nan":
        return None, None, None
    m = _PLAYER_FIELD_RE.match(raw)
    if m:
        name, team, bye = m.groups()
        return name.strip(), team, bye
    m2 = _BYE_ONLY_RE.match(raw)
    if m2:
        return m2.group(1).strip(), None, m2.group(2)
    return raw, None, None


def _fmt_bye(bye) -> str | None:
    """Always a clean whole number as text — never '6.0'."""
    if bye is None or (isinstance(bye, float) and pd.isna(bye)):
        return None
    try:
        return str(int(float(bye)))
    except (ValueError, TypeError):
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_fantasy_rankings(csv_path: str = "data/fantasy_adp.xlsx") -> pd.DataFrame:
    try:
        if csv_path.endswith((".xlsx", ".xls")):
            return pd.read_excel(csv_path)
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def clean_fantasy_rankings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    platform_cols = get_platform_columns(df)  # capture BEFORE adding any new columns

    parsed = df["Player (Bye)"].apply(_parse_player_field)
    df["Name"] = [p[0] for p in parsed]
    df["Team"] = [p[1] for p in parsed]
    df["Bye"] = [_fmt_bye(p[2]) for p in parsed]

    pos_parsed = df["POS"].fillna("").astype(str).apply(
        lambda s: _POS_RE.match(s.strip()).groups() if _POS_RE.match(s.strip()) else ("", "")
    )
    df["Position"] = [p[0] for p in pos_parsed]
    df["PositionRankRaw"] = [p[1] for p in pos_parsed]

    for col in platform_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no real player name at all — blank trailing rows in
    # the source file, not actual players.
    df = df[df["Name"].notna() & (df["Name"].astype(str).str.strip() != "")]

    df = df.drop_duplicates(subset=["Name", "Team"], keep="first").reset_index(drop=True)
    return df


def calculate_consensus_adp(df: pd.DataFrame, platform_cols: list[str]) -> pd.Series:
    """Mean of available platform values only — a missing platform is never treated as 0."""
    return df[platform_cols].mean(axis=1, skipna=True).round(1)


def calculate_position_rank(df: pd.DataFrame) -> pd.Series:
    """Positional rank computed from consensus ADP within each position group.
    Players with no valid platform data (AvgADP is NaN) get no positional
    rank rather than crashing on an int cast."""
    rank_within_pos = df.groupby("Position")["AvgADP"].rank(method="first")
    rank_str = rank_within_pos.apply(lambda x: str(int(x)) if pd.notna(x) else "")
    return df["Position"].fillna("") + rank_str


def build_ranking_table(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Runs the full clean -> consensus -> position-rank pipeline once."""
    if raw_df.empty:
        return raw_df, []
    platform_cols = get_platform_columns(raw_df)  # from the RAW file, not the cleaned one
    df = clean_fantasy_rankings(raw_df)
    if df.empty:
        return df, platform_cols

    df["AvgADP"] = calculate_consensus_adp(df, platform_cols)
    df["PosRank"] = calculate_position_rank(df)

    df = df.sort_values("AvgADP", ascending=True, na_position="last").reset_index(drop=True)
    df["ConsensusRank"] = df.index + 1
    return df, platform_cols


def filter_fantasy_rankings(df: pd.DataFrame, position: str = "Overall", search_text: str = "") -> pd.DataFrame:
    out = df
    if position and position != "Overall":
        out = out[out["Position"] == position]
    if search_text and search_text.strip():
        q = search_text.strip().lower()
        out = out[
            out["Name"].str.lower().str.contains(q, na=False)
            | out["Team"].fillna("").str.lower().str.contains(q, na=False)
        ]
    return out.reset_index(drop=True)
