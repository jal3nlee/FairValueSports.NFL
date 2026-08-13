# core/fantasy_data.py
# Data loading/cleaning for the Fantasy Draft tab. Kept separate from the
# UI on purpose — the source file can later be swapped for an API feed or
# a database without tabs/fantasy_draft.py needing to change at all.
import re
import pandas as pd
import streamlit as st

# Columns in the source file that are NOT a ranking platform. Everything
# else in the header is treated as a platform column automatically, so a
# future file with different/additional sources picks up without code
# changes — only this exclusion list would ever need updating if the
# provider renames one of these specific fields.
_META_COLS = {"Rank", "Player (Bye)", "POS", "AVG", "Real-Time"}

_PLAYER_FIELD_RE = re.compile(r"^(.*?)\s{2,}([A-Z]{2,4})\s*(?:\((\d+)\))?\s*$")
_BYE_ONLY_RE = re.compile(r"^(.*?)\s*\((\d+)\)\s*$")
_POS_RE = re.compile(r"^([A-Za-z]+?)(\d*)$")


def get_platform_columns(df: pd.DataFrame) -> list[str]:
    """Every column that isn't a known meta field is treated as a ranking source."""
    return [c for c in df.columns if c not in _META_COLS]


def _parse_player_field(raw: str) -> tuple[str, str | None, str | None]:
    """
    Splits the combined "Player   TEAM (Bye)" field into (name, team, bye).
    Handles three real shapes seen in the data:
      - "Jahmyr Gibbs   DET (6)"      -> normal player
      - "Houston Texans DST   (8)"    -> team name is part of the player name, no separate code
      - "Jermaine Jackson"            -> deep-bench player with no team/bye at all
    """
    raw = (raw or "").strip()
    if not raw:
        return "—", None, None
    m = _PLAYER_FIELD_RE.match(raw)
    if m:
        name, team, bye = m.groups()
        return name.strip(), team, bye
    m2 = _BYE_ONLY_RE.match(raw)
    if m2:
        return m2.group(1).strip(), None, m2.group(2)
    return raw, None, None


@st.cache_data(ttl=3600, show_spinner=False)
def load_fantasy_rankings(csv_path: str = "data/fantasy_adp.xlsx") -> pd.DataFrame:
    try:
        if csv_path.endswith((".xlsx", ".xls")):
            return pd.read_excel(csv_path)
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def clean_fantasy_rankings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()

    parsed = df["Player (Bye)"].apply(_parse_player_field)
    df["Name"] = [p[0] for p in parsed]
    df["Team"] = [p[1] for p in parsed]
    df["Bye"] = [p[2] for p in parsed]

    pos_parsed = df["POS"].fillna("").apply(
        lambda s: _POS_RE.match(s.strip()).groups() if _POS_RE.match(s.strip()) else ("", "")
    )
    df["Position"] = [p[0] for p in pos_parsed]
    df["PositionRankRaw"] = [p[1] for p in pos_parsed]

    for col in get_platform_columns(df):
        df[col] = pd.to_numeric(df[col], errors="coerce")  # non-numeric/blank -> NaN, i.e. null, never 0

    # Drop exact duplicate player rows (same name + team) rather than
    # letting a repeated row silently double-count in any averaging.
    df = df.drop_duplicates(subset=["Name", "Team"], keep="first").reset_index(drop=True)

    return df


def calculate_consensus_adp(df: pd.DataFrame, platform_cols: list[str]) -> pd.Series:
    """Mean of available platform values only — a missing platform is never treated as 0."""
    return df[platform_cols].mean(axis=1, skipna=True).round(1)


def calculate_platform_range(df: pd.DataFrame, platform_cols: list[str]) -> pd.Series:
    """max - min across whatever platforms actually have a value for that player."""
    return (df[platform_cols].max(axis=1, skipna=True) - df[platform_cols].min(axis=1, skipna=True))


def calculate_position_rank(df: pd.DataFrame) -> pd.Series:
    """Positional rank computed from consensus ADP within each position group —
    independent of the CSV's own POS numbering, so it stays correct even
    if the file is filtered or a future source numbers things differently."""
    rank_within_pos = df.groupby("Position")["AvgADP"].rank(method="first").astype(int)
    return df["Position"] + rank_within_pos.astype(str)


def disagreement_thresholds(df: pd.DataFrame) -> tuple[float, float]:
    """Data-driven thresholds instead of hardcoded numbers — based on the
    actual distribution of Range in this file."""
    valid = df[df["_num_platforms"] >= 2]["Range"].dropna()
    if valid.empty:
        return (999.0, 999.0)
    moderate = valid.quantile(0.75)
    high = valid.quantile(0.90)
    return (moderate, high)


def build_ranking_table(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Runs the full clean -> consensus -> range -> position-rank pipeline once."""
    df = clean_fantasy_rankings(raw_df)
    if df.empty:
        return df, []
    platform_cols = get_platform_columns(df)
    df["AvgADP"] = calculate_consensus_adp(df, platform_cols)
    df["Range"] = calculate_platform_range(df, platform_cols)
    df["_num_platforms"] = df[platform_cols].notna().sum(axis=1)
    df["PosRank"] = calculate_position_rank(df)

    moderate_th, high_th = disagreement_thresholds(df)
    def _flag(row):
        if row["_num_platforms"] < 2 or pd.isna(row["Range"]):
            return ""
        if row["Range"] >= high_th:
            return "High Disagreement"
        return ""
    df["Agreement"] = df.apply(_flag, axis=1)

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
