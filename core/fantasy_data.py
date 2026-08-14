# core/fantasy_data.py
import re
import pandas as pd
import streamlit as st

_META_COLS = {"Rank", "Player (Bye)", "POS", "AVG", "Real-Time"}

_PLAYER_FIELD_RE = re.compile(r"^(.*?)\s{2,}([A-Z]{2,4})\s*(?:\((\d+)\))?\s*$")
_BYE_ONLY_RE = re.compile(r"^(.*?)\s*\((\d+)\)\s*$")
_POS_RE = re.compile(r"^([A-Za-z]+?)(\d*)$")

SHEET_NAME_MAP = {"PPR": "PPR", "Half PPR": "Half PPR", "Standard": "STD"}


def get_platform_columns(raw_df: pd.DataFrame) -> list[str]:
    return [c for c in raw_df.columns if c not in _META_COLS]


def _parse_player_field(raw) -> tuple[str | None, str | None, str | None]:
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
    if bye is None or (isinstance(bye, float) and pd.isna(bye)):
        return None
    try:
        return str(int(float(bye)))
    except (ValueError, TypeError):
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_fantasy_rankings(scoring: str = "PPR", csv_path: str = "data/fantasy_adp.xlsx") -> pd.DataFrame:
    try:
        if csv_path.endswith((".xlsx", ".xls")):
            sheet = SHEET_NAME_MAP.get(scoring, scoring)
            return pd.read_excel(csv_path, sheet_name=sheet)
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def clean_fantasy_rankings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    platform_cols = get_platform_columns(df)

    parsed = df["Player (Bye)"].apply(_parse_player_field)
    df["Name"] = [p[0] for p in parsed]
    df["Team"] = [p[1] for p in parsed]
    df["Bye"] = [_fmt_bye(p[2]) for p in parsed]

    pos_parsed = df["POS"].fillna("").astype(str).apply(
        lambda s: _POS_RE.match(s.strip()).groups() if _POS_RE.match(s.strip()) else ("", "")
    )
    df["Position"] = [p[0] for p in pos_parsed]

    for col in platform_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Name"].notna() & (df["Name"].astype(str).str.strip() != "")]
    df = df.drop_duplicates(subset=["Name", "Team"], keep="first").reset_index(drop=True)

    # Stable identity — this is what drafted-state tracking hangs off of.
    # Never the dataframe's row index, which changes with every sort/filter.
    df["PlayerID"] = df["Name"] + "|" + df["Team"].fillna("")

    return df


def calculate_consensus_adp(df: pd.DataFrame, platform_cols: list[str]) -> pd.Series:
    return df[platform_cols].mean(axis=1, skipna=True).round(1)


def build_ranking_table(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if raw_df.empty:
        return raw_df, []
    platform_cols = get_platform_columns(raw_df)
    df = clean_fantasy_rankings(raw_df)
    if df.empty:
        return df, platform_cols

    df["AvgADP"] = calculate_consensus_adp(df, platform_cols)
    df = df.sort_values("AvgADP", ascending=True, na_position="last").reset_index(drop=True)
    df["ConsensusRank"] = df.index + 1  # fixed at build time — never renumbered after drafting
    return df, platform_cols


def filter_fantasy_rankings(df: pd.DataFrame, position: str = "Overall") -> pd.DataFrame:
    if position and position != "Overall":
        return df[df["Position"] == position].reset_index(drop=True)
    return df.reset_index(drop=True)
