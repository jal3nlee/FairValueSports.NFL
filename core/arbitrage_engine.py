# core/arbitrage_engine.py
# Arbitrage detection built entirely on top of the existing pipeline
# OUTPUT (run_market_pipeline's pivoted dataframe) — not a new raw-odds
# fetch. That output already carries the best price per side across ALL
# books for every market, which is exactly what arbitrage needs: best
# price on each side, possibly from two different sportsbooks.
import pandas as pd


def american_to_decimal(odds) -> float | None:
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    return (100 / abs(odds)) + 1 if odds < 0 else (odds / 100) + 1


def implied_probability(odds) -> float | None:
    dec = american_to_decimal(odds)
    return 1 / dec if dec else None


def find_arbitrage_opportunities(df_pipeline_output: pd.DataFrame, bankroll: float) -> list[dict]:
    """
    df_pipeline_output: the pivoted output of run_market_pipeline (same
    shape used across Matchup Center / Fair Value Model), with columns
    Game, commence_time, Market, Pick, Line, Best Odds, Best Book.

    Only evaluates groups with exactly 2 distinct picks (a clean
    two-sided market) — ambiguous groups are skipped, not guessed at.
    """
    if df_pipeline_output.empty:
        return []

    required = {"Game", "commence_time", "Market", "Pick", "Best Odds", "Best Book"}
    if not required.issubset(df_pipeline_output.columns):
        return []

    _line_col = df_pipeline_output["Line"] if "Line" in df_pipeline_output.columns else pd.Series([None] * len(df_pipeline_output))
    df = df_pipeline_output.copy()
    df["_line_key"] = _line_col.fillna("—").astype(str)

    opportunities = []
    for (game, commence, market, line_key), group in df.groupby(["Game", "commence_time", "Market", "_line_key"]):
        group = group.drop_duplicates(subset=["Pick"])
        if len(group) != 2:
            continue

        rows = group.to_dict("records")
        probs, decimals = [], []
        valid = True
        for r in rows:
            p = implied_probability(r.get("Best Odds"))
            d = american_to_decimal(r.get("Best Odds"))
            if p is None or d is None:
                valid = False
                break
            probs.append(p)
            decimals.append(d)
        if not valid:
            continue

        total_implied = sum(probs)
        if total_implied >= 1.0:
            continue  # no arbitrage — books' combined price exceeds 100%

        profit_pct = (1 / total_implied - 1) * 100
        stakes = [bankroll * (p / total_implied) / p * (1 / total_implied) for p in probs]
        # equal-payout stake split: stake_i proportional to 1/decimal_i, normalized
        raw_stakes = [1 / d for d in decimals]
        stake_total = sum(raw_stakes)
        stakes = [bankroll * (s / stake_total) for s in raw_stakes]
        payout = stakes[0] * decimals[0]  # equal for both legs by construction

        opportunities.append({
            "game": game,
            "commence_time": commence,
            "market": market,
            "line": None if line_key == "—" else line_key,
            "legs": [
                {"pick": r.get("Pick"), "odds": r.get("Best Odds"), "book": r.get("Best Book"), "stake": round(stakes[i], 2)}
                for i, r in enumerate(rows)
            ],
            "profit_pct": round(profit_pct, 2),
            "guaranteed_profit": round(payout - bankroll, 2),
        })

    opportunities.sort(key=lambda o: o["profit_pct"], reverse=True)
    return opportunities
