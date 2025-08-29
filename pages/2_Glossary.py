import streamlit as st

st.title("Glossary")

st.write(
    "Plain-English definitions for the terms used in the model and table. "
    "These are short references, not betting advice."
)

terms = [
    ("EV% (Expected Value %)",
     "A quick way to express edge. It compares the best available price to the market’s fair baseline after removing vig. "
     "Positive EV% means the offered price is better than fair; negative means worse."),
    
    ("Kelly Factor",
     "A multiplier on the Kelly stake size used to manage risk. 1.0 is full Kelly, 0.5 is half Kelly, 0.25 is quarter Kelly."),
    
    ("American Odds",
     "U.S. style pricing. Negative numbers (e.g., −140) indicate favorites; positive numbers (e.g., +120) indicate underdogs."),
    
    ("Implied Probability",
     "The chance implied by a price before removing vig. Useful for translating odds into a comparable percentage."),
    
    ("Vig (Margin)",
     "The sportsbook’s built-in margin. In a two-way market, the implied percentages usually add to more than 100% because of the vig."),
    
    ("De-vig / No-vig",
     "Removing the vig from a market to estimate a fair win percentage that’s consistent across books."),
    
    ("Fair Win %",
     "The market-based estimate of a team or outcome’s chance after removing vig. Used as the baseline for EV and Kelly."),
    
    ("Fair Line",
     "The American odds that correspond to the fair win percentage (i.e., the no-vig price)."),
    
    ("Edge",
     "How favorable the offered price is versus the fair baseline. In the app, edge is surfaced as EV%."),
    
    ("Kelly Stake",
     "Suggested stake size derived from fair win %, offered odds, and your Kelly factor. Expressed in units so it scales with bankroll."),
    
    ("Bankroll",
     "Your total betting capital. Kelly sizing and unit sizes are defined relative to this."),
    
    ("Unit",
     "A consistent bet size you track performance with (e.g., 1 unit = $25). The app expresses Kelly suggestions in units."),
    
    ("Markets",
     "Common markets include Moneyline (ML), Spread, Total (Over/Under), and Futures.")
]

for term, definition in terms:
    st.subheader(term)
    st.write(definition)

with st.expander("Quick reference: conversions and formulas"):
    st.markdown(
        """
**American odds → implied probability**

- For negative odds *A* (e.g., −140):  \n
  \\( p = \\frac{-A}{-A + 100} \\)

- For positive odds *A* (e.g., +120):  \n
  \\( p = \\frac{100}{A + 100} \\)

**Implied probability → fair line (American odds)**

- If \\( p > 0.5 \\), fair line is negative:  \n
  \\( \\text{odds} = -\\frac{p}{1-p} \\times 100 \\)

- If \\( p \\le 0.5 \\), fair line is positive:  \n
  \\( \\text{odds} = \\frac{1-p}{p} \\times 100 \\)

**EV% (intuition)**

EV% compares the payout implied by the offered odds to the market's no-vig baseline. Positive means the price pays better than fair; negative means it pays worse.
        """
    )

st.caption("Information only; not betting advice. Bet responsibly.")
