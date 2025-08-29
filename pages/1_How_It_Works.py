import streamlit as st

# Use H1 for the page title (app.py should own set_page_config)
st.title("How It Works")

# Short intro (body)
st.write("A disciplined workflow: screen books, remove the vig, quantify edge, and size with Kelly.")

# Steps (H2 + body only)
steps = [
    ("Step 1: Screen books", "Aggregate lines across sportsbooks and identify the best available price."),
    ("Step 2: De-vig odds", "Remove the bookmaker’s margin to estimate fair win probabilities."),
    ("Step 3: Quantify edge (EV%)", "Compare the best available price to the fair baseline to estimate your edge. Positive EV% means the price is better than fair."),
    ("Step 4: Size with Kelly", "Use fair win % and offered odds to suggest a Kelly fraction so stakes scale with edge and probability."),
    ("Step 5: Filter & act", "Set EV% floors, choose books, and apply your Kelly factor. Edges move with prices—use your judgment and bankroll rules."),
]

for title, body in steps:
    st.header(title)
    st.write(body)

# Disclaimer under the steps
st.caption("Information only; not betting advice. Bet responsibly.")

# Built for every betting style (H2 + body-sized lines)
st.header("Built for every betting style")

profiles = [
    ("Line Shopper",
     "Already have a favorite? Find the best-value sportsbook odds for your bet.",
     "Start here: EV% ≥ 1.0–1.5 · Kelly 0.5"),
    ("Data-Driven Bettor",
     "De-vig the market, turn prices into fair probabilities, and integrate your model with ours.",
     "Start here: EV% ≥ 1.5–2.0 · Kelly 0.75"),
    ("Bankroll-Conscious",
     "Control variance with higher EV floors and lower Kelly stakes.",
     "Start here: EV% ≥ 2–3 · Kelly 0.25"),
    ("High-Conviction Backer",
     "Maximize high-edge opportunities and filter out low-probability bets.",
     "Start here: EV% ≥ 4–5 · Kelly 0.75–1.0"),
    ("Weekend Warriors",
     "Filter by edge and lock in your favorites, every Sunday.",
     "Start here: EV% ≥ 2 · Kelly 0.25–0.75"),
    ("Just Learning",
     "Learn the workflow—fair win probabilities, EV filters, small consistent stakes.",
     "Start here: EV% ≥ 2–3 · Kelly 0.25–0.5"),
]

for name, desc, start in profiles:
    st.markdown(f"**{name}**  \n{desc}  \n{start}")
    st.write("")  # small spacing
