import streamlit as st

# Title (H1) + short intro
st.title("How It Works")
st.markdown(
    "<p class='fvb-muted'>A disciplined workflow: screen books, remove the vig, "
    "quantify edge, and size with Kelly.</p>",
    unsafe_allow_html=True,
)

# Minimal, scoped CSS for cards, chips, spacing (no new font sizes)
st.markdown(
    """
<style>
.fvb-muted { color: #475569; margin-top: .25rem; }

.fvb-grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
@media (min-width: 880px) { .fvb-grid { grid-template-columns: 1fr 1fr; } }

.fvb-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; }
.fvb-chip { display:inline-block; padding: 2px 10px; border-radius: 9999px; background: #f1f5f9; color:#475569; font-weight: 600; margin-bottom: 8px; }
.fvb-hr { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0 12px; }

.fvb-card h2 { margin: 4px 0 6px; }    /* use H2 size (Streamlit theme handles actual size) */
.fvb-card p  { margin: 0; }
</style>
""",
    unsafe_allow_html=True,
)

# --- Steps (cards in a responsive grid) ---
steps = [
    ("Step 1", "Screen books", "Aggregate lines across sportsbooks and identify the best available price."),
    ("Step 2", "De-vig odds", "Remove the bookmaker’s margin to estimate fair win probabilities."),
    ("Step 3", "Quantify edge (EV%)", "Compare the best available price to the fair baseline to estimate your edge. Positive EV% means the price is better than fair."),
    ("Step 4", "Size with Kelly", "Use fair win % and offered odds to suggest a Kelly fraction so stakes scale with edge and probability."),
    ("Step 5", "Filter & act", "Set EV% floors, choose books, and apply your Kelly factor. Edges move with prices—use your judgment and bankroll rules."),
]

def step_card_html(step_label: str, title: str, body: str) -> str:
    return f"""
    <div class="fvb-card">
      <span class="fvb-chip">{step_label}</span>
      <h2>{title}</h2>
      <p>{body}</p>
    </div>
    """

st.markdown("<div class='fvb-grid'>", unsafe_allow_html=True)
for step_label, title, body in steps:
    st.markdown(step_card_html(step_label, title, body), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Disclaimer (small text only here)
st.caption("Information only; not betting advice. Bet responsibly.")

# Divider before profiles
st.markdown("<hr class='fvb-hr' />", unsafe_allow_html=True)

# --- Built for every betting style (cards, same visual language) ---
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

def profile_card_html(name: str, desc: str, start_line: str) -> str:
    # Body-sized “Start here” line, with bold label for scannability
    return f"""
    <div class="fvb-card">
      <h2>{name}</h2>
      <p>{desc}</p>
      <p><strong>Start here:</strong> {start_line.split("Start here:")[-1].strip()}</p>
    </div>
    """

st.markdown("<div class='fvb-grid'>", unsafe_allow_html=True)
for name, desc, start_line in profiles:
    st.markdown(profile_card_html(name, desc, start_line), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
