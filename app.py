import os
import time

import pandas as pd
import streamlit as st

from data import make_scenario, make_voyages
from models import contributions, predict_days, train_model

LEADERBOARD = "leaderboard.csv"

st.set_page_config(page_title="The AI Captain", page_icon="⚓")
st.markdown("""
<style>
.stApp {
    background-image:
        linear-gradient(rgba(6, 22, 27, 0.70), rgba(6, 22, 27, 0.70)),
        url("/app/static/ghost-ship.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    """Train once at startup, then reuse."""
    voyages = make_voyages()
    model, test_mae = train_model(voyages)
    return voyages, model, test_mae


def save_result(human_mae, ai_mae):
    row = pd.DataFrame([[time.strftime("%H:%M:%S"), human_mae, ai_mae]],
                       columns=["time", "human_mae", "ai_mae"])
    row.to_csv(LEADERBOARD, mode="a", header=not os.path.exists(LEADERBOARD),
               index=False)


def load_leaderboard():
    if os.path.exists(LEADERBOARD):
        return pd.read_csv(LEADERBOARD)
    return pd.DataFrame(columns=["time", "human_mae", "ai_mae"])


def new_game():
    ss.screen = "play"
    ss.seed = int(time.time() * 1000) % (2**32)  # a new scenario per visitor
    ss.guess = None
    ss.saved = False
    for i in range(3):
        ss.pop(f"g{i}", None)


def go(screen):
    ss.screen = screen
    st.rerun()


ss = st.session_state
if "screen" not in ss:
    new_game()

voyages, model, test_mae = get_model()
scenario = make_scenario(ss.seed)
scenario["ai"] = [predict_days(model, r.distance, r.waves, r.wind)
                  for r in scenario.itertuples()]

st.title("⚓ The AI Captain")
st.caption("The Flying Dutchman must reach a mysterious island. "
           "Can you beat the machine at predicting the voyage?")

# ------------------------------------------------------------------- play
if ss.screen == "play":
    st.header("1. The logbook")
    st.write("The AI will learn from these past voyages. "
             "Look for patterns if you like.")
    st.dataframe(
        voyages.head(6).rename(columns={
            "distance": "Distance (nm)", "waves": "Waves (m)", "wind": "Headwind (knots)",
            "days": "Days taken"}),
        hide_index=True)
    st.caption(f"First 6 of {len(voyages)} voyages.")

    st.header("2. Today’s routes")
    st.write("The Flying Dutchman must reach a mysterious island. "
             "Today’s conditions:")
    st.dataframe(
        scenario[["route", "distance", "waves", "wind"]].rename(columns={
            "route": "Route", "distance": "Distance (nm)",
            "waves": "Waves (m)", "wind": "Headwind (knots)"}),
        hide_index=True)

    st.header("3. Your guess")
    st.write("How many days will each route take? Trust your intuition.")
    guesses = [st.slider(f"{name} (days)", 5.0, 25.0, 12.0, 0.5, key=f"g{i}")
               for i, name in enumerate(scenario["route"])]
    if st.button("Lock in my guesses", type="primary"):
        ss.guess = guesses
        go("ai")

# --------------------------------------------------------------------- ai
elif ss.screen == "ai":
    st.header("The AI captain predicts")
    b0, (bd, bw, bh) = model.intercept_, model.coef_
    st.write(f"The AI studied {len(voyages)} past voyages and fitted this rule "
             "(linear regression, least squares):")
    st.code(f"days = {b0:.2f} + {bd:.3f}·distance + {bw:.2f}·waves + {bh:.2f}·headwind")
    parts = pd.DataFrame(
        [contributions(model, r.distance, r.waves, r.wind)
         for r in scenario.itertuples()],
        columns=["Base", "Distance", "Waves", "Headwind"],
        index=scenario["route"])
    for r in scenario.itertuples():
        st.write(f"**{r.route}**: {r.ai:.1f} days")
    st.bar_chart(parts)
    st.caption("Each color is the number of days one factor adds to the voyage. "
               "A short route with big waves can be slower than a long calm one.")
    if st.button("Reveal the truth", type="primary"):
        go("reveal")

# ----------------------------------------------------------------- reveal
elif ss.screen == "reveal":
    st.header("The reveal")
    guess = ss.guess
    table = pd.DataFrame({
        "Route": scenario["route"],
        "You": guess,
        "AI": scenario["ai"].round(1),
        "Truth": scenario["truth"],
    })
    table["Your error"] = (table["You"] - table["Truth"]).abs().round(1)
    table["AI error"] = (table["AI"] - table["Truth"]).abs().round(1)
    st.dataframe(table, hide_index=True)
    human_mae = float(table["Your error"].mean())
    ai_mae = float(table["AI error"].mean())
    c1, c2 = st.columns(2)
    c1.metric("Your mean error (days)", f"{human_mae:.2f}")
    c2.metric("AI mean error (days)", f"{ai_mae:.2f}")
    st.caption("Mean absolute error = (|e₁| + |e₂| + |e₃|) / 3. ")
    if not ss.saved:
        save_result(human_mae, ai_mae)
        ss.saved = True
    ss.human_mae, ss.ai_mae = human_mae, ai_mae
    if st.button("See the leaderboard", type="primary"):
        go("result")

# ----------------------------------------------------------------- result
elif ss.screen == "result":
    st.header("Leaderboard")
    board = load_leaderboard()
    n = len(board)
    avg_h, avg_ai = board["human_mae"].mean(), board["ai_mae"].mean()
    st.write(f"Average error in days (lower is better). **{n}** sailors so far.")
    st.bar_chart(pd.DataFrame(
        {"Mean error (days)": [ss.human_mae, avg_h, avg_ai]},
        index=["You", "Average visitor", "AI captain"]))
    worse = int((board["human_mae"] > ss.human_mae).sum())
    if ss.human_mae < ss.ai_mae:
        st.success("You beat the AI captain this time!")
    else:
        st.info(f"The AI captain won, but you did better than {worse} of "
                f"{n} sailors.")

    if st.button("Next sailor", type="primary"):
        new_game()
        st.rerun()