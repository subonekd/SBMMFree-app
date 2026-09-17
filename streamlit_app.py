import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SBMMFree", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem; padding-bottom: 4rem;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.18); padding: 14px; border-radius: 12px;}
.small-note {opacity:.72; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("SBMMFree")
st.subheader("What happens when you change how a Warzone lobby is built?")
st.caption("An interactive matchmaking model — not a reconstruction of Activision's matchmaking code.")

# Historical/extrapolated population anchors. The important distinction is that ~0.78
# represents the middle/common player, while ~0.92–0.98 is an arithmetic/account-level
# average and should NOT be used as the center of the player distribution.
PERCENTILES = np.array([2, 17, 27, 38, 50, 62, 66, 79, 87, 91, 94, 97, 99, 99.9, 99.98, 99.99])
KD_ANCHORS = np.array([0.15, 0.40, 0.55, 0.62, 0.78, 0.92, 0.97, 1.12, 1.30, 1.45, 1.58, 1.85, 2.08, 3.57, 4.40, 5.70])


def population_sample(n, rng):
    """Sample from a smooth quantile curve fitted to historical/extrapolated anchors."""
    p = rng.uniform(0.1, 99.99, n)
    # Add endpoints so interpolation behaves sensibly at the tails.
    xp = np.r_[0.1, PERCENTILES, 100.0]
    fp = np.r_[0.10, KD_ANCHORS, 6.0]
    return np.interp(p, xp, fp)


def build_lobby(mode, lobby_size, bot_pct, churn, rng):
    bot_count = int(round(lobby_size * bot_pct / 100)) if mode == "Hybrid" else 0
    human_count = lobby_size - bot_count

    # Churn is modeled as disproportionate loss of below-midline players from the
    # active queue. We oversample then probabilistically retain players.
    pool = population_sample(max(human_count * 8, 1000), rng)
    low = pool < 0.78
    keep_prob = np.ones(pool.size)
    keep_prob[low] = 1.0 - (churn / 100.0) * 0.75
    pool = pool[rng.random(pool.size) < keep_prob]

    if mode == "Standard SBMM":
        # Illustrative average-skill SBMM lobby: select humans nearest the population
        # midline. This models skill compression, not Activision's actual algorithm.
        humans = pool[np.argsort(np.abs(pool - 0.78))[:human_count]]
    else:
        if len(pool) < human_count:
            pool = np.r_[pool, population_sample(human_count, rng)]
        humans = rng.choice(pool, human_count, replace=False)

    bots = np.array([])
    if bot_count:
        # Hybrid hypothesis: bots support only the lower part of the distribution.
        groups = rng.choice([0, 1, 2], size=bot_count, p=[0.50, 0.30, 0.20])
        bots = np.empty(bot_count)
        bots[groups == 0] = rng.uniform(0.10, 0.35, np.sum(groups == 0))
        bots[groups == 1] = rng.uniform(0.30, 0.50, np.sum(groups == 1))
        bots[groups == 2] = rng.uniform(0.50, 0.70, np.sum(groups == 2))

    values = np.r_[humans, bots]
    labels = np.array(["Human"] * len(humans) + ["Bot"] * len(bots))
    return values, labels


with st.sidebar:
    st.header("Build a lobby")
    mode = st.radio("Matchmaking model", ["Open / No SBMM", "Standard SBMM", "Hybrid"], index=0)
    lobby_size = st.slider("Lobby size", 40, 150, 120, 10)
    churn = st.slider("Low-skill churn", 0, 60, 0, 5, help="Models disproportionate loss of below-midline players from the active queue.")
    bot_pct = st.slider("Bot share", 0, 40, 10, 5, disabled=(mode != "Hybrid"))
    seed = st.number_input("Simulation seed", min_value=1, max_value=99999, value=42)

rng = np.random.default_rng(seed)
values, labels = build_lobby(mode, lobby_size, bot_pct, churn, rng)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lobby median K/D", f"{np.median(values):.2f}")
c2.metric("Lobby mean K/D", f"{np.mean(values):.2f}")
c3.metric("Skill spread", f"{np.std(values):.2f}")
c4.metric("Bots", f"{np.sum(labels == 'Bot')} / {lobby_size}")

st.markdown("### Lobby skill distribution")
chart = pd.DataFrame({"K/D": values, "Player type": labels})
bins = np.arange(0, max(3.05, values.max() + .25), .15)
hist_rows = []
for player_type in np.unique(labels):
    counts, edges = np.histogram(values[labels == player_type], bins=bins)
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        hist_rows.append({"K/D": round((left + right) / 2, 2), "Players": int(count), "Player type": player_type})
hist = pd.DataFrame(hist_rows)
pivot = hist.pivot(index="K/D", columns="Player type", values="Players").fillna(0)
st.bar_chart(pivot, height=360)

st.caption("0.78 is treated as the approximate population midline/common-player anchor — not 0.98. A near-1.0 aggregate K/D follows mechanically because most credited kills create another player's death; self-inflicted/environmental deaths can pull the global kills-to-deaths ratio below 1. The arithmetic mean of individual player K/Ds is a different statistic and can be pulled upward by the long high-skill tail.")

st.markdown("### Read the result")
if mode == "Open / No SBMM":
    st.write("Players are drawn broadly from the estimated active population. Skill diversity is therefore allowed to remain visible inside a lobby.")
elif mode == "Standard SBMM":
    st.write("This illustrative SBMM model compresses the lobby around the population midline. It demonstrates the effect of skill filtering; it does not claim to reproduce Activision's implementation.")
else:
    st.write("Humans are drawn broadly from the active population while hypothetical bots are concentrated in lower-skill bands. This tests whether lower-end support can coexist with a relatively open human matchmaking pool.")

with st.expander("Population model & evidence"):
    st.markdown("""
**The key statistical distinction**

- **~0.78 K/D:** used here as the approximate *midline/common-player* anchor for the historical Warzone population.
- **~0.92–0.98:** historical figures often described as an *average*. This is not used as the center of the distribution. A right-skewed K/D population can have a middle/common player well below the arithmetic mean.
- **Global kill/death accounting:** credited player kills and player deaths largely balance at the population level. Suicides, falls, gas/environmental deaths, and other uncredited deaths can push the aggregate kills/deaths ratio below 1. That accounting identity does **not** imply that the median player's K/D is ~1.0.

**Data quality**

Current population-wide Warzone data are scarce. The default curve combines historical tracker/Caldera-era percentile information, later community-reported anchors, interpolation between known points, and explicit extrapolation in the tails. Treat it as a plausible population model, not a current census.
""")
    anchors = pd.DataFrame({"Percentile": PERCENTILES, "Approx. K/D": KD_ANCHORS})
    st.dataframe(anchors, hide_index=True, use_container_width=True)

with st.expander("Model assumptions"):
    st.markdown("""
- **Open / No SBMM:** broad random draw from the modeled active population.
- **Standard SBMM:** demonstration of skill compression around the population midline. It is intentionally simplified.
- **Hybrid:** broad human matchmaking plus hypothetical lower-skill bot support. Bots are allocated 50% / 30% / 20% across three lower-skill bands.
- **Low-skill churn:** reduces representation of players below the 0.78 midline in the active queue. This is a scenario control, not a measured Warzone churn rate.
- The model describes consequences of assumptions. It does not identify the proprietary rules used by Call of Duty matchmaking.
""")

st.markdown("---")
st.caption("SBMMFree • Lightweight, transparent matchmaking simulation • Historical data + clearly labeled extrapolation")
