import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SBMMFree", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem; padding-bottom: 4rem;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.18); padding: 14px; border-radius: 12px;}
.rank-box {padding: 16px 18px; border: 1px solid rgba(128,128,128,.22); border-radius: 12px; background: rgba(128,128,128,.06); margin: 12px 0 22px 0;}
</style>
""", unsafe_allow_html=True)

st.title("SBMMFree")
st.subheader("What happens when you change how a Warzone lobby is built?")
st.caption("An interactive matchmaking model — not a reconstruction of Activision's matchmaking code.")

PERCENTILES = np.array([2, 17, 27, 38, 50, 62, 66, 79, 87, 91, 94, 97, 99, 99.9, 99.98, 99.99])
KD_ANCHORS = np.array([0.15, 0.40, 0.55, 0.62, 0.78, 0.92, 0.97, 1.12, 1.30, 1.45, 1.58, 1.85, 2.08, 3.57, 4.40, 5.70])

# These are scenario targets, not claims about Activision's actual SBMM brackets.
SBMM_TARGETS = {
    "Protected": {"kd": 0.35, "width": 0.18},
    "Low": {"kd": 0.60, "width": 0.24},
    "Medium": {"kd": 0.85, "width": 0.30},
    "High": {"kd": 1.30, "width": 0.42},
}


def population_sample(n, rng):
    """Sample a smooth quantile curve fitted to historical/extrapolated anchors."""
    p = rng.uniform(0.1, 99.99, n)
    xp = np.r_[0.1, PERCENTILES, 100.0]
    fp = np.r_[0.10, KD_ANCHORS, 6.0]
    return np.interp(p, xp, fp)


def active_pool(n, churn, rng):
    """Create an active queue after applying the low-skill churn scenario."""
    pool = population_sample(max(n * 10, 1500), rng)
    keep_prob = np.ones(pool.size)
    keep_prob[pool < 0.78] = 1.0 - (churn / 100.0) * 0.75
    return pool[rng.random(pool.size) < keep_prob]


def weighted_sbmm_sample(pool, n, target_name, rng):
    """Select from the same population, with probability declining away from target K/D."""
    target = SBMM_TARGETS[target_name]
    distance = (pool - target["kd"]) / target["width"]
    weights = np.exp(-0.5 * distance ** 2)

    # A small floor prevents artificial hard walls: out-of-band players are rare, not impossible.
    weights = weights + 0.015
    weights /= weights.sum()
    return rng.choice(pool, size=n, replace=False, p=weights)


def build_lobby(mode, lobby_size, bot_pct, churn, target_name, rng):
    bot_count = int(round(lobby_size * bot_pct / 100)) if mode == "Hybrid" else 0
    human_count = lobby_size - bot_count
    pool = active_pool(human_count, churn, rng)

    if len(pool) < human_count:
        pool = np.r_[pool, population_sample(human_count * 2, rng)]

    if mode == "Standard SBMM":
        humans = weighted_sbmm_sample(pool, human_count, target_name, rng)
    else:
        humans = rng.choice(pool, human_count, replace=False)

    bots = np.array([])
    if bot_count:
        groups = rng.choice([0, 1, 2], size=bot_count, p=[0.50, 0.30, 0.20])
        bots = np.empty(bot_count)
        bots[groups == 0] = rng.uniform(0.10, 0.35, np.sum(groups == 0))
        bots[groups == 1] = rng.uniform(0.30, 0.50, np.sum(groups == 1))
        bots[groups == 2] = rng.uniform(0.50, 0.70, np.sum(groups == 2))

    values = np.r_[humans, bots]
    labels = np.array(["Human"] * len(humans) + ["Bot"] * len(bots))
    return values, labels


def lobby_difficulty(values, labels):
    """Difficulty score used only for same-scenario percentile ranking."""
    humans = values[labels == "Human"]
    if len(humans) == 0:
        humans = values
    # Median captures the typical opponent while the upper quartile preserves the impact of the strong tail.
    return 0.65 * np.median(humans) + 0.35 * np.quantile(humans, 0.75)


def benchmark_lobbies(mode, lobby_size, bot_pct, churn, target_name, count=1000):
    """Monte Carlo reference distribution. It does not replace the displayed single lobby."""
    # Fixed reference seed keeps the percentile baseline stable while the displayed lobby changes.
    rng = np.random.default_rng(20260917)
    scores = np.empty(count)
    for i in range(count):
        vals, labs = build_lobby(mode, lobby_size, bot_pct, churn, target_name, rng)
        scores[i] = lobby_difficulty(vals, labs)
    return scores


if "lobby_seed" not in st.session_state:
    st.session_state.lobby_seed = 42

with st.sidebar:
    st.header("Build a lobby")
    mode = st.radio("Matchmaking model", ["Open / No SBMM", "Standard SBMM", "Hybrid"], index=0)

    target_name = "Medium"
    if mode == "Standard SBMM":
        target_name = st.radio(
            "Target skill",
            ["Protected", "Low", "Medium", "High"],
            index=2,
            help="Controls which part of the same underlying player population is favored by matchmaking.",
        )
        target = SBMM_TARGETS[target_name]
        st.caption(f"Target center: ~{target['kd']:.2f} K/D. Players outside the band remain possible.")

    lobby_size = st.slider("Lobby size", 40, 150, 120, 10)
    churn = st.slider("Low-skill churn", 0, 60, 0, 5, help="Models disproportionate loss of below-midline players from the active queue.")
    bot_pct = st.slider("Bot share", 0, 40, 10, 5, disabled=(mode != "Hybrid"))

    if st.button("🎲 New Lobby", use_container_width=True, type="primary"):
        st.session_state.lobby_seed = int(np.random.default_rng().integers(1, 2_147_483_647))

    with st.expander("Advanced"):
        seed = st.number_input(
            "Simulation seed",
            min_value=1,
            max_value=2_147_483_647,
            value=int(st.session_state.lobby_seed),
            help="Each seed represents one reproducible lobby.",
        )
        if int(seed) != st.session_state.lobby_seed:
            st.session_state.lobby_seed = int(seed)

rng = np.random.default_rng(st.session_state.lobby_seed)
values, labels = build_lobby(mode, lobby_size, bot_pct, churn, target_name, rng)
humans = values[labels == "Human"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Median K/D", f"{np.median(values):.2f}")
c2.metric("Mean K/D", f"{np.mean(values):.2f}")
c3.metric("1.5+ humans", int(np.sum(humans >= 1.5)))
c4.metric("2.0+ humans", int(np.sum(humans >= 2.0)))
c5.metric("Bots", f"{np.sum(labels == 'Bot')} / {lobby_size}")

with st.spinner("Ranking this lobby against 1,000 comparable simulations..."):
    baseline = benchmark_lobbies(mode, lobby_size, bot_pct, churn, target_name, count=1000)
current_score = lobby_difficulty(values, labels)
rank_pct = 100.0 * np.mean(baseline <= current_score)
low90, high90 = np.quantile(baseline, [0.05, 0.95])

st.markdown("### Where does this lobby rank?")
st.markdown(
    f"<div class='rank-box'><b>This lobby is harder than {rank_pct:.0f}% of comparable {mode} lobbies.</b><br>"
    f"The ranking compares this one seed against 1,000 simulated lobbies using the same lobby size, churn, bot settings, and SBMM target (when applicable). "
    f"The displayed lobby remains a single random lobby — it is not an average.</div>",
    unsafe_allow_html=True,
)

st.markdown("### Lobby skill distribution")
bins = np.arange(0, max(3.05, values.max() + .25), .15)
hist_rows = []
for player_type in np.unique(labels):
    counts, edges = np.histogram(values[labels == player_type], bins=bins)
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        hist_rows.append({"K/D": round((left + right) / 2, 2), "Players": int(count), "Player type": player_type})
hist = pd.DataFrame(hist_rows)
pivot = hist.pivot(index="K/D", columns="Player type", values="Players").fillna(0)
st.bar_chart(pivot, height=360)

st.caption("0.78 is treated as the approximate population midline/common-player anchor — not 0.98. A near-1 aggregate kill/death ratio is an accounting relationship and is not the median or mode of individual player K/Ds. The long high-skill tail can pull arithmetic averages upward.")

st.markdown("### Read the result")
if mode == "Open / No SBMM":
    st.write("This is one random draw from the modeled active population. Repeatedly choose **New Lobby** to see the natural lobby-to-lobby variance that open matchmaking permits.")
elif mode == "Standard SBMM":
    st.write(f"This **{target_name}** SBMM scenario favors players near ~{SBMM_TARGETS[target_name]['kd']:.2f} K/D, but draws them from the same underlying population as Open matchmaking. Selection probability falls with skill distance instead of imposing a hard K/D wall.")
else:
    st.write("Humans are broadly drawn from the active population while hypothetical bots are concentrated in lower-skill bands. Choose **New Lobby** repeatedly to see how much human-skill variance remains under this Hybrid scenario.")

with st.expander("Population model & evidence"):
    st.markdown("""
**The key statistical distinction**

- **~0.78 K/D:** approximate *midline/common-player* anchor for the historical Warzone population.
- **~0.92–0.98:** historical figures described as an *average* are not used as the center of the distribution. A right-skewed population can have a middle/common player well below its arithmetic mean.
- **Global kill/death accounting:** credited player kills and player deaths largely balance. Suicides, falls, gas/environmental deaths, and other uncredited deaths can pull the aggregate ratio below 1. This does not imply that the median player's K/D is ~1.0.

**Data quality**

Current population-wide Warzone data are scarce. The default curve combines historical tracker/Caldera-era information, community-reported percentile anchors, interpolation, and explicit tail extrapolation. Treat it as a plausible population model, not a current census.
""")
    anchors = pd.DataFrame({"Percentile": PERCENTILES, "Approx. K/D": KD_ANCHORS})
    st.dataframe(anchors, hide_index=True, use_container_width=True)

with st.expander("Model assumptions"):
    st.markdown("""
- **Open / No SBMM:** broad random draw from the modeled active population.
- **Standard SBMM:** uses the same population but weights selection toward Protected, Low, Medium, or High target skill. The targets and selection widths are scenario assumptions, not measured Activision brackets.
- **Hybrid:** broad human matchmaking plus hypothetical lower-skill bot support. Bots are allocated 50% / 30% / 20% across three lower-skill bands.
- **Low-skill churn:** reduces representation of players below the 0.78 midline in the active queue. This is a scenario control, not a measured Warzone churn rate.
- **Lobby rank:** compares the displayed single lobby with 1,000 Monte Carlo lobbies under identical scenario settings. Difficulty is a composite of human median K/D (65%) and human 75th-percentile K/D (35%), preserving both the typical opponent and upper-skill tail.
- The model describes consequences of assumptions. It does not identify Call of Duty's proprietary matchmaking rules.
""")

with st.expander("Reproducibility"):
    st.write(f"Current lobby seed: **{st.session_state.lobby_seed}**")
    st.caption("Use the same seed and scenario settings to reproduce this exact simulated lobby.")

st.markdown("---")
st.caption("SBMMFree • Lightweight, transparent matchmaking simulation • Historical data + clearly labeled extrapolation")
