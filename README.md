# SBMMFree

A lightweight interactive model for exploring how different matchmaking rules can change the skill composition of a Warzone-style battle royale lobby.

## What it models

- **Open / No SBMM** — broad matchmaking from the modeled active population.
- **Standard SBMM** — an intentionally simplified demonstration of skill compression around the population midline.
- **Hybrid** — broad human matchmaking with hypothetical bot support concentrated in lower-skill bands.
- **Low-skill churn** — a scenario control that reduces representation of below-midline players in the active queue.

## Population K/D model

The model deliberately separates the **middle/common player** from an **arithmetic average**. The default historical population curve uses ~0.78 K/D as the approximate midline/common-player anchor. Historical ~0.92–0.98 average figures are not treated as the center of the player distribution.

At a population-accounting level, most credited player kills correspond to another player's death, while suicides and environmental/uncredited deaths can make the global kills/deaths ratio slightly less than 1. This accounting relationship does not imply that the median individual player has a ~1.0 K/D. A right-skewed distribution can have a substantially lower median/mode while high-K/D players pull the arithmetic mean upward.

Current population-wide Warzone data are scarce. Inputs combine historical tracker/Caldera-era information, community-reported percentile anchors, interpolation, and explicit extrapolation. The model is therefore a **simulation based on plausible historical estimates**, not a current census and not a reconstruction of Activision's proprietary matchmaking algorithm.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
