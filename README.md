# Mathematical Modelling of Football (1RT001)

Coursework for **Mathematical Modelling of Football** (1RT001), period 1 of the M.Sc. in Image Analysis and Machine Learning at Uppsala University. The course applies statistical and machine-learning methods to football event-stream data from StatsBomb.

The repository contains three assignments, each in its own folder. Data files are intentionally excluded — see the [StatsBomb Open Data repository](https://github.com/statsbomb/open-data) for the event JSON used throughout.

## Contents

- **`a1-pitch-exploration/`** — first assignment. Loading StatsBomb event data, visualising player actions on the pitch with `mplsoccer`, and exploring movement patterns for individual players.
- **`a2-xDA-pass-danger/`** — main project. Expected Danger Added by Passes (xDA-P): a gradient-boosted pass-evaluation model trained on five European leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1 — 2015/16 season).
- **`a3-statistical-testing/`** — hypothesis testing on football-derived quantities, including sign tests and t-tests implemented from scratch for paired comparisons between teams or tactical regimes.

## Highlight: xDA-P (Assignment 2)

For every pass in the training data, the model predicts how much the pass is expected to increase the attacking team's threat. Aggregated over all passes a player makes, this produces a per-player pass-danger score that can be compared across leagues and normalised by minutes played.

**Pipeline:**
1. Parse StatsBomb event JSON across five leagues
2. Engineer per-pass features: start/end coordinates, distance, angle, under-pressure flag, height, body-part, technique, wide-switch indicator, match-time context
3. Label each successful pass with `max(0, xT(end) − xT(start))` using a handcrafted spatial threat proxy xT(x, y)
4. Train a Gradient Boosting Regressor with 5-fold cross-validation; compare to a linear baseline
5. Evaluate with ROC and precision-recall curves, feature importances, and calibration
6. Estimate minutes played per player from Starting XI and Substitution events
7. Rank top players per league with minimum-pass filters (≥150 or ≥200 passes)

Outputs include per-league top-10 bar charts, ROC/PR curves, feature-importance plots, and ranked player CSVs for each league.

Run with:

```bash
python a2-xDA-pass-danger/fit_xDA_model.py --events-dir ./statsbomb --output-dir ./a2-xDA-pass-danger/outputs
```

Dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and optionally `mplsoccer`.

## Data attribution

Event data: [StatsBomb Open Data](https://github.com/statsbomb/open-data), used under their free-for-non-commercial-research licence. Data files are not redistributed in this repository.
