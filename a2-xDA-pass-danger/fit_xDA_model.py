
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_xDA_model.py — Expected Danger Added by Passes (xDA-P)
Trains on Premier League events_England.json and ranks players.
Standard libraries only (+mplsoccer optional).
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

# -------------------- Utilities --------------------

PITCH_LENGTH = 120.0  # StatsBomb coordinate system
PITCH_WIDTH = 80.0

def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            return default
    return cur

def load_events(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure list of dicts
    assert isinstance(data, list), "Expected a list of events"
    # Attach match_id if missing in some rows
    # StatsBomb events are per match, but many exports include match_id in every event already
    return data

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def simple_xT(x, y):
    """
    A simple hand-crafted 'expected threat' proxy in [0,1].
    Increases with x (closer to goal) and peaks near central y.
    """
    # Normalize coordinates to pitch
    x = float(np.clip(x, 0, PITCH_LENGTH))
    y = float(np.clip(y, 0, PITCH_WIDTH))
    x_term = sigmoid(0.08 * (x - 60.0))          # rises after midfield
    y_term = np.exp(-((y - PITCH_WIDTH/2.0)**2) / (2 * (PITCH_WIDTH/3.6)**2))  # wider sigma
    return float(np.clip(x_term * y_term, 0, 1))

def pass_features_and_label(ev):
    """
    Extract features and label for a single pass event.
    Label y = max(0, xT(end) - xT(start)) for successful passes, else 0.
    """
    start = _safe_get(ev, "location", default=[None, None])
    sx, sy = (start[0] if start and len(start) > 0 else None,
              start[1] if start and len(start) > 1 else None)
    end_loc = _safe_get(ev, "pass", "end_location", default=[None, None])
    ex, ey = (end_loc[0] if end_loc and len(end_loc) > 0 else None,
              end_loc[1] if end_loc and len(end_loc) > 1 else None)

    if sx is None or sy is None:
        return None, None  # insufficient data

    # Flags and categorical fields
    under_pressure = 1 if _safe_get(ev, "under_pressure", default=False) else 0
    height = _safe_get(ev, "pass", "height", "name", default="NA")
    body = _safe_get(ev, "pass", "body_part", "name", default="NA")
    technique = _safe_get(ev, "pass", "technique", "name", default="NA")
    outcome = _safe_get(ev, "pass", "outcome", "name", default=None)  # present when unsuccessful
    minute = int(_safe_get(ev, "minute", default=0) or 0)
    period = int(_safe_get(ev, "period", default=1) or 1)

    # Geom features
    if ex is None or ey is None:
        ex, ey = sx, sy  # fallback if missing; will yield zero gain if counted as success accidentally

    dx = ex - sx
    dy = ey - sy
    dist = float(np.hypot(dx, dy))
    angle = float(np.arctan2(dy, dx))  # radians
    forward = 1 if dx > 0 else 0
    wide_switch = 1 if abs(dy) > (PITCH_WIDTH * 0.35) and dist > (PITCH_LENGTH * 0.25) else 0

    # xT values
    xt_s = simple_xT(sx, sy)
    xt_e = simple_xT(ex, ey)

    # Label
    success = outcome is None  # StatsBomb: outcome present => not completed
    y = max(0.0, xt_e - xt_s) if success else 0.0

    # Feature dict
    feats = {
        "start_x": sx, "start_y": sy,
        "end_x": ex, "end_y": ey,
        "dx": dx, "dy": dy,
        "distance": dist,
        "angle": angle,
        "forward": forward,
        "under_pressure": under_pressure,
        "wide_switch": wide_switch,
        "start_xT": xt_s, "end_xT": xt_e,
        "minute": minute, "period": period,
        "height": height, "body": body, "technique": technique,
    }
    return feats, y

def collect_passes(events):
    rows = []
    for ev in events:
        if _safe_get(ev, "type", "name") != "Pass":
            continue
        player_id = _safe_get(ev, "player", "id")
        player_name = _safe_get(ev, "player", "name", default="Unknown")
        team_id = _safe_get(ev, "team", "id")
        team_name = _safe_get(ev, "team", "name", default="Unknown")
        match_id = _safe_get(ev, "match_id", default=_safe_get(ev, "match", "id"))

        feats, y = pass_features_and_label(ev)
        if feats is None:
            continue
        feats.update({
            "player_id": player_id, "player_name": player_name,
            "team_id": team_id, "team_name": team_name,
            "match_id": match_id
        })
        rows.append((feats, y))
    if not rows:
        return pd.DataFrame(), np.array([])
    X = pd.DataFrame([r[0] for r in rows])
    y = np.array([r[1] for r in rows], dtype=float)
    return X, y

def estimate_minutes(events):
    """
    Estimate minutes played per player across matches using Starting XI and Substitution events.
    Returns a DataFrame with columns: player_id, player_name, minutes.
    """
    # Group events by match
    by_match = defaultdict(list)
    for ev in events:
        mid = _safe_get(ev, "match_id", default=_safe_get(ev, "match", "id"))
        by_match[mid].append(ev)

    minutes = defaultdict(lambda: {"name": None, "mins": 0.0})

    for mid, evs in by_match.items():
        # Find match length from last event minute in period 2 (fallback 90)
        match_len = 90.0
        for ev in evs:
            per = _safe_get(ev, "period", default=1) or 1
            if per == 2:
                m = _safe_get(ev, "minute", default=0) or 0
                s = _safe_get(ev, "second", default=0) or 0
                match_len = max(match_len, float(m) + float(s)/60.0)

        # Track stints
        starts = {}   # player_id -> start_min
        ends = {}     # player_id -> end_min

        # Starting XI
        for ev in evs:
            if _safe_get(ev, "type", "name") == "Starting XI":
                lineup = _safe_get(ev, "tactics", "lineup", default=[])
                for li in lineup or []:
                    pid = _safe_get(li, "player", "id")
                    pname = _safe_get(li, "player", "name", default="Unknown")
                    if pid is None: 
                        continue
                    starts[pid] = 0.0
                    # ensure name stored
                    if minutes[pid]["name"] is None:
                        minutes[pid]["name"] = pname

        # Substitutions
        for ev in evs:
            if _safe_get(ev, "type", "name") == "Substitution":
                off_pid = _safe_get(ev, "player", "id")
                off_pname = _safe_get(ev, "player", "name", default="Unknown")
                on_pid = _safe_get(ev, "substitution", "replacement", "id")
                on_pname = _safe_get(ev, "substitution", "replacement", "name", default="Unknown")
                m = float(_safe_get(ev, "minute", default=0) or 0)
                s = float(_safe_get(ev, "second", default=0) or 0)
                tmin = m + s/60.0

                if off_pid is not None:
                    ends[off_pid] = min(tmin, match_len)
                    if minutes[off_pid]["name"] is None:
                        minutes[off_pid]["name"] = off_pname
                if on_pid is not None:
                    if on_pid not in starts:
                        starts[on_pid] = tmin
                    if minutes[on_pid]["name"] is None:
                        minutes[on_pid]["name"] = on_pname

        # Any player with events but no recorded start
        for ev in evs:
            pid = _safe_get(ev, "player", "id")
            pname = _safe_get(ev, "player", "name", default="Unknown")
            if pid is None:
                continue
            if pid not in starts:
                m = float(_safe_get(ev, "minute", default=0) or 0)
                s = float(_safe_get(ev, "second", default=0) or 0)
                tmin = m + s/60.0
                starts[pid] = tmin
            if minutes[pid]["name"] is None:
                minutes[pid]["name"] = pname

        # Close stints
        for pid, st in starts.items():
            en = ends.get(pid, match_len)
            played = max(0.0, en - st)
            minutes[pid]["mins"] += played

    rows = []
    for pid, rec in minutes.items():
        rows.append({"player_id": pid, "player_name": rec["name"] or "Unknown", "minutes": rec["mins"]})
    return pd.DataFrame(rows)

def train_xda_model(X, y, random_state=42):
    # Feature schema
    num_cols = ["start_x","start_y","end_x","end_y","dx","dy","distance","angle",
                "forward","under_pressure","wide_switch","start_xT","end_xT","minute","period"]
    cat_cols = ["height","body","technique"]

    # Fill missing
    Xn = X.copy()
    for c in num_cols:
        if c not in Xn.columns:
            Xn[c] = 0.0
    for c in cat_cols:
        if c not in Xn.columns:
            Xn[c] = "NA"

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )

    gbr = GradientBoostingRegressor(random_state=random_state, loss="squared_error")
    pipe = Pipeline([("pre", pre), ("gbr", gbr)])

    # Out-of-fold predictions for unbiased player ranking on training league
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_pred = cross_val_predict(pipe, Xn, y, cv=kf, n_jobs=None, verbose=0, method="predict")
    # Fit final model on all data for scoring other leagues
    pipe.fit(Xn, y)

    return pipe, oof_pred, num_cols + cat_cols

def rank_players(X, preds, minutes_df, league_label, out_dir):
    df = X[["player_id","player_name"]].copy()
    df["xDA_pass"] = preds  # per pass expected danger added
    # Aggregate per player
    agg = df.groupby(["player_id","player_name"], as_index=False)["xDA_pass"].sum()
    # Join minutes to compute per 90
    m = minutes_df.copy()
    agg = agg.merge(m, on=["player_id","player_name"], how="left")
    agg["minutes"] = agg["minutes"].fillna(0.0)
    agg["xDA_per90"] = agg.apply(lambda r: (r["xDA_pass"] / r["minutes"] * 90.0) if r["minutes"] > 0 else np.nan, axis=1)
    agg = agg.sort_values("xDA_per90", ascending=False)

    out_path = Path(out_dir) / f"rankings_{league_label.replace(' ','_')}.csv"
    agg.to_csv(out_path, index=False, encoding="utf-8")
    return agg, out_path

def score_new_league(model, events_path, out_dir, league_label=None):
    events = load_events(events_path)
    X_new, _ = collect_passes(events)
    if X_new.empty:
        print(f"[WARN] No passes found in {events_path}")
        return None, None
    mins_new = estimate_minutes(events)
    preds = model.predict(X_new)
    league_label = league_label or Path(events_path).stem
    agg, out_csv = rank_players(X_new, preds, mins_new, league_label, out_dir)
    print(f"[OK] {league_label}: wrote player rankings to {out_csv}")
    print(agg.head(15).to_string(index=False))
    return agg, out_csv

def train_and_rank(train_path, out_dir):
    events = load_events(train_path)
    X, y = collect_passes(events)
    assert not X.empty, "No pass events found for training."
    assert y.size == len(X), "Label length mismatch."

    mins = estimate_minutes(events)
    model, oof_pred, used_cols = train_xda_model(X, y, random_state=42)
    league_label = "Premier_League"
    agg, out_csv = rank_players(X, oof_pred, mins, league_label, out_dir)

    # Save a small model summary
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "feature_list.txt", "w", encoding="utf-8") as f:
        for c in used_cols:
            f.write(f"{c}\n")

    print(f"[OK] Training league ranked. CSV: {out_csv}")
    print(agg.head(15).to_string(index=False))
    return model

def main():
    parser = argparse.ArgumentParser(description="Train xDA-P model on Premier League and rank players; then score other leagues.")
    parser.add_argument("--train", type=str, default="events_England.json",
                        help="Path to Premier League events JSON (StatsBomb format).")
    parser.add_argument("--score", type=str, nargs="*", default=[],
                        help="Paths to other leagues' events JSON to score with the trained model.")
    parser.add_argument("--out_dir", type=str, default="outputs",
                        help="Directory to write outputs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = train_and_rank(args.train, out_dir)

    # Score any additional leagues
    for p in args.score:
        score_new_league(model, p, out_dir, league_label=Path(p).stem)

if __name__ == "__main__":
    main()
