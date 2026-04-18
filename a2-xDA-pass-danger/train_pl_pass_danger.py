
#!/usr/bin/env python3
import json, sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42
MIN_PASSES_FOR_RANK = 200
OUT_DIR = Path("./outputs")

def ok(msg): print(f"[OK] {msg}")
def fail(msg): print(f"[ERROR] {msg}"); sys.exit(1)

def load_bundle(path: Path):
    if not path.exists(): fail(f"Missing {path}. Put events_England.json next to this script.")
    with path.open("r", encoding="utf-8") as f: b = json.load(f)
    if "events_per_match" not in b: fail("events_England.json missing key 'events_per_match'.")
    ok(f"Loaded bundle for {b.get('competition_name','?')} {b.get('season_name','?')}")
    return b

def flatten_events(b: dict) -> pd.DataFrame:
    rows, mids, evlists = [], b.get("matches", []), b["events_per_match"]
    for i, evlist in enumerate(evlists):
        mid = mids[i] if i < len(mids) else None
        for e in evlist:
            d = dict(e)
            if mid is not None: d["match_id"] = mid
            rows.append(d)
    df = pd.json_normalize(rows, sep="_")
    ok(f"Flattened events: rows={len(df)} cols={len(df.columns)}")
    return df

def build_pass_table(events: pd.DataFrame) -> pd.DataFrame:
    if "type_name" not in events.columns: fail("Event table missing 'type_name'.")
    passes = events[events["type_name"]=="Pass"].copy()
    comp = lambda v,i: v[i] if isinstance(v, list) and len(v)>=2 else np.nan
    passes["x"]     = passes["location"].apply(lambda v: comp(v,0))
    passes["y"]     = passes["location"].apply(lambda v: comp(v,1))
    passes["end_x"] = passes["pass_end_location"].apply(lambda v: comp(v,0))
    passes["end_y"] = passes["pass_end_location"].apply(lambda v: comp(v,1))
    dx, dy = passes["end_x"]-passes["x"], passes["end_y"]-passes["y"]
    passes["dist"] = np.hypot(dx, dy)
    passes["angle"] = np.degrees(np.arctan2(dy, dx))
    passes["forward"] = (dx > 0).astype(int)
    passes["height_low"]  = (passes.get("pass_height_name","")== "Low Pass").astype(int)
    passes["height_high"] = (passes.get("pass_height_name","")== "High Pass").astype(int)
    passes["body_part_foot"] = (passes.get("pass_body_part_name","")== "Foot").astype(int)
    keep = ["id","match_id","team_name","player_name","minute","second","x","y","end_x","end_y","dist","angle","forward","height_low","height_high","body_part_foot","possession","possession_team_name"]
    passes = passes[[c for c in keep if c in passes.columns]].dropna(subset=["x","y","end_x","end_y"])
    ok(f"Pass rows after cleaning: {len(passes)}")
    return passes

def build_labels(passes: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    shots = events[events["type_name"]=="Shot"][["match_id","possession","minute","second"]].copy()
    shots["t"] = shots["minute"]*60 + shots["second"]
    p = passes.copy(); p["t"] = p["minute"]*60 + p["second"]
    key = ["match_id","possession"]
    if not all(k in p.columns for k in key): fail("Missing match_id or possession for labeling.")
    next_shot = (shots.sort_values(key+["t"]).groupby(key, as_index=False)["t"].min().rename(columns={"t":"t_shot"}))
    p = p.merge(next_shot, on=key, how="left")
    p["label_shot10"] = ((~p["t_shot"].isna()) & ((p["t_shot"]-p["t"])>=0) & ((p["t_shot"]-p["t"])<=10)).astype(int)
    ok(f"Labels built. positive_rate={p['label_shot10'].mean():.4f} rows={len(p)}")
    return p

def train_model(df: pd.DataFrame):
    feat_cols = ["x","y","end_x","end_y","dist","angle","forward","height_low","height_high","body_part_foot"]
    df = df.dropna(subset=feat_cols).copy()
    X, y = df[feat_cols].values, df["label_shot10"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    pipe.fit(Xtr, ytr); probs = pipe.predict_proba(Xte)[:,1]
    ok(f"Model trained. ROC-AUC={roc_auc_score(yte, probs):.3f} AP={average_precision_score(yte, probs):.3f} n_train={len(Xtr)} n_test={len(Xte)}")
    return pipe, feat_cols, df

def rank_players(df: pd.DataFrame, pipe, feat_cols):
    df = df.copy(); df["prob"] = pipe.predict_proba(df[feat_cols].values)[:,1]
    grp = (df.groupby("player_name")["prob"].agg(passes="count", pass_danger="mean").sort_values("pass_danger", ascending=False))
    ok(f"Ranked players: {len(grp)}")
    return grp, df

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="events_England.json")
    parser.add_argument("--outdir", default=str(OUT_DIR))
    parser.add_argument("--min_passes", type=int, default=MIN_PASSES_FOR_RANK)
    # Jupyter-safe:
    args = parser.parse_args(argv) if argv is not None else parser.parse_known_args()[0]

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(Path(args.events))
    events = flatten_events(bundle)
    passes = build_pass_table(events)
    df = build_labels(passes, events)
    pipe, feat_cols, dfm = train_model(df)
    ranks, scored = rank_players(dfm, pipe, feat_cols)

    ranks.to_csv(outdir/"pl_player_pass_danger.csv")
    ok(f"Wrote {outdir/'pl_player_pass_danger.csv'}")
    sample_cols = [c for c in ["player_name","team_name","minute","second","prob","x","y","end_x","end_y","dist","angle","forward"] if c in scored.columns]
    scored.sample(min(1000,len(scored)), random_state=SEED)[sample_cols].to_csv(outdir/"pl_scored_passes_sample.csv", index=False)
    ok(f"Wrote {outdir/'pl_scored_passes_sample.csv'}")
    prev = ranks.reset_index(); prev = prev[prev["passes"]>=args.min_passes].head(15)
    print("\n[PREVIEW] Top players (>= {} passes):".format(args.min_passes))
    print(prev.to_string(index=False) if not prev.empty else "No players meet threshold.")
if __name__ == "__main__":
    main()
