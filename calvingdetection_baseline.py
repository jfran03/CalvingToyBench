"""Score the CalvingDetection authors' pretrained models on our windows (no retraining).

Usage: python calvingdetection_baseline.py --repo CalvingDetection
       -> results/cd__rf_pretrained.jsonl, results/cd__ae_only.jsonl, then run `python score.py`.

Pipeline of the authors (Cao et al. 2025): an autoencoder flags ~5% of each cow's hours as anomalous
(saved in anomalies_from_AE/*.csv); a random forest classifies the flagged hours as calving-related.
We use their saved AE flags and the RF fold that HELD OUT each cow, so the RF never saw the cow it scores.
Window prob (backward-looking, hours t-11..t):
  rf_pretrained = max RF P(calving) over flagged hours, 0 if none flagged
  ae_only       = fraction of the last 12 hours that the AE flagged
Caveats (see README): RF label is +-24h around birth (ours is [t-6h, t+24h)); the AE threshold is each cow's
season-wide 95th percentile (non-causal) and the AE trained on days before -5 of each cow's calving.
"""
import argparse, glob, json, os, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LOOKBACK_H = 12
HELD_OUT = {1: ("09B", "0C5", "14C"), 2: ("09F", "0BB", "0BD"), 3: ("0BC", "0BE", "0E1"),
            4: ("0BA", "13C", "13D"), 5: ("0E7", "12A", "12B")}
COWS = [c for k in sorted(HELD_OUT) for c in HELD_OUT[k]]
BASE_FEATS = ["KNN2", "KNN3", "KNN4", "KNN5", "Distance_moved", "ActivityCount_Difference"]


def fold_for_cow(cow):
    for k, cows in HELD_OUT.items():
        if cow in cows:
            return k
    raise ValueError(f"cow {cow!r} is not in any RF fold's held-out set")


def feature_names():
    names = list(BASE_FEATS)  # the RF also uses the 6 raw columns
    for f in BASE_FEATS:
        for w in (6, 12):
            names += [f"{f}_MA{w}", f"{f}_STD{w}"]
        names += [f"{f}_Diff1", f"{f}_Diff3"]
    return names


def add_sliding_features(df):
    """The authors' Stage_2 recipe, applied per cow (they applied it to the concatenated table; only the
    first <12 rows of each cow differ)."""
    df = df.copy()
    for f in BASE_FEATS:
        for w in (6, 12):
            df[f"{f}_MA{w}"] = df[f].rolling(window=w, min_periods=1).mean()
            df[f"{f}_STD{w}"] = df[f].rolling(window=w, min_periods=1).std()
        df[f"{f}_Diff1"] = df[f].diff(1)
        df[f"{f}_Diff3"] = df[f].diff(3)
    return df


def _in_lookback(index, t, lookback_h):
    return (index > t - pd.Timedelta(hours=lookback_h)) & (index <= t)


def window_rf_prob(flag_probs, t, lookback_h=LOOKBACK_H):
    sel = flag_probs[_in_lookback(flag_probs.index, t, lookback_h)]
    return float(sel.max()) if len(sel) else 0.0


def window_ae_frac(flag_times, t, lookback_h=LOOKBACK_H):
    return float(_in_lookback(pd.DatetimeIndex(flag_times), t, lookback_h).sum()) / lookback_h


def load_cow(repo, cow):
    """Hourly feature frame (UTC index) and the hours the authors' AE flagged for this cow."""
    path = glob.glob(os.path.join(repo, "Processed_data", f"{cow}*.csv"))[0]
    df = add_sliding_features(pd.read_csv(path, parse_dates=True, index_col=0))
    a = pd.read_csv(os.path.join(repo, "anomalies_from_AE", f"anomalies_{cow}.csv"))
    flags = pd.DatetimeIndex(pd.to_datetime(a["timestamp"]).dt.tz_localize("UTC").dt.round("h"))
    return df, flags.intersection(df.index)


def load_rf(repo, k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn version-mismatch notice
        return joblib.load(os.path.join(repo, "Trained_RF_models", f"Fold{k}.pkl"))


def _flag_probs(clf, df, flags):
    X = df.loc[flags, list(clf.feature_names_in_)]
    return pd.Series(clf.predict_proba(X)[:, list(clf.classes_).index(True)], index=flags)


def score_windows(repo, windows, lookback_h=LOOKBACK_H, fold_of=fold_for_cow):
    """Returns (rf_rows, ae_rows) in score.py's result format. `fold_of` lets the sanity check swap the fold."""
    clfs, cache = {}, {}
    rf, ae = [], []
    for w in windows:
        c = w["cow"]
        if c not in cache:
            df, flags = load_cow(repo, c)
            k = fold_of(c)
            clfs.setdefault(k, load_rf(repo, k))
            cache[c] = (_flag_probs(clfs[k], df, flags), flags)
        probs, flags = cache[c]
        t = pd.Timestamp(w["t_utc"])
        base = dict(id=w["id"], cow=c, label=w["label"], hours_to_calving=w["hours_to_calving"])
        rf.append({**base, "prob": window_rf_prob(probs, t, lookback_h)})
        ae.append({**base, "prob": window_ae_frac(flags, t, lookback_h)})
    return rf, ae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--windows", default="windows.jsonl")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    W = [json.loads(l) for l in open(a.windows)]
    rf, ae = score_windows(a.repo, W)
    os.makedirs(a.out_dir, exist_ok=True)
    for name, rows in (("cd__rf_pretrained", rf), ("cd__ae_only", ae)):
        with open(os.path.join(a.out_dir, name + ".jsonl"), "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"{name}: {len(rows)} rows, {sum(r['prob'] > 0 for r in rows)} nonzero")
    print("held-out cows per RF fold:", {k: list(v) for k, v in HELD_OUT.items()})
    print("fold sanity (RF's own task: flagged hours, authors' +-24h label), accuracy held-out vs training fold:")
    h, tr = fold_sanity(a.repo)
    print(f"  held-out {h:.2f} vs training {tr:.2f} (training should be clearly higher)")


def fold_sanity(repo):
    """Mean per-hour accuracy on AE-flagged hours, scoring each cow with its held-out fold vs a fold that trained on it."""
    from prep import CALVING_UTC
    acc = {"held": [], "train": []}
    for c in COWS:
        df, flags = load_cow(repo, c)
        calv = pd.Timestamp(CALVING_UTC[c], tz="UTC").floor("h")
        y = (flags >= calv - pd.Timedelta(days=1)) & (flags < calv + pd.Timedelta(days=1))
        for key, k in (("held", fold_for_cow(c)), ("train", fold_for_cow(c) % 5 + 1)):
            acc[key].append(((_flag_probs(load_rf(repo, k), df, flags) > 0.5).values == y).mean())
    return float(np.mean(acc["held"])), float(np.mean(acc["train"]))


if __name__ == "__main__":
    main()
