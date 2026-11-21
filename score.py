"""Score LLM runs and compare them with two classic baselines.

Usage: python score.py            # scores every file in results/
       python score.py --thr 0.5  # alert threshold on the probability

Window-level: AUROC and AUPRC (threshold-free; chance AUPRC = positive rate).
Event-level at the threshold: cows detected (any positive window alerted),
median lead time before calving, false alarms per cow-week.
Baselines are computed from windows.jsonl with leave-one-cow-out, so they never
see the test cow.
"""
import argparse, glob, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from prep import FEATURES, STEP_H


def summarize(w, feats):
    """Last-12h mean of each feature, z-scored against the cow's own baseline."""
    x = []
    for c in feats:
        v = [h[c] for h in w["hours"][-12:] if h[c] is not None]
        b = w["baseline"].get(c, {"median": 0, "p90": 1})
        spread = max(b["p90"] - b["median"], 1e-6)
        x.append(((np.mean(v) if v else b["median"]) - b["median"]) / spread)
    return x


def rule_scores(W):
    """Simple threshold-style rule: isolation from herd relative to her own baseline."""
    return [summarize(w, ["mean_4_nearest_m"])[0] for w in W]


def logreg_scores(W, feats):
    X = np.array([summarize(w, feats) for w in W]); y = np.array([w["label"] for w in W])
    cows = np.array([w["cow"] for w in W]); s = np.zeros(len(W))
    for c in np.unique(cows):
        m = cows == c
        clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[~m], y[~m])
        s[m] = clf.predict_proba(X[m])[:, 1]
    return list(s)


def metrics(recs, thr):
    recs = [r for r in recs if r.get("prob") is not None]
    y = np.array([r["label"] for r in recs]); p = np.array([r["prob"] for r in recs])
    out = {"n": len(recs)}
    if len(set(y)) == 2:
        out["auroc"] = roc_auc_score(y, p); out["auprc"] = average_precision_score(y, p)
    cows = sorted({r["cow"] for r in recs}); det, leads, fa, neg_h = 0, [], 0, 0
    for c in cows:
        rc = [r for r in recs if r["cow"] == c]
        hits = [r for r in rc if r["label"] and r["prob"] >= thr]
        if hits:
            det += 1; leads.append(max(r["hours_to_calving"] for r in hits))
        negs = [r for r in rc if not r["label"]]
        fa += sum(r["prob"] >= thr for r in negs); neg_h += len(negs) * STEP_H
    out["detected"] = f"{det}/{len(cows)}"
    out["lead_h"] = float(np.median(leads)) if leads else None
    out["fa_per_cow_week"] = fa / (neg_h / 168) if neg_h else None
    return out


def fmt(name, m):
    g = lambda k, f: (f.format(m[k]) if m.get(k) is not None else "  -  ")
    return (f"{name:<52} {m['n']:>4}  {g('auroc','{:.2f}'):>5}  {g('auprc','{:.2f}'):>5}  "
            f"{m['detected']:>6}  {g('lead_h','{:.0f}'):>5}  {g('fa_per_cow_week','{:.1f}'):>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="windows.jsonl")
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    W = [json.loads(l) for l in open(a.windows)]
    prev = np.mean([w["label"] for w in W])
    print(f"{len(W)} windows, positive rate {prev:.2f} (= chance AUPRC). Alert threshold {a.thr}.\n")
    print(f"{'run':<52} {'n':>4}  {'AUROC':>5}  {'AUPRC':>5}  {'cows':>6}  {'lead':>5}  {'FA/wk':>6}")

    base = lambda s: [dict(id=w["id"], cow=w["cow"], label=w["label"],
                           hours_to_calving=w["hours_to_calving"], prob=float(v)) for w, v in zip(W, s)]
    # rule: z > 1 means "well above her usual spread"; map to 0/1 alert at that point
    rs = rule_scores(W)
    r = base(rs); [d.update(prob=1 / (1 + np.exp(-(v - 1) * 3))) for d, v in zip(r, rs)]
    print(fmt("baseline: isolation rule (gps)", metrics(r, a.thr)))
    for fs, feats in FEATURES.items():
        print(fmt(f"baseline: logistic regression LOCO ({fs})", metrics(base(logreg_scores(W, feats)), a.thr)))
    print()
    for f in sorted(glob.glob("results/*.jsonl")):
        recs = [json.loads(l) for l in open(f)]
        err = sum(r.get("prob") is None for r in recs)
        print(fmt(os.path.basename(f)[:-6] + (f" ({err} err)" if err else ""), metrics(recs, a.thr)))


if __name__ == "__main__":
    main()
