"""Score LLM runs and compare them with two classic baselines.

Usage: python score.py            # default: per-model threshold tuned to a false-alarm budget
       python score.py --fa-lo 0.5 --fa-hi 1.0   # the budget, in false alarms per cow-week
       python score.py --thr 0.5  # fixed alert threshold on the probability instead

Window-level: AUROC and AUPRC (threshold-free; chance AUPRC = positive rate).
Event-level at the threshold: cows detected (any positive window alerted),
median lead time before calving, false alarms per cow-week.
Budget mode: for each held-out cow, the threshold is the lowest one that keeps the OTHER 14 cows'
false alarms <= --fa-hi per cow-week (lowest threshold = most cows caught within budget). It is
applied to the held-out cow and the results are pooled, so the threshold never sees the cow it
scores. The realised FA/wk can land outside [fa-lo, fa-hi]; it is flagged with * when it does.
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


REFRACTORY_H = 0  # set by --refractory; 0 = every window at/above the threshold alerts (original policy)


def alerts(rc, t):
    """Which of one cow's windows raise an alert at threshold t. rc must be in time order.
    With REFRACTORY_H > 0, an alert silences that cow's later windows for that many hours (a rancher who was
    just alerted is already watching her). Causal: depends only on earlier windows' scores."""
    out, last = [], -1e9
    for i, r in enumerate(rc):
        a = r["prob"] >= t and (REFRACTORY_H == 0 or i * STEP_H - last >= REFRACTORY_H)
        if a:
            last = i * STEP_H
        out.append(a)
    return out


def by_cow_sorted(recs):
    """{cow: its windows in time order}. Window ids end in the decision time (YYYYMMDDTHH), so they sort by time."""
    d = {}
    for r in recs:
        d.setdefault(r["cow"], []).append(r)
    return {c: sorted(v, key=lambda r: r["id"].split("_", 1)[1]) for c, v in d.items()}


_CURVES = {}


def _fa_curve(rc):
    """One cow's false-alarm count as a step function of the threshold. Her alerts can only change at her own
    distinct scores, so the count at any t equals the count at her smallest score >= t. Cached, because the
    leave-one-cow-out folds and bootstrap replicates reuse the same cows."""
    key = (REFRACTORY_H, tuple((r["prob"], r["label"]) for r in rc))
    if key not in _CURVES:
        vals = np.unique([r["prob"] for r in rc])
        _CURVES[key] = (vals, np.array([sum(a and not r["label"] for r, a in zip(rc, alerts(rc, v))) for v in vals]))
    return _CURVES[key]


def _fa_at(curve, ts):
    vals, fa = curve
    i = np.searchsorted(vals, ts, side="left")
    return np.where(i < len(vals), fa[np.minimum(i, len(vals) - 1)], 0)


def pick_threshold(recs, fa_hi):
    """Lowest probability threshold whose false alarms stay <= fa_hi per cow-week on these recs.
    inf (never alert) if even the highest score exceeds the budget."""
    if REFRACTORY_H:
        cows = by_cow_sorted(recs)
        weeks = sum(not r["label"] for r in recs) * STEP_H / 168
        cands = np.unique([r["prob"] for r in recs])  # ascending: first one within budget is the lowest
        fa = sum(_fa_at(_fa_curve(rc), cands) for rc in cows.values())  # exact, not assuming monotonicity
        ok = np.where(fa / weeks <= fa_hi)[0]
        return float(cands[ok[0]]) if len(ok) else float("inf")
    negs = np.sort([r["prob"] for r in recs if not r["label"]])
    weeks = len(negs) * STEP_H / 168
    cands = np.unique([r["prob"] for r in recs])  # ascending; FA is non-increasing in the threshold
    fa = (len(negs) - np.searchsorted(negs, cands, side="left")) / weeks  # negatives with prob >= t
    ok = np.where(fa <= fa_hi)[0]
    return float(cands[ok[0]]) if len(ok) else float("inf")


def evaluate(recs, thr, fa_lo, fa_hi):
    """metrics() at a fixed threshold (thr is a number), or, if thr is None, at the leave-one-cow-out
    false-alarm-budget thresholds. Adds 'thr' (median across folds) and 'in_range'."""
    if thr is not None:
        out = metrics(recs, thr); out["thr"] = thr; out["in_range"] = True
        return out
    recs = [r for r in recs if r.get("prob") is not None]
    per_cow = {c: pick_threshold([r for r in recs if r["cow"] != c], fa_hi) for c in {r["cow"] for r in recs}}
    out = metrics(recs, per_cow)
    finite = [t for t in per_cow.values() if np.isfinite(t)]
    out["thr"] = float(np.median(finite)) if finite else None
    fa = out["fa_per_cow_week"]
    out["in_range"] = fa is not None and fa_lo <= fa <= fa_hi
    return out


def metrics(recs, thr):
    """thr is a number, or a {cow: threshold} dict (used by the leave-one-cow-out budget mode)."""
    recs = [r for r in recs if r.get("prob") is not None]
    y = np.array([r["label"] for r in recs]); p = np.array([r["prob"] for r in recs])
    out = {"n": len(recs)}
    if len(set(y)) == 2:
        out["auroc"] = roc_auc_score(y, p); out["auprc"] = average_precision_score(y, p)
    per_cow = by_cow_sorted(recs); cows = sorted(per_cow); det, leads, fa, neg_h = 0, [], 0, 0
    for c in cows:
        rc = per_cow[c]
        t = thr[c] if isinstance(thr, dict) else thr
        al = alerts(rc, t)
        hits = [r for r, a in zip(rc, al) if r["label"] and a]
        if hits:
            det += 1; leads.append(max(r["hours_to_calving"] for r in hits))
        fa += sum(a for r, a in zip(rc, al) if not r["label"]); neg_h += sum(not r["label"] for r in rc) * STEP_H
    out["detected"] = f"{det}/{len(cows)}"
    out["lead_h"] = float(np.median(leads)) if leads else None
    out["fa_per_cow_week"] = fa / (neg_h / 168) if neg_h else None
    return out


def fmt(name, m):
    g = lambda k, f: (f.format(m[k]) if m.get(k) is not None else "  -  ")
    star = "" if m.get("in_range", True) else "*"
    return (f"{name:<52} {m['n']:>4}  {g('auroc','{:.2f}'):>5}  {g('auprc','{:.2f}'):>5}  {g('thr','{:.2f}'):>5}  "
            f"{m['detected']:>6}  {g('lead_h','{:.0f}'):>5}  {g('fa_per_cow_week','{:.1f}') + star:>6}")


def bootstrap(runs, cows, evaluate_fn, n_boot, seed=0):
    """Cluster bootstrap over cows. Each replicate resamples the cows with replacement (a cow drawn twice
    counts twice, and its windows stay together) and re-runs the whole pipeline, including the
    leave-one-cow-out threshold tuning. The same resample is applied to every run, so differences between
    runs are paired. Uses saved scores only: no model calls. Returns {name: {metric: array of n_boot}}."""
    by_cow = {name: {c: [] for c in cows} for name, _ in runs}
    for name, recs in runs:
        for r in recs:
            if r["cow"] in by_cow[name]:
                by_cow[name][r["cow"]].append(r)
    out = {name: {"auroc": [], "auprc": [], "cows": [], "fa": []} for name, _ in runs}
    for b in range(n_boot):
        draw = np.random.default_rng([seed, b]).integers(0, len(cows), len(cows))
        for name, _ in runs:
            # rename each drawn copy so duplicates of one cow are separate cows to metrics()/pick_threshold()
            recs = [dict(r, cow=f"{cows[i]}#{j}") for j, i in enumerate(draw) for r in by_cow[name][cows[i]]]
            m = evaluate_fn(recs)
            out[name]["auroc"].append(m.get("auroc", np.nan)); out[name]["auprc"].append(m.get("auprc", np.nan))
            out[name]["cows"].append(int(m["detected"].split("/")[0]))
            out[name]["fa"].append(np.nan if m["fa_per_cow_week"] is None else m["fa_per_cow_week"])
    return {n: {k: np.array(v, dtype=float) for k, v in d.items()} for n, d in out.items()}


def print_bootstrap(res, ref, n_boot, n_cows):
    ci = lambda x, f: (lambda q: f"{f.format(q[1])} [{f.format(q[0])}, {f.format(q[2])}]")(np.nanpercentile(x, [2.5, 50, 97.5]))
    print(f"\nCluster bootstrap over cows ({n_boot} resamples of {n_cows} cows): median [95% CI]. "
          f"Delta = paired difference vs '{ref}'.\n")
    print(f"{'run':<52} {'AUROC':>20}  {'AUPRC':>20}  {'cows detected':>18}  {'delta cows vs ref':>20}  {'FA/wk':>18}")
    for name, d in res.items():
        delta = "" if name == ref else ci(d["cows"] - res[ref]["cows"], "{:.0f}")
        print(f"{name:<52} {ci(d['auroc'], '{:.2f}'):>20}  {ci(d['auprc'], '{:.2f}'):>20}  "
              f"{ci(d['cows'], '{:.0f}'):>18}  {delta:>20}  {ci(d['fa'], '{:.1f}'):>18}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000, help="cow-level bootstrap resamples for the CIs (0 = skip)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--windows", default="windows.jsonl")
    ap.add_argument("--thr", type=float, default=None, help="fixed threshold; default = tune to the FA budget")
    ap.add_argument("--fa-lo", type=float, default=0.5)
    ap.add_argument("--fa-hi", type=float, default=1.0)
    ap.add_argument("--refractory", type=int, default=0, metavar="H",
                    help="after an alert, silence that cow for H hours (e.g. 24). Default 0 = original policy")
    ap.add_argument("--only", default="", help="comma-separated substrings: only score results files matching one")
    a = ap.parse_args()
    global REFRACTORY_H
    REFRACTORY_H = a.refractory
    W = [json.loads(l) for l in open(a.windows)]
    prev = np.mean([w["label"] for w in W])
    mode = (f"Alert threshold {a.thr}" if a.thr is not None else
            f"Per-model leave-one-cow-out threshold for {a.fa_lo}-{a.fa_hi} false alarms/cow-week (* = realised rate outside it)")
    print(f"{len(W)} windows, positive rate {prev:.2f} (= chance AUPRC). {mode}.\n")
    print(f"{'run':<52} {'n':>4}  {'AUROC':>5}  {'AUPRC':>5}  {'thr':>5}  {'cows':>6}  {'lead':>5}  {'FA/wk':>6}")
    metrics_ = lambda recs: evaluate(recs, a.thr, a.fa_lo, a.fa_hi)

    base = lambda s: [dict(id=w["id"], cow=w["cow"], label=w["label"],
                           hours_to_calving=w["hours_to_calving"], prob=float(v)) for w, v in zip(W, s)]
    # rule: z > 1 means "well above her usual spread"; map to 0/1 alert at that point
    rs = rule_scores(W)
    r = base(rs); [d.update(prob=1 / (1 + np.exp(-(v - 1) * 3))) for d, v in zip(r, rs)]
    runs = [("baseline: isolation rule (gps)", r)]
    for fs, feats in FEATURES.items():
        runs.append((f"baseline: logistic regression LOCO ({fs})", base(logreg_scores(W, feats))))
    n_base = len(runs)
    for f in sorted(glob.glob("results/*.jsonl")):
        if a.only and not any(k in f for k in a.only.split(",")):
            continue
        recs = [json.loads(l) for l in open(f)]
        err = sum(r.get("prob") is None for r in recs)
        runs.append((os.path.basename(f)[:-6] + (f" ({err} err)" if err else ""), recs))
    for i, (name, recs) in enumerate(runs):
        if i == n_base:
            print()
        print(fmt(name, metrics_(recs)))

    if a.boot > 0:
        cows = sorted({w["cow"] for w in W})
        res = bootstrap(runs, cows, metrics_, a.boot, a.seed)
        print_bootstrap(res, runs[0][0], a.boot, len(cows))


if __name__ == "__main__":
    main()
