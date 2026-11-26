"""Herd context for each window: what the OTHER collared cows did over the same past hours.

Usage: python herd_context.py --repo path/to/CalvingDetection      # -> herd_context.json

Why: a cow that walks less than usual on a day when the whole herd walks less (heat, wind, a move of
water point) is not showing a calving sign. Comparing her with the herd over the same hours removes those
herd-wide swings. Only the past is used: for decision time t, each period covers hours (t-h, t], the same
hours the cow's own summary uses. The other cows are anonymous (we keep only the herd median and how many
cows had data), so no cow IDs or dates reach the prompt.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

from prep import FEATURES, load_cow

PERIODS = {"last 6h": 6, "last 12h": 12, "last 24h": 24}
MIN_FRAC = 1 / 3  # a cow counts for a period only if at least a third of its hours have data


def period_mean(df, t, h, c):
    s = df[c].loc[t - pd.Timedelta(hours=h - 1): t]
    return float(s.mean()) if s.notna().sum() >= max(1, int(h * MIN_FRAC)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--windows", default="windows.jsonl")
    ap.add_argument("--out", default="herd_context.json")
    a = ap.parse_args()
    cows = {os.path.basename(f)[:3]: load_cow(f) for f in glob.glob(os.path.join(a.repo, "Processed_data", "*.csv"))}
    out = {}
    for line in open(a.windows):
        w = json.loads(line)
        t = pd.Timestamp(w["t_utc"])
        ctx = {}
        for name, h in PERIODS.items():
            ctx[name] = {}
            for c in FEATURES["both"]:
                v = [m for k, d in cows.items() if k != w["cow"] and (m := period_mean(d, t, h, c)) is not None]
                ctx[name][c] = {"median": float(np.median(v)), "n": len(v)} if v else None
        out[w["id"]] = ctx
    json.dump(out, open(a.out, "w"))
    print(f"{len(out)} windows -> {a.out}")


if __name__ == "__main__":
    main()
