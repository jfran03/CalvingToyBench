"""Build anonymized evaluation windows from the CalvingDetection dataset.

Usage: python prep.py --repo path/to/CalvingDetection --out windows.jsonl

Each window is one "decision point" t for one cow. The model sees the previous
48 hourly rows plus that cow's own baseline stats, and must say whether calving
is imminent. Dates and cow IDs are stripped so nothing can be looked up.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

# UTC calving times, taken from the authors' Stage_1.ipynb (local MST + 7h,
# start of the observed birth window).
CALVING_UTC = {
    "0BD": "2023-02-11 18:30", "0BB": "2023-02-13 19:30", "09F": "2023-02-16 16:30",
    "0E7": "2023-02-23 17:30", "0BC": "2023-02-26 06:30", "12A": "2023-02-27 19:00",
    "13D": "2023-03-07 11:30", "0E1": "2023-03-11 19:00", "09B": "2023-03-13 19:30",
    "14C": "2023-03-14 02:30", "12B": "2023-03-22 19:30", "0C5": "2023-03-27 18:30",
    "0BA": "2023-04-01 19:00", "0BE": "2023-04-02 14:30", "13C": "2023-04-12 00:30",
}

LOOKBACK_H = 48          # rows shown to the model
BASELINE_H = (168, 48)   # baseline stats from t-168h to t-48h
STEP_H = 6               # one decision every 6 hours
RANGE_D = (-10, 2)       # decision points from 10 days before to 2 days after calving
POS_H = (-6, 24)         # positive if calving is within [t-6h, t+24h)

FEATURES = {
    "gps_only":   ["dist_moved_m", "nearest_cow_m", "mean_4_nearest_m"],
    "accel_only": ["activity"],
    "both":       ["dist_moved_m", "nearest_cow_m", "mean_4_nearest_m", "activity"],
}


def load_cow(path):
    d = pd.read_csv(path)
    d["t"] = pd.to_datetime(d["Rounded_Time"], utc=True)
    d = d.set_index("t").sort_index()
    out = pd.DataFrame(index=d.index)
    out["dist_moved_m"] = d["Distance_moved"]
    out["nearest_cow_m"] = d["KNN2"]
    out["mean_4_nearest_m"] = d[["KNN2", "KNN3", "KNN4", "KNN5"]].mean(axis=1)
    out["activity"] = d["ActivityCount_Difference"]
    # regularize to an hourly grid so gaps show up as blanks
    full = pd.date_range(out.index.min(), out.index.max(), freq="h")
    out = out.reindex(full)
    out["hour_local"] = full.tz_convert("America/Denver").hour
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="windows.jsonl")
    a = ap.parse_args()

    rows, n_pos = [], 0
    for f in sorted(glob.glob(os.path.join(a.repo, "Processed_data", "*.csv"))):
        cow = os.path.basename(f)[:3]
        df = load_cow(f)
        calv = pd.Timestamp(CALVING_UTC[cow], tz="UTC")
        start = (calv + pd.Timedelta(days=RANGE_D[0])).floor("h")
        end = calv + pd.Timedelta(days=RANGE_D[1])
        for t in pd.date_range(start, end, freq=f"{STEP_H}h"):
            win = df.loc[t - pd.Timedelta(hours=LOOKBACK_H - 1): t]
            base = df.loc[t - pd.Timedelta(hours=BASELINE_H[0]): t - pd.Timedelta(hours=BASELINE_H[1])]
            if len(win) < LOOKBACK_H or win["dist_moved_m"].notna().sum() < 36 or len(base) < 48:
                continue
            h2c = (calv - t).total_seconds() / 3600
            label = int(POS_H[0] <= h2c < POS_H[1])
            n_pos += label
            stats = {c: {"median": float(np.nanmedian(base[c])), "p90": float(np.nanpercentile(base[c], 90))}
                     for c in FEATURES["both"] if base[c].notna().any()}
            rec = {
                "id": f"{cow}_{t.strftime('%Y%m%dT%H')}",
                "cow": cow, "t_utc": t.isoformat(), "hours_to_calving": round(h2c, 1), "label": label,
                "hours": [{"rel_h": int(-(t - ts).total_seconds() // 3600), "hour_local": int(r.hour_local),
                           **{c: (None if pd.isna(r[c]) else round(float(r[c]), 1)) for c in FEATURES["both"]}}
                          for ts, r in win.iterrows()],
                "baseline": stats,
            }
            rows.append(rec)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(rows)} windows, {n_pos} positive, {len(set(r['cow'] for r in rows))} cows -> {a.out}")


if __name__ == "__main__":
    main()
