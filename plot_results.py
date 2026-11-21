"""Comparison figure: hand-built baselines vs the LLM runs, plus the score.py table.

Usage: python plot_results.py            # -> figures/comparison.png
       python plot_results.py --thr 0.5  # alert threshold for the table's event-level columns

All numbers come from score.py's own functions (metrics, rule_scores, logreg_scores), so the
figure can never disagree with `python score.py`. Curves are threshold-free; the table uses --thr.
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from score import FEATURES, logreg_scores, metrics, rule_scores

# List prices in USD per million tokens (input, output), as of 2026-11. Update if prices change.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}

LLM_RUNS = [  # (label, results file, model id for pricing)
    ("Claude Haiku 5.5", "results/harbor__anthropic_claude-haiku-5-5__both.jsonl", "claude-haiku-5-5"),
    ("Claude Sonnet 5.5", "results/harbor__anthropic_claude-sonnet-5-5__both.jsonl", "claude-sonnet-5-5"),
]

# Chart ink (reference palette, light mode). Baselines are recessive grey; the LLMs carry colour.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {
    "Isolation rule (GPS)":       dict(color=INK2, ls="-"),
    "Logistic regression (both)": dict(color=MUTED, ls=(0, (4, 2))),
    "Claude Haiku 5.5":           dict(color="#2a78d6", ls="-"),
    "Claude Sonnet 5.5":          dict(color="#eb6834", ls="-"),
}


def load_runs(W):
    base = lambda s: [dict(id=w["id"], cow=w["cow"], label=w["label"],
                           hours_to_calving=w["hours_to_calving"], prob=float(v)) for w, v in zip(W, s)]
    rs = rule_scores(W)  # same sigmoid as score.py so the table's thresholded columns match it
    rule = base(rs); [d.update(prob=1 / (1 + np.exp(-(v - 1) * 3))) for d, v in zip(rule, rs)]
    runs = {"Isolation rule (GPS)": (rule, None)}
    for fs, feats in FEATURES.items():
        runs[f"Logistic regression ({fs})"] = (base(logreg_scores(W, feats)), None)
    for label, path, model in LLM_RUNS:
        if os.path.exists(path):
            runs[label] = ([json.loads(l) for l in open(path)], model)
    return runs


def cost_usd(recs, model):
    if model not in PRICES:
        return None
    pi, po = PRICES[model]
    i = sum((r.get("usage") or {}).get("input_tokens") or 0 for r in recs)
    o = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in recs)
    return i / 1e6 * pi + o / 1e6 * po


def tradeoff(recs):
    """Cows detected vs false alarms per cow-week, sweeping the threshold over every distinct score."""
    ts = sorted({r["prob"] for r in recs if r["prob"] is not None}, reverse=True)
    fa, det = [0.0], [0]
    for t in ts:
        m = metrics(recs, t)
        fa.append(m["fa_per_cow_week"]); det.append(int(m["detected"].split("/")[0]))
    return fa, det


def style_axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2); ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="windows.jsonl")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="figures/comparison.png")
    a = ap.parse_args()
    W = [json.loads(l) for l in open(a.windows)]
    prev = np.mean([w["label"] for w in W])
    runs = load_runs(W)
    plotted = [k for k in STYLE if k in runs]

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    fig = plt.figure(figsize=(15, 9.0), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.55], hspace=0.30, wspace=0.26,
                          left=0.05, right=0.98, top=0.86, bottom=0.03)
    ax_roc, ax_pr, ax_tr = (fig.add_subplot(gs[0, i]) for i in range(3))

    for name in plotted:
        recs = [r for r in runs[name][0] if r["prob"] is not None]
        y = [r["label"] for r in recs]; p = [r["prob"] for r in recs]
        st = STYLE[name]
        fpr, tpr, _ = roc_curve(y, p); ax_roc.plot(fpr, tpr, lw=2, **st, label=name)
        pr, rc, _ = precision_recall_curve(y, p); ax_pr.plot(rc[:-1], pr[:-1],  # drop sklearn's (0, 1) end point
                lw=2, drawstyle="steps-post", **st, label=name)
        fa, det = tradeoff(runs[name][0]); ax_tr.plot(fa, det, lw=2, drawstyle="steps-post", **st, label=name)

    ax_roc.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=":")
    ax_roc.text(0.62, 0.55, "chance (0.50)", color=MUTED, fontsize=8, rotation=0)
    style_axes(ax_roc, "ROC: ranking calving windows above others", "False positive rate", "True positive rate")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.01)

    ax_pr.axhline(prev, color=AXIS, lw=1, ls=":")
    ax_pr.text(0.99, prev + 0.015, f"chance ({prev:.2f})", color=MUTED, fontsize=8, ha="right")
    style_axes(ax_pr, "Precision-recall: are the top-ranked windows right?", "Recall", "Precision")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.01)

    ax_tr.axvline(1, color=AXIS, lw=1, ls=":")
    ax_tr.annotate("1 false alarm per cow-week", xy=(1.0, 0.8), xytext=(3.4, 0.8), color=MUTED, fontsize=8,
                   va="center", arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
    style_axes(ax_tr, "Alert trade-off: cows caught vs false alarms", "False alarms per cow-week", "Cows detected (of 15)")
    ax_tr.set_xlim(0, 15); ax_tr.set_ylim(0, 15.5)

    h, l = ax_roc.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.05, 0.935), ncol=4, frameon=False, fontsize=9,
               labelcolor=INK2, handlelength=2.6)
    fig.text(0.05, 0.965, "Zero-shot LLMs vs hand-built baselines: calving alerts from collar data",
             fontsize=14, color=INK, weight="bold")
    fig.text(0.98, 0.965, f"735 windows, 15 cows, {prev:.0%} positive. Baselines leave-one-cow-out.",
             fontsize=9, color=MUTED, ha="right")

    # Table: the same rows and columns as `python score.py --thr <thr>`, plus estimated cost.
    ax_t = fig.add_subplot(gs[1, :]); ax_t.axis("off")
    cols = ["Run", "n", "AUROC", "AUPRC", f"Cows detected\n(thr {a.thr})", "Median lead (h)\n(+ = before calving)",
            "False alarms\nper cow-week", "Est. cost\n(USD, list price)"]
    rows = []
    for name, (recs, model) in runs.items():
        m = metrics(recs, a.thr); c = cost_usd(recs, model) if model else None
        f = lambda v, s: "-" if v is None else s.format(v)
        rows.append([name, m["n"], f(m.get("auroc"), "{:.2f}"), f(m.get("auprc"), "{:.2f}"), m["detected"],
                     ("-" if m["lead_h"] is None else ("0" if round(m["lead_h"]) == 0 else f"{m['lead_h']:+.0f}")), f(m["fa_per_cow_week"], "{:.1f}"),
                     "-" if model is None else ("n/a" if c is None else f"${c:.2f}")])
    tbl = ax_t.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center",
                     colWidths=[0.24, 0.05, 0.08, 0.08, 0.12, 0.15, 0.12, 0.12])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.75)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_linewidth(0.6)
        cell.set_facecolor(SURFACE); cell.get_text().set_color(INK)
        if r == 0:
            cell.get_text().set_color(INK2); cell.get_text().set_weight("bold"); cell.set_height(cell.get_height() * 1.5)
        if c == 0 and r > 0:
            cell.get_text().set_ha("left"); cell._loc = "left"; cell.PAD = 0.03
    ax_t.set_title(f"Scores at alert threshold {a.thr} (same as python score.py --thr {a.thr})",
                   loc="left", fontsize=11, color=INK, pad=4)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
