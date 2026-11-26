"""Comparison figure: hand-built baselines vs the LLM runs, plus the score.py table.

Usage: python plot_results.py            # -> figures/comparison.png, table at the false-alarm budget
       python plot_results.py --thr 0.5  # fixed alert threshold for the table instead
       python plot_results.py --fa-lo 0.5 --fa-hi 1.0

All numbers come from score.py's own functions (evaluate, rule_scores, logreg_scores), so the
figure can never disagree with `python score.py`. Curves are threshold-free; the table uses the
per-model leave-one-cow-out threshold tuned to the budget (see score.py), or --thr if given.
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from score import FEATURES, bootstrap, evaluate, logreg_scores, metrics, rule_scores

# List prices in USD per million tokens (input, output), as of 2026-11. Update if prices change.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "gpt-6-luna":(0.10,0.50), "gpt-6.2-sol": (2.00, 10.00), "jev-1.14":(0.042,0.0), "gpt-6.2-astra":(8.00,40.00), "claude-opus-5-5":(4.00,20.00)}

LLM_RUNS = [  # (label, results file, model id for pricing)
    ("CalvingDetection AE only (Wang et al.)", "results/cd__ae_only.jsonl", None),  # authors' pretrained models, not LLMs
    ("CalvingDetection AE+RF (Wang et al.)", "results/cd__rf_pretrained.jsonl", None),
    ("Claude Haiku 5.5 (both)", "results/harbor__anthropic_claude-haiku-5-5__both.jsonl", "claude-haiku-5-5"),
    ("Claude Haiku 5.5 Slide_Win (both)", "results/harbor__anthropic_claude-haiku-5-5__both_SW.jsonl", "claude-haiku-5-5"),
    ("Claude Haiku 5.5 Full_Win (both)", "results/harbor__anthropic_claude-haiku-5-5__both_FW.jsonl", "claude-haiku-5-5"),
    ("Claude Sonnet 5.5 (both)", "results/harbor__anthropic_claude-sonnet-5-5__both.jsonl", "claude-sonnet-5-5"),
    ("Claude Opus 5.5 (both)", "results/harbor__anthropic_claude-opus-5-5__both.jsonl", "claude-opus-5-5"),
    ("GPT-6 Luna (both)", "results/harbor__openai_gpt-6-luna__both.jsonl", "gpt-6-luna"),
    ("GPT-6 Luna Slide_Win (both)", "results/harbor__openai_gpt-6-luna__both_SW.jsonl", "gpt-6-luna"),
    ("GPT-6.2 Sol (both)", "results/harbor__openai_gpt-6.2-sol__both.jsonl", "gpt-6.2-sol"),
    ("GPT-6.2 Astra (both)", "results/harbor__openai_gpt-6.2-astra__both.jsonl", "gpt-6.2-astra"),
    ("TypeSafe Jev 1.14 (both)", "results/harbor__jev_jev-1.14.0__both.jsonl", "jev-1.14"),
]

# Chart ink (reference palette, light mode). Baselines are recessive grey; the LLMs carry colour.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {
    "Isolation rule (GPS)":       dict(color=INK2, ls="-"),
    "Logistic regression (both)": dict(color=MUTED, ls=(0, (4, 2))),
    "CalvingDetection AE only (Wang et al.)": dict(color="#17a2b8", ls=(0, (1, 1.5))),
    "CalvingDetection AE+RF (Wang et al.)":   dict(color="#d6336c", ls=(0, (1, 1.5))),
    "Claude Haiku 5.5 (both)":    dict(color="#2a78d6", ls="-"),
    "Claude Haiku 5.5 Slide_Win (both)": dict(color="#2a78d6", ls=(0, (5, 2))),  # same blue as Haiku, dashed
    "Claude Haiku 5.5 Full_Win (both)": dict(color="#2a78d6", ls=(0, (1, 1.5))),  # same blue as Haiku, dotted
    "Claude Sonnet 5.5 (both)":   dict(color="#eb6834", ls="-"),
    "Claude Opus 5.5 (both)":     dict(color="#b5121b", ls="-"),
    "GPT-6 Luna (both)":          dict(color="#2a9d62", ls="-"),
    "GPT-6 Luna Slide_Win (both)": dict(color="#2a9d62", ls=(0, (5, 2))),  # same green as Luna, dashed
    "GPT-6.2 Sol (both)":         dict(color="#8a4fd0", ls="-"),
    "GPT-6.2 Astra (both)":       dict(color="#7a4b2a", ls="-"),
    "TypeSafe Jev 1.14 (both)":   dict(color="#c9a227", ls="-"),
}


def load_runs(W):
    base = lambda s: [dict(id=w["id"], cow=w["cow"], label=w["label"],
                           hours_to_calving=w["hours_to_calving"], prob=float(v)) for w, v in zip(W, s)]
    rs = rule_scores(W)  # same sigmoid as score.py so the table's thresholded columns match it
    rule = base(rs); [d.update(prob=1 / (1 + np.exp(-(v - 1) * 3))) for d, v in zip(rule, rs)]
    runs = {"Isolation rule (GPS)": (rule, None)}
    for fs, feats in FEATURES.items():
        runs[f"Logistic regression ({fs})"] = (base(logreg_scores(W, feats)), None)
    failed = {}  # label -> reason, for runs with no usable probabilities (e.g. every call errored)
    for label, path, model in LLM_RUNS:
        if not os.path.exists(path):
            continue
        recs = [json.loads(l) for l in open(path)]
        if not any(r.get("prob") is not None for r in recs):
            err = next((r["error"] for r in recs if r.get("error")), "no probabilities")
            failed[label] = err.split(":")[0] + (" 404 (model id not found)" if "404" in err else "")
            continue
        runs[label] = (recs, model)
    return runs, failed


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
    ap.add_argument("--thr", type=float, default=None, help="fixed threshold; default = tune to the FA budget")
    ap.add_argument("--fa-lo", type=float, default=0.5)
    ap.add_argument("--fa-hi", type=float, default=1.0)
    ap.add_argument("--boot", type=int, default=1000, help="cow-level bootstrap resamples for the table CIs (0 = skip)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--out", default="figures/comparison.png")
    a = ap.parse_args()
    W = [json.loads(l) for l in open(a.windows)]
    prev = np.mean([w["label"] for w in W])
    runs, failed = load_runs(W)
    plotted = [k for k in STYLE if k in runs]

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    fig = plt.figure(figsize=(15, 13.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.0], hspace=0.24, wspace=0.26,
                          left=0.05, right=0.98, top=0.85, bottom=0.01)
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

    ax_tr.axvspan(a.fa_lo, a.fa_hi, color=AXIS, alpha=0.45, lw=0)
    ax_tr.annotate(f"target: {a.fa_lo:g}-{a.fa_hi:g} false alarms per cow-week", xy=(a.fa_hi, 0.8), xytext=(3.4, 0.8),
                   color=MUTED, fontsize=8, va="center", arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
    style_axes(ax_tr, "Alert trade-off: cows caught vs false alarms", "False alarms per cow-week", "Cows detected (of 15)")
    ax_tr.set_xlim(0, 15); ax_tr.set_ylim(0, 15.5)

    h, l = ax_roc.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.05, 0.945), ncol=5, frameon=False, fontsize=9,
               labelcolor=INK2, handlelength=2.6)
    fig.text(0.05, 0.972, "Zero-shot LLMs vs baselines and the authors' pretrained models: calving alerts from collar data",
             fontsize=14, color=INK, weight="bold")
    fig.text(0.98, 0.972, f"735 windows, 15 cows, {prev:.0%} positive. Baselines leave-one-cow-out.",
             fontsize=9, color=MUTED, ha="right")

    # Table: the same rows and columns as `python score.py --thr <thr>`, plus estimated cost.
    ax_t = fig.add_subplot(gs[1, :]); ax_t.axis("off")
    # 95% CIs: cluster bootstrap over cows on the saved scores (no model calls); see score.bootstrap.
    boot = None
    if a.boot > 0:
        cows = sorted({w["cow"] for w in W})
        boot = bootstrap([(n, r) for n, (r, _) in runs.items()], cows,
                         lambda recs: evaluate(recs, a.thr, a.fa_lo, a.fa_hi), a.boot, a.seed)
    ref = "Isolation rule (GPS)"
    ci = lambda x, f: (lambda q: f"[{f.format(q[0])}, {f.format(q[1])}]")(np.nanpercentile(x, [2.5, 97.5]))
    cols = ["Run", "AUROC\n[95% CI]", "AUPRC\n[95% CI]", "Alert thr\n(median)", "Cows detected\n(of 15) [95% CI]",
            "Cows vs isolation rule\n[95% CI of difference]", "Median lead (h)\n(+ = before calving)",
            "False alarms\nper cow-week", "Est. cost\n(USD, list price)"]
    rows = []
    for name, (recs, model) in runs.items():
        m = evaluate(recs, a.thr, a.fa_lo, a.fa_hi); c = cost_usd(recs, model) if model else None
        f = lambda v, s: "-" if v is None else s.format(v)
        b = boot[name] if boot else None
        det = m["detected"] + (f"\n{ci(b['cows'], '{:.0f}')}" if b else "")
        delta = "-" if (not b or name == ref) else f"{ci(b['cows'] - boot[ref]['cows'], '{:+.0f}')}"
        rows.append([name,
                     f(m.get("auroc"), "{:.2f}") + (f"\n{ci(b['auroc'], '{:.2f}')}" if b else ""),
                     f(m.get("auprc"), "{:.2f}") + (f"\n{ci(b['auprc'], '{:.2f}')}" if b else ""),
                     f(m["thr"], "{:.2f}"), det, delta,
                     ("-" if m["lead_h"] is None else ("0" if round(m["lead_h"]) == 0 else f"{m['lead_h']:+.0f}")),
                     f(m["fa_per_cow_week"], "{:.1f}") + ("" if m["in_range"] else "*"),
                     "-" if model is None else ("n/a" if c is None else f"${c:.2f}")])
    tbl = ax_t.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center",
                     colWidths=[0.20, 0.09, 0.09, 0.07, 0.11, 0.14, 0.11, 0.09, 0.10])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_linewidth(0.6)
        cell.set_facecolor(SURFACE); cell.get_text().set_color(INK)
        if r == 0:
            cell.get_text().set_color(INK2); cell.get_text().set_weight("bold"); cell.set_height(cell.get_height() * 1.5)
        if c == 0 and r > 0:
            cell.get_text().set_ha("left"); cell._loc = "left"; cell.PAD = 0.03
    note = "".join(f" Not shown, all calls failed: {k} ({v})." for k, v in failed.items())
    head = (f"Scores at alert threshold {a.thr} (same as python score.py --thr {a.thr})." if a.thr is not None else
            f"Scores with each model's threshold tuned leave-one-cow-out to {a.fa_lo:g}-{a.fa_hi:g} false alarms per cow-week "
            f"(* = realised rate outside that range).")
    ci_note = (f" CIs: {a.boot} bootstrap resamples of the 15 cows (windows from a cow stay together)." if boot else "")
    ax_t.set_title(f"{head}\nboth = GPS + accelerometer.{ci_note}{note}", loc="left", fontsize=10.5, color=INK, pad=4)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
