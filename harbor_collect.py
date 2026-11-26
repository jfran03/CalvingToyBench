"""Convert a finished Harbor job into the results/*.jsonl format that score.py reads.

Usage: python harbor_collect.py jobs/<job-name>

Why: Harbor's own summary is a mean of each reward key (here: mean brier_reward). AUROC, AUPRC,
cows detected, lead time and false alarms are not means, so score.py stays the one place that
computes metrics, for bench.py runs and Harbor runs alike.

The verifier's reward.json is the source of truth for the probability. Labels, cow and timing
come from harbor_tasks/<features>.index.json (task name -> window), never from the agent.
"""
import argparse, glob, json, os


def collect(job_dir, tasks_root="harbor_tasks", tag=""):
    cfg = json.load(open(os.path.join(job_dir, "config.json")))
    features = os.path.basename(os.path.normpath(cfg["datasets"][0]["path"]))
    model = cfg["agents"][0]["model_name"] or cfg["agents"][0]["name"]
    index = json.load(open(os.path.join(tasks_root, f"{features}.index.json")))

    best = {}  # one record per task; a valid answer beats an error if Harbor retried the trial
    for f in glob.glob(os.path.join(job_dir, "*", "result.json")):
        r = json.load(open(f))
        w = index.get(r["task_name"])
        if w is None:
            continue
        rewards = (r.get("verifier_result") or {}).get("rewards") or {}
        valid = rewards.get("valid") == 1
        reason, error = None, None
        ans = os.path.join(os.path.dirname(f), "artifacts", "app", "answer.json")
        if os.path.exists(ans):
            a = json.load(open(ans))
            reason, error = a.get("reason"), a.get("error")
        if r.get("exception_info"):
            error = f"{r['exception_info'].get('exception_type')}: {r['exception_info'].get('exception_message', '')[:200]}"
        ctx = r.get("agent_result") or {}
        rec = {"id": w["id"], "cow": w["cow"], "label": w["label"], "hours_to_calving": w["hours_to_calving"],
               "prob": rewards["prob"] if valid else None, "reason": reason,
               "usage": {"input_tokens": ctx.get("n_input_tokens"), "output_tokens": ctx.get("n_output_tokens")},
               "trial": r["trial_name"]}
        if not valid:
            rec["error"] = error or "invalid or missing /app/answer.json"
        if w["id"] not in best or (valid and best[w["id"]]["prob"] is None):
            best[w["id"]] = rec

    os.makedirs("results", exist_ok=True)
    out = f"results/harbor__{model.replace('/', '_')}__{features}{tag}.jsonl"  # tag: e.g. _x3 for --samples 3
    with open(out, "w") as fh:
        for rec in sorted(best.values(), key=lambda x: x["id"]):
            fh.write(json.dumps(rec) + "\n")
    n_err = sum(r["prob"] is None for r in best.values())
    print(f"{len(best)}/{len(index)} windows collected ({n_err} without a valid answer) -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--tasks-root", default="harbor_tasks")
    ap.add_argument("--tag", default="", help="suffix for the results file, e.g. _x3")
    a = ap.parse_args()
    collect(a.job_dir, a.tasks_root, a.tag)
