"""Turn windows.jsonl into Harbor datasets: one task folder per window, one dataset per feature set.

Usage: python harbor_export.py                 # all three feature sets
       python harbor_export.py --features both

Layout written:
  harbor_tasks/<features>/w<hash>/
      instruction.md     the exact bench.py prompt (what the model sees)
      task.toml          sandbox image, offline network, separate verifier, artifact to keep
      tests/test.sh      verifier entrypoint (runs verify.py)
      tests/verify.py    scores /app/answer.json against the label
      tests/truth.json   the label. Only uploaded to the verifier sandbox, never the agent's.
      solution/solve.sh  "oracle" answer (writes the true label) to sanity-check the verifier
  With --sliding 10 the dataset is <features>_SW (e.g. both_SW): same prompt, longer history.
  harbor_tasks/<features>.index.json   task name -> window id, cow, label, hours_to_calving
                                       (kept OUTSIDE the dataset folder so it is not a task)

Task names are a hash of the window id, so nothing in a task name reveals the cow or the date.
"""
import argparse, hashlib, json, os, shutil

from bench import build_prompt, build_prompt_binned, build_prompt_digest
from prep import FEATURES

TASK_TOML = """schema_version = "1.4"
artifacts = ["/app/answer.json"]

[metadata]
benchmark = "agrivanna-calving"
features = "{features}"
category = "classification"

[environment]
docker_image = "python:3.12-slim"
network_mode = "no-network"

[agent]
timeout_sec = 600

[verifier]
timeout_sec = 60
environment_mode = "separate"
"""

TEST_SH = """#!/bin/bash
set -euo pipefail
python3 /tests/verify.py
"""

VERIFY_PY = '''"""Score one window. Writes /logs/verifier/reward.json with:
  prob          the model's probability (copied through so results can be collected)
  valid         1 if /app/answer.json held a probability in [0, 1], else 0
  brier_reward  1 - (prob - label)^2, so 1 is perfect and 0 is confidently wrong.
                Invalid answers get 0. Harbor's job summary shows the mean of this.
AUROC, AUPRC and the per-cow metrics are not averages, so they come from score.py instead.
"""
import json, math

label = json.load(open("/tests/truth.json"))["label"]
reward = {"valid": 0, "brier_reward": 0.0}
try:
    p = json.load(open("/app/answer.json"))["probability"]
    p = float(p)
    if math.isfinite(p) and 0.0 <= p <= 1.0:
        reward = {"prob": p, "valid": 1, "brier_reward": 1.0 - (p - label) ** 2}
except Exception as e:
    print(f"invalid answer: {e}")
json.dump(reward, open("/logs/verifier/reward.json", "w"))
print(reward)
'''

SOLVE_SH = """#!/bin/bash
mkdir -p /app
echo '{{"probability": {label}, "reason": "oracle"}}' > /app/answer.json
"""


def task_name(window_id):
    return "w" + hashlib.sha256(window_id.encode()).hexdigest()[:10]


def write(path, text, executable=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as fh:  # LF endings: these run in a Linux container
        fh.write(text)
    if executable:
        os.chmod(path, 0o755)


FULL_N = 10**6  # "no cap" for --full: every prior window of the cow
STEP_H = 6  # hours between consecutive windows (prep.py)


def sliding_windows(windows, n):
    """Per cow, give window k the hourly rows of windows k-n+1..k, de-duplicated.

    Consecutive windows are 6h apart but each holds 48 hourly rows, so they overlap by 42h; stacking
    them naively would repeat data. Instead take the union of hours, re-expressed relative to window k's
    own time (rel_h=0 is still "now"). Cow's first window = its own 48h; window 11 (n=10) spans
    windows 2..11 = 102h. The history never crosses cows because each cow is processed alone, and
    only ever looks back in time (no future data), so labels cannot leak.
    """
    by_cow = {}
    for w in windows:
        by_cow.setdefault(w["cow"], []).append(w)
    out = []
    for cow, ws in by_cow.items():
        ws.sort(key=lambda w: w["t_utc"])
        for k, w in enumerate(ws):
            rows = {}  # rel_h relative to w -> hour row; later windows overwrite (identical anyway)
            for j in range(max(0, k - n + 1), k + 1):
                shift = (k - j) * STEP_H  # window j is `shift` hours older than w
                for h in ws[j]["hours"]:
                    rows[h["rel_h"] - shift] = {**h, "rel_h": h["rel_h"] - shift}
            out.append({**w, "hours": [rows[r] for r in sorted(rows)], "n_windows": min(k + 1, n)})
    return out


# Redesigned tasks (docs/task-redesign-runs.md). All use the cow's full past history (never the future).
# variant -> (dataset suffix, prompt builder)
VARIANTS = {
    "dg":  ("_DG",  lambda w, f: build_prompt_digest(w, f)),
    "dgc": ("_DGC", lambda w, f: build_prompt_digest(w, f, cues=True)),
    "dgd": ("_DGD", lambda w, f: build_prompt_digest(w, f, cues=True, decompose=True)),
    "dgh": ("_DGH", lambda w, f: build_prompt_digest(w, f, cues=True, herd=_herd()[w["id"]])),
    "dga": ("_DGA", lambda w, f: build_prompt_digest(w, f, cues=True, aligned=True)),
    "dgad": ("_DGAD", lambda w, f: build_prompt_digest(w, f, cues=True, aligned=True, decompose=True)),
    "dgha": ("_DGHA", lambda w, f: build_prompt_digest(w, f, cues=True, herd=_herd()[w["id"]], aligned=True)),
}
_HERD = {}


def _herd():
    """herd_context.json (built by herd_context.py from the raw dataset), loaded on first use."""
    if not _HERD:
        _HERD.update(json.load(open("herd_context.json")))
    return _HERD


def export(windows, fs, out_root, sliding=0, suffix="_SW", binned=False, builder=None):
    feats = FEATURES[fs]
    name_ds = fs + (suffix if sliding else "")
    if sliding:
        windows = sliding_windows(windows, sliding)
    ds = os.path.join(out_root, name_ds)
    shutil.rmtree(ds, ignore_errors=True)
    index = {}
    for w in windows:
        name = task_name(w["id"])
        assert name not in index, f"hash collision on {name}"
        t = os.path.join(ds, name)
        prompt = (builder(w, feats) if builder else build_prompt_binned(w, feats) if binned
                  else build_prompt(w, feats, lookback=len(w["hours"])))
        write(os.path.join(t, "instruction.md"), prompt + "\n")
        write(os.path.join(t, "task.toml"), TASK_TOML.format(features=name_ds))
        write(os.path.join(t, "tests", "test.sh"), TEST_SH, executable=True)
        write(os.path.join(t, "tests", "verify.py"), VERIFY_PY)
        write(os.path.join(t, "tests", "truth.json"), json.dumps({"label": w["label"]}))
        write(os.path.join(t, "solution", "solve.sh"), SOLVE_SH.format(label=w["label"]), executable=True)
        index[name] = {k: w[k] for k in ("id", "cow", "label", "hours_to_calving")}
    with open(os.path.join(out_root, f"{name_ds}.index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    print(f"{name_ds}: {len(index)} tasks, {sum(v['label'] for v in index.values())} positive -> {ds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="windows.jsonl")
    ap.add_argument("--features", choices=FEATURES, help="default: all three")
    ap.add_argument("--out", default="harbor_tasks")
    ap.add_argument("--sliding", type=int, default=0, metavar="N",
                    help="sliding-window tasks (dataset name gets _SW): history = this window + up to N-1 prior windows of the same cow")
    ap.add_argument("--bins", action="store_true",
                    help="binned tasks (dataset name gets _PB): last 48h raw hourly + the 3 days before as 24h summary bins")
    ap.add_argument("--full", action="store_true",
                    help="expanding-window tasks (dataset name gets _FW): history = ALL prior windows of the same cow, "
                         "so window k holds windows 1..k (up to 336h), resetting at each new cow")
    ap.add_argument("--variant", choices=VARIANTS, help="redesigned task: digest prompts over the full past history")
    a = ap.parse_args()
    sliding, suffix = (FULL_N, "_FW") if a.full else (a.sliding, "_SW")
    builder = None
    if a.variant:
        (suffix, builder), sliding = VARIANTS[a.variant], FULL_N
    if a.bins:  # needs each window's full per-cow history, then bins it inside the prompt
        sliding, suffix = FULL_N, "_PB"
    windows = [json.loads(l) for l in open(a.windows)]
    for fs in [a.features] if a.features else FEATURES:
        export(windows, fs, a.out, sliding, suffix, a.bins, builder)


if __name__ == "__main__":
    main()
