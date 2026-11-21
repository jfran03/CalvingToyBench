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
  harbor_tasks/<features>.index.json   task name -> window id, cow, label, hours_to_calving
                                       (kept OUTSIDE the dataset folder so it is not a task)

Task names are a hash of the window id, so nothing in a task name reveals the cow or the date.
"""
import argparse, hashlib, json, os, shutil

from bench import build_prompt
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


def export(windows, fs, out_root):
    feats = FEATURES[fs]
    ds = os.path.join(out_root, fs)
    shutil.rmtree(ds, ignore_errors=True)
    index = {}
    for w in windows:
        name = task_name(w["id"])
        assert name not in index, f"hash collision on {name}"
        t = os.path.join(ds, name)
        write(os.path.join(t, "instruction.md"), build_prompt(w, feats) + "\n")
        write(os.path.join(t, "task.toml"), TASK_TOML.format(features=fs))
        write(os.path.join(t, "tests", "test.sh"), TEST_SH, executable=True)
        write(os.path.join(t, "tests", "verify.py"), VERIFY_PY)
        write(os.path.join(t, "tests", "truth.json"), json.dumps({"label": w["label"]}))
        write(os.path.join(t, "solution", "solve.sh"), SOLVE_SH.format(label=w["label"]), executable=True)
        index[name] = {k: w[k] for k in ("id", "cow", "label", "hours_to_calving")}
    with open(os.path.join(out_root, f"{fs}.index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    print(f"{fs}: {len(index)} tasks, {sum(v['label'] for v in index.values())} positive -> {ds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="windows.jsonl")
    ap.add_argument("--features", choices=FEATURES, help="default: all three")
    ap.add_argument("--out", default="harbor_tasks")
    a = ap.parse_args()
    windows = [json.loads(l) for l in open(a.windows)]
    for fs in [a.features] if a.features else FEATURES:
        export(windows, fs, a.out)


if __name__ == "__main__":
    main()
