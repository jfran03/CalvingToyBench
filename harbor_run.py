"""One command to run (or resume) a Harbor job and collect it for score.py.

Usage:
  python harbor_run.py --model mock/rule --features both --limit 20            # offline plumbing test
  python harbor_run.py --model anthropic/claude-sonnet-5-5 --features both --limit 20   # cheap smoke test
  python harbor_run.py --model anthropic/claude-sonnet-5-5 --features both     # full run (735 windows)
  python harbor_run.py --model openai/<model-id> --features gps_only
  python harbor_run.py --oracle --features both                                # verifier sanity check
  python harbor_run.py --model mock/rule --features both --sw --limit 20       # sliding-window (_SW) dataset
  python harbor_run.py --model anthropic/claude-sonnet-5-5 --features both --sw  # full _SW run (735 windows)
--sw runs harbor_tasks/<features>_SW (history = this window + up to 9 prior windows of the same cow);
it is built on first use with `harbor_export.py --sliding 10`. Job and result names get the _SW suffix.
--fw runs <features>_FW: history = ALL prior windows of the same cow (up to 336h), built with --full.
--pb runs <features>_PB: last 48h raw hourly + the 3 days before as 24h mean/max bins (built with --bins).
then: python score.py

Keys are read from .env (copy .env.example). Re-running the same command resumes the job.
"""
import argparse, os, re, subprocess, sys

from harbor_collect import collect
from prep import FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dotenv(path):
    """Minimal KEY=value parser (no extra dependency). Real environment variables win over .env."""
    out = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return {k: v for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="<provider>/<model>, provider in anthropic|openai|jev|mock")
    ap.add_argument("--oracle", action="store_true", help="run Harbor's oracle agent (writes the true label)")
    ap.add_argument("--features", default="both", choices=FEATURES)
    ap.add_argument("--sw", action="store_true", help="use the sliding-window dataset (<features>_SW)")
    ap.add_argument("--fw", action="store_true", help="use the full/expanding-window dataset (<features>_FW)")
    ap.add_argument("--pb", action="store_true", help="use the binned dataset (<features>_PB): 48h raw + 24h prior bins")
    ap.add_argument("--limit", type=int, default=0, help="only N tasks (smoke test)")
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    if not (a.model or a.oracle):
        ap.error("pass --model or --oracle")

    if a.sw + a.fw + a.pb > 1:
        ap.error("pick one of --sw / --fw / --pb")
    extra = ["--sliding", "10"] if a.sw else ["--full"] if a.fw else ["--bins"] if a.pb else []
    base = a.features
    if a.sw or a.fw or a.pb:
        a.features += "_SW" if a.sw else "_FW" if a.fw else "_PB"  # dataset folder name
    if not os.path.isdir(os.path.join(HERE, "harbor_tasks", a.features)):
        subprocess.run([sys.executable, "harbor_export.py", "--features", base] + extra, cwd=HERE, check=True)

    slug = "oracle" if a.oracle else re.sub(r"[^A-Za-z0-9._-]", "_", a.model)
    job = f"{slug}__{a.features}" + (f"__limit{a.limit}" if a.limit else "")
    job_dir = os.path.join("jobs", job)
    # The agent runs on this machine (not in the sandbox) and reads keys from its own environment.
    # Load .env here rather than via `harbor run --env-file`, because `harbor job resume` has no such flag.
    env = {**load_dotenv(os.path.join(HERE, ".env")), **os.environ,
           "PYTHONPATH": HERE + os.pathsep + os.environ.get("PYTHONPATH", "")}

    if os.path.exists(os.path.join(HERE, job_dir, "config.json")):
        print(f"resuming {job_dir}")
        cmd = ["harbor", "job", "resume", "-p", job_dir]
    else:
        agent = ["-a", "oracle"] if a.oracle else ["-a", "harbor_agent:ZeroShotAgent", "-m", a.model]
        cmd = ["harbor", "run", "-p", f"harbor_tasks/{a.features}", *agent,
               "-n", str(a.concurrency), "-o", "jobs", "--job-name", job, "-y"]
        if a.limit:
            cmd += ["-l", str(a.limit)]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=HERE, env=env, check=True)
    os.chdir(HERE)
    collect(job_dir)


if __name__ == "__main__":
    main()
