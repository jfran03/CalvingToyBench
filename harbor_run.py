"""One command to run (or resume) a Harbor job and collect it for score.py.

Usage:
  python harbor_run.py --model mock/rule --features both --limit 20            # offline plumbing test
  python harbor_run.py --model anthropic/claude-sonnet-5-5 --features both --limit 20   # cheap smoke test
  python harbor_run.py --model anthropic/claude-sonnet-5-5 --features both     # full run (735 windows)
  python harbor_run.py --model openai/<model-id> --features gps_only
  python harbor_run.py --oracle --features both                                # verifier sanity check
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
    ap.add_argument("--limit", type=int, default=0, help="only N tasks (smoke test)")
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    if not (a.model or a.oracle):
        ap.error("pass --model or --oracle")

    if not os.path.isdir(os.path.join(HERE, "harbor_tasks", a.features)):
        subprocess.run([sys.executable, "harbor_export.py", "--features", a.features], cwd=HERE, check=True)

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
