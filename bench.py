"""Zero-shot LLM baseline for calving alerts.

Usage:
  python bench.py --provider anthropic --model claude-sonnet-5-5 --features both
  python bench.py --provider openai    --model <openai-model-id> --features gps_only
  python bench.py --provider gemini    --model <gemini-model-id> --features accel_only
  python bench.py --provider mock      --model rule  --features both   # offline dry run

Keys come from OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY.
Results are appended to results/<provider>__<model>__<features>.jsonl and the
run resumes where it left off, so it is safe to Ctrl-C and restart.
"""
import argparse, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
import requests

from prep import FEATURES

UNITS = {
    "dist_moved_m": "metres moved since the previous hourly GPS fix",
    "nearest_cow_m": "metres to the nearest collared herdmate",
    "mean_4_nearest_m": "mean metres to the 4 nearest collared herdmates",
    "activity": "accelerometer activity count accumulated during the hour",
}

SYSTEM = ("You are an expert in beef cattle behaviour and precision livestock monitoring. "
          "You analyse collar sensor data and answer only with a JSON object.")


def build_prompt(w, feats):
    cols = ["rel_h", "hour_local"] + feats
    lines = [",".join(cols)]
    for h in w["hours"]:
        lines.append(",".join("" if h[c] is None else str(h[c]) for c in cols))
    base = "\n".join(f"- {c}: median {w['baseline'][c]['median']:.1f}, 90th pct {w['baseline'][c]['p90']:.1f}"
                     for c in feats if c in w["baseline"])
    defs = "\n".join(f"- {c}: {UNITS[c]}" for c in feats)
    return f"""A pregnant multiparous Brangus beef cow is grazing a large desert rangeland pasture with about 15 other collared cows. She is due to calve sometime in the coming weeks, but the exact date is unknown.

Below are her last 48 hours of hourly collar data. rel_h=0 is now; hour_local is local clock hour (0-23). Blank cells are missing data.

Columns:
{defs}

Her own typical values over the preceding 5 days:
{base}

Data:
{chr(10).join(lines)}

Question: Is this cow calving now, or will she calve within the next 24 hours?

Reply with JSON only, exactly this shape:
{{"probability": <number 0 to 1>, "reason": "<at most 25 words>"}}"""


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in: {text[:200]}")
    j = json.loads(m.group(0))
    p = float(j["probability"])
    return max(0.0, min(1.0, p)), str(j.get("reason", ""))[:300]


def call_openai(model, prompt):
    r = requests.post("https://api.openai.com/v1/responses", timeout=180,
                      headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                      json={"model": model, "instructions": SYSTEM, "input": prompt,
                            "text": {"format": {"type": "json_object"}}})
    r.raise_for_status()
    j = r.json()
    text = j.get("output_text") or "".join(
        c.get("text", "") for o in j.get("output", []) if o.get("type") == "message" for c in o.get("content", []))
    return text, j.get("usage", {})


def call_anthropic(model, prompt):
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=180,
                      headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                      json={"model": model, "max_tokens": 1024, "system": SYSTEM,
                            "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    j = r.json()
    return "".join(b.get("text", "") for b in j["content"] if b["type"] == "text"), j.get("usage", {})


def call_gemini(model, prompt):
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                      timeout=180, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
                      json={"systemInstruction": {"parts": [{"text": SYSTEM}]},
                            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                            "generationConfig": {"responseMimeType": "application/json"}})
    r.raise_for_status()
    j = r.json()
    return "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"]), j.get("usageMetadata", {})


def call_mock(model, prompt):
    """Offline stand-in: flags isolation or low movement vs baseline. Tests the plumbing only."""
    rows = [l.split(",") for l in prompt.split("Data:\n")[1].split("\n\nQuestion")[0].splitlines()]
    hdr, body = rows[0], rows[-6:]
    score = 0.1
    if "mean_4_nearest_m" in hdr:
        i = hdr.index("mean_4_nearest_m")
        vals = [float(r[i]) for r in body if r[i]]
        med = float(re.search(r"mean_4_nearest_m: median ([\d.]+)", prompt).group(1))
        if vals and sum(vals) / len(vals) > 3 * med:
            score = 0.8
    return json.dumps({"probability": score, "reason": "mock rule"}), {}


CALL = {"openai": call_openai, "anthropic": call_anthropic, "gemini": call_gemini, "mock": call_mock}


def run_one(provider, model, w, feats):
    prompt = build_prompt(w, feats)
    for attempt in range(6):
        try:
            text, usage = CALL[provider](model, prompt)
            p, reason = parse(text)
            return {"id": w["id"], "cow": w["cow"], "label": w["label"],
                    "hours_to_calving": w["hours_to_calving"], "prob": p, "reason": reason, "usage": usage}
        except requests.HTTPError as e:
            code = e.response.status_code
            if code in (400, 401, 403, 404):
                raise SystemExit(f"{provider} error {code}: {e.response.text[:500]}")
            time.sleep(2 ** attempt * 3)
        except Exception as e:
            if attempt == 5:
                return {"id": w["id"], "cow": w["cow"], "label": w["label"],
                        "hours_to_calving": w["hours_to_calving"], "prob": None, "error": str(e)[:300]}
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=CALL)
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", default="both", choices=FEATURES)
    ap.add_argument("--windows", default="windows.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="only the first N windows (cheap smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--show-prompt", action="store_true", help="print one prompt and exit")
    a = ap.parse_args()

    windows = [json.loads(l) for l in open(a.windows)]
    feats = FEATURES[a.features]
    if a.show_prompt:
        print(build_prompt(next(w for w in windows if w["label"]), feats))
        return
    if a.limit:
        windows = windows[: a.limit]

    os.makedirs("results", exist_ok=True)
    out = f"results/{a.provider}__{a.model.replace('/', '_')}__{a.features}.jsonl"
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out) if json.loads(l).get("prob") is not None}
    todo = [w for w in windows if w["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go -> {out}")

    with open(out, "a") as fh, ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(lambda w: run_one(a.provider, a.model, w, feats), todo), 1):
            fh.write(json.dumps(res) + "\n"); fh.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")


if __name__ == "__main__":
    main()
