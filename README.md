# Zero Shot Calving Benchmark

Question: Can a LLM forecast future claving events based on multimodal sensor data, specifically, GPS and accelerometer readings

# Key Ideas
- I benchmarked Claude, GPT, and TypeSafe (system one AI) models with a zero shot prompt (contained a set of GPS and Accelerometer data at a given window and asked to see if a cow was calving based on that data) to see if current LLM reasoning is able to detect the 15 cows calving. 
- To make it theoretically more viable for a rancher to adopt, I had a hard constraint with false alarms needing to be less than or equal to 1. Given this constraint, LLMs seem to be able to detect more cow calving events but does not need higher reasoning models to achieve such metrics (interestingly, cheaper models seem to work better)
- AUROC demonstrates that it can't discriminate normal cows vs pre-calving cows as well, AUPRC seems to flag real events 20% of the time. Better than the baselines but not the best in general
- **This suggests further research of using LLMs to identify livestock events (estrus, illness/injury, threats) solely on GPS and accelerometer data rather than traditional ML methods. A primary direction could be fine-tuning or better prompting to an open weight model for offline inference** 

# Motivation

[Wang et al.](https://www.mdpi.com/2077-0472/15/13/1434/pdf) utilized traditional machine learning methods (autoencoder & random forest) to successfilly determine if the systems were able to identify 15 calving events. The paper states that evaluating the model's generalization on these types of tasks is a possible open line of research, current large language models (LLMS) could provide a solution to generalizing these tasks by using its reasoning capabilities to classify calving events. There does not exist research that investigates the use of LLMs to identify these events solely on GPS and accelerometer data, this experiment attempts to address this open question.

## Alert threshold: false-alarm budget

**Step 1: count false alarms for a candidate threshold.** A false alarm is a window with no calving coming that the model scores at or above the threshold. Divide by the amount of non-calving time, in cow-weeks, to get a rate:

$$
\text{FA rate}(t) = \frac{\text{non-calving windows scored} \ge t}{\text{non-calving cow-weeks}}
$$

Raising $t$ can only lower this rate.

**Step 2: take the lowest threshold that stays within budget.** Try every distinct score the model produced and keep the lowest one whose FA rate is at most the budget $\beta$ (1.0 by default):

$$
t^{*} = \min \{\, t : \text{FA rate}(t) \le \beta \,\}
$$

Lower thresholds alert more often and so catch more cows, so this is the most sensitive threshold the budget allows. If even the highest score is over budget, $t^{*} = \infty$ and the model never alerts.

**Step 3: do it leave-one-cow-out.** Steps 1 and 2 use only the other 14 cows. The resulting $t^{*}$ is then applied to the held-out cow, and we repeat for all 15 cows. A window raises an alert when

$$
p_{\text{window}} \ge t^{*}_{\text{its cow}}
$$


## Data
CalvingDetection (Cao et al., 2025, *Agriculture* 15(13):1434), GPL-3.0.
15 Brangus cows, New Mexico rangeland, 2023. GPS fix every 60 min, accelerometer activity every 2 min (aggregated hourly).
Calving times are the authors' UTC labels; true birth time is only known to within ~12 h.

## Task
- One decision every 6 h, from 10 days before to 2 days after each cow's calving: 735 windows, 75 positive (10%).
- The model sees the last 48 hourly rows plus that cow's own 5-day baseline (median, 90th pct).
- Positive = calving within [t-6h, t+24h).
- Dates and cow IDs are removed, so the model can't look up the answer.
- Three feature sets: `gps_only` (distance moved, distance to herdmates), `accel_only` (activity), `both`.

## Run
```bash
pip install requests pandas numpy scikit-learn
git clone --depth 1 https://github.com/huipingcao/CalvingDetection.git
python prep.py --repo CalvingDetection          # rebuilds windows.jsonl (already included)

export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  GEMINI_API_KEY=...
python bench.py --provider anthropic --model claude-sonnet-5-5 --features both --limit 20   # smoke test
python bench.py --provider anthropic --model claude-sonnet-5-5 --features both              # full run
python bench.py --provider openai --model <id> --features both
python bench.py --provider gemini --model <id> --features both
# repeat with --features gps_only and accel_only
python score.py
```
- Model IDs: check each provider's model page. Claude: `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`.
- `--show-prompt` prints one full prompt. `--provider mock` runs offline to test the plumbing.
- Runs resume where they stopped. Results land in `results/`.
- Size: ~1.5k input tokens per call, so one model x one feature set is ~1.1M input tokens.

## Run with Harbor (recommended)
Same prompt, model call, parser and metrics as `bench.py`, but each window is a sandboxed
[Harbor](https://docs.harborframework.com) task with a hidden-label verifier. Needs Docker Desktop running.

```bash
cp .env.example .env        # then paste ANTHROPIC_API_KEY and OPENAI_API_KEY into .env
python harbor_run.py --oracle --features both --limit 10          # verifier sanity check: brier_reward must be 1.000
python harbor_run.py --model mock/rule --features both --limit 20 # offline plumbing test, no key
python harbor_run.py --model anthropic/claude-sonnet-5-5 --features both --limit 20   # cheap smoke test
python harbor_run.py --model anthropic/claude-sonnet-5-5 --features both              # full run, 735 windows
python harbor_run.py --model openai/<model-id> --features both
# repeat with --features gps_only and accel_only
python score.py
```
How it maps onto Harbor:

| Harbor concept | Here |
|---|---|
| Task | one window: `harbor_tasks/<features>/w<hash>/` (`python harbor_export.py` builds them) |
| `instruction.md` | the exact `bench.py` prompt |
| Agent | `harbor_agent.py:ZeroShotAgent`: one API call, writes `/app/answer.json`. Runs on your machine, so keys never enter the sandbox |
| Sandbox | `python:3.12-slim`, no network |
| Verifier | `tests/verify.py` in a separate container; the label (`tests/truth.json`) is never in the agent's sandbox |
| Reward | `brier_reward` = 1 - (p - label)^2, `valid`, `prob` |
| Job | one model x one feature set: `jobs/<model>__<features>/` |
| Dataset | one feature set: `harbor_tasks/<features>/` |

Harbor's summary table shows mean `brier_reward`. The headline metrics (AUROC, cows detected, ...) are
not means, so `harbor_collect.py` converts each job into `results/harbor__*.jsonl` and `score.py` scores
it exactly like a `bench.py` run. Re-running the same `harbor_run.py` command resumes the job.
Browse trials and raw model replies with `harbor view jobs`.

## Authors' pretrained models (AE + RF)
`calvingdetection_baseline.py` scores the CalvingDetection authors' own pipeline on the same 735 windows, with no retraining:
```bash
python calvingdetection_baseline.py --repo CalvingDetection   # -> results/cd__rf_pretrained.jsonl, results/cd__ae_only.jsonl
python score.py
```
- `cd__ae_only`: share of the last 12 h that the authors' autoencoder flagged as anomalous (their saved `anomalies_from_AE` files).
- `cd__rf_pretrained`: max random-forest probability over the flagged hours in the last 12 h (0 if none). Each cow is scored by the RF fold that held it out (3 cows per fold).
- Not like-for-like with the LLMs or leave-one-cow-out baselines:
  1. The RF was trained on +-24 h around birth (including after); our positive is [t-6 h, t+24 h).
  2. The AE threshold is each cow's season-wide 95th percentile (uses future data), and the AE trained on each cow's days before day -5, so it is not deployable as-is.
  3. Only flagged hours get a score, so most windows tie at 0; AUROC and threshold-0.5 columns are coarse.
  4. The paper reports per-cow cluster F1, a different metric that we do not reproduce.