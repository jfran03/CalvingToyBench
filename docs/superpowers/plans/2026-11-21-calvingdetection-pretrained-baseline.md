# CalvingDetection pretrained baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax. User said: do not commit.

**Goal:** Score the authors' pretrained RF (and their saved AE flags) on our 735 windows, writing `results/*.jsonl` that `score.py` already understands.

**Architecture:** One new module `calvingdetection_baseline.py` with pure helper functions (unit-tested on synthetic data) and a `main()` that reads a CalvingDetection clone, runs each cow through the RF fold that held it out, and writes two result files.

**Tech Stack:** pandas, numpy, scikit-learn (loads the .pkl), joblib.

**Spec:** `docs/superpowers/specs/2026-11-21-calvingdetection-pretrained-baseline-design.md`

## Global Constraints
- Result rows: `{id, cow, label, hours_to_calving, prob}` copied from `windows.jsonl`; files `results/cd__rf_pretrained.jsonl` and `results/cd__ae_only.jsonl`.
- `LOOKBACK_H = 12`, backward-looking only (hours t-11..t inclusive, matching how `prep.py` builds a window ending at t).
- Fold to held-out cows: F1 {09B,0C5,14C}, F2 {09F,0BB,0BD}, F3 {0BC,0BE,0E1}, F4 {0BA,13C,13D}, F5 {0E7,12A,12B}.
- RF prob = max P(class True) over AE-flagged hours in lookback, 0 if none. AE-only prob = flagged hours / 12.
- No changes to `score.py`, `plot_results.py`, `windows.jsonl`. Do not commit.

## Review Focus
- Window with no flagged hours in lookback gives prob exactly 0 (not NaN).
- Cow missing from the fold map raises a clear error, not a silent wrong fold.
- Flag timestamps are UTC-aligned to the hourly index (a tz mismatch would silently flag nothing): test asserts flags found.
- Fold sanity: in-fold (trained) cows score far higher AUROC than held-out; proves the fold map.

---

### Task 1: Pure helpers + tests
**Files:** Create `calvingdetection_baseline.py`, `tests/test_calvingdetection_baseline.py`

**Interfaces (produces):**
- `HELD_OUT: dict[int, tuple[str,...]]`, `fold_for_cow(cow: str) -> int` (ValueError if unknown)
- `add_sliding_features(df: pd.DataFrame) -> pd.DataFrame` (authors' 42-feature recipe: MA/STD 6,12 and Diff1/Diff3 for KNN2-5, Distance_moved, ActivityCount_Difference)
- `window_rf_prob(flag_probs: pd.Series, t: pd.Timestamp, lookback_h: int = 12) -> float` (flag_probs indexed by flagged-hour timestamps)
- `window_ae_frac(flag_times: pd.DatetimeIndex, t: pd.Timestamp, lookback_h: int = 12) -> float`

- [ ] Tests: `test_fold_for_cow` (every cow once, unknown raises), `test_window_rf_prob_max_and_empty` (max of in-range, exclusive of t-12h, inclusive of t, 0.0 when none), `test_window_ae_frac`, `test_sliding_features_columns` (42 feature names, MA6 equals rolling mean).
- [ ] Run, see fail; implement; run, see pass.

### Task 2: Loading + scoring + CLI
**Files:** Modify `calvingdetection_baseline.py`; Test: same file (an integration test skipped if `CalvingDetection/` absent, using `CD_REPO` env var).

**Interfaces:**
- `load_cow(repo: str, cow: str) -> tuple[pd.DataFrame, pd.DatetimeIndex]` (hourly feature frame indexed by UTC, flagged hours)
- `score_windows(repo: str, windows: list[dict], lookback_h=12) -> tuple[list[dict], list[dict]]` (rf rows, ae rows)
- `main()`: `--repo`, `--windows windows.jsonl`, `--out-dir results`; prints fold map, row counts, and fold sanity AUROCs (held-out vs a trained fold) for the report.

- [ ] Integration test: with the real repo, 735 rows each, every prob in [0,1], at least one nonzero RF prob, flags found for all 15 cows.
- [ ] Implement, run tests, run CLI, run `python score.py` and `python plot_results.py` unchanged.

### Task 3: README caveats
**Files:** Modify `README.md` (new "Authors' pretrained models" section with the four spec caveats and the command).
- [ ] Add section; verify the text matches the spec's caveats.
